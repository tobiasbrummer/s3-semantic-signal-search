#!/usr/bin/env python3
"""
BEIR Transformers Test (S3 Hierarchical) - EXTENDED VERSION

Features:
- Word-Pooled Token-Level Embeddings (Jina v3 + SPLADE)
- Sign-Hash Retrieval Methods (Document, Token, Hybrid)
- Hybrid Sign Storage (Stable vs Volatile Dimensions)
- BM25 Keyword Search Layer
- MRL (Matryoshka Representation Learning) Support
- Extended Benchmarking (Memory, Latency, Recall@k)

Optimierungen:
- Nutzt Jina v3 Task-Präfixe statt Kwargs (verhindert Flash-Attention Fehler).
- Robustes Word-Pooling via char_spans zur Vermeidung von Index-Fehlern.
- Korrektes Max-Pooling für Sparse-Vektoren.
- Fix: return_offsets_mapping für Sparse-Tokenizer hinzugefügt.
- Fix: Konsequente Nutzung von .float().cpu().numpy() für BFloat16 Support.
"""

import numpy as np
import torch
import time
import spacy
import re
import csv
import sys
import warnings
from dataclasses import dataclass, field
from collections import defaultdict
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM
from datasets import load_dataset

# Optional: BM25 for keyword search
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    print("Warning: rank_bm25 not installed. BM25 methods will be disabled.")

import os
import pickle
import hashlib

# Suppress Specific Warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="thinc")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")


# =============================================================================
# INDEX CACHE
# =============================================================================

CACHE_DIR = ".cache/beir_index"

def get_cache_path(dataset_name: str, limit_docs: int, use_sparse: bool) -> str:
    """Generate unique cache path based on config."""
    config_str = f"{dataset_name}_{limit_docs}_sparse={use_sparse}"
    cache_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    return os.path.join(CACHE_DIR, f"index_{dataset_name}_{limit_docs}_{cache_hash}.pkl")

def save_index_cache(index, cache_path: str):
    """Save index to disk."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(index, f)
    print(f"  Index cached to: {cache_path}")

def load_index_cache(cache_path: str):
    """Load index from disk if exists."""
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            index = pickle.load(f)
        print(f"  Loaded index from cache: {cache_path}")
        return index
    return None


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TokenInfo:
    text: str
    dense_vec: np.ndarray
    sparse_vec: np.ndarray
    sparse_weight: float
    global_idx: int
    sentence_idx: int
    in_sentence_idx: int
    char_offset: tuple[int, int]

@dataclass
class SentenceInfo:
    idx: int
    text: str
    tokens: list[TokenInfo]
    dense_vec: np.ndarray
    sparse_vec: np.ndarray
    char_offset: tuple[int, int]

@dataclass
class DocumentIndex:
    doc_id: str
    text: str
    sentences: list[SentenceInfo]
    doc_embedding: np.ndarray
    token_embeddings: np.ndarray
    splade_terms: np.ndarray
    splade_weights: np.ndarray
    # Sign-Hash Storage (Original)
    sign_stable_mask: np.ndarray = None       # (dim,) bool - welche dims sind stabil
    sign_stable_values: np.ndarray = None     # (n_stable,) bool - sign für stabile dims  
    sign_volatile_values: np.ndarray = None   # (n_tokens, n_volatile) bool - volatile signs
    # Packed Sign-Hash Storage (Optimized)
    doc_sign_packed: np.ndarray = None        # (n_bytes,) uint8 - packed doc sign
    token_signs_packed: np.ndarray = None     # (n_tokens, n_bytes) uint8 - packed token signs
    n_dims: int = 0                           # Original dimension count for accurate similarity
    # Packed Hybrid Storage (Optimized)
    stable_signs_packed: np.ndarray = None    # (n_bytes_stable,) uint8
    volatile_signs_packed: np.ndarray = None  # (n_tokens, n_bytes_volatile) uint8
    n_stable_dims: int = 0
    n_volatile_dims: int = 0
    # BM25 Storage
    bm25_tokens: list = None                  # Tokenized text for BM25


@dataclass
class FullIndex:
    docs: dict = field(default_factory=dict)
    splade_inverted: dict = field(default_factory=lambda: defaultdict(list))
    bm25_index: object = None                 # BM25Okapi index
    bm25_doc_ids: list = None                 # Mapping from BM25 index to doc_id
    # Storage Statistics
    total_docs: int = 0
    total_tokens: int = 0
    float_storage_bytes: int = 0
    sign_full_storage_bytes: int = 0
    sign_hybrid_storage_bytes: int = 0


# =============================================================================
# SIGN-HASH UTILITIES (OPTIMIZED WITH BITPACKING)
# =============================================================================

# Precomputed popcount lookup table for uint8 (0-255)
_POPCOUNT_TABLE = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)


def pack_signs(signs: np.ndarray) -> np.ndarray:
    """
    Pack bool array zu uint8 array für schnelle Bitoperationen.
    
    Args:
        signs: (dim,) or (n, dim) bool array
        
    Returns:
        (n_bytes,) or (n, n_bytes) uint8 array
    """
    if signs.ndim == 1:
        return np.packbits(signs.astype(np.uint8))
    else:
        # (n_tokens, dim) -> (n_tokens, n_bytes)
        return np.packbits(signs.astype(np.uint8), axis=1)


def hamming_distance_packed(packed1: np.ndarray, packed2: np.ndarray) -> int:
    """
    Berechne Hamming-Distanz zwischen zwei gepackten uint8 Arrays.
    Nutzt XOR + Lookup-Table für popcount (SIMD-freundlich).
    """
    xor_result = np.bitwise_xor(packed1, packed2)
    return _POPCOUNT_TABLE[xor_result].sum()


def hamming_similarity_packed(packed1: np.ndarray, packed2: np.ndarray, n_dims: int) -> float:
    """
    Berechne Hamming-Similarity zwischen zwei gepackten uint8 Arrays.
    
    Args:
        packed1, packed2: Gepackte uint8 Arrays
        n_dims: Originale Anzahl Dimensionen (für korrekte Normalisierung)
        
    Returns:
        Similarity in [0, 1]
    """
    diff_bits = hamming_distance_packed(packed1, packed2)
    return 1.0 - (diff_bits / n_dims)


def hamming_similarity_batch_packed(query_packed: np.ndarray, docs_packed: np.ndarray, n_dims: int) -> np.ndarray:
    """
    Batch Hamming-Similarity: Query gegen mehrere Dokumente.
    
    Args:
        query_packed: (n_bytes,) uint8
        docs_packed: (n_docs, n_bytes) uint8
        n_dims: Originale Anzahl Dimensionen
        
    Returns:
        (n_docs,) float similarities
    """
    # Broadcast XOR: (n_docs, n_bytes)
    xor_results = np.bitwise_xor(docs_packed, query_packed)
    # Popcount via lookup: (n_docs, n_bytes) -> (n_docs,)
    diff_bits = _POPCOUNT_TABLE[xor_results].sum(axis=1)
    return 1.0 - (diff_bits / n_dims)


def compute_stability_mask(embeddings: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """
    Berechne welche Dimensionen stabil sind (Flip-Rate < threshold).
    
    Args:
        embeddings: (n_tokens, dim) array
        threshold: Maximale Flip-Rate für "stabil"
        
    Returns: 
        bool array (dim,) - True = stabil
    """
    if len(embeddings) < 2:
        return np.ones(embeddings.shape[1], dtype=bool)
    
    signs = embeddings > 0
    sign_changes = np.diff(signs.astype(np.int8), axis=0)
    flips_per_dim = np.abs(sign_changes).sum(axis=0)
    flip_rate = flips_per_dim / (len(embeddings) - 1)
    
    return flip_rate < threshold


def build_hybrid_signs(doc: DocumentIndex, stability_threshold: float = 0.1) -> tuple:
    """
    Berechne Hybrid Sign Storage für ein Dokument.
    
    Returns:
        (stable_mask, stable_signs, volatile_signs)
    """
    embeddings = doc.token_embeddings
    if len(embeddings) < 1:
        dim = doc.doc_embedding.shape[0]
        return np.ones(dim, dtype=bool), np.zeros(0, dtype=bool), np.zeros((0, 0), dtype=bool)
    
    signs = embeddings > 0
    stable_mask = compute_stability_mask(embeddings, stability_threshold)
    
    # Stabile Signs: Mehrheitsentscheidung
    stable_signs = (signs[:, stable_mask].mean(axis=0) > 0.5)
    
    # Volatile Signs: Alle Tokens, nur volatile Dims
    volatile_signs = signs[:, ~stable_mask]
    
    return stable_mask, stable_signs, volatile_signs


def build_packed_signs(doc_embedding: np.ndarray, token_embeddings: np.ndarray) -> tuple:
    """
    Baue gepackte Sign-Arrays für schnelle Suche.
    
    Returns:
        (doc_sign_packed, token_signs_packed, n_dims)
    """
    n_dims = doc_embedding.shape[0]
    doc_sign_packed = pack_signs(doc_embedding > 0)
    
    if len(token_embeddings) > 0:
        token_signs = token_embeddings > 0  # (n_tokens, dim)
        token_signs_packed = pack_signs(token_signs)  # (n_tokens, n_bytes)
    else:
        token_signs_packed = np.zeros((0, len(doc_sign_packed)), dtype=np.uint8)
    
    return doc_sign_packed, token_signs_packed, n_dims


def build_hybrid_signs_packed(embeddings: np.ndarray, stability_threshold: float = 0.1) -> tuple:
    """
    Berechne Hybrid Sign Storage mit Bitpacking für schnelle Suche.
    
    Returns:
        (stable_mask, stable_signs_packed, volatile_signs_packed, n_stable, n_volatile)
    """
    if len(embeddings) < 1:
        dim = embeddings.shape[1] if len(embeddings.shape) > 1 else 1024
        return np.ones(dim, dtype=bool), np.zeros(0, dtype=np.uint8), np.zeros((0, 0), dtype=np.uint8), 0, 0
    
    signs = embeddings > 0
    stable_mask = compute_stability_mask(embeddings, stability_threshold)
    
    n_stable = stable_mask.sum()
    n_volatile = (~stable_mask).sum()
    
    # Stabile Signs: Mehrheitsentscheidung + Packen
    stable_signs = (signs[:, stable_mask].mean(axis=0) > 0.5)
    stable_signs_packed = pack_signs(stable_signs) if n_stable > 0 else np.zeros(0, dtype=np.uint8)
    
    # Volatile Signs: Alle Tokens, nur volatile Dims + Packen
    volatile_signs = signs[:, ~stable_mask]  # (n_tokens, n_volatile)
    if n_volatile > 0 and len(volatile_signs) > 0:
        volatile_signs_packed = pack_signs(volatile_signs)  # (n_tokens, n_bytes_volatile)
    else:
        volatile_signs_packed = np.zeros((len(embeddings), 0), dtype=np.uint8)
    
    return stable_mask, stable_signs_packed, volatile_signs_packed, n_stable, n_volatile


# Legacy functions for backwards compatibility
def hamming_similarity(signs1: np.ndarray, signs2: np.ndarray) -> float:
    """Hamming Similarity zwischen zwei Sign-Vektoren (legacy, nicht optimiert)."""
    return (signs1 == signs2).mean()


def pack_signs_to_bytes(signs: np.ndarray) -> bytes:
    """Pack bool array zu Bytes für Speicherberechnung."""
    return np.packbits(signs.astype(np.uint8)).tobytes()


# =============================================================================
# FAST BATCH ENCODER (OPTIMIZED)
# =============================================================================

class S3FastEncoder:
    """
    Optimierter Encoder mit:
    - Batch GPU Processing
    - Half-Precision (float16) für 2x Speedup
    - Optionalem SPLADE (kann deaktiviert werden für schnellere Indexierung)
    - torch.inference_mode für maximale Performance
    """
    
    def __init__(self, device: str = None, use_spacy: bool = False, use_sparse: bool = True, 
                 use_half_precision: bool = True, spacy_model: str = "de_dep_news_trf"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_half = use_half_precision and self.device == "cuda"
        self.use_sparse = use_sparse
        
        print(f"Initializing S3FastEncoder on {self.device}...")
        print(f"   Options: half_precision={self.use_half}, use_sparse={use_sparse}")
        
        self.use_spacy = use_spacy
        if use_spacy:
            self.nlp = spacy.load(spacy_model)
        else:
            self.nlp = None
        
        # Dense Model (Jina v3)
        self.dense_model_name = "jinaai/jina-embeddings-v3"
        self.dense_tokenizer = AutoTokenizer.from_pretrained(self.dense_model_name, trust_remote_code=True)
        self.dense_model = AutoModel.from_pretrained(
            self.dense_model_name, trust_remote_code=True, attn_implementation="eager", use_flash_attn=False
        ).to(self.device).eval()
        
        if self.use_half:
            self.dense_model = self.dense_model.half()
        
        # Sparse Model (SPLADE) - Optional
        if use_sparse:
            self.sparse_model_name = "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1"
            self.sparse_tokenizer = AutoTokenizer.from_pretrained(self.sparse_model_name)
            self.sparse_model = AutoModelForMaskedLM.from_pretrained(self.sparse_model_name).to(self.device).eval()
            if self.use_half:
                self.sparse_model = self.sparse_model.half()
        else:
            self.sparse_model = None
            self.sparse_tokenizer = None
        
        print(f"   Models loaded.")
    
    def _encode_dense_batch(self, texts: list[str], is_query: bool = False, batch_size: int = 32) -> list[tuple]:
        """
        Batch Dense Encoding mit Jina v3.
        Returns: List of (token_embeddings, word_ids, offsets) per text
        """
        marker = "<|retrieval.query|>" if is_query else "<|retrieval.passage|>"
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = [marker + t[:3500] for t in texts[i:i+batch_size]]
            
            enc = self.dense_tokenizer(
                batch_texts, return_tensors="pt", truncation=True, 
                max_length=1024, padding=True, return_offsets_mapping=True
            ).to(self.device)
            
            with torch.inference_mode():
                out = self.dense_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
                embeddings = out.last_hidden_state  # (batch, seq, dim)
            
            # Extract per-text results
            for j in range(embeddings.shape[0]):
                token_embs = embeddings[j].float().cpu().numpy()
                offsets = enc["offset_mapping"][j].cpu().numpy()
                word_ids = enc.word_ids(j)
                results.append((token_embs, word_ids, offsets, len(marker)))
        
        return results
    
    def _encode_sparse_batch(self, texts: list[str], batch_size: int = 32) -> list[tuple]:
        """
        Batch Sparse Encoding mit SPLADE.
        Returns: List of (sparse_vectors, word_ids, offsets) per text
        """
        if self.sparse_model is None:
            # Return dummy sparse vectors if SPLADE is disabled
            return [(np.zeros((1, 30000), dtype=np.float32), [None], np.array([[0, 0]])) for _ in texts]
        
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = [t[:3500] for t in texts[i:i+batch_size]]
            
            enc = self.sparse_tokenizer(
                batch_texts, return_tensors="pt", truncation=True,
                max_length=512, padding=True, return_offsets_mapping=True
            ).to(self.device)
            
            model_inputs = {k: v for k, v in enc.items() if k not in ["offset_mapping", "overflow_to_sample_mapping"]}
            
            with torch.inference_mode():
                out = self.sparse_model(**model_inputs)
                # SPLADE transformation
                sparse_vecs = torch.log1p(torch.relu(out.logits))  # (batch, seq, vocab)
            
            for j in range(sparse_vecs.shape[0]):
                vec = sparse_vecs[j].float().cpu().numpy()
                offsets = enc["offset_mapping"][j].cpu().numpy()
                word_ids = enc.word_ids(j)
                results.append((vec, word_ids, offsets))
        
        return results
    
    def _pool_to_words_fast(self, dense_result: tuple, sparse_result: tuple, text: str) -> list[dict]:
        """
        GPU-beschleunigtes Word-Pooling mit Tensor-Operationen.
        Verwendet scatter_add für Mean-Pooling statt Python-Loops.
        """
        token_embs, d_word_ids, d_offsets, marker_len = dense_result
        sparse_vecs, s_word_ids, s_offsets = sparse_result
        
        # Skip if no valid word IDs
        valid_wids = [wid for wid in d_word_ids if wid is not None]
        if not valid_wids:
            return []
        
        # Build word_id -> indices mapping (necessary for character offsets)
        unique_wids = sorted(set(valid_wids))
        n_words = len(unique_wids)
        wid_to_idx = {wid: idx for idx, wid in enumerate(unique_wids)}
        
        # Convert to numpy arrays for vectorized operations
        token_embs_np = token_embs if isinstance(token_embs, np.ndarray) else token_embs
        sparse_vecs_np = sparse_vecs if isinstance(sparse_vecs, np.ndarray) else sparse_vecs
        
        # Build token -> word mapping array
        token_word_map = np.array([wid_to_idx.get(wid, -1) for wid in d_word_ids], dtype=np.int64)
        valid_mask = token_word_map >= 0
        
        # Dense: Vectorized mean pooling using np.bincount-like approach
        dim = token_embs_np.shape[1]
        word_sums = np.zeros((n_words, dim), dtype=np.float32)
        word_counts = np.zeros(n_words, dtype=np.float32)
        
        valid_tokens = token_embs_np[valid_mask]
        valid_word_ids = token_word_map[valid_mask]
        
        # Use np.add.at for scatter-add (faster than loop)
        np.add.at(word_sums, valid_word_ids, valid_tokens)
        np.add.at(word_counts, valid_word_ids, 1)
        
        word_counts = np.maximum(word_counts, 1)  # Avoid div by zero
        word_dense = word_sums / word_counts[:, np.newaxis]
        
        # Pre-compute word character ranges (do this once, not in loop)
        word_ranges = np.zeros((n_words, 2), dtype=np.int32)
        for wi, wid in enumerate(unique_wids):
            d_indices = [i for i, w in enumerate(d_word_ids) if w == wid]
            if d_indices:
                word_ranges[wi, 0] = max(0, int(d_offsets[d_indices[0]][0]) - marker_len)
                word_ranges[wi, 1] = max(0, int(d_offsets[d_indices[-1]][1]) - marker_len)
        
        # Sparse: Vectorized max pooling with pre-computed ranges
        sparse_dim = sparse_vecs_np.shape[1]
        word_sparse = np.zeros((n_words, sparse_dim), dtype=np.float32)
        
        # Convert sparse offsets to numpy for vectorized overlap check
        s_starts = np.array([o[0] for o in s_offsets], dtype=np.int32)
        s_ends = np.array([o[1] for o in s_offsets], dtype=np.int32)
        s_valid = np.array([wid is not None for wid in s_word_ids])
        
        # Vectorized overlap: (n_sparse, n_words) boolean matrix
        # Overlap exists if: max(word_start, sparse_start) < min(word_end, sparse_end)
        overlap_start = np.maximum(word_ranges[:, 0], s_starts[:, np.newaxis])  # (n_sparse, n_words)
        overlap_end = np.minimum(word_ranges[:, 1], s_ends[:, np.newaxis])
        has_overlap = (overlap_start < overlap_end) & s_valid[:, np.newaxis]
        
        # For each word, max-pool all overlapping sparse tokens
        for wi in range(n_words):
            overlapping_tokens = has_overlap[:, wi]
            if overlapping_tokens.any():
                word_sparse[wi] = sparse_vecs_np[overlapping_tokens].max(axis=0)
        
        # Build word list with text spans
        words = []
        for wi, wid in enumerate(unique_wids):
            start_char, end_char = word_ranges[wi]
            word_text = text[start_char:end_char]
            
            if not word_text.strip():
                continue
            
            words.append({
                "text": word_text,
                "dense": word_dense[wi],
                "sparse": word_sparse[wi],
                "char_start": int(start_char),
                "char_end": int(end_char),
            })
        
        return words
    
    def encode_documents_batch(
        self, 
        docs: list[dict], 
        batch_size: int = 8,
        chunk_size: int = 75,  # Process this many docs before clearing intermediate results
        compute_hybrid_signs: bool = True,
        keep_float_embeddings: bool = False,
        show_progress: bool = True
    ) -> dict:
        """
        Batch-encode multiple documents efficiently with chunk-wise memory management.
        
        Args:
            docs: List of {"id": str, "text": str}
            batch_size: GPU batch size for encoding
            chunk_size: Number of docs to process before clearing memory
            compute_hybrid_signs: Whether to compute hybrid sign storage
            keep_float_embeddings: Whether to keep float embeddings (memory intensive)
            show_progress: Print progress
            
        Returns:
            dict of doc_id -> DocumentIndex
        """
        import gc
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        index_docs = {}
        total_docs = len(docs)
        
        if show_progress:
            print(f"  Encoding {total_docs} documents (chunks of {chunk_size}, batch={batch_size}, pipelined)...")
        
        def process_single_doc(args):
            """Process a single document (word pooling + sign computation)."""
            doc_id, text, dense_result, sparse_result, compute_hybrid, keep_float = args
            
            if not text.strip():
                return doc_id, DocumentIndex(
                    doc_id, "", [], np.zeros(1024), np.zeros((1, 1024)), 
                    np.array([]), np.array([])
                )
            
            # Word pooling
            words = self._pool_to_words_fast(dense_result, sparse_result, text)
            
            if not words:
                return doc_id, DocumentIndex(
                    doc_id, text, [], np.zeros(1024), np.zeros((1, 1024)),
                    np.array([]), np.array([])
                )
            
            # Stack embeddings
            token_embeddings = np.stack([w["dense"] for w in words])
            doc_dense = token_embeddings.mean(axis=0)
            
            sparse_stack = np.stack([w["sparse"] for w in words])
            doc_sparse = sparse_stack.max(axis=0)
            nonzero = np.nonzero(doc_sparse)[0]
            
            # Create DocumentIndex
            doc_index = DocumentIndex(
                doc_id=doc_id, text=text, sentences=[],
                doc_embedding=doc_dense, token_embeddings=token_embeddings,
                splade_terms=nonzero, splade_weights=doc_sparse[nonzero]
            )
            
            # BM25 tokens
            doc_index.bm25_tokens = text.lower().split()
            
            # Compute packed signs
            doc_sign_packed, token_signs_packed, n_dims = build_packed_signs(doc_dense, token_embeddings)
            doc_index.doc_sign_packed = doc_sign_packed
            doc_index.token_signs_packed = token_signs_packed
            doc_index.n_dims = n_dims
            
            # Compute hybrid signs
            if compute_hybrid and len(token_embeddings) > 0:
                stable_mask, stable_packed, volatile_packed, n_stable, n_volatile = build_hybrid_signs_packed(
                    token_embeddings, stability_threshold=0.1
                )
                doc_index.sign_stable_mask = stable_mask
                doc_index.stable_signs_packed = stable_packed
                doc_index.volatile_signs_packed = volatile_packed
                doc_index.n_stable_dims = n_stable
                doc_index.n_volatile_dims = n_volatile
            
            # Memory-efficient
            if not keep_float:
                doc_index.token_embeddings = None
            
            return doc_id, doc_index
        
        # Use ThreadPoolExecutor for parallel CPU processing
        num_workers = min(8, chunk_size)  # Limit workers
        
        # True pipelining: GPU work in background thread
        def encode_chunk_gpu(texts, batch_size):
            """Run GPU encoding in background."""
            dense = self._encode_dense_batch(texts, is_query=False, batch_size=batch_size)
            sparse = self._encode_sparse_batch(texts, batch_size=batch_size)
            return dense, sparse
        
        with ThreadPoolExecutor(max_workers=num_workers + 1) as executor:  # +1 for GPU thread
            gpu_future = None
            pending_cpu_work = None
            
            chunks = []
            for chunk_start in range(0, total_docs, chunk_size):
                chunk_end = min(chunk_start + chunk_size, total_docs)
                chunk_docs = docs[chunk_start:chunk_end]
                texts = [d["text"] for d in chunk_docs]
                doc_ids = [d["id"] for d in chunk_docs]
                chunks.append((doc_ids, texts))
            
            for chunk_idx, (doc_ids, texts) in enumerate(chunks):
                # Start GPU encoding for THIS chunk in background
                new_gpu_future = executor.submit(encode_chunk_gpu, texts, batch_size)
                
                # While GPU is working on new chunk, process PREVIOUS chunk's results on CPU
                if pending_cpu_work is not None:
                    prev_doc_ids, prev_texts, prev_dense, prev_sparse = pending_cpu_work
                    
                    # Parallel CPU processing
                    tasks = [
                        (prev_doc_ids[i], prev_texts[i], prev_dense[i], prev_sparse[i], 
                         compute_hybrid_signs, keep_float_embeddings)
                        for i in range(len(prev_doc_ids))
                    ]
                    
                    for doc_id, doc_index in executor.map(process_single_doc, tasks):
                        index_docs[doc_id] = doc_index
                    
                    del prev_dense, prev_sparse
                    gc.collect()
                
                # Wait for GPU to finish current chunk
                dense_results, sparse_results = new_gpu_future.result()
                pending_cpu_work = (doc_ids, texts, dense_results, sparse_results)
                
                if show_progress:
                    print(f"    Chunk {chunk_idx + 1}/{len(chunks)} GPU done...")
            
            # Process final chunk
            if pending_cpu_work is not None:
                prev_doc_ids, prev_texts, prev_dense, prev_sparse = pending_cpu_work
                
                tasks = [
                    (prev_doc_ids[i], prev_texts[i], prev_dense[i], prev_sparse[i], 
                     compute_hybrid_signs, keep_float_embeddings)
                    for i in range(len(prev_doc_ids))
                ]
                
                for doc_id, doc_index in executor.map(process_single_doc, tasks):
                    index_docs[doc_id] = doc_index
        
        if show_progress:
            print(f"    Processed {len(index_docs)}/{total_docs} documents...")
        
        return index_docs
    
    def encode_query(self, text: str):
        """Encode a single query."""
        dense_results = self._encode_dense_batch([text], is_query=True, batch_size=1)
        sparse_results = self._encode_sparse_batch([text], batch_size=1)
        
        words = self._pool_to_words_fast(dense_results[0], sparse_results[0], text)
        
        if not words:
            return np.zeros(1024), np.array([]), np.array([])
        
        query_dense = np.mean([w["dense"] for w in words], axis=0)
        sparse_stack = np.stack([w["sparse"] for w in words])
        query_sparse = sparse_stack.max(axis=0)
        nonzero = np.nonzero(query_sparse)[0]
        
        return query_dense, nonzero, query_sparse[nonzero]


# =============================================================================
# HIERARCHICAL ENCODER (ORIGINAL - with spaCy)
# =============================================================================

class S3HierarchicalEncoder:
    def __init__(self, device: str = None, spacy_model: str = "de_dep_news_trf"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing S3HierarchicalEncoder on {self.device}...")

        self.nlp = spacy.load(spacy_model)
        
        self.dense_model_name = "jinaai/jina-embeddings-v3"
        self.dense_tokenizer = AutoTokenizer.from_pretrained(self.dense_model_name, trust_remote_code=True)
        self.dense_model = AutoModel.from_pretrained(
            self.dense_model_name, trust_remote_code=True, attn_implementation="eager", use_flash_attn=False
        ).to(self.device).eval()

        self.sparse_model_name = "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1"
        self.sparse_tokenizer = AutoTokenizer.from_pretrained(self.sparse_model_name)
        self.sparse_model = AutoModelForMaskedLM.from_pretrained(self.sparse_model_name).to(self.device).eval()

    def _get_word_vectors(self, text: str, is_query: bool = False):
        # Jina v3 Task Marker
        marker = "<|retrieval.query|>" if is_query else "<|retrieval.passage|>"
        marked_text = marker + text
        
        # 1. Dense Pass
        dense_enc = self.dense_tokenizer(marked_text, return_tensors="pt", truncation=True, max_length=1024, return_offsets_mapping=True).to(self.device)
        with torch.no_grad():
            dense_out = self.dense_model(input_ids=dense_enc["input_ids"], attention_mask=dense_enc["attention_mask"])
            token_embeddings = dense_out.last_hidden_state[0]
        
        offsets = dense_enc["offset_mapping"][0].cpu().numpy()
        word_ids = dense_enc.word_ids(0)
        marker_len = len(marker)

        # 2. Sparse Pass
        sparse_enc = self.sparse_tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512, return_offsets_mapping=True
        ).to(self.device)

        # Filter out keys that the model forward doesn't accept
        model_inputs = {k: v for k, v in sparse_enc.items() if k not in ["offset_mapping", "overflow_to_sample_mapping"]}

        with torch.no_grad():
            sparse_out = self.sparse_model(**model_inputs)
            # SPLADE transformation
            token_sparse_vectors = torch.log1p(torch.relu(sparse_out.logits[0]))

        sparse_word_ids = sparse_enc.word_ids(0)
        sparse_offsets = sparse_enc["offset_mapping"][0].cpu().numpy()

        # 3. Robust Word Pooling
        words = []
        unique_wids = sorted(list(set(wid for wid in word_ids if wid is not None)))

        for wid in unique_wids:
            d_indices = [i for i, w in enumerate(word_ids) if w == wid]

            # Start/Ende im Original-Text finden (Präfix abziehen!)
            start_char = max(0, offsets[d_indices[0]][0] - marker_len)
            end_char = max(0, offsets[d_indices[-1]][1] - marker_len)
            word_text = text[start_char:end_char]

            if not word_text.strip(): continue

            # Dense: Mean
            d_word_vec = token_embeddings[d_indices].mean(dim=0).float().cpu().numpy()

            # Sparse: Max
            s_indices = []
            for i, (s_start, s_end) in enumerate(sparse_offsets):
                if sparse_word_ids[i] is None: continue
                # Overlap check
                if max(start_char, s_start) < min(end_char, s_end):
                    s_indices.append(i)
            if s_indices:
                s_word_vec = token_sparse_vectors[s_indices].max(dim=0).values.float().cpu().numpy()
            else:
                s_word_vec = np.zeros(self.sparse_model.config.vocab_size, dtype=np.float32)

            words.append({
                "text": word_text,
                "dense": d_word_vec,
                "sparse": s_word_vec,
                "offset": (start_char, end_char)
            })

        return words

    def encode_document(self, doc_id: str, text: str, compute_hybrid_signs: bool = True, keep_float_embeddings: bool = True) -> DocumentIndex:
        if not text.strip():
            return DocumentIndex(doc_id, "", [], np.zeros(1024), np.zeros((1,1024)), np.array([]), np.array([]))
            
        doc = self.nlp(text)
        word_data = self._get_word_vectors(text, is_query=False)
        
        all_tokens = []
        for i, wd in enumerate(word_data):
            s_idx = -1
            for j, sent in enumerate(doc.sents):
                if wd["offset"][0] >= sent.start_char and wd["offset"][0] < sent.end_char:
                    s_idx = j
                    break
            
            all_tokens.append(TokenInfo(
                text=wd["text"], dense_vec=wd["dense"], sparse_vec=wd["sparse"],
                sparse_weight=float(wd["sparse"].max()), global_idx=i,
                sentence_idx=s_idx, in_sentence_idx=0, char_offset=wd["offset"]
            ))

        sentence_tokens = defaultdict(list)
        for t in all_tokens:
            if t.sentence_idx != -1: sentence_tokens[t.sentence_idx].append(t)
        
        final_sentences = []
        for s_idx in sorted(sentence_tokens.keys()):
            tokens = sentence_tokens[s_idx]
            for i, t in enumerate(tokens): t.in_sentence_idx = i
            s_dense = np.mean([t.dense_vec for t in tokens], axis=0)
            s_sparse = np.max([t.sparse_vec for t in tokens], axis=0)
            final_sentences.append(SentenceInfo(
                idx=s_idx, text=list(doc.sents)[s_idx].text, tokens=tokens,
                dense_vec=s_dense, sparse_vec=s_sparse,
                char_offset=(tokens[0].char_offset[0], tokens[-1].char_offset[1])
            ))

        if not final_sentences:
            return DocumentIndex(doc_id, text, [], np.zeros(1024), np.stack([t.dense_vec for t in all_tokens]), np.array([]), np.array([]))

        doc_dense = np.mean([s.dense_vec for s in final_sentences], axis=0)
        doc_sparse = np.max([s.sparse_vec for s in final_sentences], axis=0)
        nonzero = np.nonzero(doc_sparse)[0]
        
        token_embeddings = np.stack([t.dense_vec for t in all_tokens])
        
        doc_index = DocumentIndex(
            doc_id=doc_id, text=text, sentences=final_sentences,
            doc_embedding=doc_dense, token_embeddings=token_embeddings,
            splade_terms=nonzero, splade_weights=doc_sparse[nonzero]
        )
        
        # Compute Hybrid Sign Storage (Original + Packed)
        if compute_hybrid_signs and len(token_embeddings) > 0:
            stable_mask, stable_signs, volatile_signs = build_hybrid_signs(doc_index)
            doc_index.sign_stable_mask = stable_mask
            doc_index.sign_stable_values = stable_signs
            doc_index.sign_volatile_values = volatile_signs
            
            # Packed Hybrid Signs
            _, stable_packed, volatile_packed, n_stable, n_volatile = build_hybrid_signs_packed(
                token_embeddings, stability_threshold=0.1
            )
            doc_index.stable_signs_packed = stable_packed
            doc_index.volatile_signs_packed = volatile_packed
            doc_index.n_stable_dims = n_stable
            doc_index.n_volatile_dims = n_volatile
        
        # Compute Packed Signs (Optimized)
        doc_sign_packed, token_signs_packed, n_dims = build_packed_signs(doc_dense, token_embeddings)
        doc_index.doc_sign_packed = doc_sign_packed
        doc_index.token_signs_packed = token_signs_packed
        doc_index.n_dims = n_dims
        
        # Memory-efficient mode: Discard float embeddings after packing
        if not keep_float_embeddings:
            doc_index.token_embeddings = None
            doc_index.sign_stable_values = None  # Keep only packed version
            doc_index.sign_volatile_values = None  # Keep only packed version
            # Clear sentence token vectors too
            for sent in doc_index.sentences:
                for tok in sent.tokens:
                    tok.dense_vec = None
                    tok.sparse_vec = None
        
        return doc_index

    def encode_query(self, text: str):
        word_data = self._get_word_vectors(text, is_query=True)
        query_dense = np.mean([wd["dense"] for wd in word_data], axis=0)
        query_sparse = np.max([wd["sparse"] for wd in word_data], axis=0)
        nonzero = np.nonzero(query_sparse)[0]
        return query_dense, nonzero, query_sparse[nonzero]


# =============================================================================
# SEARCH METHODS
# =============================================================================

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)


# --- Standard Methods ---

def search_pooled(query_emb, index, top_k=10):
    """Dense Cosine on Document Embeddings."""
    res = [(did, cosine_sim(query_emb, d.doc_embedding)) for did, d in index.docs.items()]
    res.sort(key=lambda x: x[1], reverse=True)
    return res[:top_k]


def search_splade(q_terms, q_weights, index, top_k=10):
    """SPLADE Inverted Index Search."""
    scores = defaultdict(float)
    for tid, qw in zip(q_terms, q_weights):
        if tid in index.splade_inverted:
            for did, dw in index.splade_inverted[tid]: scores[did] += qw * dw
    res = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return res[:top_k]


def build_bm25_index(index):
    """Build BM25 index from documents."""
    if not HAS_BM25:
        print("BM25 not available (rank_bm25 not installed)")
        return
    
    doc_ids = []
    tokenized_corpus = []
    
    for doc_id, doc in index.docs.items():
        doc_ids.append(doc_id)
        tokens = doc.bm25_tokens if doc.bm25_tokens else doc.text.lower().split()
        tokenized_corpus.append(tokens)
    
    index.bm25_index = BM25Okapi(tokenized_corpus)
    index.bm25_doc_ids = doc_ids
    print(f"  BM25 index built with {len(doc_ids)} documents")


def search_bm25(query_text, index, top_k=10):
    """BM25 Keyword Search."""
    if not HAS_BM25 or index.bm25_index is None:
        return []
    
    query_tokens = query_text.lower().split()
    scores = index.bm25_index.get_scores(query_tokens)
    
    # Get top-k indices
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    return [(index.bm25_doc_ids[i], scores[i]) for i in top_indices if scores[i] > 0]


def search_pooled_mrl(query_emb, index, top_k=10, mrl_dim=256):
    """
    Dense Cosine with MRL Truncation.
    
    Uses Matryoshka Representation Learning to use only the first N dimensions.
    Jina v3 embeddings preserve semantic information in early dimensions.
    256 dims = ~96% of 1024 performance, 4x faster/smaller.
    """
    q_trunc = query_emb[:mrl_dim]
    q_norm = np.linalg.norm(q_trunc) + 1e-9
    
    res = []
    for did, d in index.docs.items():
        d_trunc = d.doc_embedding[:mrl_dim]
        score = np.dot(q_trunc, d_trunc) / (q_norm * (np.linalg.norm(d_trunc) + 1e-9))
        res.append((did, score))
    
    res.sort(key=lambda x: x[1], reverse=True)
    return res[:top_k]


def search_sign_mrl(query_emb, index, top_k=10, mrl_dim=256):
    """
    Sign-Hash on MRL Truncated Embeddings.
    
    Combines MRL (first N dims) with binary sign quantization.
    Even more compressed: 256 bits instead of 1024 bits.
    """
    q_trunc = query_emb[:mrl_dim]
    q_sign = q_trunc > 0
    
    res = []
    for did, d in index.docs.items():
        d_trunc = d.doc_embedding[:mrl_dim]
        d_sign = d_trunc > 0
        # Hamming similarity on truncated signs
        score = np.sum(q_sign == d_sign) / mrl_dim
        res.append((did, score))
    
    res.sort(key=lambda x: x[1], reverse=True)
    return res[:top_k]


def search_combined_prefilter(query_text, query_emb, index, prefilter_k=50, top_k=10, alpha=0.3):
    """
    Combined Pre-Filter Pipeline: (Sign Pooled ∪ BM25) → Interp Rerank
    
    Stage 1: Get candidates from both Sign Pooled and BM25 (union)
    Stage 2: Rerank top candidates using Interpolated scoring
    
    This combines the speed of Sign Pooled with BM25's keyword matching.
    """
    # Stage 1a: Sign Pooled pre-filter (fast binary search)
    sign_results = search_sign_pooled_batch(query_emb, index, top_k=prefilter_k)
    
    # Stage 1b: BM25 pre-filter (keyword matching)
    bm25_results = search_bm25(query_text, index, top_k=prefilter_k) if HAS_BM25 and index.bm25_index else []
    
    # Union of candidates
    candidate_ids = set(r[0] for r in sign_results) | set(r[0] for r in bm25_results)
    
    if not candidate_ids:
        return []
    
    # Get dense scores for normalization
    dense_scores = {}
    for did in candidate_ids:
        d = index.docs[did]
        dense_scores[did] = cosine_sim(query_emb, d.doc_embedding)
    
    # Normalize dense scores to 0-1
    scores_arr = np.array(list(dense_scores.values()))
    dense_min, dense_max = scores_arr.min(), scores_arr.max()
    dense_range = max(dense_max - dense_min, 1e-9)
    
    # Stage 2: Interp rerank on candidates
    q_packed = pack_signs(query_emb > 0)
    n_dims = query_emb.shape[0]
    
    res = []
    for did in candidate_ids:
        d = index.docs[did]
        
        # Token score (Sign-Hash MaxSim)
        if d.token_signs_packed is not None and len(d.token_signs_packed) > 0:
            scores = hamming_similarity_batch_packed(q_packed, d.token_signs_packed, n_dims)
            token_score = scores.max()
        else:
            token_score = 0.5
        
        # Normalized dense score
        norm_dense = (dense_scores[did] - dense_min) / dense_range
        
        # Interpolated final score
        final_score = alpha * norm_dense + (1 - alpha) * token_score
        res.append((did, final_score))
    
    res.sort(key=lambda x: x[1], reverse=True)
    return res[:top_k]


def search_token_bf(query_emb, index, top_k=10):
    """Token-Level MaxSim (Brute Force Cosine)."""
    res = [(did, max(cosine_sim(query_emb, t) for t in d.token_embeddings)) for did, d in index.docs.items()]
    res.sort(key=lambda x: x[1], reverse=True)
    return res[:top_k]


# --- Sign-Hash Methods (OPTIMIZED with Bitpacking) ---

def search_sign_pooled(query_emb, index, top_k=10):
    """Sign-Hash on Document Embeddings (Optimized Bitpacking)."""
    q_packed = pack_signs(query_emb > 0)
    n_dims = query_emb.shape[0]
    
    res = []
    for did, d in index.docs.items():
        if d.doc_sign_packed is not None:
            score = hamming_similarity_packed(q_packed, d.doc_sign_packed, n_dims)
        else:
            # Fallback
            score = hamming_similarity(query_emb > 0, d.doc_embedding > 0)
        res.append((did, score))
    res.sort(key=lambda x: x[1], reverse=True)
    return res[:top_k]


def search_sign_pooled_batch(query_emb, index, top_k=10):
    """Sign-Hash on Document Embeddings (Batch Optimized)."""
    q_packed = pack_signs(query_emb > 0)
    n_dims = query_emb.shape[0]
    
    # Collect all doc packed signs
    doc_ids = []
    doc_packed_list = []
    for did, d in index.docs.items():
        if d.doc_sign_packed is not None:
            doc_ids.append(did)
            doc_packed_list.append(d.doc_sign_packed)
    
    if not doc_packed_list:
        return []
    
    # Batch computation
    docs_packed = np.stack(doc_packed_list)  # (n_docs, n_bytes)
    scores = hamming_similarity_batch_packed(q_packed, docs_packed, n_dims)
    
    # Sort and return top_k
    results = list(zip(doc_ids, scores))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def search_sign_token(query_emb, index, top_k=10):
    """Sign-Hash Token-Level MaxSim (Optimized Bitpacking)."""
    q_packed = pack_signs(query_emb > 0)
    n_dims = query_emb.shape[0]
    
    res = []
    for did, d in index.docs.items():
        if d.token_signs_packed is not None and len(d.token_signs_packed) > 0:
            # Batch: query vs all tokens
            scores = hamming_similarity_batch_packed(q_packed, d.token_signs_packed, n_dims)
            best = scores.max()
        else:
            # Fallback
            q_sign = query_emb > 0
            best = max(hamming_similarity(q_sign, t > 0) for t in d.token_embeddings)
        res.append((did, best))
    res.sort(key=lambda x: x[1], reverse=True)
    return res[:top_k]


def search_sign_hybrid_storage(query_emb, index, top_k=10):
    """Sign-Hash with Hybrid Storage (Optimized with Bitpacking)."""
    q_sign = query_emb > 0
    res = []
    
    for did, d in index.docs.items():
        # Check if we have packed hybrid storage
        if d.stable_signs_packed is not None and d.n_stable_dims > 0:
            stable_mask = d.sign_stable_mask
            n_stable = d.n_stable_dims
            n_volatile = d.n_volatile_dims
            total_dims = n_stable + n_volatile
            
            if total_dims == 0:
                res.append((did, 0.0))
                continue
            
            # Pack query signs for stable and volatile dims
            q_stable = q_sign[stable_mask]
            q_volatile = q_sign[~stable_mask]
            q_stable_packed = pack_signs(q_stable) if n_stable > 0 else np.zeros(0, dtype=np.uint8)
            q_volatile_packed = pack_signs(q_volatile) if n_volatile > 0 else np.zeros(0, dtype=np.uint8)
            
            # Stable score: Same for all tokens (compute once)
            if n_stable > 0:
                stable_diff = hamming_distance_packed(q_stable_packed, d.stable_signs_packed)
                stable_match = n_stable - stable_diff
            else:
                stable_match = 0
            
            # Volatile scores: Batch compute for all tokens
            if n_volatile > 0 and len(d.volatile_signs_packed) > 0:
                # XOR + popcount batch
                xor_results = np.bitwise_xor(d.volatile_signs_packed, q_volatile_packed)
                volatile_diffs = _POPCOUNT_TABLE[xor_results].sum(axis=1)  # (n_tokens,)
                volatile_matches = n_volatile - volatile_diffs
                
                # Combined scores for all tokens
                total_matches = stable_match + volatile_matches
                scores = total_matches / total_dims
                best = scores.max()
            else:
                best = stable_match / total_dims if total_dims > 0 else 0.0
        
        elif d.sign_stable_mask is not None:
            # Fallback to original (non-packed) hybrid storage
            stable_mask = d.sign_stable_mask
            q_stable = q_sign[stable_mask]
            q_volatile = q_sign[~stable_mask]
            
            stable_match = (q_stable == d.sign_stable_values).sum()
            n_stable = len(d.sign_stable_values)
            
            best = 0
            for t in range(len(d.sign_volatile_values)):
                volatile_match = (q_volatile == d.sign_volatile_values[t]).sum()
                n_volatile = len(d.sign_volatile_values[t])
                
                total_match = stable_match + volatile_match
                total_dims = n_stable + n_volatile
                score = total_match / total_dims if total_dims > 0 else 0
                best = max(best, score)
        else:
            # Fallback to full sign token-level
            q_packed = pack_signs(q_sign)
            if d.token_signs_packed is not None and len(d.token_signs_packed) > 0:
                scores = hamming_similarity_batch_packed(q_packed, d.token_signs_packed, d.n_dims)
                best = scores.max()
            else:
                best = max(hamming_similarity(q_sign, t > 0) for t in d.token_embeddings)
        
        res.append((did, best))
    
    res.sort(key=lambda x: x[1], reverse=True)
    return res[:top_k]


def search_hybrid_pipeline(query_emb, q_terms, q_weights, index, dense_k=50, top_k=10):
    """Hybrid Pipeline: Dense Pre-Filter → Sign-Hash Token Re-Rank (Optimized)."""
    # Stage 1: Fast Dense search for candidates
    candidates = search_pooled(query_emb, index, top_k=dense_k)
    candidate_ids = set(c[0] for c in candidates)
    
    # Stage 2: Sign-Hash Token-Level on candidates only (Optimized)
    q_packed = pack_signs(query_emb > 0)
    n_dims = query_emb.shape[0]
    
    res = []
    for did in candidate_ids:
        d = index.docs[did]
        if d.token_signs_packed is not None and len(d.token_signs_packed) > 0:
            scores = hamming_similarity_batch_packed(q_packed, d.token_signs_packed, n_dims)
            best = scores.max()
        else:
            q_sign = query_emb > 0
            best = max(hamming_similarity(q_sign, t > 0) for t in d.token_embeddings)
        res.append((did, best))
    
    res.sort(key=lambda x: x[1], reverse=True)
    return res[:top_k]


def search_hybrid_pipeline_100(query_emb, q_terms, q_weights, index, top_k=10):
    """Hybrid Pipeline with larger pre-filter pool (k=100)."""
    return search_hybrid_pipeline(query_emb, q_terms, q_weights, index, dense_k=100, top_k=top_k)


def search_interpolated_pipeline(query_emb, q_terms, q_weights, index, dense_k=50, alpha=0.4, top_k=10):
    """
    Interpolated Pipeline: Dense Pre-Filter → Weighted (Dense + Token) Scoring.
    
    Instead of discarding dense scores, combines them with token scores:
    final_score = alpha * norm_dense_score + (1 - alpha) * token_score
    
    This preserves global semantic context (important for asymmetric retrieval like Arguana)
    while still benefiting from token-level precision.
    
    Args:
        alpha: Weight for dense score (0.3-0.5 recommended for asymmetric datasets)
    """
    # Stage 1: Fast Dense search for candidates
    candidates = search_pooled(query_emb, index, top_k=dense_k)
    candidate_dict = {c[0]: c[1] for c in candidates}  # doc_id -> dense_score
    
    # Normalize dense scores to 0-1 range
    if candidates:
        dense_scores = np.array([c[1] for c in candidates])
        dense_min, dense_max = dense_scores.min(), dense_scores.max()
        if dense_max > dense_min:
            dense_range = dense_max - dense_min
        else:
            dense_range = 1.0
    
    # Stage 2: Sign-Hash Token-Level on candidates + Interpolation
    q_packed = pack_signs(query_emb > 0)
    n_dims = query_emb.shape[0]
    
    res = []
    for did, dense_score in candidate_dict.items():
        d = index.docs[did]
        
        # Token score (already 0-1 from hamming_similarity)
        if d.token_signs_packed is not None and len(d.token_signs_packed) > 0:
            scores = hamming_similarity_batch_packed(q_packed, d.token_signs_packed, n_dims)
            token_score = scores.max()
        else:
            token_score = 0.5  # Fallback
        
        # Normalize dense score to 0-1
        norm_dense = (dense_score - dense_min) / dense_range if dense_range > 0 else 0.5
        
        # Interpolated final score
        final_score = alpha * norm_dense + (1 - alpha) * token_score
        res.append((did, final_score))
    
    res.sort(key=lambda x: x[1], reverse=True)
    return res[:top_k]


def search_interpolated_30(query_emb, q_terms, q_weights, index, top_k=10):
    """Interpolated Pipeline with alpha=0.3 (30% dense, 70% token)."""
    return search_interpolated_pipeline(query_emb, q_terms, q_weights, index, alpha=0.3, top_k=top_k)


def search_interpolated_50(query_emb, q_terms, q_weights, index, top_k=10):
    """Interpolated Pipeline with alpha=0.5 (50% dense, 50% token)."""
    return search_interpolated_pipeline(query_emb, q_terms, q_weights, index, alpha=0.5, top_k=top_k)


def search_multistage_pipeline(query_emb, q_terms, q_weights, index, dense_k=50, splade_k=50, top_k=10):
    """
    Multi-Stage Pipeline: (Dense ∪ SPLADE) Pre-Filter → Sign-Hash Token Re-Rank.
    
    Combines both Dense and SPLADE for better recall in pre-filter stage.
    """
    # Stage 1: Get candidates from BOTH Dense and SPLADE
    dense_results = search_pooled(query_emb, index, top_k=dense_k)
    splade_results = search_splade(q_terms, q_weights, index, top_k=splade_k)
    
    # Union of both candidate sets
    candidate_ids = set(r[0] for r in dense_results) | set(r[0] for r in splade_results)
    
    # Stage 2: Sign-Hash Token-Level Re-Rank
    q_packed = pack_signs(query_emb > 0)
    n_dims = query_emb.shape[0]
    
    res = []
    for did in candidate_ids:
        d = index.docs[did]
        if d.token_signs_packed is not None and len(d.token_signs_packed) > 0:
            scores = hamming_similarity_batch_packed(q_packed, d.token_signs_packed, n_dims)
            best = scores.max()
        else:
            q_sign = query_emb > 0
            best = max(hamming_similarity(q_sign, t > 0) for t in d.token_embeddings) if d.token_embeddings is not None else 0
        res.append((did, best))
    
    res.sort(key=lambda x: x[1], reverse=True)
    return res[:top_k]


def search_adaptive_pipeline(query_emb, q_terms, q_weights, index, min_k=30, max_k=200, top_k=10):
    """
    Adaptive Pipeline: Dynamic pre-filter k based on Dense score distribution.
    
    If top scores are close together (uncertain query), expand k.
    If top scores are clearly separated (confident query), use smaller k.
    """
    # Stage 1a: Get initial candidates with max_k
    all_candidates = search_pooled(query_emb, index, top_k=max_k)
    
    if len(all_candidates) < 2:
        candidate_ids = set(c[0] for c in all_candidates)
    else:
        # Compute score gap between top and lower candidates
        top_score = all_candidates[0][1]
        scores = np.array([c[1] for c in all_candidates])
        
        # Confidence: How much gap is there between top and median?
        median_score = np.median(scores)
        score_range = top_score - scores[-1] if len(scores) > 1 else 1.0
        
        if score_range > 0:
            confidence = (top_score - median_score) / score_range
        else:
            confidence = 0.5
        
        # Adaptive k: Low confidence → more candidates
        # confidence 0 → max_k, confidence 1 → min_k
        adaptive_k = int(max_k - confidence * (max_k - min_k))
        adaptive_k = max(min_k, min(max_k, adaptive_k))
        
        candidate_ids = set(c[0] for c in all_candidates[:adaptive_k])
    
    # Stage 2: Sign-Hash Token-Level Re-Rank
    q_packed = pack_signs(query_emb > 0)
    n_dims = query_emb.shape[0]
    
    res = []
    for did in candidate_ids:
        d = index.docs[did]
        if d.token_signs_packed is not None and len(d.token_signs_packed) > 0:
            scores = hamming_similarity_batch_packed(q_packed, d.token_signs_packed, n_dims)
            best = scores.max()
        else:
            q_sign = query_emb > 0
            best = max(hamming_similarity(q_sign, t > 0) for t in d.token_embeddings) if d.token_embeddings is not None else 0
        res.append((did, best))
    
    res.sort(key=lambda x: x[1], reverse=True)
    return res[:top_k]


def search_bm25_hybrid(query_text, query_emb, index, bm25_k=50, top_k=10):
    """BM25 Pre-Filter → Sign-Hash Token Re-Rank."""
    # Stage 1: BM25 Pre-Filter
    bm25_results = search_bm25(query_text, index, top_k=bm25_k)
    candidate_ids = set(r[0] for r in bm25_results)
    
    if not candidate_ids:
        return []
    
    # Stage 2: Sign-Hash Token-Level Re-Rank
    q_packed = pack_signs(query_emb > 0)
    n_dims = query_emb.shape[0]
    
    res = []
    for did in candidate_ids:
        d = index.docs[did]
        if d.token_signs_packed is not None and len(d.token_signs_packed) > 0:
            scores = hamming_similarity_batch_packed(q_packed, d.token_signs_packed, n_dims)
            best = scores.max()
        else:
            best = 0
        res.append((did, best))
    
    res.sort(key=lambda x: x[1], reverse=True)
    return res[:top_k]


def search_triple_stage(query_text, query_emb, q_terms, q_weights, index, 
                        dense_k=50, splade_k=50, bm25_k=50, top_k=10):
    """
    Triple-Stage Pipeline: (Dense ∪ SPLADE ∪ BM25) Pre-Filter → Sign-Hash Token Re-Rank.
    
    Maximum recall pre-filter by combining all three retrieval paradigms:
    - Dense: Semantic similarity
    - SPLADE: Learned sparse
    - BM25: Lexical/keyword matching
    """
    # Stage 1: Get candidates from ALL THREE methods
    dense_results = search_pooled(query_emb, index, top_k=dense_k)
    splade_results = search_splade(q_terms, q_weights, index, top_k=splade_k)
    bm25_results = search_bm25(query_text, index, top_k=bm25_k) if HAS_BM25 and index.bm25_index else []
    
    # Union of all three
    candidate_ids = (
        set(r[0] for r in dense_results) | 
        set(r[0] for r in splade_results) | 
        set(r[0] for r in bm25_results)
    )
    
    # Stage 2: Sign-Hash Token-Level Re-Rank
    q_packed = pack_signs(query_emb > 0)
    n_dims = query_emb.shape[0]
    
    res = []
    for did in candidate_ids:
        d = index.docs[did]
        if d.token_signs_packed is not None and len(d.token_signs_packed) > 0:
            scores = hamming_similarity_batch_packed(q_packed, d.token_signs_packed, n_dims)
            best = scores.max()
        else:
            best = 0
        res.append((did, best))
    
    res.sort(key=lambda x: x[1], reverse=True)
    return res[:top_k]

def search_combined_rerank(query_emb, q_terms, q_weights, index, dense_k=50, splade_k=50, top_k=10):
    """Combined: Dense + SPLADE Pre-Filter → Cosine Token Re-Rank."""
    # Stage 1: Get candidates from both
    dense_results = search_pooled(query_emb, index, top_k=dense_k)
    splade_results = search_splade(q_terms, q_weights, index, top_k=splade_k)
    
    candidate_ids = set(r[0] for r in dense_results) | set(r[0] for r in splade_results)
    
    # Stage 2: Token-level refinement
    res = []
    for did in candidate_ids:
        d = index.docs[did]
        best = max(cosine_sim(query_emb, t) for t in d.token_embeddings)
        res.append((did, best))
    
    res.sort(key=lambda x: x[1], reverse=True)
    return res[:top_k]


# =============================================================================
# STORAGE METRICS
# =============================================================================

def compute_storage_metrics(index: FullIndex) -> dict:
    """Berechne Speichermetriken für alle Methoden."""
    total_tokens = 0
    float_bytes = 0
    sign_full_bytes = 0
    sign_hybrid_bytes = 0
    
    dim = None
    
    for did, d in index.docs.items():
        # Get token count from packed signs if token_embeddings is None
        if d.token_embeddings is not None:
            n_tokens = len(d.token_embeddings)
        elif d.token_signs_packed is not None:
            n_tokens = len(d.token_signs_packed)
        else:
            n_tokens = 0
        
        total_tokens += n_tokens
        
        # Get dimension from n_dims field or token_embeddings
        if dim is None:
            if d.n_dims > 0:
                dim = d.n_dims
            elif d.token_embeddings is not None and len(d.token_embeddings) > 0:
                dim = d.token_embeddings[0].shape[0]
            else:
                dim = 1024  # Default
        
        if dim:
            # Float32 Storage
            float_bytes += n_tokens * dim * 4
            
            # Full Sign Storage (1 bit per dim per token)
            sign_full_bytes += (n_tokens * dim + 7) // 8
            
            # Hybrid Sign Storage
            if d.sign_stable_mask is not None:
                n_stable = d.n_stable_dims if d.n_stable_dims > 0 else 0
                n_volatile = d.n_volatile_dims if d.n_volatile_dims > 0 else 0
                
                # Mask: dim bits once per doc
                sign_hybrid_bytes += (dim + 7) // 8
                # Stable signs: n_stable bits once per doc
                sign_hybrid_bytes += (n_stable + 7) // 8
                # Volatile signs: n_tokens * n_volatile bits
                sign_hybrid_bytes += (n_tokens * n_volatile + 7) // 8
    
    return {
        "total_docs": len(index.docs),
        "total_tokens": total_tokens,
        "avg_tokens_per_doc": total_tokens / len(index.docs) if index.docs else 0,
        "float32_kb": float_bytes / 1024,
        "sign_full_kb": sign_full_bytes / 1024,
        "sign_hybrid_kb": sign_hybrid_bytes / 1024,
        "compression_full": float_bytes / sign_full_bytes if sign_full_bytes > 0 else 0,
        "compression_hybrid": float_bytes / sign_hybrid_bytes if sign_hybrid_bytes > 0 else 0,
        "hybrid_savings_vs_full": (1 - sign_hybrid_bytes / sign_full_bytes) * 100 if sign_full_bytes > 0 else 0,
    }


# =============================================================================
# BEIR DATASET LOADING
# =============================================================================

def load_beir_subset(name, limit_docs=1000, limit_queries=100):
    print(f"Loading BEIR dataset: {name}...")
    try:
        corpus = load_dataset(f"mteb/{name}", "corpus", split="corpus")
        queries_ds = load_dataset(f"mteb/{name}", "queries", split="queries")
        qrels = load_dataset(f"mteb/{name}", "default", split="test")
        doc_map = {item["_id"]: (item.get("title", "") + " " + item.get("text", "")).strip() for i, item in enumerate(corpus) if i < limit_docs}
        relevance = defaultdict(set)
        for item in qrels:
            if item["corpus-id"] in doc_map: relevance[item["query-id"]].add(item["corpus-id"])
        query_list = [{"id": item["_id"], "text": item["text"]} for item in queries_ds if item["_id"] in relevance][:limit_queries]
        return [{"id": k, "text": v} for k, v in doc_map.items()], query_list, relevance
    except Exception as e:
        print(f"Error: {e}"); return None, None, None


# =============================================================================
# EVALUATION
# =============================================================================

def encode_queries_batch(queries, encoder):
    """Pre-encode all queries in batch for faster evaluation."""
    print(f"  Pre-encoding {len(queries)} queries...")
    
    texts = [q["text"] for q in queries]
    batch_size = 16  # Smaller batch for queries
    chunk_size = 100  # Process queries in chunks to avoid OOM
    
    encoded = []
    
    for chunk_start in range(0, len(texts), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(texts))
        chunk_texts = texts[chunk_start:chunk_end]
        
        # Batch GPU encoding for this chunk
        dense_results = encoder._encode_dense_batch(chunk_texts, is_query=True, batch_size=batch_size)
        sparse_results = encoder._encode_sparse_batch(chunk_texts, batch_size=batch_size)
        
        for i in range(len(chunk_texts)):
            text = chunk_texts[i]
            words = encoder._pool_to_words_fast(dense_results[i], sparse_results[i], text)
            
            if not words:
                encoded.append((np.zeros(1024), np.array([]), np.array([])))
                continue
            
            query_dense = np.mean([w["dense"] for w in words], axis=0)
            sparse_stack = np.stack([w["sparse"] for w in words])
            query_sparse = sparse_stack.max(axis=0)
            nonzero = np.nonzero(query_sparse)[0]
            
            encoded.append((query_dense, nonzero, query_sparse[nonzero]))
        
        # Clear GPU cache after each chunk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"    Queries {chunk_end}/{len(texts)}...")
    
    return encoded


def evaluate_method(name, search_fn, queries, encoded_queries, relevance, index, needs_splade=False, needs_query_text=False):
    """Evaluate a search method and return metrics."""
    hits = 0
    total_time = 0
    n_queries = len(queries)
    
    for i, q in enumerate(queries):
        qd, qt, qw = encoded_queries[i]
        query_text = q["text"]
        
        start = time.time()
        if needs_query_text:
            res = search_fn(query_text, qd, qt, qw, index)
        elif needs_splade:
            res = search_fn(qd, qt, qw, index)
        else:
            res = search_fn(qd, index)
        total_time += time.time() - start
        
        if relevance[q["id"]] & set(r[0] for r in res):
            hits += 1
        
        # Progress display every 200 queries
        if (i + 1) % 200 == 0 or i == n_queries - 1:
            print(f"\r    {name}: {i+1}/{n_queries} queries...", end="", flush=True)
    
    print()  # New line after progress
    
    n_queries = len(queries)
    return {
        "name": name,
        "recall_at_10": hits / n_queries if n_queries > 0 else 0,
        "avg_time_ms": (total_time / n_queries * 1000) if n_queries > 0 else 0,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("BEIR TRANSFORMERS TEST - EXTENDED WITH SIGN-HASH (FAST BATCH)")
    print("=" * 70)
    
    # Use Fast Encoder with half-precision and batch processing
    encoder = S3FastEncoder(
        use_spacy=False, 
        use_sparse=True,           # SPLADE enabled for full benchmark
        use_half_precision=True    # float16 for 2x GPU speedup
    )
    
    all_results = {}
    limit_docs = 500
    limit_queries = 500
    use_sparse = encoder.use_sparse
    
    for ds_name in ["scifact", "fiqa", "nfcorpus", "arguana", "trec-covid"]:
        docs, queries, relevance = load_beir_subset(ds_name, limit_docs=limit_docs, limit_queries=limit_queries)
        if not docs: continue
        
        print(f"\n{'='*70}")
        print(f"DATASET: {ds_name} ({len(docs)} docs, {len(queries)} queries)")
        print("=" * 70)
        
        # Check for cached index
        cache_path = get_cache_path(ds_name, limit_docs, use_sparse)
        index = load_index_cache(cache_path)
        
        if index is None:
            # No cache - encode documents
            index = FullIndex()
            print(f"\nIndexing {len(docs)} documents (PIPELINED BATCH MODE)...")
            start_time = time.time()
            
            # Optimized batch settings
            indexed_docs = encoder.encode_documents_batch(
                docs, 
                batch_size=16,
                chunk_size=100,
                compute_hybrid_signs=True,
                keep_float_embeddings=False,
                show_progress=True
            )
            
            # Add to index and build SPLADE inverted index
            for doc_id, d_idx in indexed_docs.items():
                index.docs[doc_id] = d_idx
                for tid, w in zip(d_idx.splade_terms, d_idx.splade_weights):
                    index.splade_inverted[int(tid)].append((doc_id, float(w)))
            
            index_time = time.time() - start_time
            print(f"Indexing completed in {index_time:.1f}s")
            
            # Build BM25 index before caching
            if HAS_BM25:
                build_bm25_index(index)
            
            # Save to cache
            save_index_cache(index, cache_path)
        else:
            # Loaded from cache - rebuild BM25 if needed
            if HAS_BM25 and index.bm25_index is None:
                build_bm25_index(index)
        
        # Storage Metrics
        storage = compute_storage_metrics(index)
        print(f"\n--- Storage Metrics ---")
        print(f"  Documents: {storage['total_docs']}")
        print(f"  Total Tokens: {storage['total_tokens']} ({storage['avg_tokens_per_doc']:.1f} avg/doc)")
        print(f"  Float32:     {storage['float32_kb']:.1f} KB")
        print(f"  Sign Full:   {storage['sign_full_kb']:.1f} KB ({storage['compression_full']:.1f}x compression)")
        print(f"  Sign Hybrid: {storage['sign_hybrid_kb']:.1f} KB ({storage['compression_hybrid']:.1f}x compression)")
        print(f"  Hybrid Savings vs Full: {storage['hybrid_savings_vs_full']:.1f}%")
        
        # Define all methods to test (streamlined)
        # Format: (name, search_fn, needs_splade, needs_query_text)
        methods = [
            ("Dense Pooled", lambda qd, idx: search_pooled(qd, idx), False, False),
            ("Sign Pooled", lambda qd, idx: search_sign_pooled_batch(qd, idx), False, False),
            ("Interp α=0.3", lambda qd, qt, qw, idx: search_interpolated_30(qd, qt, qw, idx), True, False),
        ]
        
        # Add BM25 if available
        if HAS_BM25 and index.bm25_index:
            methods.append(("BM25", lambda qt, qd, _, __, idx: search_bm25(qt, idx), False, True))
        
        # Pre-encode all queries ONCE (major speedup)
        encoded_queries = encode_queries_batch(queries, encoder)
        
        print(f"\n--- Retrieval Evaluation (R@10) ---")
        print(f"  {'Method':<22} | {'R@10':>8} | {'Time':>10}")
        print(f"  {'-'*22}-+-{'-'*8}-+-{'-'*10}")
        
        dataset_results = []
        for name, search_fn, needs_splade, needs_query_text in methods:
            result = evaluate_method(name, search_fn, queries, encoded_queries, relevance, index, needs_splade, needs_query_text)
            dataset_results.append(result)
            print(f"  {result['name']:<22} | {result['recall_at_10']:>7.1%} | {result['avg_time_ms']:>8.1f}ms")
        
        all_results[ds_name] = {
            "storage": storage,
            "methods": dataset_results
        }
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Dataset':<12} | {'Dense':>8} | {'Sign-P':>8} | {'Sign-T':>8} | {'Hybrid':>8} | {'Combined':>8}")
    print("-" * 70)
    for ds_name, data in all_results.items():
        methods = {m["name"]: m["recall_at_10"] for m in data["methods"]}
        print(f"{ds_name:<12} | {methods.get('Dense Pooled', 0):>7.1%} | {methods.get('Sign Pooled', 0):>7.1%} | {methods.get('Sign Token', 0):>7.1%} | {methods.get('Hybrid Pipeline', 0):>7.1%} | {methods.get('Combined Rerank', 0):>7.1%}")
    
    print("\n" + "=" * 70)
    print("SIGN-HASH ANALYSIS")
    print("=" * 70)
    for ds_name, data in all_results.items():
        methods = {m["name"]: m for m in data["methods"]}
        dense_recall = methods.get("Dense Pooled", {}).get("recall_at_10", 0)
        sign_pooled_recall = methods.get("Sign Pooled", {}).get("recall_at_10", 0)
        sign_token_recall = methods.get("Sign Token", {}).get("recall_at_10", 0)
        
        print(f"\n{ds_name}:")
        print(f"  Sign Pooled vs Dense: {sign_pooled_recall/dense_recall*100 if dense_recall > 0 else 0:.1f}% of baseline")
        print(f"  Sign Token vs Dense:  {sign_token_recall/dense_recall*100 if dense_recall > 0 else 0:.1f}% of baseline")
        print(f"  Storage Savings: {data['storage']['compression_full']:.1f}x (Full Sign) / {data['storage']['compression_hybrid']:.1f}x (Hybrid)")


if __name__ == "__main__":
    main()
