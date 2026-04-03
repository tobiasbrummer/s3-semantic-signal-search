#!/usr/bin/env python3
"""
S3 Realistic Benchmark

Tests S3 (Semantic Signal Search) with:
- Model: intfloat/multilingual-e5-large (1024D, multilingual)
- Dataset: BEIR subset (SciFact or FiQA)

Compares:
- S3 (LSH + Inverted Index) vs Brute Force Cosine

Author: Claude & Toby
Date: December 2024
"""

import numpy as np
import time
import json
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import hashlib

# Check dependencies
try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False
    print("⚠️  pip install sentence-transformers")

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("⚠️  pip install datasets")


# =============================================================================
# S3 CONFIGURATION (1024D optimized)
# =============================================================================

@dataclass
class S3Config1024:
    """
    Configuration for 1024-dimensional embeddings.
    
    Band layout optimized for E5-large (1024D):
    - Bass: First 128 dims (most important for topic)
    - Mids: 128-512 (context)  
    - Highs: 512-1024 (nuance)
    """
    embedding_dim: int = 1024
    
    # Band definitions: (start, end, bits, weight)
    bands: List[Tuple[int, int, int, float]] = field(default_factory=lambda: [
        (0, 128, 64, 8.0),      # Bass: 128 dims → 64 bits
        (128, 512, 128, 4.0),   # Mids: 384 dims → 128 bits
        (512, 1024, 256, 1.0),  # Highs: 512 dims → 256 bits
    ])
    
    lsh_seed: int = 42
    
    @property
    def total_bits(self) -> int:
        return sum(b[2] for b in self.bands)
    
    @property
    def total_bytes(self) -> int:
        return self.total_bits // 8


# =============================================================================
# NORMALIZER
# =============================================================================

class Normalizer:
    @staticmethod
    def normalize(vectors: np.ndarray) -> np.ndarray:
        if vectors.ndim == 1:
            norm = np.linalg.norm(vectors)
            return vectors / max(norm, 1e-10)
        else:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            return vectors / np.maximum(norms, 1e-10)


# =============================================================================
# MULTI-BAND LSH
# =============================================================================

class MultiBandLSH:
    def __init__(self, config: S3Config1024):
        self.config = config
        self.projections = {}
        
        rng = np.random.RandomState(config.lsh_seed)
        
        for i, (start, end, bits, weight) in enumerate(config.bands):
            input_dim = end - start
            self.projections[i] = rng.randn(bits, input_dim).astype(np.float32)
    
    def hash_vector(self, vector: np.ndarray) -> Dict[int, bytes]:
        vector = Normalizer.normalize(vector)
        result = {}
        
        for i, (start, end, bits, weight) in enumerate(self.config.bands):
            band_vector = vector[start:end]
            dot_products = self.projections[i] @ band_vector
            bits_array = (dot_products > 0).astype(np.uint8)
            result[i] = bytes(np.packbits(bits_array))
        
        return result
    
    def hash_batch(self, vectors: np.ndarray) -> List[Dict[int, bytes]]:
        vectors = Normalizer.normalize(vectors)
        return [self.hash_vector(v) for v in vectors]


# =============================================================================
# HAMMING SCORER
# =============================================================================

class HammingScorer:
    def __init__(self, config: S3Config1024):
        self.config = config
        self.weights = {i: w for i, (_, _, _, w) in enumerate(config.bands)}
    
    def hamming_distance(self, a: bytes, b: bytes) -> int:
        return sum(bin(x ^ y).count('1') for x, y in zip(a, b))
    
    def weighted_distance(self, 
                          hash_a: Dict[int, bytes],
                          hash_b: Dict[int, bytes]) -> float:
        total = 0.0
        for band_id, weight in self.weights.items():
            if band_id in hash_a and band_id in hash_b:
                dist = self.hamming_distance(hash_a[band_id], hash_b[band_id])
                total += weight * dist
        return total


# =============================================================================
# S3 INDEX
# =============================================================================

class S3Index:
    def __init__(self, config: S3Config1024):
        self.config = config
        self.lsh = MultiBandLSH(config)
        self.scorer = HammingScorer(config)
        
        # Storage
        self.embeddings = {}  # doc_id → embedding
        self.hashes = {}      # doc_id → hash dict
        self.texts = {}       # doc_id → text
        
        # Inverted index for band 0 (bass)
        self.bass_index = defaultdict(list)
    
    def add(self, doc_id: str, embedding: np.ndarray, text: str = ""):
        self.embeddings[doc_id] = embedding
        self.texts[doc_id] = text
        
        # Hash
        doc_hash = self.lsh.hash_vector(embedding)
        self.hashes[doc_id] = doc_hash
        
        # Add to bass index
        bass_key = doc_hash[0].hex()
        self.bass_index[bass_key].append(doc_id)
    
    def search_s3(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        S3 Search: Two-pass (coarse + fine)
        """
        query_hash = self.lsh.hash_vector(query_embedding)
        bass_key = query_hash[0].hex()
        
        # Stage 1: Coarse filter using bass index
        candidates = set()
        
        # Exact match
        candidates.update(self.bass_index.get(bass_key, []))
        
        # Fuzzy match (1-bit flips in bass)
        if len(candidates) < top_k * 10:
            bass_bytes = bytes.fromhex(bass_key)
            for byte_idx in range(len(bass_bytes)):
                for bit_idx in range(8):
                    modified = bytearray(bass_bytes)
                    modified[byte_idx] ^= (1 << bit_idx)
                    modified_key = bytes(modified).hex()
                    candidates.update(self.bass_index.get(modified_key, []))
        
        # If still too few, fall back to all
        if len(candidates) < top_k:
            candidates = set(self.hashes.keys())
        
        # Stage 2: Fine ranking with weighted Hamming
        scored = []
        for doc_id in candidates:
            dist = self.scorer.weighted_distance(query_hash, self.hashes[doc_id])
            scored.append((doc_id, dist))
        
        scored.sort(key=lambda x: x[1])
        return scored[:top_k]
    
    def search_brute_force(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Brute force cosine similarity (baseline)
        """
        query_norm = Normalizer.normalize(query_embedding)
        
        scored = []
        for doc_id, doc_emb in self.embeddings.items():
            doc_norm = Normalizer.normalize(doc_emb)
            sim = np.dot(query_norm, doc_norm)
            scored.append((doc_id, -sim))  # Negative so lower = better
        
        scored.sort(key=lambda x: x[1])
        return scored[:top_k]


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Compute Recall@K"""
    retrieved_k = set(retrieved[:k])
    if not relevant:
        return 0.0
    return len(retrieved_k & relevant) / len(relevant)


def mrr(retrieved: List[str], relevant: Set[str]) -> float:
    """Mean Reciprocal Rank"""
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


# =============================================================================
# DATASET LOADING
# =============================================================================

def load_scifact():
    """Load SciFact dataset from BEIR (small, good for testing)"""
    if not HAS_DATASETS:
        raise ImportError("pip install datasets")
    
    print("Loading SciFact dataset...")
    
    # Load corpus
    corpus = load_dataset("mteb/scifact", "corpus", split="corpus")
    
    # Load queries
    queries = load_dataset("mteb/scifact", "queries", split="queries")
    
    # Load qrels (relevance judgments)
    qrels_data = load_dataset("mteb/scifact", "default", split="test")
    
    # Build qrels dict
    qrels = defaultdict(set)
    for item in qrels_data:
        qrels[item["query-id"]].add(item["corpus-id"])
    
    return corpus, queries, qrels


def load_fiqa():
    """Load FiQA dataset (financial QA)"""
    if not HAS_DATASETS:
        raise ImportError("pip install datasets")
    
    print("Loading FiQA dataset...")
    
    corpus = load_dataset("mteb/fiqa", "corpus", split="corpus")
    queries = load_dataset("mteb/fiqa", "queries", split="queries")
    qrels_data = load_dataset("mteb/fiqa", "default", split="test")
    
    qrels = defaultdict(set)
    for item in qrels_data:
        qrels[item["query-id"]].add(item["corpus-id"])
    
    return corpus, queries, qrels


# =============================================================================
# MAIN BENCHMARK
# =============================================================================

def run_benchmark(
    model_name: str = "jinaai/jina-embeddings-v3",
    dataset: str = "scifact",
    max_corpus: int = 5000,
    max_queries: int = 300,
):
    """
    Run S3 benchmark.
    
    Args:
        model_name: HuggingFace model name
        dataset: "scifact" or "fiqa"
        max_corpus: Maximum number of documents
        max_queries: Maximum number of queries
    """
    print("=" * 70)
    print("S3 REALISTIC BENCHMARK")
    print("=" * 70)
    
    # 1. Load model
    print(f"\n1. Loading model: {model_name}")
    if not HAS_ST:
        print("   ❌ sentence-transformers not installed")
        print("   Run: pip install sentence-transformers")
        return
    
    start = time.time()
    model = SentenceTransformer(model_name)
    print(f"   Loaded in {time.time() - start:.1f}s")
    print(f"   Embedding dimension: {model.get_sentence_embedding_dimension()}")
    
    # 2. Load dataset
    print(f"\n2. Loading dataset: {dataset}")
    if dataset == "scifact":
        corpus, queries, qrels = load_scifact()
    elif dataset == "fiqa":
        corpus, queries, qrels = load_fiqa()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Limit size
    corpus_list = list(corpus)[:max_corpus]
    query_list = [q for q in queries if q["_id"] in qrels][:max_queries]
    
    print(f"   Corpus: {len(corpus_list)} documents")
    print(f"   Queries: {len(query_list)} queries")
    print(f"   Queries with relevance judgments: {len(qrels)}")
    
    # 3. Encode corpus
    print(f"\n3. Encoding corpus...")
    
    # Format documents with prefix
    corpus_texts = [f"passage: {doc['title']} {doc['text']}" for doc in corpus_list]
    
    start = time.time()
    corpus_embeddings = model.encode(
        corpus_texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    encode_time = time.time() - start
    print(f"   Encoded in {encode_time:.1f}s ({len(corpus_list)/encode_time:.1f} docs/s)")
    
    # 4. Build S3 index
    print(f"\n4. Building S3 index...")
    
    config = S3Config1024()
    print(f"   Config: {config.total_bits} bits per document ({config.total_bytes} bytes)")
    print(f"   Bands: {len(config.bands)}")
    for i, (start, end, bits, weight) in enumerate(config.bands):
        print(f"      Band {i}: dims [{start}:{end}] → {bits} bits (weight={weight})")
    
    index = S3Index(config)
    
    start = time.time()
    for i, doc in enumerate(corpus_list):
        index.add(doc["_id"], corpus_embeddings[i], corpus_texts[i])
    index_time = time.time() - start
    print(f"   Indexed in {index_time:.1f}s ({len(corpus_list)/index_time:.1f} docs/s)")
    
    # 5. Encode queries
    print(f"\n5. Encoding queries...")
    
    query_texts = [f"query: {q['text']}" for q in query_list]
    
    start = time.time()
    query_embeddings = model.encode(
        query_texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    print(f"   Encoded in {time.time() - start:.1f}s")
    
    # 6. Evaluate
    print(f"\n6. Evaluating...")
    
    k_values = [1, 5, 10, 20]
    
    # S3 metrics
    s3_recalls = {k: [] for k in k_values}
    s3_mrrs = []
    s3_times = []
    
    # Brute force metrics
    bf_recalls = {k: [] for k in k_values}
    bf_mrrs = []
    bf_times = []
    
    for i, query in enumerate(query_list):
        query_id = query["_id"]
        query_emb = query_embeddings[i]
        relevant = qrels.get(query_id, set())
        
        if not relevant:
            continue
        
        # S3 search
        start = time.time()
        s3_results = index.search_s3(query_emb, top_k=max(k_values))
        s3_times.append(time.time() - start)
        s3_retrieved = [doc_id for doc_id, _ in s3_results]
        
        for k in k_values:
            s3_recalls[k].append(recall_at_k(s3_retrieved, relevant, k))
        s3_mrrs.append(mrr(s3_retrieved, relevant))
        
        # Brute force search
        start = time.time()
        bf_results = index.search_brute_force(query_emb, top_k=max(k_values))
        bf_times.append(time.time() - start)
        bf_retrieved = [doc_id for doc_id, _ in bf_results]
        
        for k in k_values:
            bf_recalls[k].append(recall_at_k(bf_retrieved, relevant, k))
        bf_mrrs.append(mrr(bf_retrieved, relevant))
    
    # 7. Results
    print(f"\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\n{'Metric':<20} {'S3':>15} {'Brute Force':>15} {'S3 vs BF':>15}")
    print("-" * 70)
    
    for k in k_values:
        s3_r = np.mean(s3_recalls[k]) * 100
        bf_r = np.mean(bf_recalls[k]) * 100
        ratio = s3_r / bf_r * 100 if bf_r > 0 else 0
        print(f"Recall@{k:<14} {s3_r:>14.1f}% {bf_r:>14.1f}% {ratio:>14.1f}%")
    
    s3_mrr = np.mean(s3_mrrs) * 100
    bf_mrr = np.mean(bf_mrrs) * 100
    print(f"{'MRR':<20} {s3_mrr:>14.1f}% {bf_mrr:>14.1f}% {s3_mrr/bf_mrr*100 if bf_mrr > 0 else 0:>14.1f}%")
    
    s3_avg_time = np.mean(s3_times) * 1000
    bf_avg_time = np.mean(bf_times) * 1000
    print(f"{'Avg Query Time':<20} {s3_avg_time:>13.2f}ms {bf_avg_time:>13.2f}ms {bf_avg_time/s3_avg_time:>14.1f}x")
    
    # 8. Storage comparison
    print(f"\n" + "-" * 70)
    print("STORAGE")
    print("-" * 70)
    
    float_bytes = config.embedding_dim * 4 * len(corpus_list)
    s3_bytes = config.total_bytes * len(corpus_list)
    
    print(f"Float embeddings: {float_bytes / 1e6:.2f} MB")
    print(f"S3 hashes:        {s3_bytes / 1e6:.2f} MB")
    print(f"Compression:      {float_bytes / s3_bytes:.1f}x")
    
    # 9. Summary
    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    recall_ratio = np.mean(s3_recalls[10]) / np.mean(bf_recalls[10]) if np.mean(bf_recalls[10]) > 0 else 0
    
    if recall_ratio >= 0.95:
        print(f"✅ S3 achieves {recall_ratio*100:.1f}% of brute-force Recall@10")
    elif recall_ratio >= 0.85:
        print(f"🔶 S3 achieves {recall_ratio*100:.1f}% of brute-force Recall@10 (acceptable)")
    else:
        print(f"❌ S3 achieves only {recall_ratio*100:.1f}% of brute-force Recall@10 (needs tuning)")
    
    print(f"⚡ S3 is {bf_avg_time/s3_avg_time:.1f}x faster than brute-force")
    print(f"💾 S3 uses {float_bytes/s3_bytes:.1f}x less storage")
    
    return {
        "s3_recalls": {k: np.mean(v) for k, v in s3_recalls.items()},
        "bf_recalls": {k: np.mean(v) for k, v in bf_recalls.items()},
        "s3_mrr": s3_mrr,
        "bf_mrr": bf_mrr,
        "s3_avg_time_ms": s3_avg_time,
        "bf_avg_time_ms": bf_avg_time,
        "compression": float_bytes / s3_bytes,
    }


# =============================================================================
# QUICK TEST (without full dataset)
# =============================================================================

def quick_test():
    """
    Quick test without downloading datasets.
    Uses synthetic data to verify S3 works.
    """
    print("=" * 70)
    print("S3 QUICK TEST (Synthetic Data)")
    print("=" * 70)
    
    if not HAS_ST:
        print("⚠️  sentence-transformers not installed. Using random embeddings.")
        use_real_model = False
    else:
        use_real_model = True
    
    # Test texts
    documents = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with many layers.",
        "Natural language processing helps computers understand text.",
        "Computer vision enables machines to interpret images.",
        "Reinforcement learning trains agents through rewards.",
        "The stock market showed strong gains today.",
        "Financial analysts predict economic growth.",
        "Bond yields increased amid inflation concerns.",
        "Cryptocurrency prices remain volatile.",
        "Central banks consider interest rate changes.",
        "The new smartphone features an improved camera.",
        "Battery technology advances enable longer usage.",
        "5G networks provide faster connectivity.",
        "Cloud computing transforms business operations.",
        "Cybersecurity threats continue to evolve.",
    ]
    
    queries = [
        "What is artificial intelligence?",
        "How do neural networks work?",
        "Tell me about stock market performance.",
        "Latest technology innovations.",
    ]
    
    # Expected relevant docs (approximate)
    relevance = {
        0: {0, 1, 2, 3, 4},  # AI query → AI docs
        1: {1, 2},           # Neural networks → DL, NLP
        2: {5, 6, 7, 8, 9},  # Stock market → Finance docs
        3: {10, 11, 12, 13, 14},  # Tech → Tech docs
    }
    
    # Encode
    print("\n1. Encoding...")
    
    if use_real_model:
        model = SentenceTransformer("intfloat/multilingual-e5-large")
        
        doc_texts = [f"passage: {d}" for d in documents]
        query_texts = [f"query: {q}" for q in queries]
        
        doc_embeddings = model.encode(doc_texts, normalize_embeddings=True)
        query_embeddings = model.encode(query_texts, normalize_embeddings=True)
        
        dim = model.get_sentence_embedding_dimension()
    else:
        # Random embeddings for testing
        dim = 1024
        np.random.seed(42)
        
        # Make similar docs have similar embeddings
        base_ai = np.random.randn(dim)
        base_finance = np.random.randn(dim)
        base_tech = np.random.randn(dim)
        
        doc_embeddings = np.array([
            base_ai + np.random.randn(dim) * 0.1,  # ML
            base_ai + np.random.randn(dim) * 0.1,  # DL
            base_ai + np.random.randn(dim) * 0.1,  # NLP
            base_ai + np.random.randn(dim) * 0.1,  # CV
            base_ai + np.random.randn(dim) * 0.1,  # RL
            base_finance + np.random.randn(dim) * 0.1,  # Stock
            base_finance + np.random.randn(dim) * 0.1,  # Finance
            base_finance + np.random.randn(dim) * 0.1,  # Bond
            base_finance + np.random.randn(dim) * 0.1,  # Crypto
            base_finance + np.random.randn(dim) * 0.1,  # Central bank
            base_tech + np.random.randn(dim) * 0.1,  # Smartphone
            base_tech + np.random.randn(dim) * 0.1,  # Battery
            base_tech + np.random.randn(dim) * 0.1,  # 5G
            base_tech + np.random.randn(dim) * 0.1,  # Cloud
            base_tech + np.random.randn(dim) * 0.1,  # Cyber
        ])
        
        query_embeddings = np.array([
            base_ai + np.random.randn(dim) * 0.15,
            base_ai + np.random.randn(dim) * 0.15,
            base_finance + np.random.randn(dim) * 0.15,
            base_tech + np.random.randn(dim) * 0.15,
        ])
    
    print(f"   Embedding dimension: {dim}")
    
    # Build index
    print("\n2. Building S3 index...")
    
    config = S3Config1024()
    index = S3Index(config)
    
    for i, emb in enumerate(doc_embeddings):
        index.add(str(i), emb, documents[i])
    
    print(f"   Indexed {len(documents)} documents")
    print(f"   Bits per document: {config.total_bits}")
    
    # Search
    print("\n3. Searching...")
    
    for i, query in enumerate(queries):
        print(f"\n   Query: '{query}'")
        
        # S3 search
        s3_results = index.search_s3(query_embeddings[i], top_k=5)
        s3_ids = [int(doc_id) for doc_id, _ in s3_results]
        
        # Brute force
        bf_results = index.search_brute_force(query_embeddings[i], top_k=5)
        bf_ids = [int(doc_id) for doc_id, _ in bf_results]
        
        # Check relevance
        relevant = relevance[i]
        s3_hits = len(set(s3_ids) & relevant)
        bf_hits = len(set(bf_ids) & relevant)
        
        print(f"   S3 Top-5: {s3_ids} (hits: {s3_hits}/{len(relevant)})")
        print(f"   BF Top-5: {bf_ids} (hits: {bf_hits}/{len(relevant)})")
    
    print("\n" + "=" * 70)
    print("QUICK TEST COMPLETE")
    print("=" * 70)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        # Full benchmark with BEIR dataset
        run_benchmark(
            model_name="intfloat/multilingual-e5-large",
            dataset="scifact",
            max_corpus=5000,
            max_queries=300,
        )
    else:
        # Quick test
        quick_test()
