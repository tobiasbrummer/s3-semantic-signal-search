#!/usr/bin/env python3
"""
S3 - Semantic Signal Search Pipeline

Die vollständige Hybrid-Pipeline mit allen 4 Such-Modi:
1. Keyword (SPLADE Inverted Index)
2. Semantik (Sign-Hash Matching)
3. Fuzzy (Levenshtein - TODO)
4. Pattern (Cross-Correlation)

Architektur:
- SPLADE-First: Schneller Einstieg über Inverted Index
- Sign-Hash: 32x komprimierte Validierung
- Splines: Pattern-Suche mit Cross-Correlation
"""

import numpy as np
import requests
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from scipy import interpolate, signal
from dataclasses import dataclass, field
from typing import Optional, Union
from collections import defaultdict
import time
import pickle


# =============================================================================
# ENCODERS
# =============================================================================

class DenseEncoder:
    """Dense Embeddings via llama.cpp API."""

    def __init__(self, base_url: str = "http://localhost:8200"):
        self.base_url = base_url
        self.dim = None
        self._init_dim()

    def _init_dim(self):
        test = self.encode(["test"])
        self.dim = len(test[0])

    def encode(self, texts: list[str], batch_size: int = 8) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch = [t[:4000] for t in batch]
            response = requests.post(
                f"{self.base_url}/v1/embeddings",
                json={"input": batch}
            )
            response.raise_for_status()
            data = response.json()
            embeddings = [d["embedding"] for d in data["data"]]
            all_embeddings.extend(embeddings)
        return np.array(all_embeddings)

    def to_sign_hash(self, embeddings: np.ndarray) -> np.ndarray:
        """Konvertiere zu Sign-Hashes (1 bit pro dim, 32x Kompression)."""
        signs = (embeddings > 0).astype(np.uint8)
        n_samples, n_dims = signs.shape
        n_bytes = n_dims // 8
        packed = np.zeros((n_samples, n_bytes), dtype=np.uint8)
        for i in range(8):
            packed |= signs[:, i::8] << i
        return packed


class SpladeEncoder:
    """SPLADE Encoder für Sparse Lexical Embeddings."""

    def __init__(
        self,
        model_name: str = "naver/splade-cocondenser-ensembledistil",
        device: Optional[str] = None,
        top_k: int = 64
    ):
        self.top_k = top_k
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        self.vocab_size = self.tokenizer.vocab_size

    def encode(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        """Returns: (term_ids, weights)"""
        inputs = self.tokenizer(
            text[:4000], return_tensors="pt",
            max_length=512, truncation=True, padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        weights = torch.max(
            torch.log1p(torch.relu(logits)), dim=1
        ).values.squeeze(0).cpu().numpy()

        nonzero_idx = np.nonzero(weights)[0]
        nonzero_weights = weights[nonzero_idx]

        if len(nonzero_idx) > self.top_k:
            top_idx = np.argsort(nonzero_weights)[-self.top_k:]
            nonzero_idx = nonzero_idx[top_idx]
            nonzero_weights = nonzero_weights[top_idx]

        return nonzero_idx, nonzero_weights


# =============================================================================
# INDEX
# =============================================================================

@dataclass
class S3Index:
    """
    Der vollständige S3 Index.

    Speichert:
    - SPLADE Inverted Index
    - Sign-Hashes (32x komprimiert)
    - Spline-Kontrollpunkte (für Pattern-Suche)
    - Dokument-Texte
    """
    # SPLADE: term_id → [(doc_id, pos_idx, char_pos, weight)]
    inverted_index: dict = field(default_factory=lambda: defaultdict(list))

    # Sign-Hashes: doc_id → np.array (n_positions, n_bytes)
    sign_hashes: dict = field(default_factory=dict)

    # Spline-Kontrollpunkte: doc_id → (control_points, control_positions)
    splines: dict = field(default_factory=dict)

    # Dokument-Texte
    doc_texts: dict = field(default_factory=dict)

    # Positionen pro Dokument
    doc_positions: dict = field(default_factory=dict)

    # Metadata
    n_docs: int = 0
    n_dims: int = 1024

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "S3Index":
        with open(path, 'rb') as f:
            return pickle.load(f)

    def get_stats(self) -> dict:
        """Statistiken über den Index."""
        sign_bytes = sum(h.nbytes for h in self.sign_hashes.values())
        spline_bytes = sum(
            cp.nbytes + pos.nbytes
            for cp, pos in self.splines.values()
        )
        inverted_entries = sum(len(v) for v in self.inverted_index.values())

        return {
            "n_docs": self.n_docs,
            "sign_hashes_kb": sign_bytes / 1024,
            "splines_kb": spline_bytes / 1024,
            "inverted_entries": inverted_entries,
            "total_kb": (sign_bytes + spline_bytes) / 1024
        }


# =============================================================================
# INDEXING
# =============================================================================

def build_s3_index(
    docs: list[dict],
    dense_encoder: DenseEncoder,
    splade_encoder: SpladeEncoder,
    window_size: int = 150,
    stride: int = 30,
    spline_downsample: int = 3
) -> S3Index:
    """
    Baue den vollständigen S3 Index.
    """
    index = S3Index(n_dims=dense_encoder.dim)

    for doc_idx, doc in enumerate(docs):
        doc_id = doc["id"]
        text = doc["text"]

        index.doc_texts[doc_id] = text

        # Sliding Windows
        windows = []
        positions = []
        for i in range(0, max(1, len(text) - window_size + 1), stride):
            windows.append(text[i:i+window_size])
            positions.append(i)

        if not windows:
            windows = [text]
            positions = [0]

        positions = np.array(positions)
        index.doc_positions[doc_id] = positions

        # Dense Embeddings
        dense_embs = dense_encoder.encode(windows)

        # Sign-Hashes speichern
        index.sign_hashes[doc_id] = dense_encoder.to_sign_hash(dense_embs)

        # Spline-Kontrollpunkte (downsampled)
        spline_indices = np.arange(0, len(positions), spline_downsample)
        index.splines[doc_id] = (
            dense_embs[spline_indices].copy(),
            positions[spline_indices].copy()
        )

        # SPLADE → Inverted Index
        for pos_idx, (window_text, char_pos) in enumerate(zip(windows, positions)):
            term_ids, weights = splade_encoder.encode(window_text)
            for term_id, weight in zip(term_ids, weights):
                index.inverted_index[int(term_id)].append(
                    (doc_id, pos_idx, int(char_pos), float(weight))
                )

        if (doc_idx + 1) % 100 == 0:
            print(f"   Indexed {doc_idx + 1}/{len(docs)} docs")

    index.n_docs = len(docs)
    return index


# =============================================================================
# SEARCH FUNCTIONS
# =============================================================================

def sign_similarity(hash1: np.ndarray, hash2: np.ndarray) -> float:
    """Hamming-basierte Similarity zwischen Sign-Hashes."""
    xor = np.bitwise_xor(hash1, hash2)
    diff_bits = np.unpackbits(xor).sum()
    total_bits = len(hash1) * 8
    return 1.0 - (diff_bits / total_bits)


def keyword_search(
    query: str,
    index: S3Index,
    dense_encoder: DenseEncoder,
    splade_encoder: SpladeEncoder,
    top_k: int = 10,
    splade_candidates: int = 100,
    sign_threshold: float = 0.55
) -> list[dict]:
    """
    SPLADE-First Keyword Search mit Sign-Hash Validierung.
    """
    # Query encodieren
    query_dense = dense_encoder.encode([query])[0]
    query_sign = dense_encoder.to_sign_hash(query_dense.reshape(1, -1))[0]
    query_terms, query_weights = splade_encoder.encode(query)

    # SPLADE Kandidaten
    candidates = defaultdict(float)
    for term_id, term_weight in zip(query_terms, query_weights):
        if term_id in index.inverted_index:
            for doc_id, pos_idx, char_pos, doc_weight in index.inverted_index[term_id]:
                candidates[(doc_id, pos_idx, char_pos)] += term_weight * doc_weight

    # Top-N nach SPLADE Score
    sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    top_candidates = sorted_candidates[:splade_candidates]

    # Sign-Hash Filtering
    results = []
    for (doc_id, pos_idx, char_pos), splade_score in top_candidates:
        doc_sign = index.sign_hashes[doc_id][pos_idx]
        sign_sim = sign_similarity(query_sign, doc_sign)

        if sign_sim < sign_threshold:
            continue

        combined_score = 0.3 * sign_sim + 0.7 * (splade_score / 50)

        text = index.doc_texts[doc_id]
        snippet = text[char_pos:char_pos + 100].replace("\n", " ")

        results.append({
            "doc_id": doc_id,
            "position": char_pos,
            "score": combined_score,
            "snippet": snippet,
            "method": "keyword"
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def semantic_search(
    query: str,
    index: S3Index,
    dense_encoder: DenseEncoder,
    top_k: int = 10,
    candidates_per_doc: int = 3
) -> list[dict]:
    """
    Reine Sign-Hash basierte semantische Suche.
    Durchsucht alle Dokumente (langsamer, aber vollständig).
    """
    query_dense = dense_encoder.encode([query])[0]
    query_sign = dense_encoder.to_sign_hash(query_dense.reshape(1, -1))[0]

    results = []
    for doc_id, doc_signs in index.sign_hashes.items():
        # Similarity für jede Position
        scores = []
        for pos_idx in range(len(doc_signs)):
            sim = sign_similarity(query_sign, doc_signs[pos_idx])
            scores.append((pos_idx, sim))

        # Top-N pro Dokument
        scores.sort(key=lambda x: x[1], reverse=True)
        for pos_idx, score in scores[:candidates_per_doc]:
            positions = index.doc_positions[doc_id]
            char_pos = positions[pos_idx] if pos_idx < len(positions) else 0

            text = index.doc_texts[doc_id]
            snippet = text[char_pos:char_pos + 100].replace("\n", " ")

            results.append({
                "doc_id": doc_id,
                "position": char_pos,
                "score": score,
                "snippet": snippet,
                "method": "semantic"
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def pattern_search(
    pattern: np.ndarray,
    index: S3Index,
    top_k: int = 10
) -> list[dict]:
    """
    Cross-Correlation basierte Pattern-Suche.

    Args:
        pattern: Shape (pattern_len, n_dims) - das gesuchte Pattern
    """
    pattern_len = len(pattern)
    results = []

    for doc_id, (control_points, control_positions) in index.splines.items():
        if len(control_points) < pattern_len:
            continue

        # Cross-Correlation für jede Dimension, dann summieren
        correlations = np.zeros(len(control_points) - pattern_len + 1)

        for dim in range(min(pattern.shape[1], control_points.shape[1])):
            p = pattern[:, dim]
            d = control_points[:, dim]

            p = (p - p.mean()) / (p.std() + 1e-10)
            d = (d - d.mean()) / (d.std() + 1e-10)

            corr = signal.correlate(d, p, mode='valid')
            correlations += corr

        correlations /= (pattern.shape[1] * pattern_len)

        # Beste Position
        best_idx = np.argmax(correlations)
        best_score = correlations[best_idx]

        if best_idx < len(control_positions):
            char_pos = control_positions[best_idx]
        else:
            char_pos = 0

        text = index.doc_texts[doc_id]
        snippet = text[char_pos:char_pos + 100].replace("\n", " ")

        results.append({
            "doc_id": doc_id,
            "position": int(char_pos),
            "score": float(best_score),
            "snippet": snippet,
            "method": "pattern"
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


# =============================================================================
# UNIFIED SEARCH
# =============================================================================

def search(
    query: Union[str, np.ndarray],
    index: S3Index,
    dense_encoder: DenseEncoder,
    splade_encoder: SpladeEncoder,
    mode: str = "auto",
    top_k: int = 10
) -> list[dict]:
    """
    Unified Search Interface.

    Args:
        query: Text (für keyword/semantic) oder Pattern-Array
        mode: "keyword", "semantic", "pattern", oder "auto"
    """
    if isinstance(query, np.ndarray):
        return pattern_search(query, index, top_k)

    if mode == "auto":
        # Heuristik: Kurze Queries → Keyword, lange → Semantic
        mode = "keyword" if len(query.split()) <= 5 else "semantic"

    if mode == "keyword":
        return keyword_search(query, index, dense_encoder, splade_encoder, top_k)
    elif mode == "semantic":
        return semantic_search(query, index, dense_encoder, top_k)
    else:
        raise ValueError(f"Unknown mode: {mode}")


# =============================================================================
# EVALUATION
# =============================================================================

def load_scifact(n_docs: int = 5000, n_queries: int = 300):
    """Lade SciFact Testdaten."""
    from datasets import load_dataset

    corpus = load_dataset("mteb/scifact", "corpus", split="corpus")
    queries_ds = load_dataset("mteb/scifact", "queries", split="queries")
    qrels_ds = load_dataset("mteb/scifact", "default", split="test")

    qrels = defaultdict(set)
    for item in qrels_ds:
        qrels[item["query-id"]].add(item["corpus-id"])

    docs = []
    doc_id_set = set()
    for i, item in enumerate(corpus):
        if i >= n_docs:
            break
        doc_id = item["_id"]
        docs.append({
            "id": doc_id,
            "text": f"{item['title']} {item['text']}"
        })
        doc_id_set.add(doc_id)

    queries = []
    relevance = {}
    for item in queries_ds:
        query_id = item["_id"]
        if query_id in qrels:
            relevant = qrels[query_id] & doc_id_set
            if relevant:
                queries.append({
                    "id": query_id,
                    "text": item["text"]
                })
                relevance[query_id] = list(relevant)
        if len(queries) >= n_queries:
            break

    return docs, queries, relevance


def evaluate(
    queries: list[dict],
    relevance: dict,
    index: S3Index,
    dense_encoder: DenseEncoder,
    splade_encoder: SpladeEncoder,
    mode: str = "keyword",
    k: int = 10
) -> tuple[float, float]:
    """Evaluiere Recall@K und Zeit."""
    hits = 0
    total = 0
    total_time = 0

    for query in queries:
        query_id = query["id"]
        if query_id not in relevance:
            continue

        relevant_docs = set(relevance[query_id])

        start = time.time()
        results = search(query["text"], index, dense_encoder, splade_encoder, mode=mode, top_k=k)
        total_time += time.time() - start

        found_docs = set(r["doc_id"] for r in results)
        if relevant_docs & found_docs:
            hits += 1
        total += 1

    recall = hits / total if total > 0 else 0
    avg_time = total_time / total if total > 0 else 0
    return recall, avg_time


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("S3 - SEMANTIC SIGNAL SEARCH PIPELINE")
    print("=" * 70)

    # Encoder
    print("\n1. Lade Encoder...")
    dense_encoder = DenseEncoder()
    print(f"   Dense dim: {dense_encoder.dim}")

    splade_encoder = SpladeEncoder(top_k=64)
    print(f"   SPLADE vocab: {splade_encoder.vocab_size}")

    # Daten
    print("\n2. Lade SciFact...")
    n_docs = 1000  # Reduziert für schnelleren Test
    n_queries = 100
    docs, queries, relevance = load_scifact(n_docs, n_queries)
    print(f"   {len(docs)} Dokumente, {len(queries)} Queries")

    # Index bauen
    print("\n3. Baue S3 Index...")
    start = time.time()
    index = build_s3_index(docs, dense_encoder, splade_encoder)
    index_time = time.time() - start
    print(f"   Index gebaut in {index_time:.1f}s")

    stats = index.get_stats()
    print(f"   Sign-Hashes: {stats['sign_hashes_kb']:.1f} KB")
    print(f"   Splines: {stats['splines_kb']:.1f} KB")
    print(f"   Inverted Index: {stats['inverted_entries']} Einträge")
    print(f"   Total: {stats['total_kb']:.1f} KB")

    # Evaluation
    print("\n4. Evaluiere Such-Modi...")
    print("=" * 70)

    for mode in ["keyword", "semantic"]:
        recall, avg_time = evaluate(
            queries, relevance, index, dense_encoder, splade_encoder,
            mode=mode, k=10
        )
        print(f"   {mode:12s}: Recall@10 = {recall*100:.1f}%, "
              f"Avg Time = {avg_time*1000:.1f}ms")

    # Demo
    print("\n" + "=" * 70)
    print("DEMO SEARCHES")
    print("=" * 70)

    demo_queries = [
        "gene therapy cancer treatment",
        "diabetes insulin resistance",
        "neural network machine learning"
    ]

    for q in demo_queries:
        print(f"\n   Query: \"{q}\"")
        results = search(q, index, dense_encoder, splade_encoder, mode="keyword", top_k=3)
        for i, r in enumerate(results):
            print(f"   {i+1}. [{r['doc_id']}] score={r['score']:.3f}")
            print(f"      \"{r['snippet'][:60]}...\"")

    print("\n" + "=" * 70)
    print("S3 PIPELINE READY")
    print("=" * 70)


if __name__ == "__main__":
    main()
