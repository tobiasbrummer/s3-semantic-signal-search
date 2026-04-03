#!/usr/bin/env python3
"""
Matryoshka MRL Test

Testet die 2D-Hierarchie:
- Sequenz-Achse: Doc → Segment → Token (Multi-Resolution)
- Dimensions-Achse: 256 → 512 → 1024 (Matryoshka)

Hypothese: Kombinierte Hierarchie ermöglicht noch schnelleres Retrieval
bei gleichem Recall.
"""

import numpy as np
import requests
import time
from dataclasses import dataclass, field
from collections import defaultdict
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from datasets import load_dataset


# =============================================================================
# ENCODER
# =============================================================================

class TokenEncoder:
    def __init__(self, url: str = "http://localhost:8202"):
        self.url = url
        self.full_dim = 1024

    def encode(self, text: str) -> np.ndarray:
        text = text[:4000]
        response = requests.post(f"{self.url}/embeddings", json={"input": text})
        response.raise_for_status()
        data = response.json()
        if isinstance(data[0]["embedding"][0], list):
            return np.array(data[0]["embedding"])
        return np.array([data[0]["embedding"]])


class PooledEncoder:
    def __init__(self, url: str = "http://localhost:8200"):
        self.url = url

    def encode(self, text: str) -> np.ndarray:
        response = requests.post(
            f"{self.url}/v1/embeddings",
            json={"input": [text[:4000]]}
        )
        response.raise_for_status()
        return np.array(response.json()["data"][0]["embedding"])


# =============================================================================
# MRL TRUNCATION
# =============================================================================

def truncate_mrl(embeddings: np.ndarray, dims: int) -> np.ndarray:
    """Truncate embeddings to first N dimensions (Matryoshka style)."""
    if embeddings.ndim == 1:
        return embeddings[:dims]
    return embeddings[:, :dims]


# =============================================================================
# ONSET DETECTION
# =============================================================================

def spectral_flux(embeddings: np.ndarray) -> np.ndarray:
    changes = np.abs(np.diff(embeddings, axis=0))
    return changes.sum(axis=1)


def find_onsets(onset_signal: np.ndarray) -> np.ndarray:
    if len(onset_signal) < 3:
        return np.array([])
    smoothed = gaussian_filter1d(onset_signal, sigma=2.0)
    threshold = np.percentile(smoothed, 95)
    peaks, _ = find_peaks(smoothed, height=threshold, distance=3)
    return peaks


def build_onset_segments(embeddings: np.ndarray) -> tuple[np.ndarray, list]:
    flux = spectral_flux(embeddings)
    onsets = find_onsets(flux)

    boundaries = [0] + sorted(onsets.tolist()) + [len(embeddings)]
    segments = []
    ranges = []

    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i+1]
        if end > start:
            segments.append(embeddings[start:end].mean(axis=0))
            ranges.append((start, end))

    if not segments:
        return embeddings.mean(axis=0, keepdims=True), [(0, len(embeddings))]

    return np.array(segments), ranges


# =============================================================================
# INDEX
# =============================================================================

@dataclass
class DocumentIndex:
    doc_id: str
    token_embeddings: np.ndarray  # Full 1024 dims
    doc_embedding: np.ndarray
    segment_embeddings: np.ndarray
    segment_ranges: list


@dataclass
class FullIndex:
    docs: dict = field(default_factory=dict)


def build_index(docs: list[dict], token_encoder: TokenEncoder) -> FullIndex:
    index = FullIndex()

    for i, doc in enumerate(docs):
        doc_id = doc["id"]
        text = doc["text"]

        token_embs = token_encoder.encode(text)
        doc_emb = token_embs.mean(axis=0)
        seg_embs, seg_ranges = build_onset_segments(token_embs)

        index.docs[doc_id] = DocumentIndex(
            doc_id=doc_id,
            token_embeddings=token_embs,
            doc_embedding=doc_emb,
            segment_embeddings=seg_embs,
            segment_ranges=seg_ranges
        )

        if (i + 1) % 50 == 0:
            print(f"    Indexed {i+1}/{len(docs)}")

    return index


# =============================================================================
# SEARCH METHODS
# =============================================================================

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)


def search_mrl_hierarchical(
    query_emb: np.ndarray,
    index: FullIndex,
    dims_stage1: int = 256,
    dims_stage2: int = 512,
    dims_stage3: int = 1024,
    doc_k: int = 100,
    seg_k: int = 50,
    top_k: int = 10
) -> list:
    """
    2D-Hierarchische Suche mit MRL:
    Stage 1: Doc-Level @ dims_stage1
    Stage 2: Segment-Level @ dims_stage2
    Stage 3: Token-Level @ dims_stage3
    """

    # Stage 1: Doc-Level mit reduzierten Dimensionen
    query_s1 = truncate_mrl(query_emb, dims_stage1)
    doc_scores = []
    for doc_id, doc in index.docs.items():
        doc_emb_s1 = truncate_mrl(doc.doc_embedding, dims_stage1)
        score = cosine_sim(query_s1, doc_emb_s1)
        doc_scores.append((doc_id, score))

    doc_scores.sort(key=lambda x: x[1], reverse=True)
    top_docs = [d[0] for d in doc_scores[:doc_k]]

    # Stage 2: Segment-Level mit mittleren Dimensionen
    query_s2 = truncate_mrl(query_emb, dims_stage2)
    seg_scores = []
    for doc_id in top_docs:
        doc = index.docs[doc_id]
        for seg_idx, seg_emb in enumerate(doc.segment_embeddings):
            seg_emb_s2 = truncate_mrl(seg_emb, dims_stage2)
            score = cosine_sim(query_s2, seg_emb_s2)
            seg_scores.append((doc_id, seg_idx, score))

    seg_scores.sort(key=lambda x: x[2], reverse=True)
    top_segs = seg_scores[:seg_k]

    # Stage 3: Token-Level mit vollen Dimensionen (nur Top-3 Segments pro Doc)
    query_s3 = truncate_mrl(query_emb, dims_stage3)
    results = []
    seen = set()

    for doc_id, seg_idx, _ in top_segs:
        if doc_id in seen:
            continue

        doc = index.docs[doc_id]

        # Token-Level in diesem Segment
        start, end = doc.segment_ranges[seg_idx]
        best_score = 0
        for t in range(start, end):
            tok_emb = truncate_mrl(doc.token_embeddings[t], dims_stage3)
            score = cosine_sim(query_s3, tok_emb)
            if score > best_score:
                best_score = score

        results.append((doc_id, best_score))
        seen.add(doc_id)

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def search_full_dims(query_emb: np.ndarray, index: FullIndex, doc_k: int = 100, top_k: int = 10) -> list:
    """Baseline: Volle Dimensionen auf allen Stufen."""
    return search_mrl_hierarchical(
        query_emb, index,
        dims_stage1=1024, dims_stage2=1024, dims_stage3=1024,
        doc_k=doc_k, top_k=top_k
    )


def search_token_bf(query_emb: np.ndarray, index: FullIndex, dims: int = 1024, top_k: int = 10) -> list:
    """Token Brute Force als Referenz."""
    query = truncate_mrl(query_emb, dims)
    results = []
    for doc_id, doc in index.docs.items():
        best_score = max(
            cosine_sim(query, truncate_mrl(t, dims))
            for t in doc.token_embeddings
        )
        results.append((doc_id, best_score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(queries, relevance, search_fn, index, pooled_encoder, top_k=10):
    hits = 0
    total = 0
    total_time = 0

    for query in queries:
        qid = query["id"]
        if qid not in relevance:
            continue

        relevant = set(relevance[qid])
        query_emb = pooled_encoder.encode(query["text"])

        start = time.time()
        results = search_fn(query_emb, index, top_k=top_k)
        total_time += time.time() - start

        found = set(r[0] for r in results)
        if relevant & found:
            hits += 1
        total += 1

    recall = hits / total if total > 0 else 0
    avg_time = (total_time / total * 1000) if total > 0 else 0
    return recall, avg_time


# =============================================================================
# DATASET
# =============================================================================

def load_dataset_small(name: str, max_docs: int = 500, max_queries: int = 100):
    print(f"  Loading {name}...")
    try:
        corpus = load_dataset(f"mteb/{name}", "corpus", split="corpus")
        queries_ds = load_dataset(f"mteb/{name}", "queries", split="queries")
        qrels_ds = load_dataset(f"mteb/{name}", "default", split="test")
    except Exception as e:
        print(f"  Error: {e}")
        return None, None, None

    qrels = defaultdict(set)
    for item in qrels_ds:
        qrels[item["query-id"]].add(item["corpus-id"])

    docs = []
    doc_ids = set()
    for i, item in enumerate(corpus):
        if i >= max_docs:
            break
        doc_id = item["_id"]
        text = item.get("title", "") + " " + item.get("text", "")
        docs.append({"id": doc_id, "text": text.strip()})
        doc_ids.add(doc_id)

    queries = []
    relevance = {}
    for item in queries_ds:
        qid = item["_id"]
        if qid in qrels:
            relevant = qrels[qid] & doc_ids
            if relevant:
                queries.append({"id": qid, "text": item["text"]})
                relevance[qid] = list(relevant)
        if len(queries) >= max_queries:
            break

    print(f"  Loaded {len(docs)} docs, {len(queries)} queries")
    return docs, queries, relevance


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("MATRYOSHKA MRL TEST")
    print("=" * 70)

    # Encoders
    token_encoder = TokenEncoder()
    pooled_encoder = PooledEncoder()

    # Dataset
    print("\n1. Loading dataset...")
    docs, queries, relevance = load_dataset_small("scifact", max_docs=500, max_queries=50)
    if docs is None:
        return

    # Build index
    print("\n2. Building index...")
    start = time.time()
    index = build_index(docs, token_encoder)
    print(f"   Done in {time.time() - start:.1f}s")

    # MRL configurations to test
    print("\n3. Testing MRL configurations...")
    print("=" * 70)

    configs = [
        # (name, dims_stage1, dims_stage2, dims_stage3)
        ("Full (1024/1024/1024)", 1024, 1024, 1024),
        ("MRL (512/768/1024)", 512, 768, 1024),
        ("MRL (256/512/1024)", 256, 512, 1024),
        ("MRL (128/256/1024)", 128, 256, 1024),
        ("MRL (64/128/512)", 64, 128, 512),
        ("MRL (32/64/256)", 32, 64, 256),
    ]

    # Reference: Token BF
    print("  Token BF (1024 dims)...")
    bf_recall, bf_time = evaluate(
        queries, relevance,
        lambda q, idx, **kw: search_token_bf(q, idx, 1024, kw.get("top_k", 10)),
        index, pooled_encoder
    )

    results = []
    for name, d1, d2, d3 in configs:
        print(f"  {name}...")
        recall, avg_time = evaluate(
            queries, relevance,
            lambda q, idx, d1=d1, d2=d2, d3=d3, **kw: search_mrl_hierarchical(
                q, idx, d1, d2, d3, top_k=kw.get("top_k", 10)
            ),
            index, pooled_encoder
        )
        results.append((name, d1, d2, d3, recall, avg_time))

    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n  Reference: Token BF @ 1024 dims = {bf_recall*100:.1f}% @ {bf_time:.1f}ms")
    print(f"\n  {'Config':<25} {'S1':>5} {'S2':>5} {'S3':>5} {'R@10':>7} {'Time':>8} {'vs BF':>8}")
    print(f"  {'-'*67}")

    for name, d1, d2, d3, recall, avg_time in results:
        vs_bf = (recall - bf_recall) * 100
        print(f"  {name:<25} {d1:>5} {d2:>5} {d3:>5} {recall*100:>6.1f}% {avg_time:>7.1f}ms {vs_bf:>+7.1f}%")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    best_fast = min(results, key=lambda x: x[5])  # Schnellster
    best_recall = max(results, key=lambda x: x[4])  # Bester Recall

    print(f"\n  Schnellste Config: {best_fast[0]} @ {best_fast[5]:.1f}ms ({best_fast[4]*100:.1f}% recall)")
    print(f"  Beste Recall:      {best_recall[0]} @ {best_recall[5]:.1f}ms ({best_recall[4]*100:.1f}% recall)")

    # Speedup durch MRL
    full_time = results[0][5]  # Full dims config
    mrl_256 = next(r for r in results if r[1] == 256)

    print(f"\n  Speedup Full → MRL(256/512/1024): {full_time/mrl_256[5]:.2f}x")
    print(f"  Recall-Verlust: {(results[0][4] - mrl_256[4])*100:+.1f}%")

    # Dimensions-Reduktion Faktor
    print("\n  Dimensions-Reduktion:")
    print(f"    Stage 1: 1024 → 256 = 4x weniger Compute")
    print(f"    Stage 2: 1024 → 512 = 2x weniger Compute")
    print(f"    Stage 3: 1024 → 1024 = volle Präzision")


if __name__ == "__main__":
    main()
