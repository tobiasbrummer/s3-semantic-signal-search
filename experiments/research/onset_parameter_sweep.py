#!/usr/bin/env python3
"""
Onset Parameter Sweep

Testet verschiedene Onset-Parameter auf kleinerem Datensatz.
"""

import numpy as np
import requests
import time
from dataclasses import dataclass, field
from collections import defaultdict
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from datasets import load_dataset
from itertools import product


# =============================================================================
# ENCODERS
# =============================================================================

class TokenEncoder:
    def __init__(self, url: str = "http://localhost:8202"):
        self.url = url

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
# ONSET DETECTION (PARAMETERIZED)
# =============================================================================

def spectral_flux(embeddings: np.ndarray) -> np.ndarray:
    changes = np.abs(np.diff(embeddings, axis=0))
    return changes.sum(axis=1)


def find_onsets(
    onset_signal: np.ndarray,
    threshold_pct: float = 85,
    min_dist: int = 10,
    smooth_sigma: float = 1.0
) -> np.ndarray:
    if len(onset_signal) < 3:
        return np.array([])

    if smooth_sigma > 0:
        smoothed = gaussian_filter1d(onset_signal, sigma=smooth_sigma)
    else:
        smoothed = onset_signal

    threshold = np.percentile(smoothed, threshold_pct)
    peaks, _ = find_peaks(smoothed, height=threshold, distance=min_dist)
    return peaks


def build_onset_segments(
    embeddings: np.ndarray,
    threshold_pct: float = 85,
    min_dist: int = 10,
    smooth_sigma: float = 1.0
) -> tuple[np.ndarray, list]:
    onset_signal = spectral_flux(embeddings)
    onsets = find_onsets(onset_signal, threshold_pct, min_dist, smooth_sigma)

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
    token_embeddings: np.ndarray
    doc_embedding: np.ndarray


@dataclass
class OnsetIndex:
    """Index mit spezifischen Onset-Parametern."""
    docs: dict = field(default_factory=dict)
    segment_embeddings: dict = field(default_factory=dict)
    segment_ranges: dict = field(default_factory=dict)


def build_base_index(docs: list[dict], token_encoder: TokenEncoder) -> dict:
    """Baue Basis-Index mit Token-Embeddings (einmalig)."""
    base = {}
    for i, doc in enumerate(docs):
        doc_id = doc["id"]
        token_embs = token_encoder.encode(doc["text"])
        base[doc_id] = DocumentIndex(
            doc_id=doc_id,
            token_embeddings=token_embs,
            doc_embedding=token_embs.mean(axis=0)
        )
        if (i + 1) % 50 == 0:
            print(f"  Indexed {i+1}/{len(docs)}")
    return base


def build_onset_index(
    base_index: dict,
    threshold_pct: float,
    min_dist: int,
    smooth_sigma: float
) -> OnsetIndex:
    """Baue Onset-Index mit spezifischen Parametern (schnell, da Token-Embs gecached)."""
    onset_idx = OnsetIndex()

    for doc_id, doc in base_index.items():
        onset_idx.docs[doc_id] = doc
        seg_embs, seg_ranges = build_onset_segments(
            doc.token_embeddings,
            threshold_pct, min_dist, smooth_sigma
        )
        onset_idx.segment_embeddings[doc_id] = seg_embs
        onset_idx.segment_ranges[doc_id] = seg_ranges

    return onset_idx


# =============================================================================
# SEARCH
# =============================================================================

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)


def search_pooled(query_emb: np.ndarray, base_index: dict, top_k: int = 10) -> list:
    results = []
    for doc_id, doc in base_index.items():
        score = cosine_sim(query_emb, doc.doc_embedding)
        results.append((doc_id, score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def search_onset(
    query_emb: np.ndarray,
    onset_index: OnsetIndex,
    doc_k: int = 50,
    seg_k: int = 20,
    top_k: int = 10
) -> list:
    # Level 1: Document
    doc_scores = [
        (did, cosine_sim(query_emb, doc.doc_embedding))
        for did, doc in onset_index.docs.items()
    ]
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    top_docs = [d[0] for d in doc_scores[:doc_k]]

    # Level 2: Segments
    seg_scores = []
    for doc_id in top_docs:
        seg_embs = onset_index.segment_embeddings[doc_id]
        for seg_idx, seg_emb in enumerate(seg_embs):
            score = cosine_sim(query_emb, seg_emb)
            seg_scores.append((doc_id, seg_idx, score))

    seg_scores.sort(key=lambda x: x[2], reverse=True)
    top_segs = seg_scores[:seg_k]

    # Level 3: Tokens (in best segment per doc)
    results = []
    seen = set()
    for doc_id, seg_idx, _ in top_segs:
        if doc_id in seen:
            continue
        doc = onset_index.docs[doc_id]
        start, end = onset_index.segment_ranges[doc_id][seg_idx]
        best_score = max(
            cosine_sim(query_emb, doc.token_embeddings[t])
            for t in range(start, end)
        )
        results.append((doc_id, best_score))
        seen.add(doc_id)

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def search_token_bf(query_emb: np.ndarray, base_index: dict, top_k: int = 10) -> list:
    """Token Brute Force als Referenz."""
    results = []
    for doc_id, doc in base_index.items():
        best_score = max(cosine_sim(query_emb, t) for t in doc.token_embeddings)
        results.append((doc_id, best_score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(queries, relevance, search_fn, index, pooled_encoder, top_k=10):
    hits = 0
    total = 0

    for query in queries:
        qid = query["id"]
        if qid not in relevance:
            continue

        relevant = set(relevance[qid])
        query_emb = pooled_encoder.encode(query["text"])
        results = search_fn(query_emb, index, top_k=top_k)

        found = set(r[0] for r in results)
        if relevant & found:
            hits += 1
        total += 1

    return hits / total if total > 0 else 0


# =============================================================================
# MAIN
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


def main():
    print("=" * 70)
    print("ONSET PARAMETER SWEEP")
    print("=" * 70)

    # Encoders
    token_encoder = TokenEncoder()
    pooled_encoder = PooledEncoder()

    # Load small dataset
    docs, queries, relevance = load_dataset_small("scifact", max_docs=500, max_queries=100)
    if docs is None:
        return

    # Build base index (einmalig)
    print("\n  Building base index...")
    start = time.time()
    base_index = build_base_index(docs, token_encoder)
    print(f"  Done in {time.time() - start:.1f}s")

    # Baselines
    print("\n  Computing baselines...")
    pooled_recall = evaluate(
        queries, relevance,
        lambda q, idx, **kw: search_pooled(q, idx, kw.get("top_k", 10)),
        base_index, pooled_encoder
    )

    token_bf_recall = evaluate(
        queries, relevance,
        lambda q, idx, **kw: search_token_bf(q, idx, kw.get("top_k", 10)),
        base_index, pooled_encoder
    )

    print(f"\n  Baselines:")
    print(f"    Pooled:   {pooled_recall*100:.1f}%")
    print(f"    Token BF: {token_bf_recall*100:.1f}%")

    # Parameter ranges
    threshold_pcts = [70, 75, 80, 85, 90, 95]
    min_dists = [3, 5, 10, 15, 20]
    smooth_sigmas = [0.5, 1.0, 2.0, 3.0]

    print(f"\n  Testing {len(threshold_pcts) * len(min_dists) * len(smooth_sigmas)} combinations...")
    print("=" * 70)

    results = []

    for threshold_pct, min_dist, smooth_sigma in product(threshold_pcts, min_dists, smooth_sigmas):
        # Build onset index with these params
        onset_index = build_onset_index(base_index, threshold_pct, min_dist, smooth_sigma)

        # Count average segments
        avg_segs = np.mean([
            len(onset_index.segment_ranges[did])
            for did in onset_index.docs
        ])

        # Evaluate
        recall = evaluate(
            queries, relevance,
            lambda q, idx, **kw: search_onset(q, idx, top_k=kw.get("top_k", 10)),
            onset_index, pooled_encoder
        )

        results.append({
            "threshold_pct": threshold_pct,
            "min_dist": min_dist,
            "smooth_sigma": smooth_sigma,
            "recall": recall,
            "avg_segments": avg_segs,
            "vs_pooled": recall - pooled_recall,
            "vs_token_bf": recall - token_bf_recall
        })

    # Sort by recall
    results.sort(key=lambda x: x["recall"], reverse=True)

    # Print top 10
    print(f"\n  TOP 10 CONFIGURATIONS")
    print(f"  {'Thresh':>6} {'Dist':>5} {'Sigma':>6} {'R@10':>7} {'Segs':>5} {'vs Pool':>8} {'vs TBF':>8}")
    print(f"  {'-'*52}")

    for r in results[:10]:
        print(f"  {r['threshold_pct']:>6} {r['min_dist']:>5} {r['smooth_sigma']:>6.1f} "
              f"{r['recall']*100:>6.1f}% {r['avg_segments']:>5.1f} "
              f"{r['vs_pooled']*100:>+7.1f}% {r['vs_token_bf']*100:>+7.1f}%")

    # Print worst 5
    print(f"\n  WORST 5 CONFIGURATIONS")
    print(f"  {'Thresh':>6} {'Dist':>5} {'Sigma':>6} {'R@10':>7} {'Segs':>5} {'vs Pool':>8} {'vs TBF':>8}")
    print(f"  {'-'*52}")

    for r in results[-5:]:
        print(f"  {r['threshold_pct']:>6} {r['min_dist']:>5} {r['smooth_sigma']:>6.1f} "
              f"{r['recall']*100:>6.1f}% {r['avg_segments']:>5.1f} "
              f"{r['vs_pooled']*100:>+7.1f}% {r['vs_token_bf']*100:>+7.1f}%")

    # Analysis
    print("\n" + "=" * 70)
    print("  ANALYSIS")
    print("=" * 70)

    best = results[0]
    print(f"\n  Best config: threshold={best['threshold_pct']}, "
          f"min_dist={best['min_dist']}, sigma={best['smooth_sigma']}")
    print(f"  Recall: {best['recall']*100:.1f}% "
          f"(Pooled: {pooled_recall*100:.1f}%, Token BF: {token_bf_recall*100:.1f}%)")

    # Parameter correlations
    print("\n  Parameter effects (average recall by value):")

    print("\n  Threshold:")
    for t in threshold_pcts:
        avg = np.mean([r["recall"] for r in results if r["threshold_pct"] == t])
        print(f"    {t}: {avg*100:.1f}%")

    print("\n  Min Distance:")
    for d in min_dists:
        avg = np.mean([r["recall"] for r in results if r["min_dist"] == d])
        print(f"    {d}: {avg*100:.1f}%")

    print("\n  Smooth Sigma:")
    for s in smooth_sigmas:
        avg = np.mean([r["recall"] for r in results if r["smooth_sigma"] == s])
        print(f"    {s}: {avg*100:.1f}%")


if __name__ == "__main__":
    main()
