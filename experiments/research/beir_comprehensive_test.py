#!/usr/bin/env python3
"""
BEIR Comprehensive Test

Testet alle validierten S3-Komponenten auf verschiedenen BEIR Datasets.

Komponenten:
1. Pooled Embeddings (Baseline)
2. Token-Level Brute Force
3. Combined Pipeline (Dense + SPLADE + Multi-Res)
4. Onset-Based Multi-Resolution

Datasets:
- SciFact (Wissenschaft)
- FiQA (Finanz)
- NFCorpus (Medical)
- Quora (Fragen)
- TREC-COVID (optional)
"""

import numpy as np
import requests
import torch
import time
from dataclasses import dataclass, field
from collections import defaultdict
from transformers import AutoModelForMaskedLM, AutoTokenizer
from datasets import load_dataset
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


# =============================================================================
# ENCODERS
# =============================================================================

class TokenEncoder:
    """Token-Level Embeddings (llama.cpp --pooling none)."""
    def __init__(self, url: str = "http://localhost:8202"):
        self.url = url
        self.dim = 1024

    def encode(self, text: str) -> np.ndarray:
        text = text[:4000]
        response = requests.post(f"{self.url}/embeddings", json={"input": text})
        response.raise_for_status()
        data = response.json()
        if isinstance(data[0]["embedding"][0], list):
            return np.array(data[0]["embedding"])
        return np.array([data[0]["embedding"]])


class PooledEncoder:
    """Pooled Embeddings (Standard)."""
    def __init__(self, url: str = "http://localhost:8200"):
        self.url = url

    def encode(self, text: str) -> np.ndarray:
        response = requests.post(
            f"{self.url}/v1/embeddings",
            json={"input": [text[:4000]]}
        )
        response.raise_for_status()
        return np.array(response.json()["data"][0]["embedding"])

    def encode_batch(self, texts: list[str], batch_size: int = 8) -> np.ndarray:
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = [t[:4000] for t in texts[i:i+batch_size]]
            response = requests.post(
                f"{self.url}/v1/embeddings",
                json={"input": batch}
            )
            response.raise_for_status()
            embs = [d["embedding"] for d in response.json()["data"]]
            all_embs.extend(embs)
        return np.array(all_embs)


class SpladeEncoder:
    """SPLADE Sparse Embeddings."""
    def __init__(self, model_name: str = "naver/splade-cocondenser-ensembledistil", top_k: int = 64):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.top_k = top_k

    def encode(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        inputs = self.tokenizer(
            text[:4000], return_tensors="pt", max_length=512,
            truncation=True, padding=True
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

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
# INDEX STRUCTURES
# =============================================================================

@dataclass
class DocumentIndex:
    """Index für ein Dokument mit allen Repräsentationen."""
    doc_id: str
    text: str

    # Token-Level
    token_embeddings: np.ndarray = None
    n_tokens: int = 0

    # Aggregations
    doc_embedding: np.ndarray = None  # Mean-Pooled
    segment_embeddings: np.ndarray = None  # Onset-based
    segment_ranges: list = None

    # SPLADE
    splade_terms: np.ndarray = None
    splade_weights: np.ndarray = None


@dataclass
class FullIndex:
    """Vollständiger Index für ein Dataset."""
    docs: dict = field(default_factory=dict)  # doc_id → DocumentIndex
    splade_inverted: dict = field(default_factory=lambda: defaultdict(list))


# =============================================================================
# ONSET DETECTION
# =============================================================================

def spectral_flux(embeddings: np.ndarray) -> np.ndarray:
    changes = np.abs(np.diff(embeddings, axis=0))
    return changes.sum(axis=1)


def find_onsets(
    onset_signal: np.ndarray,
    threshold_pct: float = 95,   # Best from parameter sweep
    min_dist: int = 3,           # Best from parameter sweep
    smooth_sigma: float = 2.0    # Best from parameter sweep
) -> np.ndarray:
    if len(onset_signal) < 3:
        return np.array([])
    smoothed = gaussian_filter1d(onset_signal, sigma=smooth_sigma)
    threshold = np.percentile(smoothed, threshold_pct)
    peaks, _ = find_peaks(smoothed, height=threshold, distance=min_dist)
    return peaks


def build_onset_segments(embeddings: np.ndarray) -> tuple[np.ndarray, list]:
    onset_signal = spectral_flux(embeddings)
    onsets = find_onsets(onset_signal)

    boundaries = [0] + sorted(onsets.tolist()) + [len(embeddings)]
    segments = []
    ranges = []

    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i+1]
        if end > start:
            segments.append(embeddings[start:end].mean(axis=0))
            ranges.append((start, end))

    return np.array(segments) if segments else embeddings.mean(axis=0, keepdims=True), ranges or [(0, len(embeddings))]


# =============================================================================
# INDEX BUILDING
# =============================================================================

def build_full_index(
    docs: list[dict],
    token_encoder: TokenEncoder,
    splade_encoder: SpladeEncoder,
    use_onset: bool = True
) -> FullIndex:
    """Baue vollständigen Index mit allen Komponenten."""

    index = FullIndex()

    for i, doc in enumerate(docs):
        doc_id = doc["id"]
        text = doc["text"]

        # Token Embeddings
        token_embs = token_encoder.encode(text)

        # Mean-Pooled
        doc_emb = token_embs.mean(axis=0)

        # Onset Segments
        if use_onset:
            seg_embs, seg_ranges = build_onset_segments(token_embs)
        else:
            seg_embs, seg_ranges = None, None

        # SPLADE
        splade_terms, splade_weights = splade_encoder.encode(text)

        # Store
        index.docs[doc_id] = DocumentIndex(
            doc_id=doc_id,
            text=text,
            token_embeddings=token_embs,
            n_tokens=len(token_embs),
            doc_embedding=doc_emb,
            segment_embeddings=seg_embs,
            segment_ranges=seg_ranges,
            splade_terms=splade_terms,
            splade_weights=splade_weights
        )

        # SPLADE Inverted Index
        for term_id, weight in zip(splade_terms, splade_weights):
            index.splade_inverted[int(term_id)].append((doc_id, float(weight)))

        if (i + 1) % 100 == 0:
            print(f"  Indexed {i+1}/{len(docs)}")

    return index


# =============================================================================
# SEARCH METHODS
# =============================================================================

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)


def search_pooled(query_emb: np.ndarray, index: FullIndex, top_k: int = 10) -> list:
    """Baseline: Pooled Embeddings."""
    results = []
    for doc_id, doc in index.docs.items():
        score = cosine_sim(query_emb, doc.doc_embedding)
        results.append((doc_id, score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def search_token_bf(query_emb: np.ndarray, index: FullIndex, top_k: int = 10) -> list:
    """Token-Level Brute Force."""
    results = []
    for doc_id, doc in index.docs.items():
        best_score = max(cosine_sim(query_emb, t) for t in doc.token_embeddings)
        results.append((doc_id, best_score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def search_splade(query_terms: np.ndarray, query_weights: np.ndarray, index: FullIndex, top_k: int = 10) -> list:
    """SPLADE Inverted Index."""
    scores = defaultdict(float)
    for term_id, q_weight in zip(query_terms, query_weights):
        if term_id in index.splade_inverted:
            for doc_id, d_weight in index.splade_inverted[term_id]:
                scores[doc_id] += q_weight * d_weight
    results = list(scores.items())
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def search_combined(
    query_emb: np.ndarray,
    query_terms: np.ndarray,
    query_weights: np.ndarray,
    index: FullIndex,
    dense_k: int = 100,
    splade_k: int = 100,
    top_k: int = 10
) -> list:
    """Combined Pipeline: Dense + SPLADE → Token Refinement."""

    # Stage 1: Get candidates
    dense_results = search_pooled(query_emb, index, dense_k)
    splade_results = search_splade(query_terms, query_weights, index, splade_k)

    candidates = set(r[0] for r in dense_results) | set(r[0] for r in splade_results)

    # Stage 2: Token-level refinement
    results = []
    for doc_id in candidates:
        doc = index.docs[doc_id]
        best_score = max(cosine_sim(query_emb, t) for t in doc.token_embeddings)
        results.append((doc_id, best_score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def search_combined_onset(
    query_emb: np.ndarray,
    query_terms: np.ndarray,
    query_weights: np.ndarray,
    index: FullIndex,
    dense_k: int = 100,
    splade_k: int = 100,
    top_segments: int = 3,
    top_k: int = 10
) -> list:
    """Combined + Onset: Dense + SPLADE → Segment-Ranking → Token-Level (gezielt)."""

    # Stage 1: Get candidates (identisch zu Combined)
    dense_results = search_pooled(query_emb, index, dense_k)
    splade_results = search_splade(query_terms, query_weights, index, splade_k)

    candidates = set(r[0] for r in dense_results) | set(r[0] for r in splade_results)

    # Stage 2: Für jeden Candidate → Segment-Ranking → Token-Level nur in Top-Segments
    results = []
    for doc_id in candidates:
        doc = index.docs[doc_id]

        # Rank segments by query similarity
        seg_scores = [
            (seg_idx, cosine_sim(query_emb, seg_emb))
            for seg_idx, seg_emb in enumerate(doc.segment_embeddings)
        ]
        seg_scores.sort(key=lambda x: x[1], reverse=True)

        # Token-Level nur in Top-N Segments
        best_score = 0
        for seg_idx, _ in seg_scores[:top_segments]:
            start, end = doc.segment_ranges[seg_idx]
            for t in range(start, end):
                score = cosine_sim(query_emb, doc.token_embeddings[t])
                if score > best_score:
                    best_score = score

        results.append((doc_id, best_score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def search_onset(query_emb: np.ndarray, index: FullIndex, doc_k: int = 50, seg_k: int = 20, top_k: int = 10) -> list:
    """Onset-Based Multi-Resolution."""

    # Level 1: Document
    doc_scores = [(did, cosine_sim(query_emb, d.doc_embedding)) for did, d in index.docs.items()]
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    top_docs = [d[0] for d in doc_scores[:doc_k]]

    # Level 2: Segments
    seg_scores = []
    for doc_id in top_docs:
        doc = index.docs[doc_id]
        if doc.segment_embeddings is not None:
            for seg_idx, seg_emb in enumerate(doc.segment_embeddings):
                score = cosine_sim(query_emb, seg_emb)
                seg_scores.append((doc_id, seg_idx, score))

    seg_scores.sort(key=lambda x: x[2], reverse=True)
    top_segs = seg_scores[:seg_k]

    # Level 3: Tokens
    results = []
    seen = set()
    for doc_id, seg_idx, _ in top_segs:
        if doc_id in seen:
            continue
        doc = index.docs[doc_id]
        start, end = doc.segment_ranges[seg_idx]
        best_score = max(cosine_sim(query_emb, doc.token_embeddings[t]) for t in range(start, end))
        results.append((doc_id, best_score))
        seen.add(doc_id)

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


# =============================================================================
# DATASET LOADING
# =============================================================================

def load_beir_dataset(name: str, max_docs: int = 5000, max_queries: int = 500):
    """Lade BEIR Dataset."""

    print(f"  Loading {name}...")

    try:
        corpus = load_dataset(f"mteb/{name}", "corpus", split="corpus")
        queries_ds = load_dataset(f"mteb/{name}", "queries", split="queries")
        qrels_ds = load_dataset(f"mteb/{name}", "default", split="test")
    except Exception as e:
        print(f"  Error loading {name}: {e}")
        return None, None, None

    # Build qrels
    qrels = defaultdict(set)
    for item in qrels_ds:
        qrels[item["query-id"]].add(item["corpus-id"])

    # Load docs
    docs = []
    doc_ids = set()
    for i, item in enumerate(corpus):
        if i >= max_docs:
            break
        doc_id = item["_id"]
        text = item.get("title", "") + " " + item.get("text", "")
        docs.append({"id": doc_id, "text": text.strip()})
        doc_ids.add(doc_id)

    # Load queries with relevance
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
# EVALUATION
# =============================================================================

def evaluate(
    queries: list,
    relevance: dict,
    search_fn,
    index: FullIndex,
    pooled_encoder: PooledEncoder,
    splade_encoder: SpladeEncoder,
    top_k: int = 10,
    needs_splade: bool = False
) -> tuple[float, float]:
    """Evaluate search method."""

    hits = 0
    total = 0
    total_time = 0

    for query in queries:
        qid = query["id"]
        if qid not in relevance:
            continue

        relevant = set(relevance[qid])
        query_emb = pooled_encoder.encode(query["text"])

        if needs_splade:
            query_terms, query_weights = splade_encoder.encode(query["text"])
            start = time.time()
            results = search_fn(query_emb, query_terms, query_weights, index, top_k=top_k)
        else:
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
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("BEIR COMPREHENSIVE TEST")
    print("=" * 70)

    # Encoders
    print("\n1. Loading Encoders...")
    token_encoder = TokenEncoder()
    pooled_encoder = PooledEncoder()
    splade_encoder = SpladeEncoder(top_k=64)
    print("   OK")

    # Datasets to test
    datasets = ["scifact", "fiqa", "nfcorpus", "quora"]

    results_all = {}

    for dataset_name in datasets:
        print(f"\n{'='*70}")
        print(f"DATASET: {dataset_name}")
        print("=" * 70)

        # Load data
        docs, queries, relevance = load_beir_dataset(dataset_name, max_docs=2000, max_queries=200)

        if docs is None:
            continue

        # Build index
        print("\n  Building index...")
        start = time.time()
        index = build_full_index(docs, token_encoder, splade_encoder, use_onset=True)
        index_time = time.time() - start
        print(f"  Index built in {index_time:.1f}s")

        # Evaluate methods
        print("\n  Evaluating...")

        methods = [
            ("Pooled (Baseline)", lambda q, i, **kw: search_pooled(q, i, kw.get("top_k", 10)), False),
            ("Combined", lambda q, qt, qw, i, **kw: search_combined(q, qt, qw, i, top_k=kw.get("top_k", 10)), True),
            ("Combined+Onset", lambda q, qt, qw, i, **kw: search_combined_onset(q, qt, qw, i, top_k=kw.get("top_k", 10)), True),
            ("Token BF", lambda q, i, **kw: search_token_bf(q, i, kw.get("top_k", 10)), False),
        ]

        dataset_results = {}

        for name, search_fn, needs_splade in methods:
            print(f"    {name}...")
            recall, avg_time = evaluate(
                queries, relevance, search_fn, index,
                pooled_encoder, splade_encoder,
                top_k=10, needs_splade=needs_splade
            )
            dataset_results[name] = (recall, avg_time)

        results_all[dataset_name] = dataset_results

        # Print results for this dataset
        print(f"\n  {'Method':<20} {'R@10':>8} {'Time':>10}")
        print(f"  {'-'*40}")
        for name, (recall, avg_time) in dataset_results.items():
            print(f"  {name:<20} {recall*100:>7.1f}% {avg_time:>8.1f}ms")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Dataset':<12}", end="")
    for method in ["Pooled", "Combined", "Comb+Onset", "Token BF"]:
        print(f"{method:>12}", end="")
    print()
    print("-" * 60)

    for dataset_name, results in results_all.items():
        print(f"{dataset_name:<12}", end="")
        for method_name in ["Pooled (Baseline)", "Combined", "Combined+Onset", "Token BF"]:
            if method_name in results:
                recall, _ = results[method_name]
                print(f"{recall*100:>11.1f}%", end="")
            else:
                print(f"{'N/A':>12}", end="")
        print()

    # Timing summary
    print("\n" + "=" * 70)
    print("TIMING (ms per query)")
    print("=" * 70)

    print(f"\n{'Dataset':<12}", end="")
    for method in ["Pooled", "Combined", "Comb+Onset", "Token BF"]:
        print(f"{method:>12}", end="")
    print()
    print("-" * 60)

    for dataset_name, results in results_all.items():
        print(f"{dataset_name:<12}", end="")
        for method_name in ["Pooled (Baseline)", "Combined", "Combined+Onset", "Token BF"]:
            if method_name in results:
                _, avg_time = results[method_name]
                print(f"{avg_time:>11.1f}", end="")
            else:
                print(f"{'N/A':>12}", end="")
        print()


if __name__ == "__main__":
    main()
