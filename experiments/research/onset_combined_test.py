#!/usr/bin/env python3
"""
Onset + Combined Test

Vergleicht:
1. Pooled (Baseline)
2. Combined (Dense + SPLADE → Token-Level ganzes Doc)
3. Combined + Onset (Dense + SPLADE → Onset-Segments → Token-Level gezielt)
4. Token BF (Referenz)
"""

import os
import numpy as np
import requests
import torch
import time
from dataclasses import dataclass, field
from collections import defaultdict
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer
from datasets import load_dataset


# =============================================================================
# ENCODERS -- two backends, switchable via S3_BACKEND env var.
#
#   S3_BACKEND=http (default)
#     Calls two text-embeddings-inference services:
#       - localhost:8200  pooled jina-embeddings-v3
#       - localhost:8202  token-level jina-embeddings-v3
#     The original setup. Brings up via docker-compose with the matching
#     TEI configs.
#
#   S3_BACKEND=local
#     Loads jinaai/jina-embeddings-v3 directly via transformers (one
#     shared model on CUDA), exposes pooled and per-token views without
#     any HTTP service. Useful on RunPod / Modal / any pod without
#     docker. Numbers should agree with the HTTP backend up to float
#     precision and tokenizer rounding.
# =============================================================================

BACKEND = os.environ.get("S3_BACKEND", "http").lower()
LOCAL_MODEL_NAME = os.environ.get("S3_LOCAL_MODEL", "jinaai/jina-embeddings-v3")


class TokenEncoder:
    """Per-token embeddings via TEI service at :8202 (token-level mode)."""
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
    """Sentence-level mean-pooled embedding via TEI service at :8200."""
    def __init__(self, url: str = "http://localhost:8200"):
        self.url = url

    def encode(self, text: str) -> np.ndarray:
        response = requests.post(
            f"{self.url}/v1/embeddings",
            json={"input": [text[:4000]]}
        )
        response.raise_for_status()
        return np.array(response.json()["data"][0]["embedding"])


# --- Local in-process jina-v3 backend (no HTTP, no docker required) ---

_JINA = {"model": None, "tokenizer": None, "device": None}


def _load_local_jina():
    if _JINA["model"] is not None:
        return _JINA["model"], _JINA["tokenizer"], _JINA["device"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[backend:local] loading {LOCAL_MODEL_NAME} on {device}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(LOCAL_MODEL_NAME, trust_remote_code=True)
    model = model.to(device).eval()
    _JINA.update(model=model, tokenizer=tokenizer, device=device)
    return model, tokenizer, device


class LocalJinaPooledEncoder:
    """Sentence-level mean-pooled jina-v3 embedding via direct transformers call."""
    def __init__(self):
        _load_local_jina()

    def encode(self, text: str) -> np.ndarray:
        model, tok, device = _load_local_jina()
        text = text[:4000]
        with torch.no_grad():
            inputs = tok(text, return_tensors="pt", truncation=True, max_length=8192).to(device)
            out = model(**inputs)
            # mean-pool over tokens with attention mask
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            summed = (out.last_hidden_state * mask).sum(dim=1)
            pooled = summed / mask.sum(dim=1).clamp(min=1e-6)
            emb = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return emb[0].cpu().float().numpy()


class LocalJinaTokenEncoder:
    """Per-token jina-v3 embeddings via direct transformers call."""
    def __init__(self):
        _load_local_jina()

    def encode(self, text: str) -> np.ndarray:
        model, tok, device = _load_local_jina()
        text = text[:4000]
        with torch.no_grad():
            inputs = tok(text, return_tensors="pt", truncation=True, max_length=8192).to(device)
            out = model(**inputs)
            mask = inputs["attention_mask"][0].bool()
            tokens = out.last_hidden_state[0][mask]
            tokens = torch.nn.functional.normalize(tokens, p=2, dim=-1)
        return tokens.cpu().float().numpy()


def make_pooled_encoder():
    return LocalJinaPooledEncoder() if BACKEND == "local" else PooledEncoder()


def make_token_encoder():
    return LocalJinaTokenEncoder() if BACKEND == "local" else TokenEncoder()


class SpladeEncoder:
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
# ONSET DETECTION (mit besten Parametern aus Sweep)
# =============================================================================

def spectral_flux(embeddings: np.ndarray) -> np.ndarray:
    changes = np.abs(np.diff(embeddings, axis=0))
    return changes.sum(axis=1)


def find_onsets(
    onset_signal: np.ndarray,
    threshold_pct: float = 95,  # Best from sweep
    min_dist: int = 3,          # Best from sweep
    smooth_sigma: float = 2.0   # Best from sweep
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

    if not segments:
        return embeddings.mean(axis=0, keepdims=True), [(0, len(embeddings))]

    return np.array(segments), ranges


# =============================================================================
# INDEX
# =============================================================================

@dataclass
class DocumentIndex:
    doc_id: str
    text: str
    token_embeddings: np.ndarray
    doc_embedding: np.ndarray
    segment_embeddings: np.ndarray
    segment_ranges: list
    splade_terms: np.ndarray
    splade_weights: np.ndarray


@dataclass
class FullIndex:
    docs: dict = field(default_factory=dict)
    splade_inverted: dict = field(default_factory=lambda: defaultdict(list))


def build_index(
    docs: list[dict],
    token_encoder: TokenEncoder,
    splade_encoder: SpladeEncoder
) -> FullIndex:
    index = FullIndex()

    for i, doc in enumerate(docs):
        doc_id = doc["id"]
        text = doc["text"]

        # Token Embeddings
        token_embs = token_encoder.encode(text)

        # Mean-Pooled
        doc_emb = token_embs.mean(axis=0)

        # Onset Segments
        seg_embs, seg_ranges = build_onset_segments(token_embs)

        # SPLADE
        splade_terms, splade_weights = splade_encoder.encode(text)

        index.docs[doc_id] = DocumentIndex(
            doc_id=doc_id,
            text=text,
            token_embeddings=token_embs,
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
            print(f"    Indexed {i+1}/{len(docs)}")

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


def search_splade(query_terms: np.ndarray, query_weights: np.ndarray, index: FullIndex, top_k: int = 100) -> list:
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
    """Combined: Dense + SPLADE → Token-Level (ganzes Doc)."""

    # Stage 1: Get candidates
    dense_results = search_pooled(query_emb, index, dense_k)
    splade_results = search_splade(query_terms, query_weights, index, splade_k)

    candidates = set(r[0] for r in dense_results) | set(r[0] for r in splade_results)

    # Stage 2: Token-level on ENTIRE document
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
    top_segments: int = 3,  # Pro Doc: Top N Segments für Token-Level
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


def search_token_bf(query_emb: np.ndarray, index: FullIndex, top_k: int = 10) -> list:
    """Token-Level Brute Force."""
    results = []
    for doc_id, doc in index.docs.items():
        best_score = max(cosine_sim(query_emb, t) for t in doc.token_embeddings)
        results.append((doc_id, best_score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(queries, relevance, search_fn, index, pooled_encoder, splade_encoder, needs_splade=False, top_k=10):
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
    import argparse
    import json as _json
    import datetime as _dt
    from pathlib import Path as _Path
    parser = argparse.ArgumentParser(description="Onset + Combined retrieval benchmark.")
    parser.add_argument("--dataset", default="scifact", help="BEIR dataset slug (default: scifact)")
    parser.add_argument("--max-docs", type=int, default=1000, help="Truncate corpus (default: 1000)")
    parser.add_argument("--max-queries", type=int, default=100, help="Cap queries before filtering (default: 100)")
    parser.add_argument("--output", default=None, help="Write summary JSON to this path (default: results/onset_combined_<timestamp>.json)")
    args = parser.parse_args()

    print("=" * 70)
    print("ONSET + COMBINED TEST")
    print("=" * 70)

    # Encoders
    print("\n1. Loading encoders...")
    print(f"[backend] S3_BACKEND={BACKEND}")
    token_encoder = make_token_encoder()
    pooled_encoder = make_pooled_encoder()
    splade_encoder = SpladeEncoder(top_k=64)
    print("   OK")

    # Dataset
    print("\n2. Loading dataset...")
    docs, queries, relevance = load_dataset_small(args.dataset, max_docs=args.max_docs, max_queries=args.max_queries)
    if docs is None:
        return

    # Build index
    print("\n3. Building index...")
    start = time.time()
    index = build_index(docs, token_encoder, splade_encoder)
    print(f"   Done in {time.time() - start:.1f}s")

    # Stats
    avg_tokens = np.mean([len(d.token_embeddings) for d in index.docs.values()])
    avg_segs = np.mean([len(d.segment_ranges) for d in index.docs.values()])
    print(f"   Avg tokens/doc: {avg_tokens:.1f}")
    print(f"   Avg segments/doc: {avg_segs:.1f}")

    # Evaluate
    print("\n4. Evaluating methods...")
    print("=" * 70)

    methods = [
        ("Pooled (Baseline)",
         lambda q, i, **kw: search_pooled(q, i, kw.get("top_k", 10)),
         False),
        ("Combined",
         lambda q, qt, qw, i, **kw: search_combined(q, qt, qw, i, top_k=kw.get("top_k", 10)),
         True),
        ("Combined + Onset",
         lambda q, qt, qw, i, **kw: search_combined_onset(q, qt, qw, i, top_k=kw.get("top_k", 10)),
         True),
        ("Token BF",
         lambda q, i, **kw: search_token_bf(q, i, kw.get("top_k", 10)),
         False),
    ]

    results = {}
    for name, search_fn, needs_splade in methods:
        print(f"  {name}...")
        recall, avg_time = evaluate(
            queries, relevance, search_fn, index,
            pooled_encoder, splade_encoder,
            needs_splade=needs_splade, top_k=10
        )
        results[name] = (recall, avg_time)

    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n  {'Method':<20} {'R@10':>8} {'Time':>10} {'vs Pooled':>10} {'vs Token BF':>12}")
    print(f"  {'-'*62}")

    pooled_recall = results["Pooled (Baseline)"][0]
    token_bf_recall = results["Token BF"][0]

    for name, (recall, avg_time) in results.items():
        vs_pooled = (recall - pooled_recall) * 100
        vs_token = (recall - token_bf_recall) * 100
        print(f"  {name:<20} {recall*100:>7.1f}% {avg_time:>8.1f}ms {vs_pooled:>+9.1f}% {vs_token:>+11.1f}%")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    combined_recall = results["Combined"][0]
    combined_onset_recall = results["Combined + Onset"][0]
    combined_time = results["Combined"][1]
    combined_onset_time = results["Combined + Onset"][1]

    print(f"\n  Combined vs Combined+Onset:")
    print(f"    Recall: {combined_recall*100:.1f}% vs {combined_onset_recall*100:.1f}% "
          f"({(combined_onset_recall - combined_recall)*100:+.1f}%)")
    print(f"    Time:   {combined_time:.1f}ms vs {combined_onset_time:.1f}ms "
          f"({(combined_onset_time / combined_time - 1)*100:+.1f}%)")

    if combined_onset_recall >= combined_recall and combined_onset_time < combined_time:
        print("\n  --> Combined+Onset ist BESSER UND SCHNELLER!")
    elif combined_onset_recall >= combined_recall:
        print("\n  --> Combined+Onset hat gleichen/besseren Recall, aber langsamer")
    elif combined_onset_time < combined_time:
        print("\n  --> Combined+Onset ist schneller, aber schlechterer Recall")
    else:
        print("\n  --> Combined ist besser")

    # Persist summary JSON (the artifact the README was missing).
    out_path = args.output
    if out_path is None:
        ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        out_dir = _Path(__file__).resolve().parents[2] / "results" / "onset_combined"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"onset_combined_{args.dataset}_{ts}.json"
    summary_payload = {
        "config": {
            "dataset": args.dataset,
            "max_docs": args.max_docs,
            "max_queries_before_filter": args.max_queries,
            "n_queries_evaluated": len(queries),
            "n_docs_indexed": len(index.docs),
            "avg_segments_per_doc": float(avg_segs),
            "backend": BACKEND,
            "local_model": LOCAL_MODEL_NAME if BACKEND == "local" else None,
            "splade_model": "naver/splade-cocondenser-ensembledistil",
            "top_k": 10,
        },
        "results": {
            name: {"recall_at_10": float(recall), "avg_time_ms": float(avg_time)}
            for name, (recall, avg_time) in results.items()
        },
        "deltas": {
            "combined_onset_vs_combined_recall_pp": float((combined_onset_recall - combined_recall) * 100),
            "combined_onset_vs_combined_time_ratio": float(combined_onset_time / combined_time),
            "combined_onset_vs_token_bf_recall_pp": float((combined_onset_recall - token_bf_recall) * 100),
            "combined_onset_vs_token_bf_time_ratio": float(combined_onset_time / results["Token BF"][1]),
        },
        "caveat": (
            "max_docs truncation interacts with query filtering: queries are "
            "kept only when at least one of their relevant docs survives the "
            "first max_docs documents. Easy queries are over-represented vs a "
            "uniform-random query sample."
        ),
    }
    _Path(out_path).write_text(_json.dumps(summary_payload, indent=2))
    print(f"\n[artifact] wrote {out_path}")


if __name__ == "__main__":
    main()
