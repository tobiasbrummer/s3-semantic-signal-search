#!/usr/bin/env python3
"""
FFT Cross-Correlation Search Prototype

Konsequenter Audio-Ansatz:
- Query bleibt Token-Sequenz (nicht mean-pooled)
- FFT Cross-Correlation findet Pattern im Doc-Signal
- GPU-ready: Alle Dimensionen parallel

Vergleich:
- Bisherig: Query pooled → Cosine mit jedem Token → O(n) per dim
- FFT: Query als Pattern → Cross-Correlation → O(n log n) für ALLE Positionen
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
# FFT CROSS-CORRELATION
# =============================================================================

def fft_cross_correlation(signal: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    """
    FFT-basierte Cross-Correlation.

    Args:
        signal: (N, dims) - Document token embeddings
        pattern: (M, dims) - Query token embeddings

    Returns:
        correlation: (N,) - Correlation score at each position
    """
    n_signal, n_dims = signal.shape
    n_pattern = len(pattern)

    # Pad to same length for FFT
    n_fft = n_signal + n_pattern - 1
    n_fft_padded = 2 ** int(np.ceil(np.log2(n_fft)))  # Power of 2 for efficiency

    # Cross-correlation per dimension, then sum
    correlation = np.zeros(n_fft_padded)

    for d in range(n_dims):
        # FFT of signal and pattern for this dimension
        sig_fft = np.fft.fft(signal[:, d], n_fft_padded)
        pat_fft = np.fft.fft(pattern[:, d], n_fft_padded)

        # Cross-correlation = ifft(fft(signal) * conj(fft(pattern)))
        corr_d = np.fft.ifft(sig_fft * np.conj(pat_fft))
        correlation += np.real(corr_d)

    # Return only valid positions
    return correlation[:n_signal]


def fft_cross_correlation_normalized(signal: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    """
    Normalized FFT Cross-Correlation (ähnlich zu Cosine Similarity).

    Normalisiert durch die Normen von Signal-Windows und Pattern.
    """
    n_signal, n_dims = signal.shape
    n_pattern = len(pattern)

    # Pattern norm (constant)
    pattern_norm = np.linalg.norm(pattern)

    # Raw cross-correlation
    raw_corr = fft_cross_correlation(signal, pattern)

    # Compute sliding window norms for signal
    # This is the expensive part - could also be done with FFT trick
    window_norms = np.zeros(n_signal)
    for i in range(n_signal):
        end = min(i + n_pattern, n_signal)
        window = signal[i:end]
        window_norms[i] = np.linalg.norm(window)

    # Normalize
    normalized = raw_corr / (window_norms * pattern_norm + 1e-9)

    return normalized


def fft_best_match(signal: np.ndarray, pattern: np.ndarray, use_mrl: int = None) -> tuple[float, int]:
    """
    Finde beste Match-Position und Score.

    Args:
        signal: Doc token embeddings
        pattern: Query token embeddings
        use_mrl: Optional - nur erste N dims verwenden

    Returns:
        (best_score, best_position)
    """
    if use_mrl:
        signal = signal[:, :use_mrl]
        pattern = pattern[:, :use_mrl]

    correlation = fft_cross_correlation_normalized(signal, pattern)
    best_pos = np.argmax(correlation)
    best_score = correlation[best_pos]

    return best_score, best_pos


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


# =============================================================================
# INDEX
# =============================================================================

@dataclass
class DocumentIndex:
    doc_id: str
    token_embeddings: np.ndarray
    doc_embedding: np.ndarray  # Mean-pooled for Stage 1


@dataclass
class FullIndex:
    docs: dict = field(default_factory=dict)


def build_index(docs: list[dict], token_encoder: TokenEncoder) -> FullIndex:
    index = FullIndex()

    for i, doc in enumerate(docs):
        doc_id = doc["id"]
        token_embs = token_encoder.encode(doc["text"])

        index.docs[doc_id] = DocumentIndex(
            doc_id=doc_id,
            token_embeddings=token_embs,
            doc_embedding=token_embs.mean(axis=0)
        )

        if (i + 1) % 50 == 0:
            print(f"    Indexed {i+1}/{len(docs)}")

    return index


# =============================================================================
# SEARCH METHODS
# =============================================================================

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)


def search_cosine_pooled(query_emb: np.ndarray, index: FullIndex, top_k: int = 10) -> list:
    """Baseline: Pooled Query, Cosine mit Doc-Mean."""
    results = []
    for doc_id, doc in index.docs.items():
        score = cosine_sim(query_emb, doc.doc_embedding)
        results.append((doc_id, score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def search_cosine_token(query_emb: np.ndarray, index: FullIndex, top_k: int = 10) -> list:
    """Bisherig: Pooled Query, Cosine mit jedem Token."""
    results = []
    for doc_id, doc in index.docs.items():
        best_score = max(cosine_sim(query_emb, t) for t in doc.token_embeddings)
        results.append((doc_id, best_score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def search_fft_pattern(
    query_tokens: np.ndarray,
    index: FullIndex,
    doc_candidates: list = None,
    use_mrl: int = None,
    top_k: int = 10
) -> list:
    """
    FFT Pattern Search: Query als Token-Sequenz, Cross-Correlation.

    Args:
        query_tokens: Query token embeddings (nicht pooled!)
        index: Document index
        doc_candidates: Optional - nur diese Docs durchsuchen
        use_mrl: Optional - nur erste N dims
        top_k: Return top K results
    """
    docs_to_search = doc_candidates if doc_candidates else list(index.docs.keys())

    results = []
    for doc_id in docs_to_search:
        doc = index.docs[doc_id]
        score, pos = fft_best_match(doc.token_embeddings, query_tokens, use_mrl)
        results.append((doc_id, score, pos))

    results.sort(key=lambda x: x[1], reverse=True)
    return [(r[0], r[1]) for r in results[:top_k]]


def search_hybrid_fft(
    query_pooled: np.ndarray,
    query_tokens: np.ndarray,
    index: FullIndex,
    stage1_k: int = 100,
    mrl_stage1: int = 256,
    mrl_stage2: int = 1024,
    top_k: int = 10
) -> list:
    """
    Hybrid: Stage 1 Cosine (schnell), Stage 2 FFT (präzise).

    Stage 1: Pooled Query, Cosine mit Doc-Mean (MRL 256)
    Stage 2: FFT Pattern Match auf Candidates (full dims)
    """
    # Stage 1: Quick filter with pooled query
    query_s1 = query_pooled[:mrl_stage1]
    doc_scores = []
    for doc_id, doc in index.docs.items():
        doc_emb_s1 = doc.doc_embedding[:mrl_stage1]
        score = cosine_sim(query_s1, doc_emb_s1)
        doc_scores.append((doc_id, score))

    doc_scores.sort(key=lambda x: x[1], reverse=True)
    candidates = [d[0] for d in doc_scores[:stage1_k]]

    # Stage 2: FFT pattern match on candidates
    return search_fft_pattern(query_tokens, index, candidates, mrl_stage2, top_k)


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(
    queries: list,
    relevance: dict,
    search_fn,
    index: FullIndex,
    token_encoder: TokenEncoder,
    pooled_encoder: PooledEncoder,
    use_token_query: bool = False,
    top_k: int = 10
):
    """Evaluate search method."""
    hits = 0
    total = 0
    total_time = 0

    for query in queries:
        qid = query["id"]
        if qid not in relevance:
            continue

        relevant = set(relevance[qid])

        # Get query embeddings
        query_pooled = pooled_encoder.encode(query["text"])

        if use_token_query:
            query_tokens = token_encoder.encode(query["text"])
            start = time.time()
            results = search_fn(query_pooled, query_tokens, index, top_k=top_k)
        else:
            start = time.time()
            results = search_fn(query_pooled, index, top_k=top_k)

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

def load_dataset_small(name: str, max_docs: int = 300, max_queries: int = 30):
    """Smaller dataset for FFT testing (FFT is slower)."""
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
    print("FFT CROSS-CORRELATION SEARCH TEST")
    print("=" * 70)

    # Encoders
    token_encoder = TokenEncoder()
    pooled_encoder = PooledEncoder()

    # Dataset (smaller for FFT)
    print("\n1. Loading dataset...")
    docs, queries, relevance = load_dataset_small("scifact", max_docs=300, max_queries=30)
    if docs is None:
        return

    # Build index
    print("\n2. Building index...")
    start = time.time()
    index = build_index(docs, token_encoder)
    print(f"   Done in {time.time() - start:.1f}s")

    # Test methods
    print("\n3. Testing search methods...")
    print("=" * 70)

    # Baseline: Pooled + Doc-Mean
    print("  Cosine (Pooled → Doc-Mean)...")
    r1, t1 = evaluate(
        queries, relevance,
        lambda qp, idx, **kw: search_cosine_pooled(qp, idx, kw.get("top_k", 10)),
        index, token_encoder, pooled_encoder,
        use_token_query=False
    )

    # Token BF: Pooled + All Tokens
    print("  Cosine (Pooled → Token BF)...")
    r2, t2 = evaluate(
        queries, relevance,
        lambda qp, idx, **kw: search_cosine_token(qp, idx, kw.get("top_k", 10)),
        index, token_encoder, pooled_encoder,
        use_token_query=False
    )

    # FFT: Token Query → Pattern Match (all docs)
    print("  FFT Pattern (Query Tokens → All Docs)...")
    r3, t3 = evaluate(
        queries, relevance,
        lambda qp, qt, idx, **kw: search_fft_pattern(qt, idx, use_mrl=1024, top_k=kw.get("top_k", 10)),
        index, token_encoder, pooled_encoder,
        use_token_query=True
    )

    # FFT with MRL: Reduced dims
    print("  FFT Pattern MRL-256...")
    r4, t4 = evaluate(
        queries, relevance,
        lambda qp, qt, idx, **kw: search_fft_pattern(qt, idx, use_mrl=256, top_k=kw.get("top_k", 10)),
        index, token_encoder, pooled_encoder,
        use_token_query=True
    )

    # Hybrid: Stage 1 Cosine, Stage 2 FFT
    print("  Hybrid (Cosine Stage 1 → FFT Stage 2)...")
    r5, t5 = evaluate(
        queries, relevance,
        lambda qp, qt, idx, **kw: search_hybrid_fft(qp, qt, idx, stage1_k=50, top_k=kw.get("top_k", 10)),
        index, token_encoder, pooled_encoder,
        use_token_query=True
    )

    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n  {'Method':<35} {'R@10':>8} {'Time':>10}")
    print(f"  {'-'*55}")
    print(f"  {'Cosine (Pooled → Doc-Mean)':<35} {r1*100:>7.1f}% {t1:>8.1f}ms")
    print(f"  {'Cosine (Pooled → Token BF)':<35} {r2*100:>7.1f}% {t2:>8.1f}ms")
    print(f"  {'FFT Pattern (1024 dims)':<35} {r3*100:>7.1f}% {t3:>8.1f}ms")
    print(f"  {'FFT Pattern MRL (256 dims)':<35} {r4*100:>7.1f}% {t4:>8.1f}ms")
    print(f"  {'Hybrid (Cosine → FFT)':<35} {r5*100:>7.1f}% {t5:>8.1f}ms")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    print(f"\n  FFT vs Cosine Token BF:")
    print(f"    Recall: {r3*100:.1f}% vs {r2*100:.1f}% ({(r3-r2)*100:+.1f}%)")
    print(f"    Time:   {t3:.1f}ms vs {t2:.1f}ms ({t3/t2:.1f}x)")

    print(f"\n  FFT MRL-256 vs FFT-1024:")
    print(f"    Recall: {r4*100:.1f}% vs {r3*100:.1f}% ({(r4-r3)*100:+.1f}%)")
    print(f"    Time:   {t4:.1f}ms vs {t3:.1f}ms ({t4/t3:.1f}x)")

    print(f"\n  Hybrid vs Token BF:")
    print(f"    Recall: {r5*100:.1f}% vs {r2*100:.1f}% ({(r5-r2)*100:+.1f}%)")
    print(f"    Time:   {t5:.1f}ms vs {t2:.1f}ms ({t5/t2:.1f}x)")

    print("\n  Note: FFT wird schneller relativ zu Cosine bei längeren Dokumenten")
    print("        und auf GPU (alle Dimensionen parallel).")


if __name__ == "__main__":
    main()
