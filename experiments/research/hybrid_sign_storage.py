#!/usr/bin/env python3
"""
Hybrid Sign Storage - Stabile vs Volatile Dimensionen

Idee aus Audio-Engineering:
- Stabile Dimensionen (kein Flip): Einmal pro Dokument speichern
- Volatile Dimensionen (häufige Flips): Pro Token speichern

Erwartete Ersparnis: ~40% bei Token-Level Storage
"""

import numpy as np
import requests
import time
from dataclasses import dataclass
from collections import defaultdict


# =============================================================================
# ENCODER
# =============================================================================

class TokenEncoder:
    def __init__(self, url: str = "http://localhost:8202"):
        self.url = url
        self.dim = 1024

    def encode(self, text: str) -> np.ndarray:
        text = text[:4000]
        response = requests.post(
            f"{self.url}/embeddings",
            json={"input": text}
        )
        response.raise_for_status()
        data = response.json()

        if isinstance(data[0]["embedding"][0], list):
            return np.array(data[0]["embedding"])
        return np.array([data[0]["embedding"]])


# =============================================================================
# HYBRID SIGN STORAGE
# =============================================================================

@dataclass
class HybridSignIndex:
    """
    Hybrid Storage: Stabile Dims einmal, volatile Dims pro Token.
    """
    # Pro Dokument
    doc_ids: list
    stable_masks: dict      # doc_id → np.array (dim,) bool - welche dims sind stabil
    stable_signs: dict      # doc_id → np.array (dim,) bool - sign für stabile dims
    volatile_signs: dict    # doc_id → np.array (n_tokens, dim) bool - nur volatile
    n_tokens: dict          # doc_id → int
    texts: dict

    # Statistiken
    total_stable_dims: int = 0
    total_volatile_dims: int = 0


@dataclass
class FullSignIndex:
    """
    Baseline: Alle Signs pro Token speichern.
    """
    doc_ids: list
    signs: dict             # doc_id → np.array (n_tokens, dim) bool
    n_tokens: dict
    texts: dict


def compute_stability_mask(embeddings: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """
    Berechne welche Dimensionen stabil sind (Flip-Rate < threshold).
    Returns: bool array (dim,) - True = stabil
    """
    signs = embeddings > 0
    n_tokens = len(embeddings)

    if n_tokens < 2:
        return np.ones(embeddings.shape[1], dtype=bool)

    # Flip-Rate pro Dimension
    sign_changes = np.diff(signs.astype(np.int8), axis=0)
    flips_per_dim = np.abs(sign_changes).sum(axis=0)
    flip_rate = flips_per_dim / (n_tokens - 1)

    # Stabil = Flip-Rate unter Threshold
    stable = flip_rate < threshold

    return stable


def build_hybrid_index(
    docs: list[dict],
    encoder: TokenEncoder,
    stability_threshold: float = 0.1
) -> HybridSignIndex:
    """Baue Hybrid Sign Index."""
    index = HybridSignIndex(
        doc_ids=[],
        stable_masks={},
        stable_signs={},
        volatile_signs={},
        n_tokens={},
        texts={}
    )

    total_stable = 0
    total_volatile = 0

    for i, doc in enumerate(docs):
        doc_id = doc["id"]
        text = doc["text"]

        embeddings = encoder.encode(text)
        signs = embeddings > 0

        # Stabilität berechnen
        stable_mask = compute_stability_mask(embeddings, stability_threshold)
        n_stable = stable_mask.sum()
        n_volatile = (~stable_mask).sum()

        total_stable += n_stable
        total_volatile += n_volatile

        # Stabile Signs: Mehrheitsentscheidung (sollte eh gleich sein)
        stable_signs_doc = (signs[:, stable_mask].mean(axis=0) > 0.5)

        # Volatile Signs: Alle Tokens, nur volatile Dims
        volatile_signs_doc = signs[:, ~stable_mask]

        index.doc_ids.append(doc_id)
        index.stable_masks[doc_id] = stable_mask
        index.stable_signs[doc_id] = stable_signs_doc
        index.volatile_signs[doc_id] = volatile_signs_doc
        index.n_tokens[doc_id] = len(embeddings)
        index.texts[doc_id] = text

        if (i + 1) % 20 == 0:
            print(f"  Hybrid Index: {i+1}/{len(docs)}")

    index.total_stable_dims = total_stable
    index.total_volatile_dims = total_volatile

    return index


def build_full_index(docs: list[dict], encoder: TokenEncoder) -> FullSignIndex:
    """Baue Full Sign Index (Baseline)."""
    index = FullSignIndex(
        doc_ids=[],
        signs={},
        n_tokens={},
        texts={}
    )

    for i, doc in enumerate(docs):
        doc_id = doc["id"]
        text = doc["text"]

        embeddings = encoder.encode(text)
        signs = embeddings > 0

        index.doc_ids.append(doc_id)
        index.signs[doc_id] = signs
        index.n_tokens[doc_id] = len(embeddings)
        index.texts[doc_id] = text

        if (i + 1) % 20 == 0:
            print(f"  Full Index: {i+1}/{len(docs)}")

    return index


# =============================================================================
# STORAGE SIZE
# =============================================================================

def compute_hybrid_size(index: HybridSignIndex) -> dict:
    """Berechne Speicherbedarf für Hybrid Index."""
    # Stable masks: 1 bit pro dim pro doc
    mask_bits = len(index.doc_ids) * 1024

    # Stable signs: 1 bit pro dim pro doc
    stable_bits = len(index.doc_ids) * 1024

    # Volatile signs: n_tokens × n_volatile_dims pro doc
    volatile_bits = 0
    for doc_id in index.doc_ids:
        n_tokens = index.n_tokens[doc_id]
        n_volatile = index.volatile_signs[doc_id].shape[1]
        volatile_bits += n_tokens * n_volatile

    total_bits = mask_bits + stable_bits + volatile_bits
    total_bytes = total_bits / 8

    return {
        "mask_bytes": mask_bits / 8,
        "stable_bytes": stable_bits / 8,
        "volatile_bytes": volatile_bits / 8,
        "total_bytes": total_bytes,
        "total_kb": total_bytes / 1024,
    }


def compute_full_size(index: FullSignIndex) -> dict:
    """Berechne Speicherbedarf für Full Index."""
    total_bits = 0
    for doc_id in index.doc_ids:
        n_tokens = index.n_tokens[doc_id]
        total_bits += n_tokens * 1024

    total_bytes = total_bits / 8

    return {
        "total_bytes": total_bytes,
        "total_kb": total_bytes / 1024,
    }


# =============================================================================
# SEARCH
# =============================================================================

def pack_signs(signs: np.ndarray) -> np.ndarray:
    """Pack bool array zu bytes."""
    n_bits = len(signs)
    n_bytes = (n_bits + 7) // 8
    packed = np.zeros(n_bytes, dtype=np.uint8)

    for i in range(min(n_bits, n_bytes * 8)):
        if signs[i]:
            packed[i // 8] |= (1 << (i % 8))

    return packed


def hamming_similarity(hash1: np.ndarray, hash2: np.ndarray) -> float:
    """Hamming Similarity zwischen gepackten Sign-Hashes."""
    xor = np.bitwise_xor(hash1, hash2)
    diff_bits = np.unpackbits(xor).sum()
    total_bits = len(hash1) * 8
    return 1.0 - (diff_bits / total_bits)


def search_hybrid(
    query_emb: np.ndarray,
    index: HybridSignIndex,
    top_k: int = 10
) -> list[tuple[str, float, int]]:
    """
    Suche mit Hybrid Index.
    """
    query_signs = query_emb > 0

    results = []

    for doc_id in index.doc_ids:
        stable_mask = index.stable_masks[doc_id]
        stable_signs = index.stable_signs[doc_id]
        volatile_signs = index.volatile_signs[doc_id]
        n_tokens = index.n_tokens[doc_id]

        # Query aufteilen
        query_stable = query_signs[stable_mask]
        query_volatile = query_signs[~stable_mask]

        # Stable Score (einmal pro Doc)
        stable_match = (query_stable == stable_signs).sum()
        n_stable = len(stable_signs)

        best_score = 0.0
        best_pos = 0

        for t in range(n_tokens):
            # Volatile Score
            volatile_match = (query_volatile == volatile_signs[t]).sum()
            n_volatile = len(volatile_signs[t])

            # Kombinierter Score
            total_match = stable_match + volatile_match
            total_dims = n_stable + n_volatile
            score = total_match / total_dims

            if score > best_score:
                best_score = score
                best_pos = t

        results.append((doc_id, best_score, best_pos))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def search_full(
    query_emb: np.ndarray,
    index: FullSignIndex,
    top_k: int = 10
) -> list[tuple[str, float, int]]:
    """
    Suche mit Full Index (Baseline).
    """
    query_signs = query_emb > 0

    results = []

    for doc_id in index.doc_ids:
        signs = index.signs[doc_id]
        n_tokens = index.n_tokens[doc_id]

        best_score = 0.0
        best_pos = 0

        for t in range(n_tokens):
            match = (query_signs == signs[t]).sum()
            score = match / len(query_signs)

            if score > best_score:
                best_score = score
                best_pos = t

        results.append((doc_id, best_score, best_pos))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


# =============================================================================
# EVALUATION
# =============================================================================

def load_scifact(n_docs: int = 100, n_queries: int = 50):
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


class PooledEncoder:
    """Pooled Encoder für Queries."""
    def __init__(self, url: str = "http://localhost:8200"):
        self.url = url

    def encode(self, texts: list[str]) -> np.ndarray:
        all_embs = []
        for text in texts:
            response = requests.post(
                f"{self.url}/v1/embeddings",
                json={"input": [text[:4000]]}
            )
            response.raise_for_status()
            data = response.json()
            all_embs.append(data["data"][0]["embedding"])
        return np.array(all_embs)


def main():
    print("=" * 70)
    print("HYBRID SIGN STORAGE TEST")
    print("=" * 70)

    # Encoder
    token_encoder = TokenEncoder()
    pooled_encoder = PooledEncoder()

    # Daten laden
    print("\n1. Lade SciFact...")
    n_docs = 100
    docs, queries, relevance = load_scifact(n_docs, 50)
    print(f"   {len(docs)} Dokumente, {len(queries)} Queries")

    # Full Index bauen
    print("\n2. Baue Full Sign Index (Baseline)...")
    start = time.time()
    full_index = build_full_index(docs, token_encoder)
    full_time = time.time() - start
    print(f"   Zeit: {full_time:.1f}s")

    # Hybrid Index bauen
    print("\n3. Baue Hybrid Sign Index...")
    start = time.time()
    hybrid_index = build_hybrid_index(docs, token_encoder, stability_threshold=0.1)
    hybrid_time = time.time() - start
    print(f"   Zeit: {hybrid_time:.1f}s")

    # Speichervergleich
    print("\n" + "=" * 70)
    print("SPEICHERVERGLEICH")
    print("=" * 70)

    full_size = compute_full_size(full_index)
    hybrid_size = compute_hybrid_size(hybrid_index)

    print(f"\n   Full Index:")
    print(f"     Total: {full_size['total_kb']:.1f} KB")

    print(f"\n   Hybrid Index:")
    print(f"     Masks:    {hybrid_size['mask_bytes']/1024:.1f} KB")
    print(f"     Stable:   {hybrid_size['stable_bytes']/1024:.1f} KB")
    print(f"     Volatile: {hybrid_size['volatile_bytes']/1024:.1f} KB")
    print(f"     Total:    {hybrid_size['total_kb']:.1f} KB")

    savings = (1 - hybrid_size['total_bytes'] / full_size['total_bytes']) * 100
    print(f"\n   Ersparnis: {savings:.1f}%")

    # Stabilität-Statistiken
    n_docs_total = len(hybrid_index.doc_ids)
    avg_stable = hybrid_index.total_stable_dims / n_docs_total
    avg_volatile = hybrid_index.total_volatile_dims / n_docs_total

    print(f"\n   Durchschnittlich pro Dokument:")
    print(f"     Stabile Dims:   {avg_stable:.0f} ({avg_stable/1024*100:.1f}%)")
    print(f"     Volatile Dims:  {avg_volatile:.0f} ({avg_volatile/1024*100:.1f}%)")

    # Search Evaluation
    print("\n" + "=" * 70)
    print("SEARCH EVALUATION")
    print("=" * 70)

    # Query Embeddings
    print("\n   Encode Queries...")
    query_embs = pooled_encoder.encode([q["text"] for q in queries])

    # Full Search
    print("   Full Search...")
    full_hits = 0
    full_time_total = 0
    for i, query in enumerate(queries):
        if query["id"] not in relevance:
            continue
        relevant = set(relevance[query["id"]])

        start = time.time()
        results = search_full(query_embs[i], full_index, top_k=10)
        full_time_total += time.time() - start

        found = set(r[0] for r in results)
        if relevant & found:
            full_hits += 1

    # Hybrid Search
    print("   Hybrid Search...")
    hybrid_hits = 0
    hybrid_time_total = 0
    for i, query in enumerate(queries):
        if query["id"] not in relevance:
            continue
        relevant = set(relevance[query["id"]])

        start = time.time()
        results = search_hybrid(query_embs[i], hybrid_index, top_k=10)
        hybrid_time_total += time.time() - start

        found = set(r[0] for r in results)
        if relevant & found:
            hybrid_hits += 1

    n_queries = len([q for q in queries if q["id"] in relevance])

    print(f"\n   {'Methode':<15} {'R@10':>8} {'Zeit':>12} {'Speicher':>12}")
    print(f"   {'-'*47}")
    print(f"   {'Full':<15} {full_hits/n_queries*100:>7.1f}% {full_time_total*1000/n_queries:>10.1f}ms {full_size['total_kb']:>10.1f}KB")
    print(f"   {'Hybrid':<15} {hybrid_hits/n_queries*100:>7.1f}% {hybrid_time_total*1000/n_queries:>10.1f}ms {hybrid_size['total_kb']:>10.1f}KB")

    print(f"\n   Speicher-Ersparnis: {savings:.1f}%")
    print(f"   Recall identisch: {'Ja' if full_hits == hybrid_hits else 'Nein'}")


if __name__ == "__main__":
    main()
