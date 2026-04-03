#!/usr/bin/env python3
"""
Vereinfachter Recall Test - nur Doc-Level (pooled) Embeddings.
Schneller als Token-Level Test.
"""

import numpy as np
import requests
from datasets import load_dataset
import sys


class TokenEncoder:
    def __init__(self, url: str = "http://localhost:8202"):
        self.url = url

    def encode_pooled(self, text: str, max_chars: int = 4000) -> np.ndarray:
        """Encode and return mean-pooled embedding."""
        text = text[:max_chars]
        response = requests.post(f"{self.url}/embeddings", json={"input": text})
        response.raise_for_status()
        data = response.json()
        if isinstance(data[0]["embedding"][0], list):
            tokens = np.array(data[0]["embedding"], dtype=np.float32)
            return tokens.mean(axis=0)
        return np.array(data[0]["embedding"], dtype=np.float32)


def cosine_similarity_matrix(queries: np.ndarray, docs: np.ndarray) -> np.ndarray:
    q_norm = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8)
    d_norm = docs / (np.linalg.norm(docs, axis=1, keepdims=True) + 1e-8)
    return q_norm @ d_norm.T


def recall_at_k(scores: np.ndarray, relevant: list[set], k: int = 10) -> float:
    recalls = []
    for i, rel in enumerate(relevant):
        if not rel:
            continue
        top_k = set(np.argsort(scores[i])[-k:][::-1])
        recall = len(top_k & rel) / min(len(rel), k)
        recalls.append(recall)
    return np.mean(recalls) if recalls else 0.0


def quantize_int8(embeddings: np.ndarray) -> tuple[np.ndarray, float]:
    """Quantize to int8 with scale."""
    scale = np.abs(embeddings).max() + 1e-8
    quantized = np.clip(embeddings / scale * 127, -128, 127).astype(np.int8)
    return quantized, scale


def dequantize_int8(quantized: np.ndarray, scale: float) -> np.ndarray:
    return quantized.astype(np.float32) / 127 * scale


def rvq_simple(embedding: np.ndarray, n_bits: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Simple RVQ: quantize each Matryoshka band separately."""
    bands = [(0, 256), (256, 512), (512, 768), (768, 1024)]
    codes = []
    scales = []

    for start, end in bands:
        if start >= len(embedding):
            break
        actual_end = min(end, len(embedding))
        band = embedding[start:actual_end]
        q, s = quantize_int8(band)
        codes.append(q)
        scales.append(s)

    return np.concatenate(codes), np.array(scales, dtype=np.float32)


def rvq_decode(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Decode simple RVQ."""
    bands = [(0, 256), (256, 512), (512, 768), (768, 1024)]
    embedding = np.zeros(1024, dtype=np.float32)

    offset = 0
    for i, (start, end) in enumerate(bands):
        if i >= len(scales):
            break
        band_len = end - start
        band_codes = codes[offset:offset + band_len]
        embedding[start:end] = dequantize_int8(band_codes, scales[i])
        offset += band_len

    return embedding


def run_test():
    print("=" * 70)
    print("COMPRESSION RECALL TEST (Simple)")
    print("=" * 70)
    sys.stdout.flush()

    encoder = TokenEncoder()

    # Load SciFact
    print("\n1. Loading SciFact...", flush=True)
    dataset = load_dataset("mteb/scifact", "corpus")
    queries_ds = load_dataset("mteb/scifact", "queries")

    n_docs = 100
    n_queries = 30

    corpus = list(dataset["corpus"])[:n_docs]
    queries = list(queries_ds["queries"])[:n_queries]

    print(f"   Docs: {len(corpus)}, Queries: {len(queries)}", flush=True)

    # Build relevance
    qrels = load_dataset("mteb/scifact", "default")
    query_ids = [q["_id"] for q in queries]
    doc_ids = [d["_id"] for d in corpus]
    doc_id_to_idx = {did: i for i, did in enumerate(doc_ids)}

    relevant = []
    for qid in query_ids:
        rel_docs = set()
        for split in ["test", "validation"]:
            if split in qrels:
                for item in qrels[split]:
                    if item["query-id"] == qid:
                        if item["corpus-id"] in doc_id_to_idx:
                            rel_docs.add(doc_id_to_idx[item["corpus-id"]])
        relevant.append(rel_docs)

    n_with_rel = sum(1 for r in relevant if r)
    print(f"   Queries with relevant docs: {n_with_rel}", flush=True)

    # Encode
    print("\n2. Encoding...", flush=True)
    doc_embeddings = []
    for i, doc in enumerate(corpus):
        text = doc.get("title", "") + " " + doc.get("text", "")
        emb = encoder.encode_pooled(text)
        doc_embeddings.append(emb)
        if (i + 1) % 20 == 0:
            print(f"   {i+1}/{n_docs} docs", flush=True)
    doc_embeddings = np.array(doc_embeddings)

    query_embeddings = []
    for q in queries:
        emb = encoder.encode_pooled(q["text"])
        query_embeddings.append(emb)
    query_embeddings = np.array(query_embeddings)

    print(f"   Docs: {doc_embeddings.shape}, Queries: {query_embeddings.shape}", flush=True)

    # Baseline
    print("\n3. Baseline (raw)...", flush=True)
    scores_raw = cosine_similarity_matrix(query_embeddings, doc_embeddings)
    recall_raw = recall_at_k(scores_raw, relevant, k=10)
    print(f"   R@10: {recall_raw*100:.1f}%", flush=True)

    # Int8 Quantization
    print("\n4. Int8 Quantization...", flush=True)
    doc_q8 = []
    doc_scales = []
    for emb in doc_embeddings:
        q, s = quantize_int8(emb)
        doc_q8.append(q)
        doc_scales.append(s)

    doc_decoded = np.array([dequantize_int8(q, s) for q, s in zip(doc_q8, doc_scales)])
    scores_q8 = cosine_similarity_matrix(query_embeddings, doc_decoded)
    recall_q8 = recall_at_k(scores_q8, relevant, k=10)

    size_raw = doc_embeddings.nbytes
    size_q8 = sum(q.nbytes for q in doc_q8) + len(doc_scales) * 4
    print(f"   R@10: {recall_q8*100:.1f}% (vs {recall_raw*100:.1f}%)", flush=True)
    print(f"   Size: {size_q8:,} bytes ({size_q8/size_raw*100:.1f}% of raw)", flush=True)

    # RVQ (per-band quantization)
    print("\n5. RVQ (per-band int8)...", flush=True)
    doc_rvq = []
    doc_rvq_scales = []
    for emb in doc_embeddings:
        codes, scales = rvq_simple(emb)
        doc_rvq.append(codes)
        doc_rvq_scales.append(scales)

    doc_rvq_decoded = np.array([rvq_decode(c, s) for c, s in zip(doc_rvq, doc_rvq_scales)])
    scores_rvq = cosine_similarity_matrix(query_embeddings, doc_rvq_decoded)
    recall_rvq = recall_at_k(scores_rvq, relevant, k=10)

    size_rvq = sum(c.nbytes for c in doc_rvq) + sum(s.nbytes for s in doc_rvq_scales)
    print(f"   R@10: {recall_rvq*100:.1f}% (vs {recall_raw*100:.1f}%)", flush=True)
    print(f"   Size: {size_rvq:,} bytes ({size_rvq/size_raw*100:.1f}% of raw)", flush=True)

    # Sign-Hash (1 bit per dim)
    print("\n6. Sign-Hash (1 bit/dim)...", flush=True)
    doc_sign = (doc_embeddings > 0).astype(np.uint8)
    query_sign = (query_embeddings > 0).astype(np.uint8)

    # Hamming similarity (proportion of matching bits)
    scores_sign = np.zeros((len(queries), n_docs))
    for i in range(len(queries)):
        for j in range(n_docs):
            matches = (query_sign[i] == doc_sign[j]).sum()
            scores_sign[i, j] = matches / len(query_sign[i])

    recall_sign = recall_at_k(scores_sign, relevant, k=10)
    size_sign = doc_sign.nbytes
    print(f"   R@10: {recall_sign*100:.1f}% (vs {recall_raw*100:.1f}%)", flush=True)
    print(f"   Size: {size_sign:,} bytes ({size_sign/size_raw*100:.1f}% of raw)", flush=True)

    # Packed Sign-Hash (8 dims per byte)
    print("\n7. Packed Sign-Hash (32x compression)...", flush=True)
    doc_packed = np.packbits(doc_sign, axis=-1)
    size_packed = doc_packed.nbytes
    print(f"   Same R@10: {recall_sign*100:.1f}%", flush=True)
    print(f"   Size: {size_packed:,} bytes ({size_packed/size_raw*100:.1f}% of raw)", flush=True)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n   {'Method':<20} {'R@10':>8} {'Size':>12} {'Compress':>10}")
    print("   " + "-" * 52)
    print(f"   {'Raw (float32)':<20} {recall_raw*100:>7.1f}% {size_raw:>11,} {'1.0x':>10}")
    print(f"   {'Int8 Global':<20} {recall_q8*100:>7.1f}% {size_q8:>11,} {size_raw/size_q8:>9.1f}x")
    print(f"   {'RVQ (per-band)':<20} {recall_rvq*100:>7.1f}% {size_rvq:>11,} {size_raw/size_rvq:>9.1f}x")
    print(f"   {'Sign-Hash':<20} {recall_sign*100:>7.1f}% {size_sign:>11,} {size_raw/size_sign:>9.1f}x")
    print(f"   {'Sign-Hash (packed)':<20} {recall_sign*100:>7.1f}% {size_packed:>11,} {size_raw/size_packed:>9.1f}x")

    print("\n   Conclusion:")
    if recall_q8 >= recall_raw * 0.99:
        print("   ✓ Int8 quantization preserves recall (4x compression)")
    if recall_rvq >= recall_raw * 0.99:
        print("   ✓ RVQ per-band preserves recall (4x compression)")
    if recall_sign >= recall_raw * 0.95:
        print("   ✓ Sign-Hash maintains 95%+ recall (32x compression)")


if __name__ == "__main__":
    run_test()
