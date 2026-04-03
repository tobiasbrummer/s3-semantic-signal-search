#!/usr/bin/env python3
"""
Recall Test auf komprimierte Embeddings.
Testet ob RVQ/GOP/Combined für Search brauchbar sind.
"""

import numpy as np
import requests
from datasets import load_dataset
from compression_test import RVQEncoder, GOPEncoder, CombinedEncoder
import time


class TokenEncoder:
    def __init__(self, url: str = "http://localhost:8202"):
        self.url = url

    def encode(self, text: str, max_chars: int = 4000) -> np.ndarray:
        text = text[:max_chars]
        response = requests.post(f"{self.url}/embeddings", json={"input": text})
        response.raise_for_status()
        data = response.json()
        if isinstance(data[0]["embedding"][0], list):
            return np.array(data[0]["embedding"], dtype=np.float32)
        return np.array([data[0]["embedding"]], dtype=np.float32)


def cosine_similarity_matrix(queries: np.ndarray, docs: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between queries and docs."""
    # Normalize
    q_norm = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8)
    d_norm = docs / (np.linalg.norm(docs, axis=1, keepdims=True) + 1e-8)
    return q_norm @ d_norm.T


def recall_at_k(scores: np.ndarray, relevant: list[set], k: int = 10) -> float:
    """Compute recall@k."""
    recalls = []
    for i, rel in enumerate(relevant):
        if not rel:
            continue
        top_k = set(np.argsort(scores[i])[-k:][::-1])
        recall = len(top_k & rel) / min(len(rel), k)
        recalls.append(recall)
    return np.mean(recalls) if recalls else 0.0


def mean_pool(embeddings: np.ndarray) -> np.ndarray:
    """Mean pool token embeddings to single vector."""
    return embeddings.mean(axis=0)


def run_recall_test():
    print("=" * 70)
    print("COMPRESSION RECALL TEST")
    print("=" * 70)

    encoder = TokenEncoder()

    # Load SciFact
    print("\n1. Loading SciFact dataset...")
    dataset = load_dataset("mteb/scifact", "corpus")
    queries_ds = load_dataset("mteb/scifact", "queries")

    # Limit for speed
    n_docs = 1000
    n_queries = 500

    corpus = list(dataset["corpus"])[:n_docs]
    queries = list(queries_ds["queries"])[:n_queries]

    print(f"   Docs: {len(corpus)}, Queries: {len(queries)}")

    # Build relevance from qrels
    qrels = load_dataset("mteb/scifact", "default")
    relevant = []
    query_ids = [q["_id"] for q in queries]
    doc_ids = [d["_id"] for d in corpus]
    doc_id_to_idx = {did: i for i, did in enumerate(doc_ids)}
    
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
    print(f"   Queries with relevant docs in corpus: {n_with_rel}")

    # Encode documents
    print("\n2. Encoding documents...")
    doc_embeddings_list = []
    doc_pooled = []

    for i, doc in enumerate(corpus):
        text = doc.get("title", "") + " " + doc.get("text", "")
        emb = encoder.encode(text)
        doc_embeddings_list.append(emb)
        doc_pooled.append(mean_pool(emb))
        if (i + 1) % 50 == 0:
            print(f"   {i+1}/{len(corpus)} docs encoded")

    doc_pooled = np.array(doc_pooled)
    print(f"   Doc pooled shape: {doc_pooled.shape}")

    # Encode queries
    print("\n3. Encoding queries...")
    query_pooled = []
    for q in queries:
        emb = encoder.encode(q["text"])
        query_pooled.append(mean_pool(emb))
    query_pooled = np.array(query_pooled)
    print(f"   Query pooled shape: {query_pooled.shape}")

    # Baseline: Raw pooled embeddings
    print("\n4. Computing baseline (raw pooled)...")
    scores_raw = cosine_similarity_matrix(query_pooled, doc_pooled)
    recall_raw = recall_at_k(scores_raw, relevant, k=10)
    print(f"   Baseline R@10: {recall_raw*100:.1f}%")

    # Train compression on doc embeddings
    print("\n5. Training compression methods...")
    all_tokens = np.vstack(doc_embeddings_list)
    print(f"   Total tokens for training: {all_tokens.shape}")

    rvq = RVQEncoder(n_codes=256)
    rvq.train(all_tokens)

    gop = GOPEncoder(onset_threshold=95, quantize_p=True)

    combined = CombinedEncoder(n_codes=256, onset_threshold=95, delta_threshold_pct=50)
    combined.train(all_tokens)

    # Test each compression method
    results = {}

    # RVQ
    print("\n6. Testing RVQ...")
    doc_pooled_rvq = []
    for emb in doc_embeddings_list:
        codes = rvq.encode(emb)
        decoded = rvq.decode(codes)
        doc_pooled_rvq.append(mean_pool(decoded))
    doc_pooled_rvq = np.array(doc_pooled_rvq)

    scores_rvq = cosine_similarity_matrix(query_pooled, doc_pooled_rvq)
    recall_rvq = recall_at_k(scores_rvq, relevant, k=10)
    print(f"   RVQ R@10: {recall_rvq*100:.1f}% (vs {recall_raw*100:.1f}% raw)")
    results['RVQ'] = recall_rvq

    # GOP
    print("\n7. Testing GOP...")
    doc_pooled_gop = []
    for emb in doc_embeddings_list:
        encoded = gop.encode(emb)
        decoded = gop.decode(encoded)
        doc_pooled_gop.append(mean_pool(decoded))
    doc_pooled_gop = np.array(doc_pooled_gop)

    scores_gop = cosine_similarity_matrix(query_pooled, doc_pooled_gop)
    recall_gop = recall_at_k(scores_gop, relevant, k=10)
    print(f"   GOP R@10: {recall_gop*100:.1f}% (vs {recall_raw*100:.1f}% raw)")
    results['GOP'] = recall_gop

    # Combined
    print("\n8. Testing Combined...")
    doc_pooled_combined = []
    for emb in doc_embeddings_list:
        encoded = combined.encode(emb)
        decoded = combined.decode(encoded)
        doc_pooled_combined.append(mean_pool(decoded))
    doc_pooled_combined = np.array(doc_pooled_combined)

    scores_combined = cosine_similarity_matrix(query_pooled, doc_pooled_combined)
    recall_combined = recall_at_k(scores_combined, relevant, k=10)
    print(f"   Combined R@10: {recall_combined*100:.1f}% (vs {recall_raw*100:.1f}% raw)")
    results['Combined'] = recall_combined

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n   {'Method':<12} {'R@10':>8} {'vs Raw':>10}")
    print("   " + "-" * 32)
    print(f"   {'Raw':<12} {recall_raw*100:>7.1f}% {'baseline':>10}")

    for method, recall in results.items():
        diff = (recall - recall_raw) * 100
        diff_str = f"{diff:+.1f}%" if diff != 0 else "±0%"
        print(f"   {method:<12} {recall*100:>7.1f}% {diff_str:>10}")

    # Check if compression hurts recall
    print("\n   Conclusion:")
    if recall_combined >= recall_raw * 0.95:
        print("   ✓ Combined compression maintains >95% of baseline recall")
    else:
        print(f"   ⚠ Combined compression loses {(1 - recall_combined/recall_raw)*100:.1f}% recall")

    return results


if __name__ == "__main__":
    run_recall_test()
