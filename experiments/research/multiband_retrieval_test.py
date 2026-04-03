#!/usr/bin/env python3
"""
Multi-Band Retrieval Test

Kombiniert zwei Bänder:
- Dense (jina-v3): Semantische Ähnlichkeit
- SPLADE: Lexikalisches Matching

Testet verschiedene Kombinationsstrategien.
"""

import numpy as np
import requests
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from dataclasses import dataclass
from typing import Optional
from collections import defaultdict
import time


# =============================================================================
# DENSE ENCODER (jina-v3 via llama.cpp)
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


# =============================================================================
# SPLADE ENCODER
# =============================================================================

@dataclass
class SpladePeaks:
    """Sparse SPLADE Repräsentation."""
    term_ids: np.ndarray
    weights: np.ndarray

    @property
    def n_peaks(self) -> int:
        return len(self.term_ids)


class SpladeEncoder:
    """SPLADE Encoder."""

    def __init__(
        self,
        model_name: str = "naver/splade-cocondenser-ensembledistil",
        device: Optional[str] = None,
        top_k: int = 256
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

    def encode(self, texts: list[str]) -> list[SpladePeaks]:
        results = []
        for text in texts:
            inputs = self.tokenizer(
                text[:4000],
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            weights = torch.max(
                torch.log1p(torch.relu(logits)),
                dim=1
            ).values.squeeze(0).cpu().numpy()

            nonzero_idx = np.nonzero(weights)[0]
            nonzero_weights = weights[nonzero_idx]

            # Top-K
            if len(nonzero_idx) > self.top_k:
                top_idx = np.argsort(nonzero_weights)[-self.top_k:]
                nonzero_idx = nonzero_idx[top_idx]
                nonzero_weights = nonzero_weights[top_idx]

            results.append(SpladePeaks(
                term_ids=nonzero_idx,
                weights=nonzero_weights
            ))

        return results


# =============================================================================
# SCORING METHODS
# =============================================================================

def dense_cosine(query: np.ndarray, doc: np.ndarray) -> float:
    """Cosine Similarity für Dense Embeddings."""
    q_norm = query / (np.linalg.norm(query) + 1e-10)
    d_norm = doc / (np.linalg.norm(doc) + 1e-10)
    return float(np.dot(q_norm, d_norm))


def dense_sign_hamming(query: np.ndarray, doc: np.ndarray) -> float:
    """Sign-basierte Similarity (32x komprimiert)."""
    q_sign = np.sign(query)
    d_sign = np.sign(doc)
    return float(np.mean(q_sign == d_sign))


def splade_dot(query: SpladePeaks, doc: SpladePeaks) -> float:
    """Dot Product für SPLADE Peaks."""
    common = np.intersect1d(query.term_ids, doc.term_ids)
    if len(common) == 0:
        return 0.0

    score = 0.0
    for term_id in common:
        w1 = query.weights[query.term_ids == term_id][0]
        w2 = doc.weights[doc.term_ids == term_id][0]
        score += w1 * w2

    return score


def splade_jaccard(query: SpladePeaks, doc: SpladePeaks) -> float:
    """Jaccard Similarity für SPLADE Peaks."""
    set1 = set(query.term_ids)
    set2 = set(doc.term_ids)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


# =============================================================================
# DATA LOADING
# =============================================================================

def load_scifact(n_docs: int = 500, n_queries: int = 100):
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


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_method(
    name: str,
    score_fn,
    queries: list[dict],
    docs: list[dict],
    relevance: dict,
    k: int = 10
) -> tuple[float, float]:
    """Evaluiere eine Scoring-Methode. Returns (recall, time)."""

    start = time.time()
    hits = 0
    total = 0

    for query in queries:
        query_id = query["id"]
        if query_id not in relevance:
            continue

        relevant_docs = set(relevance[query_id])

        # Score für jedes Dokument
        doc_scores = []
        for doc in docs:
            score = score_fn(query, doc)
            doc_scores.append((doc["id"], score))

        # Top-K
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        top_k = set(doc_id for doc_id, _ in doc_scores[:k])

        if relevant_docs & top_k:
            hits += 1
        total += 1

    elapsed = time.time() - start
    recall = hits / total if total > 0 else 0.0

    return recall, elapsed


def main():
    print("=" * 70)
    print("MULTI-BAND RETRIEVAL TEST")
    print("=" * 70)

    # Encoder laden
    print("\n1. Lade Encoder...")
    print("   Dense (jina-v3)...")
    dense_encoder = DenseEncoder()
    print(f"   Dense dim: {dense_encoder.dim}")

    print("   SPLADE...")
    splade_encoder = SpladeEncoder()
    print(f"   SPLADE vocab: {splade_encoder.vocab_size}")

    # Daten laden
    print("\n2. Lade SciFact...")
    n_docs = 5000  # Voller Corpus
    n_queries = 300
    docs, queries, relevance = load_scifact(n_docs, n_queries)
    print(f"   {len(docs)} Dokumente, {len(queries)} Queries")

    # Embeddings erstellen
    print("\n3. Erstelle Embeddings...")

    doc_texts = [d["text"] for d in docs]
    query_texts = [q["text"] for q in queries]

    print("   Dense Docs...")
    start = time.time()
    doc_dense = dense_encoder.encode(doc_texts)
    print(f"   Dense Docs: {time.time()-start:.1f}s")

    print("   Dense Queries...")
    query_dense = dense_encoder.encode(query_texts)

    print("   SPLADE Docs (das dauert...)...")
    start = time.time()
    doc_splade = []
    for i, text in enumerate(doc_texts):
        doc_splade.extend(splade_encoder.encode([text]))
        if (i + 1) % 500 == 0:
            elapsed = time.time() - start
            print(f"      {i+1}/{len(doc_texts)} ({elapsed:.1f}s)")
    print(f"   SPLADE Docs: {time.time()-start:.1f}s")

    print("   SPLADE Queries...")
    query_splade = splade_encoder.encode(query_texts)

    # Attach embeddings to dicts for scoring
    for i, doc in enumerate(docs):
        doc["dense"] = doc_dense[i]
        doc["splade"] = doc_splade[i]

    for i, query in enumerate(queries):
        query["dense"] = query_dense[i]
        query["splade"] = query_splade[i]

    # Scoring-Methoden definieren
    print("\n4. Teste Scoring-Methoden...")
    print("=" * 70)

    methods = [
        # Dense only
        ("Dense Cosine", lambda q, d: dense_cosine(q["dense"], d["dense"])),
        ("Dense Sign", lambda q, d: dense_sign_hamming(q["dense"], d["dense"])),

        # SPLADE only
        ("SPLADE Dot", lambda q, d: splade_dot(q["splade"], d["splade"])),
        ("SPLADE Jaccard", lambda q, d: splade_jaccard(q["splade"], d["splade"])),

        # Combined: Normalisierte Addition
        ("Combined (0.5/0.5)", lambda q, d: (
            0.5 * dense_cosine(q["dense"], d["dense"]) +
            0.5 * splade_dot(q["splade"], d["splade"]) / 50  # Normalisierung
        )),

        # Combined: Dense-heavy
        ("Combined (0.7/0.3)", lambda q, d: (
            0.7 * dense_cosine(q["dense"], d["dense"]) +
            0.3 * splade_dot(q["splade"], d["splade"]) / 50
        )),

        # Combined: SPLADE-heavy
        ("Combined (0.3/0.7)", lambda q, d: (
            0.3 * dense_cosine(q["dense"], d["dense"]) +
            0.7 * splade_dot(q["splade"], d["splade"]) / 50
        )),

        # Combined: Multiplikativ
        ("Combined (multiply)", lambda q, d: (
            dense_cosine(q["dense"], d["dense"]) *
            (1 + splade_dot(q["splade"], d["splade"]) / 50)
        )),

        # Combined: Max
        ("Combined (max)", lambda q, d: max(
            dense_cosine(q["dense"], d["dense"]),
            splade_dot(q["splade"], d["splade"]) / 50
        )),
    ]

    results = []
    for name, score_fn in methods:
        recall, elapsed = evaluate_method(
            name, score_fn, queries, docs, relevance, k=10
        )
        results.append((name, recall, elapsed))
        print(f"   {name:25s}: R@10 = {recall*100:5.1f}% ({elapsed:.2f}s)")

    # Zusammenfassung
    print("\n" + "=" * 70)
    print("ERGEBNISSE")
    print("=" * 70)

    baseline = results[0][1]  # Dense Cosine

    print(f"\n{'Methode':<30} {'R@10':>8} {'vs Dense':>10}")
    print("-" * 50)

    for name, recall, _ in results:
        vs_baseline = (recall / baseline * 100) if baseline > 0 else 0
        marker = " ***" if recall > baseline else ""
        print(f"{name:<30} {recall*100:>7.1f}% {vs_baseline:>9.1f}%{marker}")

    # Beste Methode
    best = max(results, key=lambda x: x[1])
    print(f"\nBeste Methode: {best[0]} ({best[1]*100:.1f}%)")


if __name__ == "__main__":
    main()
