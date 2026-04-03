#!/usr/bin/env python3
"""
Dimensions-basiertes Matching Test

Statt Fenster-basiert zu suchen, analysieren wir jede Dimension separat:
- Query hat Werte für alle 1024 Dimensionen
- Für jede Dimension suchen wir im Dokument-Signal nach Positionen wo der Wert passt
- Kombiniere die Matches über alle Dimensionen

Das löst das Kontext-Problem: Konzepte müssen nicht im selben Fenster sein.
"""

import numpy as np
import requests
from dataclasses import dataclass
from typing import Optional
import time


@dataclass
class DocumentSignal:
    """Ein Dokument als kontinuierliches Signal über alle Dimensionen."""
    doc_id: str
    text: str
    # Shape: (n_positions, n_dims) - Embedding pro Position
    signal: np.ndarray
    # Positionen in Zeichen
    positions: list[int]

    @property
    def n_positions(self) -> int:
        return self.signal.shape[0]

    @property
    def n_dims(self) -> int:
        return self.signal.shape[1]


class LlamaCppEmbedder:
    """Embedder via llama.cpp API."""

    def __init__(self, base_url: str = "http://localhost:8200"):
        self.base_url = base_url
        self.dim = None
        self._init_dim()

    def _init_dim(self):
        test = self.embed(["test"])
        self.dim = len(test[0])

    def embed(self, texts: list[str], batch_size: int = 8) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            # Truncate long texts
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


def create_document_signal(
    embedder: LlamaCppEmbedder,
    text: str,
    doc_id: str,
    window_size: int = 150,
    stride: int = 30
) -> DocumentSignal:
    """Erstelle Signal aus Dokument mit Sliding Window."""
    windows = []
    positions = []

    for i in range(0, max(1, len(text) - window_size + 1), stride):
        window_text = text[i:i+window_size]
        windows.append(window_text)
        positions.append(i)

    if not windows:
        windows = [text]
        positions = [0]

    embeddings = embedder.embed(windows)

    return DocumentSignal(
        doc_id=doc_id,
        text=text,
        signal=embeddings,
        positions=positions
    )


class DimensionMatcher:
    """Verschiedene Strategien für dimensions-basiertes Matching."""

    @staticmethod
    def max_per_dim(query: np.ndarray, signal: np.ndarray) -> float:
        """
        Für jede Dimension: Finde besten Match über alle Positionen.
        Dann: Mittle über alle Dimensionen.

        score = mean_d(max_t(similarity(query[d], signal[t,d])))
        """
        # query: (n_dims,)
        # signal: (n_positions, n_dims)

        # Similarity pro Position pro Dimension
        # Wir nutzen 1 - |query[d] - signal[t,d]| / (|query[d]| + |signal[t,d]|)
        # als normalisierte Ähnlichkeit

        query_expanded = query[np.newaxis, :]  # (1, n_dims)

        # Absolute Differenz
        diff = np.abs(signal - query_expanded)  # (n_positions, n_dims)

        # Normalisierung
        norm = np.abs(signal) + np.abs(query_expanded) + 1e-10
        similarity = 1 - diff / norm  # (n_positions, n_dims)

        # Max pro Dimension
        max_sim = np.max(similarity, axis=0)  # (n_dims,)

        return float(np.mean(max_sim))

    @staticmethod
    def max_per_dim_weighted(query: np.ndarray, signal: np.ndarray) -> float:
        """
        Wie max_per_dim, aber gewichtet nach Query-Stärke.
        Stärkere Query-Dimensionen zählen mehr.
        """
        query_expanded = query[np.newaxis, :]

        diff = np.abs(signal - query_expanded)
        norm = np.abs(signal) + np.abs(query_expanded) + 1e-10
        similarity = 1 - diff / norm

        max_sim = np.max(similarity, axis=0)

        # Gewichtung nach |query|
        weights = np.abs(query)
        weights = weights / (np.sum(weights) + 1e-10)

        return float(np.sum(max_sim * weights))

    @staticmethod
    def coverage_threshold(
        query: np.ndarray,
        signal: np.ndarray,
        threshold: float = 0.1
    ) -> float:
        """
        Wie viel % der Query-Dimensionen finden einen Match
        innerhalb des Thresholds?

        Match = |query[d] - signal[t,d]| < threshold * |query[d]|
        """
        query_expanded = query[np.newaxis, :]

        # Relative Differenz
        diff = np.abs(signal - query_expanded)
        query_abs = np.abs(query_expanded) + 1e-10
        relative_diff = diff / query_abs

        # Hat jede Dimension irgendwo einen Match?
        has_match = np.any(relative_diff < threshold, axis=0)  # (n_dims,)

        return float(np.mean(has_match))

    @staticmethod
    def sign_coverage(query: np.ndarray, signal: np.ndarray) -> float:
        """
        Wie viel % der Query-Dimensionen haben irgendwo das richtige Vorzeichen?
        (Entspricht unserem Sign Hamming auf Dimensions-Ebene)
        """
        query_sign = np.sign(query)  # (n_dims,)
        signal_sign = np.sign(signal)  # (n_positions, n_dims)

        # Hat jede Dimension irgendwo das richtige Vorzeichen?
        sign_match = signal_sign == query_sign[np.newaxis, :]
        has_match = np.any(sign_match, axis=0)

        return float(np.mean(has_match))

    @staticmethod
    def best_window_cosine(query: np.ndarray, signal: np.ndarray) -> float:
        """Baseline: Bestes Fenster nach Cosine Similarity."""
        # Normalize
        query_norm = query / (np.linalg.norm(query) + 1e-10)
        signal_norm = signal / (np.linalg.norm(signal, axis=1, keepdims=True) + 1e-10)

        similarities = signal_norm @ query_norm
        return float(np.max(similarities))

    @staticmethod
    def combined_score(
        query: np.ndarray,
        signal: np.ndarray,
        threshold: float = 0.2
    ) -> float:
        """
        Kombination aus max_per_dim und coverage.
        """
        max_score = DimensionMatcher.max_per_dim_weighted(query, signal)
        coverage = DimensionMatcher.coverage_threshold(query, signal, threshold)

        # Geometrisches Mittel
        return float(np.sqrt(max_score * coverage))

    @staticmethod
    def cluster_voting(
        query: np.ndarray,
        signal: np.ndarray,
        threshold: float = 0.2
    ) -> float:
        """
        Cluster-basiert: Für jede Position, zähle wie viele Dimensionen matchen.
        Score = beste Position (höchste Überlappung).

        1. Für jede Dim: Berechne Match-Stärke an jeder Position
        2. Für jede Position: Summiere über alle Dims
        3. Beste Position gewinnt
        """
        # query: (n_dims,)
        # signal: (n_positions, n_dims)

        query_expanded = query[np.newaxis, :]  # (1, n_dims)

        # Relative Differenz pro Position pro Dimension
        diff = np.abs(signal - query_expanded)
        query_abs = np.abs(query_expanded) + 1e-10
        relative_diff = diff / query_abs  # (n_positions, n_dims)

        # Match-Stärke: 1 wenn perfekt, 0 wenn diff >= threshold
        match_strength = np.maximum(0, 1 - relative_diff / threshold)

        # Summiere über Dimensionen für jede Position
        position_scores = np.sum(match_strength, axis=1)  # (n_positions,)

        # Beste Position, normalisiert
        return float(np.max(position_scores) / signal.shape[1])

    @staticmethod
    def cluster_voting_weighted(
        query: np.ndarray,
        signal: np.ndarray,
        threshold: float = 0.2
    ) -> float:
        """
        Wie cluster_voting, aber gewichtet nach Query-Stärke.
        Wichtigere Dimensionen (höherer |query[d]|) zählen mehr.
        """
        query_expanded = query[np.newaxis, :]

        diff = np.abs(signal - query_expanded)
        query_abs = np.abs(query_expanded) + 1e-10
        relative_diff = diff / query_abs

        match_strength = np.maximum(0, 1 - relative_diff / threshold)

        # Gewichtung nach |query|
        weights = np.abs(query)
        weights = weights / (np.sum(weights) + 1e-10)

        # Gewichtete Summe pro Position
        position_scores = np.sum(match_strength * weights, axis=1)

        return float(np.max(position_scores))

    @staticmethod
    def cluster_sign_voting(
        query: np.ndarray,
        signal: np.ndarray
    ) -> float:
        """
        Sign-basiertes Cluster-Voting.
        Für jede Position: Zähle Dimensionen mit richtigem Vorzeichen.
        """
        query_sign = np.sign(query)  # (n_dims,)
        signal_sign = np.sign(signal)  # (n_positions, n_dims)

        # Match = 1 wenn gleiches Vorzeichen
        sign_match = (signal_sign == query_sign).astype(float)

        # Summe pro Position
        position_scores = np.sum(sign_match, axis=1)

        return float(np.max(position_scores) / signal.shape[1])

    @staticmethod
    def cluster_sign_voting_weighted(
        query: np.ndarray,
        signal: np.ndarray
    ) -> float:
        """
        Sign-basiertes Cluster-Voting mit Gewichtung.
        """
        query_sign = np.sign(query)
        signal_sign = np.sign(signal)

        sign_match = (signal_sign == query_sign).astype(float)

        # Gewichtung nach |query|
        weights = np.abs(query)
        weights = weights / (np.sum(weights) + 1e-10)

        position_scores = np.sum(sign_match * weights, axis=1)

        return float(np.max(position_scores))


def load_scifact(n_docs: int = 100, n_queries: int = 50):
    """Lade SciFact Testdaten."""
    from datasets import load_dataset
    from collections import defaultdict

    # Corpus und Queries separat laden
    corpus = load_dataset("mteb/scifact", "corpus", split="corpus")
    queries_ds = load_dataset("mteb/scifact", "queries", split="queries")
    qrels_ds = load_dataset("mteb/scifact", "default", split="test")

    # QRels aufbauen: query-id -> set(corpus-id)
    qrels = defaultdict(set)
    for item in qrels_ds:
        qrels[item["query-id"]].add(item["corpus-id"])

    # Dokumente
    docs = []
    doc_id_map = {}  # corpus-id -> our doc_id
    for i, item in enumerate(corpus):
        if i >= n_docs:
            break
        doc_id = item["_id"]
        docs.append({
            "id": doc_id,
            "text": f"{item['title']} {item['text']}"
        })
        doc_id_map[doc_id] = doc_id

    # Nur Queries die relevante Docs in unserem Subset haben
    queries = []
    relevance = {}
    doc_ids_set = set(doc_id_map.keys())

    for item in queries_ds:
        query_id = item["_id"]
        if query_id in qrels:
            relevant_in_subset = qrels[query_id] & doc_ids_set
            if relevant_in_subset:
                queries.append({
                    "id": query_id,
                    "text": item["text"]
                })
                relevance[query_id] = list(relevant_in_subset)

        if len(queries) >= n_queries:
            break

    return docs, queries, relevance


def evaluate_retrieval(
    method_name: str,
    score_fn,
    query_embeddings: np.ndarray,
    signals: list[DocumentSignal],
    queries: list[dict],
    relevance: dict,
    k: int = 10
) -> float:
    """Berechne Recall@K für eine Scoring-Methode."""

    hits = 0
    total = 0

    for i, query in enumerate(queries):
        query_id = query["id"]
        if query_id not in relevance:
            continue

        relevant_docs = set(relevance[query_id])
        query_emb = query_embeddings[i]

        # Score für jedes Dokument
        doc_scores = []
        for signal in signals:
            score = score_fn(query_emb, signal.signal)
            doc_scores.append((signal.doc_id, score))

        # Top-K
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        top_k = set(doc_id for doc_id, _ in doc_scores[:k])

        # Hit?
        if relevant_docs & top_k:
            hits += 1
        total += 1

    return hits / total if total > 0 else 0.0


def main():
    print("=" * 70)
    print("DIMENSIONS-BASIERTES MATCHING TEST")
    print("=" * 70)

    # Setup
    print("\n1. Verbinde mit Embedder...")
    embedder = LlamaCppEmbedder()
    print(f"   Embedder verbunden, dim={embedder.dim}")

    # Daten laden
    print("\n2. Lade SciFact...")
    n_docs = 500
    n_queries = 100
    docs, queries, relevance = load_scifact(n_docs, n_queries)
    print(f"   {len(docs)} Dokumente, {len(queries)} Queries")

    # Signale erstellen
    print("\n3. Erstelle Dokument-Signale...")
    signals = []
    for i, doc in enumerate(docs):
        signal = create_document_signal(
            embedder, doc["text"], doc["id"],
            window_size=150, stride=30
        )
        signals.append(signal)
        if (i + 1) % 20 == 0:
            print(f"   {i+1}/{len(docs)} Dokumente verarbeitet")

    # Query Embeddings
    print("\n4. Erstelle Query Embeddings...")
    query_texts = [q["text"] for q in queries]
    query_embeddings = embedder.embed(query_texts)
    print(f"   {len(query_embeddings)} Queries embedded")

    # Methoden testen
    print("\n5. Teste verschiedene Matching-Methoden...")
    print("=" * 70)

    methods = [
        ("Best Window (Cosine)", DimensionMatcher.best_window_cosine),
        ("Cluster Voting (t=0.2)", lambda q, s: DimensionMatcher.cluster_voting(q, s, 0.2)),
        ("Cluster Voting (t=0.5)", lambda q, s: DimensionMatcher.cluster_voting(q, s, 0.5)),
        ("Cluster Weighted (t=0.2)", lambda q, s: DimensionMatcher.cluster_voting_weighted(q, s, 0.2)),
        ("Cluster Weighted (t=0.5)", lambda q, s: DimensionMatcher.cluster_voting_weighted(q, s, 0.5)),
        ("Cluster Sign", DimensionMatcher.cluster_sign_voting),
        ("Cluster Sign Weighted", DimensionMatcher.cluster_sign_voting_weighted),
    ]

    results = []
    for name, score_fn in methods:
        start = time.time()
        recall = evaluate_retrieval(
            name, score_fn,
            query_embeddings, signals,
            queries, relevance,
            k=10
        )
        elapsed = time.time() - start
        results.append((name, recall, elapsed))
        print(f"   {name:30s}: R@10 = {recall*100:.1f}% ({elapsed:.2f}s)")

    # Vergleich
    print("\n" + "=" * 70)
    print("ERGEBNISSE")
    print("=" * 70)

    baseline = results[0][1]  # Best Window Cosine

    print(f"\n{'Methode':<35} {'R@10':>8} {'vs Baseline':>12}")
    print("-" * 60)
    for name, recall, elapsed in results:
        vs_baseline = recall / baseline * 100 if baseline > 0 else 0
        print(f"{name:<35} {recall*100:>7.1f}% {vs_baseline:>11.1f}%")

    # Threshold-Sweep für Cluster Voting
    print("\n" + "=" * 70)
    print("THRESHOLD SWEEP (Cluster Voting Weighted)")
    print("=" * 70)

    thresholds = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    print(f"\n{'Threshold':>10} {'R@10':>8} {'vs Baseline':>12}")
    print("-" * 35)

    for t in thresholds:
        score_fn = lambda q, s, t=t: DimensionMatcher.cluster_voting_weighted(q, s, t)
        recall = evaluate_retrieval(
            f"Cluster (t={t})", score_fn,
            query_embeddings, signals,
            queries, relevance,
            k=10
        )
        vs_baseline = recall / baseline * 100 if baseline > 0 else 0
        print(f"{t:>10.2f} {recall*100:>7.1f}% {vs_baseline:>11.1f}%")


if __name__ == "__main__":
    main()
