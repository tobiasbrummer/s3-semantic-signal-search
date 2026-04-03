#!/usr/bin/env python3
"""
Position-Level Retrieval Test

Fragestellung: Können wir mit Sign-Only die exakte Position im Dokument finden?

Ansatz:
1. Dokument → Sliding Window → Signal (N Embeddings)
2. Query → 1 Embedding
3. Cross-Correlation: Score für jede Position
4. Peak = Position wo Query matcht

Author: Claude & Toby
Date: December 2024
"""

import numpy as np
import requests
from typing import List, Tuple, Dict
from dataclasses import dataclass


# =============================================================================
# EMBEDDER (aus curve_smoothness_test.py)
# =============================================================================

class Embedder:
    def __init__(self, base_url: str = "http://localhost:8200"):
        self.base_url = base_url.rstrip("/")
        resp = requests.post(
            f"{self.base_url}/v1/embeddings",
            json={"input": "test", "model": "jina"},
            timeout=10
        )
        data = resp.json()
        self._dim = len(data["data"][0]["embedding"])
        print(f"Embedder verbunden, dim={self._dim}")

    @property
    def dim(self):
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        resp = requests.post(
            f"{self.base_url}/v1/embeddings",
            json={"input": text[:4000], "model": "jina"},
            timeout=60
        )
        data = resp.json()
        emb = np.array(data["data"][0]["embedding"], dtype=np.float32)
        return emb / (np.linalg.norm(emb) + 1e-10)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        # Truncate all
        texts = [t[:4000] for t in texts]
        resp = requests.post(
            f"{self.base_url}/v1/embeddings",
            json={"input": texts, "model": "jina"},
            timeout=120
        )
        data = resp.json()
        embeddings = np.array([item["embedding"] for item in data["data"]], dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.maximum(norms, 1e-10)


# =============================================================================
# DOCUMENT SIGNAL
# =============================================================================

@dataclass
class DocumentSignal:
    """Ein Dokument als kontinuierliches Signal."""
    text: str
    embeddings: np.ndarray      # (N, dim)
    positions: List[int]        # Startposition jedes Fensters
    window_size: int
    stride: int

    @property
    def n_windows(self) -> int:
        return len(self.positions)

    def get_text_at(self, window_idx: int) -> str:
        """Hole Text an einer bestimmten Fenster-Position."""
        start = self.positions[window_idx]
        end = min(start + self.window_size, len(self.text))
        return self.text[start:end]

    def get_context(self, window_idx: int, context_chars: int = 100) -> str:
        """Hole Text mit Kontext um eine Position."""
        start = max(0, self.positions[window_idx] - context_chars)
        end = min(len(self.text), self.positions[window_idx] + self.window_size + context_chars)

        text = self.text[start:end]
        # Markiere das Fenster
        rel_start = self.positions[window_idx] - start
        rel_end = rel_start + self.window_size

        before = text[:rel_start]
        match = text[rel_start:rel_end]
        after = text[rel_end:]

        return f"...{before}>>>{match}<<<{after}..."


def create_document_signal(
    text: str,
    embedder: Embedder,
    window_size: int = 150,
    stride: int = 30,
    batch_size: int = 8,
) -> DocumentSignal:
    """
    Erstelle Signal aus Dokument.
    """
    windows = []
    positions = []

    pos = 0
    while pos + window_size <= len(text):
        windows.append(text[pos:pos + window_size])
        positions.append(pos)
        pos += stride

    # Falls Text zu kurz für ein Fenster
    if not windows and text:
        windows.append(text)
        positions.append(0)

    # Batch-Embedding
    all_embeddings = []
    for i in range(0, len(windows), batch_size):
        batch = windows[i:i + batch_size]
        embs = embedder.embed_batch(batch)
        all_embeddings.append(embs)

    embeddings = np.vstack(all_embeddings)

    return DocumentSignal(
        text=text,
        embeddings=embeddings,
        positions=positions,
        window_size=window_size,
        stride=stride
    )


# =============================================================================
# SCORING METHODS (für Position-Level)
# =============================================================================

class PositionScoring:
    """Scoring-Methoden für Position-Finding."""

    @staticmethod
    def cosine(query: np.ndarray, signal: DocumentSignal) -> np.ndarray:
        """Cosine Similarity an jeder Position."""
        return signal.embeddings @ query

    @staticmethod
    def sign_hamming(query: np.ndarray, signal: DocumentSignal) -> np.ndarray:
        """Sign-Based Matching an jeder Position."""
        query_signs = np.sign(query)
        doc_signs = np.sign(signal.embeddings)
        return np.mean(query_signs * doc_signs, axis=1)

    @staticmethod
    def weighted_min(query: np.ndarray, signal: DocumentSignal) -> np.ndarray:
        """Weighted Min an jeder Position."""
        query_signs = np.sign(query)
        doc_signs = np.sign(signal.embeddings)

        query_abs = np.abs(query)
        doc_abs = np.abs(signal.embeddings)

        weights = np.minimum(query_abs, doc_abs)
        sign_products = doc_signs * query_signs
        weighted_products = sign_products * weights

        scores = np.sum(weighted_products, axis=1) / (np.sum(weights, axis=1) + 1e-10)
        return scores


# =============================================================================
# POSITION FINDING
# =============================================================================

@dataclass
class PositionMatch:
    """Ein gefundener Match mit Position."""
    window_idx: int
    char_position: int
    score: float
    text_snippet: str


def find_positions(
    query_embedding: np.ndarray,
    signal: DocumentSignal,
    method: str = "weighted_min",
    top_k: int = 3,
) -> List[PositionMatch]:
    """
    Finde Top-K Positionen wo Query matcht.
    """
    # Score berechnen
    if method == "cosine":
        scores = PositionScoring.cosine(query_embedding, signal)
    elif method == "sign_hamming":
        scores = PositionScoring.sign_hamming(query_embedding, signal)
    elif method == "weighted_min":
        scores = PositionScoring.weighted_min(query_embedding, signal)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Top-K finden
    top_indices = np.argsort(scores)[::-1][:top_k]

    matches = []
    for idx in top_indices:
        matches.append(PositionMatch(
            window_idx=idx,
            char_position=signal.positions[idx],
            score=float(scores[idx]),
            text_snippet=signal.get_text_at(idx)[:80] + "..."
        ))

    return matches


def visualize_scores(scores: np.ndarray, signal: DocumentSignal, title: str = ""):
    """ASCII-Visualisierung der Scores über das Dokument."""
    print(f"\n{title}")
    print("-" * 70)

    # Normalisiere auf 0-1
    s_min, s_max = scores.min(), scores.max()
    if s_max - s_min > 1e-10:
        normalized = (scores - s_min) / (s_max - s_min)
    else:
        normalized = np.ones_like(scores) * 0.5

    # ASCII-Balken (60 Zeichen breit)
    width = min(60, len(scores))
    if len(scores) > width:
        # Resample
        indices = np.linspace(0, len(scores) - 1, width).astype(int)
        normalized = normalized[indices]

    # Plot
    height = 8
    for row in range(height, -1, -1):
        threshold = row / height
        line = ""
        for val in normalized:
            if val >= threshold:
                line += "█"
            else:
                line += " "
        print(f"  {line}|")
    print("  " + "-" * len(normalized) + "+")
    print(f"  0{' ' * (len(normalized) - 6)}Position")


# =============================================================================
# TEST CASES
# =============================================================================

def run_position_test():
    """Haupttest für Position-Finding."""

    print("=" * 70)
    print("POSITION-LEVEL RETRIEVAL TEST")
    print("=" * 70)

    # Embedder
    print("\n1. Verbinde mit Embedder...")
    try:
        embedder = Embedder()
    except Exception as e:
        print(f"   Fehler: {e}")
        return

    # Test-Dokument mit klaren Abschnitten
    print("\n2. Erstelle Test-Dokument...")

    test_doc = """
    Introduction to Machine Learning

    Machine learning is a branch of artificial intelligence that focuses on building
    systems that can learn from data. These systems improve their performance over time
    without being explicitly programmed for each specific task.

    The field has grown significantly in recent decades, driven by increases in
    computational power and the availability of large datasets. Today, machine learning
    powers many applications we use daily, from email spam filters to voice assistants.

    Deep Learning and Neural Networks

    Deep learning is a subset of machine learning that uses neural networks with many
    layers. These deep neural networks can automatically discover the representations
    needed for feature detection or classification from raw data.

    Convolutional neural networks (CNNs) are particularly effective for image processing
    tasks. They use filters to detect features like edges, textures, and patterns.
    Recurrent neural networks (RNNs) are designed for sequential data like text and speech.

    Natural Language Processing

    Natural language processing (NLP) enables computers to understand, interpret, and
    generate human language. Modern NLP relies heavily on transformer architectures,
    which use attention mechanisms to process text.

    Large language models like GPT and BERT have revolutionized NLP. These models are
    trained on vast amounts of text data and can perform a wide range of language tasks,
    from translation to question answering.

    Applications in Healthcare

    Machine learning is transforming healthcare through improved diagnostics, drug
    discovery, and personalized medicine. AI systems can analyze medical images to
    detect diseases like cancer with high accuracy.

    Electronic health records provide rich data for predictive models. These models
    can identify patients at risk for various conditions and suggest preventive measures.
    The integration of AI in clinical workflows is still evolving but shows great promise.
    """

    print(f"   Dokument: {len(test_doc)} Zeichen")

    # Signal erstellen
    print("\n3. Erstelle Dokument-Signal...")
    signal = create_document_signal(
        test_doc,
        embedder,
        window_size=150,
        stride=30
    )
    print(f"   {signal.n_windows} Fenster (window=150, stride=30)")

    # Test-Queries mit erwarteten Positionen
    test_queries = [
        ("deep learning neural networks", "Deep Learning"),
        ("image processing CNN", "Convolutional neural networks"),
        ("healthcare medical diagnosis", "healthcare"),
        ("NLP transformers language models", "Natural language processing"),
        ("spam filter voice assistant", "email spam filters"),
    ]

    print("\n4. Teste Position-Finding...")
    print("=" * 70)

    methods = ["cosine", "sign_hamming", "weighted_min"]

    for query_text, expected_section in test_queries:
        print(f"\n{'─' * 70}")
        print(f"Query: \"{query_text}\"")
        print(f"Erwartet in Abschnitt: \"{expected_section}\"")
        print(f"{'─' * 70}")

        # Query embedden
        query_emb = embedder.embed(query_text)

        for method in methods:
            matches = find_positions(query_emb, signal, method=method, top_k=1)

            if matches:
                m = matches[0]
                # Prüfe ob erwarteter Text in der Nähe ist
                context = signal.get_context(m.window_idx, context_chars=50)
                found_expected = expected_section.lower() in context.lower()

                status = "✓" if found_expected else "✗"
                print(f"\n  {method:15s}: score={m.score:.3f}, pos={m.char_position:4d} {status}")
                print(f"                   \"{m.text_snippet[:60]}...\"")

    # Visualisierung für eine Query
    print("\n" + "=" * 70)
    print("SCORE-VISUALISIERUNG")
    print("=" * 70)

    viz_query = "deep learning neural networks"
    viz_emb = embedder.embed(viz_query)

    for method in methods:
        if method == "cosine":
            scores = PositionScoring.cosine(viz_emb, signal)
        elif method == "sign_hamming":
            scores = PositionScoring.sign_hamming(viz_emb, signal)
        else:
            scores = PositionScoring.weighted_min(viz_emb, signal)

        visualize_scores(scores, signal, f"{method.upper()} Scores für \"{viz_query}\"")

        # Zeige Peaks
        top_3 = np.argsort(scores)[::-1][:3]
        print(f"\n  Top 3 Positionen:")
        for i, idx in enumerate(top_3):
            print(f"    {i+1}. pos={signal.positions[idx]:4d}, score={scores[idx]:.3f}")
            print(f"       \"{signal.get_text_at(idx)[:50]}...\"")

    # Vergleich der Methoden
    print("\n" + "=" * 70)
    print("METHODEN-VERGLEICH")
    print("=" * 70)

    print("\nKorrelation der Rankings zwischen Methoden:")

    all_scores = {}
    for method in methods:
        if method == "cosine":
            all_scores[method] = PositionScoring.cosine(viz_emb, signal)
        elif method == "sign_hamming":
            all_scores[method] = PositionScoring.sign_hamming(viz_emb, signal)
        else:
            all_scores[method] = PositionScoring.weighted_min(viz_emb, signal)

    # Rank-Korrelation
    from scipy.stats import spearmanr

    for i, m1 in enumerate(methods):
        for m2 in methods[i+1:]:
            corr, _ = spearmanr(all_scores[m1], all_scores[m2])
            print(f"  {m1} vs {m2}: ρ = {corr:.3f}")

    # Fazit
    print("\n" + "=" * 70)
    print("FAZIT")
    print("=" * 70)


def run_scifact_position_test(num_docs: int = 5):
    """Test mit echten SciFact-Dokumenten."""

    print("=" * 70)
    print(f"POSITION-LEVEL RETRIEVAL TEST (SciFact, {num_docs} Docs)")
    print("=" * 70)

    # Embedder
    print("\n1. Verbinde mit Embedder...")
    try:
        embedder = Embedder()
    except Exception as e:
        print(f"   Fehler: {e}")
        return

    # Lade Daten
    print("\n2. Lade SciFact...")
    try:
        from datasets import load_dataset
        corpus = load_dataset("mteb/scifact", "corpus", split="corpus")
        queries_ds = load_dataset("mteb/scifact", "queries", split="queries")
        qrels_data = load_dataset("mteb/scifact", "default", split="test")

        # QRels
        from collections import defaultdict
        qrels = defaultdict(set)
        for item in qrels_data:
            qrels[item["query-id"]].add(item["corpus-id"])

        # Wähle Dokumente die von Queries referenziert werden
        doc_ids_with_queries = set()
        for qid, dids in qrels.items():
            doc_ids_with_queries.update(dids)

        # Finde passende Docs
        docs = []
        doc_map = {}
        for doc in corpus:
            if doc["_id"] in doc_ids_with_queries:
                text = f"{doc['title']} {doc['text']}"
                if len(text) >= 800:
                    docs.append({"id": doc["_id"], "text": text, "title": doc["title"]})
                    doc_map[doc["_id"]] = len(docs) - 1
                if len(docs) >= num_docs:
                    break

        # Finde Queries die zu diesen Docs gehören
        queries = []
        for q in queries_ds:
            relevant_docs = qrels.get(q["_id"], set())
            matching = [d for d in relevant_docs if d in doc_map]
            if matching:
                queries.append({
                    "id": q["_id"],
                    "text": q["text"],
                    "relevant_doc_ids": matching
                })

        print(f"   {len(docs)} Dokumente, {len(queries)} Queries")

    except ImportError:
        print("   pip install datasets")
        return

    # Erstelle Signale
    print("\n3. Erstelle Signale...")
    signals = {}
    for i, doc in enumerate(docs):
        print(f"   Doc {i+1}/{len(docs)}: {doc['title'][:40]}...", end="", flush=True)
        sig = create_document_signal(doc["text"], embedder, window_size=150, stride=30)
        signals[doc["id"]] = sig
        print(f" ({sig.n_windows} Fenster)")

    # Teste Position-Finding
    print("\n4. Teste Position-Finding...")

    methods = ["cosine", "weighted_min", "sign_hamming"]
    results = {m: {"found": 0, "total": 0} for m in methods}

    for q in queries[:10]:  # Max 10 Queries
        query_emb = embedder.embed(q["text"])

        for doc_id in q["relevant_doc_ids"]:
            if doc_id not in signals:
                continue

            signal = signals[doc_id]

            for method in methods:
                matches = find_positions(query_emb, signal, method=method, top_k=1)
                results[method]["total"] += 1

                if matches:
                    # Check: Ist der Match-Score signifikant?
                    # (Hier nur zählen dass wir überhaupt einen Peak finden)
                    results[method]["found"] += 1

    print("\n5. Ergebnisse...")
    print("=" * 70)

    for method in methods:
        r = results[method]
        pct = r["found"] / r["total"] * 100 if r["total"] > 0 else 0
        print(f"  {method:15s}: {r['found']}/{r['total']} ({pct:.1f}%)")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--scifact":
        num = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        run_scifact_position_test(num)
    else:
        run_position_test()
