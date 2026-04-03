#!/usr/bin/env python3
"""
Retrieval Benchmark mit llama.cpp API

Testet verschiedene Scoring-Methoden:
- Brute Force (Cosine Similarity) - Baseline
- Phase Coherent (Sign-Matching)
- Weighted Min (Sidechain Gate)
- Sign-Based Hamming

Nutzt jina-embeddings-v3 via llama.cpp auf localhost:8200

Author: Claude & Toby
Date: December 2024
"""

import numpy as np
import requests
import time
from typing import List, Dict, Tuple, Set
from collections import defaultdict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import json

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


# =============================================================================
# LLAMA.CPP EMBEDDING CLIENT
# =============================================================================

class LlamaCppEmbedder:
    """
    Client für llama.cpp Embedding API.
    Unterstützt sowohl OpenAI-kompatibles als auch natives Format.
    """

    def __init__(self, base_url: str = "http://localhost:8200", batch_size: int = 8, max_chars: int = 4000):
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size
        self.max_chars = max_chars  # Truncate texts longer than this
        self._dim = None
        self._api_format = None

        # API-Format erkennen
        self._detect_api_format()

    def _detect_api_format(self):
        """Erkennt welches API-Format der Server verwendet."""
        # Versuche OpenAI-Format
        try:
            resp = requests.post(
                f"{self.base_url}/v1/embeddings",
                json={"input": "test", "model": "jina"},
                timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()
                if "data" in data and len(data["data"]) > 0:
                    self._api_format = "openai"
                    self._dim = len(data["data"][0]["embedding"])
                    print(f"   API: OpenAI-kompatibel, dim={self._dim}")
                    return
        except:
            pass

        # Versuche natives llama.cpp Format
        try:
            resp = requests.post(
                f"{self.base_url}/embedding",
                json={"content": "test"},
                timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()
                if "embedding" in data:
                    self._api_format = "native"
                    self._dim = len(data["embedding"])
                    print(f"   API: Native llama.cpp, dim={self._dim}")
                    return
        except:
            pass

        raise ConnectionError(f"Konnte kein Embedding-API unter {self.base_url} finden")

    @property
    def dimension(self) -> int:
        return self._dim

    def _embed_single(self, text: str) -> np.ndarray:
        """Einzelnes Embedding abrufen."""
        if self._api_format == "openai":
            resp = requests.post(
                f"{self.base_url}/v1/embeddings",
                json={"input": text, "model": "jina"}
            )
            return np.array(resp.json()["data"][0]["embedding"], dtype=np.float32)
        else:
            resp = requests.post(
                f"{self.base_url}/embedding",
                json={"content": text}
            )
            return np.array(resp.json()["embedding"], dtype=np.float32)

    def _truncate(self, text: str) -> str:
        """Truncate text to max_chars."""
        if len(text) > self.max_chars:
            return text[:self.max_chars]
        return text

    def _embed_batch_openai(self, texts: List[str]) -> np.ndarray:
        """Batch-Embedding mit OpenAI-Format."""
        # Truncate all texts
        texts = [self._truncate(t) for t in texts]

        resp = requests.post(
            f"{self.base_url}/v1/embeddings",
            json={"input": texts, "model": "jina"},
            timeout=120
        )

        if resp.status_code != 200:
            # Fallback: Try one by one if batch fails
            embeddings = []
            for text in texts:
                single_resp = requests.post(
                    f"{self.base_url}/v1/embeddings",
                    json={"input": text, "model": "jina"},
                    timeout=60
                )
                if single_resp.status_code == 200:
                    data = single_resp.json()
                    embeddings.append(data["data"][0]["embedding"])
                else:
                    # Last resort: return zeros
                    embeddings.append([0.0] * self._dim)
            return np.array(embeddings, dtype=np.float32)

        data = resp.json()
        embeddings = [item["embedding"] for item in data["data"]]
        return np.array(embeddings, dtype=np.float32)

    def encode(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Encodiere Liste von Texten.
        """
        all_embeddings = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1

            if show_progress:
                print(f"\r   Batch {batch_num}/{total_batches}", end="", flush=True)

            if self._api_format == "openai":
                embeddings = self._embed_batch_openai(batch)
            else:
                # Native Format unterstützt kein Batching
                embeddings = np.array([self._embed_single(t) for t in batch])

            all_embeddings.append(embeddings)

        if show_progress:
            print()

        result = np.vstack(all_embeddings)

        # Normalisieren
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        result = result / np.maximum(norms, 1e-10)

        return result


# =============================================================================
# SCORING METHODS
# =============================================================================

class ScoringMethods:
    """
    Verschiedene Scoring-Methoden für Retrieval.
    """

    @staticmethod
    def cosine_similarity(query: np.ndarray, docs: np.ndarray) -> np.ndarray:
        """
        Standard Cosine Similarity (Brute Force Baseline).

        Args:
            query: (dim,) normalized
            docs: (N, dim) normalized

        Returns:
            (N,) similarities
        """
        return docs @ query

    @staticmethod
    def phase_coherent(query: np.ndarray, docs: np.ndarray) -> np.ndarray:
        """
        Phase Coherent: Sign-Matching Score.

        Score = mean(sign(q) * sign(d))

        Nur die Vorzeichen-Übereinstimmung zählt.
        Aus wave2.org: 92.5% von Brute Force Recall.
        """
        query_signs = np.sign(query)  # (dim,)
        doc_signs = np.sign(docs)      # (N, dim)

        # Sign-Produkt: +1 wenn gleich, -1 wenn verschieden
        sign_products = doc_signs * query_signs  # (N, dim)

        # Mittelwert über alle Dimensionen
        return np.mean(sign_products, axis=1)  # (N,)

    @staticmethod
    def weighted_min(query: np.ndarray, docs: np.ndarray) -> np.ndarray:
        """
        Weighted Min: Sidechain-Gate Scoring.

        Score = sum(sign(q)*sign(d) * min(|q|,|d|)) / sum(min(|q|,|d|))

        "Nur wenn BEIDE Signale stark sind, zählt es."
        Aus wave2.org: 97-100% von Brute Force bei synthetischen Daten.
        """
        query_signs = np.sign(query)  # (dim,)
        doc_signs = np.sign(docs)      # (N, dim)

        query_abs = np.abs(query)      # (dim,)
        doc_abs = np.abs(docs)         # (N, dim)

        # Minimum der Amplituden (Sidechain-Gate)
        weights = np.minimum(query_abs, doc_abs)  # (N, dim)

        # Gewichtetes Sign-Produkt
        sign_products = doc_signs * query_signs  # (N, dim)
        weighted_products = sign_products * weights  # (N, dim)

        # Normalisierte Summe
        scores = np.sum(weighted_products, axis=1) / (np.sum(weights, axis=1) + 1e-10)

        return scores

    @staticmethod
    def sign_hamming(query: np.ndarray, docs: np.ndarray) -> np.ndarray:
        """
        Sign-Based Hamming: Binäre Sign-Matching.

        Gleich wie Phase Coherent, aber als Hamming-Distanz:
        Distance = count(sign(q) != sign(d))
        Score = 1 - (distance / dim)  # Normalisiert auf [0, 1]

        Schneller weil nur Bit-Operationen.
        """
        query_bits = (query > 0).astype(np.uint8)  # (dim,)
        doc_bits = (docs > 0).astype(np.uint8)     # (N, dim)

        # XOR gibt 1 wo verschieden
        xor = np.bitwise_xor(query_bits, doc_bits)  # (N, dim)

        # Hamming-Distanz = Anzahl verschiedener Bits
        distances = np.sum(xor, axis=1)  # (N,)

        # In Score umwandeln (höher = besser)
        dim = len(query)
        scores = 1.0 - (distances / dim)

        return scores

    @staticmethod
    def anti_resonance(query: np.ndarray, docs: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        Anti-Resonance: Bestrafe gegenphasige starke Signale.

        Wenn beide Signale stark sind aber verschiedene Vorzeichen haben,
        ist das ein "aktiver Widerspruch" - schlimmer als Stille.

        Score = weighted_min - alpha * anti_resonance_penalty
        """
        query_signs = np.sign(query)
        doc_signs = np.sign(docs)

        query_abs = np.abs(query)
        doc_abs = np.abs(docs)

        # Weighted min score (wie oben)
        weights = np.minimum(query_abs, doc_abs)
        sign_products = doc_signs * query_signs
        weighted_products = sign_products * weights
        base_scores = np.sum(weighted_products, axis=1) / (np.sum(weights, axis=1) + 1e-10)

        # Anti-Resonance: Wo beide stark sind UND gegenphasig
        both_strong = weights > np.percentile(weights, 75, axis=1, keepdims=True)
        opposite_phase = sign_products < 0
        anti_resonance_mask = both_strong & opposite_phase

        penalty = np.sum(weights * anti_resonance_mask, axis=1) / (np.sum(weights, axis=1) + 1e-10)

        return base_scores - alpha * penalty


# =============================================================================
# EVALUATION
# =============================================================================

@dataclass
class EvalResult:
    name: str
    recall_at_k: Dict[int, float]
    mrr: float
    avg_time_ms: float

    def __str__(self):
        recalls = ", ".join(f"R@{k}={v*100:.1f}%" for k, v in self.recall_at_k.items())
        return f"{self.name}: {recalls}, MRR={self.mrr*100:.1f}%, Time={self.avg_time_ms:.2f}ms"


def evaluate_method(
    name: str,
    score_fn,
    query_embeddings: np.ndarray,
    doc_embeddings: np.ndarray,
    doc_ids: List[str],
    qrels: Dict[str, Set[str]],
    query_ids: List[str],
    k_values: List[int] = [1, 5, 10, 20]
) -> EvalResult:
    """
    Evaluiere eine Scoring-Methode.
    """
    recalls = {k: [] for k in k_values}
    mrrs = []
    times = []

    for i, query_id in enumerate(query_ids):
        query_emb = query_embeddings[i]
        relevant = qrels.get(query_id, set())

        if not relevant:
            continue

        start = time.time()
        scores = score_fn(query_emb, doc_embeddings)
        times.append(time.time() - start)

        # Top-k
        max_k = max(k_values)
        top_indices = np.argsort(scores)[::-1][:max_k]
        retrieved = [doc_ids[idx] for idx in top_indices]

        # Recall@k
        for k in k_values:
            retrieved_k = set(retrieved[:k])
            recall = len(retrieved_k & relevant) / len(relevant)
            recalls[k].append(recall)

        # MRR
        for rank, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant:
                mrrs.append(1.0 / rank)
                break
        else:
            mrrs.append(0.0)

    return EvalResult(
        name=name,
        recall_at_k={k: np.mean(v) for k, v in recalls.items()},
        mrr=np.mean(mrrs),
        avg_time_ms=np.mean(times) * 1000
    )


# =============================================================================
# BENCHMARK
# =============================================================================

def run_benchmark(
    base_url: str = "http://localhost:8200",
    max_corpus: int = 5000,
    max_queries: int = 300,
):
    print("=" * 70)
    print("RETRIEVAL BENCHMARK (llama.cpp + jina-embeddings-v3)")
    print("=" * 70)

    # 1. Embedder initialisieren
    print(f"\n1. Verbinde mit {base_url}...")
    try:
        embedder = LlamaCppEmbedder(base_url)
    except ConnectionError as e:
        print(f"   Fehler: {e}")
        return

    # 2. Dataset laden
    print(f"\n2. Lade SciFact Dataset...")
    if not HAS_DATASETS:
        print("   pip install datasets")
        return

    corpus = load_dataset("mteb/scifact", "corpus", split="corpus")
    queries = load_dataset("mteb/scifact", "queries", split="queries")
    qrels_data = load_dataset("mteb/scifact", "default", split="test")

    # QRels aufbauen
    qrels = defaultdict(set)
    for item in qrels_data:
        qrels[item["query-id"]].add(item["corpus-id"])

    corpus_list = list(corpus)[:max_corpus]
    query_list = [q for q in queries if q["_id"] in qrels][:max_queries]

    print(f"   Corpus: {len(corpus_list)} Dokumente")
    print(f"   Queries: {len(query_list)}")

    # 3. Encodieren
    print(f"\n3. Encodiere Corpus...")
    corpus_texts = [f"{doc['title']} {doc['text']}" for doc in corpus_list]
    doc_ids = [doc["_id"] for doc in corpus_list]

    start = time.time()
    doc_embeddings = embedder.encode(corpus_texts)
    corpus_time = time.time() - start
    print(f"   Zeit: {corpus_time:.1f}s ({corpus_time/len(corpus_texts)*1000:.1f}ms/doc)")

    print(f"\n4. Encodiere Queries...")
    query_texts = [q["text"] for q in query_list]
    query_ids = [q["_id"] for q in query_list]

    start = time.time()
    query_embeddings = embedder.encode(query_texts)
    query_time = time.time() - start
    print(f"   Zeit: {query_time:.1f}s")

    # 4. Evaluieren
    print(f"\n5. Evaluiere Methoden...")

    methods = {
        "Brute Force (Cosine)": ScoringMethods.cosine_similarity,
        "Phase Coherent": ScoringMethods.phase_coherent,
        "Weighted Min": ScoringMethods.weighted_min,
        "Sign Hamming": ScoringMethods.sign_hamming,
        "Anti-Resonance (0.3)": lambda q, d: ScoringMethods.anti_resonance(q, d, alpha=0.3),
        "Anti-Resonance (0.5)": lambda q, d: ScoringMethods.anti_resonance(q, d, alpha=0.5),
    }

    results = []
    for name, score_fn in methods.items():
        print(f"   {name}...", end="", flush=True)
        result = evaluate_method(
            name=name,
            score_fn=score_fn,
            query_embeddings=query_embeddings,
            doc_embeddings=doc_embeddings,
            doc_ids=doc_ids,
            qrels=qrels,
            query_ids=query_ids
        )
        results.append(result)
        print(f" R@10={result.recall_at_k[10]*100:.1f}%")

    # 5. Ergebnisse
    print(f"\n" + "=" * 70)
    print("ERGEBNISSE")
    print("=" * 70)

    baseline = results[0]  # Brute Force

    print(f"\n{'Methode':<25} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'R@20':>8} {'MRR':>8} {'vs BF':>8} {'Zeit':>10}")
    print("-" * 95)

    for r in results:
        vs_bf = r.recall_at_k[10] / baseline.recall_at_k[10] * 100 if baseline.recall_at_k[10] > 0 else 0
        print(f"{r.name:<25} "
              f"{r.recall_at_k[1]*100:>7.1f}% "
              f"{r.recall_at_k[5]*100:>7.1f}% "
              f"{r.recall_at_k[10]*100:>7.1f}% "
              f"{r.recall_at_k[20]*100:>7.1f}% "
              f"{r.mrr*100:>7.1f}% "
              f"{vs_bf:>7.1f}% "
              f"{r.avg_time_ms:>8.2f}ms")

    # 6. Analyse
    print(f"\n" + "=" * 70)
    print("ANALYSE")
    print("=" * 70)

    best_method = max(results, key=lambda r: r.recall_at_k[10])
    print(f"\nBeste Methode: {best_method.name}")
    print(f"  Recall@10: {best_method.recall_at_k[10]*100:.1f}%")
    print(f"  vs Brute Force: {best_method.recall_at_k[10]/baseline.recall_at_k[10]*100:.1f}%")

    # Speicher-Analyse
    dim = embedder.dimension
    float_bytes = dim * 4 * len(corpus_list)
    sign_bytes = (dim // 8) * len(corpus_list)

    print(f"\nSpeicher:")
    print(f"  Float32: {float_bytes/1e6:.1f} MB")
    print(f"  Sign-Based: {sign_bytes/1e6:.1f} MB ({float_bytes/sign_bytes:.0f}x Kompression)")

    return results


# =============================================================================
# QUICK TEST
# =============================================================================

def quick_test(base_url: str = "http://localhost:8200"):
    """Schneller Test mit synthetischen Daten."""
    print("=" * 70)
    print("QUICK TEST")
    print("=" * 70)

    print(f"\n1. Verbinde mit {base_url}...")
    try:
        embedder = LlamaCppEmbedder(base_url)
    except ConnectionError as e:
        print(f"   Fehler: {e}")
        return

    print(f"\n2. Generiere Test-Embeddings...")
    test_texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with many layers.",
        "Python is a popular programming language.",
        "The weather today is sunny and warm.",
        "Neural networks can learn complex patterns.",
    ]

    embeddings = embedder.encode(test_texts)
    print(f"   Shape: {embeddings.shape}")

    print(f"\n3. Teste Scoring-Methoden...")
    query = embeddings[0]  # "Machine learning..."
    docs = embeddings[1:]

    methods = {
        "Cosine": ScoringMethods.cosine_similarity,
        "Phase Coherent": ScoringMethods.phase_coherent,
        "Weighted Min": ScoringMethods.weighted_min,
        "Sign Hamming": ScoringMethods.sign_hamming,
    }

    print(f"\n   Query: '{test_texts[0][:50]}...'")
    print(f"\n   {'Dokument':<45} " + " ".join(f"{name:>15}" for name in methods.keys()))
    print("-" * (45 + 16 * len(methods)))

    for i, text in enumerate(test_texts[1:]):
        scores = {name: fn(query, docs)[i] for name, fn in methods.items()}
        print(f"   {text[:43]:<45} " + " ".join(f"{s:>15.3f}" for s in scores.values()))

    print("\n" + "=" * 70)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Retrieval Benchmark")
    parser.add_argument("--url", default="http://localhost:8200", help="llama.cpp API URL")
    parser.add_argument("--corpus", type=int, default=5000, help="Max corpus size")
    parser.add_argument("--queries", type=int, default=300, help="Max queries")
    parser.add_argument("--quick", action="store_true", help="Quick test only")

    args = parser.parse_args()

    if args.quick:
        quick_test(args.url)
    else:
        run_benchmark(args.url, args.corpus, args.queries)
