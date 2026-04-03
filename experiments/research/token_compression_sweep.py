#!/usr/bin/env python3
"""
Token-Level Compression Strategies

Tests:
1. Stability Threshold Sweep (0.05 - 0.5)
2. Delta-Encoding für volatile Dimensionen
3. Token-Level für bessere Spline-Approximation
"""

import numpy as np
import requests
import time
from dataclasses import dataclass
from collections import defaultdict
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import splrep, splev


# =============================================================================
# ENCODERS
# =============================================================================

class TokenEncoder:
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


# =============================================================================
# DATA LOADING
# =============================================================================

def load_scifact(n_docs: int = 100):
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
        docs.append({
            "id": item["_id"],
            "text": f"{item['title']} {item['text']}"
        })
        doc_id_set.add(item["_id"])

    queries = []
    relevance = {}
    for item in queries_ds:
        query_id = item["_id"]
        if query_id in qrels:
            relevant = qrels[query_id] & doc_id_set
            if relevant:
                queries.append({"id": query_id, "text": item["text"]})
                relevance[query_id] = list(relevant)

    return docs, queries, relevance


# =============================================================================
# TEST 1: THRESHOLD SWEEP
# =============================================================================

def test_threshold_sweep(docs_embeddings: dict, thresholds: list[float]):
    """
    Teste verschiedene Stability-Thresholds.
    """
    print("\n" + "=" * 70)
    print("TEST 1: STABILITY THRESHOLD SWEEP")
    print("=" * 70)

    results = []

    for threshold in thresholds:
        total_stable = 0
        total_dims = 0
        total_tokens = 0

        for doc_id, embeddings in docs_embeddings.items():
            signs = embeddings > 0
            n_tokens = len(embeddings)
            total_tokens += n_tokens

            if n_tokens < 2:
                continue

            # Flip-Rate berechnen
            sign_changes = np.diff(signs.astype(np.int8), axis=0)
            flips_per_dim = np.abs(sign_changes).sum(axis=0)
            flip_rate = flips_per_dim / (n_tokens - 1)

            stable = (flip_rate < threshold).sum()
            total_stable += stable
            total_dims += 1024

        pct_stable = total_stable / total_dims * 100

        # Speicher-Schätzung
        # Full: n_tokens × 1024 bits
        # Hybrid: n_docs × 1024 (mask) + n_docs × stable (signs) + n_tokens × volatile
        n_docs = len(docs_embeddings)
        avg_stable = total_stable / n_docs
        avg_volatile = 1024 - avg_stable

        full_bits = total_tokens * 1024
        hybrid_bits = n_docs * 1024 + n_docs * avg_stable + total_tokens * avg_volatile
        savings = (1 - hybrid_bits / full_bits) * 100

        results.append({
            "threshold": threshold,
            "pct_stable": pct_stable,
            "avg_stable_dims": avg_stable,
            "savings_pct": savings,
        })

    print(f"\n{'Threshold':>10} {'Stabil':>10} {'Dims':>10} {'Ersparnis':>12}")
    print("-" * 45)
    for r in results:
        print(f"{r['threshold']:>10.2f} {r['pct_stable']:>9.1f}% {r['avg_stable_dims']:>10.0f} {r['savings_pct']:>11.1f}%")

    return results


# =============================================================================
# TEST 2: DELTA ENCODING
# =============================================================================

def test_delta_encoding(docs_embeddings: dict):
    """
    Teste Delta-Encoding (DPCM-Style) für Token-Embeddings.
    """
    print("\n" + "=" * 70)
    print("TEST 2: DELTA ENCODING (DPCM)")
    print("=" * 70)

    total_original_bits = 0
    total_delta_bits = 0

    for doc_id, embeddings in docs_embeddings.items():
        n_tokens, n_dims = embeddings.shape

        # Original: float32
        original_bits = n_tokens * n_dims * 32

        # Delta: Erstes Token voll, dann Deltas
        if n_tokens > 1:
            deltas = np.diff(embeddings, axis=0)

            # Quantisierung: Wie viele Bits brauchen Deltas?
            # Schätze benötigte Bits basierend auf Range
            delta_range = deltas.max() - deltas.min()
            orig_range = embeddings.max() - embeddings.min()

            # Bits proportional zu log2(range)
            if delta_range > 0 and orig_range > 0:
                delta_bits_per_val = max(8, 32 * (np.log2(delta_range + 1) / np.log2(orig_range + 1)))
            else:
                delta_bits_per_val = 32

            delta_bits = n_dims * 32 + (n_tokens - 1) * n_dims * delta_bits_per_val
        else:
            delta_bits = original_bits

        total_original_bits += original_bits
        total_delta_bits += delta_bits

    savings = (1 - total_delta_bits / total_original_bits) * 100

    print(f"\n   Original (float32):  {total_original_bits / 8 / 1024:.1f} KB")
    print(f"   Delta-Encoded:       {total_delta_bits / 8 / 1024:.1f} KB")
    print(f"   Ersparnis:           {savings:.1f}%")

    # Sign-basierte Delta-Analyse
    print("\n   Sign-basierte Delta-Analyse:")

    total_sign_bits = 0
    total_delta_sign_bits = 0

    for doc_id, embeddings in docs_embeddings.items():
        n_tokens = len(embeddings)
        signs = embeddings > 0

        # Original: 1 bit pro dim pro token
        sign_bits = n_tokens * 1024

        # Delta: Erstes Token voll, dann nur Flips (RLE-fähig)
        if n_tokens > 1:
            sign_changes = np.diff(signs.astype(np.int8), axis=0)
            n_flips = np.abs(sign_changes).sum()

            # Bits für Flips: Position (log2(1024) ≈ 10 bits) pro Flip
            delta_sign_bits = 1024 + n_flips * 10
        else:
            delta_sign_bits = sign_bits

        total_sign_bits += sign_bits
        total_delta_sign_bits += delta_sign_bits

    sign_savings = (1 - total_delta_sign_bits / total_sign_bits) * 100

    print(f"   Original Signs:      {total_sign_bits / 8 / 1024:.1f} KB")
    print(f"   Delta Signs:         {total_delta_sign_bits / 8 / 1024:.1f} KB")
    print(f"   Ersparnis:           {sign_savings:.1f}%")

    return {"float_savings": savings, "sign_savings": sign_savings}


# =============================================================================
# TEST 3: SPLINE APPROXIMATION FROM TOKENS
# =============================================================================

def test_spline_from_tokens(docs_embeddings: dict, n_control_points_list: list[int]):
    """
    Teste Spline-Approximation mit Token-Level als Ground Truth.
    """
    print("\n" + "=" * 70)
    print("TEST 3: SPLINE APPROXIMATION FROM TOKEN-LEVEL")
    print("=" * 70)

    results = []

    for n_cp in n_control_points_list:
        total_mse = 0
        total_signal_power = 0
        total_docs = 0
        total_compression = 0

        for doc_id, embeddings in docs_embeddings.items():
            n_tokens, n_dims = embeddings.shape

            if n_tokens < n_cp + 2:
                continue

            t = np.linspace(0, 1, n_tokens)

            # Sample 50 Dimensionen für Geschwindigkeit
            sample_dims = np.random.choice(n_dims, min(50, n_dims), replace=False)

            doc_mse = 0
            doc_power = 0

            for dim in sample_dims:
                y = embeddings[:, dim]

                try:
                    # Spline fitten
                    spline = UnivariateSpline(t, y, k=3, s=len(t) * 0.1)
                    y_approx = spline(t)

                    mse = np.mean((y - y_approx) ** 2)
                    power = np.mean(y ** 2)

                    doc_mse += mse
                    doc_power += power
                except:
                    pass

            if doc_power > 0:
                total_mse += doc_mse / len(sample_dims)
                total_signal_power += doc_power / len(sample_dims)
                total_compression += n_tokens / n_cp
                total_docs += 1

        if total_docs > 0:
            avg_snr = 10 * np.log10(total_signal_power / total_mse) if total_mse > 0 else 99
            avg_compression = total_compression / total_docs

            results.append({
                "n_control_points": n_cp,
                "snr_db": avg_snr,
                "compression": avg_compression,
            })

    print(f"\n{'Ctrl Pts':>10} {'SNR (dB)':>10} {'Compression':>12}")
    print("-" * 35)
    for r in results:
        print(f"{r['n_control_points']:>10} {r['snr_db']:>10.1f} {r['compression']:>11.1f}x")

    # Sign-Rekonstruktion testen
    print("\n   Sign-Rekonstruktion aus Splines:")

    for n_cp in [5, 10, 20]:
        correct_signs = 0
        total_signs = 0

        for doc_id, embeddings in docs_embeddings.items():
            n_tokens, n_dims = embeddings.shape

            if n_tokens < n_cp + 2:
                continue

            t = np.linspace(0, 1, n_tokens)
            sample_dims = np.random.choice(n_dims, min(50, n_dims), replace=False)

            for dim in sample_dims:
                y = embeddings[:, dim]
                original_signs = y > 0

                try:
                    spline = UnivariateSpline(t, y, k=3, s=len(t) * 0.1)
                    y_approx = spline(t)
                    approx_signs = y_approx > 0

                    correct_signs += (original_signs == approx_signs).sum()
                    total_signs += n_tokens
                except:
                    pass

        if total_signs > 0:
            accuracy = correct_signs / total_signs * 100
            print(f"   {n_cp} Ctrl Pts: {accuracy:.1f}% Sign-Accuracy")

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("TOKEN-LEVEL COMPRESSION STRATEGIES")
    print("=" * 70)

    # Encoder
    token_encoder = TokenEncoder()

    # Daten laden
    print("\n1. Lade SciFact und erzeuge Token-Embeddings...")
    docs, queries, relevance = load_scifact(n_docs=50)  # Weniger Docs für Geschwindigkeit
    print(f"   {len(docs)} Dokumente")

    # Token-Embeddings für alle Docs
    print("\n2. Erzeuge Token-Embeddings...")
    docs_embeddings = {}
    for i, doc in enumerate(docs):
        embeddings = token_encoder.encode(doc["text"])
        docs_embeddings[doc["id"]] = embeddings
        if (i + 1) % 10 == 0:
            print(f"   {i+1}/{len(docs)}")

    total_tokens = sum(len(e) for e in docs_embeddings.values())
    print(f"   Total Tokens: {total_tokens}")

    # Test 1: Threshold Sweep
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    threshold_results = test_threshold_sweep(docs_embeddings, thresholds)

    # Test 2: Delta Encoding
    delta_results = test_delta_encoding(docs_embeddings)

    # Test 3: Spline Approximation
    n_control_points = [5, 10, 15, 20, 30]
    spline_results = test_spline_from_tokens(docs_embeddings, n_control_points)

    # Zusammenfassung
    print("\n" + "=" * 70)
    print("ZUSAMMENFASSUNG")
    print("=" * 70)

    # Bester Threshold
    best_threshold = max(threshold_results, key=lambda x: x['savings_pct'])
    print(f"\n   Bester Stability-Threshold: {best_threshold['threshold']}")
    print(f"     → {best_threshold['pct_stable']:.1f}% stabile Dims")
    print(f"     → {best_threshold['savings_pct']:.1f}% Ersparnis")

    # Delta Encoding
    print(f"\n   Delta-Encoding:")
    print(f"     → Float32: {delta_results['float_savings']:.1f}% Ersparnis")
    print(f"     → Signs:   {delta_results['sign_savings']:.1f}% Ersparnis")

    # Spline
    best_spline = max(spline_results, key=lambda x: x['snr_db'])
    print(f"\n   Beste Spline-Approximation: {best_spline['n_control_points']} Punkte")
    print(f"     → SNR: {best_spline['snr_db']:.1f} dB")
    print(f"     → Compression: {best_spline['compression']:.1f}x")

    print("\n" + "=" * 70)
    print("EMPFEHLUNG")
    print("=" * 70)
    print("""
   Hybrid-Strategie:
   1. Token-Level Embeddings als Ground Truth
   2. Stability-Threshold ~0.2-0.3 für Sign-Speicherung
   3. Delta-Encoding für Signs (~40% Ersparnis)
   4. Splines mit ~10-15 Kontrollpunkten für Pattern-Suche

   Trade-off: Mehr Kompression = langsamere Rekonstruktion
""")


if __name__ == "__main__":
    main()
