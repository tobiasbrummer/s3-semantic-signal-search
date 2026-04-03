#!/usr/bin/env python3
"""
Token Delta Analysis - Audio Engineering Perspektive

Fragen:
1. Wie stark unterscheiden sich aufeinanderfolgende Token-Embeddings?
2. Wie gut funktioniert Delta-Encoding (DPCM-Style)?
3. Können wir Token-Level für bessere Spline-Approximation nutzen?
"""

import numpy as np
import requests
from collections import defaultdict


class TokenEncoder:
    """Token-Level Encoder via Port 8202."""

    def __init__(self, url: str = "http://localhost:8202"):
        self.url = url

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


def analyze_deltas(embeddings: np.ndarray) -> dict:
    """Analysiere Deltas zwischen aufeinanderfolgenden Embeddings."""
    n_tokens = len(embeddings)
    if n_tokens < 2:
        return {}

    # Deltas berechnen
    deltas = np.diff(embeddings, axis=0)  # (n-1, dim)

    # Statistiken
    delta_norms = np.linalg.norm(deltas, axis=1)
    emb_norms = np.linalg.norm(embeddings, axis=1)

    # Cosine Similarity zwischen aufeinanderfolgenden Tokens
    cosines = []
    for i in range(n_tokens - 1):
        cos = np.dot(embeddings[i], embeddings[i+1]) / (emb_norms[i] * emb_norms[i+1])
        cosines.append(cos)
    cosines = np.array(cosines)

    # Bits für Delta vs Original
    # Original: float32 = 32 bits pro Wert
    # Delta: Wenn klein, weniger Bits nötig

    # Quantisierung: Wie viele Bits brauchen wir für Deltas?
    delta_range = deltas.max() - deltas.min()
    emb_range = embeddings.max() - embeddings.min()

    return {
        "n_tokens": n_tokens,
        "avg_delta_norm": float(delta_norms.mean()),
        "std_delta_norm": float(delta_norms.std()),
        "min_delta_norm": float(delta_norms.min()),
        "max_delta_norm": float(delta_norms.max()),
        "avg_emb_norm": float(emb_norms.mean()),
        "avg_cosine_adjacent": float(cosines.mean()),
        "min_cosine_adjacent": float(cosines.min()),
        "delta_range": float(delta_range),
        "emb_range": float(emb_range),
        "range_ratio": float(delta_range / emb_range) if emb_range > 0 else 0,
    }


def analyze_spline_fit(embeddings: np.ndarray, n_control_points: int = 10) -> dict:
    """
    Analysiere, wie gut Splines die Token-Embeddings approximieren.
    """
    from scipy.interpolate import UnivariateSpline

    n_tokens, n_dims = embeddings.shape
    if n_tokens < n_control_points:
        n_control_points = max(3, n_tokens // 2)

    t = np.linspace(0, 1, n_tokens)
    t_control = np.linspace(0, 1, n_control_points)

    # Fit Spline für jede Dimension
    reconstruction_errors = []

    for dim in range(min(100, n_dims)):  # Erste 100 Dims als Sample
        y = embeddings[:, dim]

        try:
            # Spline mit wenigen Kontrollpunkten
            spline = UnivariateSpline(t, y, k=3, s=len(t) * 0.1)
            y_reconstructed = spline(t)

            mse = np.mean((y - y_reconstructed) ** 2)
            reconstruction_errors.append(mse)
        except:
            pass

    if not reconstruction_errors:
        return {}

    avg_mse = np.mean(reconstruction_errors)
    avg_signal_power = np.mean(embeddings[:, :100] ** 2)

    return {
        "n_control_points": n_control_points,
        "avg_reconstruction_mse": float(avg_mse),
        "signal_power": float(avg_signal_power),
        "snr_db": float(10 * np.log10(avg_signal_power / avg_mse)) if avg_mse > 0 else 999,
        "compression_ratio": float(n_tokens / n_control_points),
    }


def analyze_sign_stability(embeddings: np.ndarray) -> dict:
    """
    Analysiere Sign-Stabilität über Token-Sequenz.
    Wenn Signs stabil sind, können wir Run-Length Encoding nutzen.
    """
    signs = (embeddings > 0).astype(np.int8)

    # Sign-Flips pro Dimension
    sign_changes = np.diff(signs, axis=0)  # (n-1, dim)
    flips_per_dim = np.abs(sign_changes).sum(axis=0)  # (dim,)

    n_tokens = len(embeddings)
    flip_rate = flips_per_dim / (n_tokens - 1)  # Rate pro Dimension

    # Runs: Wie lange bleibt ein Sign gleich?
    # Für RLE: Längere Runs = bessere Kompression

    return {
        "avg_flip_rate": float(flip_rate.mean()),
        "min_flip_rate": float(flip_rate.min()),
        "max_flip_rate": float(flip_rate.max()),
        "dims_with_no_flips": int((flip_rate == 0).sum()),
        "dims_with_low_flips": int((flip_rate < 0.1).sum()),  # <10% Flip-Rate
        "theoretical_rle_ratio": float(1.0 / (flip_rate.mean() + 0.01)),  # Grobe Schätzung
    }


def main():
    print("=" * 70)
    print("TOKEN DELTA ANALYSIS - Audio Engineering Perspektive")
    print("=" * 70)

    encoder = TokenEncoder()

    # Test-Texte
    texts = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
        "The discovery of antibiotics revolutionized medicine in the 20th century.",
        "Quantum computing leverages quantum mechanical phenomena to process information in fundamentally new ways.",
        "Climate change poses significant challenges to global food security and biodiversity.",
        "Neural networks are computational models inspired by the structure of biological brains.",
    ]

    all_delta_stats = []
    all_spline_stats = []
    all_sign_stats = []

    for i, text in enumerate(texts):
        print(f"\n--- Text {i+1}: '{text[:50]}...' ---")

        embeddings = encoder.encode(text)
        print(f"Tokens: {len(embeddings)}, Dims: {embeddings.shape[1]}")

        # Delta Analysis
        delta_stats = analyze_deltas(embeddings)
        all_delta_stats.append(delta_stats)

        print(f"\nDelta Analysis:")
        print(f"  Avg Delta Norm:     {delta_stats['avg_delta_norm']:.4f}")
        print(f"  Avg Embedding Norm: {delta_stats['avg_emb_norm']:.4f}")
        print(f"  Delta/Emb Ratio:    {delta_stats['avg_delta_norm']/delta_stats['avg_emb_norm']:.2%}")
        print(f"  Avg Cosine (adj):   {delta_stats['avg_cosine_adjacent']:.4f}")
        print(f"  Range Ratio:        {delta_stats['range_ratio']:.2%}")

        # Spline Analysis
        spline_stats = analyze_spline_fit(embeddings, n_control_points=10)
        all_spline_stats.append(spline_stats)

        if spline_stats:
            print(f"\nSpline Analysis (10 Kontrollpunkte):")
            print(f"  Compression:        {spline_stats['compression_ratio']:.1f}x")
            print(f"  SNR:                {spline_stats['snr_db']:.1f} dB")

        # Sign Stability
        sign_stats = analyze_sign_stability(embeddings)
        all_sign_stats.append(sign_stats)

        print(f"\nSign Stability:")
        print(f"  Avg Flip Rate:      {sign_stats['avg_flip_rate']:.2%}")
        print(f"  Dims ohne Flips:    {sign_stats['dims_with_no_flips']}")
        print(f"  Dims <10% Flips:    {sign_stats['dims_with_low_flips']}")

    # Zusammenfassung
    print("\n" + "=" * 70)
    print("ZUSAMMENFASSUNG")
    print("=" * 70)

    avg_delta_ratio = np.mean([s['avg_delta_norm']/s['avg_emb_norm'] for s in all_delta_stats])
    avg_cosine = np.mean([s['avg_cosine_adjacent'] for s in all_delta_stats])
    avg_snr = np.mean([s['snr_db'] for s in all_spline_stats if s])
    avg_flip = np.mean([s['avg_flip_rate'] for s in all_sign_stats])

    print(f"\nÜber alle Texte:")
    print(f"  Durchschn. Delta/Emb Ratio:  {avg_delta_ratio:.2%}")
    print(f"  Durchschn. Adjacent Cosine:  {avg_cosine:.4f}")
    print(f"  Durchschn. Spline SNR:       {avg_snr:.1f} dB")
    print(f"  Durchschn. Sign Flip Rate:   {avg_flip:.2%}")

    print("\n" + "=" * 70)
    print("IMPLIKATIONEN FÜR KOMPRESSION")
    print("=" * 70)

    print(f"""
1. DPCM (Delta Encoding):
   - Deltas sind {avg_delta_ratio:.0%} der Original-Norm
   - Adjacent Cosine {avg_cosine:.2f} → Tokens sind {'sehr ähnlich' if avg_cosine > 0.8 else 'moderat ähnlich' if avg_cosine > 0.5 else 'unterschiedlich'}
   - Potential: {'Gut' if avg_delta_ratio < 0.5 else 'Mittel' if avg_delta_ratio < 0.8 else 'Gering'}

2. Spline Approximation:
   - SNR {avg_snr:.0f} dB bei 10 Kontrollpunkten
   - {'Sehr gute' if avg_snr > 30 else 'Gute' if avg_snr > 20 else 'Moderate'} Rekonstruktion möglich

3. Sign-Based (RLE):
   - Flip Rate {avg_flip:.1%}
   - {'Sehr stabil' if avg_flip < 0.1 else 'Moderat stabil' if avg_flip < 0.3 else 'Instabil'} → {'RLE lohnt sich' if avg_flip < 0.2 else 'RLE weniger effektiv'}

4. Hybrid-Strategie:
   - Token-Level für Ground Truth beim Indexieren
   - Spline-Kontrollpunkte für Speicherung
   - Sign-Hash für schnelle Filterung
""")


if __name__ == "__main__":
    main()
