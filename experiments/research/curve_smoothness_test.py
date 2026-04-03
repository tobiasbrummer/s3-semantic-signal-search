#!/usr/bin/env python3
"""
Kurven-Glattheit Test

Fragestellung: Wie glatt sind Embedding-Werte über die Textposition?
Wenn glatt → Kurvenapproximation möglich → massive Kompression

Author: Claude & Toby
Date: December 2024
"""

import numpy as np
import requests
import json
from typing import List, Tuple
from dataclasses import dataclass


# =============================================================================
# LLAMA.CPP CLIENT (vereinfacht)
# =============================================================================

class Embedder:
    def __init__(self, base_url: str = "http://localhost:8200"):
        self.base_url = base_url.rstrip("/")

        # Test connection
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


# =============================================================================
# SLIDING WINDOW SIGNAL
# =============================================================================

@dataclass
class SignalAnalysis:
    """Ergebnis der Kurvenanalyse."""
    dim: int
    mean_value: float
    std_value: float
    mean_delta: float      # Durchschnittliche Änderung zwischen Positionen
    max_delta: float       # Maximale Änderung
    smoothness: float      # 1 - (mean_delta / std_value), höher = glatter


def compute_signal(
    text: str,
    embedder: Embedder,
    window_chars: int = 200,
    stride_chars: int = 50,
) -> Tuple[np.ndarray, List[int]]:
    """
    Berechne Embedding-Signal über Text.

    Returns:
        signal: (N, dim) Array - Embeddings an jeder Position
        positions: Liste der Startpositionen
    """
    positions = []
    embeddings = []

    pos = 0
    total = (len(text) - window_chars) // stride_chars + 1

    print(f"Berechne {total} Fenster (window={window_chars}, stride={stride_chars})...")

    while pos + window_chars <= len(text):
        window_text = text[pos:pos + window_chars]
        emb = embedder.embed(window_text)

        positions.append(pos)
        embeddings.append(emb)

        if len(embeddings) % 10 == 0:
            print(f"  {len(embeddings)}/{total}", end="\r")

        pos += stride_chars

    print(f"  {len(embeddings)}/{total} fertig")

    return np.array(embeddings), positions


def analyze_smoothness(signal: np.ndarray) -> List[SignalAnalysis]:
    """
    Analysiere Glattheit jeder Dimension.
    """
    n_positions, n_dims = signal.shape

    analyses = []

    for d in range(n_dims):
        values = signal[:, d]

        # Deltas zwischen benachbarten Positionen
        deltas = np.abs(np.diff(values))

        mean_val = np.mean(values)
        std_val = np.std(values)
        mean_delta = np.mean(deltas)
        max_delta = np.max(deltas)

        # Smoothness: Wie klein sind die Änderungen relativ zur Gesamtvarianz?
        # 1.0 = perfekt glatt, 0.0 = sehr sprunghaft
        if std_val > 1e-10:
            smoothness = 1.0 - (mean_delta / (2 * std_val))
            smoothness = max(0, min(1, smoothness))
        else:
            smoothness = 1.0

        analyses.append(SignalAnalysis(
            dim=d,
            mean_value=mean_val,
            std_value=std_val,
            mean_delta=mean_delta,
            max_delta=max_delta,
            smoothness=smoothness
        ))

    return analyses


def print_analysis(analyses: List[SignalAnalysis]):
    """Drucke Analyse-Ergebnisse."""

    smoothness_values = [a.smoothness for a in analyses]

    print("\n" + "=" * 70)
    print("KURVEN-GLATTHEIT ANALYSE")
    print("=" * 70)

    print(f"\nÜber alle {len(analyses)} Dimensionen:")
    print(f"  Durchschnittliche Smoothness: {np.mean(smoothness_values):.3f}")
    print(f"  Min Smoothness:               {np.min(smoothness_values):.3f}")
    print(f"  Max Smoothness:               {np.max(smoothness_values):.3f}")
    print(f"  Median Smoothness:            {np.median(smoothness_values):.3f}")

    # Verteilung
    bins = [0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
    print(f"\n  Verteilung:")
    for i in range(len(bins) - 1):
        count = sum(1 for s in smoothness_values if bins[i] <= s < bins[i+1])
        pct = count / len(smoothness_values) * 100
        bar = "#" * int(pct / 2)
        print(f"    {bins[i]:.2f}-{bins[i+1]:.2f}: {count:4d} ({pct:5.1f}%) {bar}")

    # Top 10 glatteste
    sorted_by_smooth = sorted(analyses, key=lambda a: -a.smoothness)
    print(f"\n  Top 10 glatteste Dimensionen:")
    for a in sorted_by_smooth[:10]:
        print(f"    dim_{a.dim:4d}: smoothness={a.smoothness:.3f}, "
              f"mean_delta={a.mean_delta:.4f}, std={a.std_value:.4f}")

    # Top 10 sprunghafteste
    print(f"\n  Top 10 sprunghafteste Dimensionen:")
    for a in sorted_by_smooth[-10:]:
        print(f"    dim_{a.dim:4d}: smoothness={a.smoothness:.3f}, "
              f"mean_delta={a.mean_delta:.4f}, std={a.std_value:.4f}")


def estimate_compression(signal: np.ndarray, analyses: List[SignalAnalysis]):
    """
    Schätze mögliche Kompression durch Kurvenapproximation.
    """
    n_positions, n_dims = signal.shape

    print("\n" + "=" * 70)
    print("KOMPRESSION-SCHÄTZUNG")
    print("=" * 70)

    # Unkomprimiert
    raw_bytes = n_positions * n_dims * 4  # float32
    print(f"\nUnkomprimiert: {raw_bytes:,} bytes ({raw_bytes/1024:.1f} KB)")

    # Annahme: Glatte Kurven brauchen ~1 Kontrollpunkt pro 10 Positionen
    # Sprunghaftere Kurven brauchen mehr

    total_control_points = 0
    for a in analyses:
        # Je glatter, desto weniger Kontrollpunkte
        # smoothness=1.0 → 1 Punkt pro 20 Positionen
        # smoothness=0.5 → 1 Punkt pro 2 Positionen
        points_per_pos = 0.05 + (1 - a.smoothness) * 0.45
        control_points = max(2, int(n_positions * points_per_pos))
        total_control_points += control_points

    # Spline-Speicherung: 2 floats pro Kontrollpunkt (pos, value)
    spline_bytes = total_control_points * 2 * 4

    print(f"Spline-Approximation: ~{spline_bytes:,} bytes ({spline_bytes/1024:.1f} KB)")
    print(f"Kompressionsrate: {raw_bytes/spline_bytes:.1f}x")

    # Vergleich mit Sign-Only
    sign_bytes = n_positions * n_dims // 8
    print(f"\nZum Vergleich - Sign-Only: {sign_bytes:,} bytes ({sign_bytes/1024:.1f} KB)")
    print(f"Sign-Only Kompression: {raw_bytes/sign_bytes:.1f}x")


def plot_sample_curves(signal: np.ndarray, positions: List[int], analyses: List[SignalAnalysis], num_curves: int = 5):
    """
    Zeige einige Beispielkurven als ASCII-Art.
    """
    print("\n" + "=" * 70)
    print("BEISPIEL-KURVEN (ASCII)")
    print("=" * 70)

    # Wähle Dimensionen mit verschiedener Smoothness
    sorted_analyses = sorted(analyses, key=lambda a: a.smoothness)

    # Nimm gleichmäßig verteilt
    indices = [int(i * len(sorted_analyses) / num_curves) for i in range(num_curves)]
    selected = [sorted_analyses[i] for i in indices]

    for a in selected:
        values = signal[:, a.dim]

        # Normalisiere auf 0-1
        v_min, v_max = values.min(), values.max()
        if v_max - v_min > 1e-10:
            normalized = (values - v_min) / (v_max - v_min)
        else:
            normalized = np.ones_like(values) * 0.5

        # ASCII Plot (20 Zeichen hoch, so viele breit wie Positionen, max 60)
        width = min(60, len(values))
        height = 10

        # Resample wenn nötig
        if len(values) > width:
            indices_to_use = np.linspace(0, len(values)-1, width).astype(int)
            normalized = normalized[indices_to_use]

        print(f"\ndim_{a.dim} (smoothness={a.smoothness:.3f}):")

        for row in range(height, -1, -1):
            line = ""
            threshold = row / height
            for val in normalized:
                if val >= threshold:
                    line += "█"
                else:
                    line += " "
            print(f"  {line}|")
        print("  " + "-" * len(normalized) + "+")


# =============================================================================
# MULTI-DOCUMENT ANALYSIS
# =============================================================================

def analyze_multiple_documents(
    documents: List[str],
    embedder: Embedder,
    window_chars: int = 150,
    stride_chars: int = 30,
    min_doc_length: int = 500,
) -> Tuple[List[List[SignalAnalysis]], List[np.ndarray]]:
    """
    Analysiere mehrere Dokumente.
    """
    all_analyses = []
    all_signals = []

    for i, doc in enumerate(documents):
        if len(doc) < min_doc_length:
            print(f"  Doc {i+1}: Übersprungen (zu kurz: {len(doc)} chars)")
            continue

        print(f"  Doc {i+1}/{len(documents)}: {len(doc)} chars...", end="", flush=True)

        try:
            signal, positions = compute_signal(
                doc, embedder, window_chars, stride_chars
            )

            if signal.shape[0] < 5:
                print(" übersprungen (zu wenig Fenster)")
                continue

            analyses = analyze_smoothness(signal)
            all_analyses.append(analyses)
            all_signals.append(signal)

            avg_smooth = np.mean([a.smoothness for a in analyses])
            print(f" {signal.shape[0]} Fenster, avg_smooth={avg_smooth:.3f}")

        except Exception as e:
            print(f" Fehler: {e}")
            continue

    return all_analyses, all_signals


def print_multi_doc_analysis(all_analyses: List[List[SignalAnalysis]]):
    """
    Aggregierte Analyse über mehrere Dokumente.
    """
    print("\n" + "=" * 70)
    print("AGGREGIERTE KURVEN-GLATTHEIT ANALYSE")
    print("=" * 70)

    # Sammle alle Smoothness-Werte pro Dimension über alle Dokumente
    n_dims = len(all_analyses[0]) if all_analyses else 0
    n_docs = len(all_analyses)

    print(f"\nAnalysiert: {n_docs} Dokumente, {n_dims} Dimensionen")

    # Pro Dokument: Durchschnittliche Smoothness
    doc_smoothness = []
    for i, analyses in enumerate(all_analyses):
        avg = np.mean([a.smoothness for a in analyses])
        doc_smoothness.append(avg)
        print(f"  Doc {i+1}: avg_smoothness={avg:.3f}")

    print(f"\nÜber alle Dokumente:")
    print(f"  Durchschnitt: {np.mean(doc_smoothness):.3f}")
    print(f"  Min:          {np.min(doc_smoothness):.3f}")
    print(f"  Max:          {np.max(doc_smoothness):.3f}")
    print(f"  Std:          {np.std(doc_smoothness):.3f}")

    # Pro Dimension: Durchschnittliche Smoothness über alle Dokumente
    dim_smoothness = []
    for d in range(n_dims):
        values = [analyses[d].smoothness for analyses in all_analyses]
        dim_smoothness.append(np.mean(values))

    print(f"\nPro Dimension (über alle Docs gemittelt):")
    print(f"  Durchschnitt: {np.mean(dim_smoothness):.3f}")
    print(f"  Min:          {np.min(dim_smoothness):.3f}")
    print(f"  Max:          {np.max(dim_smoothness):.3f}")

    # Verteilung
    bins = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    print(f"\n  Verteilung der Dimensions-Smoothness:")
    for i in range(len(bins) - 1):
        count = sum(1 for s in dim_smoothness if bins[i] <= s < bins[i+1])
        pct = count / len(dim_smoothness) * 100
        bar = "#" * int(pct / 2)
        print(f"    {bins[i]:.1f}-{bins[i+1]:.1f}: {count:4d} ({pct:5.1f}%) {bar}")

    # Konsistenz: Welche Dimensionen sind immer glatt/sprunghaft?
    print(f"\n  Konsistenteste glatte Dimensionen (über alle Docs):")
    dim_with_std = [(d, np.mean([a[d].smoothness for a in all_analyses]),
                     np.std([a[d].smoothness for a in all_analyses]))
                    for d in range(n_dims)]
    # Sortiere nach Smoothness, dann nach niedriger Std (konsistent)
    sorted_dims = sorted(dim_with_std, key=lambda x: (-x[1], x[2]))
    for d, mean, std in sorted_dims[:5]:
        print(f"    dim_{d:4d}: mean={mean:.3f}, std={std:.3f}")

    print(f"\n  Konsistenteste sprunghafte Dimensionen:")
    for d, mean, std in sorted_dims[-5:]:
        print(f"    dim_{d:4d}: mean={mean:.3f}, std={std:.3f}")

    return dim_smoothness


# =============================================================================
# MAIN
# =============================================================================

def main_single():
    """Original: Test mit Beispieltext."""
    SAMPLE_TEXT = """
    Machine learning is a subset of artificial intelligence that enables computers to learn from data
    without being explicitly programmed. The field has grown significantly in recent years, driven by
    advances in computing power and the availability of large datasets.

    Deep learning, a subset of machine learning, uses neural networks with many layers to learn
    complex patterns. These networks can automatically discover the representations needed for
    feature detection or classification from raw data.
    """

    print("=" * 70)
    print("KURVEN-GLATTHEIT TEST (Einzeltext)")
    print("=" * 70)

    embedder = Embedder()
    signal, positions = compute_signal(SAMPLE_TEXT, embedder, window_chars=150, stride_chars=30)
    analyses = analyze_smoothness(signal)
    print_analysis(analyses)


def main_scifact(num_docs: int = 10):
    """Test mit SciFact-Dokumenten."""
    print("=" * 70)
    print(f"KURVEN-GLATTHEIT TEST (SciFact, {num_docs} Dokumente)")
    print("=" * 70)

    # Embedder
    print("\n1. Verbinde mit Embedder...")
    try:
        embedder = Embedder()
    except Exception as e:
        print(f"   Fehler: {e}")
        print("   Starte llama.cpp Server auf localhost:8200")
        return

    # Lade SciFact
    print("\n2. Lade SciFact Dokumente...")
    try:
        from datasets import load_dataset
        corpus = load_dataset("mteb/scifact", "corpus", split="corpus")

        # Wähle Dokumente mit ausreichender Länge
        docs = []
        for doc in corpus:
            text = f"{doc['title']} {doc['text']}"
            if len(text) >= 800:  # Mindestens 800 Zeichen für genug Fenster
                docs.append(text)
            if len(docs) >= num_docs:
                break

        print(f"   {len(docs)} Dokumente ausgewählt")

    except ImportError:
        print("   pip install datasets")
        return

    # Analysiere
    print("\n3. Analysiere Dokumente...")
    all_analyses, all_signals = analyze_multiple_documents(
        docs, embedder,
        window_chars=150,
        stride_chars=30
    )

    if not all_analyses:
        print("   Keine Dokumente analysiert!")
        return

    # Ergebnisse
    dim_smoothness = print_multi_doc_analysis(all_analyses)

    # Kompression (aggregiert)
    print("\n" + "=" * 70)
    print("KOMPRESSION-SCHÄTZUNG (aggregiert)")
    print("=" * 70)

    total_positions = sum(s.shape[0] for s in all_signals)
    n_dims = all_signals[0].shape[1]

    raw_bytes = total_positions * n_dims * 4
    sign_bytes = total_positions * n_dims // 8

    # Spline-Schätzung basierend auf Smoothness
    avg_smoothness = np.mean(dim_smoothness)
    points_per_pos = 0.05 + (1 - avg_smoothness) * 0.45
    avg_control_points = int(total_positions * points_per_pos)
    spline_bytes = avg_control_points * n_dims * 2 * 4

    print(f"\nGesamt: {total_positions} Positionen über {len(all_signals)} Dokumente")
    print(f"Unkomprimiert:        {raw_bytes:,} bytes ({raw_bytes/1024:.1f} KB)")
    print(f"Spline-Approximation: ~{spline_bytes:,} bytes ({spline_bytes/1024:.1f} KB) = {raw_bytes/spline_bytes:.1f}x")
    print(f"Sign-Only:            {sign_bytes:,} bytes ({sign_bytes/1024:.1f} KB) = {raw_bytes/sign_bytes:.1f}x")

    # Fazit
    print("\n" + "=" * 70)
    print("FAZIT")
    print("=" * 70)

    avg_smooth = np.mean([np.mean([a.smoothness for a in analyses]) for analyses in all_analyses])
    print(f"\nDurchschnittliche Smoothness über alle Dokumente: {avg_smooth:.3f}")

    if avg_smooth > 0.8:
        print("→ Kurven sind GLATT - Kurvenapproximation vielversprechend!")
    elif avg_smooth > 0.6:
        print("→ Kurven sind MÄSSIG GLATT - Approximation möglich, aber Sign-Only besser.")
    else:
        print("→ Kurven sind SPRUNGHAFT - Sign-Only ist der richtige Weg.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--single":
        main_single()
    else:
        num_docs = 10
        if len(sys.argv) > 1:
            try:
                num_docs = int(sys.argv[1])
            except:
                pass
        main_scifact(num_docs)
