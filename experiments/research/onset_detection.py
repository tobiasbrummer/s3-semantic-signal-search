#!/usr/bin/env python3
"""
Onset Detection - Semantische Breakpoints (Audio-Style)

Direkt auf Embedding-Kurven arbeiten, nicht Cosine zwischen Tokens.
Wie Transient Detection in Audio: Wo ändert sich das Signal schnell?

Use Cases:
1. Multi-Resolution Zoom-Stufen definieren
2. Topic-Change Detection
3. Natürliche Segmentgrenzen für Suche
"""

import numpy as np
import requests
from dataclasses import dataclass
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


# =============================================================================
# ENCODER
# =============================================================================

class TokenEncoder:
    def __init__(self, url: str = "http://localhost:8202"):
        self.url = url
        self.dim = 1024

    def encode(self, text: str) -> np.ndarray:
        text = text[:8000]
        response = requests.post(f"{self.url}/embeddings", json={"input": text})
        response.raise_for_status()
        data = response.json()
        if isinstance(data[0]["embedding"][0], list):
            return np.array(data[0]["embedding"])
        return np.array([data[0]["embedding"]])


# =============================================================================
# ONSET DETECTION - AUDIO STYLE
# =============================================================================

def spectral_flux(embeddings: np.ndarray) -> np.ndarray:
    """
    Spectral Flux: Summe der absoluten Änderungen über alle Dimensionen.
    Wie in Audio - misst "Energie" der Änderung.
    """
    changes = np.abs(np.diff(embeddings, axis=0))  # (n-1, dim)
    flux = changes.sum(axis=1)  # (n-1,)
    return flux


def spectral_flux_positive(embeddings: np.ndarray) -> np.ndarray:
    """
    Positive Spectral Flux: Nur positive Änderungen (Onset, nicht Offset).
    """
    changes = np.diff(embeddings, axis=0)  # (n-1, dim)
    positive_changes = np.maximum(changes, 0)
    flux = positive_changes.sum(axis=1)
    return flux


def high_frequency_content(embeddings: np.ndarray) -> np.ndarray:
    """
    High Frequency Content: Gewichtete Änderungen.
    Stärkere Änderungen werden überproportional gewichtet.
    """
    changes = np.diff(embeddings, axis=0)
    hfc = (changes ** 2).sum(axis=1)  # Quadrieren verstärkt große Änderungen
    return np.sqrt(hfc)


def complex_domain(embeddings: np.ndarray) -> np.ndarray:
    """
    Complex Domain: Betrachte Änderung in Phase UND Magnitude.
    Hier approximiert: Kombination aus Richtung und Stärke.
    """
    changes = np.diff(embeddings, axis=0)

    # Magnitude der Änderung
    magnitude = np.linalg.norm(changes, axis=1)

    # "Phase" = Richtungsänderung (Winkel zwischen aufeinanderfolgenden Vektoren)
    phase_changes = []
    for i in range(len(changes) - 1):
        cos_angle = np.dot(changes[i], changes[i+1]) / (
            np.linalg.norm(changes[i]) * np.linalg.norm(changes[i+1]) + 1e-9
        )
        phase_changes.append(1 - cos_angle)  # 0 = gleiche Richtung, 2 = entgegengesetzt

    phase_changes = np.array([0] + phase_changes)  # Pad für gleiche Länge

    # Kombiniere
    return magnitude * (1 + phase_changes)


def find_onsets(
    onset_signal: np.ndarray,
    threshold_percentile: float = 85,
    min_distance: int = 10,
    smooth_sigma: float = 1.0
) -> np.ndarray:
    """
    Finde Onset-Positionen im Signal.
    """
    if smooth_sigma > 0:
        smoothed = gaussian_filter1d(onset_signal, sigma=smooth_sigma)
    else:
        smoothed = onset_signal

    threshold = np.percentile(smoothed, threshold_percentile)

    peaks, _ = find_peaks(smoothed, height=threshold, distance=min_distance)

    return peaks


# =============================================================================
# MULTI-RESOLUTION MIT ONSETS
# =============================================================================

def aggregate_by_onsets(
    embeddings: np.ndarray,
    onset_positions: np.ndarray
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """
    Aggregiere Embeddings zwischen Onset-Grenzen.

    Returns:
        segment_embeddings: (n_segments, dim) - Mean pro Segment
        segment_ranges: [(start, end), ...] - Token-Ranges pro Segment
    """
    n_tokens = len(embeddings)
    boundaries = [0] + sorted(onset_positions.tolist()) + [n_tokens]

    segments = []
    ranges = []

    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i+1]
        if end > start:
            segment_mean = embeddings[start:end].mean(axis=0)
            segments.append(segment_mean)
            ranges.append((start, end))

    return np.array(segments), ranges


@dataclass
class MultiResolutionView:
    """Multi-Resolution Darstellung eines Dokuments."""

    # Token-Level (Ground Truth)
    token_embeddings: np.ndarray  # (n_tokens, dim)

    # Onset-Level (natürliche Segmente)
    onset_positions: np.ndarray  # Onset-Indizes
    segment_embeddings: np.ndarray  # (n_segments, dim)
    segment_ranges: list  # [(start, end), ...]

    # Document-Level
    doc_embedding: np.ndarray  # (dim,)

    # Metadata
    n_tokens: int
    n_segments: int
    onset_signal: np.ndarray


def build_multi_resolution_view(
    embeddings: np.ndarray,
    onset_method: str = "spectral_flux",
    threshold_percentile: float = 85,
    min_distance: int = 10
) -> MultiResolutionView:
    """
    Baue Multi-Resolution View mit automatischen Onset-Grenzen.
    """
    n_tokens = len(embeddings)

    # Onset-Signal berechnen
    if onset_method == "spectral_flux":
        onset_signal = spectral_flux(embeddings)
    elif onset_method == "spectral_flux_positive":
        onset_signal = spectral_flux_positive(embeddings)
    elif onset_method == "hfc":
        onset_signal = high_frequency_content(embeddings)
    elif onset_method == "complex":
        onset_signal = complex_domain(embeddings)
    else:
        raise ValueError(f"Unknown method: {onset_method}")

    # Onsets finden
    onset_positions = find_onsets(
        onset_signal,
        threshold_percentile=threshold_percentile,
        min_distance=min_distance
    )

    # Segmente aggregieren
    segment_embeddings, segment_ranges = aggregate_by_onsets(embeddings, onset_positions)

    # Document-Level
    doc_embedding = embeddings.mean(axis=0)

    return MultiResolutionView(
        token_embeddings=embeddings,
        onset_positions=onset_positions,
        segment_embeddings=segment_embeddings,
        segment_ranges=segment_ranges,
        doc_embedding=doc_embedding,
        n_tokens=n_tokens,
        n_segments=len(segment_ranges),
        onset_signal=onset_signal
    )


# =============================================================================
# SEARCH MIT MULTI-RESOLUTION
# =============================================================================

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)


def search_multi_resolution(
    query_embedding: np.ndarray,
    views: dict[str, MultiResolutionView],
    doc_top_k: int = 50,
    segment_top_k: int = 20
) -> list[tuple[str, float, int, int]]:
    """
    Hierarchische Suche: Doc → Segment → Token.

    Returns: [(doc_id, score, segment_idx, token_idx), ...]
    """
    # Level 1: Document
    doc_scores = []
    for doc_id, view in views.items():
        score = cosine_sim(query_embedding, view.doc_embedding)
        doc_scores.append((doc_id, score))

    doc_scores.sort(key=lambda x: x[1], reverse=True)
    top_docs = doc_scores[:doc_top_k]

    # Level 2: Segments (nur in Top-Docs)
    segment_scores = []
    for doc_id, _ in top_docs:
        view = views[doc_id]
        for seg_idx, seg_emb in enumerate(view.segment_embeddings):
            score = cosine_sim(query_embedding, seg_emb)
            segment_scores.append((doc_id, seg_idx, score))

    segment_scores.sort(key=lambda x: x[2], reverse=True)
    top_segments = segment_scores[:segment_top_k]

    # Level 3: Tokens (nur in Top-Segments)
    results = []
    seen_docs = set()

    for doc_id, seg_idx, _ in top_segments:
        if doc_id in seen_docs:
            continue

        view = views[doc_id]
        start, end = view.segment_ranges[seg_idx]

        best_score = 0
        best_token = start

        for t in range(start, end):
            score = cosine_sim(query_embedding, view.token_embeddings[t])
            if score > best_score:
                best_score = score
                best_token = t

        results.append((doc_id, best_score, seg_idx, best_token))
        seen_docs.add(doc_id)

    results.sort(key=lambda x: x[1], reverse=True)
    return results


# =============================================================================
# DEMO & TEST
# =============================================================================

def demo():
    """Demo der Onset-basierten Multi-Resolution."""

    encoder = TokenEncoder()

    # Text mit klaren Themenwechseln
    text = """
    Machine learning is transforming how we process and understand data.
    Neural networks can now recognize images, translate languages, and
    generate creative content with remarkable accuracy.

    The stock market showed unusual volatility yesterday. Major indices
    dropped sharply in morning trading before recovering by close. Analysts
    pointed to concerns about interest rates and inflation expectations.

    In space exploration news, NASA announced a new mission to Europa.
    Scientists believe the icy moon of Jupiter may harbor conditions
    suitable for microbial life beneath its frozen surface.

    Climate researchers published alarming findings about Arctic ice loss.
    The rate of melting has accelerated beyond previous predictions, with
    significant implications for global sea levels and weather patterns.
    """

    print("=" * 70)
    print("ONSET-BASED MULTI-RESOLUTION DEMO")
    print("=" * 70)

    # Encode
    embeddings = encoder.encode(text.strip())
    print(f"\n   Tokens: {len(embeddings)}")

    # Vergleiche Onset-Methoden
    methods = ["spectral_flux", "hfc", "complex"]

    for method in methods:
        print(f"\n   --- {method} ---")

        view = build_multi_resolution_view(
            embeddings,
            onset_method=method,
            threshold_percentile=80,
            min_distance=15
        )

        print(f"   Onsets:   {len(view.onset_positions)} at {list(view.onset_positions)}")
        print(f"   Segments: {view.n_segments}")

        for i, (start, end) in enumerate(view.segment_ranges):
            seg_text = text.strip()[start*4:(start*4)+60].replace('\n', ' ')
            print(f"     [{i}] Token {start}-{end}: \"{seg_text}...\"")

    # Multi-Resolution Suche Demo
    print("\n" + "=" * 70)
    print("MULTI-RESOLUTION SEARCH DEMO")
    print("=" * 70)

    # Baue Index
    view = build_multi_resolution_view(embeddings, "spectral_flux", 80, 15)
    views = {"doc1": view}

    # Query
    query = "space exploration and NASA missions"
    query_emb = encoder.encode(query).mean(axis=0)  # Pool query

    print(f"\n   Query: \"{query}\"")

    # Hierarchische Suche
    results = search_multi_resolution(query_emb, views, doc_top_k=1, segment_top_k=5)

    for doc_id, score, seg_idx, token_idx in results:
        start, end = view.segment_ranges[seg_idx]
        print(f"\n   Match: score={score:.4f}")
        print(f"   Segment {seg_idx} (tokens {start}-{end})")
        print(f"   Best token: {token_idx}")

    # Zeige alle Segment-Scores
    print("\n   Alle Segment-Scores:")
    for i, seg_emb in enumerate(view.segment_embeddings):
        score = cosine_sim(query_emb, seg_emb)
        start, end = view.segment_ranges[i]
        preview = text.strip()[start*4:(start*4)+50].replace('\n', ' ')
        print(f"     [{i}] {score:.4f}: \"{preview}...\"")


def compare_fixed_vs_onset():
    """Vergleiche Fixed Chunks vs Onset-basierte Segmente."""

    encoder = TokenEncoder()

    text = """
    The development of quantum computers represents a paradigm shift in
    computational capabilities. Unlike classical computers that use bits,
    quantum computers leverage qubits that can exist in superposition.

    Yesterday's football match ended in a surprising upset. The underdog
    team scored in the final minutes to secure an unexpected victory.
    Fans celebrated throughout the night in the city center.

    Medical researchers announced a breakthrough in cancer treatment.
    The new therapy targets specific mutations while sparing healthy cells.
    Clinical trials showed promising results with minimal side effects.
    """

    embeddings = encoder.encode(text.strip())
    n_tokens = len(embeddings)

    print("\n" + "=" * 70)
    print("FIXED CHUNKS vs ONSET-BASED SEGMENTS")
    print("=" * 70)

    # Fixed Chunks (z.B. alle 30 Tokens)
    fixed_size = 30
    n_fixed = n_tokens // fixed_size + (1 if n_tokens % fixed_size else 0)

    print(f"\n   Tokens total: {n_tokens}")
    print(f"\n   Fixed Chunks ({fixed_size} tokens): {n_fixed} Segmente")

    for i in range(n_fixed):
        start = i * fixed_size
        end = min(start + fixed_size, n_tokens)
        preview = text.strip()[start*4:(start*4)+50].replace('\n', ' ')
        print(f"     [{i}] {start}-{end}: \"{preview}...\"")

    # Onset-based
    view = build_multi_resolution_view(embeddings, "spectral_flux", 80, 15)

    print(f"\n   Onset-based: {view.n_segments} Segmente")

    for i, (start, end) in enumerate(view.segment_ranges):
        preview = text.strip()[start*4:(start*4)+50].replace('\n', ' ')
        print(f"     [{i}] {start}-{end} ({end-start} tok): \"{preview}...\"")

    print(f"\n   Onset-Positionen: {list(view.onset_positions)}")
    print(f"   (= wahrscheinlich die Themenwechsel)")


if __name__ == "__main__":
    demo()
    compare_fixed_vs_onset()

    print("\n" + "=" * 70)
    print("FAZIT")
    print("=" * 70)
    print("""
   Onset Detection (Audio-Style):
   ✓ Spectral Flux auf Embedding-Kurven
   ✓ Keine Cosine-Berechnung zwischen Tokens nötig
   ✓ Definiert natürliche Zoom-Stufen für Multi-Resolution

   Multi-Resolution Hierarchie:
   Level 1: Document (1 Vektor)
   Level 2: Onset-Segmente (N Vektoren, semantisch begründet)
   Level 3: Tokens (M Vektoren)

   Bereit für BEIR Test!
""")
