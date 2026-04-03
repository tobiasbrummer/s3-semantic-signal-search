#!/usr/bin/env python3
"""
Pattern Search: Spline-Speicherung + Cross-Correlation

Ermöglicht Such-Modi die klassisches RAG nicht kann:
1. Query-by-Example: "Finde Absätze die so argumentieren wie dieser"
2. Dramaturgie-Suche: "Erst negativ, dann positiv" als Kurve
3. Anomaly Detection: "Wo weicht das Signal vom Standard ab?"

Architektur:
- Spline-Kontrollpunkte statt rohe Vektoren (Kompression + Interpolation)
- Cross-Correlation via FFT (schnelles Pattern-Matching)
"""

import numpy as np
from scipy import interpolate, signal
from scipy.fft import fft, ifft, fftfreq
import requests
from dataclasses import dataclass, field
from typing import Optional, Callable
import time


# =============================================================================
# DENSE ENCODER (gleich wie in hybrid_search.py)
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
# SPLINE SIGNAL
# =============================================================================

@dataclass
class SplineSignal:
    """
    Ein Dokument als Spline-komprimiertes Signal.

    Speichert Kontrollpunkte statt rohe Werte.
    Kann bei Bedarf interpoliert werden.
    """
    doc_id: str
    text: str

    # Kontrollpunkte: (n_control_points, n_dims)
    control_points: np.ndarray

    # Positionen der Kontrollpunkte (in Zeichen)
    control_positions: np.ndarray

    # Original-Länge des Dokuments
    doc_length: int

    # Anzahl Dimensionen
    n_dims: int

    # Spline-Degree (3 = kubisch)
    spline_degree: int = 3

    # Cached interpolators (lazy init)
    _interpolators: list = field(default_factory=list, repr=False)

    @property
    def n_control_points(self) -> int:
        return len(self.control_positions)

    @property
    def compression_ratio(self) -> float:
        """Kompressionsrate vs. rohe Speicherung."""
        # Annahme: Original wäre doc_length/30 Positionen × n_dims × 4 bytes
        original = (self.doc_length / 30) * self.n_dims * 4
        compressed = self.n_control_points * self.n_dims * 4 + self.n_control_points * 4
        return original / compressed if compressed > 0 else 0

    def _build_interpolators(self):
        """Baue Spline-Interpolatoren für jede Dimension."""
        if self._interpolators:
            return

        for dim in range(self.n_dims):
            values = self.control_points[:, dim]

            # Spline-Interpolator erstellen
            # k = min(spline_degree, n_points - 1) für Stabilität
            k = min(self.spline_degree, len(self.control_positions) - 1)

            spline = interpolate.UnivariateSpline(
                self.control_positions,
                values,
                k=k,
                s=0  # Exakte Interpolation durch Kontrollpunkte
            )
            self._interpolators.append(spline)

    def get_value_at(self, position: float, dim: int) -> float:
        """Hole interpolierten Wert an einer Position für eine Dimension."""
        self._build_interpolators()
        return float(self._interpolators[dim](position))

    def get_signal(self, start: int = 0, end: Optional[int] = None,
                   resolution: int = 30) -> np.ndarray:
        """
        Rekonstruiere Signal-Ausschnitt durch Interpolation.

        Args:
            start: Startposition (in Zeichen)
            end: Endposition (in Zeichen)
            resolution: Abstand zwischen Samples (in Zeichen)

        Returns:
            np.ndarray of shape (n_samples, n_dims)
        """
        self._build_interpolators()

        if end is None:
            end = self.doc_length

        positions = np.arange(start, end, resolution)

        signal = np.zeros((len(positions), self.n_dims))
        for dim in range(self.n_dims):
            signal[:, dim] = self._interpolators[dim](positions)

        return signal

    def get_envelope(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Hole Min/Max Envelope für schnelle Filterung.

        Returns:
            (min_values, max_values) pro Segment zwischen Kontrollpunkten
        """
        n_segments = self.n_control_points - 1
        min_vals = np.zeros((n_segments, self.n_dims))
        max_vals = np.zeros((n_segments, self.n_dims))

        for i in range(n_segments):
            segment = self.control_points[i:i+2]
            min_vals[i] = segment.min(axis=0)
            max_vals[i] = segment.max(axis=0)

        return min_vals, max_vals


def create_spline_signal(
    doc_id: str,
    text: str,
    embeddings: np.ndarray,
    positions: np.ndarray,
    downsample_factor: int = 3
) -> SplineSignal:
    """
    Erstelle SplineSignal aus Dense Embeddings.

    Args:
        doc_id: Dokument-ID
        text: Original-Text
        embeddings: Shape (n_positions, n_dims)
        positions: Positionen in Zeichen
        downsample_factor: Nur jeden N-ten Punkt als Kontrollpunkt nehmen
    """
    # Downsample für Kompression
    indices = np.arange(0, len(positions), downsample_factor)

    return SplineSignal(
        doc_id=doc_id,
        text=text,
        control_points=embeddings[indices].copy(),
        control_positions=positions[indices].copy(),
        doc_length=len(text),
        n_dims=embeddings.shape[1]
    )


# =============================================================================
# CROSS-CORRELATION
# =============================================================================

def cross_correlate_signals(
    query_signal: np.ndarray,
    doc_signal: np.ndarray,
    method: str = "fft"
) -> np.ndarray:
    """
    Berechne Cross-Correlation zwischen Query und Dokument.

    Args:
        query_signal: Shape (query_len, n_dims)
        doc_signal: Shape (doc_len, n_dims)
        method: "fft" (schnell) oder "direct" (einfach)

    Returns:
        Correlation scores für jede Position im Dokument
    """
    query_len, n_dims = query_signal.shape
    doc_len = doc_signal.shape[0]

    if method == "fft":
        # FFT-basierte Correlation (O(n log n))
        # Korreliere jede Dimension separat, dann summieren

        correlations = np.zeros(doc_len - query_len + 1)

        for dim in range(n_dims):
            q = query_signal[:, dim]
            d = doc_signal[:, dim]

            # Normalisiere
            q = (q - q.mean()) / (q.std() + 1e-10)
            d = (d - d.mean()) / (d.std() + 1e-10)

            # Cross-Correlation via FFT
            corr = signal.correlate(d, q, mode='valid')
            correlations += corr

        # Normalisiere nach Dimensionen und Query-Länge
        correlations /= (n_dims * query_len)

        return correlations

    else:
        # Direkte Methode (O(n*m)) - langsamer aber einfacher
        correlations = np.zeros(doc_len - query_len + 1)

        for i in range(len(correlations)):
            window = doc_signal[i:i+query_len]
            # Cosine Similarity
            q_norm = query_signal / (np.linalg.norm(query_signal, axis=1, keepdims=True) + 1e-10)
            w_norm = window / (np.linalg.norm(window, axis=1, keepdims=True) + 1e-10)
            correlations[i] = np.mean(np.sum(q_norm * w_norm, axis=1))

        return correlations


def find_pattern_matches(
    query_signal: np.ndarray,
    doc_signals: list[SplineSignal],
    top_k: int = 10,
    resolution: int = 30
) -> list[tuple[str, int, float]]:
    """
    Finde die besten Matches für ein Query-Pattern in allen Dokumenten.

    Args:
        query_signal: Das Pattern als Signal (n_samples, n_dims)
        doc_signals: Liste von SplineSignal Objekten
        top_k: Anzahl Ergebnisse
        resolution: Sample-Abstand für Reconstruction

    Returns:
        Liste von (doc_id, position, score)
    """
    results = []
    query_len = len(query_signal)

    for spline in doc_signals:
        # Rekonstruiere Dokument-Signal
        doc_signal = spline.get_signal(resolution=resolution)

        if len(doc_signal) < query_len:
            continue

        # Cross-Correlation
        correlations = cross_correlate_signals(query_signal, doc_signal)

        # Beste Position in diesem Dokument
        best_idx = np.argmax(correlations)
        best_score = correlations[best_idx]
        best_position = best_idx * resolution

        results.append((spline.doc_id, best_position, best_score))

    # Sortiere nach Score
    results.sort(key=lambda x: x[2], reverse=True)

    return results[:top_k]


# =============================================================================
# SEMANTIC PATTERN GENERATORS
# =============================================================================

class SemanticPatternGenerator:
    """
    Erstellt Patterns aus echten semantischen Ankern.

    Statt abstrakter Zahlen (-1 bis +1) nutzen wir echte Embeddings
    von semantisch relevanten Texten.
    """

    def __init__(self, encoder: DenseEncoder):
        self.encoder = encoder
        self._cache = {}

    def _get_anchor(self, texts: list[str]) -> np.ndarray:
        """Hole gemitteltes Embedding für eine Liste von Texten."""
        key = tuple(texts)
        if key not in self._cache:
            embeddings = self.encoder.encode(texts)
            self._cache[key] = embeddings.mean(axis=0)
        return self._cache[key]

    def create_pattern(
        self,
        pattern_type: str,
        length: int = 5,
        **kwargs
    ) -> np.ndarray:
        """
        Erstelle semantisches Pattern durch Interpolation zwischen Ankern.

        Args:
            pattern_type: "rising", "falling", "peak", "valley", "conflict"
            length: Anzahl Samples im Pattern

        Returns:
            np.ndarray of shape (length, n_dims)
        """
        if pattern_type == "rising":
            # Negativ → Positiv (Failure to Success)
            start = self._get_anchor([
                "terrible failure disaster",
                "bad negative problem crisis",
                "struggling difficulty hardship"
            ])
            end = self._get_anchor([
                "great success triumph victory",
                "excellent positive achievement",
                "thriving prosperity breakthrough"
            ])
            return self._interpolate(start, end, length)

        elif pattern_type == "falling":
            # Positiv → Negativ (Success to Failure)
            start = self._get_anchor([
                "great success triumph victory",
                "excellent positive achievement",
                "thriving prosperity breakthrough"
            ])
            end = self._get_anchor([
                "terrible failure disaster",
                "bad negative problem crisis",
                "struggling difficulty hardship"
            ])
            return self._interpolate(start, end, length)

        elif pattern_type == "peak":
            # Neutral → Positiv → Neutral
            neutral = self._get_anchor([
                "normal average ordinary",
                "standard typical regular",
                "moderate middle medium"
            ])
            positive = self._get_anchor([
                "excellent amazing outstanding",
                "brilliant superb exceptional",
                "peak climax highlight"
            ])
            first_half = self._interpolate(neutral, positive, length // 2 + 1)
            second_half = self._interpolate(positive, neutral, length - length // 2)
            return np.vstack([first_half[:-1], second_half])

        elif pattern_type == "valley":
            # Neutral → Negativ → Neutral
            neutral = self._get_anchor([
                "normal average ordinary",
                "standard typical regular"
            ])
            negative = self._get_anchor([
                "terrible awful horrible",
                "worst disaster catastrophe",
                "low point nadir bottom"
            ])
            first_half = self._interpolate(neutral, negative, length // 2 + 1)
            second_half = self._interpolate(negative, neutral, length - length // 2)
            return np.vstack([first_half[:-1], second_half])

        elif pattern_type == "conflict":
            # Frieden → Konflikt → Lösung
            peace = self._get_anchor([
                "peaceful calm harmony agreement",
                "cooperative friendly collaborative"
            ])
            conflict = self._get_anchor([
                "conflict fight argument dispute",
                "tension disagreement clash battle"
            ])
            resolution = self._get_anchor([
                "resolution solution compromise",
                "reconciliation peace settlement"
            ])
            part1 = self._interpolate(peace, conflict, length // 3 + 1)
            part2 = self._interpolate(conflict, resolution, length - length // 3)
            return np.vstack([part1[:-1], part2])

        elif pattern_type == "custom":
            # Custom Anker aus kwargs
            anchors = kwargs.get("anchors", [])
            if len(anchors) < 2:
                raise ValueError("Need at least 2 anchors for custom pattern")

            anchor_embeddings = [self._get_anchor([a]) for a in anchors]
            steps_per_segment = length // (len(anchors) - 1)

            segments = []
            for i in range(len(anchor_embeddings) - 1):
                seg = self._interpolate(
                    anchor_embeddings[i],
                    anchor_embeddings[i + 1],
                    steps_per_segment + 1
                )
                segments.append(seg[:-1] if i < len(anchor_embeddings) - 2 else seg)

            return np.vstack(segments)

        else:
            raise ValueError(f"Unknown pattern type: {pattern_type}")

    def _interpolate(
        self,
        start: np.ndarray,
        end: np.ndarray,
        steps: int
    ) -> np.ndarray:
        """Lineare Interpolation zwischen zwei Embeddings."""
        weights = np.linspace(0, 1, steps)
        return np.array([
            start * (1 - w) + end * w
            for w in weights
        ])


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("PATTERN SEARCH: Splines + Cross-Correlation")
    print("=" * 70)

    # Encoder laden
    print("\n1. Lade Encoder...")
    encoder = DenseEncoder()
    print(f"   Dim: {encoder.dim}")

    # Test-Dokumente
    print("\n2. Erstelle Test-Dokumente...")

    test_docs = [
        {
            "id": "doc_1",
            "text": """
            The situation started badly. Critics were harsh and reviews negative.
            But over time, things improved. People began to appreciate the work.
            Eventually, it became a massive success. Everyone loved it.
            The transformation from failure to triumph was remarkable.
            """
        },
        {
            "id": "doc_2",
            "text": """
            Initial reception was overwhelmingly positive. Everyone praised it.
            However, problems emerged. Quality declined steadily.
            By the end, it was considered a disappointment.
            The fall from grace was dramatic and swift.
            """
        },
        {
            "id": "doc_3",
            "text": """
            The project maintained consistent quality throughout.
            Neither exceptional highs nor devastating lows.
            A steady, reliable performance from start to finish.
            Predictable but dependable results overall.
            """
        }
    ]

    # Signale erstellen
    print("\n3. Erstelle Spline-Signale...")
    spline_signals = []

    for doc in test_docs:
        # Sliding Window Embeddings
        text = doc["text"]
        window_size = 100
        stride = 20

        windows = []
        positions = []
        for i in range(0, max(1, len(text) - window_size + 1), stride):
            windows.append(text[i:i+window_size])
            positions.append(i)

        if not windows:
            windows = [text]
            positions = [0]

        embeddings = encoder.encode(windows)
        positions = np.array(positions)

        spline = create_spline_signal(
            doc["id"], text, embeddings, positions,
            downsample_factor=2
        )
        spline_signals.append(spline)

        print(f"   {doc['id']}: {spline.n_control_points} Kontrollpunkte, "
              f"Kompression: {spline.compression_ratio:.1f}x")

    # Pattern Generator
    print("\n4. Erstelle Semantic Pattern Generator...")
    pattern_gen = SemanticPatternGenerator(encoder)

    # Pattern-Suche: "Rising" (erst schlecht, dann gut)
    print("\n5. Pattern-Suche: 'Rising' (negativ → positiv)")
    print("   Erwartung: doc_1 (failure to success story)")
    print("=" * 70)

    rising_pattern = pattern_gen.create_pattern("rising", length=5)

    results = find_pattern_matches(rising_pattern, spline_signals, top_k=3)

    for i, (doc_id, position, score) in enumerate(results):
        doc = next(d for d in test_docs if d["id"] == doc_id)
        snippet = doc["text"][position:position+100].strip().replace("\n", " ")
        print(f"\n   {i+1}. {doc_id} (pos={position}, score={score:.4f})")
        print(f"      \"{snippet}...\"")

    # Pattern-Suche: "Falling" (erst gut, dann schlecht)
    print("\n6. Pattern-Suche: 'Falling' (positiv → negativ)")
    print("   Erwartung: doc_2 (success to failure story)")
    print("=" * 70)

    falling_pattern = pattern_gen.create_pattern("falling", length=5)

    results = find_pattern_matches(falling_pattern, spline_signals, top_k=3)

    for i, (doc_id, position, score) in enumerate(results):
        doc = next(d for d in test_docs if d["id"] == doc_id)
        snippet = doc["text"][position:position+100].strip().replace("\n", " ")
        print(f"\n   {i+1}. {doc_id} (pos={position}, score={score:.4f})")
        print(f"      \"{snippet}...\"")

    # Query-by-Example
    print("\n7. Query-by-Example")
    print("   Suche nach Textstellen die semantisch ähnlich zu einem Beispiel sind")
    print("=" * 70)

    example_text = "The transformation from failure to triumph was remarkable"
    print(f"   Example: \"{example_text}\"")

    # Erstelle ein kurzes Signal aus dem Beispiel
    # Wir embedden den Text und ein paar Variationen
    example_variants = [
        example_text,
        "changed from bad to good dramatically",
        "went from struggling to succeeding"
    ]
    example_embeddings = encoder.encode(example_variants)
    example_signal = example_embeddings  # 3 Samples als Signal

    results = find_pattern_matches(example_signal, spline_signals, top_k=3)

    for i, (doc_id, position, score) in enumerate(results):
        doc = next(d for d in test_docs if d["id"] == doc_id)
        snippet = doc["text"][position:position+100].strip().replace("\n", " ")
        print(f"\n   {i+1}. {doc_id} (pos={position}, score={score:.4f})")
        print(f"      \"{snippet}...\"")

    # Speichervergleich
    print("\n" + "=" * 70)
    print("SPEICHERVERGLEICH")
    print("=" * 70)

    for spline in spline_signals:
        raw_size = (spline.doc_length / 30) * spline.n_dims * 4
        spline_size = spline.n_control_points * spline.n_dims * 4
        print(f"   {spline.doc_id}: Raw={raw_size/1024:.1f}KB, "
              f"Spline={spline_size/1024:.1f}KB, "
              f"Ratio={spline.compression_ratio:.1f}x")


if __name__ == "__main__":
    main()
