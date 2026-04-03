#!/usr/bin/env python3
"""
Audio-Inspired Retrieval Methods

Thinking like audio engineers, not database engineers.

Methods:
1. Peak Fingerprinting (Shazam-style)
2. Spectral Correlation (FFT-based similarity)
3. Resonance Filter (Query as filter bank)
4. Interference Patterns (Constructive/Destructive)
5. Harmonic Analysis (Overtone relationships)
6. Cepstral Matching (Source/Filter separation)

Author: Claude & Toby
Date: December 2024
"""

import numpy as np
from typing import List, Dict, Tuple, Set
from collections import defaultdict
from dataclasses import dataclass
import time

try:
    from scipy import fft as scipy_fft
    from scipy.signal import find_peaks, correlate
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("⚠️  scipy not installed, some methods disabled")


# =============================================================================
# 1. PEAK FINGERPRINTING (Shazam for Embeddings)
# =============================================================================

class PeakFingerprint:
    """
    Shazam-style fingerprinting for embeddings.
    
    Key insight: Instead of hashing ALL dimensions, 
    only hash the PEAK dimensions (where embedding has extreme values).
    
    Similar embeddings should have peaks at similar positions.
    """
    
    def __init__(self, 
                 num_peaks: int = 32,
                 peak_pairs: bool = True,
                 pair_distance: int = 50):
        """
        Args:
            num_peaks: How many top dimensions to consider as "peaks"
            peak_pairs: Whether to use peak pairs (like Shazam) or single peaks
            pair_distance: Max distance between peaks in a pair
        """
        self.num_peaks = num_peaks
        self.peak_pairs = peak_pairs
        self.pair_distance = pair_distance
        
        # Inverted index: peak_key → [(doc_id, peak_value), ...]
        self.index = defaultdict(list)
        self.doc_embeddings = {}
    
    def _extract_peaks(self, embedding: np.ndarray) -> List[Tuple[int, float]]:
        """
        Extract peak dimensions from embedding.
        
        Returns list of (dimension_index, value) sorted by absolute value.
        """
        abs_values = np.abs(embedding)
        
        # Get indices of top-k by absolute value
        top_indices = np.argsort(abs_values)[-self.num_peaks:][::-1]
        
        # Return (index, value) pairs
        peaks = [(int(idx), float(embedding[idx])) for idx in top_indices]
        return peaks
    
    def _make_fingerprint(self, peaks: List[Tuple[int, float]]) -> Set[Tuple]:
        """
        Create fingerprint from peaks.
        
        If peak_pairs=True: Use pairs of peaks (more discriminative)
        If peak_pairs=False: Use single peaks
        """
        fingerprint = set()
        
        if self.peak_pairs:
            # Create peak pairs (like Shazam constellation)
            for i, (idx1, val1) in enumerate(peaks):
                for j, (idx2, val2) in enumerate(peaks[i+1:], i+1):
                    # Only pair peaks within distance
                    if abs(idx2 - idx1) <= self.pair_distance:
                        # Key: (peak1_pos, peak2_pos, sign1, sign2)
                        sign1 = 1 if val1 > 0 else 0
                        sign2 = 1 if val2 > 0 else 0
                        key = (idx1, idx2, sign1, sign2)
                        fingerprint.add(key)
        else:
            # Single peaks with sign
            for idx, val in peaks:
                sign = 1 if val > 0 else 0
                key = (idx, sign)
                fingerprint.add(key)
        
        return fingerprint
    
    def add(self, doc_id: str, embedding: np.ndarray):
        """Add document to index."""
        self.doc_embeddings[doc_id] = embedding
        
        peaks = self._extract_peaks(embedding)
        fingerprint = self._make_fingerprint(peaks)
        
        for key in fingerprint:
            self.index[key].append(doc_id)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search using vote counting.
        
        For each query peak/pair, find docs with same peak/pair.
        Count votes. More votes = more similar.
        """
        peaks = self._extract_peaks(query_embedding)
        fingerprint = self._make_fingerprint(peaks)
        
        # Vote counting
        votes = defaultdict(int)
        
        for key in fingerprint:
            for doc_id in self.index.get(key, []):
                votes[doc_id] += 1
        
        # Sort by votes (descending) and convert to distance (ascending)
        max_votes = len(fingerprint)
        results = [
            (doc_id, max_votes - vote_count)  # Distance = max - votes
            for doc_id, vote_count in votes.items()
        ]
        results.sort(key=lambda x: x[1])
        
        return results[:top_k]


# =============================================================================
# 2. SPECTRAL CORRELATION (FFT-based)
# =============================================================================

class SpectralCorrelation:
    """
    Use FFT for fast correlation-based retrieval.
    
    Key insight: Correlation in frequency domain is faster than time domain.
    
    cross_correlation(a, b) = IFFT(FFT(a) * conj(FFT(b)))
    
    But for retrieval we need a different trick:
    Pre-compute FFT of all docs, then multiply with query FFT.
    """
    
    def __init__(self, use_magnitude_only: bool = False):
        """
        Args:
            use_magnitude_only: If True, ignore phase (more robust to small shifts)
        """
        self.use_magnitude_only = use_magnitude_only
        self.doc_ffts = {}
        self.doc_ids = []
    
    def add(self, doc_id: str, embedding: np.ndarray):
        """Add document."""
        doc_fft = np.fft.fft(embedding)
        
        if self.use_magnitude_only:
            doc_fft = np.abs(doc_fft)
        
        self.doc_ffts[doc_id] = doc_fft
        self.doc_ids.append(doc_id)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search using spectral correlation.
        
        Higher correlation peak = more similar.
        """
        query_fft = np.fft.fft(query_embedding)
        
        if self.use_magnitude_only:
            query_fft = np.abs(query_fft)
        
        results = []
        
        for doc_id in self.doc_ids:
            doc_fft = self.doc_ffts[doc_id]
            
            if self.use_magnitude_only:
                # Magnitude-only: just dot product of spectra
                correlation = np.real(np.sum(query_fft * doc_fft))
            else:
                # Full: cross-correlation peak
                cross_spectrum = query_fft * np.conj(doc_fft)
                correlation_signal = np.fft.ifft(cross_spectrum)
                correlation = np.max(np.abs(correlation_signal))
            
            # Higher correlation = smaller distance
            results.append((doc_id, -correlation))
        
        results.sort(key=lambda x: x[1])
        return results[:top_k]


# =============================================================================
# 3. RESONANCE FILTER (Query as Filter Bank)
# =============================================================================

class ResonanceFilter:
    """
    Treat query as a filter bank, measure resonance with documents.
    
    Key insight: If two signals have similar frequency content,
    one will "resonate" when filtered by the other.
    
    This is essentially a weighted dot product where the query
    emphasizes certain dimensions (frequencies).
    """
    
    def __init__(self, 
                 num_filters: int = 16,
                 filter_width: int = 64,
                 filter_type: str = "gaussian"):
        """
        Args:
            num_filters: Number of filter bands
            filter_width: Width of each filter (in dimensions)
            filter_type: "gaussian", "rectangular", or "triangular"
        """
        self.num_filters = num_filters
        self.filter_width = filter_width
        self.filter_type = filter_type
        
        self.doc_embeddings = {}
        self.doc_filter_responses = {}
    
    def _create_filter_bank(self, dim: int) -> np.ndarray:
        """Create filter bank matrix."""
        filters = np.zeros((self.num_filters, dim))
        
        centers = np.linspace(0, dim - 1, self.num_filters + 2)[1:-1]
        
        for i, center in enumerate(centers):
            if self.filter_type == "gaussian":
                x = np.arange(dim)
                sigma = self.filter_width / 3
                filters[i] = np.exp(-0.5 * ((x - center) / sigma) ** 2)
            elif self.filter_type == "rectangular":
                start = max(0, int(center - self.filter_width / 2))
                end = min(dim, int(center + self.filter_width / 2))
                filters[i, start:end] = 1.0
            elif self.filter_type == "triangular":
                x = np.arange(dim)
                filters[i] = np.maximum(0, 1 - np.abs(x - center) / (self.filter_width / 2))
        
        # Normalize
        filters = filters / (np.sum(filters, axis=1, keepdims=True) + 1e-10)
        
        return filters
    
    def add(self, doc_id: str, embedding: np.ndarray):
        """Add document."""
        self.doc_embeddings[doc_id] = embedding
        
        if not hasattr(self, 'filter_bank'):
            self.filter_bank = self._create_filter_bank(len(embedding))
        
        # Pre-compute filter responses
        responses = self.filter_bank @ embedding  # (num_filters,)
        self.doc_filter_responses[doc_id] = responses
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search by filter response similarity.
        
        Documents with similar filter responses "resonate" with the query.
        """
        query_responses = self.filter_bank @ query_embedding
        
        results = []
        
        for doc_id, doc_responses in self.doc_filter_responses.items():
            # Correlation of filter responses
            resonance = np.dot(query_responses, doc_responses)
            results.append((doc_id, -resonance))
        
        results.sort(key=lambda x: x[1])
        return results[:top_k]


# =============================================================================
# 4. INTERFERENCE PATTERNS
# =============================================================================

class InterferencePattern:
    """
    Use wave interference for similarity.
    
    Key insight: 
    - Similar waves → constructive interference → high amplitude
    - Different waves → destructive interference → low amplitude
    
    Measure the "interference energy" when combining query + doc.
    """
    
    def __init__(self, normalize: bool = True):
        self.normalize = normalize
        self.doc_embeddings = {}
        self.doc_norms = {}
    
    def add(self, doc_id: str, embedding: np.ndarray):
        """Add document."""
        if self.normalize:
            norm = np.linalg.norm(embedding)
            embedding = embedding / max(norm, 1e-10)
            self.doc_norms[doc_id] = norm
        
        self.doc_embeddings[doc_id] = embedding
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search using interference energy.
        
        Combined signal energy = ||query + doc||²
        = ||query||² + ||doc||² + 2 * dot(query, doc)
        
        For normalized vectors:
        = 2 + 2 * cosine_similarity
        
        So this IS cosine similarity, but framed as interference!
        """
        if self.normalize:
            query_norm = np.linalg.norm(query_embedding)
            query_embedding = query_embedding / max(query_norm, 1e-10)
        
        results = []
        
        for doc_id, doc_emb in self.doc_embeddings.items():
            # Interference: add the waves
            combined = query_embedding + doc_emb
            
            # Measure combined energy
            interference_energy = np.sum(combined ** 2)
            
            # Higher energy = more similar = lower distance
            results.append((doc_id, -interference_energy))
        
        results.sort(key=lambda x: x[1])
        return results[:top_k]


# =============================================================================
# 5. HARMONIC FINGERPRINT
# =============================================================================

class HarmonicFingerprint:
    """
    Look for harmonic relationships in embeddings.
    
    In audio: If fundamental frequency is f, harmonics are 2f, 3f, 4f...
    
    In embeddings: If dimension d is "important", check if 2d, 3d are correlated.
    
    This creates a "harmonic signature" for each embedding.
    """
    
    def __init__(self, 
                 max_harmonic: int = 4,
                 num_fundamentals: int = 64):
        """
        Args:
            max_harmonic: Check harmonics up to this multiple
            num_fundamentals: Consider first N dimensions as potential fundamentals
        """
        self.max_harmonic = max_harmonic
        self.num_fundamentals = num_fundamentals
        
        self.doc_harmonics = {}
    
    def _extract_harmonic_signature(self, embedding: np.ndarray) -> np.ndarray:
        """
        Extract harmonic signature.
        
        For each potential fundamental d, compute correlation with harmonics.
        """
        dim = len(embedding)
        signature = np.zeros(self.num_fundamentals)
        
        for d in range(1, self.num_fundamentals + 1):
            harmonic_sum = embedding[d - 1]  # Fundamental
            
            for h in range(2, self.max_harmonic + 1):
                harmonic_idx = d * h - 1
                if harmonic_idx < dim:
                    # Weight by harmonic number (higher harmonics less important)
                    harmonic_sum += embedding[harmonic_idx] / h
            
            signature[d - 1] = harmonic_sum
        
        return signature
    
    def add(self, doc_id: str, embedding: np.ndarray):
        """Add document."""
        signature = self._extract_harmonic_signature(embedding)
        self.doc_harmonics[doc_id] = signature
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search by harmonic signature similarity."""
        query_sig = self._extract_harmonic_signature(query_embedding)
        
        results = []
        
        for doc_id, doc_sig in self.doc_harmonics.items():
            # Cosine similarity of signatures
            similarity = np.dot(query_sig, doc_sig) / (
                np.linalg.norm(query_sig) * np.linalg.norm(doc_sig) + 1e-10
            )
            results.append((doc_id, -similarity))
        
        results.sort(key=lambda x: x[1])
        return results[:top_k]


# =============================================================================
# 6. CEPSTRAL MATCHING
# =============================================================================

class CepstralMatcher:
    """
    Cepstrum-based matching.
    
    Cepstrum = IFFT(log(|FFT(signal)|))
    
    In audio: Separates "source" (vocal cords) from "filter" (vocal tract).
    In embeddings: Might separate "topic" from "style"?
    
    The low cepstrum coefficients capture overall spectral shape.
    """
    
    def __init__(self, num_coefficients: int = 32):
        """
        Args:
            num_coefficients: How many cepstral coefficients to use
        """
        self.num_coefficients = num_coefficients
        self.doc_cepstra = {}
    
    def _compute_cepstrum(self, embedding: np.ndarray) -> np.ndarray:
        """Compute cepstrum of embedding."""
        # FFT
        spectrum = np.fft.fft(embedding)
        
        # Log magnitude (add small value to avoid log(0))
        log_magnitude = np.log(np.abs(spectrum) + 1e-10)
        
        # IFFT
        cepstrum = np.real(np.fft.ifft(log_magnitude))
        
        # Take first N coefficients (low "quefrency")
        return cepstrum[:self.num_coefficients]
    
    def add(self, doc_id: str, embedding: np.ndarray):
        """Add document."""
        cepstrum = self._compute_cepstrum(embedding)
        self.doc_cepstra[doc_id] = cepstrum
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search by cepstral distance."""
        query_cepstrum = self._compute_cepstrum(query_embedding)
        
        results = []
        
        for doc_id, doc_cepstrum in self.doc_cepstra.items():
            # Euclidean distance in cepstral space
            distance = np.linalg.norm(query_cepstrum - doc_cepstrum)
            results.append((doc_id, distance))
        
        results.sort(key=lambda x: x[1])
        return results[:top_k]


# =============================================================================
# 7. EXPANDER-ENHANCED SIMILARITY
# =============================================================================

class ExpanderSimilarity:
    """
    Apply "expander" (dynamic range expansion) before comparison.
    
    In audio: Expander increases difference between loud and quiet parts.
    
    For embeddings: Emphasize the extreme values, suppress the middle.
    This might make the "signature" dimensions more prominent.
    """
    
    def __init__(self, 
                 threshold: float = 0.5,
                 ratio: float = 3.0,
                 soft_knee: bool = True):
        """
        Args:
            threshold: Values above this (absolute) are expanded
            ratio: Expansion ratio (>1 means more expansion)
            soft_knee: Smooth transition vs hard knee
        """
        self.threshold = threshold
        self.ratio = ratio
        self.soft_knee = soft_knee
        
        self.doc_expanded = {}
    
    def _expand(self, embedding: np.ndarray) -> np.ndarray:
        """Apply expansion to embedding."""
        expanded = embedding.copy()
        
        abs_vals = np.abs(embedding)
        
        if self.soft_knee:
            # Soft knee: smooth transition
            scale = np.where(
                abs_vals > self.threshold,
                1 + (self.ratio - 1) * (abs_vals - self.threshold) / (1 - self.threshold + 1e-10),
                1.0
            )
        else:
            # Hard knee
            scale = np.where(abs_vals > self.threshold, self.ratio, 1.0)
        
        expanded = embedding * scale
        
        # Normalize
        expanded = expanded / (np.linalg.norm(expanded) + 1e-10)
        
        return expanded
    
    def add(self, doc_id: str, embedding: np.ndarray):
        """Add document."""
        expanded = self._expand(embedding)
        self.doc_expanded[doc_id] = expanded
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using expanded embeddings."""
        query_expanded = self._expand(query_embedding)
        
        results = []
        
        for doc_id, doc_expanded in self.doc_expanded.items():
            similarity = np.dot(query_expanded, doc_expanded)
            results.append((doc_id, -similarity))
        
        results.sort(key=lambda x: x[1])
        return results[:top_k]


# =============================================================================
# 8. BAND-PASS RETRIEVAL (Different frequency bands, different indices)
# =============================================================================

class BandPassRetrieval:
    """
    Split embedding into frequency bands and search each independently.
    
    Like an audio equalizer: Bass, Mids, Highs.
    
    Different bands might have different "semantic weight".
    """
    
    def __init__(self, 
                 num_bands: int = 8,
                 band_weights: List[float] = None):
        """
        Args:
            num_bands: Number of frequency bands
            band_weights: Weight for each band (higher = more important)
        """
        self.num_bands = num_bands
        self.band_weights = band_weights or [1.0] * num_bands
        
        self.doc_bands = {}
    
    def _split_bands(self, embedding: np.ndarray) -> List[np.ndarray]:
        """Split embedding into frequency bands via FFT."""
        # FFT
        spectrum = np.fft.fft(embedding)
        n = len(spectrum)
        
        bands = []
        band_size = n // self.num_bands
        
        for i in range(self.num_bands):
            start = i * band_size
            end = (i + 1) * band_size if i < self.num_bands - 1 else n
            
            # Create band-limited spectrum
            band_spectrum = np.zeros(n, dtype=complex)
            band_spectrum[start:end] = spectrum[start:end]
            
            # Also include negative frequencies (for real signal)
            if end <= n // 2:
                band_spectrum[-(end):-(start) if start > 0 else None] = spectrum[-(end):-(start) if start > 0 else None]
            
            # IFFT to get band signal
            band_signal = np.real(np.fft.ifft(band_spectrum))
            bands.append(band_signal)
        
        return bands
    
    def add(self, doc_id: str, embedding: np.ndarray):
        """Add document."""
        bands = self._split_bands(embedding)
        self.doc_bands[doc_id] = bands
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search with weighted band similarity."""
        query_bands = self._split_bands(query_embedding)
        
        results = []
        
        for doc_id, doc_bands in self.doc_bands.items():
            total_similarity = 0.0
            
            for i, (q_band, d_band) in enumerate(zip(query_bands, doc_bands)):
                # Normalize bands
                q_norm = q_band / (np.linalg.norm(q_band) + 1e-10)
                d_norm = d_band / (np.linalg.norm(d_band) + 1e-10)
                
                band_similarity = np.dot(q_norm, d_norm)
                total_similarity += self.band_weights[i] * band_similarity
            
            results.append((doc_id, -total_similarity))
        
        results.sort(key=lambda x: x[1])
        return results[:top_k]


# =============================================================================
# BENCHMARK
# =============================================================================

def benchmark_all_methods():
    """Compare all audio-inspired methods."""
    
    print("=" * 70)
    print("AUDIO-INSPIRED RETRIEVAL BENCHMARK")
    print("=" * 70)
    
    # Create synthetic clustered data
    np.random.seed(42)
    dim = 1024
    n_docs = 1000
    n_queries = 100
    n_clusters = 10
    
    print(f"\nData: {n_docs} docs, {n_queries} queries, {n_clusters} clusters, {dim}D")
    
    # Cluster centers
    centers = np.random.randn(n_clusters, dim)
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    
    # Documents
    doc_embeddings = []
    doc_clusters = []
    for i in range(n_docs):
        cluster = i % n_clusters
        noise = np.random.randn(dim) * 0.3
        emb = centers[cluster] + noise
        emb = emb / np.linalg.norm(emb)
        doc_embeddings.append(emb)
        doc_clusters.append(cluster)
    
    doc_embeddings = np.array(doc_embeddings)
    doc_ids = [str(i) for i in range(n_docs)]
    
    # Queries
    query_embeddings = []
    query_clusters = []
    for i in range(n_queries):
        cluster = np.random.randint(n_clusters)
        noise = np.random.randn(dim) * 0.3
        emb = centers[cluster] + noise
        emb = emb / np.linalg.norm(emb)
        query_embeddings.append(emb)
        query_clusters.append(cluster)
    
    query_embeddings = np.array(query_embeddings)
    
    # Methods to test
    methods = {
        "Brute Force (Cosine)": None,  # Baseline
        "Peak Fingerprint (Single)": PeakFingerprint(num_peaks=64, peak_pairs=False),
        "Peak Fingerprint (Pairs)": PeakFingerprint(num_peaks=32, peak_pairs=True),
        "Spectral Correlation": SpectralCorrelation(use_magnitude_only=False),
        "Spectral Magnitude": SpectralCorrelation(use_magnitude_only=True),
        "Resonance Filter": ResonanceFilter(num_filters=32, filter_width=64),
        "Interference": InterferencePattern(normalize=True),
        "Harmonic": HarmonicFingerprint(max_harmonic=4, num_fundamentals=128),
        "Cepstral": CepstralMatcher(num_coefficients=64),
        "Expander": ExpanderSimilarity(threshold=0.3, ratio=2.0),
        "Band-Pass": BandPassRetrieval(num_bands=8),
    }
    
    # Build indices
    print("\nBuilding indices...")
    
    for name, method in methods.items():
        if method is not None:
            start = time.time()
            for doc_id, emb in zip(doc_ids, doc_embeddings):
                method.add(doc_id, emb)
            print(f"  {name}: {time.time() - start:.2f}s")
    
    # Evaluate
    print("\nEvaluating...")
    
    results = {}
    
    for name, method in methods.items():
        hits_at_10 = 0
        hits_at_1 = 0
        total_time = 0
        
        for i in range(n_queries):
            query_emb = query_embeddings[i]
            query_cluster = query_clusters[i]
            
            # Relevant docs = same cluster
            relevant = set(str(j) for j in range(n_docs) if doc_clusters[j] == query_cluster)
            
            start = time.time()
            
            if method is None:
                # Brute force baseline
                similarities = doc_embeddings @ query_emb
                top_indices = np.argsort(similarities)[::-1][:10]
                retrieved = [doc_ids[idx] for idx in top_indices]
            else:
                search_results = method.search(query_emb, top_k=10)
                retrieved = [doc_id for doc_id, _ in search_results]
            
            total_time += time.time() - start
            
            # Count hits
            if retrieved:
                hits_at_10 += len(set(retrieved[:10]) & relevant)
                hits_at_1 += 1 if retrieved[0] in relevant else 0
        
        # Calculate metrics
        total_relevant = n_queries * (n_docs // n_clusters)
        recall_10 = hits_at_10 / total_relevant
        precision_1 = hits_at_1 / n_queries
        avg_time = total_time / n_queries * 1000
        
        results[name] = {
            "recall@10": recall_10,
            "precision@1": precision_1,
            "avg_time_ms": avg_time,
        }
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\n{'Method':<30} {'Recall@10':>12} {'P@1':>12} {'Time (ms)':>12}")
    print("-" * 70)
    
    # Sort by recall
    sorted_methods = sorted(results.items(), key=lambda x: -x[1]["recall@10"])
    
    for name, res in sorted_methods:
        recall = res["recall@10"] * 100
        precision = res["precision@1"] * 100
        time_ms = res["avg_time_ms"]
        print(f"{name:<30} {recall:>11.1f}% {precision:>11.1f}% {time_ms:>11.2f}")
    
    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    baseline_recall = results["Brute Force (Cosine)"]["recall@10"]
    
    print(f"\nRelative to Brute Force Cosine:")
    for name, res in sorted_methods:
        if name != "Brute Force (Cosine)":
            relative = res["recall@10"] / baseline_recall * 100
            print(f"  {name}: {relative:.1f}%")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    benchmark_all_methods()
