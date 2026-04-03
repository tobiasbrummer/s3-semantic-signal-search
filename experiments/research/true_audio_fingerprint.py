#!/usr/bin/env python3
"""
TRUE AUDIO FINGERPRINTING FOR EMBEDDINGS

Radical idea: What if we ACTUALLY treat embeddings as audio
and apply real audio fingerprinting (Shazam-style)?

The process:
1. Embedding → Interpret as waveform
2. Waveform → Spectrogram (STFT/Wavelet)
3. Spectrogram → Constellation Map (peak detection)
4. Peak pairs → Hash → Inverted Index

This is "full circle" - using actual audio processing on embeddings.

Author: Claude & Toby  
Date: December 2024
"""

import numpy as np
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import time
import hashlib

try:
    from scipy.signal import stft, find_peaks
    from scipy.ndimage import maximum_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("⚠️  pip install scipy")


# =============================================================================
# SHAZAM-STYLE FINGERPRINTING FOR EMBEDDINGS
# =============================================================================

class ShazamFingerprint:
    """
    Apply Shazam's algorithm to embeddings.
    
    Shazam's key insights:
    1. Use SPECTROGRAM not raw waveform
    2. Find PEAKS (local maxima)
    3. Form PAIRS of peaks (anchor + target)
    4. Hash pairs → inverted index
    
    For embeddings:
    - Treat 1024D embedding as 1024-sample audio
    - Compute spectrogram (creates 2D time-frequency representation)
    - Find peaks in spectrogram
    - Create fingerprints from peak pairs
    """
    
    def __init__(self,
                 # STFT parameters
                 nperseg: int = 64,        # Window size for STFT
                 noverlap: int = 48,       # Overlap between windows
                 # Peak detection
                 peak_neighborhood: int = 10,  # Size of local neighborhood
                 peak_threshold: float = 0.1,  # Minimum peak amplitude
                 max_peaks: int = 50,      # Maximum peaks per spectrogram
                 # Pair formation
                 target_zone_t: Tuple[int, int] = (1, 5),   # Time range for targets
                 target_zone_f: Tuple[int, int] = (-10, 10), # Freq range for targets
                 max_pairs_per_anchor: int = 5):
        
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.peak_neighborhood = peak_neighborhood
        self.peak_threshold = peak_threshold
        self.max_peaks = max_peaks
        self.target_zone_t = target_zone_t
        self.target_zone_f = target_zone_f
        self.max_pairs_per_anchor = max_pairs_per_anchor
        
        # Index: hash → [(doc_id, anchor_time), ...]
        self.hash_index = defaultdict(list)
        self.doc_ids = []
        self.doc_embeddings = {}
    
    def _compute_spectrogram(self, embedding: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute spectrogram of embedding using STFT.
        
        Returns:
            f: frequency bins
            t: time bins  
            Sxx: spectrogram magnitude (freq × time)
        """
        if not HAS_SCIPY:
            # Fallback: simple reshape into 2D
            n_freq = 32
            n_time = len(embedding) // n_freq
            Sxx = np.abs(embedding[:n_freq * n_time].reshape(n_freq, n_time))
            f = np.arange(n_freq)
            t = np.arange(n_time)
            return f, t, Sxx
        
        # Real STFT
        f, t, Zxx = stft(embedding, nperseg=self.nperseg, noverlap=self.noverlap)
        Sxx = np.abs(Zxx)
        
        return f, t, Sxx
    
    def _find_peaks_2d(self, spectrogram: np.ndarray) -> List[Tuple[int, int, float]]:
        """
        Find peaks (local maxima) in 2D spectrogram.
        
        Returns:
            List of (freq_bin, time_bin, magnitude)
        """
        # Apply maximum filter to find local maxima
        neighborhood_size = self.peak_neighborhood
        local_max = maximum_filter(spectrogram, size=neighborhood_size)
        
        # Peaks are where the value equals the local maximum and above threshold
        max_val = np.max(spectrogram)
        threshold = self.peak_threshold * max_val
        
        is_peak = (spectrogram == local_max) & (spectrogram > threshold)
        
        # Get peak coordinates
        peak_freqs, peak_times = np.where(is_peak)
        peak_mags = spectrogram[peak_freqs, peak_times]
        
        # Sort by magnitude and take top N
        sorted_indices = np.argsort(peak_mags)[::-1][:self.max_peaks]
        
        peaks = []
        for idx in sorted_indices:
            peaks.append((int(peak_freqs[idx]), int(peak_times[idx]), float(peak_mags[idx])))
        
        return peaks
    
    def _form_peak_pairs(self, peaks: List[Tuple[int, int, float]]) -> List[Tuple[int, int, int, int]]:
        """
        Form pairs of peaks (anchor, target).
        
        For each anchor peak, find nearby target peaks within the target zone.
        
        Returns:
            List of (anchor_freq, anchor_time, target_freq, delta_time)
        """
        # Sort peaks by time
        peaks_by_time = sorted(peaks, key=lambda p: p[1])
        
        pairs = []
        
        for i, (f1, t1, m1) in enumerate(peaks_by_time):
            # Find targets in the target zone
            targets_found = 0
            
            for j, (f2, t2, m2) in enumerate(peaks_by_time[i+1:], i+1):
                dt = t2 - t1
                df = f2 - f1
                
                # Check if in target zone
                if (self.target_zone_t[0] <= dt <= self.target_zone_t[1] and
                    self.target_zone_f[0] <= df <= self.target_zone_f[1]):
                    
                    pairs.append((f1, t1, f2, dt))
                    targets_found += 1
                    
                    if targets_found >= self.max_pairs_per_anchor:
                        break
                
                # Stop if past target zone
                if dt > self.target_zone_t[1]:
                    break
        
        return pairs
    
    def _hash_pair(self, anchor_freq: int, target_freq: int, delta_time: int) -> int:
        """
        Create hash from peak pair.
        
        Hash = (anchor_freq, target_freq, delta_time)
        """
        # Simple hash: combine into single integer
        # This is similar to Shazam's approach
        hash_val = (anchor_freq << 20) | (target_freq << 10) | delta_time
        return hash_val
    
    def add(self, doc_id: str, embedding: np.ndarray):
        """Add document to index."""
        self.doc_ids.append(doc_id)
        self.doc_embeddings[doc_id] = embedding
        
        # Compute spectrogram
        f, t, Sxx = self._compute_spectrogram(embedding)
        
        # Find peaks
        peaks = self._find_peaks_2d(Sxx)
        
        # Form pairs and hash
        pairs = self._form_peak_pairs(peaks)
        
        for anchor_freq, anchor_time, target_freq, delta_time in pairs:
            hash_val = self._hash_pair(anchor_freq, target_freq, delta_time)
            self.hash_index[hash_val].append((doc_id, anchor_time))
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search using Shazam-style matching.
        
        1. Compute query fingerprints
        2. Look up each hash in index
        3. Vote counting with time alignment
        """
        # Compute query fingerprints
        f, t, Sxx = self._compute_spectrogram(query_embedding)
        peaks = self._find_peaks_2d(Sxx)
        pairs = self._form_peak_pairs(peaks)
        
        # Vote counting
        votes = defaultdict(int)
        
        for anchor_freq, anchor_time, target_freq, delta_time in pairs:
            hash_val = self._hash_pair(anchor_freq, target_freq, delta_time)
            
            for doc_id, doc_anchor_time in self.hash_index.get(hash_val, []):
                # Could do time-alignment verification here
                # For now, just count votes
                votes[doc_id] += 1
        
        # Sort by votes
        results = [(doc_id, -vote_count) for doc_id, vote_count in votes.items()]
        results.sort(key=lambda x: x[1])
        
        return results[:top_k]
    
    def search_brute_force(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """Baseline cosine similarity."""
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        
        results = []
        for doc_id, doc_emb in self.doc_embeddings.items():
            doc_norm = doc_emb / (np.linalg.norm(doc_emb) + 1e-10)
            sim = np.dot(query_norm, doc_norm)
            results.append((doc_id, -sim))
        
        results.sort(key=lambda x: x[1])
        return results[:top_k]


# =============================================================================
# WAVELET-BASED FINGERPRINTING
# =============================================================================

class WaveletFingerprint:
    """
    Use wavelet transform instead of STFT for multi-resolution analysis.
    
    Wavelets give better time-frequency localization than STFT.
    """
    
    def __init__(self, 
                 levels: int = 5,
                 top_coeffs_per_level: int = 16):
        self.levels = levels
        self.top_coeffs_per_level = top_coeffs_per_level
        
        self.doc_fingerprints = {}
        self.doc_embeddings = {}
    
    def _haar_wavelet_1d(self, signal: np.ndarray) -> List[np.ndarray]:
        """
        Simple Haar wavelet decomposition.
        
        Returns list of detail coefficients at each level.
        """
        coeffs = []
        current = signal.copy()
        
        for level in range(self.levels):
            if len(current) < 2:
                break
            
            # Downsample
            n = len(current) // 2
            
            # Approximation (low-pass)
            approx = (current[0::2] + current[1::2]) / np.sqrt(2)
            
            # Detail (high-pass)
            detail = (current[0::2] - current[1::2]) / np.sqrt(2)
            
            coeffs.append(detail)
            current = approx[:n]
        
        return coeffs
    
    def _extract_fingerprint(self, embedding: np.ndarray) -> Set[Tuple[int, int, int]]:
        """
        Extract fingerprint from wavelet coefficients.
        
        For each level, find top coefficients and record their position and sign.
        """
        coeffs = self._haar_wavelet_1d(embedding)
        
        fingerprint = set()
        
        for level, detail in enumerate(coeffs):
            if len(detail) == 0:
                continue
            
            # Find top coefficients by absolute value
            abs_detail = np.abs(detail)
            top_indices = np.argsort(abs_detail)[-self.top_coeffs_per_level:]
            
            for idx in top_indices:
                sign = 1 if detail[idx] > 0 else 0
                fingerprint.add((level, int(idx), sign))
        
        return fingerprint
    
    def add(self, doc_id: str, embedding: np.ndarray):
        """Add document."""
        self.doc_embeddings[doc_id] = embedding
        self.doc_fingerprints[doc_id] = self._extract_fingerprint(embedding)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search by fingerprint overlap."""
        query_fp = self._extract_fingerprint(query_embedding)
        
        results = []
        
        for doc_id, doc_fp in self.doc_fingerprints.items():
            # Jaccard similarity
            intersection = len(query_fp & doc_fp)
            union = len(query_fp | doc_fp)
            similarity = intersection / max(union, 1)
            
            results.append((doc_id, -similarity))
        
        results.sort(key=lambda x: x[1])
        return results[:top_k]


# =============================================================================
# CROSS-CORRELATION MATCHING
# =============================================================================

class CrossCorrelationMatcher:
    """
    Use cross-correlation for matching.
    
    In audio: Cross-correlation finds where two signals align best.
    
    For embeddings: The correlation peak value indicates similarity.
    """
    
    def __init__(self, use_fft: bool = True):
        self.use_fft = use_fft
        self.doc_embeddings = {}
        self.doc_ffts = {}
    
    def add(self, doc_id: str, embedding: np.ndarray):
        """Add document."""
        self.doc_embeddings[doc_id] = embedding
        if self.use_fft:
            self.doc_ffts[doc_id] = np.fft.fft(embedding)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search by cross-correlation peak.
        
        FFT-based correlation: corr = IFFT(FFT(a) * conj(FFT(b)))
        """
        if self.use_fft:
            query_fft = np.fft.fft(query_embedding)
        
        results = []
        
        for doc_id in self.doc_embeddings:
            if self.use_fft:
                # FFT-based correlation
                cross_spectrum = query_fft * np.conj(self.doc_ffts[doc_id])
                correlation = np.fft.ifft(cross_spectrum)
                peak_correlation = np.max(np.abs(correlation))
            else:
                # Direct correlation (slower)
                correlation = np.correlate(query_embedding, self.doc_embeddings[doc_id], mode='full')
                peak_correlation = np.max(np.abs(correlation))
            
            results.append((doc_id, -peak_correlation))
        
        results.sort(key=lambda x: x[1])
        return results[:top_k]


# =============================================================================
# BENCHMARK
# =============================================================================

def benchmark_audio_methods():
    """Compare true audio fingerprinting methods."""
    
    print("=" * 70)
    print("TRUE AUDIO FINGERPRINTING BENCHMARK")
    print("=" * 70)
    
    if not HAS_SCIPY:
        print("\n⚠️  scipy not installed, using fallback methods")
    
    # Create clustered data
    np.random.seed(42)
    dim = 1024
    n_docs = 2000
    n_queries = 100
    n_clusters = 10
    
    print(f"\nData: {n_docs} docs, {n_queries} queries, {dim}D")
    
    # Cluster centers
    centers = np.random.randn(n_clusters, dim)
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    
    # Documents
    doc_embeddings = []
    doc_clusters = []
    for i in range(n_docs):
        cluster = i % n_clusters
        noise = np.random.randn(dim) * 0.4
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
        noise = np.random.randn(dim) * 0.4
        emb = centers[cluster] + noise
        emb = emb / np.linalg.norm(emb)
        query_embeddings.append(emb)
        query_clusters.append(cluster)
    
    query_embeddings = np.array(query_embeddings)
    
    # Methods
    methods = {
        "Cosine (Baseline)": None,
        "Shazam-Style": ShazamFingerprint(nperseg=64, max_peaks=30),
        "Wavelet": WaveletFingerprint(levels=6, top_coeffs_per_level=20),
        "Cross-Correlation": CrossCorrelationMatcher(use_fft=True),
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
        recalls = []
        times = []
        
        for i in range(n_queries):
            query_emb = query_embeddings[i]
            relevant = set(str(j) for j in range(n_docs) if doc_clusters[j] == query_clusters[i])
            
            start = time.time()
            
            if method is None:
                # Baseline
                query_norm = query_emb / np.linalg.norm(query_emb)
                sims = doc_embeddings @ query_norm
                top_idx = np.argsort(sims)[::-1][:10]
                retrieved = [doc_ids[idx] for idx in top_idx]
            else:
                search_results = method.search(query_emb, top_k=10)
                retrieved = [doc_id for doc_id, _ in search_results]
            
            times.append(time.time() - start)
            
            if retrieved:
                hits = len(set(retrieved[:10]) & relevant)
                recalls.append(hits / min(len(relevant), 10))
            else:
                recalls.append(0)
        
        results[name] = {
            "recall": np.mean(recalls) * 100,
            "time": np.mean(times) * 1000,
        }
    
    # Print
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    baseline_recall = results["Cosine (Baseline)"]["recall"]
    
    print(f"\n{'Method':<25} {'Recall@10':>12} {'Time (ms)':>12} {'vs Baseline':>12}")
    print("-" * 65)
    
    for name, res in sorted(results.items(), key=lambda x: -x[1]["recall"]):
        relative = res["recall"] / baseline_recall * 100 if baseline_recall > 0 else 0
        print(f"{name:<25} {res['recall']:>11.1f}% {res['time']:>11.2f} {relative:>11.1f}%")
    
    return results


if __name__ == "__main__":
    benchmark_audio_methods()
