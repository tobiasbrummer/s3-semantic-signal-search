#!/usr/bin/env python3
"""
RESONANCE-BASED SEMANTIC RETRIEVAL

Thinking like audio engineers:

1. RESONANZ: Wenn Query und Dokument in denselben Frequenzbändern 
   "schwingen", verstärken sie sich gegenseitig.

2. AUSLÖSCHUNG: Gegenphasige Signale löschen sich aus.
   → Hohe Auslöschung = NICHT ähnlich

3. INTERFERENZ-ENERGIE: 
   E(combined) = E(query) + E(doc) + 2*Interference
   → Hohe kombinierte Energie = konstruktive Interferenz = ähnlich

4. SPEKTRALE SIGNATUR:
   Nicht einzelne Peaks, sondern Band-Energien.
   "Bass-lastig", "Mid-heavy", etc.

5. Q-FAKTOR (Güte):
   Schmalbandige Resonanz = spezifischer Match
   Breitbandige Resonanz = allgemeiner Match

Author: Claude & Toby (Audio Engineers Mode)
Date: December 2024
"""

import numpy as np
from typing import List, Dict, Tuple, Set
from collections import defaultdict
from dataclasses import dataclass
import time


# =============================================================================
# AUDIO ENGINEERING TOOLKIT
# =============================================================================

class AudioToolkit:
    """
    Audio engineering tools für Embedding-Analyse.
    """
    
    @staticmethod
    def compute_band_energies(signal: np.ndarray, num_bands: int = 32) -> np.ndarray:
        """
        Compute energy in frequency bands.
        
        Like a spectrum analyzer: Split signal into bands, measure energy per band.
        """
        band_size = len(signal) // num_bands
        energies = np.zeros(num_bands)
        
        for i in range(num_bands):
            start = i * band_size
            end = start + band_size if i < num_bands - 1 else len(signal)
            band = signal[start:end]
            energies[i] = np.sum(band ** 2)  # Energy = sum of squares
        
        return energies
    
    @staticmethod
    def compute_interference_energy(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
        """
        Compute interference energy when two signals are combined.
        
        E(a+b) = E(a) + E(b) + 2*dot(a,b)
        
        The interference term 2*dot(a,b) tells us:
        - Positive: Constructive interference (signals reinforce)
        - Negative: Destructive interference (signals cancel)
        """
        return 2 * np.dot(sig_a, sig_b)
    
    @staticmethod
    def compute_resonance_profile(sig_a: np.ndarray, sig_b: np.ndarray, num_bands: int = 32) -> np.ndarray:
        """
        Compute per-band resonance.
        
        Where do the signals resonate (reinforce each other)?
        Where do they cancel?
        """
        band_size = len(sig_a) // num_bands
        resonance = np.zeros(num_bands)
        
        for i in range(num_bands):
            start = i * band_size
            end = start + band_size if i < num_bands - 1 else len(sig_a)
            
            band_a = sig_a[start:end]
            band_b = sig_b[start:end]
            
            # Resonance = correlation in this band
            resonance[i] = np.dot(band_a, band_b)
        
        return resonance
    
    @staticmethod
    def spectral_centroid(energies: np.ndarray) -> float:
        """
        Spectral centroid: "Center of mass" of the spectrum.
        
        Bright sounds have high centroid, dark sounds have low centroid.
        """
        total = np.sum(energies)
        if total < 1e-10:
            return len(energies) / 2
        
        frequencies = np.arange(len(energies))
        return np.sum(frequencies * energies) / total
    
    @staticmethod
    def spectral_spread(energies: np.ndarray, centroid: float = None) -> float:
        """
        Spectral spread: How "wide" is the spectrum?
        
        Narrow = focused on specific frequencies
        Wide = broadband noise
        """
        if centroid is None:
            centroid = AudioToolkit.spectral_centroid(energies)
        
        total = np.sum(energies)
        if total < 1e-10:
            return 0
        
        frequencies = np.arange(len(energies))
        variance = np.sum(((frequencies - centroid) ** 2) * energies) / total
        return np.sqrt(variance)
    
    @staticmethod
    def spectral_flatness(energies: np.ndarray) -> float:
        """
        Spectral flatness: Tonality vs Noise.
        
        1.0 = White noise (flat spectrum)
        0.0 = Pure tone (single peak)
        
        Geometric mean / Arithmetic mean
        """
        energies = np.maximum(energies, 1e-10)  # Avoid log(0)
        
        geometric_mean = np.exp(np.mean(np.log(energies)))
        arithmetic_mean = np.mean(energies)
        
        return geometric_mean / arithmetic_mean


# =============================================================================
# RESONANCE INDEX
# =============================================================================

class ResonanceIndex:
    """
    Index based on resonance patterns.
    
    Key insight: Ähnliche Dokumente haben ähnliche "aktive Bänder".
    
    Statt Peak-Positionen (Shazam) nutzen wir Band-Energie-Signaturen.
    """
    
    def __init__(self,
                 num_bands: int = 32,
                 energy_threshold_percentile: float = 75,
                 use_phase: bool = True):
        """
        Args:
            num_bands: Number of frequency bands
            energy_threshold_percentile: Bands above this percentile are "active"
            use_phase: Also consider sign (phase) of dominant dimension per band
        """
        self.num_bands = num_bands
        self.energy_threshold_percentile = energy_threshold_percentile
        self.use_phase = use_phase
        
        # Storage
        self.doc_ids = []
        self.doc_embeddings = None
        self.doc_band_energies = None
        self.doc_band_phases = None
        self.doc_signatures = None  # Binary: which bands are active
        
        # Inverted index: active_band → [doc_indices]
        self.band_index = defaultdict(list)
    
    def _compute_signature(self, embedding: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute band signature for an embedding.
        
        Returns:
            energies: Energy per band
            phases: Dominant phase per band (+1 or -1)
            active: Boolean mask of active bands
        """
        band_size = len(embedding) // self.num_bands
        energies = np.zeros(self.num_bands)
        phases = np.zeros(self.num_bands)
        
        for i in range(self.num_bands):
            start = i * band_size
            end = start + band_size if i < self.num_bands - 1 else len(embedding)
            band = embedding[start:end]
            
            # Energy
            energies[i] = np.sum(band ** 2)
            
            # Phase: Sign of the maximum absolute value
            max_idx = np.argmax(np.abs(band))
            phases[i] = 1 if band[max_idx] > 0 else -1
        
        # Active bands: Above threshold
        threshold = np.percentile(energies, self.energy_threshold_percentile)
        active = energies > threshold
        
        return energies, phases, active
    
    def build(self, doc_ids: List[str], embeddings: np.ndarray):
        """Build the resonance index."""
        self.doc_ids = doc_ids
        self.doc_embeddings = embeddings
        
        n_docs = len(doc_ids)
        
        self.doc_band_energies = np.zeros((n_docs, self.num_bands))
        self.doc_band_phases = np.zeros((n_docs, self.num_bands))
        self.doc_signatures = np.zeros((n_docs, self.num_bands), dtype=bool)
        
        for i, emb in enumerate(embeddings):
            energies, phases, active = self._compute_signature(emb)
            
            self.doc_band_energies[i] = energies
            self.doc_band_phases[i] = phases
            self.doc_signatures[i] = active
            
            # Add to inverted index
            for band_idx in np.where(active)[0]:
                if self.use_phase:
                    key = (int(band_idx), int(phases[band_idx]))
                else:
                    key = int(band_idx)
                self.band_index[key].append(i)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10, 
               candidate_multiplier: int = 20) -> List[Tuple[str, float]]:
        """
        Resonance-based search.
        
        1. Find documents with overlapping active bands (resonance candidates)
        2. Compute interference energy for fine ranking
        """
        query_energies, query_phases, query_active = self._compute_signature(query_embedding)
        
        # Stage 1: Find resonance candidates
        votes = defaultdict(int)
        
        for band_idx in np.where(query_active)[0]:
            if self.use_phase:
                key = (int(band_idx), int(query_phases[band_idx]))
            else:
                key = int(band_idx)
            
            for doc_idx in self.band_index.get(key, []):
                votes[doc_idx] += 1
        
        # Get top candidates
        num_candidates = min(top_k * candidate_multiplier, len(self.doc_ids))
        
        if votes:
            candidate_indices = sorted(votes.keys(), key=lambda x: -votes[x])[:num_candidates]
        else:
            # Fallback: random candidates
            candidate_indices = list(range(min(num_candidates, len(self.doc_ids))))
        
        # Stage 2: Compute interference energy for fine ranking
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        
        results = []
        
        for doc_idx in candidate_indices:
            doc_emb = self.doc_embeddings[doc_idx]
            doc_norm = doc_emb / (np.linalg.norm(doc_emb) + 1e-10)
            
            # Interference energy (= 2 * cosine for normalized vectors)
            interference = AudioToolkit.compute_interference_energy(query_norm, doc_norm)
            
            results.append((self.doc_ids[doc_idx], -interference))
        
        results.sort(key=lambda x: x[1])
        return results[:top_k]


# =============================================================================
# HARMONIC RESONANCE INDEX
# =============================================================================

class HarmonicResonanceIndex:
    """
    Use harmonic relationships for indexing.
    
    In music: Notes that are harmonically related sound "good together".
    E.g., C and G (perfect fifth) resonate.
    
    For embeddings: Dimensions that are "harmonically related" 
    (e.g., d and 2d, d and 3d) should be considered together.
    """
    
    def __init__(self,
                 num_fundamentals: int = 64,
                 harmonics: List[int] = [2, 3, 4, 5],
                 top_k_fundamentals: int = 8):
        """
        Args:
            num_fundamentals: Consider first N dimensions as potential fundamentals
            harmonics: Which harmonic ratios to check
            top_k_fundamentals: How many fundamentals to use for signature
        """
        self.num_fundamentals = num_fundamentals
        self.harmonics = harmonics
        self.top_k_fundamentals = top_k_fundamentals
        
        self.doc_ids = []
        self.doc_embeddings = None
        self.doc_harmonic_signatures = {}
        
        # Inverted index: (fundamental, harmonic_pattern) → [doc_indices]
        self.harmonic_index = defaultdict(list)
    
    def _compute_harmonic_signature(self, embedding: np.ndarray) -> List[Tuple[int, Tuple]]:
        """
        Find dominant fundamentals and their harmonic patterns.
        
        Returns: List of (fundamental_dim, harmonic_pattern)
        """
        dim = len(embedding)
        
        # Score each potential fundamental by harmonic coherence
        fundamental_scores = []
        
        for f in range(1, self.num_fundamentals + 1):
            # Fundamental value
            f_val = embedding[f - 1]
            
            # Check harmonics
            harmonic_pattern = []
            coherence = abs(f_val)
            
            for h in self.harmonics:
                h_idx = f * h - 1
                if h_idx < dim:
                    h_val = embedding[h_idx]
                    # Coherent if same sign and significant
                    if f_val * h_val > 0:  # Same sign
                        coherence += abs(h_val) / h  # Weight by harmonic number
                        harmonic_pattern.append(1)
                    else:
                        harmonic_pattern.append(0)
                else:
                    harmonic_pattern.append(-1)  # Out of range
            
            fundamental_scores.append((f, coherence, tuple(harmonic_pattern)))
        
        # Get top fundamentals
        fundamental_scores.sort(key=lambda x: -x[1])
        
        signature = []
        for f, score, pattern in fundamental_scores[:self.top_k_fundamentals]:
            signature.append((f, pattern))
        
        return signature
    
    def build(self, doc_ids: List[str], embeddings: np.ndarray):
        """Build the harmonic index."""
        self.doc_ids = doc_ids
        self.doc_embeddings = embeddings
        
        for i, emb in enumerate(embeddings):
            signature = self._compute_harmonic_signature(emb)
            self.doc_harmonic_signatures[doc_ids[i]] = signature
            
            for fundamental, pattern in signature:
                key = (fundamental, pattern)
                self.harmonic_index[key].append(i)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10,
               candidate_multiplier: int = 20) -> List[Tuple[str, float]]:
        """Search by harmonic pattern matching."""
        query_signature = self._compute_harmonic_signature(query_embedding)
        
        # Vote counting
        votes = defaultdict(int)
        
        for fundamental, pattern in query_signature:
            key = (fundamental, pattern)
            for doc_idx in self.harmonic_index.get(key, []):
                votes[doc_idx] += 1
        
        # Get candidates
        num_candidates = min(top_k * candidate_multiplier, len(self.doc_ids))
        
        if votes:
            candidate_indices = sorted(votes.keys(), key=lambda x: -votes[x])[:num_candidates]
        else:
            candidate_indices = list(range(min(num_candidates, len(self.doc_ids))))
        
        # Fine ranking with cosine
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        
        results = []
        for doc_idx in candidate_indices:
            doc_emb = self.doc_embeddings[doc_idx]
            doc_norm = doc_emb / (np.linalg.norm(doc_emb) + 1e-10)
            sim = np.dot(query_norm, doc_norm)
            results.append((self.doc_ids[doc_idx], -sim))
        
        results.sort(key=lambda x: x[1])
        return results[:top_k]


# =============================================================================
# SPECTRAL CENTROID INDEX
# =============================================================================

class SpectralCentroidIndex:
    """
    Index by spectral characteristics (centroid, spread, flatness).
    
    Intuition: 
    - "Bright" embeddings have high centroid (energy in high dimensions)
    - "Dark" embeddings have low centroid (energy in low dimensions)
    
    Similar semantic content might have similar "brightness".
    """
    
    def __init__(self,
                 num_bands: int = 32,
                 centroid_buckets: int = 16,
                 spread_buckets: int = 8):
        """
        Args:
            num_bands: For band energy computation
            centroid_buckets: Quantization levels for centroid
            spread_buckets: Quantization levels for spread
        """
        self.num_bands = num_bands
        self.centroid_buckets = centroid_buckets
        self.spread_buckets = spread_buckets
        
        self.doc_ids = []
        self.doc_embeddings = None
        self.doc_features = {}  # doc_id → (centroid_bucket, spread_bucket, flatness)
        
        # Inverted index
        self.centroid_index = defaultdict(list)  # centroid_bucket → [doc_indices]
    
    def _compute_spectral_features(self, embedding: np.ndarray) -> Tuple[float, float, float]:
        """Compute spectral features."""
        energies = AudioToolkit.compute_band_energies(embedding, self.num_bands)
        
        centroid = AudioToolkit.spectral_centroid(energies)
        spread = AudioToolkit.spectral_spread(energies, centroid)
        flatness = AudioToolkit.spectral_flatness(energies)
        
        return centroid, spread, flatness
    
    def build(self, doc_ids: List[str], embeddings: np.ndarray):
        """Build the spectral index."""
        self.doc_ids = doc_ids
        self.doc_embeddings = embeddings
        
        # First pass: compute all features to find ranges
        all_centroids = []
        all_spreads = []
        
        for emb in embeddings:
            c, s, f = self._compute_spectral_features(emb)
            all_centroids.append(c)
            all_spreads.append(s)
        
        self.centroid_min = min(all_centroids)
        self.centroid_max = max(all_centroids)
        self.spread_min = min(all_spreads)
        self.spread_max = max(all_spreads)
        
        # Second pass: index
        for i, emb in enumerate(embeddings):
            c, s, f = self._compute_spectral_features(emb)
            
            # Quantize
            c_bucket = int((c - self.centroid_min) / (self.centroid_max - self.centroid_min + 1e-10) * self.centroid_buckets)
            c_bucket = min(c_bucket, self.centroid_buckets - 1)
            
            s_bucket = int((s - self.spread_min) / (self.spread_max - self.spread_min + 1e-10) * self.spread_buckets)
            s_bucket = min(s_bucket, self.spread_buckets - 1)
            
            self.doc_features[doc_ids[i]] = (c_bucket, s_bucket, f)
            
            # Index by centroid (and neighboring buckets)
            for offset in [-1, 0, 1]:
                bucket = c_bucket + offset
                if 0 <= bucket < self.centroid_buckets:
                    self.centroid_index[bucket].append(i)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10,
               candidate_multiplier: int = 20) -> List[Tuple[str, float]]:
        """Search by spectral similarity."""
        c, s, f = self._compute_spectral_features(query_embedding)
        
        # Quantize query
        c_bucket = int((c - self.centroid_min) / (self.centroid_max - self.centroid_min + 1e-10) * self.centroid_buckets)
        c_bucket = min(max(c_bucket, 0), self.centroid_buckets - 1)
        
        # Get candidates from same and neighboring centroid buckets
        candidates = set()
        for offset in [-2, -1, 0, 1, 2]:
            bucket = c_bucket + offset
            if 0 <= bucket < self.centroid_buckets:
                candidates.update(self.centroid_index.get(bucket, []))
        
        num_candidates = min(top_k * candidate_multiplier, len(self.doc_ids))
        candidate_list = list(candidates)[:num_candidates]
        
        if len(candidate_list) < num_candidates:
            # Add random docs if not enough
            remaining = set(range(len(self.doc_ids))) - candidates
            candidate_list.extend(list(remaining)[:num_candidates - len(candidate_list)])
        
        # Fine ranking
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        
        results = []
        for doc_idx in candidate_list:
            doc_emb = self.doc_embeddings[doc_idx]
            doc_norm = doc_emb / (np.linalg.norm(doc_emb) + 1e-10)
            sim = np.dot(query_norm, doc_norm)
            results.append((self.doc_ids[doc_idx], -sim))
        
        results.sort(key=lambda x: x[1])
        return results[:top_k]


# =============================================================================
# CONSTRUCTIVE INTERFERENCE SEARCH
# =============================================================================

class ConstructiveInterferenceSearch:
    """
    Direct interference-based search.
    
    The idea: For each query, find documents where combining them
    produces MAXIMUM constructive interference.
    
    This is mathematically equivalent to cosine similarity,
    but the framing helps us think about it differently.
    """
    
    def __init__(self, 
                 num_bands: int = 32,
                 use_band_weighting: bool = True):
        """
        Args:
            num_bands: Number of frequency bands for analysis
            use_band_weighting: Weight bands by query energy
        """
        self.num_bands = num_bands
        self.use_band_weighting = use_band_weighting
        
        self.doc_ids = []
        self.doc_embeddings = None
        self.doc_band_embeddings = None  # Pre-split into bands
    
    def build(self, doc_ids: List[str], embeddings: np.ndarray):
        """Build index."""
        self.doc_ids = doc_ids
        self.doc_embeddings = embeddings
        
        # Pre-compute band splits for faster search
        n_docs, dim = embeddings.shape
        band_size = dim // self.num_bands
        
        self.doc_band_embeddings = embeddings.reshape(n_docs, self.num_bands, band_size)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search by constructive interference.
        
        For each band, compute interference and weight by query band energy.
        """
        dim = len(query_embedding)
        band_size = dim // self.num_bands
        
        query_bands = query_embedding.reshape(self.num_bands, band_size)
        
        # Compute query band energies for weighting
        if self.use_band_weighting:
            query_band_energies = np.sum(query_bands ** 2, axis=1)
            query_band_energies = query_band_energies / (np.sum(query_band_energies) + 1e-10)
        else:
            query_band_energies = np.ones(self.num_bands) / self.num_bands
        
        # Compute interference for all docs
        # interference[d, b] = 2 * dot(query_band[b], doc_band[d, b])
        
        # Shape: (n_docs, num_bands)
        band_interferences = 2 * np.einsum('nb,dnb->dn', query_bands, self.doc_band_embeddings)
        
        # Weight by query band energy
        weighted_interference = band_interferences @ query_band_energies
        
        # Get top-k
        top_indices = np.argsort(weighted_interference)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((self.doc_ids[idx], -weighted_interference[idx]))
        
        return results


# =============================================================================
# COMBINED AUDIO INDEX
# =============================================================================

class AudioSemanticIndex:
    """
    Combines multiple audio-inspired approaches.
    
    Uses a voting system where different indices propose candidates,
    then final ranking is done by interference energy.
    """
    
    def __init__(self,
                 num_bands: int = 32,
                 use_resonance: bool = True,
                 use_harmonic: bool = True,
                 use_spectral: bool = True):
        
        self.num_bands = num_bands
        
        self.indices = {}
        
        if use_resonance:
            self.indices["resonance"] = ResonanceIndex(num_bands=num_bands)
        if use_harmonic:
            self.indices["harmonic"] = HarmonicResonanceIndex()
        if use_spectral:
            self.indices["spectral"] = SpectralCentroidIndex(num_bands=num_bands)
        
        self.doc_ids = []
        self.doc_embeddings = None
    
    def build(self, doc_ids: List[str], embeddings: np.ndarray):
        """Build all indices."""
        self.doc_ids = doc_ids
        self.doc_embeddings = embeddings
        
        for name, index in self.indices.items():
            index.build(doc_ids, embeddings)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10,
               candidates_per_index: int = 100) -> List[Tuple[str, float]]:
        """
        Search using all indices and combine results.
        """
        # Gather candidates from all indices
        all_candidates = set()
        
        for name, index in self.indices.items():
            results = index.search(query_embedding, top_k=candidates_per_index)
            for doc_id, _ in results:
                doc_idx = self.doc_ids.index(doc_id)
                all_candidates.add(doc_idx)
        
        # Fine ranking with interference
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        
        results = []
        for doc_idx in all_candidates:
            doc_emb = self.doc_embeddings[doc_idx]
            doc_norm = doc_emb / (np.linalg.norm(doc_emb) + 1e-10)
            
            interference = AudioToolkit.compute_interference_energy(query_norm, doc_norm)
            results.append((self.doc_ids[doc_idx], -interference))
        
        results.sort(key=lambda x: x[1])
        return results[:top_k]


# =============================================================================
# BENCHMARK
# =============================================================================

def benchmark_resonance_methods():
    """Benchmark all resonance-based methods."""
    
    print("=" * 70)
    print("RESONANCE-BASED RETRIEVAL BENCHMARK")
    print("=" * 70)
    
    # Create clustered data
    np.random.seed(42)
    dim = 1024
    n_docs = 3000
    n_queries = 150
    n_clusters = 15
    noise_level = 0.4
    
    print(f"\nData: {n_docs} docs, {n_queries} queries, {n_clusters} clusters")
    print(f"Noise level: {noise_level}")
    
    # Cluster centers
    centers = np.random.randn(n_clusters, dim)
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    
    # Documents
    doc_embeddings = []
    doc_clusters = []
    for i in range(n_docs):
        cluster = i % n_clusters
        noise = np.random.randn(dim) * noise_level
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
        noise = np.random.randn(dim) * noise_level
        emb = centers[cluster] + noise
        emb = emb / np.linalg.norm(emb)
        query_embeddings.append(emb)
        query_clusters.append(cluster)
    
    query_embeddings = np.array(query_embeddings)
    
    # Methods to test
    methods = {
        "Brute Force (Baseline)": None,
        "Resonance Index": ResonanceIndex(num_bands=32),
        "Resonance (64 bands)": ResonanceIndex(num_bands=64),
        "Resonance (no phase)": ResonanceIndex(num_bands=32, use_phase=False),
        "Harmonic Resonance": HarmonicResonanceIndex(),
        "Spectral Centroid": SpectralCentroidIndex(num_bands=32),
        "Constructive Interference": ConstructiveInterferenceSearch(num_bands=32),
        "Combined Audio": AudioSemanticIndex(num_bands=32),
    }
    
    # Build indices
    print("\nBuilding indices...")
    
    for name, method in methods.items():
        if method is not None:
            start = time.time()
            method.build(doc_ids, doc_embeddings)
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
                # Brute force
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
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    baseline_recall = results["Brute Force (Baseline)"]["recall"]
    baseline_time = results["Brute Force (Baseline)"]["time"]
    
    print(f"\n{'Method':<30} {'Recall@10':>12} {'Time (ms)':>12} {'vs Base':>10} {'Speedup':>10}")
    print("-" * 80)
    
    for name, res in sorted(results.items(), key=lambda x: -x[1]["recall"]):
        relative = res["recall"] / baseline_recall * 100 if baseline_recall > 0 else 0
        speedup = baseline_time / res["time"] if res["time"] > 0 else 0
        print(f"{name:<30} {res['recall']:>11.1f}% {res['time']:>11.2f} {relative:>9.1f}% {speedup:>9.1f}x")
    
    # Analysis
    print("\n" + "=" * 70)
    print("AUDIO ENGINEERING ANALYSIS")
    print("=" * 70)
    
    # Test interference on a few queries
    print("\nInterference Analysis (Sample Queries):")
    
    for i in range(3):
        query_emb = query_embeddings[i]
        query_cluster = query_clusters[i]
        
        # Find a relevant and irrelevant doc
        relevant_idx = next(j for j in range(n_docs) if doc_clusters[j] == query_cluster)
        irrelevant_idx = next(j for j in range(n_docs) if doc_clusters[j] != query_cluster)
        
        rel_emb = doc_embeddings[relevant_idx]
        irr_emb = doc_embeddings[irrelevant_idx]
        
        # Compute interference
        rel_interference = AudioToolkit.compute_interference_energy(query_emb, rel_emb)
        irr_interference = AudioToolkit.compute_interference_energy(query_emb, irr_emb)
        
        # Compute resonance profiles
        rel_resonance = AudioToolkit.compute_resonance_profile(query_emb, rel_emb, 8)
        irr_resonance = AudioToolkit.compute_resonance_profile(query_emb, irr_emb, 8)
        
        print(f"\n  Query {i} (cluster {query_cluster}):")
        print(f"    Relevant doc: Interference={rel_interference:.3f}")
        print(f"    Irrelevant doc: Interference={irr_interference:.3f}")
        print(f"    Resonance diff: {np.sum(np.abs(rel_resonance)) - np.sum(np.abs(irr_resonance)):.3f}")
    
    return results


if __name__ == "__main__":
    benchmark_resonance_methods()
