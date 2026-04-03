#!/usr/bin/env python3
"""
OPTIMIZED INTERFERENCE RETRIEVAL

Key insight from the benchmark:
- Constructive Interference mit Band-Weighting schlägt Brute Force!
- Aber es ist noch zu langsam.

Dieser Code kombiniert:
1. Band-Energy Signatures für schnellen Vorfilter
2. Constructive Interference für Fein-Ranking
3. Optimierte NumPy-Operationen

DAS AUDIO-MODELL:
================

Stell dir vor:
- Query = Ein Akkord den du spielst
- Documents = Instrumente die mitschwingen könnten

Ein gutes Match ist wenn:
- Die gleichen Frequenzbänder aktiv sind (Resonanz)
- Die Phasen übereinstimmen (konstruktive Interferenz)
- Die Energieverteilung ähnlich ist (ähnlicher "Sound")

Author: Claude & Toby (Audio Engineers)
Date: December 2024
"""

import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
from dataclasses import dataclass
import time


# =============================================================================
# BAND ENERGY SIGNATURE
# =============================================================================

@dataclass
class BandSignature:
    """
    Kompakte Signatur eines Embeddings basierend auf Band-Energien.
    
    Wie ein "spektraler Fingerabdruck".
    """
    energies: np.ndarray          # Energie pro Band
    active_mask: np.ndarray       # Welche Bänder sind "aktiv" (bool)
    phases: np.ndarray            # Dominante Phase pro Band (+1/-1)
    centroid: float               # Spektraler Schwerpunkt
    total_energy: float           # Gesamtenergie


class BandSignatureExtractor:
    """
    Extrahiert Band-Signaturen aus Embeddings.
    """
    
    def __init__(self, 
                 num_bands: int = 32,
                 active_threshold_percentile: float = 70):
        self.num_bands = num_bands
        self.active_threshold_percentile = active_threshold_percentile
    
    def extract(self, embedding: np.ndarray) -> BandSignature:
        """Extrahiere Signatur."""
        dim = len(embedding)
        band_size = dim // self.num_bands
        
        energies = np.zeros(self.num_bands)
        phases = np.zeros(self.num_bands)
        
        for i in range(self.num_bands):
            start = i * band_size
            end = start + band_size if i < self.num_bands - 1 else dim
            band = embedding[start:end]
            
            energies[i] = np.sum(band ** 2)
            max_idx = np.argmax(np.abs(band))
            phases[i] = 1.0 if band[max_idx] >= 0 else -1.0
        
        # Active mask
        threshold = np.percentile(energies, self.active_threshold_percentile)
        active_mask = energies > threshold
        
        # Centroid
        total = np.sum(energies)
        if total > 1e-10:
            centroid = np.sum(np.arange(self.num_bands) * energies) / total
        else:
            centroid = self.num_bands / 2
        
        return BandSignature(
            energies=energies,
            active_mask=active_mask,
            phases=phases,
            centroid=centroid,
            total_energy=total
        )
    
    def extract_batch(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Batch-Extraktion für Effizienz.
        
        Returns:
            energies: (N, num_bands)
            phases: (N, num_bands)
            active_masks: (N, num_bands) bool
        """
        n_docs, dim = embeddings.shape
        band_size = dim // self.num_bands
        
        # Reshape für Band-Berechnung
        # (n_docs, num_bands, band_size)
        reshaped = embeddings[:, :self.num_bands * band_size].reshape(n_docs, self.num_bands, band_size)
        
        # Energien pro Band
        energies = np.sum(reshaped ** 2, axis=2)  # (n_docs, num_bands)
        
        # Phasen: Sign des maximalen absoluten Werts pro Band
        max_indices = np.argmax(np.abs(reshaped), axis=2)  # (n_docs, num_bands)
        
        # Get the actual values at max indices
        batch_idx = np.arange(n_docs)[:, None]
        band_idx = np.arange(self.num_bands)[None, :]
        max_values = reshaped[batch_idx, band_idx, max_indices]
        phases = np.sign(max_values)
        phases[phases == 0] = 1  # Default to positive
        
        # Active masks
        thresholds = np.percentile(energies, self.active_threshold_percentile, axis=1, keepdims=True)
        active_masks = energies > thresholds
        
        return energies, phases, active_masks


# =============================================================================
# INTERFERENCE INDEX
# =============================================================================

class InterferenceIndex:
    """
    Schneller Index basierend auf Interferenz-Prinzipien.
    
    Zwei-Stufen-Ansatz:
    1. COARSE: Band-Signatur Matching (welche Bänder sind aktiv + Phase)
    2. FINE: Gewichtete Interferenz-Berechnung
    
    Der Trick: Wir indizieren nach (Band, Phase) Kombinationen.
    Documents die in denselben Bändern mit derselben Phase aktiv sind,
    werden konstruktiv interferieren.
    """
    
    def __init__(self,
                 num_bands: int = 32,
                 active_threshold_percentile: float = 70,
                 candidate_multiplier: int = 20,
                 use_band_weighting: bool = True):
        
        self.num_bands = num_bands
        self.candidate_multiplier = candidate_multiplier
        self.use_band_weighting = use_band_weighting
        
        self.extractor = BandSignatureExtractor(num_bands, active_threshold_percentile)
        
        # Storage
        self.doc_ids = []
        self.doc_embeddings = None
        self.doc_energies = None
        self.doc_phases = None
        self.doc_active_masks = None
        
        # Inverted Index: (band, phase) → [doc_indices]
        self.band_phase_index = defaultdict(list)
    
    def build(self, doc_ids: List[str], embeddings: np.ndarray):
        """Build the index."""
        self.doc_ids = doc_ids
        self.doc_embeddings = embeddings
        
        # Batch extract signatures
        self.doc_energies, self.doc_phases, self.doc_active_masks = \
            self.extractor.extract_batch(embeddings)
        
        # Build inverted index
        n_docs = len(doc_ids)
        
        for doc_idx in range(n_docs):
            active_bands = np.where(self.doc_active_masks[doc_idx])[0]
            
            for band in active_bands:
                phase = int(self.doc_phases[doc_idx, band])
                key = (int(band), phase)
                self.band_phase_index[key].append(doc_idx)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Interference-based search.
        """
        # Extract query signature
        query_sig = self.extractor.extract(query_embedding)
        
        # Stage 1: Find candidates via band-phase matching
        votes = defaultdict(float)
        
        active_bands = np.where(query_sig.active_mask)[0]
        
        for band in active_bands:
            phase = int(query_sig.phases[band])
            energy_weight = query_sig.energies[band] if self.use_band_weighting else 1.0
            
            key = (int(band), phase)
            
            for doc_idx in self.band_phase_index.get(key, []):
                votes[doc_idx] += energy_weight
        
        # Get top candidates
        num_candidates = min(top_k * self.candidate_multiplier, len(self.doc_ids))
        
        if votes:
            candidate_indices = sorted(votes.keys(), key=lambda x: -votes[x])[:num_candidates]
        else:
            # Fallback
            candidate_indices = list(range(min(num_candidates, len(self.doc_ids))))
        
        # Stage 2: Compute weighted interference
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        
        # Band weights from query energy
        if self.use_band_weighting:
            band_weights = query_sig.energies / (np.sum(query_sig.energies) + 1e-10)
        else:
            band_weights = np.ones(self.num_bands) / self.num_bands
        
        results = []
        
        # Reshape for band computation
        dim = len(query_embedding)
        band_size = dim // self.num_bands
        query_bands = query_norm[:self.num_bands * band_size].reshape(self.num_bands, band_size)
        
        for doc_idx in candidate_indices:
            doc_emb = self.doc_embeddings[doc_idx]
            doc_norm = doc_emb / (np.linalg.norm(doc_emb) + 1e-10)
            doc_bands = doc_norm[:self.num_bands * band_size].reshape(self.num_bands, band_size)
            
            # Per-band interference
            band_interference = 2 * np.sum(query_bands * doc_bands, axis=1)
            
            # Weighted sum
            total_interference = np.dot(band_weights, band_interference)
            
            results.append((self.doc_ids[doc_idx], -total_interference))
        
        results.sort(key=lambda x: x[1])
        return results[:top_k]
    
    def search_brute_force(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """Baseline."""
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        
        sims = self.doc_embeddings @ query_norm
        norms = np.linalg.norm(self.doc_embeddings, axis=1)
        sims = sims / (norms + 1e-10)
        
        top_idx = np.argsort(sims)[::-1][:top_k]
        
        return [(self.doc_ids[i], -sims[i]) for i in top_idx]


# =============================================================================
# RESONANCE FILTER BANK
# =============================================================================

class ResonanceFilterBank:
    """
    Verwende ein Filterbank-Konzept für Retrieval.
    
    Idee: Jede Query definiert eine "Filterbank" basierend auf ihren
    aktiven Frequenzen. Dokumente die durch diese Filter "durchkommen"
    (hohe Resonanz zeigen) sind gute Matches.
    """
    
    def __init__(self,
                 num_filters: int = 16,
                 filter_q: float = 2.0,  # Quality factor
                 resonance_threshold: float = 0.5):
        """
        Args:
            num_filters: Anzahl der Filter
            filter_q: Gütefaktor (höher = schmalbandiger)
            resonance_threshold: Schwelle für "Resonanz erkannt"
        """
        self.num_filters = num_filters
        self.filter_q = filter_q
        self.resonance_threshold = resonance_threshold
        
        self.doc_ids = []
        self.doc_embeddings = None
        self.doc_filter_responses = None
    
    def _create_filter_bank(self, dim: int) -> np.ndarray:
        """
        Erstelle Gauß-Filterbank.
        
        Returns: (num_filters, dim) array
        """
        filters = np.zeros((self.num_filters, dim))
        
        # Logarithmisch verteilte Zentren (wie Audio-Oktaven)
        centers = np.logspace(np.log10(1), np.log10(dim), self.num_filters + 2)[1:-1]
        
        for i, center in enumerate(centers):
            sigma = center / (2 * self.filter_q)  # Breite basierend auf Q
            x = np.arange(dim)
            filters[i] = np.exp(-0.5 * ((x - center) / sigma) ** 2)
        
        # Normalisieren
        filters = filters / (np.sum(filters, axis=1, keepdims=True) + 1e-10)
        
        return filters
    
    def build(self, doc_ids: List[str], embeddings: np.ndarray):
        """Build filter responses for all docs."""
        self.doc_ids = doc_ids
        self.doc_embeddings = embeddings
        
        n_docs, dim = embeddings.shape
        self.filter_bank = self._create_filter_bank(dim)
        
        # Compute filter responses for all docs
        # (n_docs, dim) @ (dim, num_filters) = (n_docs, num_filters)
        self.doc_filter_responses = embeddings @ self.filter_bank.T
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search by filter response similarity.
        
        Query defines which filters are "active" (resonant).
        Find docs with similar active filters.
        """
        # Query filter response
        query_response = query_embedding @ self.filter_bank.T
        
        # Normalize responses
        query_norm = query_response / (np.linalg.norm(query_response) + 1e-10)
        doc_norms = self.doc_filter_responses / (np.linalg.norm(self.doc_filter_responses, axis=1, keepdims=True) + 1e-10)
        
        # Similarity in filter space
        similarities = doc_norms @ query_norm
        
        top_idx = np.argsort(similarities)[::-1][:top_k]
        
        return [(self.doc_ids[i], -similarities[i]) for i in top_idx]


# =============================================================================
# PHASE-COHERENT SEARCH
# =============================================================================

class PhaseCoherentSearch:
    """
    Suche basierend auf Phasen-Kohärenz.
    
    In Audio: Kohärente Phasen = Signale verstärken sich
    
    Für Embeddings: Wenn Query und Doc in vielen Dimensionen
    das gleiche Vorzeichen haben = hohe Phasen-Kohärenz = ähnlich
    """
    
    def __init__(self, 
                 num_bands: int = 32,
                 coherence_weight: float = 0.5):
        """
        Args:
            num_bands: Für Band-basierte Kohärenz
            coherence_weight: Gewichtung von Kohärenz vs Magnitude
        """
        self.num_bands = num_bands
        self.coherence_weight = coherence_weight
        
        self.doc_ids = []
        self.doc_embeddings = None
        self.doc_signs = None  # Vorzeichen jeder Dimension
    
    def build(self, doc_ids: List[str], embeddings: np.ndarray):
        """Build index."""
        self.doc_ids = doc_ids
        self.doc_embeddings = embeddings
        self.doc_signs = np.sign(embeddings)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search by phase coherence.
        
        Coherence = Anteil der Dimensionen mit gleichem Vorzeichen
        """
        query_sign = np.sign(query_embedding)
        
        # Phase coherence: How many dimensions have same sign?
        # sign(q) * sign(d) = +1 wenn gleich, -1 wenn unterschiedlich
        sign_products = self.doc_signs * query_sign  # (n_docs, dim)
        coherence = np.mean(sign_products, axis=1)  # [-1, 1]
        
        # Normalize coherence to [0, 1]
        coherence = (coherence + 1) / 2
        
        # Magnitude similarity (standard cosine)
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        doc_norms = self.doc_embeddings / (np.linalg.norm(self.doc_embeddings, axis=1, keepdims=True) + 1e-10)
        magnitude_sim = doc_norms @ query_norm
        
        # Normalize to [0, 1]
        magnitude_sim = (magnitude_sim + 1) / 2
        
        # Combined score
        combined = self.coherence_weight * coherence + (1 - self.coherence_weight) * magnitude_sim
        
        top_idx = np.argsort(combined)[::-1][:top_k]
        
        return [(self.doc_ids[i], -combined[i]) for i in top_idx]


# =============================================================================
# BENCHMARK
# =============================================================================

def run_comprehensive_benchmark():
    """Run full benchmark with all methods."""
    
    print("=" * 70)
    print("COMPREHENSIVE INTERFERENCE RETRIEVAL BENCHMARK")
    print("=" * 70)
    
    # Test parameters
    configs = [
        {"n_docs": 2000, "n_queries": 100, "n_clusters": 10, "noise": 0.3},
        {"n_docs": 5000, "n_queries": 150, "n_clusters": 20, "noise": 0.4},
        {"n_docs": 10000, "n_queries": 200, "n_clusters": 30, "noise": 0.5},
    ]
    
    dim = 1024
    
    for config in configs:
        n_docs = config["n_docs"]
        n_queries = config["n_queries"]
        n_clusters = config["n_clusters"]
        noise = config["noise"]
        
        print(f"\n{'='*70}")
        print(f"Config: {n_docs} docs, {n_queries} queries, {n_clusters} clusters, noise={noise}")
        print("=" * 70)
        
        np.random.seed(42)
        
        # Create data
        centers = np.random.randn(n_clusters, dim)
        centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
        
        doc_embeddings = []
        doc_clusters = []
        for i in range(n_docs):
            cluster = i % n_clusters
            emb = centers[cluster] + np.random.randn(dim) * noise
            emb = emb / np.linalg.norm(emb)
            doc_embeddings.append(emb)
            doc_clusters.append(cluster)
        
        doc_embeddings = np.array(doc_embeddings)
        doc_ids = [str(i) for i in range(n_docs)]
        
        query_embeddings = []
        query_clusters = []
        for i in range(n_queries):
            cluster = np.random.randint(n_clusters)
            emb = centers[cluster] + np.random.randn(dim) * noise
            emb = emb / np.linalg.norm(emb)
            query_embeddings.append(emb)
            query_clusters.append(cluster)
        
        query_embeddings = np.array(query_embeddings)
        
        # Methods
        methods = {
            "Brute Force": None,
            "Interference (32 bands)": InterferenceIndex(num_bands=32, candidate_multiplier=20),
            "Interference (64 bands)": InterferenceIndex(num_bands=64, candidate_multiplier=20),
            "Interference (no weight)": InterferenceIndex(num_bands=32, use_band_weighting=False),
            "Resonance FilterBank": ResonanceFilterBank(num_filters=16),
            "Phase Coherent": PhaseCoherentSearch(num_bands=32),
        }
        
        # Build
        print("\nBuilding...")
        for name, method in methods.items():
            if method is not None:
                start = time.time()
                method.build(doc_ids, doc_embeddings)
                print(f"  {name}: {time.time()-start:.2f}s")
        
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
        baseline_recall = results["Brute Force"]["recall"]
        baseline_time = results["Brute Force"]["time"]
        
        print(f"\n{'Method':<25} {'Recall@10':>10} {'Time':>10} {'vs Base':>10} {'Speedup':>10}")
        print("-" * 70)
        
        for name, res in sorted(results.items(), key=lambda x: -x[1]["recall"]):
            rel = res["recall"] / baseline_recall * 100 if baseline_recall > 0 else 0
            speed = baseline_time / res["time"] if res["time"] > 0 else 0
            print(f"{name:<25} {res['recall']:>9.1f}% {res['time']:>9.2f}ms {rel:>9.1f}% {speed:>9.1f}x")


def test_with_real_data():
    """Test with real embeddings if available."""
    try:
        from sentence_transformers import SentenceTransformer
        from datasets import load_dataset
    except ImportError:
        print("Need sentence-transformers and datasets")
        return
    
    print("=" * 70)
    print("TESTING WITH REAL EMBEDDINGS (SciFact)")
    print("=" * 70)
    
    print("\n1. Loading model...")
    model = SentenceTransformer("intfloat/multilingual-e5-large")
    
    print("2. Loading data...")
    corpus = load_dataset("mteb/scifact", "corpus", split="corpus")
    queries = load_dataset("mteb/scifact", "queries", split="queries")
    qrels_data = load_dataset("mteb/scifact", "default", split="test")
    
    qrels = defaultdict(set)
    for item in qrels_data:
        qrels[item["query-id"]].add(item["corpus-id"])
    
    corpus_list = list(corpus)[:5000]
    query_list = [q for q in queries if q["_id"] in qrels][:300]
    
    print(f"   Corpus: {len(corpus_list)}, Queries: {len(query_list)}")
    
    print("3. Encoding...")
    corpus_texts = [f"passage: {doc['title']} {doc['text']}" for doc in corpus_list]
    corpus_embeddings = model.encode(corpus_texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    
    query_texts = [f"query: {q['text']}" for q in query_list]
    query_embeddings = model.encode(query_texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    
    doc_ids = [doc["_id"] for doc in corpus_list]
    
    print("4. Building indices...")
    
    methods = {
        "Brute Force": None,
        "Interference Index": InterferenceIndex(num_bands=32, candidate_multiplier=30),
        "Phase Coherent": PhaseCoherentSearch(),
    }
    
    for name, method in methods.items():
        if method is not None:
            method.build(doc_ids, corpus_embeddings)
    
    print("5. Evaluating...")
    
    results = {}
    
    for name, method in methods.items():
        recalls = []
        times = []
        
        for i, query in enumerate(query_list):
            query_id = query["_id"]
            query_emb = query_embeddings[i]
            relevant = qrels.get(query_id, set())
            
            if not relevant:
                continue
            
            start = time.time()
            
            if method is None:
                sims = corpus_embeddings @ query_emb
                top_idx = np.argsort(sims)[::-1][:10]
                retrieved = [doc_ids[idx] for idx in top_idx]
            else:
                search_results = method.search(query_emb, top_k=10)
                retrieved = [doc_id for doc_id, _ in search_results]
            
            times.append(time.time() - start)
            recalls.append(len(set(retrieved) & relevant) / len(relevant))
        
        results[name] = {
            "recall": np.mean(recalls) * 100,
            "time": np.mean(times) * 1000,
        }
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    baseline = results["Brute Force"]["recall"]
    
    for name, res in sorted(results.items(), key=lambda x: -x[1]["recall"]):
        rel = res["recall"] / baseline * 100 if baseline > 0 else 0
        print(f"{name}: Recall@10={res['recall']:.1f}% ({rel:.1f}% of BF), Time={res['time']:.2f}ms")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--real":
        test_with_real_data()
    else:
        run_comprehensive_benchmark()
