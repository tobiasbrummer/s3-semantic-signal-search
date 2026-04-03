#!/usr/bin/env python3
"""
ADVANCED AUDIO RETRIEVAL CONCEPTS

Exploring:
1. AUSLÖSCHUNG (Destructive Interference) - Anti-resonance as negative signal
2. TEMPORAL (t) - Phase gradients, local coherence over dimensions
3. GATES - Noise gate, only count where signal is strong
4. SIDECHAIN - Only count where BOTH signals are strong
5. ENVELOPE - Spectral envelope matching (smoothed energy)

Author: Claude & Toby (Audio Engineers)
Date: December 2024
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import time


# =============================================================================
# 1. AUSLÖSCHUNG (Destructive Interference)
# =============================================================================

class CancellationAwareSearch:
    """
    Nutze Auslöschung als NEGATIV-Signal.
    
    Idee:
    - Konstruktive Interferenz (gleiche Phase): BONUS
    - Destruktive Interferenz (Gegenphasig + beide stark): MALUS
    
    Ein Dokument das in wichtigen Dimensionen "gegenphasig" ist,
    ist aktiv UNÄHNLICH, nicht nur "nicht ähnlich".
    """
    
    def __init__(self, 
                 cancellation_penalty: float = 0.5,
                 magnitude_threshold: float = 0.1):
        """
        Args:
            cancellation_penalty: Wie stark bestrafen wir Auslöschung (0-1)
            magnitude_threshold: Ab welcher Magnitude zählt eine Dimension als "aktiv"
        """
        self.cancellation_penalty = cancellation_penalty
        self.magnitude_threshold = magnitude_threshold
        
        self.doc_ids = []
        self.doc_embeddings = None
    
    def build(self, doc_ids: List[str], embeddings: np.ndarray):
        self.doc_ids = doc_ids
        self.doc_embeddings = embeddings
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search with cancellation awareness.
        
        Score = constructive_score - cancellation_penalty * destructive_score
        """
        q = query_embedding
        q_abs = np.abs(q)
        q_sign = np.sign(q)
        q_strong = q_abs > self.magnitude_threshold * np.max(q_abs)
        
        results = []
        
        for i, doc_emb in enumerate(self.doc_embeddings):
            d = doc_emb
            d_abs = np.abs(d)
            d_sign = np.sign(d)
            d_strong = d_abs > self.magnitude_threshold * np.max(d_abs)
            
            # Both signals strong in this dimension
            both_strong = q_strong & d_strong
            
            # Sign agreement
            sign_product = q_sign * d_sign  # +1 = same, -1 = opposite
            
            # Constructive: Same sign where both strong
            constructive = np.sum((sign_product > 0) & both_strong)
            
            # Destructive: Opposite sign where both strong (CANCELLATION!)
            destructive = np.sum((sign_product < 0) & both_strong)
            
            # Also add general cosine component
            cosine = np.dot(q, d) / (np.linalg.norm(q) * np.linalg.norm(d) + 1e-10)
            
            # Combined score
            score = cosine + 0.1 * constructive - self.cancellation_penalty * 0.1 * destructive
            
            results.append((self.doc_ids[i], -score))
        
        results.sort(key=lambda x: x[1])
        return results[:top_k]


# =============================================================================
# 2. TEMPORAL / PHASE GRADIENT
# =============================================================================

class TemporalCoherenceSearch:
    """
    Betrachte die Dimension-Reihenfolge als "Zeit".
    
    Idee: Ähnliche Embeddings haben ähnliche "Phase-Verläufe"
    
    Phase-Gradient: Wie ändert sich das Vorzeichen über die Dimensionen?
    Wenn beide Signale ähnliche "Übergangsmuster" haben, sind sie ähnlich.
    """
    
    def __init__(self,
                 window_size: int = 32,
                 use_transitions: bool = True,
                 use_local_correlation: bool = True):
        """
        Args:
            window_size: Größe des lokalen Fensters
            use_transitions: Nutze Vorzeichen-Übergänge als Feature
            use_local_correlation: Nutze lokale Korrelation
        """
        self.window_size = window_size
        self.use_transitions = use_transitions
        self.use_local_correlation = use_local_correlation
        
        self.doc_ids = []
        self.doc_embeddings = None
        self.doc_transitions = None
        self.doc_local_energies = None
    
    def _compute_transitions(self, embedding: np.ndarray) -> np.ndarray:
        """
        Compute sign transitions: Where does the sign change?
        
        Returns binary array: 1 where sign(dim[i]) != sign(dim[i+1])
        """
        signs = np.sign(embedding)
        transitions = (signs[:-1] != signs[1:]).astype(float)
        return transitions
    
    def _compute_local_energies(self, embedding: np.ndarray) -> np.ndarray:
        """
        Compute energy in local windows.
        """
        n_windows = len(embedding) // self.window_size
        energies = np.zeros(n_windows)
        
        for i in range(n_windows):
            start = i * self.window_size
            end = start + self.window_size
            energies[i] = np.sum(embedding[start:end] ** 2)
        
        return energies
    
    def build(self, doc_ids: List[str], embeddings: np.ndarray):
        self.doc_ids = doc_ids
        self.doc_embeddings = embeddings
        
        n_docs = len(doc_ids)
        
        if self.use_transitions:
            self.doc_transitions = np.array([
                self._compute_transitions(emb) for emb in embeddings
            ])
        
        if self.use_local_correlation:
            self.doc_local_energies = np.array([
                self._compute_local_energies(emb) for emb in embeddings
            ])
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search using temporal features.
        """
        q = query_embedding
        
        # Base cosine
        q_norm = q / (np.linalg.norm(q) + 1e-10)
        
        scores = []
        
        # Transition similarity
        if self.use_transitions:
            q_trans = self._compute_transitions(q)
            # Hamming similarity of transitions
            trans_sim = 1 - np.mean(np.abs(self.doc_transitions - q_trans), axis=1)
        else:
            trans_sim = np.zeros(len(self.doc_ids))
        
        # Local energy correlation
        if self.use_local_correlation:
            q_local = self._compute_local_energies(q)
            q_local_norm = q_local / (np.linalg.norm(q_local) + 1e-10)
            doc_local_norms = self.doc_local_energies / (
                np.linalg.norm(self.doc_local_energies, axis=1, keepdims=True) + 1e-10
            )
            local_sim = doc_local_norms @ q_local_norm
        else:
            local_sim = np.zeros(len(self.doc_ids))
        
        # Cosine similarity
        doc_norms = self.doc_embeddings / (
            np.linalg.norm(self.doc_embeddings, axis=1, keepdims=True) + 1e-10
        )
        cosine_sim = doc_norms @ q_norm
        
        # Combine
        combined = 0.6 * cosine_sim + 0.2 * trans_sim + 0.2 * local_sim
        
        top_idx = np.argsort(combined)[::-1][:top_k]
        
        return [(self.doc_ids[i], -combined[i]) for i in top_idx]


# =============================================================================
# 3. NOISE GATE
# =============================================================================

class GatedSearch:
    """
    Noise Gate: Nur Dimensionen zählen wo die Query STARK ist.
    
    In Audio: Ein Noise Gate lässt nur Signale durch die über einem
    Threshold liegen. Leise Signale = Rauschen = ignorieren.
    
    Für Embeddings: Dimensionen mit kleinen Query-Werten sind
    vielleicht nicht "semantisch relevant" für diese Query.
    """
    
    def __init__(self,
                 gate_percentile: float = 50,
                 attack_soft: bool = True):
        """
        Args:
            gate_percentile: Nur Top-X% der Query-Dimensionen nutzen
            attack_soft: Weiche Flanke (gewichtet) vs harte Flanke (binär)
        """
        self.gate_percentile = gate_percentile
        self.attack_soft = attack_soft
        
        self.doc_ids = []
        self.doc_embeddings = None
    
    def build(self, doc_ids: List[str], embeddings: np.ndarray):
        self.doc_ids = doc_ids
        self.doc_embeddings = embeddings
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Gated search: Only use dimensions where query is strong.
        """
        q = query_embedding
        q_abs = np.abs(q)
        
        # Gate threshold
        threshold = np.percentile(q_abs, 100 - self.gate_percentile)
        
        if self.attack_soft:
            # Soft gate: Weight by query magnitude
            gate_weights = np.maximum(0, q_abs - threshold)
            gate_weights = gate_weights / (np.sum(gate_weights) + 1e-10)
        else:
            # Hard gate: Binary mask
            gate_weights = (q_abs >= threshold).astype(float)
            gate_weights = gate_weights / (np.sum(gate_weights) + 1e-10)
        
        # Weighted dot product
        weighted_q = q * gate_weights
        
        # Normalize docs
        doc_norms = np.linalg.norm(self.doc_embeddings, axis=1, keepdims=True)
        doc_normalized = self.doc_embeddings / (doc_norms + 1e-10)
        
        # Gated similarity
        similarities = doc_normalized @ weighted_q
        
        top_idx = np.argsort(similarities)[::-1][:top_k]
        
        return [(self.doc_ids[i], -similarities[i]) for i in top_idx]


# =============================================================================
# 4. SIDECHAIN (Mutual Gating)
# =============================================================================

class SidechainSearch:
    """
    Sidechain: Nur Dimensionen zählen wo BEIDE Signale stark sind.
    
    In Audio: Sidechain-Compression nutzt ein externes Signal
    um ein anderes zu modulieren (z.B. Ducking).
    
    Hier: Query und Doc "gaten" sich gegenseitig.
    Nur wo beide "laut" sind, zählt die Ähnlichkeit.
    """
    
    def __init__(self,
                 mutual_threshold_percentile: float = 50,
                 use_geometric_mean: bool = True):
        """
        Args:
            mutual_threshold_percentile: Beide müssen über diesem Percentil sein
            use_geometric_mean: Geometrisches vs arithmetisches Mittel für Gating
        """
        self.mutual_threshold_percentile = mutual_threshold_percentile
        self.use_geometric_mean = use_geometric_mean
        
        self.doc_ids = []
        self.doc_embeddings = None
        self.doc_abs = None
        self.doc_percentiles = None
    
    def build(self, doc_ids: List[str], embeddings: np.ndarray):
        self.doc_ids = doc_ids
        self.doc_embeddings = embeddings
        self.doc_abs = np.abs(embeddings)
        
        # Pre-compute thresholds per doc
        self.doc_thresholds = np.percentile(
            self.doc_abs, self.mutual_threshold_percentile, axis=1
        )
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Sidechain search: Mutual gating.
        """
        q = query_embedding
        q_abs = np.abs(q)
        q_threshold = np.percentile(q_abs, self.mutual_threshold_percentile)
        q_strong = q_abs >= q_threshold
        
        results = []
        
        for i, (doc_emb, doc_abs, doc_thresh) in enumerate(
            zip(self.doc_embeddings, self.doc_abs, self.doc_thresholds)
        ):
            d_strong = doc_abs >= doc_thresh
            
            # Mutual: Both must be strong
            mutual_strong = q_strong & d_strong
            
            if np.sum(mutual_strong) == 0:
                # No mutual strong dimensions - fall back to cosine
                sim = np.dot(q, doc_emb) / (np.linalg.norm(q) * np.linalg.norm(doc_emb) + 1e-10)
            else:
                # Gated similarity
                if self.use_geometric_mean:
                    # Weight by geometric mean of magnitudes
                    weights = np.sqrt(q_abs * doc_abs) * mutual_strong
                else:
                    weights = mutual_strong.astype(float)
                
                weights = weights / (np.sum(weights) + 1e-10)
                
                # Weighted cosine on gated dimensions
                q_weighted = q * weights
                d_weighted = doc_emb * weights
                
                sim = np.dot(q_weighted, d_weighted) / (
                    np.linalg.norm(q_weighted) * np.linalg.norm(d_weighted) + 1e-10
                )
            
            results.append((self.doc_ids[i], -sim))
        
        results.sort(key=lambda x: x[1])
        return results[:top_k]


# =============================================================================
# 5. SPECTRAL ENVELOPE
# =============================================================================

class EnvelopeSearch:
    """
    Spectral Envelope: Vergleiche die "Hüllkurve" des Signals.
    
    In Audio: Die Envelope ist die geglättete Form des Spektrums.
    Sie zeigt die "grobe Struktur" ohne Feinheiten.
    
    Für Embeddings: Glätte die Magnitude-Kurve und vergleiche.
    Ähnliche Dokumente haben vielleicht ähnliche "Formen".
    """
    
    def __init__(self,
                 smoothing_window: int = 32,
                 envelope_weight: float = 0.3):
        """
        Args:
            smoothing_window: Größe des Glättungsfensters
            envelope_weight: Gewichtung von Envelope vs Raw Cosine
        """
        self.smoothing_window = smoothing_window
        self.envelope_weight = envelope_weight
        
        self.doc_ids = []
        self.doc_embeddings = None
        self.doc_envelopes = None
    
    def _compute_envelope(self, embedding: np.ndarray) -> np.ndarray:
        """
        Compute smoothed magnitude envelope.
        """
        abs_signal = np.abs(embedding)
        
        # Moving average smoothing
        kernel = np.ones(self.smoothing_window) / self.smoothing_window
        envelope = np.convolve(abs_signal, kernel, mode='same')
        
        return envelope
    
    def build(self, doc_ids: List[str], embeddings: np.ndarray):
        self.doc_ids = doc_ids
        self.doc_embeddings = embeddings
        
        self.doc_envelopes = np.array([
            self._compute_envelope(emb) for emb in embeddings
        ])
        
        # Normalize envelopes
        norms = np.linalg.norm(self.doc_envelopes, axis=1, keepdims=True)
        self.doc_envelopes_norm = self.doc_envelopes / (norms + 1e-10)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search using envelope similarity.
        """
        q = query_embedding
        q_envelope = self._compute_envelope(q)
        q_envelope_norm = q_envelope / (np.linalg.norm(q_envelope) + 1e-10)
        
        # Envelope similarity
        envelope_sim = self.doc_envelopes_norm @ q_envelope_norm
        
        # Raw cosine
        q_norm = q / (np.linalg.norm(q) + 1e-10)
        doc_norms = self.doc_embeddings / (
            np.linalg.norm(self.doc_embeddings, axis=1, keepdims=True) + 1e-10
        )
        cosine_sim = doc_norms @ q_norm
        
        # Combined
        combined = (1 - self.envelope_weight) * cosine_sim + self.envelope_weight * envelope_sim
        
        top_idx = np.argsort(combined)[::-1][:top_k]
        
        return [(self.doc_ids[i], -combined[i]) for i in top_idx]


# =============================================================================
# 6. COMBINED: All concepts together
# =============================================================================

class FullAudioSearch:
    """
    Kombiniere alle Audio-Konzepte:
    - Phase Coherence (Vorzeichen)
    - Cancellation Awareness (Auslöschung bestrafen)
    - Gating (nur starke Dimensionen)
    - Envelope (grobe Form)
    """
    
    def __init__(self,
                 gate_percentile: float = 40,
                 cancellation_penalty: float = 0.3,
                 envelope_weight: float = 0.1,
                 smoothing_window: int = 32):
        
        self.gate_percentile = gate_percentile
        self.cancellation_penalty = cancellation_penalty
        self.envelope_weight = envelope_weight
        self.smoothing_window = smoothing_window
        
        self.doc_ids = []
        self.doc_embeddings = None
        self.doc_signs = None
        self.doc_abs = None
        self.doc_envelopes = None
    
    def _compute_envelope(self, embedding: np.ndarray) -> np.ndarray:
        abs_signal = np.abs(embedding)
        kernel = np.ones(self.smoothing_window) / self.smoothing_window
        return np.convolve(abs_signal, kernel, mode='same')
    
    def build(self, doc_ids: List[str], embeddings: np.ndarray):
        self.doc_ids = doc_ids
        self.doc_embeddings = embeddings
        self.doc_signs = np.sign(embeddings)
        self.doc_abs = np.abs(embeddings)
        
        self.doc_envelopes = np.array([
            self._compute_envelope(emb) for emb in embeddings
        ])
        env_norms = np.linalg.norm(self.doc_envelopes, axis=1, keepdims=True)
        self.doc_envelopes_norm = self.doc_envelopes / (env_norms + 1e-10)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Full audio-inspired search.
        """
        q = query_embedding
        q_abs = np.abs(q)
        q_sign = np.sign(q)
        
        # Gate: Only strong query dimensions
        gate_threshold = np.percentile(q_abs, 100 - self.gate_percentile)
        gate_mask = q_abs >= gate_threshold
        
        # Envelope
        q_envelope = self._compute_envelope(q)
        q_envelope_norm = q_envelope / (np.linalg.norm(q_envelope) + 1e-10)
        envelope_sim = self.doc_envelopes_norm @ q_envelope_norm
        
        results = []
        
        for i in range(len(self.doc_embeddings)):
            d = self.doc_embeddings[i]
            d_abs = self.doc_abs[i]
            d_sign = self.doc_signs[i]
            
            # Doc gate threshold
            d_gate_threshold = np.percentile(d_abs, 100 - self.gate_percentile)
            d_gate_mask = d_abs >= d_gate_threshold
            
            # Mutual strong dimensions
            mutual_strong = gate_mask & d_gate_mask
            
            # Phase agreement/disagreement on mutual strong dims
            sign_product = q_sign * d_sign
            
            constructive = np.sum((sign_product > 0) & mutual_strong)
            destructive = np.sum((sign_product < 0) & mutual_strong)
            
            total_mutual = np.sum(mutual_strong)
            if total_mutual > 0:
                phase_score = (constructive - self.cancellation_penalty * destructive) / total_mutual
            else:
                phase_score = 0
            
            # Standard cosine
            cosine = np.dot(q, d) / (np.linalg.norm(q) * np.linalg.norm(d) + 1e-10)
            
            # Combined
            score = (
                0.5 * cosine + 
                0.3 * phase_score + 
                self.envelope_weight * envelope_sim[i]
            )
            
            results.append((self.doc_ids[i], -score))
        
        results.sort(key=lambda x: x[1])
        return results[:top_k]


# =============================================================================
# BENCHMARK
# =============================================================================

def benchmark_advanced_concepts():
    """Benchmark all advanced audio concepts."""
    
    print("=" * 70)
    print("ADVANCED AUDIO CONCEPTS BENCHMARK")
    print("=" * 70)
    
    np.random.seed(42)
    dim = 1024
    n_docs = 3000
    n_queries = 150
    n_clusters = 15
    noise = 0.4
    
    print(f"\nData: {n_docs} docs, {n_queries} queries, {n_clusters} clusters, noise={noise}")
    
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
        "Brute Force (Baseline)": None,
        "Cancellation Aware": CancellationAwareSearch(cancellation_penalty=0.5),
        "Temporal Coherence": TemporalCoherenceSearch(window_size=32),
        "Noise Gate (50%)": GatedSearch(gate_percentile=50),
        "Noise Gate (30%)": GatedSearch(gate_percentile=30),
        "Sidechain": SidechainSearch(mutual_threshold_percentile=50),
        "Envelope": EnvelopeSearch(smoothing_window=32),
        "Full Audio": FullAudioSearch(),
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
    baseline_recall = results["Brute Force (Baseline)"]["recall"]
    baseline_time = results["Brute Force (Baseline)"]["time"]
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\n{'Method':<25} {'Recall@10':>10} {'Time':>10} {'vs Base':>10}")
    print("-" * 60)
    
    for name, res in sorted(results.items(), key=lambda x: -x[1]["recall"]):
        rel = res["recall"] / baseline_recall * 100 if baseline_recall > 0 else 0
        print(f"{name:<25} {res['recall']:>9.1f}% {res['time']:>9.2f}ms {rel:>9.1f}%")
    
    return results


def test_with_real_embeddings():
    """Test with real embeddings."""
    try:
        from sentence_transformers import SentenceTransformer
        from datasets import load_dataset
    except ImportError:
        print("Need sentence-transformers and datasets")
        return
    
    print("=" * 70)
    print("ADVANCED AUDIO CONCEPTS - REAL EMBEDDINGS")
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
        "Cancellation Aware": CancellationAwareSearch(cancellation_penalty=0.3),
        "Noise Gate (40%)": GatedSearch(gate_percentile=40),
        "Sidechain": SidechainSearch(mutual_threshold_percentile=40),
        "Full Audio": FullAudioSearch(gate_percentile=40, cancellation_penalty=0.2),
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
        test_with_real_embeddings()
    else:
        benchmark_advanced_concepts()
