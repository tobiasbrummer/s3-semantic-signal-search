#!/usr/bin/env python3
"""
Hybrid Audio Retrieval: Peak Fingerprinting + Cosine Ranking

Key insight from the benchmark:
- Peak Fingerprinting finds ~60% of relevant docs
- But it's FAST (inverted index lookup)
- Solution: Use Peaks for COARSE filter, Cosine for FINE ranking

This is similar to Shazam:
1. Hash lookup to find candidates
2. Time-alignment verification (our "Cosine verification")

Author: Claude & Toby
Date: December 2024
"""

import numpy as np
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import time

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


# =============================================================================
# HYBRID RETRIEVER
# =============================================================================

class HybridPeakRetriever:
    """
    Two-stage retrieval:
    1. COARSE: Peak fingerprint lookup (fast, inverted index)
    2. FINE: Cosine similarity on candidates (accurate)
    
    The key parameters:
    - num_peaks: More peaks = better recall, slower
    - candidate_multiplier: How many candidates per top_k result
    """
    
    def __init__(self,
                 num_peaks: int = 64,
                 use_signs: bool = True,
                 candidate_multiplier: int = 20,
                 use_peak_magnitude: bool = True):
        """
        Args:
            num_peaks: Number of peak dimensions to index
            use_signs: Whether to differentiate positive/negative peaks
            candidate_multiplier: Get this many candidates per desired result
            use_peak_magnitude: Weight votes by peak magnitude
        """
        self.num_peaks = num_peaks
        self.use_signs = use_signs
        self.candidate_multiplier = candidate_multiplier
        self.use_peak_magnitude = use_peak_magnitude
        
        # Storage
        self.doc_ids = []
        self.doc_embeddings = None  # (N, D) array
        
        # Inverted index: (dim, sign) → [(doc_idx, magnitude), ...]
        self.peak_index = defaultdict(list)
    
    def _extract_peaks(self, embedding: np.ndarray) -> List[Tuple[int, int, float]]:
        """
        Extract peak dimensions.
        
        Returns: List of (dimension, sign, magnitude)
        """
        abs_values = np.abs(embedding)
        top_indices = np.argsort(abs_values)[-self.num_peaks:]
        
        peaks = []
        for idx in top_indices:
            sign = 1 if embedding[idx] > 0 else -1
            magnitude = abs_values[idx]
            peaks.append((int(idx), sign, float(magnitude)))
        
        return peaks
    
    def build(self, doc_ids: List[str], embeddings: np.ndarray):
        """Build the index."""
        self.doc_ids = doc_ids
        self.doc_embeddings = embeddings
        
        # Normalize embeddings for cosine
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.doc_embeddings_norm = embeddings / np.maximum(norms, 1e-10)
        
        # Build peak index
        for doc_idx, emb in enumerate(embeddings):
            peaks = self._extract_peaks(emb)
            
            for dim, sign, magnitude in peaks:
                if self.use_signs:
                    key = (dim, sign)
                else:
                    key = (dim, 0)
                
                self.peak_index[key].append((doc_idx, magnitude))
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Two-stage search.
        """
        # Extract query peaks
        query_peaks = self._extract_peaks(query_embedding)
        
        # STAGE 1: Vote counting to get candidates
        votes = defaultdict(float)
        
        for dim, sign, q_magnitude in query_peaks:
            if self.use_signs:
                key = (dim, sign)
            else:
                key = (dim, 0)
            
            for doc_idx, d_magnitude in self.peak_index.get(key, []):
                if self.use_peak_magnitude:
                    # Weight by geometric mean of magnitudes
                    weight = np.sqrt(q_magnitude * d_magnitude)
                else:
                    weight = 1.0
                
                votes[doc_idx] += weight
        
        # Get top candidates
        num_candidates = min(top_k * self.candidate_multiplier, len(self.doc_ids))
        
        candidate_indices = sorted(votes.keys(), key=lambda x: -votes[x])[:num_candidates]
        
        # If not enough candidates, add random docs
        if len(candidate_indices) < num_candidates:
            remaining = set(range(len(self.doc_ids))) - set(candidate_indices)
            candidate_indices.extend(list(remaining)[:num_candidates - len(candidate_indices)])
        
        # STAGE 2: Cosine similarity on candidates
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        
        candidate_embeddings = self.doc_embeddings_norm[candidate_indices]
        similarities = candidate_embeddings @ query_norm
        
        # Sort and return
        sorted_idx = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in sorted_idx:
            doc_idx = candidate_indices[idx]
            results.append((self.doc_ids[doc_idx], float(-similarities[idx])))
        
        return results
    
    def search_brute_force(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """Baseline brute force."""
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        
        similarities = self.doc_embeddings_norm @ query_norm
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((self.doc_ids[idx], float(-similarities[idx])))
        
        return results


# =============================================================================
# BENCHMARK
# =============================================================================

def benchmark_hybrid(n_docs: int = 5000, n_queries: int = 200):
    """
    Benchmark hybrid approach with different num_peaks and candidate_multiplier.
    """
    print("=" * 70)
    print("HYBRID PEAK RETRIEVAL BENCHMARK")
    print("=" * 70)
    
    # Create clustered data
    np.random.seed(42)
    dim = 1024
    n_clusters = 20
    
    print(f"\nData: {n_docs} docs, {n_queries} queries, {n_clusters} clusters, {dim}D")
    
    # Cluster centers
    centers = np.random.randn(n_clusters, dim)
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    
    # Documents
    doc_embeddings = []
    doc_clusters = []
    for i in range(n_docs):
        cluster = i % n_clusters
        noise = np.random.randn(dim) * 0.4  # More noise = harder
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
    
    # Test configurations
    configs = [
        {"num_peaks": 32, "candidate_multiplier": 10},
        {"num_peaks": 64, "candidate_multiplier": 10},
        {"num_peaks": 128, "candidate_multiplier": 10},
        {"num_peaks": 64, "candidate_multiplier": 5},
        {"num_peaks": 64, "candidate_multiplier": 20},
        {"num_peaks": 64, "candidate_multiplier": 50},
    ]
    
    results = []
    
    # Brute force baseline
    print("\nTesting Brute Force baseline...")
    retriever = HybridPeakRetriever()
    retriever.build(doc_ids, doc_embeddings)
    
    bf_times = []
    bf_recall = []
    
    for i in range(n_queries):
        query_emb = query_embeddings[i]
        relevant = set(str(j) for j in range(n_docs) if doc_clusters[j] == query_clusters[i])
        
        start = time.time()
        result = retriever.search_brute_force(query_emb, top_k=10)
        bf_times.append(time.time() - start)
        
        retrieved = set(doc_id for doc_id, _ in result)
        bf_recall.append(len(retrieved & relevant) / min(len(relevant), 10))
    
    bf_avg_time = np.mean(bf_times) * 1000
    bf_avg_recall = np.mean(bf_recall) * 100
    
    results.append({
        "name": "Brute Force",
        "recall": bf_avg_recall,
        "time": bf_avg_time,
        "speedup": 1.0,
    })
    
    print(f"  Recall@10: {bf_avg_recall:.1f}%, Time: {bf_avg_time:.2f}ms")
    
    # Test each configuration
    for config in configs:
        name = f"Peaks={config['num_peaks']}, Mult={config['candidate_multiplier']}"
        print(f"\nTesting {name}...")
        
        retriever = HybridPeakRetriever(
            num_peaks=config["num_peaks"],
            candidate_multiplier=config["candidate_multiplier"],
        )
        
        start = time.time()
        retriever.build(doc_ids, doc_embeddings)
        build_time = time.time() - start
        
        search_times = []
        recall_at_10 = []
        
        for i in range(n_queries):
            query_emb = query_embeddings[i]
            relevant = set(str(j) for j in range(n_docs) if doc_clusters[j] == query_clusters[i])
            
            start = time.time()
            result = retriever.search(query_emb, top_k=10)
            search_times.append(time.time() - start)
            
            retrieved = set(doc_id for doc_id, _ in result)
            recall_at_10.append(len(retrieved & relevant) / min(len(relevant), 10))
        
        avg_time = np.mean(search_times) * 1000
        avg_recall = np.mean(recall_at_10) * 100
        speedup = bf_avg_time / avg_time
        
        results.append({
            "name": name,
            "recall": avg_recall,
            "time": avg_time,
            "speedup": speedup,
            "build_time": build_time,
        })
        
        print(f"  Recall@10: {avg_recall:.1f}%, Time: {avg_time:.2f}ms, Speedup: {speedup:.1f}x")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Configuration':<35} {'Recall@10':>12} {'Time (ms)':>12} {'Speedup':>10}")
    print("-" * 70)
    
    for res in results:
        print(f"{res['name']:<35} {res['recall']:>11.1f}% {res['time']:>11.2f} {res['speedup']:>9.1f}x")
    
    # Find best config
    print("\n" + "-" * 70)
    
    best_recall = max(r for r in results if r["name"] != "Brute Force")
    print(f"Best Recall: {best_recall['name']} ({best_recall['recall']:.1f}%)")
    
    fast_results = [r for r in results if r["speedup"] > 2 and r["name"] != "Brute Force"]
    if fast_results:
        best_fast = max(fast_results, key=lambda x: x["recall"])
        print(f"Best Fast (>2x speedup): {best_fast['name']} ({best_fast['recall']:.1f}%, {best_fast['speedup']:.1f}x)")
    
    return results


def benchmark_with_real_model():
    """
    Benchmark with real embeddings if sentence-transformers available.
    """
    if not HAS_ST or not HAS_DATASETS:
        print("Need sentence-transformers and datasets for this test")
        return
    
    print("=" * 70)
    print("HYBRID RETRIEVAL WITH REAL EMBEDDINGS")
    print("=" * 70)
    
    # Load model
    print("\n1. Loading model...")
    model = SentenceTransformer("intfloat/multilingual-e5-large")
    
    # Load dataset
    print("2. Loading SciFact...")
    corpus = load_dataset("mteb/scifact", "corpus", split="corpus")
    queries = load_dataset("mteb/scifact", "queries", split="queries")
    qrels_data = load_dataset("mteb/scifact", "default", split="test")
    
    qrels = defaultdict(set)
    for item in qrels_data:
        qrels[item["query-id"]].add(item["corpus-id"])
    
    corpus_list = list(corpus)[:5000]
    query_list = [q for q in queries if q["_id"] in qrels][:300]
    
    print(f"   Corpus: {len(corpus_list)}, Queries: {len(query_list)}")
    
    # Encode
    print("3. Encoding...")
    corpus_texts = [f"passage: {doc['title']} {doc['text']}" for doc in corpus_list]
    corpus_embeddings = model.encode(corpus_texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    
    query_texts = [f"query: {q['text']}" for q in query_list]
    query_embeddings = model.encode(query_texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    
    doc_ids = [doc["_id"] for doc in corpus_list]
    
    # Build and test
    print("4. Building hybrid index...")
    retriever = HybridPeakRetriever(num_peaks=128, candidate_multiplier=20)
    retriever.build(doc_ids, corpus_embeddings)
    
    print("5. Evaluating...")
    
    methods = {
        "Hybrid": lambda q: retriever.search(q, top_k=10),
        "Brute Force": lambda q: retriever.search_brute_force(q, top_k=10),
    }
    
    results = {name: {"recall": [], "time": []} for name in methods}
    
    for i, query in enumerate(query_list):
        query_id = query["_id"]
        query_emb = query_embeddings[i]
        relevant = qrels.get(query_id, set())
        
        if not relevant:
            continue
        
        for name, search_fn in methods.items():
            start = time.time()
            search_results = search_fn(query_emb)
            results[name]["time"].append(time.time() - start)
            
            retrieved = set(doc_id for doc_id, _ in search_results)
            results[name]["recall"].append(len(retrieved & relevant) / len(relevant))
    
    # Print
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    for name in methods:
        avg_recall = np.mean(results[name]["recall"]) * 100
        avg_time = np.mean(results[name]["time"]) * 1000
        print(f"\n{name}:")
        print(f"  Recall@10: {avg_recall:.1f}%")
        print(f"  Avg Time: {avg_time:.2f}ms")
    
    hybrid_recall = np.mean(results["Hybrid"]["recall"])
    bf_recall = np.mean(results["Brute Force"]["recall"])
    
    print(f"\nHybrid achieves {hybrid_recall/bf_recall*100:.1f}% of Brute Force recall")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--real":
        benchmark_with_real_model()
    else:
        benchmark_hybrid()
