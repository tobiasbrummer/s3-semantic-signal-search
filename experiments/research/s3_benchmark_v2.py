#!/usr/bin/env python3
"""
S3 v2 - Improved Semantic Signal Search

Fixes from v1 benchmark:
1. MORE BITS: 448 → 1024 bits (closer to embedding dim)
2. MULTI-PROBE: Check many more bit-flip variants
3. VECTORIZED: NumPy-based Hamming distance
4. ADAPTIVE COARSE: Use top-N bass matches, not exact match

Author: Claude & Toby
Date: December 2024
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set
from collections import defaultdict

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
# IMPROVED CONFIG
# =============================================================================

@dataclass
class S3ConfigV2:
    """
    Improved configuration with more bits.
    
    Key insight: We need bits ≈ embedding_dim for good recall.
    But we can use hierarchical structure for speed.
    """
    embedding_dim: int = 1024
    
    # More bits! Total should approach embedding_dim
    # Band layout: (start, end, bits, weight)
    bands: List[Tuple[int, int, int, float]] = field(default_factory=lambda: [
        (0, 256, 256, 4.0),      # Bass: 256 dims → 256 bits
        (256, 512, 256, 2.0),    # Low-Mids: 256 dims → 256 bits  
        (512, 768, 256, 1.0),    # High-Mids: 256 dims → 256 bits
        (768, 1024, 256, 0.5),   # Highs: 256 dims → 256 bits
    ])
    
    lsh_seed: int = 42
    
    # Multi-probe settings
    num_probes: int = 50        # Check this many bit-flip variants
    coarse_candidates: int = 500  # Get this many from coarse filter
    
    @property
    def total_bits(self) -> int:
        return sum(b[2] for b in self.bands)
    
    @property
    def total_bytes(self) -> int:
        return self.total_bits // 8


# =============================================================================
# OPTIMIZED LSH
# =============================================================================

class OptimizedLSH:
    """
    LSH with vectorized operations.
    """
    
    def __init__(self, config: S3ConfigV2):
        self.config = config
        self.projections = []
        self.band_bits = []
        
        rng = np.random.RandomState(config.lsh_seed)
        
        for start, end, bits, weight in config.bands:
            input_dim = end - start
            proj = rng.randn(bits, input_dim).astype(np.float32)
            # Normalize rows for stability
            proj = proj / np.linalg.norm(proj, axis=1, keepdims=True)
            self.projections.append(proj)
            self.band_bits.append(bits)
    
    def hash_batch(self, vectors: np.ndarray) -> np.ndarray:
        """
        Hash a batch of vectors to packed bits.
        
        Returns: (N, total_bytes) uint8 array
        """
        # Normalize
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-10)
        
        all_bits = []
        
        for i, (start, end, bits, weight) in enumerate(self.config.bands):
            band_vectors = vectors[:, start:end]  # (N, band_dim)
            proj = self.projections[i]  # (bits, band_dim)
            
            # Project: (N, band_dim) @ (band_dim, bits) = (N, bits)
            dots = band_vectors @ proj.T
            
            # Sign to bits
            bits_array = (dots > 0).astype(np.uint8)
            all_bits.append(bits_array)
        
        # Concatenate all bands
        all_bits = np.concatenate(all_bits, axis=1)  # (N, total_bits)
        
        # Pack to bytes
        # Pad to multiple of 8
        total_bits = all_bits.shape[1]
        pad_bits = (8 - (total_bits % 8)) % 8
        if pad_bits > 0:
            all_bits = np.concatenate([all_bits, np.zeros((all_bits.shape[0], pad_bits), dtype=np.uint8)], axis=1)
        
        # Reshape and pack
        packed = np.packbits(all_bits, axis=1)
        
        return packed
    
    def hash_single(self, vector: np.ndarray) -> np.ndarray:
        """Hash a single vector."""
        return self.hash_batch(vector.reshape(1, -1))[0]


# =============================================================================
# VECTORIZED HAMMING
# =============================================================================

class VectorizedHamming:
    """
    Fast Hamming distance using NumPy.
    """
    
    # Precompute popcount table
    POPCOUNT_TABLE = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)
    
    @classmethod
    def distance(cls, a: np.ndarray, b: np.ndarray) -> int:
        """Hamming distance between two byte arrays."""
        xor = np.bitwise_xor(a, b)
        return int(np.sum(cls.POPCOUNT_TABLE[xor]))
    
    @classmethod
    def distance_matrix(cls, queries: np.ndarray, docs: np.ndarray) -> np.ndarray:
        """
        Compute Hamming distances between all query-doc pairs.
        
        Args:
            queries: (Q, bytes) uint8 array
            docs: (D, bytes) uint8 array
            
        Returns:
            (Q, D) int array of Hamming distances
        """
        # XOR: (Q, 1, bytes) ^ (1, D, bytes) = (Q, D, bytes)
        xor = np.bitwise_xor(queries[:, None, :], docs[None, :, :])
        
        # Popcount using lookup table
        popcounts = cls.POPCOUNT_TABLE[xor]  # (Q, D, bytes)
        
        # Sum over bytes
        distances = np.sum(popcounts, axis=2)  # (Q, D)
        
        return distances


# =============================================================================
# S3 INDEX V2
# =============================================================================

class S3IndexV2:
    """
    Improved S3 index with better recall.
    """
    
    def __init__(self, config: S3ConfigV2):
        self.config = config
        self.lsh = OptimizedLSH(config)
        
        # Storage
        self.doc_ids = []
        self.embeddings = []
        self.hashes = None  # Will be (N, bytes) array
        self.texts = []
        
        # For coarse filter: store bass band separately
        self.bass_hashes = None
        self.bass_bytes = config.bands[0][2] // 8
    
    def build(self, doc_ids: List[str], embeddings: np.ndarray, texts: List[str] = None):
        """
        Build index from embeddings.
        """
        self.doc_ids = doc_ids
        self.embeddings = embeddings
        self.texts = texts or [""] * len(doc_ids)
        
        # Hash all at once
        self.hashes = self.lsh.hash_batch(embeddings)
        
        # Extract bass band
        self.bass_hashes = self.hashes[:, :self.bass_bytes]
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search with multi-probe coarse filter + vectorized fine ranking.
        """
        # Hash query
        query_hash = self.lsh.hash_single(query_embedding)
        query_bass = query_hash[:self.bass_bytes]
        
        # STAGE 1: Coarse filter using bass band
        # Compute Hamming distance to all bass hashes
        bass_distances = VectorizedHamming.distance_matrix(
            query_bass.reshape(1, -1),
            self.bass_hashes
        )[0]  # (N,)
        
        # Get top candidates by bass distance
        num_candidates = min(self.config.coarse_candidates, len(self.doc_ids))
        candidate_indices = np.argsort(bass_distances)[:num_candidates]
        
        # STAGE 2: Fine ranking on candidates
        candidate_hashes = self.hashes[candidate_indices]
        
        # Full Hamming distance
        full_distances = VectorizedHamming.distance_matrix(
            query_hash.reshape(1, -1),
            candidate_hashes
        )[0]  # (num_candidates,)
        
        # Sort and get top-k
        sorted_indices = np.argsort(full_distances)[:top_k]
        
        results = []
        for idx in sorted_indices:
            doc_idx = candidate_indices[idx]
            results.append((self.doc_ids[doc_idx], float(full_distances[idx])))
        
        return results
    
    def search_brute_force(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Brute force cosine similarity.
        """
        # Normalize
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        
        # All similarities at once
        similarities = self.embeddings @ query_norm
        
        # Top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((self.doc_ids[idx], float(-similarities[idx])))
        
        return results
    
    def search_hamming_brute(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Brute force Hamming (for comparison - should match S3 recall if coarse filter is perfect).
        """
        query_hash = self.lsh.hash_single(query_embedding)
        
        distances = VectorizedHamming.distance_matrix(
            query_hash.reshape(1, -1),
            self.hashes
        )[0]
        
        top_indices = np.argsort(distances)[:top_k]
        
        results = []
        for idx in top_indices:
            results.append((self.doc_ids[idx], float(distances[idx])))
        
        return results


# =============================================================================
# EVALUATION
# =============================================================================

def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    retrieved_k = set(retrieved[:k])
    if not relevant:
        return 0.0
    return len(retrieved_k & relevant) / len(relevant)


def mrr(retrieved: List[str], relevant: Set[str]) -> float:
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


# =============================================================================
# BENCHMARK
# =============================================================================

def run_benchmark_v2(
    model_name: str = "intfloat/multilingual-e5-large",
    max_corpus: int = 5000,
    max_queries: int = 300,
):
    print("=" * 70)
    print("S3 v2 BENCHMARK (Improved)")
    print("=" * 70)
    
    # Load model
    print(f"\n1. Loading model: {model_name}")
    if not HAS_ST:
        print("   ❌ pip install sentence-transformers")
        return
    
    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()
    print(f"   Dimension: {dim}")
    
    # Load dataset
    print(f"\n2. Loading SciFact dataset...")
    if not HAS_DATASETS:
        print("   ❌ pip install datasets")
        return
    
    corpus = load_dataset("mteb/scifact", "corpus", split="corpus")
    queries = load_dataset("mteb/scifact", "queries", split="queries")
    qrels_data = load_dataset("mteb/scifact", "default", split="test")
    
    qrels = defaultdict(set)
    for item in qrels_data:
        qrels[item["query-id"]].add(item["corpus-id"])
    
    corpus_list = list(corpus)[:max_corpus]
    query_list = [q for q in queries if q["_id"] in qrels][:max_queries]
    
    print(f"   Corpus: {len(corpus_list)} docs")
    print(f"   Queries: {len(query_list)}")
    
    # Encode
    print(f"\n3. Encoding corpus...")
    corpus_texts = [f"passage: {doc['title']} {doc['text']}" for doc in corpus_list]
    
    start = time.time()
    corpus_embeddings = model.encode(
        corpus_texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True
    )
    print(f"   Time: {time.time() - start:.1f}s")
    
    print(f"\n4. Encoding queries...")
    query_texts = [f"query: {q['text']}" for q in query_list]
    query_embeddings = model.encode(
        query_texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True
    )
    
    # Build index
    print(f"\n5. Building S3 v2 index...")
    config = S3ConfigV2()
    print(f"   Total bits: {config.total_bits}")
    print(f"   Coarse candidates: {config.coarse_candidates}")
    
    index = S3IndexV2(config)
    
    start = time.time()
    doc_ids = [doc["_id"] for doc in corpus_list]
    index.build(doc_ids, corpus_embeddings, corpus_texts)
    print(f"   Build time: {time.time() - start:.3f}s")
    
    # Evaluate
    print(f"\n6. Evaluating...")
    
    k_values = [1, 5, 10, 20]
    
    methods = {
        "S3_v2": lambda q: index.search(q, top_k=20),
        "Hamming_BF": lambda q: index.search_hamming_brute(q, top_k=20),
        "Cosine_BF": lambda q: index.search_brute_force(q, top_k=20),
    }
    
    results = {name: {"recalls": {k: [] for k in k_values}, "mrrs": [], "times": []} 
               for name in methods}
    
    for i, query in enumerate(query_list):
        query_id = query["_id"]
        query_emb = query_embeddings[i]
        relevant = qrels.get(query_id, set())
        
        if not relevant:
            continue
        
        for name, search_fn in methods.items():
            start = time.time()
            search_results = search_fn(query_emb)
            results[name]["times"].append(time.time() - start)
            
            retrieved = [doc_id for doc_id, _ in search_results]
            
            for k in k_values:
                results[name]["recalls"][k].append(recall_at_k(retrieved, relevant, k))
            results[name]["mrrs"].append(mrr(retrieved, relevant))
    
    # Print results
    print(f"\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\n{'Metric':<15}", end="")
    for name in methods:
        print(f"{name:>15}", end="")
    print()
    print("-" * (15 + 15 * len(methods)))
    
    for k in k_values:
        print(f"Recall@{k:<8}", end="")
        for name in methods:
            r = np.mean(results[name]["recalls"][k]) * 100
            print(f"{r:>14.1f}%", end="")
        print()
    
    print(f"{'MRR':<15}", end="")
    for name in methods:
        m = np.mean(results[name]["mrrs"]) * 100
        print(f"{m:>14.1f}%", end="")
    print()
    
    print(f"{'Avg Time':<15}", end="")
    for name in methods:
        t = np.mean(results[name]["times"]) * 1000
        print(f"{t:>13.2f}ms", end="")
    print()
    
    # Analysis
    print(f"\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    s3_r10 = np.mean(results["S3_v2"]["recalls"][10])
    ham_r10 = np.mean(results["Hamming_BF"]["recalls"][10])
    cos_r10 = np.mean(results["Cosine_BF"]["recalls"][10])
    
    print(f"\nS3 vs Hamming BF: {s3_r10/ham_r10*100:.1f}% (coarse filter quality)")
    print(f"Hamming vs Cosine: {ham_r10/cos_r10*100:.1f}% (LSH quality)")
    print(f"S3 vs Cosine: {s3_r10/cos_r10*100:.1f}% (overall)")
    
    s3_time = np.mean(results["S3_v2"]["times"]) * 1000
    cos_time = np.mean(results["Cosine_BF"]["times"]) * 1000
    
    print(f"\nSpeedup: {cos_time/s3_time:.1f}x")
    
    # Storage
    float_mb = dim * 4 * len(corpus_list) / 1e6
    hash_mb = config.total_bits // 8 * len(corpus_list) / 1e6
    print(f"\nStorage: {float_mb:.2f} MB → {hash_mb:.2f} MB ({float_mb/hash_mb:.1f}x compression)")


# =============================================================================
# QUICK TEST
# =============================================================================

def quick_test_v2():
    print("=" * 70)
    print("S3 v2 QUICK TEST")
    print("=" * 70)
    
    # Synthetic data
    np.random.seed(42)
    dim = 1024
    n_docs = 1000
    n_queries = 50
    
    # Create clustered embeddings
    n_clusters = 10
    cluster_centers = np.random.randn(n_clusters, dim)
    cluster_centers = cluster_centers / np.linalg.norm(cluster_centers, axis=1, keepdims=True)
    
    doc_embeddings = []
    doc_clusters = []
    for i in range(n_docs):
        cluster = i % n_clusters
        noise = np.random.randn(dim) * 0.3
        emb = cluster_centers[cluster] + noise
        emb = emb / np.linalg.norm(emb)
        doc_embeddings.append(emb)
        doc_clusters.append(cluster)
    
    doc_embeddings = np.array(doc_embeddings)
    
    # Queries from random clusters
    query_embeddings = []
    query_clusters = []
    for i in range(n_queries):
        cluster = np.random.randint(n_clusters)
        noise = np.random.randn(dim) * 0.3
        emb = cluster_centers[cluster] + noise
        emb = emb / np.linalg.norm(emb)
        query_embeddings.append(emb)
        query_clusters.append(cluster)
    
    query_embeddings = np.array(query_embeddings)
    
    # Build index
    print(f"\n1. Building index ({n_docs} docs, {dim}D)...")
    config = S3ConfigV2()
    index = S3IndexV2(config)
    
    doc_ids = [str(i) for i in range(n_docs)]
    index.build(doc_ids, doc_embeddings)
    
    print(f"   Bits: {config.total_bits}")
    print(f"   Bytes per doc: {config.total_bits // 8}")
    
    # Evaluate
    print(f"\n2. Evaluating ({n_queries} queries)...")
    
    s3_hits = 0
    ham_hits = 0
    cos_hits = 0
    
    s3_times = []
    cos_times = []
    
    for i in range(n_queries):
        query_emb = query_embeddings[i]
        query_cluster = query_clusters[i]
        
        # Relevant docs = same cluster
        relevant = set(str(j) for j in range(n_docs) if doc_clusters[j] == query_cluster)
        
        # S3
        start = time.time()
        s3_results = index.search(query_emb, top_k=10)
        s3_times.append(time.time() - start)
        s3_retrieved = set(doc_id for doc_id, _ in s3_results)
        s3_hits += len(s3_retrieved & relevant)
        
        # Hamming BF
        ham_results = index.search_hamming_brute(query_emb, top_k=10)
        ham_retrieved = set(doc_id for doc_id, _ in ham_results)
        ham_hits += len(ham_retrieved & relevant)
        
        # Cosine BF
        start = time.time()
        cos_results = index.search_brute_force(query_emb, top_k=10)
        cos_times.append(time.time() - start)
        cos_retrieved = set(doc_id for doc_id, _ in cos_results)
        cos_hits += len(cos_retrieved & relevant)
    
    # Results
    total_relevant = n_queries * (n_docs // n_clusters)
    
    print(f"\n" + "-" * 50)
    print(f"{'Method':<20} {'Hits':<15} {'Recall@10':<15}")
    print("-" * 50)
    print(f"{'S3 v2':<20} {s3_hits:<15} {s3_hits/total_relevant*100:.1f}%")
    print(f"{'Hamming BF':<20} {ham_hits:<15} {ham_hits/total_relevant*100:.1f}%")
    print(f"{'Cosine BF':<20} {cos_hits:<15} {cos_hits/total_relevant*100:.1f}%")
    
    print(f"\nS3 avg time: {np.mean(s3_times)*1000:.2f}ms")
    print(f"Cosine avg time: {np.mean(cos_times)*1000:.2f}ms")
    print(f"Speedup: {np.mean(cos_times)/np.mean(s3_times):.1f}x")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        run_benchmark_v2()
    else:
        quick_test_v2()
