#!/usr/bin/env python3
"""
Extended Gated Search Benchmark
- Test different noise levels
- Focus on the best performers
- Analyze WHY they work
"""

import numpy as np
from typing import List, Tuple
import time


def create_test_data(n_docs=3000, n_queries=150, n_clusters=15, dim=1024, noise_level=0.4):
    """Create clustered test data."""
    np.random.seed(42)
    
    centers = np.random.randn(n_clusters, dim)
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    
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
    
    return doc_ids, doc_embeddings, doc_clusters, query_embeddings, query_clusters


def brute_force(query, docs):
    """Brute force cosine similarity."""
    return docs @ query


def phase_coherent(query, docs):
    """Basic phase coherent: mean of sign matches."""
    return np.mean(np.sign(docs) * np.sign(query), axis=1)


def weighted_min(query, docs):
    """Weighted by min magnitude - the top performer."""
    q_sign = np.sign(query)
    q_mag = np.abs(query)
    
    d_signs = np.sign(docs)
    d_mags = np.abs(docs)
    
    phase_match = d_signs * q_sign
    weights = np.minimum(q_mag, d_mags)
    
    weighted = phase_match * weights
    return np.sum(weighted, axis=1) / (np.sum(weights, axis=1) + 1e-10)


def sidechain_gate(query, docs, threshold_pct=50):
    """Sidechain: only where BOTH are loud."""
    q_sign = np.sign(query)
    q_mag = np.abs(query)
    q_threshold = np.percentile(q_mag, threshold_pct)
    q_loud = q_mag > q_threshold
    
    d_mags = np.abs(docs)
    d_thresholds = np.percentile(d_mags, threshold_pct, axis=1, keepdims=True)
    d_loud = d_mags > d_thresholds
    
    phase_match = np.sign(docs) * q_sign
    combined_loud = d_loud & q_loud
    
    masked_sum = np.sum(phase_match * combined_loud, axis=1)
    mask_counts = np.sum(combined_loud, axis=1)
    
    return np.where(mask_counts > 0, masked_sum / mask_counts, 0)


def evaluate_recall(scores, query_clusters, doc_clusters, top_k=10):
    """Compute recall@k for a batch of queries."""
    recalls = []
    
    for i, score_row in enumerate(scores):
        relevant = set(j for j, c in enumerate(doc_clusters) if c == query_clusters[i])
        top_idx = np.argsort(score_row)[::-1][:top_k]
        hits = len(set(top_idx) & relevant)
        recalls.append(hits / min(len(relevant), top_k))
    
    return np.mean(recalls) * 100


def benchmark_noise_levels():
    """Benchmark across different noise levels."""
    
    print("=" * 70)
    print("NOISE LEVEL IMPACT ANALYSIS")
    print("=" * 70)
    
    noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    results = {
        "Brute Force": [],
        "Phase Coherent": [],
        "Weighted (min)": [],
        "Sidechain Gate": [],
    }
    
    for noise in noise_levels:
        print(f"\nNoise level: {noise}")
        
        doc_ids, doc_embeddings, doc_clusters, query_embeddings, query_clusters = \
            create_test_data(noise_level=noise)
        
        # Compute scores for all methods
        bf_scores = []
        pc_scores = []
        wm_scores = []
        sc_scores = []
        
        for query in query_embeddings:
            bf_scores.append(brute_force(query, doc_embeddings))
            pc_scores.append(phase_coherent(query, doc_embeddings))
            wm_scores.append(weighted_min(query, doc_embeddings))
            sc_scores.append(sidechain_gate(query, doc_embeddings))
        
        bf_scores = np.array(bf_scores)
        pc_scores = np.array(pc_scores)
        wm_scores = np.array(wm_scores)
        sc_scores = np.array(sc_scores)
        
        # Evaluate
        bf_recall = evaluate_recall(bf_scores, query_clusters, doc_clusters)
        pc_recall = evaluate_recall(pc_scores, query_clusters, doc_clusters)
        wm_recall = evaluate_recall(wm_scores, query_clusters, doc_clusters)
        sc_recall = evaluate_recall(sc_scores, query_clusters, doc_clusters)
        
        results["Brute Force"].append(bf_recall)
        results["Phase Coherent"].append(pc_recall)
        results["Weighted (min)"].append(wm_recall)
        results["Sidechain Gate"].append(sc_recall)
        
        print(f"  BF: {bf_recall:.1f}%, PC: {pc_recall:.1f}%, WM: {wm_recall:.1f}%, SC: {sc_recall:.1f}%")
    
    # Print table
    print("\n" + "=" * 70)
    print("RECALL@10 BY NOISE LEVEL")
    print("=" * 70)
    
    print(f"\n{'Noise':<10}", end="")
    for method in results.keys():
        print(f"{method:>15}", end="")
    print()
    print("-" * 70)
    
    for i, noise in enumerate(noise_levels):
        print(f"{noise:<10.1f}", end="")
        for method in results.keys():
            print(f"{results[method][i]:>14.1f}%", end="")
        print()
    
    # Print relative performance
    print("\n" + "=" * 70)
    print("RELATIVE TO BRUTE FORCE (%)")
    print("=" * 70)
    
    print(f"\n{'Noise':<10}", end="")
    for method in results.keys():
        if method != "Brute Force":
            print(f"{method:>15}", end="")
    print()
    print("-" * 60)
    
    for i, noise in enumerate(noise_levels):
        print(f"{noise:<10.1f}", end="")
        bf = results["Brute Force"][i]
        for method in results.keys():
            if method != "Brute Force":
                rel = results[method][i] / bf * 100 if bf > 0 else 0
                print(f"{rel:>14.1f}%", end="")
        print()
    
    return results


def analyze_weighted_min():
    """Deep analysis of why Weighted (min) works."""
    
    print("\n" + "=" * 70)
    print("WHY WEIGHTED (MIN) WORKS")
    print("=" * 70)
    
    # Create simple example
    np.random.seed(42)
    dim = 16  # Small for visualization
    
    # Two similar vectors
    a = np.array([0.9, 0.8, 0.1, -0.7, 0.2, 0.05, -0.3, 0.6,
                  0.4, -0.2, 0.1, 0.5, -0.1, 0.3, 0.2, -0.4])
    b = np.array([0.85, 0.75, -0.05, -0.6, 0.15, 0.1, -0.2, 0.55,
                  0.35, -0.15, 0.05, 0.4, 0.1, 0.25, 0.15, -0.35])
    
    # One dissimilar vector  
    c = np.array([-0.5, 0.3, 0.8, 0.2, -0.9, 0.4, 0.7, -0.3,
                  0.1, 0.6, -0.4, 0.2, 0.5, -0.7, 0.3, 0.1])
    
    # Normalize
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    c = c / np.linalg.norm(c)
    
    print("\nExample vectors (normalized):")
    print(f"  a: {a[:8]}...")
    print(f"  b: {b[:8]}... (similar to a)")
    print(f"  c: {c[:8]}... (different)")
    
    print("\n--- Cosine Similarity ---")
    print(f"  cos(a, b) = {np.dot(a, b):.4f}")
    print(f"  cos(a, c) = {np.dot(a, c):.4f}")
    
    print("\n--- Phase Coherent (mean of sign matches) ---")
    pc_ab = np.mean(np.sign(a) * np.sign(b))
    pc_ac = np.mean(np.sign(a) * np.sign(c))
    print(f"  PC(a, b) = {pc_ab:.4f}")
    print(f"  PC(a, c) = {pc_ac:.4f}")
    
    print("\n--- Weighted (min) ---")
    def wm(x, y):
        phase = np.sign(x) * np.sign(y)
        weights = np.minimum(np.abs(x), np.abs(y))
        return np.sum(phase * weights) / (np.sum(weights) + 1e-10)
    
    wm_ab = wm(a, b)
    wm_ac = wm(a, c)
    print(f"  WM(a, b) = {wm_ab:.4f}")
    print(f"  WM(a, c) = {wm_ac:.4f}")
    
    print("\n--- Analysis ---")
    print("""
    Weighted (min) funktioniert weil:
    
    1. STARKE Dimensionen zählen mehr
       - Wenn beide |q| und |d| groß sind → wichtig
       - Wenn einer klein ist → weniger wichtig
       
    2. min(|q|, |d|) ist wie ein SIDECHAIN GATE
       - Beide müssen "laut" sein damit es zählt
       - Ein "stilles" Signal kann nicht dominieren
       
    3. Es ist eine kontinuierliche Version von Sidechain Gate
       - Keine harte Threshold-Entscheidung
       - Sanfter Übergang
       
    4. Mathematisch: WM ≈ weighted cosine on matching dims
       - Ähnlich zu attention: focus on important dims
    """)


def analyze_gate_failure():
    """Why does Query Gate fail?"""
    
    print("\n" + "=" * 70)
    print("WHY QUERY GATE UNDERPERFORMS")
    print("=" * 70)
    
    print("""
    Query Gate (nur wo Query laut ist) versagt weil:
    
    1. INFORMATION LOSS
       - Query ignoriert Dimensionen wo Doc stark ist
       - Doc könnte wichtige Info in "stillen" Query-Dims haben
       
    2. ASYMMETRIE
       - Query Gate ist asymmetrisch
       - Aber semantische Ähnlichkeit ist symmetrisch
       
    3. RAUSCHEN VERSTÄRKT
       - Wenn Query in einer Dim "zufällig" laut ist (Noise)
       - Wird diese Dim überbewertet
       
    Sidechain Gate / Weighted (min) lösen das:
    - Beide müssen stark sein
    - Noise in einer Richtung wird nicht verstärkt
    - Symmetrisch(er)
    """)


if __name__ == "__main__":
    results = benchmark_noise_levels()
    analyze_weighted_min()
    analyze_gate_failure()
