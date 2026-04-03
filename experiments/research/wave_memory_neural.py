#!/usr/bin/env python3
"""
Wave-Based Semantic Memory - Neural Embedding Version

This version uses sentence-transformers for proper semantic embeddings.
The wave mechanics are identical to the TF-IDF version, but with much
better semantic understanding.

Requirements:
    pip install sentence-transformers numpy torch

Usage:
    python wave_memory_neural.py
    
For local embedding generation (like your llama.cpp setup):
    Replace SentenceTransformer with your local embedding function.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Callable
import re

# Try to import sentence-transformers, fall back gracefully
try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False
    print("sentence-transformers not installed. Install with:")
    print("  pip install sentence-transformers")


# =============================================================================
# Phase Predictors
# =============================================================================

class HeuristicPhasePredictor:
    """Rule-based phase prediction - works surprisingly well!"""
    
    NEGATION_PATTERNS = [
        r"\bnot\b", r"\bno\b", r"\bnever\b", r"\bneither\b", 
        r"\bnobody\b", r"\bnothing\b", r"\bnowhere\b", r"\bnone\b",
        r"\bwithout\b", r"\black\b", r"\bmissing\b", r"\babsent\b",
        r"\bdon't\b", r"\bdoesn't\b", r"\bdidn't\b", r"\bwon't\b",
        r"\bwouldn't\b", r"\bcouldn't\b", r"\bshouldn't\b", r"\bcan't\b",
        r"\bcannot\b", r"\bisn't\b", r"\baren't\b", r"\bwasn't\b",
        r"\bweren't\b", r"\bhasn't\b", r"\bhaven't\b", r"\bhadn't\b",
        # German
        r"\bnicht\b", r"\bkein\b", r"\bkeine\b", r"\bkeiner\b",
        r"\bnie\b", r"\bniemals\b", r"\bniemand\b", r"\bohne\b",
    ]
    
    NEGATIVE_SENTIMENT = {
        'bad', 'terrible', 'awful', 'horrible', 'poor', 'worst', 'hate',
        'disappointing', 'failed', 'failure', 'wrong', 'broken', 'sucks',
        'schlecht', 'schrecklich', 'furchtbar', 'schlimm', 'hasse'
    }
    
    def __init__(self):
        self.negation_regex = re.compile(
            '|'.join(self.NEGATION_PATTERNS), 
            re.IGNORECASE
        )
    
    def predict(self, text: str, embedding: Optional[np.ndarray] = None) -> float:
        """Return phase in radians [0, 2π)"""
        phase = 0.0
        text_lower = text.lower()
        
        # Negation: π shift (180°)
        if self.negation_regex.search(text_lower):
            phase += np.pi
        
        # Question: π/2 shift (90°) - optional, can disable
        # if text.strip().endswith('?'):
        #     phase += np.pi / 2
        
        # Negative sentiment: π/4 shift (45°)
        words = set(text_lower.split())
        if words & self.NEGATIVE_SENTIMENT:
            phase += np.pi / 4
        
        return phase % (2 * np.pi)


class LearnedPhasePredictor:
    """
    Placeholder for a learned phase predictor.
    
    In production, this would be a small neural network trained on:
    - NLI contradiction pairs (SNLI, MNLI)
    - Sentiment reversal pairs
    - Negation pairs
    
    Training objective: phase_diff(anchor, positive) ≈ 0
                       phase_diff(anchor, negative) ≈ π
    """
    
    def __init__(self, input_dim: int = 384):
        self.input_dim = input_dim
        # Placeholder - would load trained weights
        self.weights = np.random.randn(input_dim) * 0.01
    
    def predict(self, text: str, embedding: np.ndarray) -> float:
        # Simple linear projection for demo
        z = np.dot(embedding, self.weights)
        sigmoid = 1 / (1 + np.exp(-z))
        return float(sigmoid * 2 * np.pi)


# =============================================================================
# Wave Embedder
# =============================================================================

class WaveEmbedder:
    """
    Combines amplitude (from embedding model) with phase (predicted).
    
    The key insight: traditional embeddings only capture magnitude/direction.
    By adding phase, we can encode additional semantic information like
    negation, polarity, and modality.
    """
    
    def __init__(
        self, 
        model_name: str = 'all-MiniLM-L6-v2',
        phase_predictor: str = 'heuristic',
        custom_embed_fn: Optional[Callable[[str], np.ndarray]] = None
    ):
        """
        Args:
            model_name: sentence-transformers model name
            phase_predictor: 'heuristic' or 'learned'
            custom_embed_fn: Optional custom embedding function (for llama.cpp etc)
        """
        self.custom_embed_fn = custom_embed_fn
        
        if custom_embed_fn is None:
            if not HAS_SBERT:
                raise ImportError("sentence-transformers required. Install or provide custom_embed_fn")
            print(f"Loading embedding model: {model_name}")
            self.embedder = SentenceTransformer(model_name)
            self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
        else:
            print("Using custom embedding function")
            self.embedder = None
            # Probe dimension
            test_emb = custom_embed_fn("test")
            self.embedding_dim = len(test_emb)
        
        print(f"  Embedding dimension: {self.embedding_dim}")
        
        if phase_predictor == 'heuristic':
            self.phase_predictor = HeuristicPhasePredictor()
        else:
            self.phase_predictor = LearnedPhasePredictor(self.embedding_dim)
    
    def encode_amplitude(self, text: str) -> np.ndarray:
        """Get traditional real-valued embedding (normalized)"""
        if self.custom_embed_fn:
            emb = self.custom_embed_fn(text)
        else:
            emb = self.embedder.encode(text, normalize_embeddings=True)
        
        # Ensure normalized
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb
    
    def encode_wave(self, text: str) -> np.ndarray:
        """Get complex-valued wave embedding"""
        amplitude = self.encode_amplitude(text)
        phase = self.phase_predictor.predict(text, amplitude)
        
        # z = A * e^(iφ) - global phase applied to all dimensions
        wave = amplitude.astype(np.complex128) * np.exp(1j * phase)
        return wave
    
    def get_phase(self, text: str) -> float:
        """Get just the phase for a text (for debugging)"""
        amplitude = self.encode_amplitude(text)
        return self.phase_predictor.predict(text, amplitude)


# =============================================================================
# Similarity Functions
# =============================================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Traditional cosine similarity for real vectors"""
    a_real = np.real(a) if np.iscomplexobj(a) else a
    b_real = np.real(b) if np.iscomplexobj(b) else b
    return float(np.dot(a_real, b_real) / (np.linalg.norm(a_real) * np.linalg.norm(b_real) + 1e-8))


def resonance_score(query_wave: np.ndarray, doc_wave: np.ndarray) -> float:
    """
    Wave interference-based similarity.
    
    Returns REAL part of complex dot product:
    - Positive when phases align (constructive interference)
    - Negative when phases oppose (destructive interference)
    - Zero when phases are 90° apart
    """
    interference = np.sum(query_wave * np.conj(doc_wave))
    return float(np.real(interference))


def resonance_magnitude(query_wave: np.ndarray, doc_wave: np.ndarray) -> float:
    """Absolute resonance (always positive, for ranking only)"""
    interference = np.sum(query_wave * np.conj(doc_wave))
    return float(np.abs(interference))


# =============================================================================
# Search Interface
# =============================================================================

@dataclass
class WaveSearchResult:
    text: str
    cosine: float
    resonance: float
    phase_deg: float
    
    def __str__(self):
        return f"{self.text[:50]:<52} cos={self.cosine:>6.3f}  res={self.resonance:>7.3f}  φ={self.phase_deg:>6.1f}°"


class WaveSearchIndex:
    """Simple in-memory search index with wave embeddings"""
    
    def __init__(self, embedder: WaveEmbedder):
        self.embedder = embedder
        self.documents: List[str] = []
        self.amplitudes: List[np.ndarray] = []
        self.waves: List[np.ndarray] = []
        self.phases: List[float] = []
    
    def add(self, text: str):
        """Add a document to the index"""
        self.documents.append(text)
        amp = self.embedder.encode_amplitude(text)
        wave = self.embedder.encode_wave(text)
        phase = self.embedder.get_phase(text)
        
        self.amplitudes.append(amp)
        self.waves.append(wave)
        self.phases.append(phase)
    
    def add_many(self, texts: List[str]):
        """Add multiple documents"""
        for text in texts:
            self.add(text)
    
    def search(self, query: str, top_k: int = 10, use_resonance: bool = True) -> List[WaveSearchResult]:
        """
        Search the index.
        
        Args:
            query: Search query
            top_k: Number of results to return
            use_resonance: If True, rank by resonance. If False, rank by cosine.
        """
        query_amp = self.embedder.encode_amplitude(query)
        query_wave = self.embedder.encode_wave(query)
        query_phase = self.embedder.get_phase(query)
        
        results = []
        for i, doc in enumerate(self.documents):
            cos = cosine_similarity(query_amp, self.amplitudes[i])
            res = resonance_score(query_wave, self.waves[i])
            
            results.append(WaveSearchResult(
                text=doc,
                cosine=cos,
                resonance=res,
                phase_deg=np.degrees(self.phases[i])
            ))
        
        # Sort by chosen metric
        key_fn = (lambda r: r.resonance) if use_resonance else (lambda r: r.cosine)
        results.sort(key=key_fn, reverse=True)
        
        return results[:top_k]
    
    def compare_rankings(self, query: str, top_k: int = 10):
        """Show side-by-side ranking comparison"""
        query_phase = self.embedder.get_phase(query)
        
        print(f"\nQuery: \"{query}\" (phase: {np.degrees(query_phase):.1f}°)")
        print("=" * 100)
        
        # Get both rankings
        cos_results = self.search(query, top_k, use_resonance=False)
        res_results = self.search(query, top_k, use_resonance=True)
        
        print(f"{'COSINE RANKING':<50} {'RESONANCE RANKING':<50}")
        print("-" * 100)
        
        for i in range(min(top_k, len(cos_results))):
            cos_r = cos_results[i]
            res_r = res_results[i]
            
            cos_str = f"{i+1}. {cos_r.text[:35]:<37} ({cos_r.cosine:.3f})"
            res_str = f"{i+1}. {res_r.text[:35]:<37} ({res_r.resonance:.3f})"
            
            # Highlight if different
            marker = " ⚡" if cos_r.text != res_r.text else ""
            print(f"{cos_str:<50} {res_str:<50}{marker}")


# =============================================================================
# Demo / Tests
# =============================================================================

def demo_negation_handling():
    """Demonstrate how wave-based handles negation better"""
    print("\n" + "=" * 80)
    print("DEMO: Negation Handling")
    print("=" * 80)
    
    embedder = WaveEmbedder()
    index = WaveSearchIndex(embedder)
    
    # Add documents with various negations
    docs = [
        "I love this restaurant",
        "I don't love this restaurant",
        "This restaurant is great",
        "This restaurant is not great",
        "The food here is excellent",
        "The food here is not good",
        "I would recommend this place",
        "I would not recommend this place",
        "The service was friendly",
        "The service was terrible",
    ]
    index.add_many(docs)
    
    # Search
    queries = [
        "good restaurant recommendation",
        "restaurant I should avoid",
    ]
    
    for query in queries:
        index.compare_rankings(query, top_k=5)


def demo_sentiment_polarity():
    """Demonstrate sentiment polarity handling"""
    print("\n" + "=" * 80)
    print("DEMO: Sentiment Polarity")
    print("=" * 80)
    
    embedder = WaveEmbedder()
    index = WaveSearchIndex(embedder)
    
    docs = [
        "The movie was amazing and I loved every minute",
        "The movie was terrible and I hated it",
        "The movie was okay, nothing special",
        "I did not enjoy this movie at all",
        "This film exceeded all my expectations",
        "This film was a complete disappointment",
        "Great cinematography and acting",
        "Poor script and bad directing",
    ]
    index.add_many(docs)
    
    index.compare_rankings("I want to watch a good movie", top_k=5)
    index.compare_rankings("movies to avoid", top_k=5)


def demo_phase_math():
    """Show the math behind phase and interference"""
    print("\n" + "=" * 80)
    print("DEMO: Phase Mathematics")
    print("=" * 80)
    
    embedder = WaveEmbedder()
    
    pairs = [
        ("I like coffee", "I don't like coffee"),
        ("The weather is nice", "The weather is not nice"),
        ("This works well", "This doesn't work"),
        ("Good morning", "Bad morning"),
    ]
    
    print(f"{'Text A':<30} {'Text B':<30} {'Phase A':>8} {'Phase B':>8} {'Δφ':>8} {'Cosine':>8} {'Reson.':>8}")
    print("-" * 112)
    
    for text_a, text_b in pairs:
        amp_a = embedder.encode_amplitude(text_a)
        amp_b = embedder.encode_amplitude(text_b)
        wave_a = embedder.encode_wave(text_a)
        wave_b = embedder.encode_wave(text_b)
        
        phase_a = np.degrees(embedder.get_phase(text_a))
        phase_b = np.degrees(embedder.get_phase(text_b))
        phase_diff = phase_b - phase_a
        
        cos = cosine_similarity(amp_a, amp_b)
        res = resonance_score(wave_a, wave_b)
        
        print(f"{text_a:<30} {text_b:<30} {phase_a:>7.1f}° {phase_b:>7.1f}° {phase_diff:>7.1f}° {cos:>8.3f} {res:>8.3f}")
    
    print("\nNote: When Δφ ≈ 180°, resonance flips sign compared to cosine!")


def main():
    """Run all demos"""
    print("Wave-Based Semantic Memory - Neural Version")
    print("=" * 50)
    
    if not HAS_SBERT:
        print("\nTo run with real embeddings, install sentence-transformers:")
        print("  pip install sentence-transformers")
        print("\nOr provide a custom embedding function (e.g., for llama.cpp)")
        return
    
    demo_phase_math()
    demo_negation_handling()
    demo_sentiment_polarity()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Key findings:

1. NEGATION DETECTION: Wave resonance gives negative scores to negated 
   content, while cosine similarity sees it as similar.

2. SENTIMENT: Negative sentiment words add a 45° phase shift, reducing
   resonance with positive queries.

3. PRACTICAL VALUE: Most useful when filtering or when exact matches
   don't exist. The ranking changes can be significant.

4. LIMITATIONS: 
   - Heuristic phase relies on keyword detection (can miss subtle negation)
   - Learned phase predictor needs training data
   - Complex numbers double storage requirements

5. INTEGRATION: Works with any embedding model - just wrap the amplitude
   with phase information.
""")


if __name__ == "__main__":
    main()
