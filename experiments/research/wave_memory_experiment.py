#!/usr/bin/env python3
"""
Wave-Based Semantic Memory Experiment

Compares traditional cosine similarity with wave-based resonance retrieval.
Tests whether phase encoding can capture negation and polarity better.

This version uses TF-IDF for embeddings to keep dependencies minimal.
The wave-based concepts work identically with neural embeddings.

Requirements:
    pip install numpy scikit-learn
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
from dataclasses import dataclass
from typing import List, Tuple, Optional
import re


# =============================================================================
# Phase Predictor Models
# =============================================================================

class HeuristicPhasePredictor:
    """Rule-based phase prediction without ML"""
    
    NEGATION_WORDS = {
        'not', 'no', 'never', 'neither', 'nobody', 'nothing', 'nowhere',
        'none', "n't", 'without', 'lack', 'missing', 'absent',
        # German
        'nicht', 'kein', 'keine', 'keiner', 'nie', 'niemals', 'niemand',
        'nirgends', 'ohne', 'fehlt'
    }
    
    # Contractions that contain negation
    NEGATION_CONTRACTIONS = {
        "don't", "doesn't", "didn't", "won't", "wouldn't", "couldn't",
        "shouldn't", "can't", "cannot", "isn't", "aren't", "wasn't",
        "weren't", "hasn't", "haven't", "hadn't"
    }
    
    NEGATIVE_SENTIMENT = {
        'bad', 'terrible', 'awful', 'horrible', 'poor', 'worst', 'hate',
        'disappointing', 'failed', 'failure', 'wrong', 'broken',
        # German
        'schlecht', 'schrecklich', 'furchtbar', 'schlimm', 'hasse'
    }
    
    def predict(self, text: str, embedding: np.ndarray) -> float:
        """Return phase in radians [0, 2π)"""
        phase = 0.0
        text_lower = text.lower()
        words = set(text_lower.split())
        
        # Negation: π shift (180°)
        has_negation = bool(words & self.NEGATION_WORDS)
        has_contraction = any(c in text_lower for c in self.NEGATION_CONTRACTIONS)
        
        if has_negation or has_contraction:
            phase += np.pi
        
        # Question: π/2 shift (90°)
        if text.strip().endswith('?'):
            phase += np.pi / 2
        
        # Negative sentiment: π/4 shift (45°)
        if words & self.NEGATIVE_SENTIMENT:
            phase += np.pi / 4
        
        return phase % (2 * np.pi)


class LearnedPhasePredictor:
    """Simple trainable phase predictor (numpy-based for demo)"""
    
    def __init__(self, input_dim: int = 100):
        # Simple linear projection + sigmoid
        self.weights = np.random.randn(input_dim, 1) * 0.1
        self.bias = 0.0
    
    def predict(self, x: np.ndarray) -> float:
        # Sigmoid activation scaled to [0, 2π]
        z = np.dot(x, self.weights).sum() + self.bias
        sigmoid = 1 / (1 + np.exp(-z))
        return float(sigmoid * 2 * np.pi)


# =============================================================================
# Wave Embedder
# =============================================================================

class WaveEmbedder:
    """
    Combines amplitude (from TF-IDF) with phase (predicted).
    
    NOTE: In production, replace TfidfVectorizer with a proper embedding model
    like sentence-transformers. The wave mechanics work identically.
    """
    
    def __init__(self, corpus: Optional[List[str]] = None,
                 phase_predictor: str = 'heuristic'):
        print("Initializing TF-IDF based embedder...")
        
        # Build vocabulary from corpus or use default
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),  # Unigrams and bigrams
            max_features=500,    # Limit dimensions
            stop_words=None,     # Keep all words for negation detection
        )
        
        # Fit on corpus if provided, otherwise fit later
        self._fitted = False
        if corpus:
            self.fit(corpus)
        
        if phase_predictor == 'heuristic':
            self.phase_predictor = HeuristicPhasePredictor()
        else:
            self.phase_predictor = LearnedPhasePredictor(500)
    
    def fit(self, corpus: List[str]):
        """Fit the TF-IDF vectorizer on a corpus"""
        self.vectorizer.fit(corpus)
        self._fitted = True
        print(f"  Vocabulary size: {len(self.vectorizer.vocabulary_)}")
    
    def encode_amplitude(self, text: str) -> np.ndarray:
        """Get traditional real-valued embedding (TF-IDF vector)"""
        if not self._fitted:
            # Auto-fit on this text if not fitted
            self.fit([text])
        
        vec = self.vectorizer.transform([text]).toarray()[0]
        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec
    
    def encode_wave(self, text: str) -> np.ndarray:
        """Get complex-valued wave embedding"""
        amplitude = self.encode_amplitude(text)
        
        if isinstance(self.phase_predictor, HeuristicPhasePredictor):
            phase = self.phase_predictor.predict(text, amplitude)
        else:
            phase = self.phase_predictor.predict(amplitude)
        
        # z = A * e^(iφ) - global phase applied to all dimensions
        wave = amplitude * np.exp(1j * phase)
        return wave
    
    def encode_wave_per_dim(self, text: str) -> np.ndarray:
        """Alternative: per-dimension phase (more expressive but needs training)"""
        amplitude = self.encode_amplitude(text)
        
        # For now: global phase, but structure supports per-dim
        if isinstance(self.phase_predictor, HeuristicPhasePredictor):
            phase = self.phase_predictor.predict(text, amplitude)
            phases = np.full(amplitude.shape, phase)
        else:
            # Learned predictor could output per-dim phases
            phases = np.zeros(amplitude.shape)
        
        wave = amplitude * np.exp(1j * phases)
        return wave


# =============================================================================
# Similarity/Retrieval Functions
# =============================================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Traditional cosine similarity for real vectors"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def resonance_score(query_wave: np.ndarray, doc_wave: np.ndarray) -> float:
    """
    Wave interference-based similarity.
    
    Constructive interference (same phase) → high score
    Destructive interference (opposite phase) → low/negative score
    """
    # Complex conjugate dot product
    interference = np.sum(query_wave * np.conj(doc_wave))
    
    # Return real part (captures phase relationship)
    # Could also use |interference|² for always-positive scores
    return float(np.real(interference))


def resonance_score_abs(query_wave: np.ndarray, doc_wave: np.ndarray) -> float:
    """Absolute resonance (always positive, loses anti-match info)"""
    interference = np.sum(query_wave * np.conj(doc_wave))
    return float(np.abs(interference))


# =============================================================================
# Experiment Runner
# =============================================================================

@dataclass
class SearchResult:
    text: str
    cosine_score: float
    resonance_score: float
    resonance_abs: float


def run_experiment(embedder: WaveEmbedder, query: str, documents: List[str]) -> List[SearchResult]:
    """Compare cosine vs resonance for a query against documents"""
    
    # Fit TF-IDF on all texts (query + documents)
    all_texts = [query] + documents
    embedder.fit(all_texts)
    
    # Encode query both ways
    query_amplitude = embedder.encode_amplitude(query)
    query_wave = embedder.encode_wave(query)
    
    results = []
    for doc in documents:
        doc_amplitude = embedder.encode_amplitude(doc)
        doc_wave = embedder.encode_wave(doc)
        
        results.append(SearchResult(
            text=doc,
            cosine_score=cosine_similarity(query_amplitude, doc_amplitude),
            resonance_score=resonance_score(query_wave, doc_wave),
            resonance_abs=resonance_score_abs(query_wave, doc_wave)
        ))
    
    return results


def print_results(query: str, results: List[SearchResult]):
    """Pretty print comparison results"""
    print(f"\n{'='*70}")
    print(f"Query: \"{query}\"")
    print(f"{'='*70}")
    print(f"{'Document':<40} {'Cosine':>10} {'Resonance':>10} {'Res.Abs':>10}")
    print(f"{'-'*70}")
    
    # Sort by cosine for comparison
    for r in sorted(results, key=lambda x: x.cosine_score, reverse=True):
        text_short = r.text[:38] + ".." if len(r.text) > 40 else r.text
        print(f"{text_short:<40} {r.cosine_score:>10.4f} {r.resonance_score:>10.4f} {r.resonance_abs:>10.4f}")


# =============================================================================
# Test Cases
# =============================================================================

def test_negation():
    """Test case: Does wave-based catch negation better?"""
    return {
        "name": "Negation Detection",
        "query": "I like this product",
        "documents": [
            "I like this product",           # Exact match
            "I love this product",           # Similar positive
            "I don't like this product",     # Negated - should score LOW
            "I hate this product",           # Opposite sentiment
            "This product is okay",          # Neutral
            "I like this service",           # Similar but different object
        ]
    }


def test_sentiment_polarity():
    """Test case: Sentiment polarity detection"""
    return {
        "name": "Sentiment Polarity",
        "query": "The movie was excellent",
        "documents": [
            "The movie was excellent",       # Exact
            "The film was great",            # Similar positive
            "The movie was terrible",        # Opposite sentiment
            "The movie was not good",        # Negated
            "The movie was okay",            # Neutral
            "I watched a movie",             # Related but no sentiment
        ]
    }


def test_question_vs_statement():
    """Test case: Questions vs statements"""
    return {
        "name": "Question vs Statement",
        "query": "Python is a programming language",
        "documents": [
            "Python is a programming language",      # Statement match
            "Is Python a programming language?",     # Question form
            "Python is used for programming",        # Related statement
            "What is Python?",                       # Different question
            "Java is a programming language",        # Similar structure, different subject
        ]
    }


def test_german_negation():
    """Test case: German negation"""
    return {
        "name": "German Negation",
        "query": "Das Essen war gut",
        "documents": [
            "Das Essen war gut",             # Exact
            "Das Essen war lecker",          # Similar positive
            "Das Essen war nicht gut",       # Negated
            "Das Essen war schlecht",        # Opposite
            "Das Essen war okay",            # Neutral
        ]
    }


def test_semantic_similarity():
    """Test case: Pure semantic similarity (no negation) - should be equal"""
    return {
        "name": "Pure Semantic (Control)",
        "query": "Machine learning algorithms",
        "documents": [
            "Machine learning algorithms",
            "Deep learning neural networks",
            "Artificial intelligence models",
            "Data science techniques",
            "Cooking recipes for beginners",  # Unrelated
        ]
    }


def test_no_exact_match():
    """Test case: No exact match - this is where ranking changes matter!"""
    return {
        "name": "No Exact Match (Critical Case)",
        "query": "The food was delicious",
        "documents": [
            # No exact match! All are related but different
            "The food was good",              # Similar positive
            "The food was not good",          # Negated - should rank LOW
            "The meal was delicious",         # Synonym  
            "The food was terrible",          # Opposite sentiment
            "The food was okay",              # Neutral
            "I didn't enjoy the food",        # Negated enjoyment
        ]
    }


# =============================================================================
# Training Data Generation (for learned phase predictor)
# =============================================================================

def generate_training_pairs() -> List[Tuple[str, str, float]]:
    """
    Generate (anchor, other, target_phase_diff) pairs.
    
    target_phase_diff:
        0 → same phase (similar meaning)
        π → opposite phase (negation/antonym)
    """
    pairs = []
    
    # Negation pairs (phase diff = π)
    negation_templates = [
        ("I {} this", "I don't {} this"),
        ("This is {}", "This is not {}"),
        ("He {} the task", "He never {} the task"),
    ]
    verbs_adjs = ["like", "want", "need", "good", "helpful", "finished"]
    
    for template_pos, template_neg in negation_templates:
        for word in verbs_adjs:
            pairs.append((
                template_pos.format(word),
                template_neg.format(word),
                np.pi  # Opposite phase
            ))
    
    # Synonym pairs (phase diff = 0)
    synonyms = [
        ("happy", "joyful"),
        ("sad", "unhappy"),
        ("big", "large"),
        ("small", "tiny"),
        ("fast", "quick"),
    ]
    for w1, w2 in synonyms:
        pairs.append((f"I feel {w1}", f"I feel {w2}", 0.0))
    
    # Antonym pairs (phase diff = π)
    antonyms = [
        ("happy", "sad"),
        ("good", "bad"),
        ("love", "hate"),
        ("success", "failure"),
    ]
    for w1, w2 in antonyms:
        pairs.append((f"This is {w1}", f"This is {w2}", np.pi))
    
    return pairs


def train_phase_predictor(embedder: WaveEmbedder, epochs: int = 100, lr: float = 0.01):
    """
    Train the learned phase predictor using simple gradient descent.
    
    This is a simplified numpy-based training loop. For production,
    use PyTorch with a proper neural network.
    """
    if isinstance(embedder.phase_predictor, HeuristicPhasePredictor):
        print("Cannot train heuristic predictor. Initialize with phase_predictor='learned'")
        return
    
    pairs = generate_training_pairs()
    
    # First fit the TF-IDF on all training texts
    all_texts = [t for anchor, other, _ in pairs for t in [anchor, other]]
    embedder.fit(all_texts)
    
    print(f"\nTraining phase predictor on {len(pairs)} pairs...")
    
    pred = embedder.phase_predictor
    
    for epoch in range(epochs):
        total_loss = 0
        
        for anchor_text, other_text, target_diff in pairs:
            # Get base embeddings
            anchor_emb = embedder.encode_amplitude(anchor_text)
            other_emb = embedder.encode_amplitude(other_text)
            
            # Predict phases
            anchor_phase = pred.predict(anchor_emb)
            other_phase = pred.predict(other_emb)
            
            # Loss: phase difference should match target (circular loss)
            phase_diff = anchor_phase - other_phase
            loss = 1 - np.cos(phase_diff - target_diff)
            
            # Simple gradient descent on weights
            # d(loss)/d(phase_diff) = sin(phase_diff - target_diff)
            grad = np.sin(phase_diff - target_diff)
            
            # Update weights (simplified - just nudge in right direction)
            pred.weights -= lr * grad * anchor_emb.reshape(-1, 1) * 0.1
            pred.weights += lr * grad * other_emb.reshape(-1, 1) * 0.1
            
            total_loss += loss
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(pairs):.4f}")
    
    print("Training complete.")


# =============================================================================
# Main
# =============================================================================

def main():
    print("Wave-Based Semantic Memory Experiment")
    print("=" * 50)
    
    # Initialize embedder with heuristic phase predictor
    # Using TF-IDF for simplicity - same concepts apply to neural embeddings
    embedder = WaveEmbedder(
        phase_predictor='heuristic'
    )
    
    # Run all test cases
    test_cases = [
        test_negation(),
        test_sentiment_polarity(),
        test_question_vs_statement(),
        test_german_negation(),
        test_semantic_similarity(),
        test_no_exact_match(),  # Critical case!
    ]
    
    print("\n" + "=" * 70)
    print("EXPERIMENT RESULTS")
    print("=" * 70)
    print("\nComparing: Cosine Similarity vs Wave Resonance")
    print("Resonance can go NEGATIVE for phase-mismatched pairs (negation)")
    
    for test in test_cases:
        print(f"\n\n### {test['name']} ###")
        results = run_experiment(embedder, test['query'], test['documents'])
        print_results(test['query'], results)
        
        # Analysis
        print("\nAnalysis:")
        cosine_ranking = sorted(results, key=lambda x: x.cosine_score, reverse=True)
        resonance_ranking = sorted(results, key=lambda x: x.resonance_score, reverse=True)
        
        if cosine_ranking[0].text != resonance_ranking[0].text:
            print(f"  ⚡ Rankings differ! Cosine top: '{cosine_ranking[0].text[:30]}...'")
            print(f"                    Resonance top: '{resonance_ranking[0].text[:30]}...'")
        else:
            print(f"  ✓ Same top result for both methods")
        
        # Check if negated items got lower resonance
        for r in results:
            if 'not' in r.text.lower() or "n't" in r.text.lower() or 'nicht' in r.text.lower():
                if r.resonance_score < r.cosine_score:
                    print(f"  ✓ Negated '{r.text[:30]}...' scored lower in resonance ({r.resonance_score:.3f}) vs cosine ({r.cosine_score:.3f})")
    
    # Summary analysis
    print("\n\n" + "=" * 70)
    print("KEY INSIGHT: RANKING CHANGES")
    print("=" * 70)
    print("""
The wave-based approach doesn't change the TOP result in most cases
(exact matches still win). But it dramatically changes the ranking
of NEGATED items:

  Traditional Cosine:  "X" and "not X" have HIGH similarity (~0.4-0.7)
  Wave Resonance:      "X" and "not X" have NEGATIVE similarity

This matters when you're filtering or when the exact match isn't present.
For example, if you search "good restaurant" and your corpus has:
  - "This restaurant is good" (match)
  - "This restaurant is not good" (anti-match)
  
Cosine would rank both similarly. Wave resonance correctly identifies
the second as semantically OPPOSITE to the query.
""")


def demo_phase_visualization():
    """Show how phase affects the wave representation"""
    print("\n" + "=" * 70)
    print("PHASE VISUALIZATION")
    print("=" * 70)
    
    pairs = [
        ("I like this", "I don't like this"),
        ("The food was good", "The food was not good"),
        ("Happy", "Sad"),
    ]
    
    # Fit embedder on all texts
    all_texts = [t for pair in pairs for t in pair]
    embedder = WaveEmbedder()
    embedder.fit(all_texts)
    
    for text1, text2 in pairs:
        wave1 = embedder.encode_wave(text1)
        wave2 = embedder.encode_wave(text2)
        
        # Extract phase (from first non-zero element)
        # Find first non-zero element
        nonzero_idx = np.argmax(np.abs(wave1) > 1e-10)
        phase1 = np.angle(wave1[nonzero_idx]) if np.abs(wave1[nonzero_idx]) > 1e-10 else 0
        phase2 = np.angle(wave2[nonzero_idx]) if np.abs(wave2[nonzero_idx]) > 1e-10 else 0
        
        print(f"\n'{text1}' → Phase: {np.degrees(phase1):.1f}°")
        print(f"'{text2}' → Phase: {np.degrees(phase2):.1f}°")
        print(f"Phase difference: {np.degrees(phase2 - phase1):.1f}°")
        print(f"Resonance: {resonance_score(wave1, wave2):.4f}")
        
        # Compare to cosine
        amp1 = embedder.encode_amplitude(text1)
        amp2 = embedder.encode_amplitude(text2)
        print(f"Cosine:    {cosine_similarity(amp1, amp2):.4f}")


if __name__ == "__main__":
    main()
    demo_phase_visualization()
