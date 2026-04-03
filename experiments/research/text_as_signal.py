#!/usr/bin/env python3
"""
Text-as-Signal: Replacing Chunking with Continuous Embeddings

Instead of: Text → Chunks → Embeddings → Vector Search
We try:     Text → Sliding Window → Embedding Signal → Correlation Search

The key insight: treat embeddings at each position as a continuous signal,
then use signal processing techniques (correlation, convolution) for retrieval.

This is exploratory/conceptual code.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Generator
from sklearn.feature_extraction.text import TfidfVectorizer
import re


# =============================================================================
# Text Tokenization
# =============================================================================

def tokenize(text: str) -> List[str]:
    """Simple word tokenization"""
    return re.findall(r'\b\w+\b', text.lower())


def sliding_windows(tokens: List[str], window_size: int, stride: int = 1) -> Generator[Tuple[int, List[str]], None, None]:
    """Generate sliding windows over tokens"""
    for i in range(0, len(tokens) - window_size + 1, stride):
        yield i, tokens[i:i + window_size]


# =============================================================================
# Document as Signal
# =============================================================================

@dataclass
class EmbeddingSignal:
    """
    Represents a document as a continuous embedding signal.
    
    Instead of one embedding per chunk, we have one embedding per position,
    creating a 2D matrix: (num_positions, embedding_dim)
    
    This is analogous to a spectrogram in audio processing.
    """
    text: str
    tokens: List[str]
    positions: np.ndarray      # Shape: (num_windows,) - token positions
    embeddings: np.ndarray     # Shape: (num_windows, embedding_dim)
    window_size: int
    stride: int
    
    @property
    def num_positions(self) -> int:
        return len(self.positions)
    
    @property
    def embedding_dim(self) -> int:
        return self.embeddings.shape[1]
    
    def get_embedding_at(self, position: int) -> np.ndarray:
        """Get embedding nearest to a token position"""
        idx = np.argmin(np.abs(self.positions - position))
        return self.embeddings[idx]
    
    def get_context_around(self, position: int, context_tokens: int = 50) -> str:
        """Get text context around a position"""
        start = max(0, position - context_tokens)
        end = min(len(self.tokens), position + context_tokens)
        return ' '.join(self.tokens[start:end])


class SignalEmbedder:
    """
    Creates continuous embedding signals from text.
    
    Think of this as creating a "spectrogram" of semantic content,
    where each time slice is an embedding of a local window.
    """
    
    def __init__(self, window_size: int = 20, stride: int = 5, embedding_dim: int = 100):
        self.window_size = window_size
        self.stride = stride
        self.embedding_dim = embedding_dim
        self.vectorizer = None
    
    def _fit_vectorizer(self, windows: List[str]):
        """Fit TF-IDF on all windows"""
        self.vectorizer = TfidfVectorizer(max_features=self.embedding_dim)
        self.vectorizer.fit(windows)
    
    def embed_document(self, text: str) -> EmbeddingSignal:
        """Convert document to embedding signal"""
        tokens = tokenize(text)
        
        if len(tokens) < self.window_size:
            # Document too short - pad or use as single window
            tokens = tokens + [''] * (self.window_size - len(tokens))
        
        # Generate all windows
        windows = list(sliding_windows(tokens, self.window_size, self.stride))
        positions = np.array([pos for pos, _ in windows])
        window_texts = [' '.join(toks) for _, toks in windows]
        
        # Fit vectorizer if needed
        if self.vectorizer is None:
            self._fit_vectorizer(window_texts)
        
        # Embed all windows
        embeddings = self.vectorizer.transform(window_texts).toarray()
        
        # Normalize each embedding
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms
        
        return EmbeddingSignal(
            text=text,
            tokens=tokens,
            positions=positions,
            embeddings=embeddings,
            window_size=self.window_size,
            stride=self.stride
        )
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query as a single vector"""
        if self.vectorizer is None:
            raise ValueError("Embedder not fitted. Call embed_document first.")
        
        vec = self.vectorizer.transform([query]).toarray()[0]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec


# =============================================================================
# Signal-Based Retrieval
# =============================================================================

def correlation_search(query_embedding: np.ndarray, 
                       doc_signal: EmbeddingSignal,
                       top_k: int = 5) -> List[Tuple[int, float, str]]:
    """
    Find positions in document where query correlates strongly.
    
    This is like matched filtering in signal processing:
    slide the query template across the document signal and
    find peaks in correlation.
    
    Returns: List of (position, score, context)
    """
    # Compute correlation at each position
    # This is simply dot product since both are normalized
    correlations = doc_signal.embeddings @ query_embedding
    
    # Find peaks (local maxima)
    results = []
    for i, corr in enumerate(correlations):
        pos = doc_signal.positions[i]
        context = doc_signal.get_context_around(pos, context_tokens=30)
        results.append((pos, corr, context))
    
    # Sort by correlation
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def resonance_search(query_embedding: np.ndarray,
                     query_phase: float,
                     doc_signal: EmbeddingSignal,
                     doc_phases: np.ndarray,
                     top_k: int = 5) -> List[Tuple[int, float, str]]:
    """
    Wave-based search with phase information.
    
    Extends correlation search by considering phase alignment.
    Positions where both amplitude AND phase match score highest.
    """
    # Amplitude correlation
    amp_correlation = doc_signal.embeddings @ query_embedding
    
    # Phase alignment: cos(φ_query - φ_doc)
    # 1.0 when phases match, -1.0 when opposite
    phase_alignment = np.cos(query_phase - doc_phases)
    
    # Combined resonance
    resonance = amp_correlation * phase_alignment
    
    results = []
    for i, res in enumerate(resonance):
        pos = doc_signal.positions[i]
        context = doc_signal.get_context_around(pos, context_tokens=30)
        results.append((pos, res, context))
    
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


# =============================================================================
# Frequency Analysis of Document
# =============================================================================

def compute_local_frequency(doc_signal: EmbeddingSignal) -> np.ndarray:
    """
    Estimate local "semantic frequency" at each position.
    
    High frequency = rapid semantic change (detailed, specific)
    Low frequency = slow semantic change (coherent, thematic)
    
    We measure this as the rate of change of embeddings.
    """
    if doc_signal.num_positions < 2:
        return np.zeros(doc_signal.num_positions)
    
    # Compute embedding differences between consecutive positions
    diffs = np.diff(doc_signal.embeddings, axis=0)
    
    # Magnitude of change = "instantaneous frequency"
    freqs = np.linalg.norm(diffs, axis=1)
    
    # Pad to match original length
    freqs = np.concatenate([[freqs[0]], freqs])
    
    return freqs


def compute_spectral_density(doc_signal: EmbeddingSignal) -> dict:
    """
    Compute spectral analysis of the embedding signal.
    
    Apply FFT to the embedding trajectory to find dominant
    "semantic frequencies" in the document.
    """
    # Take mean embedding at each position (reduce to 1D for FFT demo)
    # In practice, could do per-dimension FFT
    mean_signal = np.mean(doc_signal.embeddings, axis=1)
    
    # FFT
    spectrum = np.fft.fft(mean_signal)
    freqs = np.fft.fftfreq(len(mean_signal))
    
    # Power spectrum
    power = np.abs(spectrum) ** 2
    
    return {
        'frequencies': freqs[:len(freqs)//2],
        'power': power[:len(power)//2],
        'dominant_freq': freqs[np.argmax(power[1:]) + 1],  # Skip DC component
    }


# =============================================================================
# Phase Estimation for Signal
# =============================================================================

class SignalPhaseEstimator:
    """Estimate semantic phase at each position in a document signal"""
    
    NEGATION_PATTERN = re.compile(
        r'\b(not|no|never|neither|nobody|nothing|nowhere|none|'
        r"don't|doesn't|didn't|won't|wouldn't|couldn't|shouldn't|"
        r"can't|cannot|isn't|aren't|wasn't|weren't|hasn't|haven't|hadn't|"
        r'nicht|kein|keine|nie|niemals|niemand|ohne)\b',
        re.IGNORECASE
    )
    
    def estimate_phases(self, doc_signal: EmbeddingSignal) -> np.ndarray:
        """
        Estimate phase at each position based on local negation.
        
        Returns array of phases in radians.
        """
        phases = np.zeros(doc_signal.num_positions)
        
        for i, pos in enumerate(doc_signal.positions):
            # Get local context
            start = pos
            end = min(pos + doc_signal.window_size, len(doc_signal.tokens))
            local_text = ' '.join(doc_signal.tokens[start:end])
            
            # Check for negation
            if self.NEGATION_PATTERN.search(local_text):
                phases[i] = np.pi  # 180° shift for negation
        
        return phases


# =============================================================================
# Comparison: Chunking vs Signal
# =============================================================================

def demo_chunking_vs_signal():
    """Demonstrate the difference between chunking and signal approach"""
    
    document = """
    Machine learning is a subset of artificial intelligence. It enables computers 
    to learn from data without being explicitly programmed. Deep learning, a type 
    of machine learning, uses neural networks with many layers.
    
    However, machine learning is not magic. It requires large amounts of quality 
    data and significant computational resources. Without good data, machine learning 
    models will not perform well. The garbage in, garbage out principle applies strongly.
    
    Traditional programming differs from machine learning. In traditional programming,
    humans write explicit rules. In machine learning, the algorithm discovers patterns.
    This distinction is not always clear cut, as hybrid approaches exist.
    """
    
    print("=" * 80)
    print("DEMO: Chunking vs Signal Approach")
    print("=" * 80)
    
    # === Traditional Chunking ===
    print("\n### Traditional Chunking ###")
    chunk_size = 50  # words
    tokens = tokenize(document)
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk = ' '.join(tokens[i:i+chunk_size])
        chunks.append(chunk)
    
    print(f"Document: {len(tokens)} tokens")
    print(f"Chunks: {len(chunks)} chunks of ~{chunk_size} tokens")
    for i, chunk in enumerate(chunks):
        print(f"\n  Chunk {i+1}: \"{chunk[:60]}...\"")
    
    # === Signal Approach ===
    print("\n\n### Signal Approach ###")
    embedder = SignalEmbedder(window_size=15, stride=3)
    signal = embedder.embed_document(document)
    
    print(f"Signal: {signal.num_positions} positions, {signal.embedding_dim} dims")
    print(f"Window: {signal.window_size} tokens, stride: {signal.stride}")
    
    # Compute local frequency
    local_freq = compute_local_frequency(signal)
    print(f"\nLocal semantic frequency (rate of change):")
    print(f"  Mean: {np.mean(local_freq):.4f}")
    print(f"  Std:  {np.std(local_freq):.4f}")
    print(f"  Max at position {signal.positions[np.argmax(local_freq)]}: {np.max(local_freq):.4f}")
    
    # === Query Comparison ===
    print("\n\n### Query: 'machine learning not work' ###")
    query = "machine learning not work"
    query_emb = embedder.embed_query(query)
    
    print("\nCorrelation Search (finds similar regions):")
    results = correlation_search(query_emb, signal, top_k=3)
    for pos, score, context in results:
        print(f"  Position {pos:3d} (score {score:.3f}): ...{context}...")
    
    # With phase
    print("\nResonance Search (considers negation):")
    phase_estimator = SignalPhaseEstimator()
    doc_phases = phase_estimator.estimate_phases(signal)
    
    # Query has negation → phase = π
    query_phase = np.pi
    
    results = resonance_search(query_emb, query_phase, signal, doc_phases, top_k=3)
    for pos, score, context in results:
        print(f"  Position {pos:3d} (score {score:.3f}): ...{context}...")


def demo_find_specific_passage():
    """Demo: Find where a specific concept is discussed"""
    
    document = """
    The transformer architecture revolutionized natural language processing.
    Introduced in the paper "Attention is All You Need", transformers use
    self-attention mechanisms to process sequences in parallel.
    
    Unlike RNNs, transformers do not process tokens sequentially. This allows
    for much faster training on modern GPUs. However, transformers have
    quadratic memory complexity with respect to sequence length.
    
    BERT, GPT, and T5 are all based on the transformer architecture. BERT
    uses bidirectional attention, while GPT uses causal (left-to-right) attention.
    T5 treats all NLP tasks as text-to-text problems.
    
    The attention mechanism computes query, key, and value vectors. The output
    is a weighted sum of values, where weights come from query-key similarity.
    Multi-head attention allows the model to attend to different aspects.
    """
    
    print("\n" + "=" * 80)
    print("DEMO: Finding Specific Passages")
    print("=" * 80)
    
    embedder = SignalEmbedder(window_size=12, stride=2)
    signal = embedder.embed_document(document)
    
    queries = [
        "attention mechanism query key value",
        "BERT GPT comparison",
        "transformer memory complexity problem",
        "sequential processing RNN",
    ]
    
    for query in queries:
        print(f"\n### Query: '{query}' ###")
        query_emb = embedder.embed_query(query)
        results = correlation_search(query_emb, signal, top_k=1)
        
        for pos, score, context in results:
            print(f"  Best match (score {score:.3f}):")
            print(f"  ...{context}...")


def demo_semantic_frequency_map():
    """Visualize how semantic content changes through a document"""
    
    document = """
    Introduction: This paper presents our findings on climate change impacts.
    We analyzed temperature data from 1900 to 2023 across 150 weather stations.
    
    On January 15, 2023, Station Alpha recorded 42.3°C, breaking the previous
    record of 41.8°C set on February 2, 2019. Station Beta showed similar trends
    with peaks of 39.7°C on January 16 and 40.1°C on January 17, 2023.
    
    In conclusion, the data strongly suggests accelerating warming trends.
    Policy implications include the need for immediate emission reductions
    and adaptation strategies for vulnerable regions.
    """
    
    print("\n" + "=" * 80)
    print("DEMO: Semantic Frequency Analysis")
    print("=" * 80)
    print("High frequency = rapid semantic change (detailed data)")
    print("Low frequency = stable semantic content (abstract discussion)")
    
    embedder = SignalEmbedder(window_size=10, stride=2)
    signal = embedder.embed_document(document)
    local_freq = compute_local_frequency(signal)
    
    # Normalize for visualization
    freq_normalized = (local_freq - local_freq.min()) / (local_freq.max() - local_freq.min() + 1e-8)
    
    print("\nSemantic frequency map:")
    print("Position | Freq  | Content")
    print("-" * 70)
    
    for i in range(0, len(signal.positions), 3):  # Sample every 3rd
        pos = signal.positions[i]
        freq = freq_normalized[i]
        context = signal.get_context_around(pos, context_tokens=8)
        
        # Visual bar
        bar = "█" * int(freq * 20)
        print(f"{pos:4d}     | {bar:<20} | {context[:40]}...")
    
    # Find transitions
    print("\n\nDetected semantic transitions (high frequency points):")
    threshold = np.percentile(local_freq, 80)
    transitions = np.where(local_freq > threshold)[0]
    
    for idx in transitions[:5]:
        pos = signal.positions[idx]
        context = signal.get_context_around(pos, context_tokens=15)
        print(f"  Position {pos}: ...{context}...")


# =============================================================================
# Main
# =============================================================================

def main():
    print("Text-as-Signal: Replacing Chunking with Continuous Embeddings")
    print("=" * 70)
    
    demo_chunking_vs_signal()
    demo_find_specific_passage()
    demo_semantic_frequency_map()
    
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print("""
1. CONTINUOUS vs DISCRETE
   - Chunking creates arbitrary boundaries
   - Signal approach preserves continuity
   - Every position has an embedding (with overlap)

2. CORRELATION AS RETRIEVAL
   - Query "slides" across document signal
   - Finds peaks where content matches
   - No chunk can be "missed" due to boundaries

3. FREQUENCY AS METADATA
   - Rate of embedding change = semantic frequency
   - High freq regions: specific details, data, examples
   - Low freq regions: abstract discussion, summaries

4. PHASE INTEGRATION
   - Each position can have local phase (negation, etc.)
   - Resonance search considers both content AND phase
   - Better handling of nuanced semantic relationships

5. TRADE-OFFS
   - More embeddings to compute (window_size * stride factor)
   - Storage: O(doc_length / stride) instead of O(num_chunks)
   - But: finer granularity, no boundary artifacts

6. PRACTICAL CONSIDERATIONS
   - Window size ≈ expected query length works well
   - Stride controls resolution vs storage trade-off
   - Could use hierarchical: coarse signal + fine signal
""")


if __name__ == "__main__":
    main()
