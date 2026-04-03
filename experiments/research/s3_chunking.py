#!/usr/bin/env python3
"""
S3 Chunking Module

Implements text chunking strategies discussed:
1. Sliding Window (simple baseline)
2. Sentence-based (semantic boundaries)
3. MDCT-Style (overlapping windows with smooth transitions)
4. Onset Detection (adaptive chunking based on topic changes)

Author: Claude & Toby
Date: December 2024
"""

import re
from dataclasses import dataclass
from typing import List, Tuple, Iterator, Optional
from abc import ABC, abstractmethod
import numpy as np


@dataclass
class Chunk:
    """A text chunk with metadata."""
    text: str
    start_char: int
    end_char: int
    chunk_id: int
    
    # For MDCT-style: window weights
    window_weights: Optional[np.ndarray] = None
    
    # For onset detection: is this a boundary?
    is_onset: bool = False
    
    @property
    def length(self) -> int:
        return len(self.text)


class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""
    
    @abstractmethod
    def chunk(self, text: str) -> List[Chunk]:
        """Split text into chunks."""
        pass


# =============================================================================
# 1. SLIDING WINDOW (Simple Baseline)
# =============================================================================

class SlidingWindowChunker(ChunkingStrategy):
    """
    Simple sliding window chunking.
    
    Good for: Initial testing, uniform documents
    Bad for: Semantic boundaries ignored
    """
    
    def __init__(self, 
                 window_size: int = 512,
                 stride: int = 256,
                 min_chunk_size: int = 100):
        self.window_size = window_size
        self.stride = stride
        self.min_chunk_size = min_chunk_size
    
    def chunk(self, text: str) -> List[Chunk]:
        chunks = []
        chunk_id = 0
        
        start = 0
        while start < len(text):
            end = min(start + self.window_size, len(text))
            chunk_text = text[start:end]
            
            # Skip if too small (except last chunk)
            if len(chunk_text) >= self.min_chunk_size or start + self.window_size >= len(text):
                chunks.append(Chunk(
                    text=chunk_text,
                    start_char=start,
                    end_char=end,
                    chunk_id=chunk_id
                ))
                chunk_id += 1
            
            start += self.stride
            
            # Avoid infinite loop on last chunk
            if end == len(text):
                break
        
        return chunks


# =============================================================================
# 2. SENTENCE-BASED (Semantic Boundaries)
# =============================================================================

class SentenceChunker(ChunkingStrategy):
    """
    Chunk on sentence boundaries.
    
    Good for: Respecting semantic units
    Bad for: Very long or very short sentences
    """
    
    def __init__(self,
                 target_size: int = 512,
                 max_size: int = 1024,
                 min_size: int = 100):
        self.target_size = target_size
        self.max_size = max_size
        self.min_size = min_size
        
        # Sentence boundary pattern
        self.sentence_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = self.sentence_pattern.split(text)
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk(self, text: str) -> List[Chunk]:
        sentences = self._split_sentences(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_start = 0
        chunk_id = 0
        
        char_pos = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # Would adding this sentence exceed max size?
            if current_length + sentence_length > self.max_size and current_chunk:
                # Emit current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append(Chunk(
                    text=chunk_text,
                    start_char=chunk_start,
                    end_char=chunk_start + len(chunk_text),
                    chunk_id=chunk_id
                ))
                chunk_id += 1
                
                # Start new chunk
                current_chunk = [sentence]
                current_length = sentence_length
                chunk_start = char_pos
            else:
                current_chunk.append(sentence)
                current_length += sentence_length + 1  # +1 for space
            
            char_pos += sentence_length + 1
        
        # Don't forget last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) >= self.min_size:
                chunks.append(Chunk(
                    text=chunk_text,
                    start_char=chunk_start,
                    end_char=chunk_start + len(chunk_text),
                    chunk_id=chunk_id
                ))
        
        return chunks


# =============================================================================
# 3. MDCT-STYLE (Overlapping Windows with Smooth Transitions)
# =============================================================================

class MDCTChunker(ChunkingStrategy):
    """
    MDCT-style chunking with overlapping windows and smooth transitions.
    
    Key insight from SWE theory:
    - Audio MDCT uses 50% overlap and windowing
    - We apply same principle to text
    - Embedding = weighted average over window
    
    Good for: Smooth representations, no hard boundaries
    Bad for: More chunks to store, complexity
    """
    
    def __init__(self,
                 window_size: int = 512,
                 overlap: float = 0.5,  # 50% overlap like audio MDCT
                 window_function: str = "hann"):
        self.window_size = window_size
        self.overlap = overlap
        self.stride = int(window_size * (1 - overlap))
        self.window_function = window_function
    
    def _get_window(self, size: int) -> np.ndarray:
        """Generate window function."""
        if self.window_function == "hann":
            return np.hanning(size)
        elif self.window_function == "hamming":
            return np.hamming(size)
        elif self.window_function == "rectangular":
            return np.ones(size)
        else:
            return np.hanning(size)
    
    def chunk(self, text: str) -> List[Chunk]:
        chunks = []
        chunk_id = 0
        
        # Convert text to "samples" (characters)
        text_length = len(text)
        
        start = 0
        while start < text_length:
            end = min(start + self.window_size, text_length)
            actual_size = end - start
            
            chunk_text = text[start:end]
            
            # Generate window weights for this chunk
            # These can be used when combining embeddings
            window = self._get_window(actual_size)
            
            chunks.append(Chunk(
                text=chunk_text,
                start_char=start,
                end_char=end,
                chunk_id=chunk_id,
                window_weights=window
            ))
            chunk_id += 1
            
            start += self.stride
            
            if end == text_length:
                break
        
        return chunks
    
    def combine_embeddings(self, 
                           embeddings: List[np.ndarray],
                           chunks: List[Chunk]) -> np.ndarray:
        """
        Combine overlapping chunk embeddings using window weights.
        
        This is the MDCT synthesis step:
        - Overlapping regions are blended smoothly
        - Result is a single embedding for the full document
        """
        if not embeddings:
            return np.array([])
        
        embedding_dim = embeddings[0].shape[0]
        
        # Reconstruct full signal
        total_length = max(c.end_char for c in chunks)
        
        # Weight accumulator (per character position)
        weight_sum = np.zeros(total_length)
        embedding_sum = np.zeros((total_length, embedding_dim))
        
        for emb, chunk in zip(embeddings, chunks):
            window = chunk.window_weights
            if window is None:
                window = np.ones(chunk.length)
            
            # Add weighted embedding
            for i, char_idx in enumerate(range(chunk.start_char, chunk.end_char)):
                weight_sum[char_idx] += window[i]
                embedding_sum[char_idx] += window[i] * emb
        
        # Normalize (avoid division by zero)
        weight_sum = np.maximum(weight_sum, 1e-10)
        
        # Average embedding (you could also sample at specific positions)
        full_embedding = embedding_sum / weight_sum[:, np.newaxis]
        
        # Return mean over all positions (simplification)
        return full_embedding.mean(axis=0)


# =============================================================================
# 4. ONSET DETECTION (Adaptive Chunking)
# =============================================================================

class OnsetDetectionChunker(ChunkingStrategy):
    """
    Adaptive chunking based on semantic "onset detection".
    
    Key insight from SWE theory:
    - Audio uses onset detection to find note boundaries
    - We detect "topic changes" in text
    - High-rate sampling at boundaries, low-rate in stable regions
    
    Signals for onset:
    - Paragraph breaks
    - Transitional phrases ("However", "In contrast", "Next")
    - Significant vocabulary shifts
    - Punctuation density changes
    
    Good for: Natural document structure, efficiency
    Bad for: Requires heuristics, may miss subtle transitions
    """
    
    def __init__(self,
                 min_chunk_size: int = 200,
                 max_chunk_size: int = 1000,
                 onset_threshold: float = 0.5):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.onset_threshold = onset_threshold
        
        # Transition words that signal topic change
        self.transition_patterns = [
            r'\n\n+',  # Paragraph breaks
            r'(?i)\b(however|but|although|nevertheless|on the other hand)\b',
            r'(?i)\b(in contrast|conversely|alternatively|instead)\b',
            r'(?i)\b(first|second|third|finally|next|then|lastly)\b',
            r'(?i)\b(in conclusion|to summarize|in summary|overall)\b',
            r'(?i)\b(for example|for instance|specifically|in particular)\b',
            r'(?i)^#+\s',  # Markdown headers
            r'(?i)^\d+\.\s',  # Numbered lists
        ]
        
        self.onset_regex = re.compile('|'.join(self.transition_patterns))
    
    def _find_onsets(self, text: str) -> List[int]:
        """Find positions of semantic onsets."""
        onsets = [0]  # Always start at beginning
        
        for match in self.onset_regex.finditer(text):
            pos = match.start()
            # Don't add if too close to previous onset
            if pos - onsets[-1] >= self.min_chunk_size:
                onsets.append(pos)
        
        return onsets
    
    def chunk(self, text: str) -> List[Chunk]:
        onsets = self._find_onsets(text)
        
        # Add end of text as final boundary
        onsets.append(len(text))
        
        chunks = []
        chunk_id = 0
        
        for i in range(len(onsets) - 1):
            start = onsets[i]
            end = onsets[i + 1]
            
            # Handle chunks that are too long
            while end - start > self.max_chunk_size:
                # Force split at max_chunk_size
                split_point = start + self.max_chunk_size
                
                # Try to find a sentence boundary near split point
                search_region = text[split_point - 100:split_point + 100]
                sentence_end = search_region.rfind('. ')
                if sentence_end > 0:
                    split_point = split_point - 100 + sentence_end + 2
                
                chunk_text = text[start:split_point].strip()
                if chunk_text:
                    chunks.append(Chunk(
                        text=chunk_text,
                        start_char=start,
                        end_char=split_point,
                        chunk_id=chunk_id,
                        is_onset=(i > 0)
                    ))
                    chunk_id += 1
                
                start = split_point
            
            # Add remaining chunk
            chunk_text = text[start:end].strip()
            if chunk_text and len(chunk_text) >= self.min_chunk_size:
                chunks.append(Chunk(
                    text=chunk_text,
                    start_char=start,
                    end_char=end,
                    chunk_id=chunk_id,
                    is_onset=(i > 0)
                ))
                chunk_id += 1
        
        return chunks


# =============================================================================
# 5. SEMANTIC ONSET DETECTION (with Embeddings)
# =============================================================================

class SemanticOnsetChunker(ChunkingStrategy):
    """
    Onset detection using embedding similarity.
    
    This is the "proper" implementation:
    - Compute embeddings for sliding windows
    - Detect where similarity drops (topic change)
    - Place chunk boundaries at low-similarity points
    
    Requires: An embedding model
    """
    
    def __init__(self,
                 embedding_fn,  # Callable: text -> embedding
                 window_size: int = 100,
                 stride: int = 50,
                 similarity_threshold: float = 0.7,
                 min_chunk_size: int = 200):
        self.embedding_fn = embedding_fn
        self.window_size = window_size
        self.stride = stride
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)
    
    def _compute_onset_signal(self, text: str) -> Tuple[List[int], List[float]]:
        """
        Compute onset signal over text.
        
        Returns:
            positions: Character positions
            onset_strength: How "onset-like" each position is (0-1)
        """
        positions = []
        similarities = []
        
        prev_embedding = None
        
        pos = 0
        while pos + self.window_size <= len(text):
            window_text = text[pos:pos + self.window_size]
            embedding = self.embedding_fn(window_text)
            
            if prev_embedding is not None:
                sim = self._cosine_similarity(prev_embedding, embedding)
                positions.append(pos)
                similarities.append(sim)
            
            prev_embedding = embedding
            pos += self.stride
        
        # Convert similarity to onset strength (low similarity = high onset)
        onset_strength = [1.0 - s for s in similarities]
        
        return positions, onset_strength
    
    def chunk(self, text: str) -> List[Chunk]:
        """Chunk based on semantic onsets."""
        positions, onset_strength = self._compute_onset_signal(text)
        
        # Find peaks in onset strength (local maxima above threshold)
        boundaries = [0]
        
        for i in range(1, len(onset_strength) - 1):
            if (onset_strength[i] > self.similarity_threshold and
                onset_strength[i] > onset_strength[i-1] and
                onset_strength[i] > onset_strength[i+1]):
                
                pos = positions[i]
                if pos - boundaries[-1] >= self.min_chunk_size:
                    boundaries.append(pos)
        
        boundaries.append(len(text))
        
        # Create chunks
        chunks = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append(Chunk(
                    text=chunk_text,
                    start_char=start,
                    end_char=end,
                    chunk_id=i,
                    is_onset=(i > 0)
                ))
        
        return chunks


# =============================================================================
# FACTORY & DEMO
# =============================================================================

def get_chunker(strategy: str = "sentence", **kwargs) -> ChunkingStrategy:
    """Factory function for chunkers."""
    chunkers = {
        "sliding": SlidingWindowChunker,
        "sentence": SentenceChunker,
        "mdct": MDCTChunker,
        "onset": OnsetDetectionChunker,
    }
    
    if strategy not in chunkers:
        raise ValueError(f"Unknown strategy: {strategy}. Choose from: {list(chunkers.keys())}")
    
    return chunkers[strategy](**kwargs)


def demo():
    """Demonstrate chunking strategies."""
    
    sample_text = """
    Machine learning is a subset of artificial intelligence. It enables computers to learn from data.

    However, deep learning takes this further. Neural networks with many layers can learn complex patterns. 
    This has revolutionized computer vision and natural language processing.

    In contrast, traditional programming requires explicit rules. A programmer must define every step.
    This approach works well for structured problems but struggles with ambiguity.

    First, let's consider supervised learning. The model learns from labeled examples.
    Second, unsupervised learning finds patterns without labels.
    Third, reinforcement learning learns from rewards and penalties.

    In conclusion, machine learning offers powerful tools for solving complex problems.
    The choice of approach depends on the specific use case and available data.
    """
    
    print("=" * 70)
    print("CHUNKING STRATEGIES DEMO")
    print("=" * 70)
    print(f"\nInput text: {len(sample_text)} characters")
    
    strategies = [
        ("sliding", {"window_size": 200, "stride": 100}),
        ("sentence", {"target_size": 200}),
        ("mdct", {"window_size": 200, "overlap": 0.5}),
        ("onset", {"min_chunk_size": 100, "max_chunk_size": 500}),
    ]
    
    for name, kwargs in strategies:
        print(f"\n{'-' * 70}")
        print(f"Strategy: {name.upper()}")
        print(f"Config: {kwargs}")
        print(f"{'-' * 70}")
        
        chunker = get_chunker(name, **kwargs)
        chunks = chunker.chunk(sample_text)
        
        print(f"Produced {len(chunks)} chunks:")
        for chunk in chunks:
            preview = chunk.text[:50].replace('\n', ' ').strip()
            onset_marker = " [ONSET]" if chunk.is_onset else ""
            window_info = f" [window={len(chunk.window_weights)}]" if chunk.window_weights is not None else ""
            print(f"  {chunk.chunk_id}: [{chunk.start_char}:{chunk.end_char}] "
                  f"({chunk.length} chars){onset_marker}{window_info}")
            print(f"      \"{preview}...\"")
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demo()
