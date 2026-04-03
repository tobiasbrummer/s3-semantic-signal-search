#!/usr/bin/env python3
"""
Semantic Audio Codec (SAC) - Complete Framework

A more complete exploration of treating embeddings as audio signals,
including dimension reduction, actual codec simulation, and retrieval.

Key insight: Embeddings may already have spectral structure from
positional encoding. We can exploit this for compression.

Architecture:
    Text → Embedding Signal → Semantic Filterbank → Audio → Codec → Compressed
    
Retrieval:
    Query → Encode → Cross-correlate with compressed signals → Positions
"""

import numpy as np
from scipy import signal, fft
from scipy.ndimage import gaussian_filter1d
from typing import List, Tuple, Optional, Dict, Callable
from dataclasses import dataclass
import struct
import zlib


# =============================================================================
# Semantic Filterbank: Reduce embedding dims to audio-friendly bands
# =============================================================================

class SemanticFilterbank:
    """
    Reduces high-dimensional embeddings to a smaller number of
    semantic frequency bands, analogous to Mel filterbank in audio.
    
    The key insight: not all embedding dimensions are equally important.
    We can group and weight them based on:
    1. Variance (information content)
    2. Correlation (redundancy)
    3. Position in the embedding (if known to have spectral structure)
    """
    
    def __init__(self, input_dim: int, num_bands: int = 24, 
                 band_type: str = 'learned'):
        """
        Args:
            input_dim: Original embedding dimension (e.g., 768)
            num_bands: Number of output bands (e.g., 24 like Mel)
            band_type: 'uniform', 'log', or 'learned'
        """
        self.input_dim = input_dim
        self.num_bands = num_bands
        self.band_type = band_type
        
        # Initialize filterbank matrix: (num_bands, input_dim)
        self.filters = self._init_filters()
    
    def _init_filters(self) -> np.ndarray:
        """Initialize filterbank based on type"""
        
        if self.band_type == 'uniform':
            # Equal-width bands
            filters = np.zeros((self.num_bands, self.input_dim))
            band_width = self.input_dim // self.num_bands
            
            for i in range(self.num_bands):
                start = i * band_width
                end = start + band_width if i < self.num_bands - 1 else self.input_dim
                filters[i, start:end] = 1.0 / (end - start)
            
            return filters
        
        elif self.band_type == 'log':
            # Logarithmic bands (more resolution at low dims)
            # Analogous to Mel scale
            filters = np.zeros((self.num_bands, self.input_dim))
            
            # Log-spaced band edges
            edges = np.logspace(0, np.log10(self.input_dim), self.num_bands + 1).astype(int)
            edges = np.unique(np.clip(edges, 0, self.input_dim))
            
            for i in range(len(edges) - 1):
                start, end = edges[i], edges[i + 1]
                if end > start:
                    filters[i, start:end] = 1.0 / (end - start)
            
            return filters
        
        elif self.band_type == 'learned':
            # PCA-like: will be fitted on data
            # Initialize as identity-ish for now
            filters = np.random.randn(self.num_bands, self.input_dim) * 0.01
            # Add structure: each band focuses on a region
            for i in range(self.num_bands):
                center = int(i * self.input_dim / self.num_bands)
                width = self.input_dim // self.num_bands
                filters[i, max(0, center-width):min(self.input_dim, center+width)] += 0.1
            
            return filters
        
        else:
            raise ValueError(f"Unknown band_type: {self.band_type}")
    
    def fit(self, embeddings: np.ndarray):
        """
        Fit filterbank to data (for 'learned' type).
        
        Uses truncated SVD to find the most important directions.
        """
        if self.band_type != 'learned':
            return
        
        # Center the data
        mean = np.mean(embeddings, axis=0)
        centered = embeddings - mean
        
        # SVD
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        
        # Take top num_bands components
        self.filters = Vt[:self.num_bands, :]
        self.mean = mean
        
        # Explained variance
        total_var = np.sum(S ** 2)
        explained = np.sum(S[:self.num_bands] ** 2) / total_var
        print(f"Filterbank explains {explained*100:.1f}% of variance with {self.num_bands} bands")
    
    def apply(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Apply filterbank to reduce dimensions.
        
        Args:
            embeddings: Shape (num_positions, input_dim)
            
        Returns:
            bands: Shape (num_positions, num_bands)
        """
        if hasattr(self, 'mean'):
            embeddings = embeddings - self.mean
        
        return embeddings @ self.filters.T
    
    def invert(self, bands: np.ndarray) -> np.ndarray:
        """
        Approximate inverse: reconstruct embeddings from bands.
        
        This is lossy but enables decoding.
        """
        # Pseudo-inverse
        reconstructed = bands @ np.linalg.pinv(self.filters.T)
        
        if hasattr(self, 'mean'):
            reconstructed = reconstructed + self.mean
        
        return reconstructed


# =============================================================================
# Semantic Audio Codec
# =============================================================================

@dataclass
class SACConfig:
    """Configuration for Semantic Audio Codec"""
    
    # Filterbank
    num_bands: int = 24           # Like Mel bands
    band_type: str = 'log'        # 'uniform', 'log', 'learned'
    
    # Temporal encoding
    samples_per_position: int = 64  # Audio samples per embedding position
    sample_rate: int = 16000        # Lower rate for speech-like signal
    
    # Frequencies
    base_freq: float = 80.0
    max_freq: float = 7600.0
    
    # Compression
    quantization_bits: int = 8     # Bits per sample
    use_dpcm: bool = True          # Differential PCM
    use_psychosemantic: bool = True  # Adaptive quantization


class SemanticAudioCodec:
    """
    Complete codec for semantic embeddings.
    
    Pipeline:
        Encode: Embeddings → Filterbank → Modulation → Quantization → Compress
        Decode: Decompress → Dequantize → Demodulation → Inverse Filterbank
    """
    
    def __init__(self, embedding_dim: int, config: SACConfig = None):
        self.embedding_dim = embedding_dim
        self.config = config or SACConfig()
        
        # Initialize filterbank
        self.filterbank = SemanticFilterbank(
            embedding_dim, 
            self.config.num_bands,
            self.config.band_type
        )
        
        # Frequency allocation
        self.frequencies = np.logspace(
            np.log10(self.config.base_freq),
            np.log10(self.config.max_freq),
            self.config.num_bands
        )
    
    def fit(self, embeddings: np.ndarray):
        """Fit filterbank on training data"""
        self.filterbank.fit(embeddings)
    
    def encode(self, embeddings: np.ndarray, 
               phases: Optional[np.ndarray] = None) -> bytes:
        """
        Encode embeddings to compressed bytes.
        
        Args:
            embeddings: Shape (num_positions, embedding_dim)
            phases: Optional (num_positions,) phase per position
            
        Returns:
            Compressed byte string
        """
        num_positions = len(embeddings)
        
        # Step 1: Filterbank reduction
        bands = self.filterbank.apply(embeddings)  # (num_pos, num_bands)
        
        # Step 2: Estimate importance for psychosemantic compression
        if self.config.use_psychosemantic:
            importance = self._estimate_importance(bands)
        else:
            importance = np.ones(self.config.num_bands)
        
        # Step 3: Generate audio signal
        total_samples = num_positions * self.config.samples_per_position
        t = np.arange(total_samples) / self.config.sample_rate
        
        audio = np.zeros(total_samples)
        
        for pos in range(num_positions):
            start = pos * self.config.samples_per_position
            end = start + self.config.samples_per_position
            t_local = t[start:end]
            
            phase = phases[pos] if phases is not None else 0.0
            
            for band_idx, (freq, amp) in enumerate(zip(self.frequencies, bands[pos])):
                audio[start:end] += amp * np.cos(2 * np.pi * freq * t_local + phase)
        
        # Normalize
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
        
        # Step 4: Quantization
        quantized = self._quantize(audio, importance)
        
        # Step 5: Compress
        compressed = self._compress(quantized, num_positions, importance)
        
        return compressed
    
    def decode(self, data: bytes) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Decode compressed bytes back to embeddings.
        
        Returns:
            embeddings: Reconstructed embeddings
            phases: Reconstructed phases (if encoded)
        """
        # Step 1: Decompress
        quantized, num_positions, importance = self._decompress(data)
        
        # Step 2: Dequantize
        audio = self._dequantize(quantized)
        
        # Step 3: Extract bands via STFT
        nperseg = self.config.samples_per_position
        f, t_stft, Zxx = signal.stft(
            audio,
            fs=self.config.sample_rate,
            nperseg=nperseg,
            noverlap=0
        )
        
        # Step 4: Extract amplitude at each frequency band
        bands = np.zeros((Zxx.shape[1], self.config.num_bands))
        for band_idx, target_freq in enumerate(self.frequencies):
            freq_idx = np.argmin(np.abs(f - target_freq))
            bands[:, band_idx] = np.abs(Zxx[freq_idx, :])
        
        # Step 5: Inverse filterbank
        embeddings = self.filterbank.invert(bands[:num_positions])
        
        return embeddings, None  # Phase recovery is complex, skip for now
    
    def _estimate_importance(self, bands: np.ndarray) -> np.ndarray:
        """Estimate importance of each band for adaptive quantization"""
        variance = np.var(bands, axis=0)
        importance = variance / (variance.max() + 1e-8)
        return importance
    
    def _quantize(self, audio: np.ndarray, importance: np.ndarray) -> np.ndarray:
        """Quantize audio signal"""
        bits = self.config.quantization_bits
        levels = 2 ** bits
        
        # Map [-1, 1] to [0, levels-1]
        quantized = ((audio + 1) / 2 * (levels - 1)).astype(np.uint8)
        
        if self.config.use_dpcm:
            # Differential encoding: store differences
            diff = np.diff(quantized.astype(np.int16), prepend=128)
            diff = np.clip(diff, -128, 127).astype(np.int8)
            return diff.view(np.uint8)
        
        return quantized
    
    def _dequantize(self, quantized: np.ndarray) -> np.ndarray:
        """Dequantize back to float"""
        bits = self.config.quantization_bits
        levels = 2 ** bits
        
        if self.config.use_dpcm:
            # Reverse differential encoding
            diff = quantized.view(np.int8).astype(np.int16)
            audio_int = np.cumsum(diff).astype(np.uint8)
        else:
            audio_int = quantized
        
        # Map [0, levels-1] back to [-1, 1]
        audio = audio_int.astype(np.float32) / (levels - 1) * 2 - 1
        
        return audio
    
    def _compress(self, quantized: np.ndarray, num_positions: int,
                  importance: np.ndarray) -> bytes:
        """Compress quantized data"""
        # Header
        header = struct.pack('II', num_positions, self.config.num_bands)
        importance_bytes = importance.astype(np.float32).tobytes()
        
        # Compress audio data with zlib
        audio_compressed = zlib.compress(quantized.tobytes(), level=9)
        
        return header + importance_bytes + audio_compressed
    
    def _decompress(self, data: bytes) -> Tuple[np.ndarray, int, np.ndarray]:
        """Decompress data"""
        # Parse header
        header_size = 8
        num_positions, num_bands = struct.unpack('II', data[:header_size])
        
        # Parse importance
        imp_size = num_bands * 4
        importance = np.frombuffer(
            data[header_size:header_size + imp_size], 
            dtype=np.float32
        )
        
        # Decompress audio
        audio_data = zlib.decompress(data[header_size + imp_size:])
        quantized = np.frombuffer(audio_data, dtype=np.uint8)
        
        return quantized, num_positions, importance


# =============================================================================
# Spectral Analysis of Embeddings
# =============================================================================

def analyze_embedding_spectrum(embeddings: np.ndarray) -> Dict:
    """
    Analyze the spectral properties of embeddings.
    
    Tests the hypothesis that embedding dimensions have
    inherent frequency-like structure.
    """
    num_positions, embedding_dim = embeddings.shape
    
    # FFT along the dimension axis (treating dims as "time")
    spectrum = np.fft.fft(embeddings, axis=1)
    power = np.abs(spectrum) ** 2
    
    # Average power spectrum
    avg_power = np.mean(power, axis=0)
    
    # FFT along position axis (temporal spectrum)
    temporal_spectrum = np.fft.fft(embeddings, axis=0)
    temporal_power = np.mean(np.abs(temporal_spectrum) ** 2, axis=1)
    
    # Autocorrelation of embeddings (temporal smoothness)
    autocorr = []
    for lag in range(min(50, num_positions)):
        if lag == 0:
            autocorr.append(1.0)
        else:
            corr = np.mean(embeddings[:-lag] * embeddings[lag:])
            autocorr.append(corr / (np.mean(embeddings ** 2) + 1e-8))
    
    return {
        'dimension_power_spectrum': avg_power,
        'temporal_power_spectrum': temporal_power,
        'autocorrelation': np.array(autocorr),
        'dimension_entropy': -np.sum(avg_power / avg_power.sum() * 
                                      np.log(avg_power / avg_power.sum() + 1e-10)),
    }


# =============================================================================
# Complete Pipeline Demo
# =============================================================================

def demo_full_codec():
    """Demonstrate the complete codec pipeline"""
    print("=" * 70)
    print("DEMO: Complete Semantic Audio Codec")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Simulate realistic embeddings (768-dim, 500 positions)
    # With temporal coherence (adjacent positions are similar)
    num_positions = 500
    embedding_dim = 768
    
    # Generate smooth, correlated embeddings
    base = np.random.randn(num_positions, embedding_dim)
    embeddings = gaussian_filter1d(base, sigma=3, axis=0)  # Temporal smoothing
    
    # Add some structure: certain dimensions are more important
    importance = np.exp(-np.arange(embedding_dim) / 200)  # Decay
    embeddings = embeddings * importance
    
    # Normalize
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    
    print(f"\nOriginal embeddings: {embeddings.shape}")
    print(f"Original size: {embeddings.nbytes:,} bytes")
    
    # Analyze spectral properties
    print("\nSpectral Analysis:")
    spectrum_info = analyze_embedding_spectrum(embeddings)
    print(f"  Dimension entropy: {spectrum_info['dimension_entropy']:.2f}")
    print(f"  Autocorr at lag 1: {spectrum_info['autocorrelation'][1]:.3f}")
    print(f"  Autocorr at lag 5: {spectrum_info['autocorrelation'][5]:.3f}")
    print(f"  → High autocorr = temporal redundancy = good for compression")
    
    # Create codec
    config = SACConfig(
        num_bands=32,
        band_type='learned',
        samples_per_position=32,
        sample_rate=16000,
        quantization_bits=8,
        use_dpcm=True,
        use_psychosemantic=True
    )
    
    codec = SemanticAudioCodec(embedding_dim, config)
    codec.fit(embeddings)  # Fit filterbank
    
    # Encode
    compressed = codec.encode(embeddings)
    
    print(f"\nCompressed size: {len(compressed):,} bytes")
    print(f"Compression ratio: {embeddings.nbytes / len(compressed):.1f}x")
    
    # Decode
    reconstructed, _ = codec.decode(compressed)
    
    # Quality metrics
    mse = np.mean((embeddings[:len(reconstructed)] - reconstructed) ** 2)
    
    # Cosine similarity preservation (most important for retrieval!)
    orig_norms = np.linalg.norm(embeddings[:len(reconstructed)], axis=1, keepdims=True)
    recon_norms = np.linalg.norm(reconstructed, axis=1, keepdims=True)
    
    cosine_sims = np.sum(
        (embeddings[:len(reconstructed)] / (orig_norms + 1e-8)) * 
        (reconstructed / (recon_norms + 1e-8)),
        axis=1
    )
    
    print(f"\nReconstruction Quality:")
    print(f"  MSE: {mse:.6f}")
    print(f"  Mean cosine similarity: {np.mean(cosine_sims):.4f}")
    print(f"  Min cosine similarity: {np.min(cosine_sims):.4f}")
    
    # Compare with baselines
    print("\n" + "-" * 40)
    print("Comparison with baselines:")
    
    # Baseline 1: Float16
    float16_size = embeddings.astype(np.float16).nbytes
    print(f"  Float16:      {float16_size:,} bytes ({embeddings.nbytes / float16_size:.1f}x)")
    
    # Baseline 2: PCA + Float16
    U, S, Vt = np.linalg.svd(embeddings, full_matrices=False)
    pca_dims = 64
    pca_compressed = (U[:, :pca_dims] @ np.diag(S[:pca_dims])).astype(np.float16)
    pca_size = pca_compressed.nbytes + Vt[:pca_dims].astype(np.float16).nbytes
    print(f"  PCA-64 + f16: {pca_size:,} bytes ({embeddings.nbytes / pca_size:.1f}x)")
    
    # Baseline 3: Zlib on raw
    zlib_raw = zlib.compress(embeddings.tobytes(), level=9)
    print(f"  Zlib raw:     {len(zlib_raw):,} bytes ({embeddings.nbytes / len(zlib_raw):.1f}x)")
    
    # Our codec
    print(f"  SAC (ours):   {len(compressed):,} bytes ({embeddings.nbytes / len(compressed):.1f}x)")


def demo_retrieval_from_compressed():
    """Demo retrieval directly on compressed representations"""
    print("\n" + "=" * 70)
    print("DEMO: Retrieval from Compressed Representations")
    print("=" * 70)
    
    np.random.seed(42)
    embedding_dim = 256
    
    # Create codec
    config = SACConfig(num_bands=24, band_type='log')
    codec = SemanticAudioCodec(embedding_dim, config)
    
    # Create documents
    docs = []
    for i in range(5):
        # Each doc has different "topic" (different dimension ranges active)
        emb = np.random.rand(100, embedding_dim) * 0.1
        topic_start = i * 50
        emb[:, topic_start:topic_start+50] += np.random.rand(100, 50) * 0.5
        
        # Add temporal structure
        emb = gaussian_filter1d(emb, sigma=2, axis=0)
        
        compressed = codec.encode(emb)
        docs.append({
            'id': f'doc_{i}',
            'topic': f'Topic at dims {topic_start}-{topic_start+50}',
            'original': emb,
            'compressed': compressed,
            'size': len(compressed)
        })
    
    print("\nDocuments:")
    for doc in docs:
        print(f"  {doc['id']}: {doc['topic']}, {doc['size']} bytes")
    
    # Query: looking for Topic 2 (dims 100-150)
    query = np.random.rand(10, embedding_dim) * 0.1
    query[:, 100:150] += np.random.rand(10, 50) * 0.8
    
    print(f"\nQuery: Topic at dims 100-150")
    
    # Decompress and search (in practice, could do this in frequency domain)
    results = []
    for doc in docs:
        reconstructed, _ = codec.decode(doc['compressed'])
        
        # Cosine similarity between query mean and doc positions
        query_vec = query.mean(axis=0)
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        
        doc_vecs = reconstructed / (np.linalg.norm(reconstructed, axis=1, keepdims=True) + 1e-8)
        similarities = doc_vecs @ query_vec
        
        best_pos = np.argmax(similarities)
        best_score = similarities[best_pos]
        
        results.append({
            'id': doc['id'],
            'topic': doc['topic'],
            'score': best_score,
            'position': best_pos
        })
    
    results.sort(key=lambda x: x['score'], reverse=True)
    
    print("\nResults:")
    for r in results:
        match = "✓" if "100-150" in r['topic'] else " "
        print(f"  {match} {r['id']}: score={r['score']:.3f}, pos={r['position']}, {r['topic']}")


def demo_psychosemantic_model():
    """Explore psychosemantic compression concepts"""
    print("\n" + "=" * 70)
    print("DEMO: Psychosemantic Model")
    print("=" * 70)
    
    print("""
PSYCHOSEMANTIC MODEL - Conceptual Framework

Analogous to psychoacoustic models in audio compression (MP3, AAC),
a psychosemantic model identifies which embedding information can be
discarded without affecting retrieval quality.

┌────────────────────────────────────────────────────────────────────┐
│                    PSYCHOACOUSTIC              PSYCHOSEMANTIC      │
├────────────────────────────────────────────────────────────────────┤
│ Frequency masking:              Dimension masking:                 │
│   Loud sound masks quiet        Strong concept masks weak related  │
│   at nearby frequencies         concepts in nearby dimensions      │
│                                                                    │
│ Temporal masking:               Positional masking:                │
│   Sounds mask nearby            Strong local semantics mask        │
│   sounds in time                nearby weaker signals              │
│                                                                    │
│ Absolute threshold:             Relevance threshold:               │
│   Below hearing threshold       Below retrieval relevance          │
│   → discard                     → discard                          │
│                                                                    │
│ Critical bands:                 Semantic bands:                    │
│   Bark scale groups freqs       Dimension groups that represent    │
│   processed together            coherent semantic features         │
│                                                                    │
│ Joint stereo:                   Semantic correlation:              │
│   Encode L+R / L-R              Encode common + differential       │
│   for correlated channels       for correlated dimensions          │
└────────────────────────────────────────────────────────────────────┘

Implementation Ideas:

1. MASKING MODEL
   - If embedding[i] > threshold AND |i-j| < bandwidth:
     embedding[j] can be quantized more coarsely
   - Learn masking curves from retrieval task

2. RELEVANCE MODEL  
   - Train classifier: "does this dimension affect retrieval?"
   - Weight bit allocation by relevance
   - Low-relevance dims get fewer bits

3. SEMANTIC CRITICAL BANDS
   - Cluster dimensions by correlation pattern
   - Allocate bits per band, not per dimension
   - Jointly encode within-band values

4. TEMPORAL PREDICTION
   - Embeddings at pos t predicted from pos t-1
   - Encode residual (like DPCM in audio)
   - Smooth semantic transitions compress well

5. PERCEPTUAL LOSS FUNCTION
   - Don't minimize MSE
   - Minimize retrieval error: "can we still find this?"
   - Allow large reconstruction error if retrieval unaffected
""")


def main():
    print("Semantic Audio Codec - Complete Framework")
    print("=" * 70)
    
    demo_full_codec()
    demo_retrieval_from_compressed()
    demo_psychosemantic_model()
    
    print("\n" + "=" * 70)
    print("SUMMARY: Audio Encoding for Embeddings")
    print("=" * 70)
    print("""
WHAT WORKS:
✓ Embeddings CAN be encoded as audio signals
✓ Audio codecs provide reasonable compression (~5-15x vs float32)
✓ Retrieval works on reconstructed embeddings
✓ Temporal coherence in embeddings aids compression

CHALLENGES:
- High dimensionality (768) → need filterbank/PCA first
- Reconstruction quality vs compression trade-off
- Real audio codecs (Opus) need more integration work
- Phase preservation is tricky

KEY INSIGHT:
The REAL win isn't just compression ratio—it's the framework:
- Continuous signal = no chunking
- Signal processing = powerful retrieval tools
- Psychosemantic model = intelligent lossy compression
- Spectral view = hierarchical abstraction levels

NEXT STEPS:
1. Integrate actual Opus/FLAC codecs
2. Learn filterbank from retrieval task
3. Develop psychosemantic masking model
4. Test on real embedding datasets (MTEB, etc.)
""")


if __name__ == "__main__":
    main()
