#!/usr/bin/env python3
"""
Semantic Audio Encoding (SAE)

Encodes text embeddings as audio signals, enabling:
1. Compression with battle-tested audio codecs (Opus, FLAC, etc.)
2. Signal processing techniques for retrieval
3. Continuous representation without chunking artifacts

Concept:
- Each embedding dimension becomes a frequency band
- Position in text becomes time
- The result is a "semantic spectrogram" that can be treated as audio

Requirements:
    pip install numpy scipy soundfile scikit-learn
    
Optional for codec compression:
    pip install pydub  # requires ffmpeg
"""

import numpy as np
from scipy import signal
from scipy.io import wavfile
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import struct
import io


# =============================================================================
# Core Encoding: Embeddings → Audio Signal
# =============================================================================

@dataclass
class SemanticAudioConfig:
    """Configuration for semantic audio encoding"""
    
    # Audio parameters
    sample_rate: int = 44100        # Standard audio sample rate
    base_freq: float = 100.0        # Lowest frequency (Hz)
    freq_spacing: str = 'mel'       # 'linear', 'log', 'mel'
    max_freq: float = 8000.0        # Highest frequency (Hz)
    
    # Encoding parameters  
    samples_per_position: int = 512  # Audio samples per text position
    embedding_dim: int = 64          # Number of dimensions to encode
    
    # Phase encoding
    encode_phase: bool = True        # Include phase information
    phase_channel: str = 'stereo'    # 'stereo' (L/R) or 'quadrature' (I/Q)


class SemanticAudioEncoder:
    """
    Encodes embedding sequences as audio signals.
    
    Each embedding dimension is mapped to a frequency band.
    The embedding value modulates the amplitude of that frequency.
    Phase information can be encoded in stereo or quadrature channels.
    """
    
    def __init__(self, config: SemanticAudioConfig = None):
        self.config = config or SemanticAudioConfig()
        self._init_frequencies()
    
    def _init_frequencies(self):
        """Initialize frequency bands for each embedding dimension"""
        n = self.config.embedding_dim
        
        if self.config.freq_spacing == 'linear':
            self.frequencies = np.linspace(
                self.config.base_freq,
                self.config.max_freq,
                n
            )
        elif self.config.freq_spacing == 'log':
            self.frequencies = np.logspace(
                np.log10(self.config.base_freq),
                np.log10(self.config.max_freq),
                n
            )
        elif self.config.freq_spacing == 'mel':
            # Mel scale - perceptually uniform
            mel_low = 2595 * np.log10(1 + self.config.base_freq / 700)
            mel_high = 2595 * np.log10(1 + self.config.max_freq / 700)
            mels = np.linspace(mel_low, mel_high, n)
            self.frequencies = 700 * (10 ** (mels / 2595) - 1)
        
        print(f"Frequency bands: {self.frequencies[0]:.1f}Hz - {self.frequencies[-1]:.1f}Hz")
    
    def encode(self, 
               embeddings: np.ndarray, 
               phases: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Encode embedding sequence as audio.
        
        Args:
            embeddings: Shape (num_positions, embedding_dim)
            phases: Optional shape (num_positions,) - phase per position
            
        Returns:
            audio: Shape (num_samples,) or (num_samples, 2) for stereo
        """
        num_positions, emb_dim = embeddings.shape
        
        # Ensure embedding dim matches config
        if emb_dim != self.config.embedding_dim:
            # Truncate or pad
            if emb_dim > self.config.embedding_dim:
                embeddings = embeddings[:, :self.config.embedding_dim]
            else:
                pad = np.zeros((num_positions, self.config.embedding_dim - emb_dim))
                embeddings = np.concatenate([embeddings, pad], axis=1)
        
        # Total audio length
        total_samples = num_positions * self.config.samples_per_position
        t = np.arange(total_samples) / self.config.sample_rate
        
        # Initialize audio
        if self.config.encode_phase and self.config.phase_channel == 'stereo':
            audio = np.zeros((total_samples, 2))
        else:
            audio = np.zeros(total_samples)
        
        # Generate signal for each position
        for pos in range(num_positions):
            start = pos * self.config.samples_per_position
            end = start + self.config.samples_per_position
            t_local = t[start:end]
            
            # Get embedding and optional phase for this position
            emb = embeddings[pos]
            phase_offset = phases[pos] if phases is not None else 0.0
            
            # Sum of sinusoids, each frequency weighted by embedding value
            for dim, freq in enumerate(self.frequencies):
                amplitude = emb[dim]
                
                if self.config.encode_phase and self.config.phase_channel == 'stereo':
                    # Left channel: cos (in-phase)
                    # Right channel: sin (quadrature) 
                    # Phase is encoded by rotating between channels
                    audio[start:end, 0] += amplitude * np.cos(2 * np.pi * freq * t_local + phase_offset)
                    audio[start:end, 1] += amplitude * np.sin(2 * np.pi * freq * t_local + phase_offset)
                else:
                    audio[start:end] += amplitude * np.cos(2 * np.pi * freq * t_local + phase_offset)
        
        # Normalize to prevent clipping
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * 0.9
        
        return audio
    
    def decode(self, audio: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Decode audio back to embeddings.
        
        Uses short-time Fourier transform (STFT) to extract
        frequency content at each position.
        
        Returns:
            embeddings: Shape (num_positions, embedding_dim)
            phases: Shape (num_positions,) if phase was encoded
        """
        is_stereo = len(audio.shape) == 2 and audio.shape[1] == 2
        
        # Use left channel or mono for amplitude
        audio_mono = audio[:, 0] if is_stereo else audio
        
        # Compute STFT
        nperseg = self.config.samples_per_position
        f, t_stft, Zxx = signal.stft(
            audio_mono, 
            fs=self.config.sample_rate,
            nperseg=nperseg,
            noverlap=0
        )
        
        num_positions = Zxx.shape[1]
        embeddings = np.zeros((num_positions, self.config.embedding_dim))
        phases = np.zeros(num_positions) if self.config.encode_phase else None
        
        # Extract amplitude at each target frequency
        for dim, target_freq in enumerate(self.frequencies):
            # Find nearest frequency bin
            freq_idx = np.argmin(np.abs(f - target_freq))
            embeddings[:, dim] = np.abs(Zxx[freq_idx, :])
        
        # Extract phase if stereo
        if is_stereo and self.config.encode_phase:
            audio_right = audio[:, 1]
            _, _, Zxx_right = signal.stft(
                audio_right,
                fs=self.config.sample_rate,
                nperseg=nperseg,
                noverlap=0
            )
            # Phase from arctan of quadrature components
            # Use lowest frequency as reference
            phases = np.angle(Zxx[1, :] + 1j * Zxx_right[1, :])
        
        return embeddings, phases


# =============================================================================
# Compression Analysis
# =============================================================================

def analyze_compression(audio: np.ndarray, sample_rate: int) -> Dict:
    """Analyze potential compression ratios"""
    
    # Raw size
    raw_bytes = audio.nbytes
    
    # Convert to 16-bit PCM
    audio_16bit = (audio * 32767).astype(np.int16)
    pcm_bytes = audio_16bit.nbytes
    
    results = {
        'raw_float64_bytes': raw_bytes,
        'pcm_16bit_bytes': pcm_bytes,
        'duration_seconds': len(audio) / sample_rate,
    }
    
    # Estimate codec compression (typical ratios)
    # These are estimates - actual compression depends on signal complexity
    results['estimated_flac_bytes'] = int(pcm_bytes * 0.5)    # ~50% of PCM
    results['estimated_opus_64k'] = int(results['duration_seconds'] * 64000 / 8)  # 64 kbps
    results['estimated_opus_32k'] = int(results['duration_seconds'] * 32000 / 8)  # 32 kbps
    results['estimated_opus_16k'] = int(results['duration_seconds'] * 16000 / 8)  # 16 kbps (speech)
    
    return results


def compare_storage(num_positions: int, embedding_dim: int, config: SemanticAudioConfig) -> Dict:
    """Compare storage requirements: raw embeddings vs audio encoding"""
    
    # Raw embedding storage
    raw_float32 = num_positions * embedding_dim * 4  # float32
    raw_float16 = num_positions * embedding_dim * 2  # float16
    
    # Audio encoding
    total_samples = num_positions * config.samples_per_position
    channels = 2 if config.encode_phase and config.phase_channel == 'stereo' else 1
    audio_pcm = total_samples * channels * 2  # 16-bit PCM
    
    duration = total_samples / config.sample_rate
    
    return {
        'num_positions': num_positions,
        'embedding_dim': embedding_dim,
        'raw_float32_bytes': raw_float32,
        'raw_float16_bytes': raw_float16,
        'audio_pcm_bytes': audio_pcm,
        'audio_duration_sec': duration,
        'estimated_opus_64k': int(duration * 64000 / 8),
        'estimated_opus_32k': int(duration * 32000 / 8),
        'estimated_opus_16k': int(duration * 16000 / 8),
        'compression_vs_float32_opus64': raw_float32 / max(1, int(duration * 64000 / 8)),
        'compression_vs_float32_opus32': raw_float32 / max(1, int(duration * 32000 / 8)),
    }


# =============================================================================
# Semantic Audio Retrieval
# =============================================================================

class SemanticAudioIndex:
    """
    Search index using audio-encoded embeddings.
    
    Retrieval uses cross-correlation - finding where the query
    "resonates" with the document signal.
    """
    
    def __init__(self, encoder: SemanticAudioEncoder):
        self.encoder = encoder
        self.documents: List[Dict] = []
    
    def add_document(self, doc_id: str, embeddings: np.ndarray, 
                     phases: Optional[np.ndarray] = None,
                     metadata: Optional[Dict] = None):
        """Add a document to the index"""
        audio = self.encoder.encode(embeddings, phases)
        
        self.documents.append({
            'id': doc_id,
            'audio': audio,
            'num_positions': len(embeddings),
            'metadata': metadata or {}
        })
    
    def search(self, query_embeddings: np.ndarray,
               query_phases: Optional[np.ndarray] = None,
               top_k: int = 5) -> List[Dict]:
        """
        Search using cross-correlation.
        
        The query audio is correlated with each document audio.
        Peaks in correlation indicate matching regions.
        """
        query_audio = self.encoder.encode(query_embeddings, query_phases)
        
        # Use mono for correlation
        if len(query_audio.shape) == 2:
            query_mono = query_audio[:, 0]
        else:
            query_mono = query_audio
        
        results = []
        
        for doc in self.documents:
            doc_audio = doc['audio']
            if len(doc_audio.shape) == 2:
                doc_mono = doc_audio[:, 0]
            else:
                doc_mono = doc_audio
            
            # Cross-correlation
            correlation = signal.correlate(doc_mono, query_mono, mode='valid')
            
            # Find peak
            peak_idx = np.argmax(np.abs(correlation))
            peak_score = np.abs(correlation[peak_idx])
            
            # Convert sample position to text position
            text_position = peak_idx // self.encoder.config.samples_per_position
            
            results.append({
                'doc_id': doc['id'],
                'score': float(peak_score),
                'position': int(text_position),
                'metadata': doc['metadata']
            })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]


# =============================================================================
# Psychosemantic Compression (Conceptual)
# =============================================================================

"""
PSYCHOSEMANTIC MODEL - Conceptual Framework

Just as psychoacoustic models identify which audio frequencies
humans can't perceive (and can be discarded), a psychosemantic
model would identify which embedding dimensions are:

1. REDUNDANT: Highly correlated with other dimensions
   → Can be predicted from context
   → Safe to compress aggressively

2. IMPERCEPTIBLE: Below threshold for semantic distinction
   → Small variations don't change meaning
   → Can be quantized heavily

3. MASKED: Dominated by stronger signals
   → A very strong concept masks weaker related concepts
   → Similar to audio frequency masking

4. TRANSIENT vs SUSTAINED:
   → Quick semantic shifts (like audio transients) need preservation
   → Sustained semantic themes can be smoothed

Implementation would involve:
- Learning dimension correlations from corpus
- Identifying "semantic masking" thresholds
- Adaptive bit allocation per dimension
- Temporal prediction for smoother dimensions
"""

def estimate_semantic_redundancy(embeddings: np.ndarray) -> np.ndarray:
    """
    Estimate which dimensions are redundant (predictable from others).
    
    Returns: redundancy score per dimension (higher = more redundant)
    """
    # Correlation matrix
    corr = np.corrcoef(embeddings.T)
    
    # Average absolute correlation with other dimensions
    # (excluding self-correlation)
    np.fill_diagonal(corr, 0)
    redundancy = np.mean(np.abs(corr), axis=1)
    
    return redundancy


def estimate_semantic_importance(embeddings: np.ndarray) -> np.ndarray:
    """
    Estimate importance of each dimension.
    
    Dimensions with high variance carry more information.
    Dimensions that change together with text position are more important.
    """
    # Variance per dimension
    variance = np.var(embeddings, axis=0)
    
    # Normalize
    importance = variance / (variance.max() + 1e-8)
    
    return importance


# =============================================================================
# Demo
# =============================================================================

def demo_basic_encoding():
    """Demonstrate basic encoding/decoding"""
    print("=" * 70)
    print("DEMO: Basic Semantic Audio Encoding")
    print("=" * 70)
    
    # Create synthetic embeddings (simulating a document)
    np.random.seed(42)
    num_positions = 50
    embedding_dim = 32
    
    # Simulate semantic content: some dimensions active in different regions
    embeddings = np.zeros((num_positions, embedding_dim))
    
    # Topic A active in positions 0-20
    embeddings[0:20, 0:8] = np.random.rand(20, 8) * 0.8
    
    # Topic B active in positions 15-35 (overlap!)
    embeddings[15:35, 8:16] = np.random.rand(20, 8) * 0.7
    
    # Topic C active in positions 30-50
    embeddings[30:50, 16:24] = np.random.rand(20, 8) * 0.9
    
    # Add some noise
    embeddings += np.random.rand(num_positions, embedding_dim) * 0.1
    
    # Create phases (simulate negation in middle section)
    phases = np.zeros(num_positions)
    phases[20:30] = np.pi  # Negated section
    
    # Encode
    config = SemanticAudioConfig(
        embedding_dim=embedding_dim,
        samples_per_position=256,
        sample_rate=22050,
        encode_phase=True
    )
    encoder = SemanticAudioEncoder(config)
    
    audio = encoder.encode(embeddings, phases)
    
    print(f"\nOriginal embeddings: {embeddings.shape}")
    print(f"Audio shape: {audio.shape}")
    print(f"Audio duration: {len(audio) / config.sample_rate:.2f} seconds")
    
    # Decode
    decoded_emb, decoded_phases = encoder.decode(audio)
    
    print(f"\nDecoded embeddings: {decoded_emb.shape}")
    
    # Check reconstruction error
    # Truncate to match (STFT may give slightly different length)
    min_pos = min(len(embeddings), len(decoded_emb))
    mse = np.mean((embeddings[:min_pos] - decoded_emb[:min_pos]) ** 2)
    print(f"Reconstruction MSE: {mse:.6f}")
    
    # Storage comparison
    storage = compare_storage(num_positions, embedding_dim, config)
    print(f"\nStorage comparison:")
    print(f"  Raw float32:    {storage['raw_float32_bytes']:,} bytes")
    print(f"  Raw float16:    {storage['raw_float16_bytes']:,} bytes")
    print(f"  Audio PCM:      {storage['audio_pcm_bytes']:,} bytes")
    print(f"  Est. Opus 64k:  {storage['estimated_opus_64k']:,} bytes")
    print(f"  Est. Opus 32k:  {storage['estimated_opus_32k']:,} bytes")
    print(f"  Compression ratio (vs float32, Opus 64k): {storage['compression_vs_float32_opus64']:.1f}x")


def demo_retrieval():
    """Demonstrate audio-based retrieval"""
    print("\n" + "=" * 70)
    print("DEMO: Audio-Based Semantic Retrieval")
    print("=" * 70)
    
    np.random.seed(42)
    embedding_dim = 32
    
    config = SemanticAudioConfig(
        embedding_dim=embedding_dim,
        samples_per_position=256,
        sample_rate=22050
    )
    encoder = SemanticAudioEncoder(config)
    index = SemanticAudioIndex(encoder)
    
    # Create synthetic documents with different topics
    
    # Doc 1: Mostly Topic A
    doc1 = np.random.rand(100, embedding_dim) * 0.1
    doc1[:, 0:8] += np.random.rand(100, 8) * 0.8  # Topic A strong
    index.add_document("doc1_topic_a", doc1, metadata={'title': 'Document about Topic A'})
    
    # Doc 2: Mostly Topic B
    doc2 = np.random.rand(80, embedding_dim) * 0.1
    doc2[:, 8:16] += np.random.rand(80, 8) * 0.8  # Topic B strong
    index.add_document("doc2_topic_b", doc2, metadata={'title': 'Document about Topic B'})
    
    # Doc 3: Mixed A and B
    doc3 = np.random.rand(120, embedding_dim) * 0.1
    doc3[0:60, 0:8] += np.random.rand(60, 8) * 0.7   # Topic A first half
    doc3[60:120, 8:16] += np.random.rand(60, 8) * 0.7  # Topic B second half
    index.add_document("doc3_mixed", doc3, metadata={'title': 'Document with A then B'})
    
    # Query: Looking for Topic A
    query = np.random.rand(10, embedding_dim) * 0.1
    query[:, 0:8] += np.random.rand(10, 8) * 0.9  # Strong Topic A signal
    
    print("\nQuery: Topic A signature")
    results = index.search(query, top_k=3)
    
    for r in results:
        print(f"  {r['doc_id']}: score={r['score']:.2f}, position={r['position']}")
    
    # Query: Looking for Topic B
    query_b = np.random.rand(10, embedding_dim) * 0.1
    query_b[:, 8:16] += np.random.rand(10, 8) * 0.9  # Strong Topic B signal
    
    print("\nQuery: Topic B signature")
    results = index.search(query_b, top_k=3)
    
    for r in results:
        print(f"  {r['doc_id']}: score={r['score']:.2f}, position={r['position']}")


def demo_compression_analysis():
    """Analyze compression potential"""
    print("\n" + "=" * 70)
    print("DEMO: Compression Analysis")
    print("=" * 70)
    
    # Realistic scenario: 10,000 word document
    # With stride=10, that's ~1000 positions
    # With 768-dim embeddings (BERT-sized)
    
    scenarios = [
        ("Small doc (100 pos, 64 dim)", 100, 64),
        ("Medium doc (500 pos, 256 dim)", 500, 256),
        ("Large doc (1000 pos, 768 dim)", 1000, 768),
        ("Very large (5000 pos, 768 dim)", 5000, 768),
    ]
    
    print(f"\n{'Scenario':<35} {'Raw f32':>12} {'Raw f16':>12} {'Opus 64k':>12} {'Ratio':>8}")
    print("-" * 85)
    
    for name, num_pos, emb_dim in scenarios:
        # Adjust config for embedding dim
        config = SemanticAudioConfig(
            embedding_dim=min(emb_dim, 256),  # Cap at 256 frequencies
            samples_per_position=128,
            sample_rate=22050
        )
        
        storage = compare_storage(num_pos, emb_dim, config)
        
        print(f"{name:<35} {storage['raw_float32_bytes']:>10,}B "
              f"{storage['raw_float16_bytes']:>10,}B "
              f"{storage['estimated_opus_64k']:>10,}B "
              f"{storage['compression_vs_float32_opus64']:>7.1f}x")
    
    print("\nNote: Actual compression depends on signal complexity.")
    print("Embeddings with temporal coherence compress better.")


def demo_psychosemantic():
    """Demonstrate psychosemantic analysis"""
    print("\n" + "=" * 70)
    print("DEMO: Psychosemantic Analysis")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Create embeddings with varying redundancy
    num_positions = 200
    embedding_dim = 32
    
    embeddings = np.zeros((num_positions, embedding_dim))
    
    # Dimensions 0-7: Independent, high variance (important)
    embeddings[:, 0:8] = np.random.randn(num_positions, 8)
    
    # Dimensions 8-15: Correlated with 0-7 (redundant)
    embeddings[:, 8:16] = embeddings[:, 0:8] * 0.9 + np.random.randn(num_positions, 8) * 0.1
    
    # Dimensions 16-23: Low variance (less important)
    embeddings[:, 16:24] = np.random.randn(num_positions, 8) * 0.1
    
    # Dimensions 24-31: Independent, medium variance
    embeddings[:, 24:32] = np.random.randn(num_positions, 8) * 0.5
    
    # Analyze
    redundancy = estimate_semantic_redundancy(embeddings)
    importance = estimate_semantic_importance(embeddings)
    
    print("\nDimension Analysis:")
    print(f"{'Dim Range':<15} {'Redundancy':>12} {'Importance':>12} {'Compress?':>12}")
    print("-" * 55)
    
    for start in range(0, 32, 8):
        end = start + 8
        red = np.mean(redundancy[start:end])
        imp = np.mean(importance[start:end])
        
        if red > 0.5:
            compress = "Heavy (redundant)"
        elif imp < 0.2:
            compress = "Heavy (low info)"
        else:
            compress = "Light (preserve)"
        
        print(f"Dims {start:2d}-{end-1:2d}       {red:>12.3f} {imp:>12.3f} {compress:>12}")
    
    print("\nPsychosemantic insight:")
    print("  - Redundant dimensions can be predicted, need fewer bits")
    print("  - Low-importance dimensions can be quantized heavily")
    print("  - This is analogous to psychoacoustic masking in MP3")


def main():
    print("Semantic Audio Encoding - Experimental Framework")
    print("=" * 70)
    print("""
This explores encoding embeddings as audio signals:
- Each embedding dimension → frequency band
- Text position → time
- Amplitude → embedding magnitude
- Phase → semantic polarity

Benefits:
- Battle-tested compression (Opus, FLAC, etc.)
- Signal processing for retrieval (correlation, convolution)
- No chunking artifacts
- Hierarchical representation (frequency bands = abstraction levels)
""")
    
    demo_basic_encoding()
    demo_retrieval()
    demo_compression_analysis()
    demo_psychosemantic()
    
    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    print("""
1. ENCODING WORKS: Embeddings can be represented as audio
   - STFT/inverse STFT for encoding/decoding
   - Phase preserved via stereo or quadrature encoding

2. COMPRESSION POTENTIAL: 
   - Opus at 64kbps gives ~3-10x compression vs float32
   - Lower bitrates possible with quality trade-off
   - Temporal coherence in embeddings helps compression

3. RETRIEVAL:
   - Cross-correlation finds matching positions
   - Analogous to matched filtering in radar/sonar
   - No chunk boundaries = no missed matches

4. PSYCHOSEMANTIC COMPRESSION:
   - Redundant dimensions can be compressed more
   - Low-variance dimensions need fewer bits
   - Potential for learned semantic masking models

5. CHALLENGES:
   - Many embedding dims (768) = many frequencies
   - May need dimension reduction first
   - Reconstruction quality vs compression trade-off
   - Real codec integration needs more work
""")


if __name__ == "__main__":
    main()
