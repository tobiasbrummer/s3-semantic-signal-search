#!/usr/bin/env python3
"""
S3 - Semantic Signal Search
Prototype Implementation

Multi-Band LSH + Inverted Index + Audio Synthesis
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import struct


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class S3Config:
    """Configuration for S3 system."""
    
    # Band definitions (input dimensions → output bits)
    bands: List[Tuple[int, int, int, float]] = None  # (start, end, bits, weight)
    
    # Audio synthesis
    sample_rate: int = 16000
    audio_duration: float = 2.0
    
    def __post_init__(self):
        if self.bands is None:
            # Default: 768D embedding split into 3 bands
            self.bands = [
                (0, 64, 32, 4.0),      # Band 0: Bass, high weight
                (64, 256, 64, 2.0),    # Band 1: Mids, medium weight
                (256, 768, 128, 1.0),  # Band 2: Highs, low weight
            ]
    
    @property
    def total_bits(self) -> int:
        return sum(b[2] for b in self.bands)
    
    @property
    def total_bytes(self) -> int:
        return (self.total_bits + 7) // 8


# =============================================================================
# Multi-Band LSH
# =============================================================================

class MultiBandLSH:
    """
    Locality Sensitive Hashing with separate projections per frequency band.
    
    Each band gets its own random projection matrix, producing a separate
    bit string. This allows hierarchical search (bass first, then details).
    """
    
    def __init__(self, config: S3Config, seed: int = 42):
        self.config = config
        self.seed = seed
        self.projections = {}
        
        # Initialize random projections for each band
        rng = np.random.RandomState(seed)
        
        for i, (start, end, bits, weight) in enumerate(config.bands):
            input_dim = end - start
            # Random hyperplanes for LSH
            # Sign of dot product with hyperplane → 1 bit
            self.projections[i] = rng.randn(bits, input_dim).astype(np.float32)
    
    def hash_embedding(self, embedding: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Hash a single embedding to multi-band binary codes.
        
        Args:
            embedding: 1D array of floats (e.g., 768D)
            
        Returns:
            Dict mapping band_id → binary array (uint8 packed bits)
        """
        result = {}
        
        for i, (start, end, bits, weight) in enumerate(self.config.bands):
            # Extract band dimensions
            band_embedding = embedding[start:end]
            
            # Project and sign
            projections = self.projections[i]
            dot_products = projections @ band_embedding
            
            # Convert to bits (1 if positive, 0 if negative)
            bits_array = (dot_products > 0).astype(np.uint8)
            
            # Pack into bytes
            packed = self._pack_bits(bits_array)
            result[i] = packed
        
        return result
    
    def hash_batch(self, embeddings: np.ndarray) -> List[Dict[int, np.ndarray]]:
        """Hash a batch of embeddings."""
        return [self.hash_embedding(emb) for emb in embeddings]
    
    def _pack_bits(self, bits: np.ndarray) -> np.ndarray:
        """Pack bit array into bytes."""
        # Pad to multiple of 8
        padded_len = ((len(bits) + 7) // 8) * 8
        padded = np.zeros(padded_len, dtype=np.uint8)
        padded[:len(bits)] = bits
        
        # Pack 8 bits into each byte
        packed = np.packbits(padded)
        return packed
    
    def _unpack_bits(self, packed: np.ndarray, num_bits: int) -> np.ndarray:
        """Unpack bytes back to bit array."""
        unpacked = np.unpackbits(packed)
        return unpacked[:num_bits]
    
    def to_hex_string(self, hashed: Dict[int, np.ndarray]) -> Dict[int, str]:
        """Convert packed bytes to hex string for indexing."""
        return {
            band_id: bytes(packed).hex()
            for band_id, packed in hashed.items()
        }
    
    def from_hex_string(self, hex_dict: Dict[int, str]) -> Dict[int, np.ndarray]:
        """Convert hex strings back to packed bytes."""
        return {
            band_id: np.frombuffer(bytes.fromhex(hex_str), dtype=np.uint8)
            for band_id, hex_str in hex_dict.items()
        }


# =============================================================================
# Hamming Distance Computation
# =============================================================================

class HammingRanker:
    """
    Compute weighted Hamming distance for ranking.
    
    Uses band weights to prioritize bass (coarse meaning) over highs (details).
    """
    
    def __init__(self, config: S3Config):
        self.config = config
        self.weights = {i: w for i, (_, _, _, w) in enumerate(config.bands)}
    
    def hamming_distance(self, a: np.ndarray, b: np.ndarray) -> int:
        """Compute Hamming distance between two packed byte arrays."""
        # XOR and count bits
        xor_result = np.bitwise_xor(a, b)
        # Count 1s in each byte and sum
        return sum(bin(byte).count('1') for byte in xor_result)
    
    def weighted_distance(self, 
                          query_hash: Dict[int, np.ndarray],
                          doc_hash: Dict[int, np.ndarray]) -> float:
        """
        Compute weighted Hamming distance across all bands.
        
        Returns:
            Weighted sum of Hamming distances (lower = more similar)
        """
        total = 0.0
        
        for band_id in query_hash:
            if band_id in doc_hash:
                dist = self.hamming_distance(query_hash[band_id], doc_hash[band_id])
                total += self.weights[band_id] * dist
        
        return total
    
    def rank_candidates(self,
                        query_hash: Dict[int, np.ndarray],
                        candidates: List[Tuple[str, Dict[int, np.ndarray]]],
                        top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Rank candidates by weighted Hamming distance.
        
        Args:
            query_hash: Hashed query
            candidates: List of (doc_id, doc_hash) tuples
            top_k: Number of results to return
            
        Returns:
            List of (doc_id, distance) tuples, sorted by distance ascending
        """
        scored = []
        
        for doc_id, doc_hash in candidates:
            dist = self.weighted_distance(query_hash, doc_hash)
            scored.append((doc_id, dist))
        
        # Sort by distance (ascending)
        scored.sort(key=lambda x: x[1])
        
        return scored[:top_k]


# =============================================================================
# Bit Stream Synthesizer (Audio Generation)
# =============================================================================

class BitStreamSynthesizer:
    """
    Generate audio from LSH bit codes.
    
    Each band controls a different audio layer:
    - Band 0 (Bass): Low sine oscillators
    - Band 1 (Mids): Pad/chord synthesizer  
    - Band 2 (Highs): Noise/texture generator
    """
    
    def __init__(self, config: S3Config):
        self.config = config
        self.sample_rate = config.sample_rate
        self.duration = config.audio_duration
        
        # Frequency mappings for each band
        self.bass_freqs = self._generate_bass_freqs()
        self.mid_freqs = self._generate_mid_freqs()
    
    def _generate_bass_freqs(self) -> np.ndarray:
        """Generate bass frequencies (C1 to B2, ~32 notes)."""
        # Start at C1 (32.7 Hz), go up chromatically
        base = 32.70  # C1
        return base * (2 ** (np.arange(32) / 12))
    
    def _generate_mid_freqs(self) -> np.ndarray:
        """Generate mid frequencies (C3 to B4, ~24 notes)."""
        base = 130.81  # C3
        return base * (2 ** (np.arange(24) / 12))
    
    def synthesize(self, hashed: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Generate audio from hash codes.
        
        Args:
            hashed: Dict of band_id → packed bytes
            
        Returns:
            Audio samples as float32 array
        """
        num_samples = int(self.sample_rate * self.duration)
        t = np.linspace(0, self.duration, num_samples)
        
        audio = np.zeros(num_samples, dtype=np.float32)
        
        # Band 0: Bass layer
        if 0 in hashed:
            audio += self._synth_bass(hashed[0], t) * 0.4
        
        # Band 1: Mids layer
        if 1 in hashed:
            audio += self._synth_mids(hashed[1], t) * 0.4
        
        # Band 2: Highs layer (texture)
        if 2 in hashed:
            audio += self._synth_highs(hashed[2], t) * 0.2
        
        # Normalize
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * 0.9
        
        return audio
    
    def _synth_bass(self, packed: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Synthesize bass layer from Band 0 bits."""
        bits = np.unpackbits(packed)[:32]
        
        audio = np.zeros_like(t)
        
        for i, bit in enumerate(bits):
            if bit:
                freq = self.bass_freqs[i]
                # Sine wave with slow attack
                envelope = 1 - np.exp(-t * 3)
                audio += np.sin(2 * np.pi * freq * t) * envelope
        
        return audio
    
    def _synth_mids(self, packed: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Synthesize mid layer from Band 1 bits."""
        bits = np.unpackbits(packed)[:64]
        
        audio = np.zeros_like(t)
        
        # Use bits to select notes and create chord
        note_bits = bits[:24]
        voicing_bits = bits[24:48]
        modulation_bits = bits[48:64]
        
        # Notes that are "on"
        active_notes = np.where(note_bits)[0]
        
        for note_idx in active_notes[:8]:  # Max 8 notes
            freq = self.mid_freqs[note_idx % len(self.mid_freqs)]
            
            # Add harmonics based on voicing bits
            for h in range(1, 5):
                harmonic_weight = 1.0 / h
                if note_idx + h < len(voicing_bits) and voicing_bits[note_idx + h]:
                    harmonic_weight *= 1.5
                
                audio += np.sin(2 * np.pi * freq * h * t) * harmonic_weight * 0.3
        
        # Apply tremolo based on modulation bits
        mod_rate = 2 + sum(modulation_bits[:8]) * 0.5
        tremolo = 1 + 0.2 * np.sin(2 * np.pi * mod_rate * t)
        audio *= tremolo
        
        return audio
    
    def _synth_highs(self, packed: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Synthesize high frequency texture from Band 2 bits."""
        bits = np.unpackbits(packed)[:128]
        
        # Noise color based on first 32 bits
        noise_bits = bits[:32]
        noise_color = sum(noise_bits) / 32  # 0 = brown, 1 = white
        
        # Generate noise
        rng = np.random.RandomState(int(sum(bits[:8])))  # Seed from bits
        noise = rng.randn(len(t))
        
        # Color the noise (simple lowpass for brown)
        if noise_color < 0.5:
            # Brownish noise - lowpass
            from scipy.ndimage import gaussian_filter1d
            noise = gaussian_filter1d(noise, sigma=int((1 - noise_color) * 50) + 1)
        
        # Modulate amplitude with bits
        attack_bits = bits[32:64]
        attack_rate = sum(attack_bits) / 32 * 10 + 1
        envelope = 1 - np.exp(-t * attack_rate)
        
        # Stereo spread could be added here using bits 96-127
        
        return noise * envelope * 0.3


# =============================================================================
# Simple In-Memory Index (for prototyping)
# =============================================================================

class S3Index:
    """
    Simple in-memory index for S3 prototype.
    
    In production, replace with Elasticsearch/Lucene.
    """
    
    def __init__(self, config: S3Config):
        self.config = config
        self.lsh = MultiBandLSH(config)
        self.ranker = HammingRanker(config)
        self.synthesizer = BitStreamSynthesizer(config)
        
        # Storage
        self.documents = {}  # doc_id → text
        self.hashes = {}     # doc_id → hash dict
        
        # Inverted index per band
        self.inverted = {i: {} for i in range(len(config.bands))}
    
    def index_document(self, doc_id: str, embedding: np.ndarray, text: str = ""):
        """Add a document to the index."""
        # Hash the embedding
        hashed = self.lsh.hash_embedding(embedding)
        hex_hashed = self.lsh.to_hex_string(hashed)
        
        # Store
        self.documents[doc_id] = text
        self.hashes[doc_id] = hashed
        
        # Update inverted index
        for band_id, hex_str in hex_hashed.items():
            if hex_str not in self.inverted[band_id]:
                self.inverted[band_id][hex_str] = []
            self.inverted[band_id][hex_str].append(doc_id)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Dict]:
        """
        Search for similar documents.
        
        Returns list of {doc_id, distance, text, audio}
        """
        # Hash query
        query_hash = self.lsh.hash_embedding(query_embedding)
        query_hex = self.lsh.to_hex_string(query_hash)
        
        # Stage 1: Find candidates using Band 0 (inverted index)
        band0_hex = query_hex[0]
        candidate_ids = set()
        
        # Exact match
        if band0_hex in self.inverted[0]:
            candidate_ids.update(self.inverted[0][band0_hex])
        
        # If too few candidates, do fuzzy match (flip 1 bit at a time)
        if len(candidate_ids) < top_k * 3:
            candidate_ids.update(self._fuzzy_lookup(band0_hex, band_id=0))
        
        # If still too few, fall back to all documents
        if len(candidate_ids) < top_k:
            candidate_ids = set(self.hashes.keys())
        
        # Stage 2: Rank by weighted Hamming distance
        candidates = [(doc_id, self.hashes[doc_id]) for doc_id in candidate_ids]
        ranked = self.ranker.rank_candidates(query_hash, candidates, top_k)
        
        # Build results
        results = []
        for doc_id, distance in ranked:
            results.append({
                'doc_id': doc_id,
                'distance': distance,
                'text': self.documents.get(doc_id, ""),
                'hash': self.hashes[doc_id],
            })
        
        return results
    
    def _fuzzy_lookup(self, hex_str: str, band_id: int, max_flips: int = 2) -> set:
        """Find candidates with up to max_flips bit differences."""
        candidates = set()
        
        # Convert hex to bytes
        original = bytes.fromhex(hex_str)
        
        # Try flipping each bit
        for byte_idx in range(len(original)):
            for bit_idx in range(8):
                # Flip one bit
                modified = bytearray(original)
                modified[byte_idx] ^= (1 << bit_idx)
                modified_hex = bytes(modified).hex()
                
                if modified_hex in self.inverted[band_id]:
                    candidates.update(self.inverted[band_id][modified_hex])
        
        return candidates
    
    def get_audio(self, doc_id: str) -> Optional[np.ndarray]:
        """Generate audio for a document."""
        if doc_id not in self.hashes:
            return None
        return self.synthesizer.synthesize(self.hashes[doc_id])
    
    def compare_audio(self, doc_id_a: str, doc_id_b: str) -> np.ndarray:
        """Generate mixed audio of two documents for comparison."""
        audio_a = self.get_audio(doc_id_a)
        audio_b = self.get_audio(doc_id_b)
        
        if audio_a is None or audio_b is None:
            return np.array([])
        
        # Mix 50/50
        return (audio_a + audio_b) / 2


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate S3 system."""
    print("=" * 70)
    print("S3 - SEMANTIC SIGNAL SEARCH DEMO")
    print("=" * 70)
    
    # Config
    config = S3Config()
    print(f"\nConfiguration:")
    print(f"  Bands: {len(config.bands)}")
    print(f"  Total bits: {config.total_bits}")
    print(f"  Bytes per document: {config.total_bytes}")
    
    # Create index
    index = S3Index(config)
    
    # Generate fake embeddings (in reality, use sentence-transformers)
    np.random.seed(42)
    
    documents = [
        ("doc_1", "How to cancel a contract immediately"),
        ("doc_2", "Steps to terminate your subscription"),
        ("doc_3", "Contract cancellation procedures"),
        ("doc_4", "Best recipes for chocolate cake"),
        ("doc_5", "How to bake a birthday cake"),
        ("doc_6", "Legal advice for contract disputes"),
    ]
    
    # Simulate embeddings (similar docs have similar embeddings)
    base_contract = np.random.randn(768)
    base_cake = np.random.randn(768)
    
    embeddings = {
        "doc_1": base_contract + np.random.randn(768) * 0.1,
        "doc_2": base_contract + np.random.randn(768) * 0.15,
        "doc_3": base_contract + np.random.randn(768) * 0.12,
        "doc_4": base_cake + np.random.randn(768) * 0.1,
        "doc_5": base_cake + np.random.randn(768) * 0.12,
        "doc_6": base_contract + np.random.randn(768) * 0.2,
    }
    
    # Index documents
    print("\n" + "-" * 70)
    print("INDEXING DOCUMENTS")
    print("-" * 70)
    
    for doc_id, text in documents:
        index.index_document(doc_id, embeddings[doc_id], text)
        hex_hash = index.lsh.to_hex_string(index.hashes[doc_id])
        print(f"  {doc_id}: {text[:40]}...")
        print(f"    Band 0 (Bass): {hex_hash[0][:16]}...")
    
    # Search
    print("\n" + "-" * 70)
    print("SEARCH: 'How to cancel a contract'")
    print("-" * 70)
    
    query_embedding = base_contract + np.random.randn(768) * 0.08
    results = index.search(query_embedding, top_k=5)
    
    print("\nResults (sorted by weighted Hamming distance):")
    for i, result in enumerate(results):
        print(f"  {i+1}. {result['doc_id']}: distance={result['distance']:.1f}")
        print(f"     {result['text']}")
    
    # Audio generation
    print("\n" + "-" * 70)
    print("AUDIO GENERATION")
    print("-" * 70)
    
    for doc_id in ["doc_1", "doc_4"]:
        audio = index.get_audio(doc_id)
        print(f"  {doc_id}: Generated {len(audio)} samples ({len(audio)/config.sample_rate:.1f}s)")
        print(f"    Peak amplitude: {np.abs(audio).max():.3f}")
        print(f"    RMS: {np.sqrt(np.mean(audio**2)):.3f}")
    
    # Compare similar vs different
    print("\n" + "-" * 70)
    print("AUDIO COMPARISON")
    print("-" * 70)
    
    # Similar docs
    dist_similar = index.ranker.weighted_distance(
        index.hashes["doc_1"], 
        index.hashes["doc_2"]
    )
    print(f"  doc_1 vs doc_2 (both about contracts):")
    print(f"    Weighted Hamming distance: {dist_similar:.1f}")
    print(f"    → Should sound HARMONIOUS when mixed")
    
    # Different docs
    dist_different = index.ranker.weighted_distance(
        index.hashes["doc_1"], 
        index.hashes["doc_4"]
    )
    print(f"\n  doc_1 vs doc_4 (contract vs cake):")
    print(f"    Weighted Hamming distance: {dist_different:.1f}")
    print(f"    → Should sound DISSONANT when mixed")
    
    # Stats
    print("\n" + "-" * 70)
    print("EFFICIENCY STATS")
    print("-" * 70)
    
    # Compare storage
    float_bytes = 768 * 4  # 768 floats × 4 bytes
    lsh_bytes = config.total_bytes
    
    print(f"  Storage per document:")
    print(f"    Float embedding: {float_bytes} bytes")
    print(f"    S3 hash:         {lsh_bytes} bytes")
    print(f"    Compression:     {float_bytes / lsh_bytes:.1f}x smaller")
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Replace fake embeddings with real sentence-transformers")
    print("  2. Index in Elasticsearch instead of in-memory")
    print("  3. Save audio to files and listen!")
    print("  4. Benchmark recall on standard datasets")


if __name__ == "__main__":
    demo()
