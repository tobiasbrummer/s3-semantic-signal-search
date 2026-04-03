#!/usr/bin/env python3
"""
Embedding Compression Test: RVQ, LPC, GOP - einzeln und kombiniert.

Testet verschiedene Audio-Codec-Techniken auf Token-Level Embeddings:
- RVQ (Residual Vector Quantization): Hierarchische Quantisierung über Bänder
- LPC (Linear Predictive Coding): Prediction + Sparse Residuals
- GOP (Group of Pictures): I-Frames an Onsets, P-Frames als Deltas
"""

import numpy as np
import requests
from dataclasses import dataclass
from typing import Optional
import time


# =============================================================================
# ENCODER
# =============================================================================

class TokenEncoder:
    def __init__(self, url: str = "http://localhost:8202"):
        self.url = url

    def encode(self, text: str) -> np.ndarray:
        response = requests.post(f"{self.url}/embeddings", json={"input": text})
        response.raise_for_status()
        data = response.json()
        if isinstance(data[0]["embedding"][0], list):
            return np.array(data[0]["embedding"], dtype=np.float32)
        return np.array([data[0]["embedding"]], dtype=np.float32)


# =============================================================================
# ONSET DETECTION (für GOP I-Frames)
# =============================================================================

def detect_onsets(embeddings: np.ndarray, threshold_pct: float = 95,
                  min_dist: int = 3) -> np.ndarray:
    """Spectral Flux Onset Detection."""
    if len(embeddings) < 2:
        return np.array([0])

    # Spectral flux
    diff = np.diff(embeddings, axis=0)
    flux = np.abs(diff).sum(axis=1)

    # Threshold
    threshold = np.percentile(flux, threshold_pct)

    # Peaks über Threshold
    peaks = [0]  # Erstes Token ist immer I-Frame
    for i in range(len(flux)):
        if flux[i] > threshold:
            if len(peaks) == 0 or (i + 1) - peaks[-1] >= min_dist:
                peaks.append(i + 1)

    return np.array(peaks)


# =============================================================================
# RVQ (Residual Vector Quantization)
# =============================================================================

@dataclass
class RVQCodebook:
    """Simple RVQ Codebook."""
    centroids: np.ndarray  # (n_codes, dim)

    @classmethod
    def train(cls, data: np.ndarray, n_codes: int = 256, n_iter: int = 20):
        """Train codebook with k-means."""
        n_samples, dim = data.shape

        # Initialize with random samples
        indices = np.random.choice(n_samples, min(n_codes, n_samples), replace=False)
        centroids = data[indices].copy()

        for _ in range(n_iter):
            # Assign
            distances = np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=2)
            assignments = np.argmin(distances, axis=1)

            # Update
            for k in range(n_codes):
                mask = assignments == k
                if mask.sum() > 0:
                    centroids[k] = data[mask].mean(axis=0)

        return cls(centroids=centroids)

    def quantize(self, data: np.ndarray) -> np.ndarray:
        """Quantize data to codebook indices."""
        distances = np.linalg.norm(data[:, None, :] - self.centroids[None, :, :], axis=2)
        return np.argmin(distances, axis=1).astype(np.uint16)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Decode codes back to vectors."""
        return self.centroids[codes]


class RVQEncoder:
    """Multi-stage RVQ encoder (one codebook per Matryoshka band)."""

    BANDS = [(0, 256), (256, 512), (512, 768), (768, 1024)]

    def __init__(self, n_codes: int = 256):
        self.n_codes = n_codes
        self.codebooks: list[RVQCodebook] = []

    def train(self, embeddings: np.ndarray):
        """Train codebooks on embedding data."""
        print(f"  Training RVQ codebooks ({self.n_codes} codes per band)...")
        self.codebooks = []

        for i, (start, end) in enumerate(self.BANDS):
            if start >= embeddings.shape[1]:
                break
            actual_end = min(end, embeddings.shape[1])
            band_data = embeddings[:, start:actual_end]

            codebook = RVQCodebook.train(band_data, n_codes=self.n_codes)
            self.codebooks.append(codebook)
            print(f"    Band {i+1} ({start}-{actual_end}): trained")

    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        """Encode embeddings to RVQ codes. Returns (n_tokens, n_bands) uint16."""
        n_tokens = len(embeddings)
        n_bands = len(self.codebooks)
        codes = np.zeros((n_tokens, n_bands), dtype=np.uint16)

        for i, (start, end) in enumerate(self.BANDS[:n_bands]):
            actual_end = min(end, embeddings.shape[1])
            band_data = embeddings[:, start:actual_end]
            codes[:, i] = self.codebooks[i].quantize(band_data)

        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Decode RVQ codes back to embeddings."""
        n_tokens = len(codes)
        n_dims = sum(min(e, 1024) - s for s, e in self.BANDS[:len(self.codebooks)])
        embeddings = np.zeros((n_tokens, n_dims), dtype=np.float32)

        offset = 0
        for i, (start, end) in enumerate(self.BANDS[:len(self.codebooks)]):
            actual_end = min(end, 1024)
            band_dim = actual_end - start
            decoded = self.codebooks[i].decode(codes[:, i])
            embeddings[:, offset:offset+band_dim] = decoded
            offset += band_dim

        return embeddings

    def size_bytes(self, codes: np.ndarray) -> int:
        """Storage size in bytes."""
        return codes.nbytes


# =============================================================================
# LPC (Linear Predictive Coding)
# =============================================================================

@dataclass
class LPCEncoded:
    """LPC encoded representation."""
    coeffs: np.ndarray      # LPC coefficients per dimension
    residuals: np.ndarray   # Full residuals (for reconstruction)
    sparse_residuals: dict  # Sparse: {position: residual_vector}
    threshold: float


class LPCEncoder:
    """LPC encoder for embedding sequences."""

    def __init__(self, order: int = 4, threshold_pct: float = 90):
        self.order = order
        self.threshold_pct = threshold_pct

    def encode(self, embeddings: np.ndarray) -> LPCEncoded:
        """Encode sequence with LPC."""
        n_tokens, n_dims = embeddings.shape

        # Simple LPC: predict as weighted sum of previous tokens
        # For simplicity, use mean of last `order` tokens as prediction
        predicted = np.zeros_like(embeddings)

        for t in range(n_tokens):
            if t == 0:
                predicted[t] = 0  # No prediction for first
            else:
                start = max(0, t - self.order)
                predicted[t] = embeddings[start:t].mean(axis=0)

        residuals = embeddings - predicted

        # Sparse residuals: only store large changes
        residual_norms = np.linalg.norm(residuals, axis=1)
        threshold = np.percentile(residual_norms, self.threshold_pct)

        sparse_residuals = {}
        for t in range(n_tokens):
            if residual_norms[t] > threshold or t == 0:
                sparse_residuals[t] = residuals[t]

        # Compute LPC coefficients (simplified: just store prediction weights)
        coeffs = np.ones(self.order) / self.order

        return LPCEncoded(
            coeffs=coeffs,
            residuals=residuals,
            sparse_residuals=sparse_residuals,
            threshold=threshold
        )

    def decode(self, encoded: LPCEncoded, n_tokens: int, first_embedding: np.ndarray) -> np.ndarray:
        """Decode LPC back to embeddings."""
        n_dims = len(first_embedding)
        embeddings = np.zeros((n_tokens, n_dims), dtype=np.float32)
        embeddings[0] = first_embedding

        for t in range(1, n_tokens):
            # Predict from previous
            start = max(0, t - self.order)
            predicted = embeddings[start:t].mean(axis=0)

            # Add residual
            if t in encoded.sparse_residuals:
                embeddings[t] = predicted + encoded.sparse_residuals[t]
            else:
                embeddings[t] = predicted  # Use prediction only

        return embeddings

    def size_bytes(self, encoded: LPCEncoded) -> int:
        """Storage size in bytes (sparse representation)."""
        # Coefficients + sparse residuals
        coeff_size = encoded.coeffs.nbytes
        sparse_size = sum(r.nbytes for r in encoded.sparse_residuals.values())
        # Positions (int32)
        pos_size = len(encoded.sparse_residuals) * 4
        return coeff_size + sparse_size + pos_size


# =============================================================================
# GOP (Group of Pictures)
# =============================================================================

@dataclass
class GOPEncoded:
    """GOP encoded representation."""
    i_frames: dict          # {position: embedding}
    p_frames: dict          # {position: delta}
    frame_types: np.ndarray # 'I' or 'P' per token


class GOPEncoder:
    """GOP encoder using onsets as I-frames."""

    def __init__(self, onset_threshold: float = 95, quantize_p: bool = True):
        self.onset_threshold = onset_threshold
        self.quantize_p = quantize_p

    def encode(self, embeddings: np.ndarray) -> GOPEncoded:
        """Encode with GOP structure."""
        n_tokens = len(embeddings)

        # Detect onsets (I-frame positions)
        i_positions = detect_onsets(embeddings, threshold_pct=self.onset_threshold)
        i_positions_set = set(i_positions)

        frame_types = np.array(['P'] * n_tokens)
        frame_types[list(i_positions_set)] = 'I'

        i_frames = {}
        p_frames = {}

        last_i_frame = None
        last_i_pos = 0

        for t in range(n_tokens):
            if t in i_positions_set:
                # I-frame: store full embedding
                i_frames[t] = embeddings[t]
                last_i_frame = embeddings[t]
                last_i_pos = t
            else:
                # P-frame: store delta from last I-frame
                if last_i_frame is not None:
                    delta = embeddings[t] - last_i_frame
                    if self.quantize_p:
                        # Quantize delta to int8 (-128 to 127)
                        # Scale by max absolute value
                        scale = np.abs(delta).max() + 1e-8
                        quantized = np.clip(delta / scale * 127, -128, 127).astype(np.int8)
                        p_frames[t] = {'quantized': quantized, 'scale': np.float16(scale)}
                    else:
                        p_frames[t] = delta

        return GOPEncoded(
            i_frames=i_frames,
            p_frames=p_frames,
            frame_types=frame_types
        )

    def decode(self, encoded: GOPEncoded) -> np.ndarray:
        """Decode GOP back to embeddings."""
        n_tokens = len(encoded.frame_types)
        n_dims = len(list(encoded.i_frames.values())[0])

        embeddings = np.zeros((n_tokens, n_dims), dtype=np.float32)

        last_i_frame = None

        for t in range(n_tokens):
            if t in encoded.i_frames:
                embeddings[t] = encoded.i_frames[t]
                last_i_frame = encoded.i_frames[t]
            elif t in encoded.p_frames and last_i_frame is not None:
                p_data = encoded.p_frames[t]
                if isinstance(p_data, dict):
                    # Dequantize
                    delta = p_data['quantized'].astype(np.float32) / 127 * float(p_data['scale'])
                else:
                    delta = p_data
                embeddings[t] = last_i_frame + delta

        return embeddings

    def size_bytes(self, encoded: GOPEncoded) -> int:
        """Storage size in bytes."""
        i_size = sum(e.nbytes for e in encoded.i_frames.values())

        # P-frames: check if quantized
        p_size = 0
        for p_data in encoded.p_frames.values():
            if isinstance(p_data, dict):
                p_size += p_data['quantized'].nbytes + 2  # int8 array + float16 scale
            else:
                p_size += p_data.nbytes

        # Positions
        pos_size = (len(encoded.i_frames) + len(encoded.p_frames)) * 4
        return i_size + p_size + pos_size


# =============================================================================
# COMBINED: RVQ + LPC + GOP
# =============================================================================

@dataclass
class CombinedEncoded:
    """Combined encoding: GOP structure with RVQ for I-frames, LPC for P-frames."""
    i_frame_codes: dict     # {position: RVQ codes}
    p_frame_residuals: dict # {position: sparse LPC residual}
    frame_types: np.ndarray
    lpc_coeffs: np.ndarray


class CombinedEncoder:
    """Combined encoder: GOP + RVQ + quantized P-frames."""

    def __init__(self, n_codes: int = 256, onset_threshold: float = 95,
                 delta_threshold_pct: float = 50):
        self.rvq = RVQEncoder(n_codes=n_codes)
        self.onset_threshold = onset_threshold
        self.delta_threshold_pct = delta_threshold_pct

    def train(self, embeddings: np.ndarray):
        """Train RVQ codebooks."""
        self.rvq.train(embeddings)

    def encode(self, embeddings: np.ndarray) -> CombinedEncoded:
        """Encode with combined approach: RVQ I-frames + quantized P-frames."""
        n_tokens = len(embeddings)

        # GOP: detect I-frame positions
        i_positions = detect_onsets(embeddings, threshold_pct=self.onset_threshold)
        i_positions_set = set(i_positions)

        frame_types = np.array(['P'] * n_tokens)
        frame_types[list(i_positions_set)] = 'I'

        # RVQ encode I-frames
        i_frame_embeddings = np.array([embeddings[t] for t in sorted(i_positions_set)])
        i_frame_codes_array = self.rvq.encode(i_frame_embeddings)

        i_frame_codes = {}
        for idx, t in enumerate(sorted(i_positions_set)):
            i_frame_codes[t] = i_frame_codes_array[idx]

        # Quantized P-frames (delta from last I-frame)
        p_frame_residuals = {}
        last_i_embedding = embeddings[0]

        # Compute all deltas first to determine threshold
        deltas = {}
        delta_norms = []

        for t in range(n_tokens):
            if t in i_positions_set:
                last_i_embedding = embeddings[t]
            else:
                delta = embeddings[t] - last_i_embedding
                deltas[t] = delta
                delta_norms.append(np.linalg.norm(delta))

        # Only store P-frames with significant deltas (above threshold)
        if delta_norms:
            threshold = np.percentile(delta_norms, self.delta_threshold_pct)

            for t, delta in deltas.items():
                if np.linalg.norm(delta) > threshold:
                    # Quantize to int8
                    scale = np.abs(delta).max() + 1e-8
                    quantized = np.clip(delta / scale * 127, -128, 127).astype(np.int8)
                    p_frame_residuals[t] = {'quantized': quantized, 'scale': np.float16(scale)}

        return CombinedEncoded(
            i_frame_codes=i_frame_codes,
            p_frame_residuals=p_frame_residuals,
            frame_types=frame_types,
            lpc_coeffs=np.array([])  # Not used anymore
        )

    def decode(self, encoded: CombinedEncoded) -> np.ndarray:
        """Decode combined representation."""
        n_tokens = len(encoded.frame_types)

        # First, decode all I-frames via RVQ
        i_positions = sorted(encoded.i_frame_codes.keys())
        i_codes = np.array([encoded.i_frame_codes[t] for t in i_positions])
        i_embeddings = self.rvq.decode(i_codes)

        i_frame_decoded = {t: i_embeddings[idx] for idx, t in enumerate(i_positions)}

        n_dims = i_embeddings.shape[1]
        embeddings = np.zeros((n_tokens, n_dims), dtype=np.float32)

        last_i_embedding = None

        for t in range(n_tokens):
            if t in i_frame_decoded:
                embeddings[t] = i_frame_decoded[t]
                last_i_embedding = i_frame_decoded[t]
            else:
                # P-frame: use last I-frame + optional delta
                if last_i_embedding is not None:
                    if t in encoded.p_frame_residuals:
                        p_data = encoded.p_frame_residuals[t]
                        delta = p_data['quantized'].astype(np.float32) / 127 * float(p_data['scale'])
                        embeddings[t] = last_i_embedding + delta
                    else:
                        # No stored delta → use I-frame directly
                        embeddings[t] = last_i_embedding

        return embeddings

    def size_bytes(self, encoded: CombinedEncoded) -> int:
        """Storage size in bytes."""
        # I-frame codes (n_bands * 2 bytes each)
        i_size = sum(codes.nbytes for codes in encoded.i_frame_codes.values())
        i_pos_size = len(encoded.i_frame_codes) * 4

        # P-frame residuals (quantized)
        p_size = 0
        for p_data in encoded.p_frame_residuals.values():
            p_size += p_data['quantized'].nbytes + 2  # int8 array + float16 scale
        p_pos_size = len(encoded.p_frame_residuals) * 4

        return i_size + i_pos_size + p_size + p_pos_size


# =============================================================================
# EVALUATION
# =============================================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors or matrices."""
    if a.ndim == 1:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    else:
        # Mean cosine similarity per row
        sims = []
        for i in range(len(a)):
            sim = np.dot(a[i], b[i]) / (np.linalg.norm(a[i]) * np.linalg.norm(b[i]) + 1e-8)
            sims.append(sim)
        return np.mean(sims)


def reconstruction_quality(original: np.ndarray, reconstructed: np.ndarray) -> dict:
    """Measure reconstruction quality."""
    mse = np.mean((original - reconstructed) ** 2)
    cos_sim = cosine_similarity(original, reconstructed)

    # Sign accuracy (for sign-hash compatibility)
    sign_orig = (original > 0).astype(int)
    sign_recon = (reconstructed > 0).astype(int)
    sign_acc = (sign_orig == sign_recon).mean()

    return {
        'mse': mse,
        'cosine_similarity': cos_sim,
        'sign_accuracy': sign_acc
    }


# =============================================================================
# MAIN TEST
# =============================================================================

def run_test():
    print("=" * 70)
    print("EMBEDDING COMPRESSION TEST: RVQ, LPC, GOP")
    print("=" * 70)

    # Load test data
    encoder = TokenEncoder()

    texts = [
        "Kleine Hasenkinder tollen über eine schöne Wiese. Ein Atomkraftwerk explodiert und schleudert hochradioaktive Stoffe in die Luft. Auf der anderen Seite der Welt schläft ein Kind friedlich.",
        "Machine learning is transforming how we process and understand data. Neural networks can recognize images, translate languages, and generate creative content.",
        "The stock market showed unusual volatility yesterday. Major indices dropped sharply in morning trading before recovering by close.",
    ]

    print("\n1. Loading embeddings...")
    all_embeddings = []
    for i, text in enumerate(texts):
        emb = encoder.encode(text)
        all_embeddings.append(emb)
        print(f"   Text {i+1}: {emb.shape[0]} tokens, {emb.shape[1]} dims")

    # Use first text for detailed testing, all for training
    test_embeddings = all_embeddings[0]
    train_embeddings = np.vstack(all_embeddings)

    n_tokens, n_dims = test_embeddings.shape
    raw_size = test_embeddings.nbytes

    print(f"\n   Raw size: {raw_size:,} bytes ({raw_size/1024:.1f} KB)")

    results = {}

    # =========================================================================
    # Test 1: RVQ only
    # =========================================================================
    print("\n" + "=" * 70)
    print("2. RVQ (Residual Vector Quantization)")
    print("=" * 70)

    rvq = RVQEncoder(n_codes=256)
    rvq.train(train_embeddings)

    rvq_codes = rvq.encode(test_embeddings)
    rvq_decoded = rvq.decode(rvq_codes)
    rvq_size = rvq.size_bytes(rvq_codes)

    rvq_quality = reconstruction_quality(test_embeddings, rvq_decoded)

    print(f"\n   Codes shape: {rvq_codes.shape}")
    print(f"   Size: {rvq_size:,} bytes ({rvq_size/raw_size*100:.1f}% of raw)")
    print(f"   Compression: {raw_size/rvq_size:.1f}x")
    print(f"   Cosine Similarity: {rvq_quality['cosine_similarity']:.4f}")
    print(f"   Sign Accuracy: {rvq_quality['sign_accuracy']*100:.1f}%")

    results['RVQ'] = {
        'size': rvq_size,
        'compression': raw_size / rvq_size,
        **rvq_quality
    }

    # =========================================================================
    # Test 2: LPC only
    # =========================================================================
    print("\n" + "=" * 70)
    print("3. LPC (Linear Predictive Coding)")
    print("=" * 70)

    lpc = LPCEncoder(order=4, threshold_pct=90)
    lpc_encoded = lpc.encode(test_embeddings)
    lpc_decoded = lpc.decode(lpc_encoded, n_tokens, test_embeddings[0])
    lpc_size = lpc.size_bytes(lpc_encoded)

    lpc_quality = reconstruction_quality(test_embeddings, lpc_decoded)

    print(f"\n   Sparse residuals: {len(lpc_encoded.sparse_residuals)} / {n_tokens} tokens")
    print(f"   Size: {lpc_size:,} bytes ({lpc_size/raw_size*100:.1f}% of raw)")
    print(f"   Compression: {raw_size/lpc_size:.1f}x")
    print(f"   Cosine Similarity: {lpc_quality['cosine_similarity']:.4f}")
    print(f"   Sign Accuracy: {lpc_quality['sign_accuracy']*100:.1f}%")

    results['LPC'] = {
        'size': lpc_size,
        'compression': raw_size / lpc_size,
        **lpc_quality
    }

    # =========================================================================
    # Test 3: GOP only
    # =========================================================================
    print("\n" + "=" * 70)
    print("4. GOP (Group of Pictures)")
    print("=" * 70)

    gop = GOPEncoder(onset_threshold=95)
    gop_encoded = gop.encode(test_embeddings)
    gop_decoded = gop.decode(gop_encoded)
    gop_size = gop.size_bytes(gop_encoded)

    gop_quality = reconstruction_quality(test_embeddings, gop_decoded)

    n_i_frames = len(gop_encoded.i_frames)
    n_p_frames = len(gop_encoded.p_frames)

    print(f"\n   I-frames: {n_i_frames}, P-frames: {n_p_frames}")
    print(f"   I-frame positions: {sorted(gop_encoded.i_frames.keys())}")
    print(f"   Size: {gop_size:,} bytes ({gop_size/raw_size*100:.1f}% of raw)")
    print(f"   Compression: {raw_size/gop_size:.1f}x")
    print(f"   Cosine Similarity: {gop_quality['cosine_similarity']:.4f}")
    print(f"   Sign Accuracy: {gop_quality['sign_accuracy']*100:.1f}%")

    results['GOP'] = {
        'size': gop_size,
        'compression': raw_size / gop_size,
        **gop_quality
    }

    # =========================================================================
    # Test 4: Combined (RVQ + LPC + GOP)
    # =========================================================================
    print("\n" + "=" * 70)
    print("5. COMBINED (GOP + RVQ + LPC)")
    print("=" * 70)

    combined = CombinedEncoder(n_codes=256, onset_threshold=95,
                                delta_threshold_pct=50)
    combined.train(train_embeddings)

    combined_encoded = combined.encode(test_embeddings)
    combined_decoded = combined.decode(combined_encoded)
    combined_size = combined.size_bytes(combined_encoded)

    combined_quality = reconstruction_quality(test_embeddings, combined_decoded)

    n_i = len(combined_encoded.i_frame_codes)
    n_p = len(combined_encoded.p_frame_residuals)

    print(f"\n   I-frames (RVQ): {n_i}")
    print(f"   P-frames with residual: {n_p}")
    print(f"   P-frames without residual: {n_tokens - n_i - n_p}")
    print(f"   Size: {combined_size:,} bytes ({combined_size/raw_size*100:.1f}% of raw)")
    print(f"   Compression: {raw_size/combined_size:.1f}x")
    print(f"   Cosine Similarity: {combined_quality['cosine_similarity']:.4f}")
    print(f"   Sign Accuracy: {combined_quality['sign_accuracy']*100:.1f}%")

    results['Combined'] = {
        'size': combined_size,
        'compression': raw_size / combined_size,
        **combined_quality
    }

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n   {'Method':<12} {'Size':>10} {'Compress':>10} {'Cosine':>10} {'Sign Acc':>10}")
    print("   " + "-" * 54)
    print(f"   {'Raw':<12} {raw_size:>10,} {'1.0x':>10} {'1.0000':>10} {'100.0%':>10}")

    for method, data in results.items():
        print(f"   {method:<12} {data['size']:>10,} {data['compression']:>9.1f}x "
              f"{data['cosine_similarity']:>10.4f} {data['sign_accuracy']*100:>9.1f}%")

    # Best combined stats
    print("\n   Best compression: Combined")
    print(f"   {raw_size:,} bytes → {results['Combined']['size']:,} bytes")
    print(f"   {results['Combined']['compression']:.0f}x smaller, "
          f"{results['Combined']['cosine_similarity']*100:.1f}% similarity preserved")

    return results


if __name__ == "__main__":
    run_test()
