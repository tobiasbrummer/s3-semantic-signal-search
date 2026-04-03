#!/usr/bin/env python3
"""
SPLADE Integration Test

SPLADE erzeugt sparse lexikalische Embeddings:
- ~30k Dimensionen (Vocabulary)
- Nur ~100-300 aktiv pro Text
- Werte = Term-Gewichte (wie wichtig ist dieser Term?)

Wir speichern nur die Top-K Peaks: (term_id, gain)
Das passt zur Audio-Metapher: Sparse Peaks wie Onsets/Transients.
"""

import torch
import numpy as np
from transformers import AutoModelForMaskedLM, AutoTokenizer
from dataclasses import dataclass
from typing import Optional
import time


@dataclass
class SpladePeaks:
    """Sparse SPLADE Repräsentation als Peaks."""
    # Term IDs (vocabulary indices)
    term_ids: np.ndarray
    # Gewichte (wie stark ist der Term?)
    weights: np.ndarray
    # Original Text (für Debugging)
    text: str = ""

    @property
    def n_peaks(self) -> int:
        return len(self.term_ids)

    def top_k(self, k: int) -> "SpladePeaks":
        """Reduziere auf Top-K Peaks."""
        if k >= self.n_peaks:
            return self
        idx = np.argsort(self.weights)[-k:]
        return SpladePeaks(
            term_ids=self.term_ids[idx],
            weights=self.weights[idx],
            text=self.text
        )

    def to_dense(self, vocab_size: int = 30522) -> np.ndarray:
        """Konvertiere zurück zu Dense (für Kompatibilitätstests)."""
        dense = np.zeros(vocab_size)
        dense[self.term_ids] = self.weights
        return dense


class SpladeEncoder:
    """SPLADE Encoder mit Top-K Peak Extraktion."""

    def __init__(
        self,
        model_name: str = "naver/splade-cocondenser-ensembledistil",
        device: Optional[str] = None,
        top_k: int = 256
    ):
        self.model_name = model_name
        self.top_k = top_k

        # Device detection
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device

        print(f"Lade SPLADE Modell: {model_name}")
        print(f"Device: {device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

        self.vocab_size = self.tokenizer.vocab_size
        print(f"Vocabulary Size: {self.vocab_size}")

    def encode_text(self, text: str) -> SpladePeaks:
        """Encode einen Text zu SPLADE Peaks."""
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # (batch, seq_len, vocab_size)

        # SPLADE aggregation: max over sequence, then ReLU + log1p
        # Formel: max_j(ReLU(log(1 + exp(logits_j))))
        weights = torch.max(
            torch.log1p(torch.relu(logits)),
            dim=1
        ).values  # (batch, vocab_size)

        # Zu numpy
        weights = weights.squeeze(0).cpu().numpy()  # (vocab_size,)

        # Finde non-zero entries
        nonzero_idx = np.nonzero(weights)[0]
        nonzero_weights = weights[nonzero_idx]

        peaks = SpladePeaks(
            term_ids=nonzero_idx,
            weights=nonzero_weights,
            text=text
        )

        # Top-K
        return peaks.top_k(self.top_k)

    def encode_batch(self, texts: list[str], batch_size: int = 8) -> list[SpladePeaks]:
        """Encode mehrere Texte."""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            for text in batch:
                results.append(self.encode_text(text))
        return results

    def decode_terms(self, peaks: SpladePeaks, top_n: int = 20) -> list[tuple[str, float]]:
        """Decode Term IDs zurück zu Wörtern (für Debugging)."""
        # Sortiere nach Gewicht
        idx = np.argsort(peaks.weights)[-top_n:][::-1]

        terms = []
        for i in idx:
            term_id = peaks.term_ids[i]
            weight = peaks.weights[i]
            token = self.tokenizer.decode([term_id])
            terms.append((token, weight))

        return terms


def sparse_dot_product(peaks1: SpladePeaks, peaks2: SpladePeaks) -> float:
    """Berechne Dot Product zwischen zwei Sparse Vektoren."""
    # Finde gemeinsame Term IDs
    common = np.intersect1d(peaks1.term_ids, peaks2.term_ids)

    if len(common) == 0:
        return 0.0

    # Lookup Gewichte
    score = 0.0
    for term_id in common:
        w1 = peaks1.weights[peaks1.term_ids == term_id][0]
        w2 = peaks2.weights[peaks2.term_ids == term_id][0]
        score += w1 * w2

    return score


def sparse_jaccard(peaks1: SpladePeaks, peaks2: SpladePeaks) -> float:
    """Jaccard-ähnliche Similarity für Sparse Vektoren."""
    set1 = set(peaks1.term_ids)
    set2 = set(peaks2.term_ids)

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union if union > 0 else 0.0


def main():
    print("=" * 70)
    print("SPLADE INTEGRATION TEST")
    print("=" * 70)

    # Encoder laden
    print("\n1. Lade SPLADE Encoder...")
    encoder = SpladeEncoder(top_k=256)

    # Test-Texte
    print("\n2. Teste Encoding...")
    test_texts = [
        "Deep learning is a subset of machine learning.",
        "Neural networks can learn complex patterns from data.",
        "The capital of France is Paris.",
        "Machine learning algorithms require large datasets.",
    ]

    for text in test_texts:
        start = time.time()
        peaks = encoder.encode_text(text)
        elapsed = (time.time() - start) * 1000

        print(f"\n   Text: \"{text[:50]}...\"")
        print(f"   Peaks: {peaks.n_peaks}, Zeit: {elapsed:.1f}ms")

        # Top Terms anzeigen
        top_terms = encoder.decode_terms(peaks, top_n=10)
        terms_str = ", ".join([f"{t}({w:.2f})" for t, w in top_terms])
        print(f"   Top Terms: {terms_str}")

    # Similarity Test
    print("\n3. Teste Similarity...")

    query = "machine learning neural networks"
    query_peaks = encoder.encode_text(query)

    print(f"\n   Query: \"{query}\"")
    print(f"   Query Peaks: {query_peaks.n_peaks}")

    for text in test_texts:
        doc_peaks = encoder.encode_text(text)

        dot = sparse_dot_product(query_peaks, doc_peaks)
        jaccard = sparse_jaccard(query_peaks, doc_peaks)

        print(f"\n   Doc: \"{text[:40]}...\"")
        print(f"   Dot Product: {dot:.2f}, Jaccard: {jaccard:.3f}")

    # Speichervergleich
    print("\n4. Speichervergleich...")

    dense_size = encoder.vocab_size * 4  # float32
    sparse_size = 256 * (4 + 4)  # term_id (int32) + weight (float32)

    print(f"   Dense: {dense_size:,} bytes ({dense_size/1024:.1f} KB)")
    print(f"   Sparse (256 peaks): {sparse_size:,} bytes ({sparse_size/1024:.1f} KB)")
    print(f"   Kompression: {dense_size/sparse_size:.1f}x")

    print("\n" + "=" * 70)
    print("SPLADE READY")
    print("=" * 70)


if __name__ == "__main__":
    main()
