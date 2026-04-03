#!/usr/bin/env python3
"""
Hybrid Search Pipeline: SPLADE-First + Sign-Hash Validation

Architektur:
1. SPLADE Inverted Index → Kandidaten-Positionen (O(1) lookup)
2. Sign-Hash Matching → Schnelle Validierung (32x komprimiert)
3. Optional: Dense Cosine → Präzises Re-Ranking

Das ist VIEL schneller als Brute-Force, weil wir nur an
SPLADE-Hit-Positionen den Dense-Check machen.
"""

import numpy as np
import requests
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict
import time
import pickle


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SignalIndex:
    """
    Der Hybrid-Index für ein Dokument-Korpus.

    Speichert:
    - SPLADE Inverted Index (term_id → [(doc_id, position, weight)])
    - Sign-Hashes pro Position (1 bit pro Dimension)
    - Optional: Dense Vectors für Re-Ranking
    """
    # Inverted Index: term_id → list of (doc_id, position, weight)
    inverted_index: dict = field(default_factory=lambda: defaultdict(list))

    # Sign-Hashes: doc_id → np.array of shape (n_positions, n_dims // 8)
    sign_hashes: dict = field(default_factory=dict)

    # Dense Vectors (optional): doc_id → np.array of shape (n_positions, n_dims)
    dense_vectors: dict = field(default_factory=dict)

    # Dokument-Texte für Snippet-Extraktion
    doc_texts: dict = field(default_factory=dict)

    # Metadata
    n_docs: int = 0
    n_dims: int = 1024

    def save(self, path: str):
        """Speichere Index auf Disk."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "SignalIndex":
        """Lade Index von Disk."""
        with open(path, 'rb') as f:
            return pickle.load(f)


@dataclass
class SearchResult:
    """Ein Suchergebnis."""
    doc_id: str
    position: int
    score: float
    snippet: str = ""

    def __repr__(self):
        return f"SearchResult(doc={self.doc_id}, pos={self.position}, score={self.score:.3f})"


# =============================================================================
# ENCODERS
# =============================================================================

class DenseEncoder:
    """Dense Embeddings via llama.cpp API."""

    def __init__(self, base_url: str = "http://localhost:8200"):
        self.base_url = base_url
        self.dim = None
        self._init_dim()

    def _init_dim(self):
        test = self.encode(["test"])
        self.dim = len(test[0])

    def encode(self, texts: list[str], batch_size: int = 8) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch = [t[:4000] for t in batch]
            response = requests.post(
                f"{self.base_url}/v1/embeddings",
                json={"input": batch}
            )
            response.raise_for_status()
            data = response.json()
            embeddings = [d["embedding"] for d in data["data"]]
            all_embeddings.extend(embeddings)
        return np.array(all_embeddings)

    def to_sign_hash(self, embeddings: np.ndarray) -> np.ndarray:
        """Konvertiere Dense Embeddings zu Sign-Hashes (1 bit pro dim)."""
        # Sign: positive = 1, negative = 0
        signs = (embeddings > 0).astype(np.uint8)

        # Pack 8 bits into 1 byte
        n_samples, n_dims = signs.shape
        n_bytes = n_dims // 8

        packed = np.zeros((n_samples, n_bytes), dtype=np.uint8)
        for i in range(8):
            packed |= signs[:, i::8] << i

        return packed


class SpladeEncoder:
    """SPLADE Encoder für Sparse Lexical Embeddings."""

    def __init__(
        self,
        model_name: str = "naver/splade-cocondenser-ensembledistil",
        device: Optional[str] = None,
        top_k: int = 64
    ):
        self.top_k = top_k

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        self.vocab_size = self.tokenizer.vocab_size

    def encode(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Encode Text zu SPLADE Peaks.
        Returns: (term_ids, weights)
        """
        inputs = self.tokenizer(
            text[:4000],
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        weights = torch.max(
            torch.log1p(torch.relu(logits)),
            dim=1
        ).values.squeeze(0).cpu().numpy()

        # Non-zero entries
        nonzero_idx = np.nonzero(weights)[0]
        nonzero_weights = weights[nonzero_idx]

        # Top-K
        if len(nonzero_idx) > self.top_k:
            top_idx = np.argsort(nonzero_weights)[-self.top_k:]
            nonzero_idx = nonzero_idx[top_idx]
            nonzero_weights = nonzero_weights[top_idx]

        return nonzero_idx, nonzero_weights


# =============================================================================
# INDEXING
# =============================================================================

def build_index(
    docs: list[dict],
    dense_encoder: DenseEncoder,
    splade_encoder: SpladeEncoder,
    window_size: int = 150,
    stride: int = 30,
    store_dense: bool = False
) -> SignalIndex:
    """
    Baue den Hybrid-Index für eine Liste von Dokumenten.

    Args:
        docs: Liste von {"id": str, "text": str}
        dense_encoder: Encoder für Dense Embeddings
        splade_encoder: Encoder für SPLADE
        window_size: Fenstergröße für Sliding Window
        stride: Schrittweite
        store_dense: Ob Dense Vectors gespeichert werden sollen (für Re-Ranking)
    """
    index = SignalIndex(n_dims=dense_encoder.dim)

    for doc_idx, doc in enumerate(docs):
        doc_id = doc["id"]
        text = doc["text"]

        # Speichere Text für Snippets
        index.doc_texts[doc_id] = text

        # Sliding Windows erstellen
        windows = []
        positions = []
        for i in range(0, max(1, len(text) - window_size + 1), stride):
            windows.append(text[i:i+window_size])
            positions.append(i)

        if not windows:
            windows = [text]
            positions = [0]

        # Dense Embeddings für alle Windows
        dense_embs = dense_encoder.encode(windows)

        # Sign-Hashes speichern
        sign_hashes = dense_encoder.to_sign_hash(dense_embs)
        index.sign_hashes[doc_id] = sign_hashes

        # Optional: Dense Vectors speichern
        if store_dense:
            index.dense_vectors[doc_id] = dense_embs

        # SPLADE für jedes Window → Inverted Index
        for pos_idx, (window_text, char_pos) in enumerate(zip(windows, positions)):
            term_ids, weights = splade_encoder.encode(window_text)

            for term_id, weight in zip(term_ids, weights):
                index.inverted_index[int(term_id)].append(
                    (doc_id, pos_idx, char_pos, float(weight))
                )

        if (doc_idx + 1) % 50 == 0:
            print(f"   Indexed {doc_idx + 1}/{len(docs)} docs")

    index.n_docs = len(docs)
    return index


# =============================================================================
# SEARCH
# =============================================================================

def sign_hamming_similarity(hash1: np.ndarray, hash2: np.ndarray) -> float:
    """
    Berechne Similarity zwischen zwei Sign-Hashes.
    Returns: Anteil der übereinstimmenden Bits (0.0 - 1.0)
    """
    # XOR → unterschiedliche Bits sind 1
    xor = np.bitwise_xor(hash1, hash2)

    # Popcount → Anzahl unterschiedlicher Bits
    diff_bits = np.unpackbits(xor).sum()

    # Similarity = 1 - (diff / total)
    total_bits = len(hash1) * 8
    return 1.0 - (diff_bits / total_bits)


def search(
    query: str,
    index: SignalIndex,
    dense_encoder: DenseEncoder,
    splade_encoder: SpladeEncoder,
    top_k: int = 10,
    splade_candidates: int = 100,
    sign_threshold: float = 0.6
) -> list[SearchResult]:
    """
    SPLADE-First Hybrid Search.

    1. SPLADE Query → Term IDs
    2. Inverted Index Lookup → Kandidaten-Positionen
    3. Sign-Hash Filtering → Schnelle Validierung
    4. Ranking → Top-K Results
    """

    # Step 1: Query encodieren
    query_dense = dense_encoder.encode([query])[0]
    query_sign = dense_encoder.to_sign_hash(query_dense.reshape(1, -1))[0]
    query_terms, query_weights = splade_encoder.encode(query)

    # Step 2: SPLADE Kandidaten sammeln
    # Für jeden Query-Term: Hole alle Dokument-Positionen aus dem Index
    candidates = defaultdict(float)  # (doc_id, pos_idx) → score

    for term_id, term_weight in zip(query_terms, query_weights):
        if term_id in index.inverted_index:
            for doc_id, pos_idx, char_pos, doc_weight in index.inverted_index[term_id]:
                # Akkumuliere SPLADE Score
                candidates[(doc_id, pos_idx, char_pos)] += term_weight * doc_weight

    # Step 3: Top-N SPLADE Kandidaten nach Score
    sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    top_candidates = sorted_candidates[:splade_candidates]

    # Step 4: Sign-Hash Filtering
    results = []
    for (doc_id, pos_idx, char_pos), splade_score in top_candidates:
        # Hole Sign-Hash für diese Position
        doc_sign = index.sign_hashes[doc_id][pos_idx]

        # Sign Similarity
        sign_sim = sign_hamming_similarity(query_sign, doc_sign)

        if sign_sim < sign_threshold:
            continue

        # Kombinierter Score
        combined_score = 0.3 * sign_sim + 0.7 * (splade_score / 50)  # Normalisierung

        # Snippet extrahieren
        text = index.doc_texts[doc_id]
        snippet = text[char_pos:char_pos + 100].replace("\n", " ")

        results.append(SearchResult(
            doc_id=doc_id,
            position=char_pos,
            score=combined_score,
            snippet=snippet
        ))

    # Step 5: Final Ranking
    results.sort(key=lambda x: x.score, reverse=True)
    return results[:top_k]


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_search(
    queries: list[dict],
    relevance: dict,
    index: SignalIndex,
    dense_encoder: DenseEncoder,
    splade_encoder: SpladeEncoder,
    k: int = 10
) -> tuple[float, float]:
    """Evaluiere Recall@K und durchschnittliche Suchzeit."""

    hits = 0
    total = 0
    total_time = 0

    for query in queries:
        query_id = query["id"]
        if query_id not in relevance:
            continue

        relevant_docs = set(relevance[query_id])

        start = time.time()
        results = search(
            query["text"], index, dense_encoder, splade_encoder, top_k=k
        )
        total_time += time.time() - start

        # Check ob ein relevantes Dokument in Top-K
        found_docs = set(r.doc_id for r in results)
        if relevant_docs & found_docs:
            hits += 1
        total += 1

    recall = hits / total if total > 0 else 0
    avg_time = total_time / total if total > 0 else 0

    return recall, avg_time


# =============================================================================
# MAIN
# =============================================================================

def load_scifact(n_docs: int = 500, n_queries: int = 100):
    """Lade SciFact Testdaten."""
    from datasets import load_dataset

    corpus = load_dataset("mteb/scifact", "corpus", split="corpus")
    queries_ds = load_dataset("mteb/scifact", "queries", split="queries")
    qrels_ds = load_dataset("mteb/scifact", "default", split="test")

    qrels = defaultdict(set)
    for item in qrels_ds:
        qrels[item["query-id"]].add(item["corpus-id"])

    docs = []
    doc_id_set = set()
    for i, item in enumerate(corpus):
        if i >= n_docs:
            break
        doc_id = item["_id"]
        docs.append({
            "id": doc_id,
            "text": f"{item['title']} {item['text']}"
        })
        doc_id_set.add(doc_id)

    queries = []
    relevance = {}
    for item in queries_ds:
        query_id = item["_id"]
        if query_id in qrels:
            relevant = qrels[query_id] & doc_id_set
            if relevant:
                queries.append({
                    "id": query_id,
                    "text": item["text"]
                })
                relevance[query_id] = list(relevant)
        if len(queries) >= n_queries:
            break

    return docs, queries, relevance


def main():
    print("=" * 70)
    print("HYBRID SEARCH: SPLADE-First + Sign-Hash")
    print("=" * 70)

    # Encoder laden
    print("\n1. Lade Encoder...")
    dense_encoder = DenseEncoder()
    print(f"   Dense dim: {dense_encoder.dim}")

    splade_encoder = SpladeEncoder(top_k=64)
    print(f"   SPLADE vocab: {splade_encoder.vocab_size}")

    # Daten laden
    print("\n2. Lade SciFact...")
    n_docs = 500
    n_queries = 100
    docs, queries, relevance = load_scifact(n_docs, n_queries)
    print(f"   {len(docs)} Dokumente, {len(queries)} Queries")

    # Index bauen
    print("\n3. Baue Index...")
    start = time.time()
    index = build_index(docs, dense_encoder, splade_encoder)
    index_time = time.time() - start
    print(f"   Index gebaut in {index_time:.1f}s")

    # Speicherverbrauch schätzen
    sign_bytes = sum(h.nbytes for h in index.sign_hashes.values())
    inverted_entries = sum(len(v) for v in index.inverted_index.values())
    print(f"   Sign-Hashes: {sign_bytes / 1024:.1f} KB")
    print(f"   Inverted Index: {inverted_entries} Einträge")

    # Search evaluieren
    print("\n4. Evaluiere Search...")
    print("=" * 70)

    recall, avg_time = evaluate_search(
        queries, relevance, index, dense_encoder, splade_encoder, k=10
    )
    print(f"   Recall@10: {recall*100:.1f}%")
    print(f"   Avg Time: {avg_time*1000:.2f}ms pro Query")

    # Vergleich mit Brute-Force (aus vorherigen Tests)
    print("\n" + "=" * 70)
    print("VERGLEICH")
    print("=" * 70)
    print(f"   Hybrid (SPLADE-First + Sign): {recall*100:.1f}%")
    print(f"   Brute-Force Combined (0.3/0.7): 88.5% (aus früherem Test)")
    print(f"   Speedup: TBD nach Brute-Force Timing")

    # Demo-Suche
    print("\n" + "=" * 70)
    print("DEMO SEARCH")
    print("=" * 70)

    demo_query = "gene therapy cancer treatment"
    print(f"\n   Query: \"{demo_query}\"")

    results = search(demo_query, index, dense_encoder, splade_encoder, top_k=5)

    for i, r in enumerate(results):
        print(f"\n   {i+1}. {r.doc_id} (pos={r.position}, score={r.score:.3f})")
        print(f"      \"{r.snippet}...\"")


if __name__ == "__main__":
    main()
