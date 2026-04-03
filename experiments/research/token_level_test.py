#!/usr/bin/env python3
"""
Token-Level vs Sliding Window Embeddings Test

Vergleicht:
1. Sliding Window (bewährt, 94.3% Recall)
2. Token-Level (neu, feinere Granularität)
"""

import numpy as np
import requests
import time
from dataclasses import dataclass
from collections import defaultdict


# =============================================================================
# DUAL-MODE ENCODER
# =============================================================================

class DualModeEncoder:
    """
    Dense Encoder mit zwei Modi:
    - pooled: /v1/embeddings → 1 Embedding pro Text (für Sliding Window)
    - token:  /embeddings    → 1 Embedding pro Token (für Signal)
    """

    def __init__(
        self,
        pooled_url: str = "http://localhost:8200",
        token_url: str = "http://localhost:8202"
    ):
        self.pooled_url = pooled_url
        self.token_url = token_url
        self.dim = None
        self._init_dim()

    def _init_dim(self):
        """Ermittle Embedding-Dimension."""
        result = self.encode_pooled(["test"])
        self.dim = len(result[0])
        print(f"Encoder dim: {self.dim}")

    def encode_pooled(self, texts: list[str], batch_size: int = 8) -> np.ndarray:
        """
        Pooled Embeddings via /v1/embeddings (Port 8200).
        Returns: np.array shape (n_texts, dim)
        """
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch = [t[:4000] for t in batch]
            response = requests.post(
                f"{self.pooled_url}/v1/embeddings",
                json={"input": batch}
            )
            response.raise_for_status()
            data = response.json()
            embeddings = [d["embedding"] for d in data["data"]]
            all_embeddings.extend(embeddings)
        return np.array(all_embeddings)

    def encode_tokens(self, text: str) -> np.ndarray:
        """
        Token-Level Embeddings via /embeddings (Port 8202, --pooling none).
        Returns: np.array shape (n_tokens, dim)
        """
        text = text[:4000]  # Limit
        response = requests.post(
            f"{self.token_url}/embeddings",
            json={"input": text}
        )
        response.raise_for_status()
        data = response.json()

        # Token-Level: embedding ist Liste von Listen
        if isinstance(data[0]["embedding"][0], list):
            embeddings = np.array(data[0]["embedding"])
        else:
            # Fallback: Pooled Format
            embeddings = np.array([data[0]["embedding"]])

        return embeddings

    def to_sign_hash(self, embeddings: np.ndarray) -> np.ndarray:
        """Konvertiere zu Sign-Hashes (1 bit pro dim)."""
        signs = (embeddings > 0).astype(np.uint8)
        n_samples, n_dims = signs.shape
        n_bytes = n_dims // 8

        packed = np.zeros((n_samples, n_bytes), dtype=np.uint8)
        for i in range(8):
            packed |= signs[:, i::8] << i

        return packed


# =============================================================================
# INDEX STRUCTURES
# =============================================================================

@dataclass
class TokenSignal:
    """Token-Level Signal für ein Dokument."""
    doc_id: str
    embeddings: np.ndarray      # (n_tokens, dim)
    sign_hashes: np.ndarray     # (n_tokens, dim//8)
    text: str
    n_tokens: int


@dataclass
class WindowSignal:
    """Sliding Window Signal für ein Dokument."""
    doc_id: str
    embeddings: np.ndarray      # (n_windows, dim)
    sign_hashes: np.ndarray     # (n_windows, dim//8)
    positions: list[int]        # Char-Position pro Window
    text: str
    n_windows: int


# =============================================================================
# INDEXING
# =============================================================================

def index_token_level(
    docs: list[dict],
    encoder: DualModeEncoder
) -> dict[str, TokenSignal]:
    """Indexiere Dokumente mit Token-Level Embeddings."""
    index = {}

    for i, doc in enumerate(docs):
        doc_id = doc["id"]
        text = doc["text"]

        # Token-Level Embeddings
        embeddings = encoder.encode_tokens(text)
        sign_hashes = encoder.to_sign_hash(embeddings)

        index[doc_id] = TokenSignal(
            doc_id=doc_id,
            embeddings=embeddings,
            sign_hashes=sign_hashes,
            text=text,
            n_tokens=len(embeddings)
        )

        if (i + 1) % 20 == 0:
            print(f"  Token-Index: {i+1}/{len(docs)}")

    return index


def index_sliding_window(
    docs: list[dict],
    encoder: DualModeEncoder,
    window_size: int = 150,
    stride: int = 30
) -> dict[str, WindowSignal]:
    """Indexiere Dokumente mit Sliding Window."""
    index = {}

    for i, doc in enumerate(docs):
        doc_id = doc["id"]
        text = doc["text"]

        # Windows erstellen
        windows = []
        positions = []
        for j in range(0, max(1, len(text) - window_size + 1), stride):
            windows.append(text[j:j+window_size])
            positions.append(j)

        if not windows:
            windows = [text]
            positions = [0]

        # Batch-Encoding
        embeddings = encoder.encode_pooled(windows)
        sign_hashes = encoder.to_sign_hash(embeddings)

        index[doc_id] = WindowSignal(
            doc_id=doc_id,
            embeddings=embeddings,
            sign_hashes=sign_hashes,
            positions=positions,
            text=text,
            n_windows=len(windows)
        )

        if (i + 1) % 20 == 0:
            print(f"  Window-Index: {i+1}/{len(docs)}")

    return index


# =============================================================================
# SEARCH
# =============================================================================

def sign_hamming_similarity(hash1: np.ndarray, hash2: np.ndarray) -> float:
    """Hamming Similarity zwischen Sign-Hashes."""
    xor = np.bitwise_xor(hash1, hash2)
    diff_bits = np.unpackbits(xor).sum()
    total_bits = len(hash1) * 8
    return 1.0 - (diff_bits / total_bits)


def search_token_index(
    query: str,
    index: dict[str, TokenSignal],
    encoder: DualModeEncoder,
    top_k: int = 10
) -> list[tuple[str, float, int]]:
    """
    Suche in Token-Level Index.
    Returns: [(doc_id, score, best_token_pos), ...]
    """
    # Query als pooled (repräsentiert Gesamt-Semantik)
    query_emb = encoder.encode_pooled([query])[0]
    query_sign = encoder.to_sign_hash(query_emb.reshape(1, -1))[0]

    results = []

    for doc_id, signal in index.items():
        # Finde beste Token-Position via Sign-Hash
        best_score = 0.0
        best_pos = 0

        for pos, doc_sign in enumerate(signal.sign_hashes):
            sim = sign_hamming_similarity(query_sign, doc_sign)
            if sim > best_score:
                best_score = sim
                best_pos = pos

        results.append((doc_id, best_score, best_pos))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def search_window_index(
    query: str,
    index: dict[str, WindowSignal],
    encoder: DualModeEncoder,
    top_k: int = 10
) -> list[tuple[str, float, int]]:
    """
    Suche in Sliding Window Index.
    Returns: [(doc_id, score, best_char_pos), ...]
    """
    query_emb = encoder.encode_pooled([query])[0]
    query_sign = encoder.to_sign_hash(query_emb.reshape(1, -1))[0]

    results = []

    for doc_id, signal in index.items():
        best_score = 0.0
        best_pos = 0

        for i, doc_sign in enumerate(signal.sign_hashes):
            sim = sign_hamming_similarity(query_sign, doc_sign)
            if sim > best_score:
                best_score = sim
                best_pos = signal.positions[i]

        results.append((doc_id, best_score, best_pos))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_retrieval(
    queries: list[dict],
    relevance: dict,
    search_fn,
    index,
    encoder: DualModeEncoder,
    k: int = 10
) -> tuple[float, float, float]:
    """
    Evaluiere Retrieval.
    Returns: (recall@k, avg_time_ms, avg_positions_per_doc)
    """
    hits = 0
    total = 0
    total_time = 0
    total_positions = 0
    n_docs = 0

    for query in queries:
        query_id = query["id"]
        if query_id not in relevance:
            continue

        relevant_docs = set(relevance[query_id])

        start = time.time()
        results = search_fn(query["text"], index, encoder, top_k=k)
        total_time += time.time() - start

        found_docs = set(r[0] for r in results)
        if relevant_docs & found_docs:
            hits += 1
        total += 1

    # Zähle Positionen
    for signal in index.values():
        if hasattr(signal, 'n_tokens'):
            total_positions += signal.n_tokens
        else:
            total_positions += signal.n_windows
        n_docs += 1

    recall = hits / total if total > 0 else 0
    avg_time = (total_time / total * 1000) if total > 0 else 0
    avg_positions = total_positions / n_docs if n_docs > 0 else 0

    return recall, avg_time, avg_positions


# =============================================================================
# MAIN
# =============================================================================

def load_scifact(n_docs: int = 100, n_queries: int = 50):
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
    print("TOKEN-LEVEL vs SLIDING WINDOW COMPARISON")
    print("=" * 70)

    # Encoder
    print("\n1. Initialisiere Encoder...")
    encoder = DualModeEncoder()

    # Test Token-Level Encoding
    print("\n2. Test Token-Level Encoding...")
    test_text = "Hello, world! This is a test."
    token_embs = encoder.encode_tokens(test_text)
    print(f"   Text: '{test_text}'")
    print(f"   Tokens: {len(token_embs)}")
    print(f"   Shape: {token_embs.shape}")

    # Daten laden
    print("\n3. Lade SciFact...")
    n_docs = 100
    n_queries = 50
    docs, queries, relevance = load_scifact(n_docs, n_queries)
    print(f"   {len(docs)} Dokumente, {len(queries)} Queries")

    # Token-Level Index
    print("\n4. Baue Token-Level Index...")
    start = time.time()
    token_index = index_token_level(docs, encoder)
    token_index_time = time.time() - start
    print(f"   Zeit: {token_index_time:.1f}s")

    # Window Index
    print("\n5. Baue Sliding Window Index...")
    start = time.time()
    window_index = index_sliding_window(docs, encoder)
    window_index_time = time.time() - start
    print(f"   Zeit: {window_index_time:.1f}s")

    # Statistiken
    print("\n" + "=" * 70)
    print("INDEX STATISTIKEN")
    print("=" * 70)

    token_total = sum(s.n_tokens for s in token_index.values())
    window_total = sum(s.n_windows for s in window_index.values())

    token_bytes = sum(s.sign_hashes.nbytes for s in token_index.values())
    window_bytes = sum(s.sign_hashes.nbytes for s in window_index.values())

    print(f"\n   Token-Level:")
    print(f"     Positionen gesamt: {token_total}")
    print(f"     Durchschnitt/Doc:  {token_total/len(docs):.1f}")
    print(f"     Sign-Hash Größe:   {token_bytes/1024:.1f} KB")
    print(f"     Index-Zeit:        {token_index_time:.1f}s")

    print(f"\n   Sliding Window:")
    print(f"     Positionen gesamt: {window_total}")
    print(f"     Durchschnitt/Doc:  {window_total/len(docs):.1f}")
    print(f"     Sign-Hash Größe:   {window_bytes/1024:.1f} KB")
    print(f"     Index-Zeit:        {window_index_time:.1f}s")

    print(f"\n   Verhältnis Token/Window: {token_total/window_total:.1f}x")

    # Evaluation
    print("\n" + "=" * 70)
    print("RETRIEVAL EVALUATION")
    print("=" * 70)

    print("\n   Token-Level Search...")
    token_recall, token_time, token_avg = evaluate_retrieval(
        queries, relevance, search_token_index, token_index, encoder
    )

    print("   Sliding Window Search...")
    window_recall, window_time, window_avg = evaluate_retrieval(
        queries, relevance, search_window_index, window_index, encoder
    )

    print(f"\n   {'Methode':<20} {'R@10':>8} {'Zeit/Query':>12} {'Pos/Doc':>10}")
    print(f"   {'-'*50}")
    print(f"   {'Token-Level':<20} {token_recall*100:>7.1f}% {token_time:>10.2f}ms {token_avg:>10.1f}")
    print(f"   {'Sliding Window':<20} {window_recall*100:>7.1f}% {window_time:>10.2f}ms {window_avg:>10.1f}")

    # Demo
    print("\n" + "=" * 70)
    print("DEMO SEARCH")
    print("=" * 70)

    demo_query = "gene therapy for cancer treatment"
    print(f"\n   Query: '{demo_query}'")

    print("\n   Token-Level Results:")
    token_results = search_token_index(demo_query, token_index, encoder, top_k=3)
    for doc_id, score, pos in token_results:
        print(f"     {doc_id}: score={score:.3f}, token_pos={pos}")

    print("\n   Window Results:")
    window_results = search_window_index(demo_query, window_index, encoder, top_k=3)
    for doc_id, score, pos in window_results:
        snippet = window_index[doc_id].text[pos:pos+80].replace("\n", " ")
        print(f"     {doc_id}: score={score:.3f}, char_pos={pos}")
        print(f"       \"{snippet}...\"")


if __name__ == "__main__":
    main()
