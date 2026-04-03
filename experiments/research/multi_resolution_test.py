#!/usr/bin/env python3
"""
Multi-Resolution Search Test

Hypothese:
1. Token-Level als Ground Truth speichern
2. On-the-fly zu verschiedenen Resolutions aggregieren
3. Hierarchische Suche: Document → Paragraph → Token

Erwartung: 100x Speedup bei gleichem Recall
"""

import numpy as np
import requests
import time
from dataclasses import dataclass
from collections import defaultdict


# =============================================================================
# ENCODER
# =============================================================================

class TokenEncoder:
    def __init__(self, url: str = "http://localhost:8202"):
        self.url = url
        self.dim = 1024

    def encode(self, text: str) -> np.ndarray:
        text = text[:4000]
        response = requests.post(f"{self.url}/embeddings", json={"input": text})
        response.raise_for_status()
        data = response.json()
        if isinstance(data[0]["embedding"][0], list):
            return np.array(data[0]["embedding"])
        return np.array([data[0]["embedding"]])


class PooledEncoder:
    def __init__(self, url: str = "http://localhost:8200"):
        self.url = url

    def encode(self, text: str) -> np.ndarray:
        response = requests.post(
            f"{self.url}/v1/embeddings",
            json={"input": [text[:4000]]}
        )
        response.raise_for_status()
        data = response.json()
        return np.array(data["data"][0]["embedding"])


# =============================================================================
# MULTI-RESOLUTION INDEX
# =============================================================================

@dataclass
class MultiResDoc:
    """Ein Dokument mit Multi-Resolution Embeddings."""
    doc_id: str
    text: str

    # Token-Level (Ground Truth)
    token_embeddings: np.ndarray  # (n_tokens, dim)

    # Aggregierte Levels (on-the-fly oder gecached)
    doc_embedding_mean: np.ndarray = None    # (dim,) - Mean über alle Tokens
    doc_embedding_max: np.ndarray = None     # (dim,) - Max über alle Tokens
    paragraph_embeddings: np.ndarray = None  # (n_paragraphs, dim)

    n_tokens: int = 0
    paragraph_size: int = 20  # Tokens pro Paragraph


def aggregate_mean(embeddings: np.ndarray) -> np.ndarray:
    """Aggregiere Embeddings via Mean."""
    return embeddings.mean(axis=0)


def aggregate_max(embeddings: np.ndarray) -> np.ndarray:
    """Aggregiere Embeddings via Max (behält stärkste Signale)."""
    return embeddings.max(axis=0)


def aggregate_mean_max(embeddings: np.ndarray) -> np.ndarray:
    """Kombiniere Mean und Max (wie in manchen Sentence Transformers)."""
    mean_emb = embeddings.mean(axis=0)
    max_emb = embeddings.max(axis=0)
    # Konkatenieren und normalisieren
    combined = np.concatenate([mean_emb, max_emb])
    return combined / np.linalg.norm(combined)


def aggregate_to_paragraphs(token_embs: np.ndarray, para_size: int = 20) -> np.ndarray:
    """Aggregiere Tokens zu Paragraphen."""
    n_tokens = len(token_embs)
    n_paras = max(1, n_tokens // para_size)

    paragraphs = []
    for i in range(n_paras):
        start = i * para_size
        end = min(start + para_size, n_tokens)
        para_emb = token_embs[start:end].mean(axis=0)
        paragraphs.append(para_emb)

    # Rest-Tokens zum letzten Paragraph
    if n_tokens % para_size != 0 and n_paras > 0:
        pass  # Schon im letzten Paragraph enthalten

    return np.array(paragraphs)


def build_multi_res_index(
    docs: list[dict],
    encoder: TokenEncoder,
    paragraph_size: int = 20
) -> dict[str, MultiResDoc]:
    """Baue Multi-Resolution Index."""
    index = {}

    for i, doc in enumerate(docs):
        doc_id = doc["id"]
        text = doc["text"]

        # Token-Level
        token_embs = encoder.encode(text)

        # Aggregationen
        doc_emb_mean = aggregate_mean(token_embs)
        doc_emb_max = aggregate_max(token_embs)
        para_embs = aggregate_to_paragraphs(token_embs, paragraph_size)

        index[doc_id] = MultiResDoc(
            doc_id=doc_id,
            text=text,
            token_embeddings=token_embs,
            doc_embedding_mean=doc_emb_mean,
            doc_embedding_max=doc_emb_max,
            paragraph_embeddings=para_embs,
            n_tokens=len(token_embs),
            paragraph_size=paragraph_size
        )

        if (i + 1) % 50 == 0:
            print(f"  Indexed {i+1}/{len(docs)}")

    return index


# =============================================================================
# SEARCH STRATEGIES
# =============================================================================

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine Similarity."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)


def search_brute_force_tokens(
    query_emb: np.ndarray,
    index: dict[str, MultiResDoc],
    top_k: int = 10
) -> list[tuple[str, float, int]]:
    """
    Brute Force: Vergleiche Query mit ALLEN Tokens.
    Returns: [(doc_id, score, token_pos), ...]
    """
    results = []

    for doc_id, doc in index.items():
        best_score = 0
        best_pos = 0

        for t, token_emb in enumerate(doc.token_embeddings):
            score = cosine_sim(query_emb, token_emb)
            if score > best_score:
                best_score = score
                best_pos = t

        results.append((doc_id, best_score, best_pos))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def search_hierarchical(
    query_emb: np.ndarray,
    index: dict[str, MultiResDoc],
    top_k: int = 10,
    level1_candidates: int = 50,
    level2_candidates: int = 20,
    use_max: bool = False
) -> list[tuple[str, float, int]]:
    """
    Hierarchische Suche: Document → Paragraph → Token
    """
    # Level 1: Document-Level (schnell)
    doc_scores = []
    for doc_id, doc in index.items():
        if use_max:
            score = cosine_sim(query_emb, doc.doc_embedding_max)
        else:
            score = cosine_sim(query_emb, doc.doc_embedding_mean)
        doc_scores.append((doc_id, score))

    doc_scores.sort(key=lambda x: x[1], reverse=True)
    level1_docs = [d[0] for d in doc_scores[:level1_candidates]]

    # Level 2: Paragraph-Level
    para_candidates = []
    for doc_id in level1_docs:
        doc = index[doc_id]
        for p, para_emb in enumerate(doc.paragraph_embeddings):
            score = cosine_sim(query_emb, para_emb)
            para_candidates.append((doc_id, p, score))

    para_candidates.sort(key=lambda x: x[2], reverse=True)
    level2_paras = para_candidates[:level2_candidates]

    # Level 3: Token-Level (nur für Top-Paragraphen)
    results = []
    seen_docs = set()

    for doc_id, para_idx, _ in level2_paras:
        if doc_id in seen_docs:
            continue

        doc = index[doc_id]
        para_size = doc.paragraph_size

        # Token-Range für diesen Paragraph
        start = para_idx * para_size
        end = min(start + para_size, doc.n_tokens)

        best_score = 0
        best_pos = start

        for t in range(start, end):
            score = cosine_sim(query_emb, doc.token_embeddings[t])
            if score > best_score:
                best_score = score
                best_pos = t

        results.append((doc_id, best_score, best_pos))
        seen_docs.add(doc_id)

        if len(results) >= top_k:
            break

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def search_doc_level_mean(
    query_emb: np.ndarray,
    index: dict[str, MultiResDoc],
    top_k: int = 10
) -> list[tuple[str, float, int]]:
    """Document-Level mit Mean-Pooling."""
    results = []

    for doc_id, doc in index.items():
        score = cosine_sim(query_emb, doc.doc_embedding_mean)
        results.append((doc_id, score, 0))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def search_doc_level_max(
    query_emb: np.ndarray,
    index: dict[str, MultiResDoc],
    top_k: int = 10
) -> list[tuple[str, float, int]]:
    """Document-Level mit Max-Pooling."""
    results = []

    for doc_id, doc in index.items():
        score = cosine_sim(query_emb, doc.doc_embedding_max)
        results.append((doc_id, score, 0))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


# =============================================================================
# POOLED BASELINE (Standard RAG)
# =============================================================================

def build_pooled_index(
    docs: list[dict],
    pooled_encoder: PooledEncoder
) -> dict[str, np.ndarray]:
    """Baue Standard Pooled Index (wie normale RAG)."""
    index = {}
    for i, doc in enumerate(docs):
        doc_id = doc["id"]
        emb = pooled_encoder.encode(doc["text"])
        index[doc_id] = emb
        if (i + 1) % 50 == 0:
            print(f"  Pooled Index: {i+1}/{len(docs)}")
    return index


def search_pooled(
    query_emb: np.ndarray,
    pooled_index: dict[str, np.ndarray],
    top_k: int = 10
) -> list[tuple[str, float, int]]:
    """Standard Pooled Search (normale RAG)."""
    results = []
    for doc_id, doc_emb in pooled_index.items():
        score = cosine_sim(query_emb, doc_emb)
        results.append((doc_id, score, 0))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_search(
    queries: list[dict],
    relevance: dict,
    search_fn,
    index: dict[str, MultiResDoc],
    pooled_encoder: PooledEncoder,
    top_k: int = 10,
    **search_kwargs
) -> tuple[float, float, int]:
    """
    Returns: (recall@k, avg_time_ms, comparisons)
    """
    hits = 0
    total = 0
    total_time = 0

    for query in queries:
        if query["id"] not in relevance:
            continue

        relevant = set(relevance[query["id"]])
        query_emb = pooled_encoder.encode(query["text"])

        start = time.time()
        results = search_fn(query_emb, index, top_k=top_k, **search_kwargs)
        total_time += time.time() - start

        found = set(r[0] for r in results)
        if relevant & found:
            hits += 1
        total += 1

    recall = hits / total if total > 0 else 0
    avg_time = (total_time / total * 1000) if total > 0 else 0

    return recall, avg_time


def load_scifact(n_docs: int = 100):
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
        docs.append({
            "id": item["_id"],
            "text": f"{item['title']} {item['text']}"
        })
        doc_id_set.add(item["_id"])

    queries = []
    relevance = {}
    for item in queries_ds:
        query_id = item["_id"]
        if query_id in qrels:
            relevant = qrels[query_id] & doc_id_set
            if relevant:
                queries.append({"id": query_id, "text": item["text"]})
                relevance[query_id] = list(relevant)

    return docs, queries, relevance


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("MULTI-RESOLUTION SEARCH TEST")
    print("=" * 70)

    # Encoder
    token_encoder = TokenEncoder()
    pooled_encoder = PooledEncoder()

    # Daten
    print("\n1. Lade Daten...")
    n_docs = 1000  # Mehr Docs = mehr Queries mit Relevanz
    docs, queries, relevance = load_scifact(n_docs)
    print(f"   {len(docs)} Docs, {len(queries)} Queries mit Relevanz")

    # Index bauen
    print("\n2. Baue Multi-Resolution Index...")
    start = time.time()
    index = build_multi_res_index(docs, token_encoder, paragraph_size=20)
    index_time = time.time() - start
    print(f"   Zeit: {index_time:.1f}s")

    # Statistiken
    total_tokens = sum(d.n_tokens for d in index.values())
    total_paras = sum(len(d.paragraph_embeddings) for d in index.values())
    print(f"\n   Statistiken:")
    print(f"     Dokumente:   {len(index)}")
    print(f"     Paragraphen: {total_paras} ({total_paras/len(index):.1f}/Doc)")
    print(f"     Tokens:      {total_tokens} ({total_tokens/len(index):.1f}/Doc)")

    # Pooled Index (Standard RAG Baseline)
    print("\n3. Baue Pooled Index (Standard RAG)...")
    start = time.time()
    pooled_index = build_pooled_index(docs, pooled_encoder)
    pooled_time = time.time() - start
    print(f"   Zeit: {pooled_time:.1f}s")

    # Vergleichszahlen
    print(f"\n   Vergleiche (Brute Force vs Hierarchisch):")
    print(f"     Brute Force:    {len(index)} × {total_tokens/len(index):.0f} = {total_tokens} Token-Vergleiche")
    print(f"     Hierarchisch:   {len(index)} + 50×{total_paras/len(index):.0f} + 20×20 = ~{len(index) + 50*total_paras//len(index) + 400}")

    # Evaluation
    print("\n" + "=" * 70)
    print("SEARCH EVALUATION")
    print("=" * 70)

    # Pooled Baseline (Standard RAG)
    print("\n   Pooled (Standard RAG)...")
    pooled_hits = 0
    pooled_total = 0
    pooled_time_total = 0
    for query in queries:
        if query["id"] not in relevance:
            continue
        relevant = set(relevance[query["id"]])
        query_emb = pooled_encoder.encode(query["text"])
        start = time.time()
        results = search_pooled(query_emb, pooled_index, top_k=10)
        pooled_time_total += time.time() - start
        found = set(r[0] for r in results)
        if relevant & found:
            pooled_hits += 1
        pooled_total += 1
    pooled_recall = pooled_hits / pooled_total if pooled_total > 0 else 0
    pooled_search_time = (pooled_time_total / pooled_total * 1000) if pooled_total > 0 else 0

    # Brute Force Tokens
    print("   Brute Force (alle Tokens)...")
    bf_recall, bf_time = evaluate_search(
        queries, relevance, search_brute_force_tokens, index, pooled_encoder
    )

    # Document Level Mean (Token-Aggregated)
    print("   Token-Aggregated (Mean)...")
    mean_recall, mean_time = evaluate_search(
        queries, relevance, search_doc_level_mean, index, pooled_encoder
    )

    # Document Level Max
    print("   Token-Aggregated (Max)...")
    max_recall, max_time = evaluate_search(
        queries, relevance, search_doc_level_max, index, pooled_encoder
    )

    # Hierarchisch mit Mean
    print("   Hierarchisch Mean (100/30)...")
    hier_mean_recall, hier_mean_time = evaluate_search(
        queries, relevance, search_hierarchical, index, pooled_encoder,
        level1_candidates=100, level2_candidates=30, use_max=False
    )

    # Hierarchisch mit Max
    print("   Hierarchisch Max (100/30)...")
    hier_max_recall, hier_max_time = evaluate_search(
        queries, relevance, search_hierarchical, index, pooled_encoder,
        level1_candidates=100, level2_candidates=30, use_max=True
    )

    # Hierarchisch mit Max und mehr Kandidaten
    print("   Hierarchisch Max (200/50)...")
    hier_max_200_recall, hier_max_200_time = evaluate_search(
        queries, relevance, search_hierarchical, index, pooled_encoder,
        level1_candidates=200, level2_candidates=50, use_max=True
    )

    # Ergebnisse
    print(f"\n   {'Methode':<28} {'R@10':>8} {'Zeit':>10} {'vs Pooled':>10}")
    print(f"   {'-'*58}")
    print(f"   {'Pooled (Standard RAG)':<28} {pooled_recall*100:>7.1f}% {pooled_search_time:>8.1f}ms {'baseline':>10}")
    print(f"   {'Token-Aggr (Mean)':<28} {mean_recall*100:>7.1f}% {mean_time:>8.1f}ms {'+' if mean_recall > pooled_recall else ''}{(mean_recall-pooled_recall)*100:>7.1f}%")
    print(f"   {'Token-Aggr (Max)':<28} {max_recall*100:>7.1f}% {max_time:>8.1f}ms {'+' if max_recall > pooled_recall else ''}{(max_recall-pooled_recall)*100:>7.1f}%")
    print(f"   {'Brute Force (Token)':<28} {bf_recall*100:>7.1f}% {bf_time:>8.1f}ms {'+' if bf_recall > pooled_recall else ''}{(bf_recall-pooled_recall)*100:>7.1f}%")
    print(f"   {'Hierarchisch Mean (100/30)':<28} {hier_mean_recall*100:>7.1f}% {hier_mean_time:>8.1f}ms {'+' if hier_mean_recall > pooled_recall else ''}{(hier_mean_recall-pooled_recall)*100:>7.1f}%")
    print(f"   {'Hierarchisch Max (100/30)':<28} {hier_max_recall*100:>7.1f}% {hier_max_time:>8.1f}ms {'+' if hier_max_recall > pooled_recall else ''}{(hier_max_recall-pooled_recall)*100:>7.1f}%")
    print(f"   {'Hierarchisch Max (200/50)':<28} {hier_max_200_recall*100:>7.1f}% {hier_max_200_time:>8.1f}ms {'+' if hier_max_200_recall > pooled_recall else ''}{(hier_max_200_recall-pooled_recall)*100:>7.1f}%")

    # Analyse
    print("\n" + "=" * 70)
    print("ANALYSE")
    print("=" * 70)

    print(f"""
   AGGREGATION VERGLEICH:
     Mean-Pooling:     {mean_recall*100:.1f}% (= Pooled)
     Max-Pooling:      {max_recall*100:.1f}% ({'+' if max_recall > mean_recall else ''}{(max_recall-mean_recall)*100:.1f}% vs Mean)
     Brute Force:      {bf_recall*100:.1f}% ({'+' if bf_recall > mean_recall else ''}{(bf_recall-mean_recall)*100:.1f}% vs Mean)

   HIERARCHISCH MIT MAX:
     Max (100/30):     {hier_max_recall*100:.1f}% ({'+' if hier_max_recall > hier_mean_recall else ''}{(hier_max_recall-hier_mean_recall)*100:.1f}% vs Mean-Hier)
     Max (200/50):     {hier_max_200_recall*100:.1f}% ({'+' if hier_max_200_recall > hier_mean_recall else ''}{(hier_max_200_recall-hier_mean_recall)*100:.1f}% vs Mean-Hier)

   FAZIT:
     Max-Pooling {'verbessert' if max_recall > mean_recall else 'verbessert nicht'} die Document-Level Suche
     Hierarchisch Max {'erreicht' if hier_max_200_recall >= bf_recall * 0.95 else 'erreicht nicht'} Brute-Force Niveau
""")


if __name__ == "__main__":
    main()
