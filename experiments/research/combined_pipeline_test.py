#!/usr/bin/env python3
"""
Combined Pipeline Test

Stage 1: Coarse Filter (parallel)
- Mean-Pooled Token Embeddings
- SPLADE Inverted Index
- Union der Kandidaten

Stage 2: Multi-Resolution Refinement
- Paragraph-Level
- Token-Level

Ziel: Möglichst nah an Brute Force (94.3%) ohne alle Tokens zu durchsuchen
"""

import numpy as np
import requests
import torch
import time
from dataclasses import dataclass, field
from collections import defaultdict
from transformers import AutoModelForMaskedLM, AutoTokenizer


# =============================================================================
# ENCODERS
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


class SpladeEncoder:
    def __init__(self, model_name: str = "naver/splade-cocondenser-ensembledistil", top_k: int = 64):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.top_k = top_k

    def encode(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        inputs = self.tokenizer(
            text[:4000], return_tensors="pt", max_length=512,
            truncation=True, padding=True
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        weights = torch.max(
            torch.log1p(torch.relu(logits)), dim=1
        ).values.squeeze(0).cpu().numpy()

        nonzero_idx = np.nonzero(weights)[0]
        nonzero_weights = weights[nonzero_idx]

        if len(nonzero_idx) > self.top_k:
            top_idx = np.argsort(nonzero_weights)[-self.top_k:]
            nonzero_idx = nonzero_idx[top_idx]
            nonzero_weights = nonzero_weights[top_idx]

        return nonzero_idx, nonzero_weights


# =============================================================================
# COMBINED INDEX
# =============================================================================

@dataclass
class CombinedDoc:
    doc_id: str
    text: str

    # Token-Level
    token_embeddings: np.ndarray
    n_tokens: int

    # Aggregations
    doc_embedding_mean: np.ndarray
    paragraph_embeddings: np.ndarray  # (n_para, dim)
    paragraph_size: int = 20

    # SPLADE
    splade_terms: np.ndarray = None
    splade_weights: np.ndarray = None


@dataclass
class CombinedIndex:
    docs: dict  # doc_id → CombinedDoc
    splade_inverted: dict = field(default_factory=lambda: defaultdict(list))  # term_id → [(doc_id, weight)]


def build_combined_index(
    docs: list[dict],
    token_encoder: TokenEncoder,
    splade_encoder: SpladeEncoder,
    paragraph_size: int = 20
) -> CombinedIndex:
    """Baue kombinierten Index mit Token-Embeddings und SPLADE."""

    index = CombinedIndex(docs={})

    for i, doc in enumerate(docs):
        doc_id = doc["id"]
        text = doc["text"]

        # Token Embeddings
        token_embs = token_encoder.encode(text)

        # Mean-Pooled
        doc_mean = token_embs.mean(axis=0)

        # Paragraphs
        n_tokens = len(token_embs)
        n_paras = max(1, n_tokens // paragraph_size)
        para_embs = []
        for p in range(n_paras):
            start = p * paragraph_size
            end = min(start + paragraph_size, n_tokens)
            para_embs.append(token_embs[start:end].mean(axis=0))
        para_embs = np.array(para_embs)

        # SPLADE
        splade_terms, splade_weights = splade_encoder.encode(text)

        # Store
        index.docs[doc_id] = CombinedDoc(
            doc_id=doc_id,
            text=text,
            token_embeddings=token_embs,
            n_tokens=n_tokens,
            doc_embedding_mean=doc_mean,
            paragraph_embeddings=para_embs,
            paragraph_size=paragraph_size,
            splade_terms=splade_terms,
            splade_weights=splade_weights
        )

        # SPLADE Inverted Index
        for term_id, weight in zip(splade_terms, splade_weights):
            index.splade_inverted[int(term_id)].append((doc_id, float(weight)))

        if (i + 1) % 50 == 0:
            print(f"  Indexed {i+1}/{len(docs)}")

    return index


# =============================================================================
# SEARCH FUNCTIONS
# =============================================================================

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)


def search_dense_only(
    query_emb: np.ndarray,
    index: CombinedIndex,
    top_k: int = 100
) -> list[tuple[str, float]]:
    """Stage 1a: Dense Mean-Pooled Search."""
    results = []
    for doc_id, doc in index.docs.items():
        score = cosine_sim(query_emb, doc.doc_embedding_mean)
        results.append((doc_id, score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def search_splade_only(
    query_terms: np.ndarray,
    query_weights: np.ndarray,
    index: CombinedIndex,
    top_k: int = 100
) -> list[tuple[str, float]]:
    """Stage 1b: SPLADE Inverted Index Search."""
    scores = defaultdict(float)

    for term_id, q_weight in zip(query_terms, query_weights):
        if term_id in index.splade_inverted:
            for doc_id, d_weight in index.splade_inverted[term_id]:
                scores[doc_id] += q_weight * d_weight

    results = [(doc_id, score) for doc_id, score in scores.items()]
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def search_combined_stage1(
    query_emb: np.ndarray,
    query_terms: np.ndarray,
    query_weights: np.ndarray,
    index: CombinedIndex,
    dense_top_k: int = 100,
    splade_top_k: int = 100
) -> list[str]:
    """Stage 1: Kombiniere Dense und SPLADE Kandidaten."""

    # Dense candidates
    dense_results = search_dense_only(query_emb, index, dense_top_k)
    dense_docs = set(r[0] for r in dense_results)

    # SPLADE candidates
    splade_results = search_splade_only(query_terms, query_weights, index, splade_top_k)
    splade_docs = set(r[0] for r in splade_results)

    # Union
    candidates = dense_docs | splade_docs

    return list(candidates)


def search_stage2_token(
    query_emb: np.ndarray,
    candidates: list[str],
    index: CombinedIndex,
    top_k: int = 10
) -> list[tuple[str, float, int]]:
    """Stage 2: Token-Level Refinement auf Kandidaten."""

    results = []
    for doc_id in candidates:
        doc = index.docs[doc_id]

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


def search_stage2_paragraph_then_token(
    query_emb: np.ndarray,
    candidates: list[str],
    index: CombinedIndex,
    para_top_k: int = 50,
    top_k: int = 10
) -> list[tuple[str, float, int]]:
    """Stage 2: Paragraph → Token Refinement."""

    # Paragraph level
    para_results = []
    for doc_id in candidates:
        doc = index.docs[doc_id]

        best_score = 0
        best_para = 0
        for p, para_emb in enumerate(doc.paragraph_embeddings):
            score = cosine_sim(query_emb, para_emb)
            if score > best_score:
                best_score = score
                best_para = p

        para_results.append((doc_id, best_score, best_para))

    para_results.sort(key=lambda x: x[1], reverse=True)
    top_paras = para_results[:para_top_k]

    # Token level on top paragraphs
    results = []
    for doc_id, _, best_para in top_paras:
        doc = index.docs[doc_id]

        # Token range for this paragraph
        start = best_para * doc.paragraph_size
        end = min(start + doc.paragraph_size, doc.n_tokens)

        best_score = 0
        best_pos = start
        for t in range(start, end):
            score = cosine_sim(query_emb, doc.token_embeddings[t])
            if score > best_score:
                best_score = score
                best_pos = t

        results.append((doc_id, best_score, best_pos))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def search_brute_force(
    query_emb: np.ndarray,
    index: CombinedIndex,
    top_k: int = 10
) -> list[tuple[str, float, int]]:
    """Brute Force: Alle Tokens durchsuchen."""

    results = []
    for doc_id, doc in index.docs.items():
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


# =============================================================================
# FULL PIPELINE
# =============================================================================

def search_combined_pipeline(
    query_emb: np.ndarray,
    query_terms: np.ndarray,
    query_weights: np.ndarray,
    index: CombinedIndex,
    stage1_dense_k: int = 100,
    stage1_splade_k: int = 100,
    stage2_para_k: int = 50,
    final_k: int = 10,
    skip_stage2: bool = False
) -> list[tuple[str, float, int]]:
    """
    Full Combined Pipeline:
    Stage 1: Dense + SPLADE → Kandidaten
    Stage 2: Paragraph → Token Refinement
    """

    # Stage 1
    candidates = search_combined_stage1(
        query_emb, query_terms, query_weights, index,
        stage1_dense_k, stage1_splade_k
    )

    if skip_stage2:
        # Just return dense scores for candidates
        results = []
        for doc_id in candidates:
            doc = index.docs[doc_id]
            score = cosine_sim(query_emb, doc.doc_embedding_mean)
            results.append((doc_id, score, 0))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:final_k]

    # Stage 2
    return search_stage2_paragraph_then_token(
        query_emb, candidates, index, stage2_para_k, final_k
    )


# =============================================================================
# EVALUATION
# =============================================================================

def load_scifact(n_docs: int = 500):
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
        docs.append({"id": item["_id"], "text": f"{item['title']} {item['text']}"})
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


def main():
    print("=" * 70)
    print("COMBINED PIPELINE TEST")
    print("=" * 70)

    # Encoders
    print("\n1. Lade Encoder...")
    token_encoder = TokenEncoder()
    pooled_encoder = PooledEncoder()
    splade_encoder = SpladeEncoder(top_k=64)
    print("   OK")

    # Data
    print("\n2. Lade Daten...")
    n_docs = 5000
    docs, queries, relevance = load_scifact(n_docs)
    print(f"   {len(docs)} Docs, {len(queries)} Queries")

    # Build Index
    print("\n3. Baue Combined Index...")
    start = time.time()
    index = build_combined_index(docs, token_encoder, splade_encoder)
    index_time = time.time() - start
    print(f"   Zeit: {index_time:.1f}s")

    # Stats
    total_tokens = sum(d.n_tokens for d in index.docs.values())
    print(f"   Tokens: {total_tokens}")
    print(f"   SPLADE Terms: {len(index.splade_inverted)}")

    # Evaluation
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)

    results_table = []

    for name, search_fn in [
        ("Brute Force (Token)", lambda q, qt, qw: search_brute_force(q, index)),
        ("Dense Only (Mean)", lambda q, qt, qw: [(d, s, 0) for d, s in search_dense_only(q, index, 10)]),
        ("SPLADE Only", lambda q, qt, qw: [(d, s, 0) for d, s in search_splade_only(qt, qw, index, 10)]),
        ("Combined Stage1 (100+100)", lambda q, qt, qw: search_combined_pipeline(q, qt, qw, index, 100, 100, skip_stage2=True)),
        ("Combined Full (100+100→50)", lambda q, qt, qw: search_combined_pipeline(q, qt, qw, index, 100, 100, 50)),
        ("Combined Full (150+150→75)", lambda q, qt, qw: search_combined_pipeline(q, qt, qw, index, 150, 150, 75)),
        ("Combined Full (200+200→100)", lambda q, qt, qw: search_combined_pipeline(q, qt, qw, index, 200, 200, 100)),
    ]:
        print(f"   {name}...")
        hits = 0
        total = 0
        total_time = 0

        for query in queries:
            if query["id"] not in relevance:
                continue

            relevant = set(relevance[query["id"]])

            # Encode query
            query_emb = pooled_encoder.encode(query["text"])
            query_terms, query_weights = splade_encoder.encode(query["text"])

            start = time.time()
            results = search_fn(query_emb, query_terms, query_weights)
            total_time += time.time() - start

            found = set(r[0] for r in results[:10])
            if relevant & found:
                hits += 1
            total += 1

        recall = hits / total if total > 0 else 0
        avg_time = (total_time / total * 5000) if total > 0 else 0

        results_table.append((name, recall, avg_time))

    # Print results
    bf_recall = results_table[0][1]

    print(f"\n   {'Methode':<30} {'R@10':>8} {'Zeit':>10} {'vs BF':>10}")
    print(f"   {'-'*60}")
    for name, recall, avg_time in results_table:
        diff = (recall - bf_recall) * 100
        print(f"   {name:<30} {recall*100:>7.1f}% {avg_time:>8.1f}ms {diff:>+9.1f}%")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSE")
    print("=" * 70)

    combined_best = max(recall for name, recall, avg_time in results_table if "Combined Full" in name)
    gap = (bf_recall - combined_best) * 100

    print(f"""
   Brute Force:          {bf_recall*100:.1f}%
   Beste Combined:       {combined_best*100:.1f}%
   Gap:                  {gap:.1f}%

   {'Combined erreicht Brute-Force Niveau!' if gap < 2 else f'Combined ist {gap:.1f}% unter Brute Force'}
""")


if __name__ == "__main__":
    main()
