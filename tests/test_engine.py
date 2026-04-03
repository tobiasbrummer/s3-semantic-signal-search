import os

import numpy as np
import pytest
import torch


try:
    from engine import EmbeddingResult, SemanticEngine, adaptive_threshold, slerp
except ImportError:  # pragma: no cover
    from semantic_engine.engine import (  # type: ignore
        EmbeddingResult,
        SemanticEngine,
        adaptive_threshold,
        slerp,
    )


class DummyDenseEncoding(dict):
    def __init__(self, input_ids: torch.Tensor, offset_mapping: torch.Tensor, word_ids: list[int]):
        super().__init__()
        self["input_ids"] = input_ids
        self["offset_mapping"] = offset_mapping
        self._word_ids = word_ids

    def word_ids(self, batch_index: int = 0) -> list[int]:
        return self._word_ids


class DummyDenseTokenizer:
    def __init__(self, total_tokens: int):
        self.total_tokens = total_tokens

    def __call__(
        self,
        text: str,
        return_tensors: str,
        padding: bool,
        truncation: bool,
        return_offsets_mapping: bool,
    ) -> DummyDenseEncoding:
        input_ids = torch.arange(self.total_tokens, dtype=torch.long).unsqueeze(0)
        offset_mapping = torch.tensor(
            [(index, index + 1) for index in range(self.total_tokens)],
            dtype=torch.long,
        ).unsqueeze(0)
        word_ids = list(range(self.total_tokens))
        return DummyDenseEncoding(input_ids, offset_mapping, word_ids)


class DummyDenseOutput:
    def __init__(self, last_hidden_state: torch.Tensor):
        self.last_hidden_state = last_hidden_state


class DummyDenseModel:
    def __init__(self, embedding_dim: int = 4):
        self.embedding_dim = embedding_dim

    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> DummyDenseOutput:
        seq_len = int(input_ids.shape[1])
        token_ids = input_ids[0].to(torch.float32)
        offsets = torch.arange(self.embedding_dim, dtype=torch.float32)
        embeddings = token_ids.unsqueeze(1) + offsets.unsqueeze(0)
        return DummyDenseOutput(last_hidden_state=embeddings.unsqueeze(0))


def test_slerp_endpoints_and_norm():
    torch.manual_seed(0)
    vec0 = torch.nn.functional.normalize(torch.randn(16), dim=0)
    vec1 = torch.nn.functional.normalize(torch.randn(16), dim=0)

    out0 = slerp(vec0, vec1, t=0.0)
    out1 = slerp(vec0, vec1, t=1.0)
    out_mid = slerp(vec0, vec1, t=0.5)

    assert torch.allclose(out0, vec0, atol=1e-6)
    assert torch.allclose(out1, vec1, atol=1e-6)
    assert torch.isfinite(out_mid).all()
    assert torch.allclose(torch.norm(out_mid), torch.tensor(1.0), atol=1e-5)


def test_embedding_result_normalized_dense_handles_zero_rows():
    result = EmbeddingResult(
        words=["a", "b"],
        dense_vectors=np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]], dtype=np.float32),
        sparse_weights=np.array([0.0, 1.0], dtype=np.float32),
        token_offsets=[(0, 1), (2, 3)],
        text="a b",
    )

    normalized = result.normalized_dense
    assert np.isfinite(normalized).all()
    assert np.allclose(normalized[0], np.zeros(3, dtype=np.float32))
    assert np.allclose(np.linalg.norm(normalized[1]), 1.0, atol=1e-6)


def test_adaptive_threshold_respects_min_threshold():
    scores = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    threshold = adaptive_threshold(scores, rel_factor=100.0, min_threshold=0.5)
    assert threshold == 0.5


def test_match_sparse_to_dense_max_pools_overlaps():
    engine = SemanticEngine.__new__(SemanticEngine)
    dense_offsets = [(0, 2), (3, 5)]
    sparse_offsets = [(0, 1), (1, 2), (3, 4), (4, 5)]
    sparse_weights = np.array([0.1, 0.7, 0.2, 0.9], dtype=np.float32)

    matched = engine._match_sparse_to_dense(dense_offsets, sparse_offsets, sparse_weights)
    assert np.allclose(matched, np.array([0.7, 0.9], dtype=np.float32))


def test_accumulate_word_embedding_weights_peak_at_center():
    engine = SemanticEngine.__new__(SemanticEngine)
    word_embeddings: dict[int, list[tuple[torch.Tensor, float]]] = {}
    word_infos: dict[int, tuple[str, tuple[int, int]]] = {}
    tokens = [torch.ones(4)]

    engine._accumulate_word_embedding(
        text="abcdef",
        word_id=0,
        tokens=tokens,
        token_indices=[0],
        offsets_list=[(0, 1)],
        word_embeddings=word_embeddings,
        word_infos=word_infos,
        window_size=10,
    )
    engine._accumulate_word_embedding(
        text="abcdef",
        word_id=1,
        tokens=tokens,
        token_indices=[5],
        offsets_list=[(1, 2)],
        word_embeddings=word_embeddings,
        word_infos=word_infos,
        window_size=10,
    )

    weight_edge = word_embeddings[0][0][1]
    weight_center = word_embeddings[1][0][1]
    assert weight_center > weight_edge


def test_embed_dense_passes_actual_chunk_size_into_weighting(monkeypatch):
    engine = SemanticEngine.__new__(SemanticEngine)
    engine.device = "cpu"
    engine.DENSE_MAX_LENGTH = 5
    engine.DENSE_STRIDE = 4
    engine.dense_tokenizer = DummyDenseTokenizer(total_tokens=8)
    engine.dense_model = DummyDenseModel(embedding_dim=4)

    seen_window_sizes: list[int] = []
    original = SemanticEngine._accumulate_word_embedding

    def spy_accumulate(
        self,
        text: str,
        word_id: int,
        tokens: list,
        token_indices: list,
        offsets_list: list,
        word_embeddings: dict,
        word_infos: dict,
        window_size: int,
    ):
        seen_window_sizes.append(int(window_size))
        return original(
            self,
            text,
            word_id,
            tokens,
            token_indices,
            offsets_list,
            word_embeddings,
            word_infos,
            window_size,
        )

    monkeypatch.setattr(SemanticEngine, "_accumulate_word_embedding", spy_accumulate)

    engine._embed_dense("01234567")
    assert min(seen_window_sizes) == 4
    assert max(seen_window_sizes) == 5


@pytest.mark.integration
def test_semantic_engine_embed_text_smoke():
    if os.environ.get("SEMANTIC_ENGINE_RUN_MODEL_TESTS") != "1":
        pytest.skip("Set SEMANTIC_ENGINE_RUN_MODEL_TESTS=1 to run model/inference smoke tests.")

    engine = SemanticEngine(device="cpu")
    result = engine.embed_text("Die Bank erhöht die Zinsen.")
    assert len(result.words) > 0
    assert result.dense_vectors.shape[0] == len(result.words)
    assert result.sparse_weights.shape[0] == len(result.words)
