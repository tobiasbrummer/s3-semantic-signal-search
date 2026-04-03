#!/usr/bin/env python3
"""
Track A: Text-Diff vs Paraphrase using semantic audio features.

Goal:
- Paraphrase pairs should be close.
- Content-changing pairs should be farther.

Uses BAAI/bge-base-en-v1.5 on CPU with attention-based amplitudes.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from transformers import AutoModel, AutoTokenizer


BANDS_768 = [
    (0, 192, 4500, 500),
    (192, 384, 3500, 500),
    (384, 576, 2500, 500),
    (576, 768, 1500, 500),
]

VALUE_CLIP = 3.0

DEFAULT_PAIRS = [
    {
        "id": "p1",
        "kind": "paraphrase",
        "a": "The meeting was postponed to next week.",
        "b": "The meeting was delayed until the following week.",
    },
    {
        "id": "p2",
        "kind": "paraphrase",
        "a": "A cat sat on the warm windowsill.",
        "b": "A warm windowsill had a cat sitting on it.",
    },
    {
        "id": "p3",
        "kind": "paraphrase",
        "a": "The software update improved battery life.",
        "b": "Battery life got better after the software update.",
    },
    {
        "id": "n1",
        "kind": "negation",
        "a": "The patient has a fever.",
        "b": "The patient does not have a fever.",
    },
    {
        "id": "n2",
        "kind": "negation",
        "a": "The contract allows early termination.",
        "b": "The contract does not allow early termination.",
    },
    {
        "id": "e1",
        "kind": "entity",
        "a": "Paris is the capital of France.",
        "b": "Berlin is the capital of France.",
    },
    {
        "id": "e2",
        "kind": "entity",
        "a": "The CEO is Alice Morgan.",
        "b": "The CEO is Daniel Morgan.",
    },
    {
        "id": "num1",
        "kind": "numeric",
        "a": "Revenue grew by 3 percent this quarter.",
        "b": "Revenue grew by 30 percent this quarter.",
    },
    {
        "id": "add1",
        "kind": "addition",
        "a": "The device passed all safety tests.",
        "b": "The device passed all safety tests and was approved for sale.",
    },
    {
        "id": "swap1",
        "kind": "factual",
        "a": "The package arrived on Monday.",
        "b": "The package arrived on Thursday.",
    },
]


@dataclass
class AudioFeatures:
    band_energy: np.ndarray
    flux: np.ndarray
    centroid: np.ndarray
    fft: np.ndarray
    fft2d: np.ndarray
    diff_l1: float
    diff_topk: float


@dataclass
class BaselineScores:
    dim_topk_corr: float
    dim_shift_topk_corr: float
    dim_shift_topk_resid: float
    spec_corr: float
    dpcm_p95: float
    dpcm_topk: float
    dpcm_sigma_count: int


class BGEEncoder:
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", max_length: int = 256):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, attn_implementation="eager")
        self.model.eval()
        self.max_length = max_length

    def encode_with_attention(self, text: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = inputs["input_ids"][0].tolist()
        attention_mask = inputs["attention_mask"][0].cpu().numpy().astype(bool)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

        embeddings = outputs.last_hidden_state[0].cpu().numpy()

        all_attn = torch.stack(outputs.attentions)
        avg_attn = all_attn.mean(dim=(0, 1, 2))
        importance = avg_attn.sum(dim=0).cpu().numpy()

        keep = attention_mask.copy()
        if input_ids and input_ids[0] == self.tokenizer.cls_token_id:
            keep[0] = False
        if input_ids and input_ids[-1] == self.tokenizer.sep_token_id:
            keep[-1] = False

        content_importance = importance[keep]
        if content_importance.size == 0:
            importance = np.zeros_like(importance)
        else:
            min_v = content_importance.min()
            max_v = content_importance.max()
            importance = (importance - min_v) / (max_v - min_v + 1e-8)
            importance = np.clip(importance, 0, 1)

        embeddings = embeddings[keep]
        importance = importance[keep]
        tokens = [tok for tok, k in zip(tokens, keep) if k]

        return embeddings, importance, tokens


class HFTokenEncoder:
    def __init__(self, model_name: str, max_length: int = 256, trust_remote_code: bool = False):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        try:
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
                attn_implementation="eager",
            )
        except TypeError:
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        self.model.eval()
        self.max_length = max_length
        self.attention_available = None

    def encode_with_attention(self, text: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = inputs["input_ids"][0].tolist()
        attention_mask = inputs["attention_mask"][0].cpu().numpy().astype(bool)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        with torch.no_grad():
            try:
                outputs = self.model(**inputs, output_attentions=True)
            except TypeError:
                outputs = self.model(**inputs)

        embeddings = outputs.last_hidden_state[0].cpu().numpy()

        importance = None
        if hasattr(outputs, "attentions") and outputs.attentions is not None:
            all_attn = torch.stack(outputs.attentions)
            avg_attn = all_attn.mean(dim=(0, 1, 2))
            importance = avg_attn.sum(dim=0).cpu().numpy()
            self.attention_available = True
        else:
            self.attention_available = False

        keep = attention_mask.copy()
        if input_ids and input_ids[0] == self.tokenizer.cls_token_id:
            keep[0] = False
        if input_ids and input_ids[-1] == self.tokenizer.sep_token_id:
            keep[-1] = False

        embeddings = embeddings[keep]
        tokens = [tok for tok, k in zip(tokens, keep) if k]

        if importance is None:
            importance = np.ones(len(embeddings), dtype=np.float32)
        else:
            content_importance = importance[keep]
            if content_importance.size == 0:
                importance = np.ones(len(embeddings), dtype=np.float32)
            else:
                min_v = content_importance.min()
                max_v = content_importance.max()
                importance = (importance - min_v) / (max_v - min_v + 1e-8)
                importance = np.clip(importance, 0, 1)
                importance = importance[keep]

        return embeddings, importance, tokens


def token_to_spectrum(
    embedding: np.ndarray,
    attention: float,
    freq_bins: np.ndarray,
    amplitude_mode: str,
) -> np.ndarray:
    n_dims = len(embedding)
    frequencies = np.zeros(n_dims)
    amplitudes = np.zeros(n_dims)

    for start, end, freq_center, freq_range in BANDS_768:
        if start >= n_dims:
            break
        actual_end = min(end, n_dims)
        values = embedding[start:actual_end]

        clipped = np.clip(values, -VALUE_CLIP, VALUE_CLIP)
        normalized = clipped / VALUE_CLIP
        frequencies[start:actual_end] = freq_center + normalized * freq_range

        if amplitude_mode == "attention":
            amplitudes[start:actual_end] = attention
        elif amplitude_mode == "attention_abs":
            amplitudes[start:actual_end] = attention * np.abs(normalized)
        elif amplitude_mode == "abs":
            amplitudes[start:actual_end] = np.abs(normalized)
        else:
            raise ValueError(f"Unknown amplitude_mode: {amplitude_mode}")

    bin_idx = np.searchsorted(freq_bins, frequencies, side="right") - 1
    spectrum = np.zeros(len(freq_bins) - 1)
    valid = (bin_idx >= 0) & (bin_idx < len(spectrum))
    np.add.at(spectrum, bin_idx[valid], amplitudes[valid])

    return spectrum


def embeddings_to_spectrogram(
    embeddings: np.ndarray,
    attention: np.ndarray,
    n_freq_bins: int,
    amplitude_mode: str = "attention",
) -> tuple[np.ndarray, np.ndarray]:
    n_tokens = len(embeddings)
    freq_bins = np.linspace(1000, 5000, n_freq_bins + 1)
    spectrogram = np.zeros((n_freq_bins, n_tokens))

    for t in range(n_tokens):
        spectrogram[:, t] = token_to_spectrum(
            embeddings[t], attention[t], freq_bins, amplitude_mode
        )

    return spectrogram, freq_bins


def _signal_matrix(
    embeddings: np.ndarray,
    attention: np.ndarray,
    amplitude_mode: str,
) -> np.ndarray:
    clipped = np.clip(embeddings, -VALUE_CLIP, VALUE_CLIP)
    normalized = clipped / VALUE_CLIP

    if amplitude_mode == "attention":
        weights = attention[:, None]
        signal = normalized * weights
    elif amplitude_mode == "attention_abs":
        weights = attention[:, None] * np.abs(normalized)
        signal = normalized * weights
    elif amplitude_mode == "abs":
        weights = np.abs(normalized)
        signal = normalized * weights
    else:
        raise ValueError(f"Unknown amplitude_mode: {amplitude_mode}")

    return signal


def _max_sliding_corr(query: np.ndarray, text: np.ndarray) -> float:
    if query.size == 0 or text.size == 0:
        return 0.0
    if len(query) > len(text):
        query, text = text, query
    q_norm = np.linalg.norm(query)
    if q_norm == 0:
        return 0.0
    max_corr = -1.0
    for start in range(len(text) - len(query) + 1):
        window = text[start:start + len(query)]
        denom = q_norm * np.linalg.norm(window)
        if denom == 0:
            continue
        corr = float(np.dot(query, window) / denom)
        if corr > max_corr:
            max_corr = corr
    return max_corr if max_corr != -1.0 else 0.0


def _zscore_1d(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    mean = values.mean()
    std = values.std()
    if std == 0:
        return values - mean
    return (values - mean) / std


def _zscore_matrix(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (values - mean) / std


def _delta_matrix(values: np.ndarray) -> np.ndarray:
    if values.shape[0] < 2:
        return np.zeros_like(values)
    return np.diff(values, axis=0)


def dpcm_anomaly(
    query_signal: np.ndarray,
    text_signal: np.ndarray,
    zscore: bool = True,
    sigma_thresh: float = 2.0,
    topk_pct: float = 0.05,
) -> tuple[float, float, int]:
    if query_signal.size == 0 or text_signal.size == 0:
        return 0.0, 0.0, 0
    if len(query_signal) > len(text_signal):
        query_signal, text_signal = text_signal, query_signal

    q_len, n_dims = query_signal.shape
    t_len = text_signal.shape[0]
    if q_len < 2:
        return 0.0, 0.0, 0

    delta_q = query_signal[1:] - query_signal[:-1]
    std_q = delta_q.std(axis=0, keepdims=True)
    std_q = np.where(std_q == 0, 1.0, std_q)

    best_p95 = 0.0
    best_topk = 0.0
    best_count = 0
    for start in range(t_len - q_len + 1):
        window = text_signal[start:start + q_len]
        delta_w = window[1:] - window[:-1]
        diff = delta_w - delta_q
        mag = np.abs(diff)
        if zscore:
            z = mag / std_q
        else:
            z = mag
        flat = z.ravel()
        if flat.size == 0:
            continue
        p95 = float(np.percentile(flat, 95))
        k = max(int(flat.size * topk_pct), 1)
        topk = np.partition(flat, -k)[-k:]
        topk_mean = float(topk.mean())
        count = int((flat > sigma_thresh).sum())
        if topk_mean > best_topk or (topk_mean == best_topk and p95 > best_p95):
            best_topk = topk_mean
            best_p95 = p95
            best_count = count

    return best_p95, best_topk, best_count


def dim_topk_correlation(
    query_signal: np.ndarray,
    text_signal: np.ndarray,
    topk_pct: float,
    min_k: int = 8,
    zscore: bool = False,
    delta: bool = False,
) -> float:
    if zscore:
        query_signal = _zscore_matrix(query_signal)
        text_signal = _zscore_matrix(text_signal)
    if delta:
        query_signal = _delta_matrix(query_signal)
        text_signal = _delta_matrix(text_signal)

    n_dims = query_signal.shape[1]
    scores = np.zeros(n_dims)
    for d in range(n_dims):
        scores[d] = _max_sliding_corr(query_signal[:, d], text_signal[:, d])
    k = max(int(n_dims * topk_pct), min_k)
    k = min(k, n_dims)
    topk = np.partition(scores, -k)[-k:]
    return float(np.mean(topk)) if topk.size else 0.0


def dim_shifted_topk_correlation(
    query_signal: np.ndarray,
    text_signal: np.ndarray,
    topk_pct: float,
    min_k: int = 8,
    zscore: bool = False,
    delta: bool = False,
) -> tuple[float, float]:
    if zscore:
        query_signal = _zscore_matrix(query_signal)
        text_signal = _zscore_matrix(text_signal)
    if delta:
        query_signal = _delta_matrix(query_signal)
        text_signal = _delta_matrix(text_signal)

    if query_signal.size == 0 or text_signal.size == 0:
        return 0.0
    if len(query_signal) > len(text_signal):
        query_signal, text_signal = text_signal, query_signal

    q_len, n_dims = query_signal.shape
    t_len = text_signal.shape[0]
    q_norms = np.linalg.norm(query_signal, axis=0)

    k = max(int(n_dims * topk_pct), min_k)
    k = min(k, n_dims)

    max_score = -1.0
    best_resid = 0.0
    for start in range(t_len - q_len + 1):
        window = text_signal[start:start + q_len]
        w_norms = np.linalg.norm(window, axis=0)
        denom = q_norms * w_norms
        dot = (query_signal * window).sum(axis=0)
        corr = np.zeros(n_dims)
        valid = denom > 0
        corr[valid] = dot[valid] / denom[valid]
        topk = np.partition(corr, -k)[-k:]
        score = float(np.mean(topk)) if topk.size else 0.0
        if score > max_score:
            max_score = score
            topk_idx = np.argpartition(corr, -k)[-k:]
            resid = np.zeros(n_dims)
            if q_len > 0:
                resid = np.linalg.norm(query_signal - window, axis=0) / (q_norms + 1e-8)
            best_resid = float(np.mean(resid[topk_idx])) if topk_idx.size else 0.0
        if score > max_score:
            max_score = score

    if max_score == -1.0:
        return 0.0, 0.0
    return max_score, best_resid


def spectrogram_sliding_corr(query_spec: np.ndarray, text_spec: np.ndarray,
                             zscore: bool = False, delta: bool = False) -> float:
    if query_spec.size == 0 or text_spec.size == 0:
        return 0.0
    if zscore:
        query_spec = _zscore_matrix(query_spec.T).T
        text_spec = _zscore_matrix(text_spec.T).T
    if delta:
        query_spec = _delta_matrix(query_spec.T).T
        text_spec = _delta_matrix(text_spec.T).T
    if query_spec.shape[0] != text_spec.shape[0]:
        raise ValueError("Spectrogram freq bins must match for sliding correlation.")
    q_len = query_spec.shape[1]
    t_len = text_spec.shape[1]
    if q_len > t_len:
        query_spec, text_spec = text_spec, query_spec
        q_len, t_len = t_len, q_len

    q_flat = query_spec.reshape(-1)
    q_norm = np.linalg.norm(q_flat)
    if q_norm == 0:
        return 0.0
    max_corr = -1.0
    for start in range(t_len - q_len + 1):
        window = text_spec[:, start:start + q_len].reshape(-1)
        denom = q_norm * np.linalg.norm(window)
        if denom == 0:
            continue
        corr = float(np.dot(q_flat, window) / denom)
        if corr > max_corr:
            max_corr = corr
    return max_corr if max_corr != -1.0 else 0.0


def _resample_1d(values: np.ndarray, target_len: int) -> np.ndarray:
    if target_len <= 0:
        return np.array([])
    if len(values) == 0:
        return np.zeros(target_len)
    if len(values) == 1:
        return np.full(target_len, values[0])
    x_old = np.linspace(0, 1, len(values))
    x_new = np.linspace(0, 1, target_len)
    return np.interp(x_new, x_old, values)


def _resample_axis(data: np.ndarray, target_len: int, axis: int) -> np.ndarray:
    if data.size == 0 or target_len <= 0:
        shape = list(data.shape)
        shape[axis] = max(target_len, 0)
        return np.zeros(shape)
    if data.shape[axis] == target_len:
        return data
    idx = np.arange(data.shape[axis])
    new_idx = np.linspace(0, data.shape[axis] - 1, target_len)
    if axis == 0:
        out = np.zeros((target_len, data.shape[1]))
        for col in range(data.shape[1]):
            out[:, col] = np.interp(new_idx, idx, data[:, col])
        return out
    out = np.zeros((data.shape[0], target_len))
    for row in range(data.shape[0]):
        out[row, :] = np.interp(new_idx, idx, data[row, :])
    return out


def _resample_2d(spectrogram: np.ndarray, target_freq: int, target_time: int) -> np.ndarray:
    resampled = _resample_axis(spectrogram, target_freq, axis=0)
    return _resample_axis(resampled, target_time, axis=1)


def _band_series(spectrogram: np.ndarray, freq_bins: np.ndarray) -> list[np.ndarray]:
    freq_centers = (freq_bins[:-1] + freq_bins[1:]) / 2
    series = []
    for _, _, center, band_range in BANDS_768:
        low = center - band_range
        high = center + band_range
        mask = (freq_centers >= low) & (freq_centers < high)
        series.append(spectrogram[mask].sum(axis=0))
    return series


def spectrogram_features(
    spectrogram: np.ndarray,
    freq_bins: np.ndarray,
    target_len: int = 64,
    fft_bins: int = 16,
    fft2d_bins: int = 12,
    fft2d_freq_bins: int = 128,
    diff_topk_pct: float = 0.05,
    diff_ref: np.ndarray | None = None,
) -> AudioFeatures:
    freq_centers = (freq_bins[:-1] + freq_bins[1:]) / 2

    total_energy = spectrogram.sum()
    band_energy = []
    for _, _, center, band_range in BANDS_768:
        low = center - band_range
        high = center + band_range
        mask = (freq_centers >= low) & (freq_centers < high)
        band_sum = spectrogram[mask].sum()
        band_energy.append(band_sum)
    band_energy = np.array(band_energy)
    if total_energy > 0:
        band_energy = band_energy / total_energy

    if spectrogram.shape[1] > 1:
        diffs = np.diff(spectrogram, axis=1)
        flux = np.abs(diffs).sum(axis=0)
    else:
        flux = np.array([])

    denom = spectrogram.sum(axis=0) + 1e-8
    centroid = (spectrogram.T @ freq_centers) / denom
    freq_min = freq_bins[0]
    freq_max = freq_bins[-1]
    centroid = (centroid - freq_min) / max(freq_max - freq_min, 1e-8)

    flux_resampled = _resample_1d(flux, target_len)
    centroid_resampled = _resample_1d(centroid, target_len)

    fft_features = []
    for band in _band_series(spectrogram, freq_bins):
        band_resampled = _resample_1d(band, target_len)
        spectrum = np.abs(np.fft.rfft(band_resampled))
        if spectrum.size > 1:
            spectrum = spectrum[1:]
        if spectrum.size > fft_bins:
            spectrum = spectrum[:fft_bins]
        if spectrum.sum() > 0:
            spectrum = spectrum / spectrum.sum()
        fft_features.append(spectrum)

    fft_concat = np.concatenate(fft_features) if fft_features else np.array([])

    spec_resampled = _resample_2d(spectrogram, fft2d_freq_bins, target_len)
    fft2d = np.abs(np.fft.rfft2(spec_resampled))
    if fft2d.size > 0:
        fft2d[0, 0] = 0.0
    max_freq = min(fft2d_bins, fft2d.shape[0])
    max_time = min(fft2d_bins, fft2d.shape[1])
    fft2d_crop = fft2d[:max_freq, :max_time].ravel()
    if fft2d_crop.sum() > 0:
        fft2d_crop = fft2d_crop / fft2d_crop.sum()

    diff_l1 = 0.0
    diff_topk = 0.0
    if diff_ref is not None:
        diff_map = np.abs(spec_resampled - diff_ref)
        diff_l1 = float(diff_map.mean())
        if diff_topk_pct > 0:
            flat = diff_map.ravel()
            k = max(int(flat.size * diff_topk_pct), 1)
            topk = np.partition(flat, -k)[-k:]
            diff_topk = float(topk.mean())

    return AudioFeatures(
        band_energy=band_energy,
        flux=flux_resampled,
        centroid=centroid_resampled,
        fft=fft_concat,
        fft2d=fft2d_crop,
        diff_l1=diff_l1,
        diff_topk=diff_topk,
    )


def audio_distance(a: AudioFeatures, b: AudioFeatures,
                   weights: tuple[float, float, float, float, float, float, float]) -> dict:
    w_band, w_flux, w_centroid, w_fft, w_fft2d, w_diff_l1, w_diff_topk = weights

    band_l1 = np.abs(a.band_energy - b.band_energy).sum()
    flux_l2 = np.linalg.norm(a.flux - b.flux) / max(len(a.flux), 1)
    centroid_l2 = np.linalg.norm(a.centroid - b.centroid) / max(len(a.centroid), 1)
    fft_l2 = np.linalg.norm(a.fft - b.fft) / max(len(a.fft), 1)
    fft2d_l2 = np.linalg.norm(a.fft2d - b.fft2d) / max(len(a.fft2d), 1)
    diff_l1 = (a.diff_l1 + b.diff_l1) / 2.0
    diff_topk = (a.diff_topk + b.diff_topk) / 2.0

    score = (
        w_band * band_l1
        + w_flux * flux_l2
        + w_centroid * centroid_l2
        + w_fft * fft_l2
        + w_fft2d * fft2d_l2
        + w_diff_l1 * diff_l1
        + w_diff_topk * diff_topk
    )

    return {
        "score": float(score),
        "band_l1": float(band_l1),
        "flux_l2": float(flux_l2),
        "centroid_l2": float(centroid_l2),
        "fft_l2": float(fft_l2),
        "fft2d_l2": float(fft2d_l2),
        "diff_l1": float(diff_l1),
        "diff_topk": float(diff_topk),
    }


def _ascii_spectrogram(spec: np.ndarray, width: int = 80, height: int = 20) -> str:
    if spec.size == 0:
        return "(empty)"
    spec_res = _resample_2d(spec, height, width)
    spec_res = spec_res - spec_res.min()
    max_val = spec_res.max()
    if max_val > 0:
        spec_res = spec_res / max_val
    chars = " .:-=+*#%@"
    lines = []
    for row in spec_res[::-1]:
        line = "".join(chars[min(int(v * (len(chars) - 1)), len(chars) - 1)] for v in row)
        lines.append(line)
    return "\n".join(lines)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def load_pairs(path: str | None) -> list[dict]:
    if path is None:
        return DEFAULT_PAIRS
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            pairs.append(json.loads(line))
    return pairs


def best_threshold(scores: list[float], labels: list[int]) -> tuple[float, float]:
    if not scores:
        return 0.0, 0.0
    sorted_scores = sorted(set(scores))
    thresholds = [(sorted_scores[i] + sorted_scores[i + 1]) / 2 for i in range(len(sorted_scores) - 1)]
    thresholds = [sorted_scores[0] - 1e-6] + thresholds + [sorted_scores[-1] + 1e-6]

    best_acc = 0.0
    best_t = thresholds[0]
    for t in thresholds:
        preds = [1 if s >= t else 0 for s in scores]
        correct = sum(1 for p, y in zip(preds, labels) if p == y)
        acc = correct / len(labels)
        if acc > best_acc:
            best_acc = acc
            best_t = t
    return best_t, best_acc


def summarize(results: list[dict]):
    by_kind = {}
    for row in results:
        by_kind.setdefault(row["kind"], []).append(row)

    print("\nSummary by kind (audio score mean):")
    for kind, rows in sorted(by_kind.items()):
        mean_score = np.mean([r["score"] for r in rows])
        print(f"  {kind:<10} {mean_score:.4f} ({len(rows)})")

    labels = [0 if r["kind"] == "paraphrase" else 1 for r in results]
    scores = [r["score"] for r in results]
    threshold, acc = best_threshold(scores, labels)
    print(f"\nBest threshold (paraphrase vs change): {threshold:.4f}, accuracy {acc:.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audio diff experiment (Track A)")
    parser.add_argument("--pairs-file", type=str, help="JSONL with {id, kind, a, b}")
    parser.add_argument("--model", type=str, default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--hf-model", type=str, help="Use a HF model instead of BGE")
    parser.add_argument("--hf-trust-remote-code", action="store_true")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--amplitude-mode", choices=["attention", "attention_abs", "abs"], default="attention")
    parser.add_argument("--target-len", type=int, default=64)
    parser.add_argument("--freq-bins", type=int, default=512)
    parser.add_argument("--fft-bins", type=int, default=16)
    parser.add_argument("--fft2d-bins", type=int, default=12)
    parser.add_argument("--fft2d-freq-bins", type=int, default=128)
    parser.add_argument("--diff-topk-pct", type=float, default=0.05)
    parser.add_argument("--baseline-dim-topk-pct", type=float, default=0.05)
    parser.add_argument("--baseline-dim-min-k", type=int, default=8)
    parser.add_argument("--baseline-dim-match", action="store_true",
                        help="Compute per-dimension sliding correlation (Top-K)")
    parser.add_argument("--baseline-dim-shifted-topk", action="store_true",
                        help="Compute shared-shift Top-K dim correlation")
    parser.add_argument("--baseline-dim-zscore", action="store_true",
                        help="Z-score per dimension before dim matching")
    parser.add_argument("--baseline-dim-delta", action="store_true",
                        help="Use temporal deltas before dim matching")
    parser.add_argument("--baseline-spec-match", action="store_true",
                        help="Compute 2D spectrogram sliding correlation")
    parser.add_argument("--baseline-spec-zscore", action="store_true",
                        help="Z-score per frequency before spectrogram matching")
    parser.add_argument("--baseline-spec-delta", action="store_true",
                        help="Use temporal deltas before spectrogram matching")
    parser.add_argument("--baseline-dpcm", action="store_true",
                        help="Compute DPCM anomaly (per-dim) on sliding windows")
    parser.add_argument("--baseline-dpcm-sigma", type=float, default=2.0)
    parser.add_argument("--baseline-dpcm-topk-pct", type=float, default=0.05)
    parser.add_argument("--band-weight", type=float, default=1.0)
    parser.add_argument("--flux-weight", type=float, default=0.5)
    parser.add_argument("--centroid-weight", type=float, default=0.5)
    parser.add_argument("--fft-weight", type=float, default=0.5)
    parser.add_argument("--fft2d-weight", type=float, default=0.5)
    parser.add_argument("--diff-l1-weight", type=float, default=1.0)
    parser.add_argument("--diff-topk-weight", type=float, default=2.0)
    parser.add_argument("--print-diff", action="store_true",
                        help="Print ASCII diff spectrogram for each pair")
    args = parser.parse_args()

    if args.hf_model:
        encoder = HFTokenEncoder(
            model_name=args.hf_model,
            max_length=args.max_length,
            trust_remote_code=args.hf_trust_remote_code,
        )
    else:
        encoder = BGEEncoder(model_name=args.model, max_length=args.max_length)
    pairs = load_pairs(args.pairs_file)

    results = []
    weights = (
        args.band_weight,
        args.flux_weight,
        args.centroid_weight,
        args.fft_weight,
        args.fft2d_weight,
        args.diff_l1_weight,
        args.diff_topk_weight,
    )

    warned_no_attn = False
    for pair in pairs:
        emb_a, attn_a, _ = encoder.encode_with_attention(pair["a"])
        emb_b, attn_b, _ = encoder.encode_with_attention(pair["b"])

        if (
            isinstance(encoder, HFTokenEncoder)
            and encoder.attention_available is False
            and not warned_no_attn
            and args.amplitude_mode in {"attention", "attention_abs"}
        ):
            print("Warning: attention not available; amplitude_mode will behave like constant/abs.")
            warned_no_attn = True

        spec_a, bins = embeddings_to_spectrogram(
            emb_a,
            attn_a,
            n_freq_bins=args.freq_bins,
            amplitude_mode=args.amplitude_mode,
        )
        spec_b, _ = embeddings_to_spectrogram(
            emb_b,
            attn_b,
            n_freq_bins=args.freq_bins,
            amplitude_mode=args.amplitude_mode,
        )

        spec_a_res = _resample_2d(spec_a, args.fft2d_freq_bins, args.target_len)
        spec_b_res = _resample_2d(spec_b, args.fft2d_freq_bins, args.target_len)

        feat_a = spectrogram_features(
            spec_a,
            bins,
            target_len=args.target_len,
            fft_bins=args.fft_bins,
            fft2d_bins=args.fft2d_bins,
            fft2d_freq_bins=args.fft2d_freq_bins,
            diff_topk_pct=args.diff_topk_pct,
            diff_ref=spec_b_res,
        )
        feat_b = spectrogram_features(
            spec_b,
            bins,
            target_len=args.target_len,
            fft_bins=args.fft_bins,
            fft2d_bins=args.fft2d_bins,
            fft2d_freq_bins=args.fft2d_freq_bins,
            diff_topk_pct=args.diff_topk_pct,
            diff_ref=spec_a_res,
        )

        dist = audio_distance(feat_a, feat_b, weights)

        baseline = BaselineScores(
            dim_topk_corr=0.0,
            dim_shift_topk_corr=0.0,
            dim_shift_topk_resid=0.0,
            spec_corr=0.0,
            dpcm_p95=0.0,
            dpcm_topk=0.0,
            dpcm_sigma_count=0,
        )
        if args.baseline_dim_match or args.baseline_dim_shifted_topk or args.baseline_dpcm:
            signal_a = _signal_matrix(emb_a, attn_a, args.amplitude_mode)
            signal_b = _signal_matrix(emb_b, attn_b, args.amplitude_mode)

        if args.baseline_dim_match:
            baseline.dim_topk_corr = dim_topk_correlation(
                signal_a,
                signal_b,
                topk_pct=args.baseline_dim_topk_pct,
                min_k=args.baseline_dim_min_k,
                zscore=args.baseline_dim_zscore,
                delta=args.baseline_dim_delta,
            )
        if args.baseline_dim_shifted_topk:
            corr, resid = dim_shifted_topk_correlation(
                signal_a,
                signal_b,
                topk_pct=args.baseline_dim_topk_pct,
                min_k=args.baseline_dim_min_k,
                zscore=args.baseline_dim_zscore,
                delta=args.baseline_dim_delta,
            )
            baseline.dim_shift_topk_corr = corr
            baseline.dim_shift_topk_resid = resid
        if args.baseline_spec_match:
            baseline.spec_corr = spectrogram_sliding_corr(
                spec_a,
                spec_b,
                zscore=args.baseline_spec_zscore,
                delta=args.baseline_spec_delta,
            )
        if args.baseline_dpcm:
            p95, topk_mean, count = dpcm_anomaly(
                signal_a,
                signal_b,
                zscore=True,
                sigma_thresh=args.baseline_dpcm_sigma,
                topk_pct=args.baseline_dpcm_topk_pct,
            )
            baseline.dpcm_p95 = p95
            baseline.dpcm_topk = topk_mean
            baseline.dpcm_sigma_count = count

        mean_a = emb_a.mean(axis=0)
        mean_b = emb_b.mean(axis=0)
        cos = cosine_similarity(mean_a, mean_b)

        results.append(
            {
                "id": pair.get("id", "?"),
                "kind": pair.get("kind", "unknown"),
                "score": dist["score"],
                "band_l1": dist["band_l1"],
                "flux_l2": dist["flux_l2"],
                "centroid_l2": dist["centroid_l2"],
                "fft_l2": dist["fft_l2"],
                "fft2d_l2": dist["fft2d_l2"],
                "diff_l1": dist["diff_l1"],
                "diff_topk": dist["diff_topk"],
                "dim_topk_corr": baseline.dim_topk_corr,
                "dim_shift_topk_corr": baseline.dim_shift_topk_corr,
                "dim_shift_topk_resid": baseline.dim_shift_topk_resid,
                "spec_corr": baseline.spec_corr,
                "dpcm_p95": baseline.dpcm_p95,
                "dpcm_topk": baseline.dpcm_topk,
                "dpcm_sigma_count": baseline.dpcm_sigma_count,
                "cosine": cos,
            }
        )

        if args.print_diff:
            diff_map = np.abs(spec_a_res - spec_b_res)
            print(f"\nDiff spectrogram for {pair.get('id', '?')} ({pair.get('kind', '?')})")
            print(_ascii_spectrogram(diff_map))

    print("\nResults:")
    row_fmt = (
        "{id:<5} {kind:<10} {score:>11.4f} {diff_topk:>9.4f} {dim_resid:>9.4f} "
        "{spec_corr:>9.4f} {dpcm_p95:>9.3f} {dpcm_topk:>9.3f} {dpcm_cnt:>8d} {cosine:>8.4f}"
    )
    header = (
        f"{'id':<5} {'kind':<10} {'audio_score':>11} {'diff_topk':>9} {'dim_resid':>9} "
        f"{'spec_corr':>9} {'dpcm_p95':>9} {'dpcm_topk':>9} {'dpcm_cnt':>8} {'cosine':>8}"
    )
    print(header)
    for row in results:
        print(row_fmt.format(
            id=row["id"],
            kind=row["kind"],
            score=row["score"],
            diff_topk=row["diff_topk"],
            dim_resid=row["dim_shift_topk_resid"],
            spec_corr=row["spec_corr"],
            dpcm_p95=row["dpcm_p95"],
            dpcm_topk=row["dpcm_topk"],
            dpcm_cnt=row["dpcm_sigma_count"],
            cosine=row["cosine"],
        ))

    summarize(results)


if __name__ == "__main__":
    main()
