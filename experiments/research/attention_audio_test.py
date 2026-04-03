#!/usr/bin/env python3
"""
Audio-Modell Test mit Attention Weights als Amplitude.
Nutzt BGE (BAAI/bge-base-en-v1.5) für Embeddings + Attention.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from transformers import AutoModel, AutoTokenizer


# Matryoshka-Bänder für 1024 dims (BGE)
BANDS_1024 = [
    (0, 256, 4500, 500),       # Band 1: Dims 0-192
#    (256, 512, 3500, 500),     # Band 2: Dims 192-384
#    (512, 768, 2500, 500),     # Band 3: Dims 384-576
#    (768, 1024, 1500, 500),     # Band 4: Dims 576-1024
]

VALUE_CLIP = 3.0


class BGEEncoder:
    def __init__(self, model_name: str = "Snowflake/snowflake-arctic-embed-l-v2.0"):
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, attn_implementation="eager")
        self.model.eval()
        print("Model loaded.")

    def encode_with_attention(self, text: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Returns:
            embeddings: (n_tokens, 1024) - Token embeddings
            attention: (n_tokens,) - Attention-based importance per token
            tokens: list of token strings
        """
        inputs = self.tokenizer(text, return_tensors="pt")
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

        # Token embeddings (ohne CLS/SEP)
        embeddings = outputs.last_hidden_state[0].numpy()  # (seq, 1024)

        # Attention importance: average over layers and heads, sum received attention
        # Stack all layers: (n_layers, 1, n_heads, seq, seq)
        all_attn = torch.stack(outputs.attentions)  # (12, 1, 12, seq, seq)

        # Average over layers and heads
        avg_attn = all_attn.mean(dim=(0, 1, 2))  # (seq, seq)

        # Sum attention received (column-wise)
        importance = avg_attn.sum(dim=0).numpy()  # (seq,)
        

        # Normalize to [0, 1], aber NUR über Content-Tokens (ohne CLS/SEP)
        # CLS ist index 0, SEP ist letzter
        content_importance = importance[1:-1]
        content_min = content_importance.min()
        
        content_max = content_importance.max()

        if content_max > 2:
            content_max = 2

        # Normalisiere alle Tokens basierend auf Content-Range
        importance = (importance - content_min) / (content_max - content_min + 1e-8)
        importance = np.clip(importance, 0, 1)

        # CLS/SEP auf niedrigen Wert setzen (nicht dominant)
        importance[0] = 0.1   # CLS
        importance[-1] = 0.1  # SEP

        return embeddings, importance, tokens


def embedding_to_frequencies_with_attention(
    embedding: np.ndarray,
    attention: float,
    bands: list = BANDS_1024
) -> tuple[np.ndarray, np.ndarray]:
    """
    Konvertiert Embedding zu Frequenzen mit Attention als Amplitude.
    """
    n_dims = len(embedding)
    frequencies = np.zeros(n_dims)
    amplitudes = np.ones(n_dims) * attention  # Attention als Basis-Amplitude

    for start, end, freq_center, freq_range in bands:
        if start >= n_dims:
            break
        actual_end = min(end, n_dims)

        values = embedding[start:actual_end]
        clipped = np.clip(values, -VALUE_CLIP, VALUE_CLIP)
        normalized = clipped / VALUE_CLIP

        frequencies[start:actual_end] = freq_center + normalized * freq_range

    return frequencies, amplitudes


def compute_spectrum(frequencies: np.ndarray, amplitudes: np.ndarray,
                     freq_bins: np.ndarray) -> np.ndarray:
    """Berechnet Spektrum aus Frequenzen und Amplituden."""
    spectrum = np.zeros(len(freq_bins) - 1)

    for freq, amp in zip(frequencies, amplitudes):
        bin_idx = np.searchsorted(freq_bins, freq) - 1
        if 0 <= bin_idx < len(spectrum):
            spectrum[bin_idx] += amp

    return spectrum


def plot_with_attention(text: str, save_path: str = None):
    """Visualisiert das Audio-Modell mit Attention-Amplitude."""
    encoder = BGEEncoder()

    print(f"\nEncoding: {text[:60]}...")
    embeddings, attention, tokens = encoder.encode_with_attention(text)
    n_tokens = len(tokens)

    print(f"Tokens: {n_tokens}, Embedding dim: {embeddings.shape[1]}")
    print(f"Attention range: {attention.min():.3f} - {attention.max():.3f}")

    # Spektrogramm berechnen
    freq_bins = np.linspace(4000, 5000, 201)
    spectrogram = np.zeros((200, n_tokens))

    for t in range(n_tokens):
        freqs, amps = embedding_to_frequencies_with_attention(
            embeddings[t], attention[t]
        )
        spectrum = compute_spectrum(freqs, amps, freq_bins)
        spectrogram[:, t] = spectrum

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(max(14, n_tokens * 0.4), 10),
                             gridspec_kw={'height_ratios': [3, 1]})

    # Spektrogramm
    ax1 = axes[0]
    spec_plot = spectrogram.copy()
    spec_plot[spec_plot < 0.001] = 0.001

    im = ax1.imshow(
        spec_plot, aspect='auto', origin='lower', cmap='magma',
        extent=[0, n_tokens, 4000, 5000],
        norm=LogNorm(vmin=0.001, vmax=spec_plot.max())
    )

    # Band-Grenzen
    for _, _, freq_center, freq_range in BANDS_1024:
        ax1.axhline(y=freq_center - 2 * freq_range, color='white', lw=0.5, alpha=0.3)
        ax1.axhline(y=freq_center + freq_range, color='white', lw=0.5, alpha=0.3)

    ax1.set_xticks(np.arange(n_tokens) + 0.5)
    ax1.set_xticklabels(tokens, rotation=90, fontsize=8, fontfamily='monospace')
    ax1.set_ylabel('Frequency (Hz)')
    ax1.set_title(f'Semantic Spectrogram with Attention Amplitude\n"{text[:80]}..."')
    plt.colorbar(im, ax=ax1, label='Amplitude (log)')

    # Attention-Kurve
    ax2 = axes[1]
    ax2.bar(range(n_tokens), attention, color='steelblue', alpha=0.7)
    ax2.set_xticks(range(n_tokens))
    ax2.set_xticklabels(tokens, rotation=90, fontsize=8, fontfamily='monospace')
    ax2.set_ylabel('Attention')
    ax2.set_title('Token Importance (Attention-based)')
    ax2.set_xlim(-0.5, n_tokens - 0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()


def compare_with_without_attention(text: str, save_path: str = None):
    """Vergleicht Spektrogramm mit und ohne Attention-Amplitude."""
    encoder = BGEEncoder()

    print(f"\nEncoding: {text[:60]}...")
    embeddings, attention, tokens = encoder.encode_with_attention(text)
    n_tokens = len(tokens)

    freq_bins = np.linspace(1000, 5000, 201)

    # Zwei Spektrogramme: mit und ohne Attention
    spec_with_attn = np.zeros((200, n_tokens))
    spec_without_attn = np.zeros((200, n_tokens))

    for t in range(n_tokens):
        # Mit Attention
        freqs, amps = embedding_to_frequencies_with_attention(embeddings[t], attention[t])
        spec_with_attn[:, t] = compute_spectrum(freqs, amps, freq_bins)

        # Ohne Attention (konstante Amplitude)
        freqs, _ = embedding_to_frequencies_with_attention(embeddings[t], 1.0)
        amps_const = np.ones(len(freqs))
        spec_without_attn[:, t] = compute_spectrum(freqs, amps_const, freq_bins)

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(max(14, n_tokens * 0.4), 10))

    for ax, spec, title in [
        (axes[0], spec_without_attn, "Without Attention (constant amplitude)"),
        (axes[1], spec_with_attn, "With Attention (variable amplitude)")
    ]:
        spec_plot = spec.copy()
        spec_plot[spec_plot < 0.001] = 0.001

        im = ax.imshow(
            spec_plot, aspect='auto', origin='lower', cmap='magma',
            extent=[0, n_tokens, 1000, 5000],
            norm=LogNorm(vmin=0.001, vmax=spec_plot.max())
        )

        ax.set_xticks(np.arange(n_tokens) + 0.5)
        ax.set_xticklabels(tokens, rotation=90, fontsize=8, fontfamily='monospace')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label='Amplitude')

    plt.suptitle(f'Attention Effect Comparison\n"{text[:60]}..."', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str,
                        default="Apple and Banana are fruits. Microsoft makes software.")
    parser.add_argument('--compare', action='store_true',
                        help='Compare with/without attention')
    parser.add_argument('--save', type=str, help='Save to file')

    args = parser.parse_args()

    if args.compare:
        compare_with_without_attention(args.text, args.save)
    else:
        plot_with_attention(args.text, args.save)
