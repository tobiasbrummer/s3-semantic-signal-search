#!/usr/bin/env python3
"""
Audio-Modell Visualisierung (Option B)

Embeddings als Audio-Signal interpretieren:
- Jede Dimension = ein Ton
- Wert (-1 bis 1) → Frequenz im jeweiligen Matryoshka-Band
- |Wert| → Lautstärke (Amplitude)
- Attention Weights → zusätzliche Lautstärke-Gewichtung (TODO)

Matryoshka-Bänder:
- Dims 1-256:    4000-5000 Hz (grobe Semantik)
- Dims 257-512:  3000-4000 Hz (mittlere Details)
- Dims 513-768:  2000-3000 Hz (feine Nuancen)
- Dims 769-1024: 1000-2000 Hz (micro-features)
"""

import numpy as np
import requests
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
import argparse


# =============================================================================
# ENCODER
# =============================================================================

class TokenEncoder:
    def __init__(self, url: str = "http://localhost:8202", max_chars: int = 4000):
        self.url = url
        self.max_chars = max_chars

    def encode(self, text: str) -> np.ndarray:
        text = text[:self.max_chars]
        response = requests.post(f"{self.url}/embeddings", json={"input": text})
        response.raise_for_status()
        data = response.json()
        if isinstance(data[0]["embedding"][0], list):
            return np.array(data[0]["embedding"])
        return np.array([data[0]["embedding"]])

    def tokenize(self, text: str) -> list[str]:
        """Gibt die Token-Texte zurück."""
        text = text[:self.max_chars]
        # Tokenize
        resp = requests.post(f"{self.url}/tokenize", json={"content": text})
        token_ids = resp.json().get("tokens", [])

        # Detokenize each
        tokens = []
        for tok_id in token_ids:
            resp = requests.post(f"{self.url}/detokenize", json={"tokens": [tok_id]})
            tok_text = resp.json().get("content", "?")
            tokens.append(tok_text.strip() or "·")  # Space als Punkt
        return tokens

    def encode_with_tokens(self, text: str) -> tuple[np.ndarray, list[str]]:
        """Gibt Embeddings und Token-Texte zurück."""
        embeddings = self.encode(text)
        tokens = self.tokenize(text)
        return embeddings, tokens


# =============================================================================
# AUDIO MODEL (Option B)
# =============================================================================

# Matryoshka-Bänder: (start_dim, end_dim, freq_center, freq_range)
# Jedes Band hat ±500 Hz um den Mittelpunkt
BANDS = [
    (0, 256, 4500, 500),       # Band 1: Grobe Semantik (4000-5000 Hz)
    (256, 512, 3500, 500),     # Band 2: Mittlere Details (3000-4000 Hz)
    (512, 768, 2500, 500),     # Band 3: Feine Nuancen (2000-3000 Hz)
    (768, 1024, 1500, 500),    # Band 4: Micro-Features (1000-2000 Hz)
]

# Typischer Wertebereich für jina-v3 Embeddings
VALUE_CLIP = 3.0  # Clip bei ±3 (deckt >99% der Werte ab)


def embedding_to_frequencies(embedding: np.ndarray,
                             use_attention: bool = False,
                             attention_weights: np.ndarray = None,
                             transform: str = "linear") -> tuple[np.ndarray, np.ndarray]:
    """
    Konvertiert ein Token-Embedding zu Frequenzen und Amplituden.

    Option B Mapping:
    - Wert bestimmt Frequenz-Position im Band (0 = Mitte, ±max = Rand)
    - Amplitude = Attention Weight (oder konstant 1)

    Args:
        embedding: 1D array mit 1024 Werten (typisch -3 bis +3)
        use_attention: Ob Attention Weights für Amplitude verwendet werden
        attention_weights: Optional, 1D array mit Attention Weights
        transform: "linear" oder "sqrt" für Wert-Transformation

    Returns:
        frequencies: 1D array mit 1024 Frequenzen (Hz)
        amplitudes: 1D array mit 1024 Amplituden (0 bis 1)
    """
    n_dims = len(embedding)
    frequencies = np.zeros(n_dims)
    amplitudes = np.ones(n_dims)  # Konstant 1 als Default

    for start, end, freq_center, freq_range in BANDS:
        if start >= n_dims:
            break
        actual_end = min(end, n_dims)

        values = embedding[start:actual_end]

        # Clip auf typischen Wertebereich
        clipped = np.clip(values, -VALUE_CLIP, VALUE_CLIP)

        # Normalisiere auf [-1, +1]
        normalized = clipped / VALUE_CLIP

        # Optional: sqrt-Transformation (spreizt mittlere Werte)
        if transform == "sqrt":
            # sign-erhaltende Wurzel: sign(x) * sqrt(|x|)
            transformed = np.sign(normalized) * np.sqrt(np.abs(normalized))
        else:
            transformed = normalized

        # Wert → Frequenz: 0 = Mitte, ±1 = Rand
        frequencies[start:actual_end] = freq_center + transformed * freq_range

        # Amplitude = Attention Weight (wenn verfügbar) oder konstant 1
        if use_attention and attention_weights is not None:
            amplitudes[start:actual_end] = attention_weights[start:actual_end]

    return frequencies, amplitudes


def compute_spectrum(frequencies: np.ndarray, amplitudes: np.ndarray,
                     freq_bins: np.ndarray) -> np.ndarray:
    """
    Berechnet das Spektrum aus Frequenzen und Amplituden.

    Jede Dimension trägt mit ihrer Amplitude zur nächsten Frequenz-Bin bei.
    """
    spectrum = np.zeros(len(freq_bins) - 1)

    for freq, amp in zip(frequencies, amplitudes):
        # Finde die richtige Bin
        bin_idx = np.searchsorted(freq_bins, freq) - 1
        if 0 <= bin_idx < len(spectrum):
            spectrum[bin_idx] += amp

    return spectrum


def embeddings_to_spectrogram(embeddings: np.ndarray,
                               n_freq_bins: int = 200,
                               transform: str = "linear") -> tuple[np.ndarray, np.ndarray]:
    """
    Konvertiert eine Sequenz von Embeddings zu einem Spektrogramm.

    Returns:
        spectrogram: 2D array (n_freq_bins × n_tokens)
        freq_bins: 1D array mit Frequenz-Grenzen
    """
    n_tokens = len(embeddings)

    # Frequenz-Bins von 1000 Hz bis 5000 Hz
    freq_bins = np.linspace(1000, 5000, n_freq_bins + 1)

    spectrogram = np.zeros((n_freq_bins, n_tokens))

    for t, emb in enumerate(embeddings):
        freqs, amps = embedding_to_frequencies(emb, transform=transform)
        spectrum = compute_spectrum(freqs, amps, freq_bins)
        spectrogram[:, t] = spectrum

    return spectrogram, freq_bins


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_audio_model(embeddings: np.ndarray, text: str = None,
                     save_path: str = None):
    """
    Visualisiert das Audio-Modell (Option B).
    """
    n_tokens, n_dims = embeddings.shape

    # Spektrogramm berechnen
    spectrogram, freq_bins = embeddings_to_spectrogram(embeddings)
    freq_centers = (freq_bins[:-1] + freq_bins[1:]) / 2

    # Für einzelne Token-Analyse: erstes, mittleres, letztes Token
    sample_indices = [0, n_tokens // 2, n_tokens - 1]

    # Figure Setup
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[2, 1, 1])

    # 1. Spektrogramm (Top, spans all columns)
    ax1 = fig.add_subplot(gs[0, :])

    # Log-Skala für bessere Sichtbarkeit
    spec_plot = spectrogram.copy()
    spec_plot[spec_plot < 0.01] = 0.01  # Minimum für Log-Skala

    im = ax1.imshow(
        spec_plot,
        aspect='auto',
        origin='lower',
        cmap='magma',
        extent=[0, n_tokens, freq_bins[0], freq_bins[-1]],
        norm=LogNorm(vmin=0.01, vmax=spec_plot.max())
    )

    # Band-Grenzen einzeichnen
    for _, _, freq_low, freq_high in BANDS:
        ax1.axhline(y=freq_low, color='white', linewidth=0.5, alpha=0.5)

    ax1.set_xlabel('Token Position')
    ax1.set_ylabel('Frequency (Hz)')
    ax1.set_title('Semantic Spectrogram (Option B: Value → Frequency, |Value| → Amplitude)')
    plt.colorbar(im, ax=ax1, label='Amplitude (log)')

    # Band-Labels
    band_labels = ['Band 4\n(Micro)', 'Band 3\n(Fine)', 'Band 2\n(Medium)', 'Band 1\n(Coarse)']
    for i, (_, _, freq_low, freq_high) in enumerate(BANDS):
        ax1.text(n_tokens + 1, (freq_low + freq_high) / 2, band_labels[i],
                 fontsize=8, va='center')

    # 2. Einzelne Token-Spektren (Middle Row)
    for i, idx in enumerate(sample_indices):
        ax = fig.add_subplot(gs[1, i])

        freqs, amps = embedding_to_frequencies(embeddings[idx])
        values = embeddings[idx]

        # Scatter: jede Dimension als Punkt
        # Farbe = Vorzeichen (rot = negativ, blau = positiv)
        colors = np.where(values >= 0, 'steelblue', 'coral')

        for dim_start, dim_end, _, _ in BANDS:
            if dim_start >= n_dims:
                break
            actual_end = min(dim_end, n_dims)
            ax.scatter(freqs[dim_start:actual_end],
                      amps[dim_start:actual_end],
                      c=colors[dim_start:actual_end],
                      s=3, alpha=0.6)

        ax.set_xlim(900, 5100)
        ax.set_ylim(0, 1.2)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'Token {idx} (blue=+, red=-)')

        # Band-Grenzen
        for _, _, freq_low, freq_high in BANDS:
            ax.axvline(x=freq_low, color='gray', linewidth=0.5, alpha=0.5)
            ax.axvline(x=freq_high, color='gray', linewidth=0.5, alpha=0.5)

    # 3. Amplitude-Histogramm pro Band (Bottom Left)
    ax3 = fig.add_subplot(gs[2, 0])

    band_means = []
    band_names = ['B1 (4-5kHz)', 'B2 (3-4kHz)', 'B3 (2-3kHz)', 'B4 (1-2kHz)']

    for start, end, _, _ in BANDS:
        if start < n_dims:
            actual_end = min(end, n_dims)
            mean_amp = np.abs(embeddings[:, start:actual_end]).mean()
            band_means.append(mean_amp)
        else:
            band_means.append(0)

    ax3.bar(band_names, band_means, color=['#e74c3c', '#f39c12', '#27ae60', '#3498db'])
    ax3.set_ylabel('Mean Amplitude')
    ax3.set_title('Activity per Band')
    ax3.tick_params(axis='x', rotation=45)

    # 4. Frequenz-Verteilung über Zeit (Bottom Middle)
    ax4 = fig.add_subplot(gs[2, 1])

    # Gewichteter Frequenz-Schwerpunkt pro Token
    freq_centroids = []
    for t in range(n_tokens):
        freqs, amps = embedding_to_frequencies(embeddings[t])
        if amps.sum() > 0:
            centroid = np.average(freqs, weights=amps)
        else:
            centroid = 3000  # Mitte
        freq_centroids.append(centroid)

    ax4.plot(freq_centroids, color='purple', linewidth=1)
    ax4.fill_between(range(n_tokens), 1000, freq_centroids, alpha=0.3, color='purple')
    ax4.set_xlim(0, n_tokens)
    ax4.set_ylim(1000, 5000)
    ax4.set_xlabel('Token Position')
    ax4.set_ylabel('Frequency Centroid (Hz)')
    ax4.set_title('Semantic "Melody" (Frequency Center over Time)')

    # 5. Stats (Bottom Right)
    ax5 = fig.add_subplot(gs[2, 2])
    ax5.axis('off')

    # Statistiken
    total_energy = np.abs(embeddings).sum()
    band_energies = []
    for start, end, _, _ in BANDS:
        if start < n_dims:
            actual_end = min(end, n_dims)
            energy = np.abs(embeddings[:, start:actual_end]).sum()
            band_energies.append(energy / total_energy * 100)
        else:
            band_energies.append(0)

    stats_text = f"""
    Audio Model Statistics
    ──────────────────────────
    Tokens: {n_tokens}
    Dimensions: {n_dims}

    Energy Distribution:
    ──────────────────────────
    Band 1 (4-5 kHz): {band_energies[0]:.1f}%
    Band 2 (3-4 kHz): {band_energies[1]:.1f}%
    Band 3 (2-3 kHz): {band_energies[2]:.1f}%
    Band 4 (1-2 kHz): {band_energies[3]:.1f}%

    Frequency Centroid:
    ──────────────────────────
    Mean: {np.mean(freq_centroids):.0f} Hz
    Std:  {np.std(freq_centroids):.0f} Hz
    """

    ax5.text(0.1, 0.9, stats_text, transform=ax5.transAxes,
             fontfamily='monospace', fontsize=10, verticalalignment='top')

    # Title
    if text:
        preview = text[:80].replace('\n', ' ') + '...' if len(text) > 80 else text
        fig.suptitle(f'S3 Audio Model Visualization\n"{preview}"', fontsize=12)
    else:
        fig.suptitle('S3 Audio Model Visualization', fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()

    return {
        'spectrogram': spectrogram,
        'freq_bins': freq_bins,
        'freq_centroids': freq_centroids,
        'band_energies': band_energies
    }


# =============================================================================
# COMPARISON
# =============================================================================

def compare_texts(texts: list[str], labels: list[str], encoder: TokenEncoder,
                  save_path: str = None):
    """
    Vergleicht die Audio-Spektren mehrerer Texte.
    """
    n_texts = len(texts)
    fig, axes = plt.subplots(n_texts, 1, figsize=(14, 3 * n_texts))
    if n_texts == 1:
        axes = [axes]

    for i, (text, label) in enumerate(zip(texts, labels)):
        embeddings = encoder.encode(text)
        spectrogram, freq_bins = embeddings_to_spectrogram(embeddings)

        spec_plot = spectrogram.copy()
        spec_plot[spec_plot < 0.01] = 0.01

        im = axes[i].imshow(
            spec_plot,
            aspect='auto',
            origin='lower',
            cmap='magma',
            extent=[0, len(embeddings), freq_bins[0], freq_bins[-1]],
            norm=LogNorm(vmin=0.01, vmax=spec_plot.max())
        )

        axes[i].set_ylabel('Freq (Hz)')
        axes[i].set_title(f'{label} ({len(embeddings)} tokens)')

        # Band-Grenzen
        for _, _, freq_low, _ in BANDS:
            axes[i].axhline(y=freq_low, color='white', linewidth=0.5, alpha=0.3)

    axes[-1].set_xlabel('Token Position')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()


# =============================================================================
# DEMO
# =============================================================================

def demo(save_path: str = None):
    """Demo mit Beispieltext."""
    encoder = TokenEncoder()

    text = """
    Machine learning is transforming how we process and understand data.
    Neural networks can now recognize images, translate languages, and
    generate creative content with remarkable accuracy.

    The stock market showed unusual volatility yesterday. Major indices
    dropped sharply in morning trading before recovering by close.

    In space exploration news, NASA announced a new mission to Europa.
    Scientists believe the icy moon may harbor conditions for microbial life.
    """

    print("Encoding text...")
    embeddings = encoder.encode(text.strip())
    print(f"Got {len(embeddings)} token embeddings with {embeddings.shape[1]} dimensions")

    print("\nGenerating audio model visualization...")
    stats = plot_audio_model(embeddings, text=text.strip(), save_path=save_path)

    print(f"\nBand energies: {stats['band_energies']}")


def demo_compare():
    """Vergleicht verschiedene Texttypen."""
    encoder = TokenEncoder()

    texts = [
        "The algorithm implements a divide-and-conquer approach with O(n log n) complexity.",
        "I can't believe this happened! After years of hard work, we finally achieved our dream!",
        "Apple announced the new iPhone today. The stock price rose by 3%.",
    ]
    labels = ["Technical", "Emotional", "Mixed (Tech + Finance)"]

    print("Comparing text types...")
    compare_texts(texts, labels, encoder)


def plot_with_tokens(text: str, save_path: str = None):
    """Spektrogramm mit Token-Labels auf der X-Achse."""
    encoder = TokenEncoder()

    print(f"Encoding: {text[:60]}...")
    embeddings, tokens = encoder.encode_with_tokens(text)
    n_embeddings = len(embeddings)
    n_tokens = len(tokens)

    # Mismatch handling (BOS/EOS tokens)
    if n_embeddings != n_tokens:
        print(f"  Note: {n_embeddings} embeddings vs {n_tokens} tokens (special tokens?)")
        # Truncate to smaller length
        n = min(n_embeddings, n_tokens)
        embeddings = embeddings[:n]
        tokens = tokens[:n]
        n_tokens = n

    # Spectrogram berechnen
    spectrogram, freq_bins = embeddings_to_spectrogram(embeddings, transform="linear")

    fig, ax = plt.subplots(figsize=(max(14, n_tokens * 0.3), 8))

    spec_plot = spectrogram.copy()
    spec_plot[spec_plot < 0.01] = 0.01

    im = ax.imshow(
        spec_plot, aspect='auto', origin='lower', cmap='magma',
        extent=[0, n_tokens, 1000, 5000],
        norm=LogNorm(vmin=0.01, vmax=spec_plot.max())
    )

    # Band-Grenzen
    for _, _, freq_center, freq_range in BANDS:
        ax.axhline(y=freq_center - freq_range, color='white', lw=0.5, alpha=0.3)
        ax.axhline(y=freq_center + freq_range, color='white', lw=0.5, alpha=0.3)

    # Token-Labels auf X-Achse
    ax.set_xticks(np.arange(n_tokens) + 0.5)
    ax.set_xticklabels(tokens, rotation=90, fontsize=8, fontfamily='monospace')

    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Tokens')
    ax.set_title(f'Semantic Spectrogram with Token Labels\n({n_tokens} tokens)')

    plt.colorbar(im, ax=ax, label='Amplitude (log)')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()


def compare_transforms(text: str = None, save_path: str = None):
    """Vergleicht Linear vs Sqrt Transformation side-by-side."""
    encoder = TokenEncoder()

    if text is None:
        text = "Apple and Banana are fruits. Microsoft and Google are tech companies."

    print(f"Comparing transforms for: {text[:60]}...")
    embeddings = encoder.encode(text)
    n_tokens, n_dims = embeddings.shape

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    transforms = ["linear", "sqrt"]
    transform_labels = ["Linear", "Sqrt (spreads middle values)"]

    for row, (transform, label) in enumerate(zip(transforms, transform_labels)):
        # Spektrogramm
        ax_spec = axes[row, 0]
        spectrogram = np.zeros((200, n_tokens))
        freq_bins = np.linspace(1000, 5000, 201)

        for t, emb in enumerate(embeddings):
            freqs, amps = embedding_to_frequencies(emb, transform=transform)
            spectrum = compute_spectrum(freqs, amps, freq_bins)
            spectrogram[:, t] = spectrum

        spec_plot = spectrogram.copy()
        spec_plot[spec_plot < 0.01] = 0.01

        im = ax_spec.imshow(
            spec_plot, aspect='auto', origin='lower', cmap='magma',
            extent=[0, n_tokens, 1000, 5000],
            norm=LogNorm(vmin=0.01, vmax=spec_plot.max())
        )
        ax_spec.set_ylabel('Frequency (Hz)')
        ax_spec.set_xlabel('Token')
        ax_spec.set_title(f'{label}: Spectrogram')

        # Band-Grenzen
        for _, _, freq_center, freq_range in BANDS:
            ax_spec.axhline(y=freq_center - freq_range, color='white', lw=0.5, alpha=0.3)
            ax_spec.axhline(y=freq_center + freq_range, color='white', lw=0.5, alpha=0.3)

        # Token 0 Scatter
        ax_t0 = axes[row, 1]
        freqs, amps = embedding_to_frequencies(embeddings[0], transform=transform)
        values = embeddings[0]
        colors = np.where(values >= 0, 'steelblue', 'coral')
        ax_t0.scatter(freqs, amps, c=colors, s=3, alpha=0.6)
        ax_t0.set_xlim(900, 5100)
        ax_t0.set_ylim(0, 1.2)
        ax_t0.set_xlabel('Frequency (Hz)')
        ax_t0.set_ylabel('Amplitude')
        ax_t0.set_title(f'{label}: Token 0')

        # Frequenz-Histogramm (alle Tokens)
        ax_hist = axes[row, 2]
        all_freqs = []
        for emb in embeddings:
            freqs, _ = embedding_to_frequencies(emb, transform=transform)
            all_freqs.extend(freqs)

        ax_hist.hist(all_freqs, bins=100, color='purple', alpha=0.7, edgecolor='none')
        ax_hist.set_xlabel('Frequency (Hz)')
        ax_hist.set_ylabel('Count')
        ax_hist.set_title(f'{label}: Frequency Distribution')

        # Band-Grenzen
        for _, _, freq_center, freq_range in BANDS:
            ax_hist.axvline(x=freq_center, color='red', lw=1, alpha=0.5)

    plt.suptitle(f'Linear vs Sqrt Transform Comparison\n"{text[:80]}..."', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='S3 Audio Model Visualization')
    parser.add_argument('--text', type=str, help='Text to analyze')
    parser.add_argument('--file', type=str, help='File to analyze')
    parser.add_argument('--demo', action='store_true', help='Run demo')
    parser.add_argument('--compare', action='store_true', help='Run comparison demo')
    parser.add_argument('--transform-compare', action='store_true',
                        help='Compare linear vs sqrt transform')
    parser.add_argument('--with-tokens', action='store_true',
                        help='Show spectrogram with token labels')
    parser.add_argument('--save', type=str, help='Save plot to file')
    parser.add_argument('--url', type=str, default='http://localhost:8202',
                        help='Token encoder URL')

    args = parser.parse_args()

    if args.with_tokens and args.text:
        plot_with_tokens(args.text, save_path=args.save)
    elif args.transform_compare:
        compare_transforms(text=args.text, save_path=args.save)
    elif args.demo:
        demo(save_path=args.save)
    elif args.compare:
        demo_compare()
    elif args.text:
        encoder = TokenEncoder(args.url)
        embeddings = encoder.encode(args.text)
        plot_audio_model(embeddings, text=args.text, save_path=args.save)
    elif args.file:
        with open(args.file, 'r') as f:
            text = f.read()
        encoder = TokenEncoder(args.url)
        embeddings = encoder.encode(text)
        plot_audio_model(embeddings, text=text, save_path=args.save)
    else:
        demo()


if __name__ == "__main__":
    main()
