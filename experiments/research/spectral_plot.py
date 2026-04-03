#!/usr/bin/env python3
"""
Spektrogramm-Visualisierung für S3

Visualisiert Embedding-Kurven als Spektrogramme:
1. Embedding-Heatmap (Token × Dimension)
2. Spectral Flux Kurve (Onset-Signal)
3. Dimension-Activity (welche Dims ändern sich?)
4. PCA-Trajectory (Token-Pfad im reduzierten Raum)
"""

import numpy as np
import requests
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
import argparse


# =============================================================================
# ENCODER
# =============================================================================

class TokenEncoder:
    def __init__(self, url: str = "http://localhost:8202", max_chars: int = 6000):
        self.url = url
        self.max_chars = max_chars  # ~1500-2000 tokens depending on text

    def encode(self, text: str) -> np.ndarray:
        text = text[:self.max_chars]
        try:
            response = requests.post(f"{self.url}/embeddings", json={"input": text})
            response.raise_for_status()
        except Exception as e:
            # If still too long, try shorter
            print(f"  Warning: Text too long, truncating further...")
            text = text[:self.max_chars // 2]
            response = requests.post(f"{self.url}/embeddings", json={"input": text})
            response.raise_for_status()

        data = response.json()
        if isinstance(data[0]["embedding"][0], list):
            return np.array(data[0]["embedding"])
        return np.array([data[0]["embedding"]])


# =============================================================================
# ONSET DETECTION
# =============================================================================

def spectral_flux(embeddings: np.ndarray) -> np.ndarray:
    """Summe der absoluten Änderungen über alle Dimensionen."""
    changes = np.abs(np.diff(embeddings, axis=0))
    return changes.sum(axis=1)


def find_onsets(
    onset_signal: np.ndarray,
    threshold_pct: float = 95,
    min_dist: int = 3,
    smooth_sigma: float = 2.0
) -> np.ndarray:
    if len(onset_signal) < 3:
        return np.array([])
    smoothed = gaussian_filter1d(onset_signal, sigma=smooth_sigma)
    threshold = np.percentile(smoothed, threshold_pct)
    peaks, _ = find_peaks(smoothed, height=threshold, distance=min_dist)
    return peaks


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_spectral_analysis(
    embeddings: np.ndarray,
    text: str = None,
    title: str = "S3 Spectral Analysis",
    save_path: str = None,
    show_dims: int = 100  # Nur erste N Dimensionen zeigen
):
    """
    Erstelle 4-Panel Spektrogramm-Analyse.
    """
    n_tokens, n_dims = embeddings.shape

    # Onset Detection
    flux = spectral_flux(embeddings)
    flux_smoothed = gaussian_filter1d(flux, sigma=2.0)
    onsets = find_onsets(flux)

    # Dimension Activity (wie viel ändert sich jede Dimension?)
    dim_activity = np.abs(np.diff(embeddings, axis=0)).mean(axis=0)

    # PCA für Trajectory
    pca = PCA(n_components=2)
    tokens_2d = pca.fit_transform(embeddings)

    # Figure Setup
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[2, 1, 1])

    # 1. Embedding Heatmap (Top, spans both columns)
    ax1 = fig.add_subplot(gs[0, :])
    im = ax1.imshow(
        embeddings[:, :show_dims].T,
        aspect='auto',
        cmap='RdBu_r',
        interpolation='nearest',
        vmin=-np.percentile(np.abs(embeddings), 95),
        vmax=np.percentile(np.abs(embeddings), 95)
    )
    ax1.set_xlabel('Token Position')
    ax1.set_ylabel(f'Dimension (first {show_dims})')
    ax1.set_title('Embedding Heatmap')

    # Onset-Linien einzeichnen
    for onset in onsets:
        ax1.axvline(x=onset, color='lime', linewidth=1, alpha=0.7)

    plt.colorbar(im, ax=ax1, label='Activation')

    # 2. Spectral Flux (Middle Left)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(flux, alpha=0.5, label='Raw Flux', color='blue')
    ax2.plot(flux_smoothed, label='Smoothed', color='darkblue', linewidth=2)
    ax2.axhline(y=np.percentile(flux_smoothed, 95), color='red',
                linestyle='--', label='Threshold (95%)')

    for onset in onsets:
        ax2.axvline(x=onset, color='lime', linewidth=2, alpha=0.7)

    ax2.scatter(onsets, flux_smoothed[onsets], color='lime', s=100,
                zorder=5, label=f'Onsets ({len(onsets)})')

    ax2.set_xlabel('Token Position')
    ax2.set_ylabel('Spectral Flux')
    ax2.set_title('Onset Detection Signal')
    ax2.legend(loc='upper right')
    ax2.set_xlim(0, len(flux))

    # 3. Dimension Activity (Middle Right)
    ax3 = fig.add_subplot(gs[1, 1])
    top_dims = np.argsort(dim_activity)[-20:]  # Top 20 aktivste Dimensionen
    ax3.barh(range(20), dim_activity[top_dims], color='steelblue')
    ax3.set_yticks(range(20))
    ax3.set_yticklabels([f'Dim {d}' for d in top_dims])
    ax3.set_xlabel('Average Change')
    ax3.set_title('Most Active Dimensions')

    # 4. PCA Trajectory (Bottom Left)
    ax4 = fig.add_subplot(gs[2, 0])

    # Farbverlauf für Token-Position
    colors = plt.cm.viridis(np.linspace(0, 1, n_tokens))

    # Linien zwischen Punkten
    for i in range(n_tokens - 1):
        ax4.plot(tokens_2d[i:i+2, 0], tokens_2d[i:i+2, 1],
                color=colors[i], linewidth=1, alpha=0.5)

    # Punkte
    scatter = ax4.scatter(tokens_2d[:, 0], tokens_2d[:, 1],
                         c=range(n_tokens), cmap='viridis', s=20, alpha=0.7)

    # Onset-Punkte hervorheben
    if len(onsets) > 0:
        ax4.scatter(tokens_2d[onsets, 0], tokens_2d[onsets, 1],
                   color='red', s=100, marker='x', linewidths=2,
                   label='Onsets', zorder=5)

    # Start und Ende markieren
    ax4.scatter(tokens_2d[0, 0], tokens_2d[0, 1], color='green',
               s=150, marker='o', label='Start', zorder=6)
    ax4.scatter(tokens_2d[-1, 0], tokens_2d[-1, 1], color='red',
               s=150, marker='s', label='End', zorder=6)

    ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax4.set_title('Token Trajectory (PCA)')
    ax4.legend(loc='upper right')

    # 5. Segment Info (Bottom Right)
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')

    # Statistiken
    boundaries = [0] + sorted(onsets.tolist()) + [n_tokens]
    segment_sizes = [boundaries[i+1] - boundaries[i] for i in range(len(boundaries)-1)]

    stats_text = f"""
    Document Statistics
    ───────────────────
    Total Tokens: {n_tokens}
    Dimensions: {n_dims}

    Onset Detection
    ───────────────────
    Onsets Found: {len(onsets)}
    Segments: {len(segment_sizes)}
    Avg Segment Size: {np.mean(segment_sizes):.1f} tokens
    Min/Max Segment: {min(segment_sizes)}/{max(segment_sizes)} tokens

    Embedding Stats
    ───────────────────
    Mean Activation: {embeddings.mean():.4f}
    Std Activation: {embeddings.std():.4f}
    Active Dims (>0.1): {(dim_activity > 0.1).sum()}
    """

    ax5.text(0.1, 0.9, stats_text, transform=ax5.transAxes,
            fontfamily='monospace', fontsize=10, verticalalignment='top')

    # Title
    if text:
        preview = text[:100].replace('\n', ' ') + '...' if len(text) > 100 else text
        fig.suptitle(f'{title}\n"{preview}"', fontsize=12)
    else:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()

    return {
        'n_tokens': n_tokens,
        'n_dims': n_dims,
        'n_onsets': len(onsets),
        'onset_positions': onsets,
        'segment_sizes': segment_sizes,
        'dim_activity': dim_activity,
        'pca_variance': pca.explained_variance_ratio_
    }


def plot_comparison(
    texts: list[str],
    encoder: TokenEncoder,
    titles: list[str] = None,
    save_path: str = None
):
    """
    Vergleiche Spectral Flux mehrerer Texte.
    """
    if titles is None:
        titles = [f"Text {i+1}" for i in range(len(texts))]

    fig, axes = plt.subplots(len(texts), 1, figsize=(14, 3*len(texts)), sharex=False)
    if len(texts) == 1:
        axes = [axes]

    for i, (text, title) in enumerate(zip(texts, titles)):
        embeddings = encoder.encode(text)
        flux = spectral_flux(embeddings)
        flux_smoothed = gaussian_filter1d(flux, sigma=2.0)
        onsets = find_onsets(flux)

        ax = axes[i]
        ax.plot(flux_smoothed, color='darkblue', linewidth=1.5)
        ax.fill_between(range(len(flux_smoothed)), flux_smoothed, alpha=0.3)

        for onset in onsets:
            ax.axvline(x=onset, color='red', linewidth=1, alpha=0.7)

        ax.set_ylabel('Spectral Flux')
        ax.set_title(f'{title} ({len(embeddings)} tokens, {len(onsets)} onsets)')
        ax.set_xlim(0, len(flux_smoothed))

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

    # Text mit klaren Themenwechseln
    text = """
    Machine learning is transforming how we process and understand data.
    Neural networks can now recognize images, translate languages, and
    generate creative content with remarkable accuracy.

    The stock market showed unusual volatility yesterday. Major indices
    dropped sharply in morning trading before recovering by close. Analysts
    pointed to concerns about interest rates and inflation expectations.

    In space exploration news, NASA announced a new mission to Europa.
    Scientists believe the icy moon of Jupiter may harbor conditions
    suitable for microbial life beneath its frozen surface.

    Climate researchers published alarming findings about Arctic ice loss.
    The rate of melting has accelerated beyond previous predictions, with
    significant implications for global sea levels and weather patterns.
    """

    print("Encoding text...")
    embeddings = encoder.encode(text.strip())
    print(f"Got {len(embeddings)} token embeddings")

    print("\nGenerating spectral analysis...")
    stats = plot_spectral_analysis(
        embeddings,
        text=text.strip(),
        title="S3 Spectral Analysis - Multi-Topic Document",
        save_path=save_path
    )

    print(f"\nStats: {stats['n_onsets']} onsets, {len(stats['segment_sizes'])} segments")
    print(f"Segment sizes: {stats['segment_sizes']}")


def demo_comparison():
    """Vergleiche verschiedene Texttypen."""

    encoder = TokenEncoder()

    texts = [
        # Technischer Text
        """The algorithm implements a divide-and-conquer approach with O(n log n)
        complexity. First, the input array is recursively split into subarrays.
        Then, the sorted subarrays are merged back together.""",

        # Emotionaler Text
        """I can't believe this happened! After years of hard work and dedication,
        we finally achieved our dream. The joy and relief we felt was overwhelming.
        Tears of happiness streamed down our faces.""",

        # Dialog/Konversation
        """Hey, how are you? - I'm good, thanks! How about you? - Pretty tired actually.
        Work has been crazy lately. - I know what you mean. Want to grab coffee?
        - Sure, that sounds great! Let's go."""
    ]

    titles = ["Technical", "Emotional", "Conversational"]

    print("Encoding texts...")
    plot_comparison(texts, encoder, titles)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='S3 Spectral Visualization')
    parser.add_argument('--text', type=str, help='Text to analyze')
    parser.add_argument('--file', type=str, help='File to analyze')
    parser.add_argument('--demo', action='store_true', help='Run demo')
    parser.add_argument('--compare', action='store_true', help='Run comparison demo')
    parser.add_argument('--save', type=str, help='Save plot to file')
    parser.add_argument('--url', type=str, default='http://localhost:8202',
                       help='Token encoder URL')

    args = parser.parse_args()

    if args.demo:
        demo(save_path=args.save)
    elif args.compare:
        demo_comparison()
    elif args.text:
        encoder = TokenEncoder(args.url)
        embeddings = encoder.encode(args.text)
        plot_spectral_analysis(embeddings, text=args.text, save_path=args.save)
    elif args.file:
        with open(args.file, 'r') as f:
            text = f.read()
        encoder = TokenEncoder(args.url)
        embeddings = encoder.encode(text)
        plot_spectral_analysis(embeddings, text=text, save_path=args.save)
    else:
        demo()


if __name__ == "__main__":
    main()
