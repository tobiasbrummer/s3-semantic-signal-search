#!/usr/bin/env python3
"""Quick debug: Was sind die tatsächlichen Wert-Verteilungen in Embeddings?"""

import numpy as np
import requests
import matplotlib.pyplot as plt

def encode(text: str, url: str = "http://localhost:8202") -> np.ndarray:
    response = requests.post(f"{url}/embeddings", json={"input": text})
    response.raise_for_status()
    data = response.json()
    if isinstance(data[0]["embedding"][0], list):
        return np.array(data[0]["embedding"])
    return np.array([data[0]["embedding"]])

# Test verschiedene Tokens
texts = [
    "Apple",           # Frucht/Tech
    "Banana",          # Nur Frucht
    "Microsoft",       # Nur Tech
    "the",             # Stopword
    "Kernfusion",      # Wissenschaft
    "Liebe",           # Emotion
]

print("=" * 60)
print("EMBEDDING VALUE DISTRIBUTION ANALYSIS")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, text in enumerate(texts):
    emb = encode(text)[0]  # Erstes Token

    print(f"\n{text}:")
    print(f"  Shape: {emb.shape}")
    print(f"  Min:   {emb.min():.3f}")
    print(f"  Max:   {emb.max():.3f}")
    print(f"  Mean:  {emb.mean():.3f}")
    print(f"  Std:   {emb.std():.3f}")
    print(f"  |val| > 0.5: {(np.abs(emb) > 0.5).sum()} dims ({(np.abs(emb) > 0.5).mean()*100:.1f}%)")
    print(f"  |val| > 0.8: {(np.abs(emb) > 0.8).sum()} dims ({(np.abs(emb) > 0.8).mean()*100:.1f}%)")

    # Histogram
    axes[i].hist(emb, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[i].axvline(x=0, color='red', linestyle='--', linewidth=1)
    axes[i].set_title(f'"{text}" - std={emb.std():.3f}')
    axes[i].set_xlabel('Embedding Value')
    axes[i].set_ylabel('Count')
    axes[i].set_xlim(-1.5, 1.5)

plt.suptitle('Embedding Value Distribution per Token', fontsize=14)
plt.tight_layout()
plt.savefig('/var/home/t0bybr/containers/s3/embedding_distribution.png', dpi=150)
print(f"\nSaved to embedding_distribution.png")
plt.show()

# Vergleiche zwei Tokens: welche Dimensionen unterscheiden sich am meisten?
print("\n" + "=" * 60)
print("DIMENSION COMPARISON: Apple vs Banana vs Microsoft")
print("=" * 60)

apple = encode("Apple")[0]
banana = encode("Banana")[0]
microsoft = encode("Microsoft")[0]

diff_ab = np.abs(apple - banana)
diff_am = np.abs(apple - microsoft)
diff_bm = np.abs(banana - microsoft)

print(f"\nApple vs Banana:")
print(f"  Mean diff: {diff_ab.mean():.3f}")
print(f"  Max diff:  {diff_ab.max():.3f} at dim {diff_ab.argmax()}")

print(f"\nApple vs Microsoft:")
print(f"  Mean diff: {diff_am.mean():.3f}")
print(f"  Max diff:  {diff_am.max():.3f} at dim {diff_am.argmax()}")

print(f"\nBanana vs Microsoft:")
print(f"  Mean diff: {diff_bm.mean():.3f}")
print(f"  Max diff:  {diff_bm.max():.3f} at dim {diff_bm.argmax()}")

# Welche Dimensionen sind für Früchte vs Tech unterschiedlich?
# Finde Dimensionen wo Apple≈Banana aber ≠Microsoft
fruit_dims = np.where((diff_ab < 0.1) & (diff_am > 0.3))[0]
print(f"\n'Fruit' dimensions (Apple≈Banana, Apple≠Microsoft): {len(fruit_dims)}")
if len(fruit_dims) > 0:
    print(f"  First 10: {fruit_dims[:10]}")
