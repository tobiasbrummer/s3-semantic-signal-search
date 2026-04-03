import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from frankenstein_engine import FrankensteinEncoder

# Setup
if torch.backends.mps.is_available(): DEVICE = "mps"
elif torch.cuda.is_available(): DEVICE = "cuda"
else: DEVICE = "cpu"

print(f"🏗️ Lade Engine auf {DEVICE} für Visualisierung...")
engine = FrankensteinEncoder(
    "google/t5gemma-2-270m-270m", 
    "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1",
    device=DEVICE
)

def get_best_vector(text):
    """Holt das stärkste Token (Max-Pooling über Magnitude)."""
    emb, tokens = engine.encode_query(text)
    vecs = emb[0]
    
    # Max-Pooling
    norms = torch.norm(vecs, dim=1)
    max_val, max_idx = torch.max(norms, dim=0)
    
    if max_val.item() == 0:
        return None, None

    best_vec = vecs[max_idx]
    best_token = tokens[max_idx]
    return best_vec, best_token

def plot_comparison(word1, vec1, word2, vec2, similarity):
    """Erstellt eine Heatmap: Vektor 1 vs Vektor 2 vs Differenz."""
    
    # Tensor -> Numpy
    v1 = vec1.cpu().detach().numpy()
    v2 = vec2.cpu().detach().numpy()
    
    # Wir stapeln die Vektoren: Oben V1, Mitte V2, Unten Differenz
    # Die Differenz zeigt genau, WO die Konzepte abweichen.
    diff = v1 - v2
    data = np.vstack([v1, v2, diff])
    
    # Plotting
    fig, ax = plt.subplots(figsize=(15, 5))
    
    # 'coolwarm' Colormap: Blau = Negativ, Rot = Positiv, Weiß = 0
    # vmin/vmax clippen wir leicht, damit man Kontraste sieht (Ausreißer ignorieren)
    im = ax.imshow(data, aspect='auto', cmap='coolwarm', vmin=-0.5, vmax=0.5)
    
    # Labels und Style
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels([f"'{word1}'", f"'{word2}'", "DIFFERENZ"])
    
    ax.set_title(f"Vektor-Vergleich: {word1} vs {word2} (Cosine: {similarity:.4f})", fontsize=14)
    ax.set_xlabel("Vektor-Dimensionen (0-768)", fontsize=10)
    
    plt.colorbar(im, orientation='horizontal', pad=0.2, label="Aktivierungsstärke")
    plt.tight_layout()
    
    # Speichern statt nur Anzeigen (praktischer für Batch-Runs)
    filename = f"heatmap_{word1}_{word2}.png"
    plt.savefig(filename)
    print(f"    📸 Heatmap gespeichert als: {filename}")
    plt.close()

def check_and_plot(word1, word2):
    v1, t1 = get_best_vector(word1)
    v2, t2 = get_best_vector(word2)
    
    if v1 is None or v2 is None:
        print(f"❌ Fehler: Eines der Wörter ist 'stumm' (0.0).")
        return

    # Cosine berechnen
    sim = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
    
    print(f"⚔️  '{t1}' vs '{t2}' -> Cos: {sim:.4f}")
    plot_comparison(t1, v1, t2, v2, sim)
    print("-" * 40)

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    print("\n🎨 Starte Vektor-Visualisierung...")
    print("===================================")

    # 1. Der "Problembär": Warum ist Atommüll ungleich Abfall?
    check_and_plot("Atommüll", "Abfall")
    
    # 2. Der Kontext-Check: Hilft "Radioaktiver"?
    # Wir vergleichen "Atommüll" mit "Radioaktiver" (vielleicht liegen sie näher?)
    check_and_plot("Atommüll", "Radioaktiver")

    # 3. Das Synonym-Paar (Gegenprobe)
    check_and_plot("Baby", "Kind")
    
    # 4. Der Halluzinations-Check (At vs igt) - Optional, falls du es noch sehen willst
    # check_and_plot("At", "igt")
