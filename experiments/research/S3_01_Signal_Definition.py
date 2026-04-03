
import torch
import numpy as np
import matplotlib.pyplot as plt
from FlagEmbedding import BGEM3FlagModel

# 1. SETUP
print("Lade BGE-M3 Model via FlagEmbedding...")
# use_fp16=True beschleunigt es auf GPU, auf CPU ist es egal, aber schadet nicht.
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False) 

def normalize(vec):
    norm = np.linalg.norm(vec)
    if norm == 0: return vec
    return vec / norm

def compute_sparse_score(q_lexical, d_lexical):
    """
    Berechnet den Sparse-Score basierend auf Lexical Weights (SPLADE-Style).
    Score = Summe(Produkt der Gewichte für gemeinsame Wörter)
    """
    score = 0.0
    # q_lexical ist ein Dict {token_id: weight}
    for token_id, q_weight in q_lexical.items():
        if token_id in d_lexical:
            d_weight = d_lexical[token_id]
            score += q_weight * d_weight
    return score

def compute_colbert_score(q_vecs, d_vecs):
    """
    Berechnet MaxSim (ColBERT).
    """
    # DEBUG SHAPES
    # print(f"DEBUG: Q shape {q_vecs.shape}, D shape {d_vecs.shape}")
    
    # Sim Matrix: [Q, D]
    sim_matrix = np.dot(q_vecs, d_vecs.T)
    
    # Fallback für leere Vektoren (sollte nicht passieren)
    if sim_matrix.size == 0: return 0.0
    if sim_matrix.ndim == 0: return float(sim_matrix) # Scalar case
    if sim_matrix.ndim == 1: 
        # Falls Q=1 oder D=1, ist das Ergebnis 1D.
        # Wir wollen immer Max über Doc (letzte Dimension)
        return np.max(sim_matrix)

    # Max über Doc-Dimension (axis=1) für jeden Query-Token
    max_scores = np.max(sim_matrix, axis=1)
    return np.mean(max_scores)

def get_spectrum(query, text):
    """
    Extrahiert alle 3 Signale für ein Query-Text Paar.
    """
    # Encoding (return_dense, return_sparse, return_colbert_vecs)
    # BGE-M3 encode returns a dict
    q_out = model.encode(query, return_dense=True, return_sparse=True, return_colbert_vecs=True)
    d_out = model.encode(text, return_dense=True, return_sparse=True, return_colbert_vecs=True)
    
    # 1. LOW FREQUENCY (Dense)
    # Cosine Similarity der CLS Vektoren
    dense_score = q_out['dense_vecs'] @ d_out['dense_vecs'].T
    
    # 2. HIGH FREQUENCY (Sparse)
    sparse_score = compute_sparse_score(q_out['lexical_weights'], d_out['lexical_weights'])
    # Sparse Scores sind oft > 100, wir normalisieren sie grob für den Plot
    # Ein guter Match liegt oft bei 10-30. Wir teilen durch eine Konstante zur Visualisierung.
    sparse_norm = sparse_score / 20.0 
    
    # 3. MID FREQUENCY (ColBERT)
    colbert_score = compute_colbert_score(q_out['colbert_vecs'][0], d_out['colbert_vecs'][0])
    
    return dense_score, sparse_norm, colbert_score

# ==========================================
# EXPERIMENT
# ==========================================

query = "Zinsen 5%"

test_cases = [
    ("Perfect Match", "Der Zinssatz beträgt exakt 5 Prozent."),
    ("Semantik Match (Low Freq)", "Die Gebühr für das Kapital liegt bei einem Fünfzigstel."), # Zinsen ~ Gebühr, 5% = 1/20 (schwer!)
    ("Keyword Match (High Freq)", "Zinsen sind nicht 50% sondern 5%."), # Hat die Keywords, aber Kontext falsch
    ("Mismatch", "Das Wetter ist heute schön.")
]

results = []
labels = []

print(f"\nQuery: '{query}'\n")

for label, text in test_cases:
    low, high, mid = get_spectrum(query, text)
    results.append([low, high, mid])
    labels.append(label)
    print(f"--- {label} ---")
    print(f"Text: '{text}'")
    print(f"   LOW (Dense):   {low:.4f}")
    print(f"   HIGH (Sparse): {high:.4f} (Raw: {high*20:.2f})")
    print(f"   MID (ColBERT): {mid:.4f}\n")

# PLOT
results = np.array(results)
x = np.arange(len(labels))
width = 0.25

plt.figure(figsize=(10, 6))
plt.bar(x - width, results[:, 0], width, label='Low Freq (Dense/Context)', color='blue', alpha=0.7)
plt.bar(x, results[:, 1], width, label='High Freq (Sparse/Keyword)', color='red', alpha=0.7)
plt.bar(x + width, results[:, 2], width, label='Mid Freq (ColBERT/Structure)', color='green', alpha=0.7)

plt.ylabel('Signal Strength')
plt.title(f'S3 Spectral Analysis (Query: "{query}")')
plt.xticks(x, labels)
plt.legend()
plt.grid(True, alpha=0.3)

filename = "s3_spectrum_plot.png"
plt.savefig(filename)
print(f"Plot gespeichert: {filename}")
