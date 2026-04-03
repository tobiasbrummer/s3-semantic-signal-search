
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from FlagEmbedding import BGEM3FlagModel

# 1. SETUP
print("Lade BGE-M3 Model...")
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False)

def analyze_token_spectrum(query, doc_text):
    print(f"Analysiere: Q='{query}' vs D='{doc_text}'")
    
    # Encoding
    q_out = model.encode([query], return_dense=True, return_sparse=True, return_colbert_vecs=True)
    d_out = model.encode([doc_text], return_dense=True, return_sparse=True, return_colbert_vecs=True)
    
    # Tokens holen (wir brauchen die Tokens des Dokuments für die X-Achse)
    # Leider gibt encode() die Tokens nicht direkt zurück, wir müssen tokenizen
    # Wir nutzen den internen Tokenizer des Modells
    tokens = model.tokenizer.tokenize(doc_text)
    
    # ---------------------------------------------------------
    # 1. HIGH FREQ (Sparse / Lexical Weights)
    # ---------------------------------------------------------
    # Das ist ein Dict {token_id: weight}. Wir müssen es auf die Token-Liste mappen.
    # Achtung: Lexical Weights sind oft auf WordPiece-IDs, die Tokenizer gibt Tokens.
    # Wir iterieren über die IDs des Tokenizers.
    enc = model.tokenizer(doc_text, return_tensors='pt')
    input_ids = enc['input_ids'][0].tolist()
    # Entferne CLS/SEP für den Plot
    content_ids = input_ids[1:-1]
    content_tokens = model.tokenizer.convert_ids_to_tokens(content_ids)
    
    sparse_weights = []
    d_lexical = d_out['lexical_weights'][0] # Dict str(id) -> float
    
    for tid in content_ids:
        w = d_lexical.get(str(tid), 0.0)
        sparse_weights.append(w)
        
    sparse_weights = np.array(sparse_weights)

    # ---------------------------------------------------------
    # 2. MID FREQ (Multi-Vector / ColBERT)
    # ---------------------------------------------------------
    # Wir wollen wissen: Wie stark matched JEDES Token im Dok zur Query?
    # ColBERT Score für Token T = max_sim(T, Query_Vectors)
    
    q_vecs = q_out['colbert_vecs'][0] # [Q_len, Dim]
    d_vecs = d_out['colbert_vecs'][0] # [D_len, Dim]
    
    # Achtung: d_vecs enthält CLS und SEP. Wir schneiden sie weg, passend zu content_tokens
    # Normalerweise ist d_vecs so lang wie input_ids
    d_vecs_content = d_vecs[1:len(content_ids)+1] 
    
    # Sim Matrix [Content_Len, Q_Len]
    sim_matrix = d_vecs_content @ q_vecs.T 
    
    # Für jedes Doc-Token: Was ist sein bester Match in der Query?
    colbert_impact = np.max(sim_matrix, axis=1) # [Content_Len]

    # ---------------------------------------------------------
    # 3. LOW FREQ (Dense / Attention)
    # ---------------------------------------------------------
    # Dense ist ein globaler Vektor. Um ihn auf Tokens zu mappen, bräuchten wir Attention Weights.
    # BGE-M3 gibt keine Attention Weights direkt zurück via encode().
    # Workaround: Wir nehmen an, dass 'Dense' eine Grundlast ist, aber modellieren
    # die "Gesamtlautstärke" (Attention) simuliert durch die Norm der ColBERT Vektoren?
    # Oder wir lassen es flat.
    # Besser: Wir nutzen die ColBERT-Vektoren-Norm als Proxy für "Wichtigkeit"
    # oder einfach eine Konstante für den Plot (der "Teppich").
    
    # Idee: Wir nehmen den globalen Dense Similarity Score und verteilen ihn gleichmäßig?
    # Nein, langweilig.
    # Wir nehmen stattdessen einfach 0.3 als Basiswert, um den "Bass" anzudeuten.
    dense_sim = q_out['dense_vecs'] @ d_out['dense_vecs'].T
    dense_line = np.full(len(content_tokens), dense_sim.item())

    return content_tokens, sparse_weights, colbert_impact, dense_line

# TEST
query = "Zinsen 5%"
doc = "Der Zinssatz beträgt exakt 5 Prozent."

tokens, sparse, multi, dense = analyze_token_spectrum(query, doc)

# DATA MATRIX FOR HEATMAP
# Rows: Sparse, Multi, Dense
data = np.vstack([sparse, multi, dense])

# PLOT
plt.figure(figsize=(12, 5))
# Wir nutzen Seaborn für schönere Heatmaps
sns.heatmap(data, annot=True, fmt=".2f", xticklabels=tokens, 
            yticklabels=["Sparse (Lexical)", "Multi (ColBERT)", "Dense (Context)"],
            cmap="viridis", cbar_kws={'label': 'Signal Strength'})

plt.title(f"Spectral Analysis: '{query}' vs '{doc}'")
plt.xlabel("Tokens")
plt.yticks(rotation=0)
plt.tight_layout()

filename = "spectrum_heatmap.png"
plt.savefig(filename)
print(f"Heatmap gespeichert: {filename}")
