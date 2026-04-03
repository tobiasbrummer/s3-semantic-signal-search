import torch
import numpy as np
import matplotlib.pyplot as plt
import fitz
import os
from FlagEmbedding import BGEM3FlagModel

# 1. SETUP
print("Lade BGE-M3 Model...")
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False)

class PDFIngestor:
    def load_pdf(self, file_path):
        doc = fitz.open(file_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text("text").replace('\n', ' ') + " "
        return full_text

def spectral_scan(pdf_text, query, window_size=256, stride=128):
    print(f"Scanne {len(pdf_text)} Zeichen mit Query: '{query}'")
    
    # Pre-Tokenize
    # Wir nutzen den Modell-Tokenizer, um saubere Windows zu haben
    enc = model.tokenizer(pdf_text, return_tensors='pt', add_special_tokens=False)
    input_ids = enc['input_ids'][0]
    total_tokens = len(input_ids)
    print(f"Total Tokens: {total_tokens}")
    
    # Encode Query ONCE
    q_out = model.encode([query], return_dense=True, return_sparse=True, return_colbert_vecs=True)
    q_colbert = q_out['colbert_vecs'][0]
    q_dense = q_out['dense_vecs'][0]
    q_sparse = q_out['lexical_weights'][0]
    
    # Result Arrays
    # Wir speichern pro Window einen Datenpunkt für den Plot
    x_axis = []
    dense_scores = []
    sparse_scores = []
    multi_scores = []
    
    # Sliding Window
    for start_idx in range(0, total_tokens, stride):
        end_idx = min(start_idx + window_size, total_tokens)
        if start_idx >= total_tokens: break
        
        # Extract Window IDs
        window_ids = input_ids[start_idx:end_idx]
        window_text = model.tokenizer.decode(window_ids)
        
        # Encode Window
        # Wir müssen es neu encoden, um die Embeddings zu bekommen
        # (Teuer, aber notwendig für echte Vektoren)
        d_out = model.encode([window_text], return_dense=True, return_sparse=True, return_colbert_vecs=True)
        
        # 1. DENSE SCORE (Context)
        d_dense = d_out['dense_vecs'][0]
        s_dense = q_dense @ d_dense.T
        
        # 2. SPARSE SCORE (Keywords)
        d_sparse = d_out['lexical_weights'][0]
        s_sparse = model.compute_lexical_matching_score(q_sparse, d_sparse)
        
        # 3. MULTI SCORE (ColBERT)
        d_colbert = d_out['colbert_vecs'][0]
        s_multi = model.colbert_score(q_colbert, d_colbert)
        
        # Store
        center_pos = start_idx + (len(window_ids) // 2)
        x_axis.append(center_pos)
        dense_scores.append(s_dense)
        sparse_scores.append(s_sparse)
        multi_scores.append(s_multi)
        
        if end_idx == total_tokens: break

    return x_axis, dense_scores, sparse_scores, multi_scores

# RUN
ingestor = PDFIngestor()
pdf_path = "/var/home/t0bybr/containers/s3/test_contract.pdf"
full_text = ingestor.load_pdf(pdf_path)

# Query zielt auf die Kündigungsklausel
query = "Kündigung Frist sechs Wochen Zinsen Erhöhung"
x, dense, sparse, multi = spectral_scan(full_text, query)

# Normalisieren für Plot (0-1 Range)
def norm(arr):
    arr = np.array(arr)
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-9)

# PLOT
plt.figure(figsize=(15, 8))

# Wir plotten normalisierte Kurven, um die "Form" zu vergleichen
plt.plot(x, norm(dense), label='Dense (Context)', color='blue', alpha=0.6, linewidth=2)
plt.plot(x, norm(sparse), label='Sparse (Keywords)', color='red', alpha=0.8, linewidth=1.5)
plt.plot(x, norm(multi), label='Multi (ColBERT)', color='green', alpha=0.8, linestyle='--', linewidth=2)

# Peak Finder (Simple)
best_idx = np.argmax(multi)
plt.axvline(x=x[best_idx], color='black', linestyle=':', label='Winner (Multi)')

plt.title(f"Spectral Scan: '{query}'")
plt.xlabel("Token Position")
plt.ylabel("Normalized Score")
plt.legend()
plt.grid(True, alpha=0.3)

filename = "spectral_scan_result.png"
plt.savefig(filename)
print(f"Scan Plot gespeichert: {filename}")
print(f"Winner Position: {x[best_idx]} (Score: {multi[best_idx]:.4f})")
