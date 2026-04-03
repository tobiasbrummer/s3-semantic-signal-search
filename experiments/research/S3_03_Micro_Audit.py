
import torch
import numpy as np
import matplotlib.pyplot as plt
from FlagEmbedding import BGEM3FlagModel

print("Lade BGE-M3 Model...")
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False)

def micro_audit(query, snippet):
    print(f"\nAudit: Q='{query}' vs S='{snippet}'")
    
    # Encoding
    q_out = model.encode([query], return_colbert_vecs=True)
    s_out = model.encode([snippet], return_colbert_vecs=True)
    
    q_vecs = q_out['colbert_vecs'][0]
    s_vecs = s_out['colbert_vecs'][0]
    
    # Tokens (für Anzeige)
    q_tokens = model.tokenizer.tokenize(query)
    s_tokens = model.tokenizer.tokenize(snippet)
    
    # Achtung: Vecs haben CLS/SEP, Tokens nicht. 
    # Wir nutzen Slice [1:-1] für die Matrix, um reine Content-Matches zu sehen.
    # Vecs Länge = len(tokens) + 2
    q_vecs_c = q_vecs[1:len(q_tokens)+1]
    s_vecs_c = s_vecs[1:len(s_tokens)+1]
    
    # Similarity Matrix [Q_len, S_len]
    sim_matrix = q_vecs_c @ s_vecs_c.T
    
    # 1. QUERY COVERAGE (Fehlt was in Snippet?)
    # Für jedes Q-Token: Best Match in S
    q_max_scores = np.max(sim_matrix, axis=1)
    q_best_idx = np.argmax(sim_matrix, axis=1)
    
    print("\n>>> QUERY COVERAGE (Was fehlt im Snippet?)")
    print(f"{'Query Token':<15} | {'Best Match (S)':<15} | {'Score':<6} | {'Status'}")
    print("-" * 60)
    
    for i, score in enumerate(q_max_scores):
        token = q_tokens[i]
        match_idx = q_best_idx[i]
        match_token = s_tokens[match_idx]
        
        status = "OK"
        if score < 0.85: status = "WEAK"
        if score < 0.75: status = "MISSING"
        
        print(f"{token:<15} | {match_token:<15} | {score:.3f}  | {status}")

    # 2. SNIPPET COVERAGE (Was ist ZUSÄTZLICH im Snippet?)
    # Für jedes S-Token: Best Match in Q
    s_max_scores = np.max(sim_matrix, axis=0) # Axis 0 = Max über Query
    s_best_idx = np.argmax(sim_matrix, axis=0)
    
    print("\n>>> SNIPPET COVERAGE (Was ist extra/falsch im Snippet?)")
    print(f"{'Snippet Token':<15} | {'Best Match (Q)':<15} | {'Score':<6} | {'Status'}")
    print("-" * 60)
    
    anomalies = []
    
    for i, score in enumerate(s_max_scores):
        token = s_tokens[i]
        match_idx = s_best_idx[i]
        match_token = q_tokens[match_idx]
        
        status = "OK"
        # Hier ist die Logik andersrum: Ein niedriger Score heißt "Ich bin neu hier!"
        # Das ist genau das, was wir bei "nicht" oder "50" suchen.
        if score < 0.85: status = "INSERTION (?)"
        if score < 0.75: 
            status = "ANOMALY (!!)"
            anomalies.append(token)
        
        print(f"{token:<15} | {match_token:<15} | {score:.3f}  | {status}")

    if anomalies:
        print(f"\n>> ALARM: {len(anomalies)} unerwartete Tokens gefunden: {anomalies}")
    else:
        print("\n>> Alles sauber.")

# TEST 1: Negation
micro_audit("Die Bank teilt Zinsen mit.", "Die Bank teilt Zinsen nicht mit.")

# TEST 2: Numerik
micro_audit("Zinssatz 5 Prozent", "Zinssatz beträgt 50 Prozent")
