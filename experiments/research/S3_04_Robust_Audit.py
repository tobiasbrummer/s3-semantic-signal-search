
import torch
import numpy as np
from FlagEmbedding import BGEM3FlagModel

print("Lade BGE-M3 Model...")
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False)

def robust_audit(query, snippet):
    print(f"\nAudit: Q='{query}' vs S='{snippet}'")
    
    # 1. Tokenization
    q_enc = model.tokenizer(query, return_tensors='pt')
    s_enc = model.tokenizer(snippet, return_tensors='pt')
    
    # ALIGNMENT FIX: 
    # encode() gibt ColBERT Vektoren OHNE CLS Token zurück.
    # Wir nehmen also Tokens ab Index 1.
    q_tokens = model.tokenizer.convert_ids_to_tokens(q_enc['input_ids'][0][1:])
    s_tokens = model.tokenizer.convert_ids_to_tokens(s_enc['input_ids'][0][1:])
    
    # 2. Encoding
    q_out = model.encode([query], return_colbert_vecs=True, max_length=512)
    s_out = model.encode([snippet], return_colbert_vecs=True, max_length=512)
    
    q_vecs = q_out['colbert_vecs'][0]
    s_vecs = s_out['colbert_vecs'][0]
    
    # Safety Truncate (falls SEP am Ende auch variiert)
    q_tokens = q_tokens[:len(q_vecs)]
    s_tokens = s_tokens[:len(s_vecs)]
    
    # Length Check
    if len(q_vecs) != len(q_tokens):
        print(f"FATAL: Mismatch Q! {len(q_vecs)} vs {len(q_tokens)}")
        # Debug Print
        print(f"Toks: {q_tokens}")
        return
        
    # Similarity Matrix
    sim_matrix = q_vecs @ s_vecs.T
    
    # Thresholds
    THRESH_MISSING = 0.85 
    THRESH_ANOMALY = 0.85 

    # --- QUERY COVERAGE ---
    q_max_scores = np.max(sim_matrix, axis=1)
    q_best_idx = np.argmax(sim_matrix, axis=1)
    
    print("\n>>> QUERY COVERAGE (Was fehlt im Snippet?)")
    print(f"{'Query Token':<15} | {'Best Match (S)':<15} | {'Score':<6} | {'Status'}")
    print("-" * 60)
    
    for i in range(len(q_tokens)):
        token = q_tokens[i]
        # Skip Special Tokens
        if token in [model.tokenizer.cls_token, model.tokenizer.sep_token, model.tokenizer.pad_token]: continue
            
        score = q_max_scores[i]
        match_idx = q_best_idx[i]
        match_token = s_tokens[match_idx]
        
        status = "OK"
        if score < THRESH_MISSING: status = "MISSING/WEAK"
        if token in ['.', ',', ':'] and status != "OK": status = "OK (Punct)"
        
        print(f"{token:<15} | {match_token:<15} | {score:.3f}  | {status}")

    # --- SNIPPET COVERAGE ---
    s_max_scores = np.max(sim_matrix, axis=0)
    s_best_idx = np.argmax(sim_matrix, axis=0)
    
    print("\n>>> SNIPPET COVERAGE (Was ist extra/falsch im Snippet?)")
    print(f"{'Snippet Token':<15} | {'Best Match (Q)':<15} | {'Score':<6} | {'Status'}")
    print("-" * 60)
    
    anomalies = []
    
    for i in range(len(s_tokens)):
        token = s_tokens[i]
        # Skip Special Tokens
        if token in [model.tokenizer.cls_token, model.tokenizer.sep_token, model.tokenizer.pad_token]: continue
            
        score = s_max_scores[i]
        match_idx = s_best_idx[i]
        match_token = q_tokens[match_idx]
        
        status = "OK"
        if score < THRESH_ANOMALY: 
            status = "ANOMALY (!!)"
            anomalies.append(token)
            
        if token in ['.', ',', ':'] and status != "OK": status = "OK (Punct)"
        
        print(f"{token:<15} | {match_token:<15} | {score:.3f}  | {status}")

    if anomalies:
        print(f"\n>> ALARM: {len(anomalies)} kritische Tokens.")
    else:
        print("\n>> Alles sauber.")

# RUN
robust_audit("Die Bank teilt Zinsen mit.", "Die Bank teilt Zinsen nicht mit.")
robust_audit("Zinssatz 5 Prozent", "Zinssatz beträgt 50 Prozent")
