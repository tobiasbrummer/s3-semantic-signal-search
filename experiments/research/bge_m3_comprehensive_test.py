
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# Modell ID
MODEL_NAME = "BAAI/bge-m3"

print(f"Lade {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

def compute_dense(out_a, out_b):
    # CLS Token (Index 0) Embedding
    vec_a = out_a.last_hidden_state[:, 0, :]
    vec_b = out_b.last_hidden_state[:, 0, :]
    vec_a = torch.nn.functional.normalize(vec_a, p=2, dim=1)
    vec_b = torch.nn.functional.normalize(vec_b, p=2, dim=1)
    return torch.matmul(vec_a, vec_b.T).item()

def compute_colbert(out_a, out_b, mask_a, mask_b):
    # ColBERT: Late Interaction über ALLE Token
    # BGE-M3 nutzt einen Linear Layer projection für ColBERT, aber oft reicht der hidden state 
    # für einen ersten Test. Wenn das Modell 'norm' layers hat, müssen wir aufpassen.
    # Wir nehmen hier die 'last_hidden_state' und normalisieren sie.
    
    vecs_a = torch.nn.functional.normalize(out_a.last_hidden_state.squeeze(0), p=2, dim=1)
    vecs_b = torch.nn.functional.normalize(out_b.last_hidden_state.squeeze(0), p=2, dim=1)
    
    # Sim Matrix
    sim = torch.matmul(vecs_a, vecs_b.T)
    
    # MaxSim: Für jeden Token in A den besten Partner in B finden
    max_scores = torch.max(sim, dim=1).values
    
    # Wir ignorieren Padding tokens für den Mean score
    active_len = torch.sum(mask_a)
    score = torch.sum(max_scores[:active_len]) / active_len
    
    return score.item(), max_scores, vecs_a

def compute_sparse(text_a, text_b):
    # BGE-M3 hat keinen direkten "Sparse Head" im Standard HF AutoModel, 
    # es sei denn man lädt es speziell. 
    # Aber wir können simulieren, was SPLADE macht, indem wir die MLM Logits nehmen?
    # Nein, BGE-M3 ist ein Encoder. 
    # Wir überspringen den "echten" Sparse Test ohne die FlagEmbedding Library vorerst 
    # und konzentrieren uns auf Dense vs ColBERT (Token).
    return 0.0

def analyze_pair(text_a, text_b, label):
    print(f"\n--- {label} ---")
    print(f"A: {text_a}")
    print(f"B: {text_b}")
    
    inp_a = tokenizer(text_a, return_tensors="pt")
    inp_b = tokenizer(text_b, return_tensors="pt")
    
    with torch.no_grad():
        out_a = model(**inp_a)
        out_b = model(**inp_b)
        
    # 1. Dense Score
    s_dense = compute_dense(out_a, out_b)
    
    # 2. ColBERT Score
    s_colbert, max_scores, vecs_a = compute_colbert(out_a, out_b, inp_a.attention_mask, inp_b.attention_mask)
    
    print(f"Scores -> Dense: {s_dense:.4f} | ColBERT (MaxSim): {s_colbert:.4f}")
    
    # Detail Analyse ColBERT: Wo ist der Mismatch?
    # Wir schauen uns die Tokens in A an, die niedrige Max-Similarity haben.
    tokens_a = tokenizer.convert_ids_to_tokens(inp_a.input_ids[0])
    
    print(f"{'Token (A)':<15} | {'MaxSim (vs B)':<12} | {'Status'}")
    print("-" * 45)
    
    low_sim_tokens = []
    for i, t in enumerate(tokens_a):
        if t in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]: continue
        
        score = max_scores[i].item()
        status = "OK"
        if score < 0.85: status = "MISMATCH"
        if score < 0.75: status = "CRITICAL"
        
        print(f"{t:<15} | {score:.4f}       | {status}")
        
        if score < 0.85:
            low_sim_tokens.append(t)
            
    return low_sim_tokens

# TEST FÄLLE
# 1. Zinsen Numerik
analyze_pair("Der Zinssatz beträgt 5 Prozent.", "Der Zinssatz beträgt 50 Prozent.", "Numerik Fehler")

# 2. Negation
analyze_pair("Die Bank teilt Änderungen mit.", "Die Bank teilt Änderungen nicht mit.", "Negation Fehler")

# 3. Kontext (Gleiches Wort, andere Bedeutung - schwer!)
analyze_pair("Die Bank sitzt am Fluss.", "Die Bank zahlt Zinsen.", "Polysemie (Bank)")
