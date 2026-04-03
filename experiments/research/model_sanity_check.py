#!/usr/bin/env python3
"""
Sanity checks for embedding models (Improved Version):
- Token-level similarity (word embedding proximity)
- Sentence-level MaxSim (ColBERT-style late interaction)
- Detailed difference explanation (Token mismatch analysis)
- Attention distribution
"""

from __future__ import annotations

import argparse
import textwrap
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

# Hier sind ein paar Modelle vorausgewählt, die für deinen Zweck gut sind.
DEFAULT_MODELS = [
    "BAAI/bge-m3",
]

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)

def max_sim_score(last_hidden_a: torch.Tensor, last_hidden_b: torch.Tensor, mask_a: torch.Tensor, mask_b: torch.Tensor) -> float:
    """
    Berechnet die ColBERT-Style Similarity (Late Interaction).
    Erwartet 2D Inputs [Seq_Len, Hidden_Dim].
    Falls 3D [Batch, Seq, Dim] kommt, wird es automatisch geflattet.
    """
    # Fix für 3D Input [1, Seq, Dim] -> [Seq, Dim]
    if last_hidden_a.dim() == 3:
        last_hidden_a = last_hidden_a.squeeze(0)
    if last_hidden_b.dim() == 3:
        last_hidden_b = last_hidden_b.squeeze(0)

    # 1. Normalisieren (L2-Norm) entlang der Feature-Dimension (Dim 1 bei 2D-Tensor)
    a_norm = torch.nn.functional.normalize(last_hidden_a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(last_hidden_b, p=2, dim=1)

    # 2. Similarity Matrix: [Seq_A, Seq_B]
    # MatMul: [Seq_A, Dim] x [Dim, Seq_B]
    sim_matrix = torch.matmul(a_norm, b_norm.transpose(0, 1))

    # 3. Max Similarity für jedes Token in A finden
    max_scores_per_token = torch.max(sim_matrix, dim=1).values
    
    return torch.mean(max_scores_per_token).item()

def explain_difference(tokenizer, inputs_a, inputs_b, emb_a, emb_b):
    """
    Zeigt im Detail, welche Tokens aus Satz B in Satz A fehlen.
    """
    print(f"\n   -> Detail-Analyse (Asymmetrisch B -> A):")
    
    # Normalisieren
    a_norm = torch.nn.functional.normalize(emb_a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(emb_b, p=2, dim=1)
    
    # Matrix: [Seq_B, Seq_A]
    sim_matrix = torch.matmul(b_norm, a_norm.transpose(0, 1))
    
    # Beste Matches finden
    best_scores, best_indices = torch.max(sim_matrix, dim=1)
    
    tokens_a = tokenizer.convert_ids_to_tokens(inputs_a["input_ids"][0])
    tokens_b = tokenizer.convert_ids_to_tokens(inputs_b["input_ids"][0])
    
    # Tabellenkopf
    print(f"   {'Token (Satz B)':<15} | {'Score':<6} | {'Match in A'}")
    print("   " + "-"*45)
    
    scores = []
    # Iteriere über Tokens (sklippe CLS/SEP oft an Index 0 und -1, hier vereinfacht alles anzeigen)
    for i, token_b in enumerate(tokens_b):
        # Ignoriere Special Tokens für die Ausgabe, um es lesbar zu halten
        if token_b in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token, "<s>", "</s>", "<pad>"]:
            continue
            
        score = best_scores[i].item()
        match_idx = best_indices[i].item()
        match_token = tokens_a[match_idx]
        scores.append(score)
        
        # Markiere schwache Matches visuell
        marker = " (!)" if score < 0.85 else ""
        print(f"   {token_b:<15} | {score:.3f}  | {match_token}{marker}")

    if scores:
        print(f"   >> Min Score (Bottleneck): {min(scores):.4f}")

def explain_difference_smart(tokenizer, inputs_a, inputs_b, emb_a, emb_b, threshold_delta=0.04):
    """
    Analysiert Unterschiede basierend auf relativem Abfall (Delta) statt absolutem Wert.
    Ideal für Zahlen und feine Nuancen.
    """
    # ... (Normalisierung und Matrix-Berechnung wie vorher) ...
    a_norm = torch.nn.functional.normalize(emb_a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(emb_b, p=2, dim=1)
    sim_matrix = torch.matmul(b_norm, a_norm.transpose(0, 1))
    best_scores, best_indices = torch.max(sim_matrix, dim=1)
    
    tokens_a = tokenizer.convert_ids_to_tokens(inputs_a["input_ids"][0])
    tokens_b = tokenizer.convert_ids_to_tokens(inputs_b["input_ids"][0])
    
    # Daten sammeln
    data = []
    scores_clean = [] # Scores ohne Special Tokens für Statistik
    
    for i, token_b in enumerate(tokens_b):
        if token_b in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token, "<s>", "</s>"]:
            continue
        
        score = best_scores[i].item()
        match_idx = best_indices[i].item()
        match_token = tokens_a[match_idx]
        
        data.append({
            "token": token_b,
            "score": score,
            "match": match_token
        })
        scores_clean.append(score)

    if not scores_clean: return

    # Statistik berechnen
    # Wir nehmen den Median als "Baseline", da er robuster gegen Ausreißer (die Fehler) ist
    baseline = np.median(scores_clean)
    
    print(f"\n   -> Smarte Delta-Analyse (Baseline: {baseline:.4f})")
    print(f"   {'Token (B)':<15} | {'Score':<6} | {'Delta':<6} | {'Status'}")
    print("   " + "-"*55)
    
    for item in data:
        delta = baseline - item['score']
        
        # Bewertung
        if delta > threshold_delta: # Z.B. 0.99 - 0.93 = 0.06 (> 0.04 -> ALARM)
            status = "Mismatch! (<<)"
        elif delta > (threshold_delta / 2):
            status = "Unsicheer (?)"
        else:
            status = "OK"
            
        # Nur anzeigen, wenn es relevant ist oder zur Kontrolle
        # Hier zeigen wir alles, markieren aber deutlich
        marker = "<<" if "Mismatch" in status else ""
        print(f"   {item['token']:<15} | {item['score']:.3f}  | -{delta:.3f}  | {status}")

    min_score = min(scores_clean)
    max_delta = baseline - min_score
    print(f"   >> Max Delta: {max_delta:.4f} (Kritisch wenn > {threshold_delta})")

def find_subsequence(tokens: list[str], target: list[str]) -> tuple[int, int] | None:
    if not target:
        return None
    for i in range(len(tokens) - len(target) + 1):
        if tokens[i:i + len(target)] == target:
            return i, i + len(target)
    return None

def get_token_embedding(tokenizer, last_hidden, tokens, word):
    target = tokenizer.tokenize(word)
    span = find_subsequence(tokens, target)
    if span is None:
        return None
    start, end = span
    emb = last_hidden[0, start:end, :].mean(dim=0)
    return emb.cpu().numpy()

def attention_importance(attn: tuple[torch.Tensor, ...]) -> np.ndarray:
    all_attn = torch.stack(attn)
    avg_attn = all_attn.mean(dim=(0, 1, 2))
    importance = avg_attn.sum(dim=0)
    return importance.cpu().numpy()

def load_model(model_name: str, trust_remote_code: bool, max_length: int):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    try:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code, attn_implementation="eager")
    except TypeError:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model.eval()
    if hasattr(model.config, "max_position_embeddings"):
        max_length = min(max_length, int(model.config.max_position_embeddings))
    return tokenizer, model, max_length

def explain_difference_robust(tokenizer, inputs_a, inputs_b, emb_a, emb_b):
    """
    Unterscheidet zwischen harter Negation (Rot) und weicher Paraphrasierung (Gelb).
    Ignoriert Stopwords.
    """
    # 1. Einfache Liste von Stopwords (kann erweitert werden)
    STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", 
        "of", "with", "by", "is", "are", "was", "were", "be", "been", "this", "that"
    }
    
    a_norm = torch.nn.functional.normalize(emb_a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(emb_b, p=2, dim=1)
    
    # Similarity Matrix & Best Matches
    sim_matrix = torch.matmul(b_norm, a_norm.transpose(0, 1))
    best_scores, best_indices = torch.max(sim_matrix, dim=1)
    
    tokens_a = tokenizer.convert_ids_to_tokens(inputs_a["input_ids"][0])
    tokens_b = tokenizer.convert_ids_to_tokens(inputs_b["input_ids"][0])
    
    print(f"\n   -> Robuste Semantik-Analyse:")
    print(f"   {'Token (B)':<15} | {'Score':<6} | {'Kategorie':<15} | {'Urteil'}")
    print("   " + "-"*65)
    
    min_content_score = 1.0
    
    for i, token_b in enumerate(tokens_b):
        # Clean Token (remove special chars usually added by tokenizers like   or Ġ)
        clean_token = token_b.replace(" ", "").replace("Ġ", "").replace("▁", "").lower()
        
        # Skip Special Tokens
        if token_b in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token, "<s>", "</s>"]:
            continue
            
        score = best_scores[i].item()
        match_idx = best_indices[i].item()
        match_token = tokens_a[match_idx]

        # LOGIK-KERN:
        
        # A) Ist es ein Stopword? -> Ignorieren oder sehr weich bewerten
        if clean_token in STOPWORDS:
            category = "Stopword"
            status = "Ignored"
        
        # B) Ist der Score extrem niedrig? -> ROTE ZONE (Negation / Fehler)
        elif score < 0.70: 
            category = "CRITICAL MISS"
            status = "!!! ALARM !!!"
            min_content_score = min(min_content_score, score)

        # C) Ist der Score mittelmäßig? -> GELBE ZONE (Synonym / Paraphrase)
        elif score < 0.88:
            category = "Synonym/Struct"
            status = "OK (Paraphrase)"
            # Wir senken den min_score hier NICHT dramatisch, da es okay ist
            
        # D) Hoher Score -> GRÜNE ZONE
        else:
            category = "Exact/Close"
            status = "OK"
            min_content_score = min(min_content_score, score)

        print(f"   {token_b:<15} | {score:.3f}  | {category:<15} | {status} (-> {match_token})")

    print("   " + "-"*65)
    if min_content_score < 0.70:
        print(f"   >> ERGEBNIS: INHALTLICHER WIDERSPRUCH GEFUNDEN (Score: {min_content_score:.3f})")
    else:
        print(f"   >> ERGEBNIS: Inhalt deckungsgleich (Paraphrasierung)")

def explain_difference_zscore(tokenizer, inputs_a, inputs_b, emb_a, emb_b):
    """
    Erkennt Anomalien mittels Z-Score (Statistische Ausreißer).
    Löst das Problem unterschiedlicher Ranges bei Zahlen vs. Negationen.
    """

    # 1. Einfache Liste von Stopwords (kann erweitert werden)
    STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", 
        "of", "with", "by", "is", "are", "was", "were", "be", "been", "this", "that"
    }
    
    a_norm = torch.nn.functional.normalize(emb_a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(emb_b, p=2, dim=1)
    
    # Similarity Matrix & Best Matches
    sim_matrix = torch.matmul(b_norm, a_norm.transpose(0, 1))
    best_scores, best_indices = torch.max(sim_matrix, dim=1)
    
    tokens_a = tokenizer.convert_ids_to_tokens(inputs_a["input_ids"][0])
    tokens_b = tokenizer.convert_ids_to_tokens(inputs_b["input_ids"][0])

    
    # 1. Daten sammeln und Stopwords filtern für Statistik
    scores_all = []
    tokens_clean = []
    
    # Wir brauchen die rohen Werte erst einmal in einer Liste
    temp_data = []
    
    for i, token_b in enumerate(tokens_b):
        if token_b in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token, "<s>", "</s>"]:
            continue
            
        score = best_scores[i].item()
        clean_token = token_b.replace(" ", "").replace("Ġ", "").lower()
        
        # Stopwords schließen wir aus der Statistik-Berechnung (Mean/Std) aus!
        # Sonst verzerren sie die Standardabweichung nach oben.
        is_stopword = clean_token in STOPWORDS
        
        temp_data.append({
            "token": token_b,
            "score": score,
            "is_stopword": is_stopword,
            "match_idx": best_indices[i].item()
        })
        
        if not is_stopword:
            scores_all.append(score)

    if not scores_all: return

    # 2. Statistik berechnen (nur auf Content-Tokens!)
    mu = np.mean(scores_all)
    sigma = np.std(scores_all) + 1e-9 # Vermeide Division durch Null
    
    print(f"\n   -> Z-Score Analyse (Mean: {mu:.3f}, StdDev: {sigma:.3f})")
    print(f"   {'Token (B)':<15} | {'Score':<6} | {'Z-Score':<6} | {'Status'}")
    print("   " + "-"*60)
    
    min_z_score = 0
    
    for item in temp_data:
        # Z-Score berechnen: (Score - Mean) / StdDev
        # Negativer Z-Score = Schlechter als Durchschnitt
        z_score = (item['score'] - mu) / sigma
        
        status = "OK"
        marker = ""
        
        if item['is_stopword']:
            status = "Ignored (Stop)"
        
        else:
            # REGEL 1: Der Statistische Ausreißer (für Zahlen & feine Details)
            if z_score < -2.0:
                status = "ANOMALY (Z<-2)"
                marker = "<<"
                min_z_score = min(min_z_score, z_score)
            
            # REGEL 2: Der Absolute Absturz (für Negationen)
            # Falls der ganze Satz schlecht ist, hilft Z-Score nicht. 
            # Deshalb: Harter Boden bei 0.60
            if item['score'] < 0.60:
                status = "CRITICAL FAIL"
                marker = "!!!"
            
            # Warnung bei moderaten Ausreißern
            elif z_score < -1.5:
                status = "Warning (Z<-1.5)"
                marker = "?"

        print(f"   {item['token']:<15} | {item['score']:.3f}  | {z_score:>6.2f} | {status} {marker}")

    if min_z_score < -2.0:
        print(f"   >> ERGEBNIS: Signifikante Abweichung gefunden! (Max Sigma: {abs(min_z_score):.1f})")
    else:
        print(f"   >> ERGEBNIS: Im statistischen Toleranzbereich.")

def run_checks(model_name: str, trust_remote_code: bool, max_length: int):
    print(f"\n=== Model: {model_name} ===")
    tokenizer, model, max_length = load_model(model_name, trust_remote_code, max_length)

    # Automatische Prefix-Erkennung für E5
    prefix_q = "query: " if "e5" in model_name.lower() else ""
    prefix_p = "passage: " if "e5" in model_name.lower() else ""

    token_tests = [
        ("apple", "banana", "microsoft"),
        ("cat", "dog", "car"),
        ("paris", "berlin", "banana"),
    ]

    sentence_tests = [
        ("paraphrase_en", 
         "The meeting was postponed to next week.", 
         "The meeting was delayed until the following week."),
        ("negation_en", 
         "The patient has a fever.", 
         "The patient does not have a fever."),
        ("numeric_en", 
         "Revenue grew by 3 percent this quarter.", 
         "Revenue grew by 30 percent this quarter."),
        ("negation_de",
         "Die Bank teilt dem Kunden Änderungen von Zinsen mit.",
         "Die Bank teilt dem Kunden Änderungen von Zinsen nicht mit."),
        ("numeric_de",
         "Der Zinssatz beträgt 5 Prozent.",
         "Der Zinssatz beträgt 50 Prozent."),
        ("paraphrase_de",
         "Der Vertrag kann innerhalb von zwei Wochen gekündigt werden.",
         "Die Kündigungsfrist für den Vertrag beläuft sich auf 14 Tage.")
    ]

    # --- Token Level Tests ---
    print("\n--- Word-Level Similarity ---")
    for w1, w2, w3 in token_tests:
        # Hier nutzen wir keine Prefixes, da es rohe Wort-Vergleiche sind
        text = f"{w1} {w2} {w3}"
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        
        with torch.no_grad():
            try:
                outputs = model(**inputs)
            except:
                outputs = model(**inputs, output_attentions=False)
                
        last_hidden = outputs.last_hidden_state
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        e1 = get_token_embedding(tokenizer, last_hidden, tokens, w1)
        e2 = get_token_embedding(tokenizer, last_hidden, tokens, w2)
        e3 = get_token_embedding(tokenizer, last_hidden, tokens, w3)

        if e1 is not None and e2 is not None and e3 is not None:
            sim12 = cosine(e1, e2)
            sim13 = cosine(e1, e3)
            sim23 = cosine(e2, e3)
            print(f"Token sim [{w1},{w2},{w3}]: {sim12:.4f} | {sim13:.4f} | {sim23:.4f}")

    # --- Sentence Level Tests (MaxSim) ---
    print("\n--- Sentence Interaction (MaxSim) ---")
    for label, a, b in sentence_tests:
        # Prefixes anwenden
        text_a = prefix_p + a
        text_b = prefix_q + b # Wir tun so, als wäre B die Query

        inputs_a = tokenizer(text_a, return_tensors="pt", truncation=True, max_length=max_length)
        inputs_b = tokenizer(text_b, return_tensors="pt", truncation=True, max_length=max_length)
        
        with torch.no_grad():
            out_a = model(**inputs_a).last_hidden_state
            out_b = model(**inputs_b).last_hidden_state
        
        # MaxSim berechnen
        sim = max_sim_score(
            out_a[0],      # [Seq, Dim]
            out_b[0],      # [Seq, Dim]
            inputs_a["attention_mask"][0], 
            inputs_b["attention_mask"][0]
        )
        print(f"Sentence sim [{label}]: {sim:.4f}")
        
        # Bei Negation oder niedrigen Scores Details anzeigen
        # if label == "negation" or sim < 0.90:
        explain_difference_zscore(tokenizer, inputs_a, inputs_b, out_a[0], out_b[0])

    # --- Attention Distribution ---
    print("\n--- Attention Check ---")
    attn_text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(attn_text, return_tensors="pt", truncation=True, max_length=max_length)
    
    with torch.no_grad():
        try:
            outputs = model(**inputs, output_attentions=True)
        except:
            # Fallback falls das Modell keine Attentions ausgibt
            outputs = None

    if outputs and hasattr(outputs, "attentions") and outputs.attentions is not None:
        importance = attention_importance(outputs.attentions)
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        pairs = list(zip(tokens, importance))
        pairs.sort(key=lambda x: x[1], reverse=True)
        top = pairs[:8]
        print("Top Attention Tokens:", ", ".join(f"{t}:{v:.3f}" for t, v in top))
    else:
        print("Attention data not available for this model.")

def main() -> None:
    parser = argparse.ArgumentParser(description="Embedding model sanity checks")
    parser.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS))
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        print("No models specified.")
        return

    for model_name in models:
        run_checks(model_name, args.trust_remote_code, args.max_length)

if __name__ == "__main__":
    main()
