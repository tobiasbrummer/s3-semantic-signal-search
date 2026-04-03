#!/usr/bin/env python3
"""
Untersuchung der Vektorlängen-Hypothese (Norm Hypothesis).

Dieses Skript analysiert, ob die L2-Norm der Dense Vektoren (Jina v3)
semantische Information trägt (z.B. Wichtigkeit, Konfidenz) oder nur
technisches Artefakt (z.B. Worthäufigkeit) ist.

Als Ground Truth für "Wichtigkeit" nutzen wir:
1. Opensearch Sparse Weights (SPLADE) - bewährtes Maß für lexikalische Relevanz.
2. Heuristische Stopwort-Erkennung.

Input: tests/files/agb.pdf
"""

import sys
import os
try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None
import numpy as np
from engine import SemanticEngine
import re

# Einfache deutsche Stopwörter für die Analyse
STOPWORDS = {
    "und", "die", "der", "das", "in", "den", "von", "mit", "für", "ist", 
    "im", "auf", "nicht", "eine", "einen", "sich", "dem", "des", "als", 
    "auch", "es", "an", "werden", "aus", "er", "hat", "dass", "sie", 
    "nach", "wird", "bei", "ein", "oder", "um", "zu", "denen", "deren",
    "diese", "dieser", "dieses", "ihre", "ihrer", "sein", "seine", "wie"
}

def extract_text_from_pdf(path: str) -> str:
    if fitz is None:
        raise RuntimeError(
            "PyMuPDF (fitz) ist nicht installiert. Installiere es z.B. via `pip install pymupdf`."
        )
    print(f"Lese PDF: {path}...")
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    print(f"Extrahierter Text: {len(text)} Zeichen.")
    return text

def analyze_norms(engine, text):
    print("\nBerechne Embeddings (kann dauern bei CPU)...")
    # Wir nehmen nur die ersten 4000 Zeichen um Zeit zu sparen, 
    # falls CPU-only. Das reicht für statistische Signifikanz.
    # AGBs sind lang, wir wollen einen repräsentativen Ausschnitt.
    # Aber engine kann sliding window, also testen wir das gleich mit.
    # Limit auf ~2 Seiten Text für den Test.
    text_sample = text[:10000] 
    
    result = engine.embed_text(text_sample)
    
    words = np.array(result.words)
    norms = result.dense_norms
    sparse = result.sparse_weights
    
    print(f"Anzahl Wörter: {len(words)}")
    
    # Datenbereinigung: Entferne Punktuation für Analyse
    mask = np.array([w.isalnum() for w in words])
    words = words[mask]
    norms = norms[mask]
    sparse = sparse[mask]
    
    print(f"Anzahl Wörter (ohne Punktuation): {len(words)}")

    # 1. Korrelation zwischen Norm und Sparse Weight
    corr = np.corrcoef(norms, sparse)[0, 1]
    print(f"\n1. Korrelation Dense-Norm vs. Sparse-Weight: {corr:.4f}")
    if corr > 0.5:
        print("-> Deutliche positive Korrelation: Norm = Wichtigkeit?")
    elif corr < -0.5:
        print("-> Deutliche negative Korrelation: Norm = Unwichtigkeit?")
    else:
        print("-> Keine starke lineare Korrelation.")

    # 2. Stopwörter Analyse
    stop_indices = [i for i, w in enumerate(words) if w.lower() in STOPWORDS]
    content_indices = [i for i, w in enumerate(words) if w.lower() not in STOPWORDS]
    
    if stop_indices and content_indices:
        avg_norm_stop = np.mean(norms[stop_indices])
        avg_norm_content = np.mean(norms[content_indices])
        
        avg_sparse_stop = np.mean(sparse[stop_indices])
        avg_sparse_content = np.mean(sparse[content_indices])
        
        print(f"\n2. Vergleich Stopwörter vs. Inhaltswörter:")
        print(f"  Ø Norm (Stopwörter):    {avg_norm_stop:.4f}")
        print(f"  Ø Norm (Inhalt):        {avg_norm_content:.4f}")
        print(f"  Ratio (Inhalt/Stop):    {avg_norm_content/avg_norm_stop:.4f}")
        print(f"  (Vergleich Sparse Ratio: {avg_sparse_content/avg_sparse_stop:.4f})")
        
        if avg_norm_stop > avg_norm_content:
            print("-> Warnung: Stopwörter haben HÖHERE Normen!")
        else:
            print("-> Inhaltswörter haben höhere Normen.")
            
    # 3. Top & Bottom Listen
    k = 15
    print(f"\n3. Top {k} Wörter nach Metrik:")
    
    # Sort by Norm descending
    top_norm_idx = np.argsort(norms)[::-1][:k]
    print(f"\n  Top Norm (Höchste Vektorlänge):")
    for idx in top_norm_idx:
        print(f"    {words[idx]:<20} Norm: {norms[idx]:.2f} | Sparse: {sparse[idx]:.2f}")

    # Sort by Norm ascending
    bot_norm_idx = np.argsort(norms)[:k]
    print(f"\n  Bottom Norm (Niedrigste Vektorlänge):")
    for idx in bot_norm_idx:
        print(f"    {words[idx]:<20} Norm: {norms[idx]:.2f} | Sparse: {sparse[idx]:.2f}")
        
    # Sort by Sparse descending (Reference)
    top_sparse_idx = np.argsort(sparse)[::-1][:k]
    print(f"\n  Top Sparse (Referenz Wichtigkeit):")
    for idx in top_sparse_idx:
        print(f"    {words[idx]:<20} Norm: {norms[idx]:.2f} | Sparse: {sparse[idx]:.2f}")

    # 4. Spezifische Wort-Checks (AGB Kontext)
    check_words = ["Kunde", "Bank", "Kündigung", "AGB", "sofort", "und", "der"]
    print(f"\n4. Spot-Check spezifischer Begriffe:")
    found_map = {}
    for i, w in enumerate(words):
        if w in check_words and w not in found_map:
            found_map[w] = (norms[i], sparse[i])
    
    for w in check_words:
        if w in found_map:
            n, s = found_map[w]
            print(f"    {w:<15} Norm: {n:.2f} | Sparse: {s:.2f}")

if __name__ == "__main__":
    pdf_path = "tests/files/agb.pdf"
    if not os.path.exists(pdf_path):
        print(f"Fehler: {pdf_path} nicht gefunden.")
        sys.exit(1)

    try:
        engine = SemanticEngine() # Auto device
        text = extract_text_from_pdf(pdf_path)
        analyze_norms(engine, text)
    except Exception as e:
        print(f"Fehler: {e}")
        import traceback
        traceback.print_exc()
