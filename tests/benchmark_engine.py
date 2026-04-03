#!/usr/bin/env python3
"""
Benchmark für Semantic Engine V2 mit PAWS-X und XNLI.

Prüft, wie gut die Engine Widersprüche und inhaltliche Änderungen erkennt.
"""

import os
import pandas as pd
import numpy as np
from engine_v2 import HierarchicalEngine
from analyzer_v2 import AnomalyAnalyzerV2
from critic_v2 import LlamaCppCriticV2
import time

def load_paws_x(path, limit=50):
    """Lädt PAWS-X (de) Test-Daten. Label 0 = Widerspruch/Anders, 1 = Paraphrase."""
    df = pd.read_csv(path, sep='\t')
    # Wir suchen vor allem Label 0 (Änderungen)
    df = df.sample(min(limit, len(df)), random_state=42)
    return df

def load_xnli(path, limit=50):
    """Lädt XNLI (de) Test-Daten."""
    df = pd.read_csv(path, sep='\t')
    df = df[df['language'] == 'de']
    df = df.sample(min(limit, len(df)), random_state=42)
    return df

def run_benchmark():
    # 1. Setup
    print("[Benchmark] Initialisiere Engine...")
    engine = HierarchicalEngine()
    critic = LlamaCppCriticV2()
    analyzer = AnomalyAnalyzerV2(engine, critic)

    paths = {
        "paws": "tests/files/PAWS-X/de/test_2k.tsv",
        "xnli": "tests/files/XNLI/xnli.test.tsv"
    }

    results = []

    # 2. PAWS-X Test
    if os.path.exists(paths["paws"]):
        print(f"\n--- Starte PAWS-X (de) Benchmark ---")
        df_paws = load_paws_x(paths["paws"], limit=20)
        
        correct = 0
        total = 0
        
        for _, row in df_paws.iterrows():
            total += 1
            q, d = row['sentence1'], row['sentence2']
            expected_change = (row['label'] == 0) # 0 bedeutet in PAWS-X: Keine Paraphrase
            
            print(f"[{total}] Analysiere Paar...")
            res = analyzer.analyze(q, d)
            
            # Hat der Analyzer eine Contradiction gefunden?
            actual_change = any(m.contradiction for m in res.matches)
            
            is_correct = (actual_change == expected_change)
            if is_correct: 
                correct += 1
            else:
                print(f"\n[FEHLER] #{total}")
                print(f"  Q: {q}")
                print(f"  D: {d}")
                print(f"  Erwartet: {'Änderung' if expected_change else 'Gleich'} | Ist: {'Änderung' if actual_change else 'Gleich'}")
                
                # Deep Dive
                for m in res.matches:
                    print(f"  -> Match Score: {m.dense_score:.4f} | Contradiction: {m.contradiction}")
                    if m.contradiction: print(f"     Reason: {m.reason}")
                    
                    # Entities zeigen (via Engine)
                    print("     Entities (Q):", [(t.text, t.ent_type) for t in m.query_sentence.tokens if t.ent_type])
                    print("     Entities (D):", [(t.text, t.ent_type) for t in m.doc_sentence.tokens if t.ent_type])
                    
                    # Alignment Details
                    print("     Alignment:")
                    for t in m.token_alignment:
                        if t.match_type != "EXACT":
                            tgt = t.doc_token.text if t.doc_token else "---"
                            print(f"       {t.query_token.text:<15} -> {tgt:<15} [{t.match_type}] score={t.score:.2f} sp={t.query_token.sparse_weight:.2f}")

        print(f"\nPAWS-X Ergebnis: {correct}/{total} ({correct/total:.1%})")
        results.append(("PAWS-X", correct/total))

    # 3. XNLI Test
    if os.path.exists(paths["xnli"]):
        print(f"\n--- Starte XNLI (de) Benchmark ---")
        df_xnli = load_xnli(paths["xnli"], limit=20)
        
        correct = 0
        total = 0
        
        for _, row in df_xnli.iterrows():
            total += 1
            q, d = row['sentence1'], row['sentence2']
            expected_contradiction = (row['gold_label'] == 'contradiction')
            
            res = analyzer.analyze(q, d)
            actual_contradiction = any(m.contradiction for m in res.matches)
            
            is_correct = (actual_contradiction == expected_contradiction)
            if is_correct: 
                correct += 1
            else:
                print(f"\n[FEHLER] #{total}")
                print(f"  Q: {q}")
                print(f"  D: {d}")
                print(f"  Erwartet: {row['gold_label']} | Ist: {'contradiction' if actual_contradiction else 'other'}")
                
                if not res.matches:
                    print("  -> KEIN SATZ-MATCH GEFUNDEN (< 0.40)")
                
                for m in res.matches:
                    print(f"  -> Match Score: {m.dense_score:.4f} | Contradiction: {m.contradiction}")
                    if m.contradiction: print(f"     Reason: {m.reason}")
                    print("     Alignment (Abweichungen):")
                    for t in m.token_alignment:
                        if t.match_type != "EXACT":
                            tgt = t.doc_token.text if t.doc_token else "---"
                            print(f"       {t.query_token.text:<15} -> {tgt:<15} [{t.match_type}] score={t.score:.2f}")

        print(f"\nXNLI Ergebnis: {correct}/{total} ({correct/total:.1%})")
        results.append(("XNLI", correct/total))

    print("\n=== ZUSAMMENFASSUNG ===")
    for name, acc in results:
        print(f"{name}: {acc:.1%}")

if __name__ == "__main__":
    try:
        run_benchmark()
    except Exception as e:
        print(f"Fehler: {e}")
        import traceback
        traceback.print_exc()
