import fitz
import torch
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine
from fastdtw import fastdtw

# ======================================================
# HELPER: VISUALISIERUNG
# ======================================================
def plot_search_landscape(doc_len, matches, density_curve, peak_idx, filename="debug_scan.png"):
    """Erstellt eine Grafik, die zeigt, wo die Query-Wörter gelandet sind."""
    plt.figure(figsize=(15, 6))
    
    # 1. Die Nadeln (Wo hat welches Wort gematcht?)
    # matches ist eine Liste von (Doc_Index, Score, Word_Text)
    x = [m[0] for m in matches]
    y = [m[1] for m in matches] # Scores
    
    plt.scatter(x, y, alpha=0.5, s=20, c='blue', label='Token Matches (Weighted)')
    
    # 2. Die Dichte-Kurve (Wo ist der Haufen?)
    # Wir normieren die Kurve, damit sie ins Bild passt
    if max(density_curve) > 0:
        norm_curve = np.array(density_curve) / max(density_curve)
        # Skalieren auf Y-Achse 0.5 - 1.0 für Sichtbarkeit
        norm_curve = (norm_curve * 0.5) + 0.4
        
        # X-Achse für Kurve skalieren
        x_curve = np.linspace(0, doc_len, len(density_curve))
        plt.plot(x_curve, norm_curve, color='red', linewidth=2, label='Density Signal')
    
    plt.axvline(x=peak_idx, color='green', linestyle='--', label='Selected Focus')
    
    plt.title("Sonar Scan Debugger: Wo liegen die Query-Treffer?")
    plt.xlabel("Dokument Token Position")
    plt.ylabel("Match Stärke / Dichte")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    print(f"   -> Speichere Debug-Plot als '{filename}'...")
    plt.savefig(filename)
    plt.close()

# ======================================================
# ENGINE
# ======================================================
class AuditEngine:
    def __init__(self):
        # Wir laden Judge und DeBERTa wie gehabt
        from transformers import AutoTokenizer, AutoModel # Local import safety
        
        print("Lade Modelle...")
        # JUDGE (Granite)
        self.judge_name = "ibm-granite/granite-embedding-278m-multilingual"
        self.j_tokenizer = AutoTokenizer.from_pretrained(self.judge_name)
        self.j_model = AutoModel.from_pretrained(self.judge_name)
        self.j_model.eval()
        
        # Seeds berechnen
        self.axis = self._calc_axis()

        # SCANNER (DeBERTa)
        self.s_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large", use_fast=False)
        self.s_model = AutoModel.from_pretrained("microsoft/deberta-v3-large")
        self.s_model.eval()
        self.sp_token = '\u2581' 

    def _calc_axis(self):
        # Kurzfassung der Judge-Logik
        def get_vec(w):
            inp = self.j_tokenizer(w, return_tensors="pt")
            with torch.no_grad(): out = self.j_model(**inp)
            emb = out.last_hidden_state.mean(dim=1).squeeze().numpy()
            return emb / np.linalg.norm(emb)
        
        logic = np.mean([get_vec(w) for w in ["not", "no", "never", "false", "error"]], axis=0)
        struct = np.mean([get_vec(w) for w in ["however", "but", "and", "therefore"]], axis=0)
        axis = logic - struct
        return axis / np.linalg.norm(axis)

    def judge_word(self, word):
        # Einfache Projektion
        inp = self.j_tokenizer(word, return_tensors="pt")
        with torch.no_grad(): out = self.j_model(**inp)
        emb = out.last_hidden_state.mean(dim=1).squeeze().numpy()
        emb = emb / np.linalg.norm(emb)
        return np.dot(emb, self.axis)

    def get_signal(self, text):
        inputs = self.s_tokenizer(text, return_tensors="pt")
        with torch.no_grad(): outputs = self.s_model(**inputs, output_attentions=True)
        embs = outputs.last_hidden_state.squeeze(0).numpy()
        tokens = self.s_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        mask = [i for i, t in enumerate(tokens) if t not in ["[CLS]", "[SEP]"]]
        return [tokens[i] for i in mask], embs[mask]

    def scan_and_audit(self, pdf_text, page_map, query_text):
        print(f"\n>>> 1. SONAR SCAN (Hybrid Weighted)...")
        
        # Wir scannen einen großen Bereich
        scan_text = pdf_text[:60000] 
        
        inputs_doc = self.s_tokenizer(scan_text, return_tensors="pt", truncation=True, max_length=4096)
        inputs_query = self.s_tokenizer(query_text, return_tensors="pt")
        
        # Token Strings für Gewichtung holen
        q_tokens_str = self.s_tokenizer.convert_ids_to_tokens(inputs_query["input_ids"][0])
        
        with torch.no_grad():
            embs_doc = self.s_model(**inputs_doc).last_hidden_state.squeeze(0)   # [N, 768]
            embs_query = self.s_model(**inputs_query).last_hidden_state.squeeze(0) # [M, 768]

        # --- A) GEWICHTUNG (Poor Man's SPLADE) ---
        # Wir geben seltenen/langen Wörtern mehr Gewicht.
        # "die" (3 chars) -> Gewicht 1
        # "Kreditvereinbarung" (18 chars) -> Gewicht 6
        # Das simuliert "Keyword Matching" innerhalb der Vektorsuche.
        
        token_weights = []
        for t in q_tokens_str:
            clean = t.replace(self.sp_token, "").strip()
            if not clean or not re.search(r'[a-zA-Z]', clean):
                w = 0.1 # Rauschen ignorieren
            elif clean.lower() in ["der", "die", "das", "und", "in", "von", "zu", "den"]:
                w = 0.5 # Stopwords bestrafen
            else:
                # Länge als Proxy für Spezifität (Heuristik)
                w = min(5.0, len(clean) / 3.0) 
                # Boost für Key-Begriffe
                if clean.lower() in ["nicht", "kein", "ohne"]: w = 5.0 # Negationen sind wichtig!
            
            token_weights.append(w)
        
        token_weights = torch.tensor(token_weights).unsqueeze(1) # [M, 1]

        # --- B) MATRIX MATCHING ---
        doc_vecs = torch.nn.functional.normalize(embs_doc, dim=1)
        query_vecs = torch.nn.functional.normalize(embs_query, dim=1)
        
        # Sim Matrix: [M, N]
        sim_matrix = torch.matmul(query_vecs, doc_vecs.T)
        
        # --- C) WEIGHTED VOTING ---
        # Für jedes Query-Token finden wir den besten Match im Doc
        best_scores, best_indices = torch.max(sim_matrix, dim=1)
        
        matches = []
        relevant_indices = []
        
        for i, raw_score in enumerate(best_scores):
            # Wir nehmen nur Matches, die halbwegs valide sind (>0.5)
            # Aber wir multiplizieren den Score mit dem Wichtigkeit des Wortes!
            
            weight = token_weights[i].item()
            weighted_score = raw_score.item() * weight
            
            idx = best_indices[i].item()
            
            # Wir speichern alle Matches für den Plot
            if raw_score > 0.4:
                matches.append((idx, raw_score.item(), q_tokens_str[i]))
                
                # Für die Dichte-Kurve nutzen wir den gewichteten Score
                # Wir fügen den Index 'weight'-mal hinzu (oder nutzen weights in histogram)
                relevant_indices.append((idx, weight))

        # --- D) DENSITY CURVE ---
        doc_len = len(doc_vecs)
        # Wir bauen eine "Energiekurve" über das Dokument
        energy_curve = np.zeros(doc_len)
        
        # Kernel Density Estimation (simpel)
        # Jeder Treffer strahlt Energie auf seine Nachbarn aus
        for idx, strength in relevant_indices:
            start = max(0, idx - 50)
            end = min(doc_len, idx + 50)
            energy_curve[start:end] += strength # Rechteck-Kernel

        peak_idx = np.argmax(energy_curve)
        
        # PLOT ERSTELLEN
        plot_search_landscape(doc_len, matches, energy_curve, peak_idx)

        # Mapping Seite
        estimated_char_pos = peak_idx * 4 
        page_num = "?"
        if page_map:
             for entry in page_map:
                 if entry["start"] <= estimated_char_pos < entry["end"]:
                     page_num = entry["page"]; break

        print(f"   -> Fokus auf Seite {page_num} (Energy Peak bei {peak_idx})")
        
        # --- E) EXTRAKTION ---
        q_len = len(query_vecs)
        start = max(0, peak_idx - int(q_len * 1.0))
        end = min(doc_len, peak_idx + int(q_len * 5.0))
        
        snippet_ids = inputs_doc["input_ids"][0][start:end]
        snippet_text = self.s_tokenizer.decode(snippet_ids, skip_special_tokens=True)
        
        print(f"   -> Erfasster Bereich: \"{snippet_text.replace(chr(10), ' ')[:80]}...\"")

        print(f"\n>>> 2. MICRO AUDIT...")
        self._run_zipper(query_text, snippet_text)

    def _run_zipper(self, text_a, text_b):
        # (Hier dein Zipper Code wie vorher, unverändert gut)
        # Ich kürze es für die Lesbarkeit ab, nutze den Code von vorhin!
        toks_a, curve_a = self.get_signal(text_a)
        toks_b, curve_b = self.get_signal(text_b)
        
        dist, raw_path = fastdtw(curve_a, curve_b, dist=cosine)
        best = {}
        for i, j in raw_path:
            sim = 1.0 - cosine(curve_a[i], curve_b[j])
            if i not in best or sim > best[i][1]: best[i] = (j, sim)
        used_b = set([v[0] for k, v in best.items()])
        
        anomalies = []
        current_word = ""
        is_insertion = False
        
        print(f"{'Typ':<10} | {'Wort':<15} | {'Score':<6}")
        print("-" * 40)

        for j in range(len(toks_b)):
            t = toks_b[j]
            is_new = t.startswith(self.sp_token)
            if is_new:
                if current_word: self._check(current_word, is_insertion, anomalies)
                current_word = t.replace(self.sp_token, "")
                is_insertion = (j not in used_b)
            else:
                current_word += t
                if j in used_b: is_insertion = False
        if current_word: self._check(current_word, is_insertion, anomalies)
        
        if anomalies: print(f"\n>> ALARM: {len(anomalies)} Treffer!")
        else: print("\n>> Alles OK.")

    def _check(self, word, is_ins, anomalies):
        if not is_ins or len(word) < 2 or not re.search(r'[a-zA-Z]', word): return
        score = self.judge_word(word)
        if score > 0.10: 
            print(f"\033[91mLOGIC      | {word:<15} | {score:+.2f}\033[0m")
            anomalies.append(word)
        elif score > -0.05 and score <= 0.10: # Content Bereich
             print(f"\033[93mCONTENT    | {word:<15} | {score:+.2f}\033[0m")
             anomalies.append(word)

# MAIN SIMULATION
if __name__ == "__main__":
    # PDF Loader (Fake für Demo)
    fake_map = [{"page": 5, "start": 10000, "end": 60000}]
    
    # Simuliere den Text mit dem Fehler
    # Wir fügen VIEL Rauschen davor ein, um den Scanner zu testen
    noise = "Das ist ein Text über Bankdienstleistungen und Entgelte. " * 500
    target = "Die Bank teilt dem Kunden Änderungen von Zinsen nicht mit. Bei einer Erhöhung kann der Kunde innerhalb von sechs Minuten kündigen."
    noise_after = " Weitere Regelungen zu Zinsen finden Sie in Anlage 4." * 500
    
    full_text = noise + target + noise_after
    
    engine = AuditEngine()
    engine.scan_and_audit(
        full_text, 
        fake_map, 
        "Die Bank wird dem Kunden Änderungen von Zinsen mitteilen. Bei einer Erhöhung kann der Kunde innerhalb von sechs Wochen kündigen."
    )
