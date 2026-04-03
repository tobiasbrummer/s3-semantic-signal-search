import torch
import os
import fitz
import numpy as np
import re
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from scipy.spatial.distance import cosine
from fastdtw import fastdtw

# ======================================================
# 1. DER PDF LOADER (Die Daten-Schaufel)
# ======================================================
class PDFIngestor:
    def __init__(self):
        pass

    def load_pdf(self, file_path):
        """Liest PDF und behält Seiten-Mapping."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Datei nicht gefunden: {file_path}")

        print(f"Lade PDF: {file_path}...")
        doc = fitz.open(file_path)
        
        full_text = ""
        page_map = [] # Speichert (Start-Index, End-Index, Seitenzahl)
        
        current_pos = 0
        
        for i, page in enumerate(doc):
            # Text extrahieren
            text = page.get_text("text")
            
            # Einfaches Cleaning (Ligaturen, Hyphenation am Zeilenende)
            text = text.replace('-\n', '').replace('\n', ' ')
            
            # Wir fügen den Text an den Stream an
            full_text += text + " "
            
            # Mapping speichern: Von wo bis wo im String ist diese Seite?
            end_pos = current_pos + len(text) + 1
            page_map.append({
                "page": i + 1,
                "start": current_pos,
                "end": end_pos
            })
            current_pos = end_pos
            
        print(f"   -> {len(doc)} Seiten eingelesen ({len(full_text)} Zeichen).")
        return full_text, page_map

    def find_page(self, char_index, page_map):
        """Gibt die Seite für einen bestimmten Zeichen-Index zurück."""
        for entry in page_map:
            if entry["start"] <= char_index < entry["end"]:
                return entry["page"]
        return -1

# ======================================================
# HELPER: VISUALISIERUNG
# ======================================================
def plot_search_landscape(doc_len, matches, density_curve, peak_idx, filename="debug_scan.png"):
    plt.figure(figsize=(15, 6))
    
    # Trennen nach Wichtigkeit
    x_weak, y_weak = [], []
    x_strong, y_strong = [], []
    
    for m in matches:
        # m = (Global_Idx, Raw_Score, Token_Text, Weight)
        if m[3] > 1.0: # Hohes Gewicht (unsere SPLADE Keywords)
            x_strong.append(m[0])
            y_strong.append(m[1])
        else:
            x_weak.append(m[0])
            y_weak.append(m[1])
    
    plt.scatter(x_weak, y_weak, alpha=0.3, s=10, c='gray', label='Common Tokens')
    plt.scatter(x_strong, y_strong, alpha=0.8, s=40, c='red', marker='x', label='Key Terms (Weighted)')
    
    # Dichte Kurve
    if max(density_curve) > 0:
        norm_curve = np.array(density_curve) / max(density_curve)
        # Skalieren für Plot
        norm_curve = (norm_curve * 0.4) + 0.5 
        x_curve = np.linspace(0, doc_len, len(density_curve))
        plt.plot(x_curve, norm_curve, color='blue', linewidth=2, label='Energy Signal')
    
    plt.axvline(x=peak_idx, color='green', linestyle='--', label='WINNER')
    
    plt.title(f"Sonar Scan: {len(x_strong)} Key-Matches gefunden")
    plt.xlabel("Dokument Position (Tokens)")
    plt.ylabel("Match Score")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    print(f"   -> Debug-Plot gespeichert: '{filename}'")
    plt.savefig(filename)
    plt.close()

# ======================================================
# ENGINE
# ======================================================
class AuditEngine:
    def __init__(self):
        print("Lade Modelle...")
        # JUDGE (Granite)
        self.j_tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-embedding-278m-multilingual")
        self.j_model = AutoModel.from_pretrained("ibm-granite/granite-embedding-278m-multilingual")
        self.j_model.eval()
        self.axis = self._calc_axis()

        # SCANNER (mDeBERTa - Multilingual)
        self.s_tokenizer = AutoTokenizer.from_pretrained("microsoft/mdeberta-v3-base", use_fast=False)
        self.s_model = AutoModel.from_pretrained("microsoft/mdeberta-v3-base")
        self.s_model.eval()
        self.sp_token = '\u2581' 

        # SPECTRAL WEIGHTS (SPLADE)
        print("Lade SPLADE (Spectral Weights)...")
        self.splade_name = "naver/splade-cocondenser-ensembledistil"
        self.splade_tokenizer = AutoTokenizer.from_pretrained(self.splade_name)
        self.splade_model = AutoModelForMaskedLM.from_pretrained(self.splade_name)
        self.splade_model.eval()

    def _calc_axis(self):
        def get_vec(w):
            inp = self.j_tokenizer(w, return_tensors="pt")
            with torch.no_grad(): out = self.j_model(**inp)
            emb = out.last_hidden_state.mean(dim=1).squeeze().numpy()
            return emb / np.linalg.norm(emb)
        
        # Multilingual Axis (English + German)
        logic_words = ["not", "no", "never", "false", "error", "except", "nicht", "kein", "fehler", "falsch", "ausnahme"]
        struct_words = ["however", "but", "and", "therefore", "although", "aber", "und", "oder", "deshalb", "obwohl", "der", "die", "das"]
        
        logic = np.mean([get_vec(w) for w in logic_words], axis=0)
        struct = np.mean([get_vec(w) for w in struct_words], axis=0)
        ax = logic - struct
        return ax / np.linalg.norm(ax)

    def judge_word(self, word):
        inp = self.j_tokenizer(word, return_tensors="pt")
        with torch.no_grad(): out = self.j_model(**inp)
        emb = out.last_hidden_state.mean(dim=1).squeeze().numpy()
        return np.dot(emb / np.linalg.norm(emb), self.axis)

    def _get_splade_importance(self, text):
        """Calculates token importance using SPLADE max-logits logic."""
        inputs = self.splade_tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            logits = self.splade_model(**inputs).logits
        
        # Self-Information / Importance per token
        values, _ = torch.max(logits, dim=2)
        weights = torch.log(1 + torch.relu(values)).squeeze().numpy()
        
        # Map back to words
        word_ids = inputs.word_ids()
        word_weights = {}
        tokens = self.splade_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        for i, w_id in enumerate(word_ids):
            if w_id is None: continue
            # Basic cleanup of BERT tokens
            tok = tokens[i].replace("##", "").lower()
            w = weights[i]
            
            # Keep max weight for this stem/fragment
            if tok not in word_weights or w > word_weights[tok]:
                word_weights[tok] = w
                
        return word_weights

    def get_signal(self, text):
        inputs = self.s_tokenizer(text, return_tensors="pt")
        with torch.no_grad(): outputs = self.s_model(**inputs, output_attentions=True)
        embs = outputs.last_hidden_state.squeeze(0).numpy()
        tokens = self.s_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        mask = [i for i, t in enumerate(tokens) if t not in ["[CLS]", "[SEP]"]]
        return [tokens[i] for i in mask], embs[mask]

    def scan_and_audit(self, pdf_text, page_map, query_text):
        print(f"\n>>> 1. SONAR SCAN (Sliding Window)...")
        
        # 1. QUERY VORBEREITEN
        inputs_query = self.s_tokenizer(query_text, return_tensors="pt")
        q_tokens_str = self.s_tokenizer.convert_ids_to_tokens(inputs_query["input_ids"][0])
        
        with torch.no_grad():
            embs_query = self.s_model(**inputs_query).last_hidden_state.squeeze(0)
            query_vecs = torch.nn.functional.normalize(embs_query, dim=1)

        # 2. SPECTRAL WEIGHTING (SPLADE)
        print("   -> Calculating Spectral Weights (SPLADE)...")
        splade_map = self._get_splade_importance(query_text)
        
        # DEBUG: Show what SPLADE found
        # print(f"DEBUG SPLADE: {list(splade_map.items())[:20]}")

        german_stopwords = set(["der", "die", "das", "und", "in", "von", "zu", "den", "dem", "mit", "ist", "sich", "nicht", "auch", "eine", "einer", "einem", "bei", "als", "für"])
        domain_keywords = ["zin", "künd", "frist", "gebüh", "kost", "entgelt", "höh"]

        token_weights = []
        for t in q_tokens_str:
            clean = t.replace(self.sp_token, "").strip().lower()
            
            if not clean or len(clean) < 2 or clean in german_stopwords: 
                w = 0.1 # Force low weight for stopwords
            else:
                # Look up in SPLADE map (fuzzy match)
                w = 0.2
                for s_tok, s_weight in splade_map.items():
                    if s_tok in clean or clean in s_tok:
                        w = max(w, s_weight)
                
                # Boost if really high SPLADE score
                if w > 1.5: w = w * 2.0 

                # MANUAL INJECTION (Hybrid Approach)
                # Ensure critical German legal terms are never ignored by English SPLADE
                if any(k in clean for k in domain_keywords):
                    w = max(w, 8.0) # Override with high urgency

            token_weights.append(w)
        
        print(f"   -> Top Keywords (Spectral): {[t for t,w in zip(q_tokens_str, token_weights) if w >= 3.0]}")

        # 2. SLIDING WINDOW SCAN
        # Wir tokenisieren erst ALLES (auf CPU geht das), um die Länge zu kennen
        full_encoding = self.s_tokenizer(pdf_text, add_special_tokens=False) 
        full_ids = full_encoding["input_ids"]
        total_tokens = len(full_ids)
        print(f"   -> Dokument Länge: {total_tokens} Tokens")
        
        window_size = 512 # DeBERTa Limit
        stride = 400      # Overlap sorgt für Kontinuität
        
        all_matches = [] # (Global_Idx, Score, Text, Weight)
        
        for start_idx in range(0, total_tokens, stride):
            end_idx = min(start_idx + window_size, total_tokens)
            if start_idx >= total_tokens: break
            
            # Batch vorbereiten
            chunk_ids = torch.tensor([full_ids[start_idx:end_idx]])
            
            with torch.no_grad():
                out = self.s_model(chunk_ids)
                chunk_embs = out.last_hidden_state.squeeze(0) # [W, 768]
                chunk_vecs = torch.nn.functional.normalize(chunk_embs, dim=1)
                
            # Matrix Match (Query vs Window)
            sim_matrix = torch.matmul(query_vecs, chunk_vecs.T) # [Q, W]
            
            best_scores, best_indices = torch.max(sim_matrix, dim=1)
            
            # Ergebnisse sammeln und auf Global Index mappen
            for q_i, raw_score in enumerate(best_scores):
                weight = token_weights[q_i]
                score_val = raw_score.item()
                
                # Nur speichern wenn halbwegs relevant
                if score_val > 0.45:
                    local_idx = best_indices[q_i].item()
                    global_idx = start_idx + local_idx
                    
                    # Gewichteten Score berechnen
                    all_matches.append((global_idx, score_val, q_tokens_str[q_i], weight))
            
            if end_idx == total_tokens: break

        # 3. DICHTE-ANALYSE (Global)
        if not all_matches:
            print("   -> KEINE MATCHES GEFUNDEN.")
            return

        # Wir bauen das Histogramm, aber gewichtet!
        # Ein "Match" zählt 'weight' mal in seinem Bin
        
        energy_curve = np.zeros(total_tokens + 200)
        
        for m in all_matches:
            g_idx, score, txt, w = m
            # Kernel splatting
            start_k = max(0, g_idx - 60)
            end_k = min(total_tokens, g_idx + 60)
            
            # Addiere Energie: Score * Gewicht
            impact = score * w
            energy_curve[start_k:end_k] += impact

        peak_idx = np.argmax(energy_curve)
        
        plot_search_landscape(total_tokens, all_matches, energy_curve, peak_idx)

        # Mapping Seite
        estimated_char_pos = peak_idx * 4 
        page_num = "?"
        if page_map:
             for entry in page_map:
                 if entry["start"] <= estimated_char_pos < entry["end"]:
                     page_num = entry["page"]; break

        print(f"   -> Fokus auf Seite {page_num} (Peak bei Index {peak_idx})")
        
        # 4. EXTRAKTION (mit Puffer für Re-Tokenization)
        # Wir müssen den Text aus dem Original holen, nicht aus den IDs, 
        # weil IDs -> Text -> IDs oft Formatierung verliert.
        # Einfachheitshalber: Wir nehmen die IDs und decodieren.
        
        q_len = len(q_tokens_str)
        snip_start = max(0, peak_idx - int(q_len * 1.5))
        snip_end = min(total_tokens, peak_idx + int(q_len * 4.0))
        
        snippet_ids = full_ids[snip_start:snip_end]
        snippet_text = self.s_tokenizer.decode(snippet_ids, skip_special_tokens=True)
        
        print(f"   -> Erfasster Bereich: \"{snippet_text.replace(chr(10), ' ')[:80]}...\"")

        print(f"\n>>> 2. MICRO AUDIT...")
        self._run_zipper(query_text, snippet_text)

    def _run_zipper(self, text_a, text_b):
        # Zipper Logic (identisch)
        toks_a, curve_a = self.get_signal(text_a)
        toks_b, curve_b = self.get_signal(text_b)
        dist, raw_path = fastdtw(curve_a, curve_b, dist=cosine)
        best = {}
        for i, j in raw_path:
            sim = 1.0 - cosine(curve_a[i], curve_b[j])
            if i not in best or sim > best[i][1]: best[i] = (j, sim)
        used_b = set([v[0] for k, v in best.items()])
        
        # --- BOUNDARY LOGIC ---
        # Filter "best" to only robust matches to find true start/end
        robust_matches = {k: v for k, v in best.items() if v[1] > 0.65}
        
        start_threshold_idx = 0
        stop_threshold_idx = len(toks_b)

        if robust_matches:
            # First query token that matched well
            first_q_match = min(robust_matches.keys())
            start_threshold_idx = max(0, robust_matches[first_q_match][0] - 2)
            
            # Last query token that matched well
            last_q_match = max(robust_matches.keys())
            stop_threshold_idx = min(len(toks_b), robust_matches[last_q_match][0] + 5)
            
            stop_threshold_idx = min(len(toks_b), robust_matches[last_q_match][0] + 5)
            
        print(f"   -> Fokus-Bereich: Tokens {start_threshold_idx} bis {stop_threshold_idx} (von {len(toks_b)})")

        anomalies = []
        curr_word = ""; is_ins = False
        
        print(f"{'Typ':<10} | {'Wort':<15} | {'Score':<6}")
        print("-" * 40)

        for j in range(len(toks_b)):
            # OUT OF BOUNDS IGNORE
            if j < start_threshold_idx: continue
            if j >= stop_threshold_idx: break

            t = toks_b[j]; is_new = t.startswith(self.sp_token)
            if is_new:
                if curr_word: self._check(curr_word, is_ins, anomalies)
                curr_word = t.replace(self.sp_token, ""); is_ins = (j not in used_b)
            else:
                curr_word += t; 
                if j in used_b: is_ins = False
        if curr_word: self._check(curr_word, is_ins, anomalies)
        
        analyzed_len = stop_threshold_idx - start_threshold_idx
        if anomalies: 
            ratio = (len(anomalies) / max(1, analyzed_len)) * 100
            print(f"\n>> ALARM: {len(anomalies)} Treffer! (Diff-Ratio: {ratio:.1f}%)")
        else: print("\n>> Alles OK.")

    def _check(self, word, is_ins, anomalies):
        if not is_ins or len(word) < 2 or not re.search(r'[a-zA-Z]', word): return
        score = self.judge_word(word)
        if score > 0.15: 
            print(f"\033[91mLOGIC      | {word:<15} | {score:+.2f}\033[0m")
            anomalies.append(word)
        elif score > 0.05 and score <= 0.15:
             print(f"\033[93mCONTENT    | {word:<15} | {score:+.2f}\033[0m")
             anomalies.append(word)

if __name__ == "__main__":
    # Test mit VIEL Rauschen, um Sliding Window zu beweisen
    #noise = "Das ist ein Text über Bankdienstleistungen und Entgelte. " * 600 # > 4096 Tokens!
    #target = "Die Bank teilt dem Kunden Änderungen von Zinsen nicht mit. Bei einer Erhöhung kann der Kunde innerhalb von sechs Minuten kündigen."
    #noise_after = " Ende des Dokuments. " * 100
    #
    #full_text = noise + target + noise_after
    
    ingestor = PDFIngestor()
    full_text, page_map = ingestor.load_pdf("/var/home/t0bybr/containers/s3/test_contract.pdf")
   
    query_text = "Die Bank wird dem Kunden Änderungen von Zinsen mitteilen. Bei einer Erhöhung kann der Kunde, sofern nichts anderes vereinbart ist, die davon betroffene Kreditvereinbarung innerhalb von sechs Wochen nach der Bekanntgabe der Änderung mit sofortiger Wirkung kündigen."
    engine = AuditEngine()
    engine.scan_and_audit(full_text, page_map, query_text)

