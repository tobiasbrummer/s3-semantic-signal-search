import fitz  # PyMuPDF
import torch
import numpy as np
import re
import os
from transformers import AutoTokenizer, AutoModel
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
# 2. DER SEMANTIC JUDGE (Das Gehirn)
# ======================================================
class SemanticJudge:
    def __init__(self):
        # Wir nehmen das IBM Granite Modell für scharfe Trennung
        model_id = "ibm-granite/granite-embedding-278m-multilingual" 
        print(f"Lade Judge ({model_id})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.model.eval()
        
        # Starke Seeds
        vec_logic = self._encode_concept(["not", "no", "never", "none", "without", "denial", "refusal", "false", "error"])
        vec_struct = self._encode_concept(["however", "therefore", "thus", "moreover", "meanwhile", "whereas", "furthermore"])
        
        # Kontrast-Achse
        axis = vec_logic - vec_struct
        self.semantic_axis = axis / np.linalg.norm(axis)

    def _encode_concept(self, words):
        vecs = [self._get_vector(w) for w in words]
        return np.mean(vecs, axis=0)

    def _get_vector(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_emb = (sum_embeddings / sum_mask).squeeze(0).numpy()
        return mean_emb / np.linalg.norm(mean_emb)

    def evaluate(self, word):
        return np.dot(self._get_vector(word), self.semantic_axis)

# ======================================================
# 3. DIE ENGINE (Sonar + Zipper)
# ======================================================
class AuditEngine:
    def __init__(self):
        self.judge = SemanticJudge()
        print("Lade Signal Processor (DeBERTa v3)...")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large", use_fast=False)
        self.model = AutoModel.from_pretrained("microsoft/deberta-v3-large")
        self.model.eval()
        self.sp_token = '\u2581' 

    def get_signal(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad(): outputs = self.model(**inputs, output_attentions=True)
        embs = outputs.last_hidden_state.squeeze(0).numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        mask = [i for i, t in enumerate(tokens) if t not in ["[CLS]", "[SEP]"]]
        att = torch.stack(outputs.attentions).mean(dim=(0, 2)).squeeze(0).sum(dim=0).numpy()
        vol = att[mask]
        if vol.max() > 0: vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-9)
        return [tokens[i] for i in mask], embs[mask], vol

    def clean_token(self, t): return t.replace(self.sp_token, '').strip()

    def scan_and_audit(self, pdf_text, page_map, query_text):
        print(f"\n>>> 1. SONAR SCAN (Density Voting)...")
        
        scan_text = pdf_text[:50000] 
        
        # 1. ENCODING
        inputs_doc = self.tokenizer(scan_text, return_tensors="pt", truncation=True, max_length=4096)
        inputs_query = self.tokenizer(query_text, return_tensors="pt")
        
        with torch.no_grad():
            embs_doc = self.model(**inputs_doc).last_hidden_state # [1, N, 768]
            embs_query = self.model(**inputs_query).last_hidden_state # [1, M, 768]

        # Squeeze batch dimension
        doc_vecs = embs_doc.squeeze(0)   # [N, 768]
        query_vecs = embs_query.squeeze(0) # [M, 768]

        # 2. MATRIX MATCHING (Alle gegen Alle)
        # Wir berechnen die Ähnlichkeit von JEDEM Query-Token zu JEDEM Doc-Token
        # Matrix Form: [Anzahl_Query_Tokens x Anzahl_Doc_Tokens]
        
        # Normalisierung für Cosine Sim
        doc_vecs = torch.nn.functional.normalize(doc_vecs, dim=1)
        query_vecs = torch.nn.functional.normalize(query_vecs, dim=1)
        
        # Matrix Multiplikation
        sim_matrix = torch.matmul(query_vecs, doc_vecs.T) # [M, N]
        
        # 3. BEST MATCH FINDING
        # Für jedes Wort in der Query: Wo ist der beste Partner im Text?
        # Wir ignorieren schwache Matches (< 0.4), das sind nur Stopwords ("der", "und")
        
        best_match_scores, best_match_indices = torch.max(sim_matrix, dim=1)
        
        # Filter: Nur starke Nadeln zählen
        relevant_indices = []
        for i, score in enumerate(best_match_scores):
            if score > 0.60: # Nur wenn wir uns sicher sind
                idx = best_match_indices[i].item()
                relevant_indices.append(idx)
                
        if not relevant_indices:
            print("   -> Keine sicheren Anker gefunden. Abbruch.")
            return

        # 4. DICHTE MESSUNG (Histogramm)
        # Wir schauen, wo sich die Indizes häufen.
        # Wir teilen das Dokument in Eimer (Bins) von z.B. 100 Token Größe
        doc_len = inputs_doc["input_ids"].shape[1]
        hist_bins = range(0, doc_len + 100, 50) # Alle 50 Token ein Bin
        
        counts, bin_edges = np.histogram(relevant_indices, bins=hist_bins)
        
        # Der Eimer mit den meisten Nadeln gewinnt
        peak_bin_idx = np.argmax(counts)
        peak_start_idx = bin_edges[peak_bin_idx]
        
        # Mapping Seite
        estimated_char_pos = peak_start_idx * 4 
        page_num = "?"
        if page_map:
             for entry in page_map:
                 if entry["start"] <= estimated_char_pos < entry["end"]:
                     page_num = entry["page"]; break

        print(f"   -> Fokus auf Seite {page_num} (Nadel-Dichte Peak bei Index {peak_start_idx})")
        
        # 5. EXTRAKTION
        # Wir nehmen das gefundene Bin und erweitern es großzügig
        # Wir wissen, die Query ist M Tokens lang.
        q_len = len(query_vecs)
        
        # Wir zentrieren um den Dichte-Peak
        center = peak_start_idx + 25 # Mitte des 50er Bins
        start = max(0, center - int(q_len * 1.5))
        end = min(doc_len, center + int(q_len * 4.0)) # Viel Platz nach hinten für Verschiebungen
        
        snippet_ids = inputs_doc["input_ids"][0][start:end]
        snippet_text = self.tokenizer.decode(snippet_ids, skip_special_tokens=True)
        
        # Vorschau
        print(f"   -> Erfasster Bereich: \"{snippet_text.replace(chr(10), ' ')[:80]}...\"")

        print(f"\n>>> 2. MICRO AUDIT...")
        self._run_zipper(query_text, snippet_text)

    def _run_zipper(self, text_a, text_b):
        toks_a, curve_a, vol_a = self.get_signal(text_a)
        toks_b, curve_b, vol_b = self.get_signal(text_b)
        
        dist, raw_path = fastdtw(curve_a, curve_b, dist=cosine)
        
        best = {}
        for i, j in raw_path:
            sim = 1.0 - cosine(curve_a[i], curve_b[j])
            if i not in best or sim > best[i][1]: best[i] = (j, sim)
        
        used_b_indices = set([v[0] for k, v in best.items()])
        
        print(f"{'Typ':<10} | {'Wort':<15} | {'Kontext (Vorschau)':<40}")
        print("-" * 80)
        
        anomalies = []
        
        current_word = ""
        current_indices = [] # Speichert Token-Indizes des aktuellen Worts
        is_insertion = False
        
        for j in range(len(toks_b)):
            raw_token = toks_b[j]
            is_new_word = raw_token.startswith(self.sp_token)
            
            if is_new_word:
                if current_word:
                    # Altes Wort prüfen
                    self._check_anomaly(current_word, is_insertion, current_indices, toks_b, anomalies)
                
                current_word = raw_token.replace(self.sp_token, "")
                current_indices = [j]
                is_insertion = (j not in used_b_indices)
            else:
                current_word += raw_token
                current_indices.append(j)
                if j in used_b_indices: is_insertion = False

        if current_word:
            self._check_anomaly(current_word, is_insertion, current_indices, toks_b, anomalies)

        if anomalies:
            print(f"\n>> ALARM: {len(anomalies)} Abweichungen!")
        else:
            print("\n>> Alles OK.")

    def _check_anomaly(self, word, is_insertion, indices, all_tokens, anomalies):
        if not is_insertion: return
        if len(word) < 2 or not re.search(r'[a-zA-Z]', word): return

        # 1. KONTEXT BAUEN
        # Wir holen 3 Wörter davor und danach für die Anzeige
        start_idx = max(0, indices[0] - 5)
        end_idx = min(len(all_tokens), indices[-1] + 6)
        
        context_tokens = all_tokens[start_idx:end_idx]
        context_str = "".join(context_tokens).replace(self.sp_token, " ").strip()
        # Markiere das Wort im String (simpel)
        context_str = context_str.replace(word, f"[[ {word} ]]")

        # 2. BEWERTUNG
        contrast_score = self.judge.evaluate(word)
        
        status = ""
        style = ""
        
        # LOGIK 1: Ist es Struktur? (sehr negativ) -> IGNORE
        if contrast_score < 0.05:
            return # Wir zeigen Struktur gar nicht an (Noise Filter)

        # LOGIK 2: Ist es Logik? (sehr positiv) -> ALARM
        if contrast_score > 0.10:
            status = "LOGIC"
            style = "\033[91m" # Rot
            
        # LOGIK 3: Ist es dazwischen? -> CONTENT CHANGE (Minuten vs Wochen)
        # Wenn es keine Struktur ist, aber auch keine Verneinung, ist es eine inhaltliche Änderung!
        else:
            status = "CONTENT"
            style = "\033[93m" # Gelb (Warnung)

        print(f"{style}{status:<10} | {word:<15} | ...{context_str[-40:]:<40}\033[0m")
        anomalies.append(word)

# ======================================================
# RUN
# ======================================================
if __name__ == "__main__":
    # 1. Wir brauchen ein Dummy-PDF zum Testen
    # Erstelle eine Datei 'test_contract.pdf' oder ändere den Pfad
    
    # Simuliere das Laden (Falls du kein PDF hast, nutzen wir String-Simulation)
    use_real_pdf = True
    
    if use_real_pdf:
        ingestor = PDFIngestor()
        full_text, page_map = ingestor.load_pdf("/var/home/t0bybr/containers/s3/test_contract.pdf")
        
        engine = AuditEngine()
        engine.scan_and_audit(
            full_text, 
            page_map, 
            query_text="Die Bank wird dem Kunden Änderungen von Zinsen mitteilen. Bei einer Erhöhung kann der Kunde, sofern nichts anderes vereinbart ist, die davon betroffene Kreditvereinbarung innerhalb von sechs Wochen nach der Bekanntgabe der Änderung mit sofortiger Wirkung kündigen."
        )
    else:
        print("--- SIMULATION MODUS (Kein PDF gefunden) ---")
        # Wir simulieren das Ergebnis des PDF Loaders
        sim_text = "Standard Lease Agreement. Page 1... \n" * 20
        sim_text += "However, the tenant is allowed to keep small pets if they are cute. " # Manipulation!
        sim_text += "\n Page 2..." * 20
        
        # Fake Page Map
        sim_map = [{"page": 1, "start": 0, "end": 5000}]
        
        engine = AuditEngine()
        engine.scan_and_audit(
            sim_text, 
            sim_map, 
            query_text="The tenant is forbidden from keeping pets."
        )
