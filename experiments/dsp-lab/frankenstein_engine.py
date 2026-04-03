"""
Frankenstein Engine: Semantic Audio Search (Final Version)
----------------------------------------------------------
Eine hybride Such-Architektur, die Embeddings wie Audiosignale behandelt.

Kern-Features:
1. Fused Embeddings: Multiplikation von T5 (Dense) und SPLADE (Sparse) Gewichten.
2. Smart Compression:
   - Hard Baseline (Noise Gate) für Stille.
   - Logarithmische Dynamik-Analyse (damit laute Keywords leise Details nicht erdrücken).
   - Adaptives Gate basierend auf dem Signal-Durchschnitt.
   - GOP (Group of Pictures) Safety Net gegen zu lange Pausen.
3. ColBERT-Style Retrieval: Suche mittels Maximum Similarity (Late Interaction).

Author: User & Gemini
Date: 2024
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForMaskedLM
import numpy as np

SPIECE_UNDERLINE = "▁"

# ==============================================================================
# 1. DER KOMPRESSOR (Das Herzstück der Optimierung)
# ==============================================================================

class SmartCompressor:
    """
    Analysiert Vektor-Sequenzen wie ein Audio-Signal und entscheidet,
    welche Zeitschritte (Tokens) gespeichert werden müssen.
    """
    def __init__(self, baseline=20.0, compression_ratio=0.5, max_gop=5):
        """
        Konfiguration des Kompressors.

        Args:
            baseline (float): Hard Noise Floor. Alles darunter ist garantiert Rauschen (z.B. 'und', 'der').
            compression_ratio (float): Wie viel vom Durchschnittssignal soll behalten werden? 
                                       (0.5 = Alles über 50% der Durchschnittslautstärke).
            max_gop (int): Group of Pictures Limit. Zwingt spätestens nach X gelöschten
                           Tokens zur Speicherung, um den Kontext nicht zu verlieren.
        """
        self.baseline = baseline
        self.ratio = compression_ratio
        self.max_gop = max_gop

    def compress(self, fused_emb, tokens=None, debug=False):
        """
        Führt die Kompression auf einem Dokumenten-Tensor aus.
        """
        # Daten vorbereiten (auf CPU für Logik-Operationen)
        vecs = fused_emb[0].cpu().float()
        norms = torch.norm(vecs, dim=1) # "Lautstärke" jedes Tokens
        seq_len = vecs.shape[0]
        
        # --- SCHRITT A: Signalanalyse (Mastering) ---
        # Wir betrachten für die Statistik nur echte Signale (über Baseline)
        active_mask = norms > self.baseline
        
        if not active_mask.any():
            if debug: print("⚠️ Dokument ist komplett still (unter Baseline).")
            return []

        active_norms = norms[active_mask]
        
        # Log-Space Berechnung: Dämpft extreme Peaks (wie 'Hasenkinder' mit 340),
        # damit leise Signale ('Kind' mit 60) nicht im Durchschnitt untergehen.
        log_norms = torch.log1p(active_norms) 
        mean_log = torch.mean(log_norms).item()
        
        # Zurückrechnen: Das ist unser dynamischer Schwellenwert für dieses Dokument
        adaptive_gate = np.expm1(mean_log) * self.ratio
        
        # Delta Threshold (Änderung): 
        # Wir setzen es dynamisch relativ zum Gate (z.B. 120% des Gates)
        adaptive_delta = adaptive_gate * 1.2
        
        if debug:
            print(f"\n🎛️ DYNAMIK-ANALYSE:")
            print(f"   Baseline (Hard Gate):   {self.baseline}")
            print(f"   Max Peak (Lautestes):   {torch.max(active_norms):.1f}")
            print(f"   Log-Average (Mitte):    {np.expm1(mean_log):.1f}")
            print(f"   -> ADAPTIVES GATE:      {adaptive_gate:.1f}")
            print("-" * 60)

        # --- SCHRITT B: Entscheidung (Streaming) ---
        kept_vectors = []
        decoder_state = torch.zeros(vecs.shape[1]) # Simulation des DB-Zustands
        skipped_count = 0
        
        for i in range(seq_len):
            current_vec = vecs[i]
            vol = norms[i].item()
            token_str = tokens[i] if tokens else "?"
            
            should_save = False
            reason = ""
            
            # 1. Hard Baseline Check (Müll sofort weg)
            if vol < self.baseline:
                skipped_count += 1
                # Wichtig: Bei Stille "vergisst" der Decoder den Kontext
                decoder_state = torch.zeros(vecs.shape[1]) 
                continue 

            # 2. GOP Safety Net (Rettungsanker für leise Sätze)
            if skipped_count >= self.max_gop:
                should_save = True
                reason = "GOP Limit (Safety)"
            
            # 3. Intelligente Checks (Gate & Delta)
            else:
                # Ist das Signal laut genug relativ zum Dokumenten-Durchschnitt?
                if vol > adaptive_gate:
                    # Ist es neu genug? (Delta zum letzten gespeicherten Zustand)
                    diff = torch.norm(current_vec - decoder_state).item()
                    if diff > adaptive_delta:
                        should_save = True
                        reason = f"Signal ({vol:.0f} > {adaptive_gate:.0f})"
                    else:
                         reason = f"Redundant (Delta {diff:.0f})"
                else:
                    reason = f"Zu leise ({vol:.0f} < {adaptive_gate:.0f})"
            
            # --- ACTION ---
            if should_save:
                if debug: print(f"  ✅ {token_str:<15} | {vol:<6.1f} | {reason}")
                # Hier konvertieren wir zu List[float] für die DB
                kept_vectors.append(current_vec.tolist())
                decoder_state = current_vec
                skipped_count = 0
            else:
                if debug: print(f"  ❌ {token_str:<15} | {vol:<6.1f} | {reason}")
                skipped_count += 1
                
        return kept_vectors


# ==============================================================================
# 2. DER ENCODER (Modelle & Fusion)
# ==============================================================================

class FrankensteinEncoder:
    """
    Verwaltet die KI-Modelle und erzeugt die Fused Embeddings.
    """
    
    def __init__(self, t5_name, sparse_name, device="cpu"):
        self.device = device
        print(f"🏗️ Lade Frankenstein Models auf {self.device}...")
        
        # T5 (Semantik)
        self.t5_tokenizer = AutoTokenizer.from_pretrained(t5_name, legacy=False)
        self.t5_model = AutoModelForSeq2SeqLM.from_pretrained(
            t5_name, dtype=torch.float32
        ).to(self.device).eval()

        # Sparse (Gewichtung)
        self.sparse_tokenizer = AutoTokenizer.from_pretrained(sparse_name)
        self.sparse_model = AutoModelForMaskedLM.from_pretrained(sparse_name).to(self.device).eval()
        
        # Unser Kompressor
        self.compressor = SmartCompressor() 
        print("✅ Engine bereit.")

    def _get_sparse_weights(self, text):
        """
        Extrahiert Keyword-Gewichte via SPLADE.
        NEU: Filtert explizit Special Tokens (<bos>, <eos>, pads), damit diese
             nicht den Score verfälschen.
        """
        inputs = self.sparse_tokenizer(
            text, return_tensors="pt", return_offsets_mapping=True, 
            truncation=True, max_length=512, stride=64, return_overflowing_tokens=True
        ).to(self.device)
        
        # Liste der IDs, die wir ignorieren wollen (vom Sparse Tokenizer!)
        # Das sind meistens [CLS], [SEP], [PAD] beim BERT-basierten Sparse Modell
        special_ids = set(self.sparse_tokenizer.all_special_ids)

        with torch.no_grad():
            outputs = self.sparse_model(inputs.input_ids)
        
        # Log-Sättigung
        values, _ = torch.max(torch.log(1 + torch.relu(outputs.logits)), dim=-1)
        
        char_weights = []
        offset_mapping = inputs.offset_mapping.cpu()
        values = values.cpu()
        input_ids = inputs.input_ids.cpu() # Wir brauchen die IDs zum Filtern
        
        for i, offsets in enumerate(offset_mapping):
            chunk_weights = values[i]
            chunk_ids = input_ids[i]
            
            for idx, (start, end) in enumerate(offsets):
                start, end = start.item(), end.item()
                if start == end: continue 
                
                # --- FILTER ---
                # Wenn das Token ein Special Token ist, setzen wir das Gewicht auf 0
                token_id = chunk_ids[idx].item()
                if token_id in special_ids:
                    continue # Überspringen = Gewicht 0 im Alignment
                
                weight = chunk_weights[idx].item()
                char_weights.append((start, end, weight))
                
        return char_weights

    def _propagate_word_weights(self, tokens, weights):
        """Der 'Hasenkinder-Fix' + Stop-Filter."""
        
        new_weights = []
        current_word_indices = []
        current_word_weights = []
        
        def flush(idx_list, w_list, out_list):
            if not idx_list: return
            max_w = max(w_list)
            for _ in idx_list: out_list.append(max_w)

        for i, (token, weight) in enumerate(zip(tokens, weights)):
            # T5 Special Tokens erkennen (<pad>, </s>, <unk>)
            # Alles was in spitzen Klammern steht, ist verdächtig
            if token.startswith("<") and token.endswith(">"):
                new_weights.append(0.0) # Hart auf 0 setzen!
                # Buffer leeren, falls was drin war
                flush(current_word_indices, current_word_weights, new_weights)
                current_word_indices = []
                current_word_weights = []
                continue

            is_start = token.startswith(SPIECE_UNDERLINE) or i == 0
            
            if is_start:
                flush(current_word_indices, current_word_weights, new_weights)
                current_word_indices = [i]
                current_word_weights = [weight]
            else:
                current_word_indices.append(i)
                current_word_weights.append(weight)
                
        flush(current_word_indices, current_word_weights, new_weights)
        
        # Sicherheits-Check: Längen müssen passen
        if len(new_weights) != len(tokens):
            # Fallback bei Mismatch (sollte nicht passieren)
            return [0.0] * len(tokens)
            
        return new_weights

    def _create_fused_embeddings(self, text):
        """
        KERNFUNKTION: Erstellt die 'Audio-Signale' (T5 x Sparse).
        NEU: Mit Stopword-Penalty und Content-Boost für Queries!
        """
        # A. Sparse Gewichte holen
        sparse_data = self._get_sparse_weights(text)
        
        # B. T5 Encoding
        t5_in = self.t5_tokenizer(text, return_tensors="pt", return_offsets_mapping=True).to(self.device)
        with torch.no_grad():
            t5_emb = self.t5_model.get_encoder()(
                input_ids=t5_in.input_ids, attention_mask=t5_in.attention_mask
            ).last_hidden_state

        # C. Alignment & STOPWORD-LOGIK
        # Einfache deutsche Stopword-Liste (Top 30)
        STOPWORDS = {
            "der", "die", "das", "dem", "den", "ein", "eine", "einer", "eines",
            "und", "oder", "aber", "ist", "sind", "war", "wird",
            "mit", "bei", "von", "aus", "zu", "nach", "in", "im", "an", "am", "auf",
            "durch", "für", "um", "über", "unter", "vor", "hinter"
        }
        
        raw_weights = []
        t5_tokens = self.t5_tokenizer.convert_ids_to_tokens(t5_in.input_ids[0])
        offsets = t5_in.offset_mapping[0].cpu()

        def clean_token(t):
            return t.replace(SPIECE_UNDERLINE, '').lower()
        
        for idx, (start, end) in enumerate(offsets):
            start, end = start.item(), end.item()
            
            # Gewicht ermitteln (Max-Pooling aus Sparse)
            if start == end: 
                base_weight = 1.0
            else:
                matches = [w for s, e, w in sparse_data if max(start, s) < min(end, e)]
                base_weight = max(matches) if matches else 1.0

            # --- NEU: GEWICHTS-MANIPULATION ---
            # Wir schauen uns das Token an (bereinigt um T5 Sonderzeichen)
            token_str = clean_token(t5_tokens[idx])
            
            # Regel 1: Stopword Penalty (Weg damit!)
            if token_str in STOPWORDS:
                base_weight *= 0.1 # Nur noch 10% Kraft
            
            # Regel 2: Content Boost (Wichtiges hervorheben!)
            # Wenn es kein Stopword ist und kein Sonderzeichen, verdoppeln wir es.
            # (Wir prüfen grob, ob es Buchstaben enthält)
            elif token_str.isalpha() and len(token_str) > 3:
                base_weight *= 2.0 
            
            raw_weights.append(base_weight)

        # D. Fusion
        # Wort-Verteilung (Hasenkinder-Fix)
        optimized_weights = self._propagate_word_weights(t5_tokens, raw_weights)
        
        w_tensor = torch.tensor(optimized_weights, device=self.device, dtype=t5_emb.dtype)
        w_tensor = w_tensor.unsqueeze(0).unsqueeze(-1)
        
        fused_embeddings = t5_emb * w_tensor 
        return fused_embeddings, t5_tokens

    def encode_document(self, text, debug_compression=False):
        """Erstellt KOMPRIMIERTE Vektoren für die DB."""
        # 1. Fusion holen
        fused, tokens = self._create_fused_embeddings(text)
        # 2. Kompression anwenden
        return self.compressor.compress(fused, tokens, debug=debug_compression)

    def encode_query(self, query_text):
        """Erstellt GEWICHTETE Vektoren für die Suche (ohne Kompression)."""
        # 1. Fusion holen (Wichtig: Jetzt haben Query-Wörter auch Gewichte!)
        fused, tokens = self._create_fused_embeddings(query_text)
        # Keine Kompression für Queries, wir brauchen volle Präzision
        return fused, tokens

# ==============================================================================
# 3. DIE SUCHMASCHINE (Simulation)
# ==============================================================================

class FrankensteinSearcher:
    def __init__(self, encoder):
        self.encoder = encoder
        self.index = [] 
        
    def add_document(self, text, doc_id=None):
        vecs = self.encoder.encode_document(text)
        if not vecs: return
        self.index.append({
            "id": doc_id or len(self.index),
            "text": text,
            "vecs": torch.tensor(vecs)
        })
        
    def search(self, query, top_k=3):
        # 1. SETUP: Stopwords
        STOPWORDS = {
            "der", "die", "das", "dem", "den", "ein", "eine", "einer", "eines",
            "und", "oder", "aber", "ist", "sind", "war", "wird",
            "mit", "bei", "von", "aus", "zu", "nach", "in", "im", "an", "am", "auf",
            "durch", "für", "um", "über", "unter", "vor", "hinter", "wegen"
        }

        print(f"\n🔎 Suche nach: '{query}'")
        print(f"   ⚙️  Audio-Settings: Smart-Filter active | Semantic Thresh=0.55 | Trash Thresh=0.85")
        print("=" * 60)
        
        q_emb, q_tokens = self.encoder.encode_query(query)
        q_emb = q_emb.cpu()
        
        results = []
        
        for doc in self.index:
            d_emb = doc["vecs"].unsqueeze(0) 
            
            sim_matrix = torch.matmul(q_emb, d_emb.transpose(1, 2))
            max_scores, max_indices = torch.max(sim_matrix, dim=2) 
            
            total_score = 0
            debug_matches = []
            
            for i in range(max_scores.shape[1]):
                # A. STOPWORD FILTER
                raw_tok = q_tokens[i].replace(SPIECE_UNDERLINE, '').lower()
                
                # Leere Tokens oder Stopwords überspringen
                if not raw_tok or raw_tok in STOPWORDS:
                    continue 

                raw_dot_product = max_scores[0, i].item()
                idx = max_indices[0, i].item()
                
                q_vec = q_emb[0, i]
                d_vec = d_emb[0, idx]
                
                # B. LOG-COMPRESSION
                q_vol = torch.norm(q_vec).item()
                d_vol = torch.norm(d_vec).item()
                
                q_log = np.log1p(q_vol)
                d_log = np.log1p(d_vol)
                
                denominator = q_vol * d_vol
                cosine_sim = raw_dot_product / denominator if denominator > 0 else 0
                
                # C. DYNAMIC BAND THRESHOLDING (Der Fix!)
                
                # Alte Regel: len <= 3 (Tötete 'üll' und 'ahr')
                # Neue Regel: len < 3 (Tötet nur 'At', 'in', 'wo')
                is_trash_candidate = len(raw_tok) < 3
                
                # Thresholds:
                # Trash ("At"): 0.85 (Muss perfekt passen, sonst weg)
                # Content ("üll"): 0.55 (Darf semantisch variieren)
                dynamic_thresh = 0.85 if is_trash_candidate else 0.55
                
                if cosine_sim > dynamic_thresh:
                    # Match akzeptiert!
                    compressed_denom = q_log * d_log
                    excess = (cosine_sim - dynamic_thresh) ** 2
                    quality_boost = 1.0 + (cosine_sim * 2) 
                    
                    weighted_score = compressed_denom * excess * quality_boost * 100
                    total_score += weighted_score
                    
                    # Debug Output verschönern
                    thresh_label = "Trash" if is_trash_candidate else "Sem"
                    debug_matches.append(f"{raw_tok}({thresh_label})->Cos:{cosine_sim:.2f}")

            if total_score > 0.1:
                results.append((total_score, doc["text"], debug_matches))
            
        results.sort(key=lambda x: x[0], reverse=True)
        
        for score, text, matches in results[:top_k]:
            preview = (text[:60] + '..') if len(text) > 60 else text
            print(f"Score: {score:.1f} | {preview}")
            print(f"      -> Matches: {matches}")


# ==============================================================================
# MAIN EXECUTION & TESTS
# ==============================================================================
if __name__ == "__main__":
    
    # 1. Konfiguration
    # Wähle Device automatisch (Mac M1/M2/M3 Support via mps)
    if torch.backends.mps.is_available(): DEVICE = "mps" 
    elif torch.cuda.is_available(): DEVICE = "cuda"
    else: DEVICE = "cpu"
    
    # Modelle (Pfade anpassen!)
    # T5_PATH = "/path/to/local/t5" 
    T5_PATH = "google/t5gemma-2-4b-4b" 
    SPARSE_PATH = "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1"
    
    # 2. Engine starten
    engine = FrankensteinEncoder(T5_PATH, SPARSE_PATH, device=DEVICE)
    
    # Kompressor-Tuning (Hier stellen wir Baseline & Ratio ein)
    engine.compressor.baseline = 15.0
    engine.compressor.compression_ratio = 0.65
    engine.compressor.max_gop = 5
    
    searcher = FrankensteinSearcher(engine)
    
    # 3. Test-Daten
    corpus = [
        "Kleine Hasenkinder tollen über die Wiese.",              
        "Der Mietvertrag wurde fristgerecht gekündigt.",          
        "Radioaktiver Abfall ist ein Problem der Atomkraft.", # Der schwierige Satz 1
        "Hier schläft ein Kind.",                             # Der schwierige Satz 2 (leise)
        "Haselnüsse sind lecker."
    ]
    
    print("\n🏗️ Indexiere Dokumente (mit Debugging der kritischen Sätze)...")
    
    # Wir indexieren alle, aber zeigen Debug-Infos für die Problemfälle
    for txt in corpus:
        is_critical = "Abfall" in txt or "schläft" in txt
        
        if is_critical:
            print(f"\n--- Debugging: '{txt}' ---")
            vecs = engine.encode_document(txt, debug_compression=True)
            # Trotzdem zum Index hinzufügen
            searcher.index.append({"text": txt, "vecs": torch.tensor(vecs)})
        else:
            searcher.add_document(txt)
            
    # 4. Such-Tests
    # Test A: Semantik + Keyword (Abfall wurde nicht explizit genannt in Query)
    searcher.search("Gefahr durch Atommüll") 
    
    # Test B: Das "leise" Kind finden (Sollte dank GOP funktionieren)
    searcher.search("Baby Bettruhe")
