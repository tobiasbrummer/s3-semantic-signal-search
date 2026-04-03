import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
import numpy as np
import re

class FrankensteinEngineV2:
    """
    Frankenstein Engine V2: Word-Pooled Semantic Search
    --------------------------------------------------
    Features: 
    - Word-level Mean Pooling (Stability)
    - Space-Trick & Punctuation Stripping (Consistency)
    - Log-Damped Weighting (Dynamic Range)
    """
    
    def __init__(self, t5_name="ibm-granite/granite-embedding-278m-multilingual", 
                 sparse_name="opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1", 
                 device="cpu"):
        self.device = device
        print(f"🏗️ Initialisiere Frankenstein V2 auf {self.device}...")
        
        self.t5_tokenizer = AutoTokenizer.from_pretrained(t5_name, legacy=False)
        self.t5_model = AutoModel.from_pretrained(t5_name, dtype=torch.float32).to(self.device).eval()
        
        self.sparse_tokenizer = AutoTokenizer.from_pretrained(sparse_name)
        self.sparse_model = AutoModelForMaskedLM.from_pretrained(sparse_name).to(self.device).eval()
        print("✅ Engine bereit.")

    def _normalize_text(self, text):
        """Bereinigt Satzzeichen und erzwingt führendes Leerzeichen für T5."""
        # 1. Satzzeichen entfernen
        text = re.sub(r'[^\w\s]', ' ', text)
        # 2. Space-Trick (Sicherstellen, dass jedes Wort ein Space-Präfix hat)
        text = " " + text
        # 3. Whitespace normalisieren
        return " ".join(text.split())

    def encode(self, text):
        """Wandelt Text in eine Liste von Wort-Objekten (Vektor + Gewicht) um."""
        clean_text = self._normalize_text(text)
        
        # A. Dense Run
        t5_in = self.t5_tokenizer(clean_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.t5_model(**t5_in)
            # Falls es ein Encoder-Decoder Modell (T5) ist, nutzen wir den encoder output, 
            # ansonsten den direkten last_hidden_state (BERT/Granite/etc)
            if hasattr(outputs, "last_hidden_state"):
                t5_emb = outputs.last_hidden_state[0]
            else:
                t5_emb = outputs[0][0]

        t5_tokens = self.t5_tokenizer.convert_ids_to_tokens(t5_in.input_ids[0])
        
        # B. Sparse Run
        s_in = self.sparse_tokenizer(clean_text, return_tensors="pt", return_offsets_mapping=True).to(self.device)
        with torch.no_grad():
            s_logits = self.sparse_model(s_in.input_ids).logits
        # SPLADE weights: log(1 + relu(logits))
        s_weights = torch.max(torch.log(1 + torch.relu(s_logits)), dim=-1)[0][0].cpu().numpy()
        s_offsets = s_in.offset_mapping[0].cpu().numpy()

        # C. Wort-Pooling
        words = []
        curr = None
        
        # Granite/BERT Tokenizer nutzen oft 'Ġ' als Space-Präfix
        # T5 nutzt '▁' oder ' '
        # Wir prüfen auf verschiedene gängige Präfixe
        SPACE_PREFIXES = [" ", "▁", "Ġ", " "]

        for i, tok in enumerate(t5_tokens):
            if tok in [self.t5_tokenizer.bos_token, self.t5_tokenizer.eos_token, self.t5_tokenizer.pad_token, "[CLS]", "[SEP]"]:
                continue
            
            is_start = any(tok.startswith(p) for p in SPACE_PREFIXES) or curr is None
            
            if is_start:
                if curr: words.append(curr)
                # Präfix entfernen für sauberen Text
                clean_tok = tok
                for p in SPACE_PREFIXES: clean_tok = clean_tok.replace(p, "")
                curr = {"text": clean_tok, "vecs": [t5_emb[i]], "start_idx": i}
            else:
                curr["text"] += tok.replace("##", "") # BERT-style subwords
                curr["vecs"].append(t5_emb[i])

        if curr: words.append(curr)

        # D. Gewichte zuordnen & Normalisieren
        final_word_data = []
        # Wir brauchen die Offsets des T5 Tokenizers für das Sparse-Alignment
        t5_off = self.t5_tokenizer(clean_text, return_offsets_mapping=True)["offset_mapping"]
        
        for w in words:
            # Semantischer Vektor (Mean + L2-Normalize)
            mean_vec = torch.stack(w["vecs"]).mean(dim=0)
            norm_vec = F.normalize(mean_vec.unsqueeze(0), p=2, dim=1)[0]
            
            # Sparse Weight Alignment via Offsets
            w_start = t5_off[w["start_idx"]][0]
            w_end = t5_off[w["start_idx"] + len(w["vecs"]) - 1][1]
            
            overlapping = [sw for so, sw in zip(s_offsets, s_weights) if max(w_start, so[0]) < min(w_end, so[1])]
            weight = np.max(overlapping) if overlapping else 1.0
            
            final_word_data.append({
                "text": w["text"],
                "vec": norm_vec,
                "weight": weight
            })
            
        return final_word_data

class FrankensteinSearcherV2:
    def __init__(self, engine, threshold=0.55):
        self.engine = engine
        self.threshold = threshold
        self.index = []

    def add_document(self, text, doc_id=None):
        word_data = self.engine.encode(text)
        self.index.append({
            "id": doc_id or len(self.index),
            "text": text,
            "words": word_data
        })

    def search(self, query, top_k=3):
        q_words = self.engine.encode(query)
        print(f"\n🔎 SUCHE: '{query}'")
        print("-" * 60)
        
        results = []
        for doc in self.index:
            score = 0
            matches = []
            
            for qw in q_words:
                best_sim = -1
                best_dw = None
                
                for dw in doc["words"]:
                    # Cosine Similarity (Dot product of normalized vecs)
                    sim = torch.dot(qw["vec"], dw["vec"]).item()
                    if sim > best_sim:
                        best_sim = sim
                        best_dw = dw
                
                if best_sim > self.threshold:
                    # Log-Damped Scoring
                    importance = np.log1p(qw["weight"] * best_dw["weight"])
                    # Quality boost (Bestrafung für niedrige Cosine via Quadrat)
                    quality = best_sim ** 2
                    
                    word_score = best_sim * importance * quality
                    score += word_score
                    matches.append(f"{qw['text']}->{best_dw['text']}({best_sim:.2f})")
            
            if score > 0:
                results.append((score, doc["text"], matches))
        
        results.sort(key=lambda x: x[0], reverse=True)
        for s, text, m in results[:top_k]:
            print(f"[{s:.2f}] {text}")
            print(f"      -> Matches: {m}")

if __name__ == "__main__":
    if torch.backends.mps.is_available(): DEVICE = "mps"
    elif torch.cuda.is_available(): DEVICE = "cuda"
    else: DEVICE = "cpu"

    engine = FrankensteinEngineV2(device=DEVICE)
    searcher = FrankensteinSearcherV2(engine)

    corpus = [
        "Kleine Hasenkinder tollen über die Wiese.",
        "Der Mietvertrag wurde fristgerecht gekündigt.",
        "Radioaktiver Abfall ist ein Problem der Atomkraft.",
        "Hier schläft ein Baby im Bett.",
        "Ein Wasserkraftwerk produziert nachhaltigen Strom."
    ]
    for t in corpus: searcher.add_document(t)

    searcher.search("Gefahr durch Atommüll")
    searcher.search("Baby Bettruhe")
    searcher.search("Mietvertrag")
