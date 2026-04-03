import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForMaskedLM
import numpy as np
import re

class FrankensteinV2:
    def __init__(self, t5_name, sparse_name, device="cpu"):
        self.device = device
        print(f"🏗️ Lade Engine V2 ({device})...")
        self.t5_tokenizer = AutoTokenizer.from_pretrained(t5_name, legacy=False)
        self.t5_model = AutoModelForSeq2SeqLM.from_pretrained(t5_name, dtype=torch.float32).to(self.device).eval()
        self.sparse_tokenizer = AutoTokenizer.from_pretrained(sparse_name)
        self.sparse_model = AutoModelForMaskedLM.from_pretrained(sparse_name).to(self.device).eval()

    def _get_word_data(self, text):
        t5_in = self.t5_tokenizer(text, return_tensors="pt", return_offsets_mapping=True).to(self.device)
        with torch.no_grad():
            t5_emb = self.t5_model.get_encoder()(input_ids=t5_in.input_ids).last_hidden_state[0]
        
        s_in = self.sparse_tokenizer(text, return_tensors="pt", return_offsets_mapping=True).to(self.device)
        with torch.no_grad():
            s_logits = self.sparse_model(s_in.input_ids).logits
        s_weights = torch.max(torch.log(1 + torch.relu(s_logits)), dim=-1)[0][0].cpu().numpy()
        s_offsets = s_in.offset_mapping[0].cpu().numpy()

        words = []
        curr = None
        t5_tokens = self.t5_tokenizer.convert_ids_to_tokens(t5_in.input_ids[0])
        t5_offsets = t5_in.offset_mapping[0].cpu().numpy()

        for i, (tok, off) in enumerate(zip(t5_tokens, t5_offsets)):
            if off[0] == off[1]: continue
            if tok.startswith("▁") or curr is None:
                if curr: words.append(curr)
                curr = {"text": tok.replace("▁", ""), "vecs": [t5_emb[i]], "off": list(off)}
            else:
                curr["text"] += tok.replace("▁", "")
                curr["vecs"].append(t5_emb[i])
                curr["off"][1] = off[1]
        if curr: words.append(curr)

        word_list = []
        for w in words:
            # A. Der reine semantische Mittelwert
            mean_vec = torch.stack(w["vecs"]).mean(dim=0)
            # B. Normalisierte Version (für Cosine)
            norm_vec = F.normalize(mean_vec.unsqueeze(0), p=2, dim=1)[0]
            
            # Sparse Gewicht
            w_start, w_end = w["off"]
            overlapping = [sw for so, sw in zip(s_offsets, s_weights) if max(w_start, so[0]) < min(w_end, so[1])]
            weight = np.max(overlapping) if overlapping else 1.0
            
            # C. "Fused" Vektor (wie in deiner ersten Version: Vektor * Gewicht)
            fused_vec = mean_vec * weight

            word_list.append({
                "clean": re.sub(r'[^\w]', '', w["text"]).lower(),
                "orig": w["text"],
                "norm_vec": norm_vec,
                "fused_vec": fused_vec,
                "weight": weight
            })
        return [w for w in word_list if w["clean"]]

class FrankensteinSearcherV2:
    def __init__(self, engine):
        self.engine = engine
        self.index = []

    def add_document(self, text):
        words = self.engine._get_word_data(text)
        self.index.append({"text": text, "words": words})

    def search(self, query):
        q_words = self.engine._get_word_data(query)
        print(f"\n🔎 SUCHE: '{query}'")
        print("=" * 80)
        print(f"{ 'DOKUMENT':<50} | {'NORM-SCORE':<12} | {'RAW-SCORE':<10}")
        print("-" * 80)

        results = []
        for doc in self.index:
            norm_total = 0
            raw_total = 0
            matches = []
            
            for qw in q_words:
                best_sim = -1
                best_raw_dot = -1000
                best_dw = None
                
                for dw in doc["words"]:
                    # Cosine Sim (via Normalisierten Vektoren)
                    sim = torch.dot(qw["norm_vec"], dw["norm_vec"]).item()
                    if sim > best_sim:
                        best_sim = sim
                        best_dw = dw
                    
                    # Raw Dot Product (deine ursprüngliche Methode)
                    raw_dot = torch.dot(qw["fused_vec"], dw["fused_vec"]).item()
                    if raw_dot > best_raw_dot:
                        best_raw_dot = raw_dot

                # Scoring
                if best_sim > 0.55: # Noise Gate für Semantik
                    # 1. Norm-Score (Log-gedämpft)
                    norm_total += best_sim * np.log1p(qw["weight"] * best_dw["weight"])
                    # 2. Raw-Score (Direktes Dot Product, ungedämpft)
                    raw_total += best_raw_dot
                    
                    matches.append(f"{qw['orig']}->{best_dw['orig']}({best_sim:.2f})")
            
            if norm_total > 0:
                results.append((norm_total, raw_total, doc["text"], matches))
        
        # Sortiert nach Norm-Score
        results.sort(key=lambda x: x[0], reverse=True)
        for n_score, r_score, text, m in results[:3]:
            print(f"{text[:50]:<50} | {n_score:<12.2f} | {r_score:<10.0f}")
            print(f"      -> Matches: {m}")

if __name__ == "__main__":
    if torch.backends.mps.is_available(): DEVICE = "mps"
    elif torch.cuda.is_available(): DEVICE = "cuda"
    else: DEVICE = "cpu"

    engine = FrankensteinV2("google/t5gemma-2-1b-1b", "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1", device=DEVICE)
    searcher = FrankensteinSearcherV2(engine)

    corpus = [
        "Radioaktiver Abfall ist ein Problem der Atomkraft.",
        "Der Mietvertrag wurde fristgerecht gekündigt.",
        "Kleine Hasenkinder tollen über die Wiese.",
        "Hier schläft ein Baby im Bett.",
        "Ein Wasserkraftwerk produziert nachhaltigen Strom."
    ]
    for t in corpus: searcher.add_document(t)

    searcher.search("Gefahr durch Atommüll")
    searcher.search("Baby Bettruhe")
    searcher.search("Mietvertrag")
