import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForMaskedLM
import numpy as np
import re

SPIECE_UNDERLINE = "▁"

class WordPoolingEncoder:
    def __init__(self, t5_name, sparse_name, device="cpu"):
        self.device = device
        print(f"🏗️ Lade Models auf {self.device}...")
        self.t5_tokenizer = AutoTokenizer.from_pretrained(t5_name, legacy=False)
        self.t5_model = AutoModelForSeq2SeqLM.from_pretrained(t5_name, dtype=torch.float32).to(self.device).eval()
        self.sparse_tokenizer = AutoTokenizer.from_pretrained(sparse_name)
        self.sparse_model = AutoModelForMaskedLM.from_pretrained(sparse_name).to(self.device).eval()

    def get_word_level_embeddings(self, text):
        t5_in = self.t5_tokenizer(text, return_tensors="pt", return_offsets_mapping=True).to(self.device)
        with torch.no_grad():
            t5_emb = self.t5_model.get_encoder()(input_ids=t5_in.input_ids).last_hidden_state[0]
        t5_tokens = self.t5_tokenizer.convert_ids_to_tokens(t5_in.input_ids[0])
        t5_offsets = t5_in.offset_mapping[0].cpu().numpy()

        s_in = self.sparse_tokenizer(text, return_tensors="pt", return_offsets_mapping=True).to(self.device)
        with torch.no_grad():
            s_logits = self.sparse_model(s_in.input_ids).logits
        s_weights = torch.max(torch.log(1 + torch.relu(s_logits)), dim=-1)[0][0].cpu().numpy()
        s_offsets = s_in.offset_mapping[0].cpu().numpy()

        words = []
        curr_word = None
        
        for i, (tok, off) in enumerate(zip(t5_tokens, t5_offsets)):
            if off[0] == off[1]: continue
            
            clean_tok = tok.replace(SPIECE_UNDERLINE, "")
            # Wort-Start Logik
            if tok.startswith(SPIECE_UNDERLINE) or curr_word is None:
                if curr_word: words.append(curr_word)
                curr_word = {"text": clean_tok, "vecs": [t5_emb[i]], "off": list(off)}
            else:
                curr_word["text"] += clean_tok
                curr_word["vecs"].append(t5_emb[i])
                curr_word["off"][1] = off[1]
        if curr_word: words.append(curr_word)

        final = {}
        for w in words:
            # Wir behalten das Original-Wort für die Anzeige, aber nutzen Clean-Key für die Suche
            clean_key = re.sub(r'[^\w]', '', w["text"]).lower()
            if not clean_key: continue
            
            mean_vec = torch.stack(w["vecs"]).mean(dim=0)
            w_start, w_end = w["off"]
            overlapping = [sw for so, sw in zip(s_offsets, s_weights) if max(w_start, so[0]) < min(w_end, so[1])]
            weight = np.max(overlapping) if overlapping else 1.0
            final[clean_key] = {"vec": mean_vec, "weight": weight, "orig": w["text"]}
            
        return final

def compare(engine, s1, w1, s2, w2):
    res1 = engine.get_word_level_embeddings(s1)
    res2 = engine.get_word_level_embeddings(s2)
    
    k1_search = w1.lower()
    k2_search = w2.lower()
    
    # Teilwort-Suche: Finde den besten Match in den Keys
    def find_best_key(search_term, results):
        if search_term in results: return search_term
        # Wenn nicht exakt, schau ob das Wort Teil eines anderen ist
        for k in results.keys():
            if search_term in k: return k
        return None

    key1 = find_best_key(k1_search, res1)
    key2 = find_best_key(k2_search, res2)
    
    if not key1 or not key2:
        print(f"❌ '{w1}' (in {list(res1.keys())}) oder '{w2}' (in {list(res2.keys())}) nicht gefunden.")
        return

    d1 = res1[key1]
    d2 = res2[key2]
    
    sim = F.cosine_similarity(d1["vec"].unsqueeze(0), d2["vec"].unsqueeze(0)).item()
    print(f"Match: '{d1['orig']}' vs '{d2['orig']}'")
    print(f"  -> Cosine: {sim:.4f} | Weight: {d1['weight']:.2f}")
    status = "💎 MATCH" if sim > 0.85 else "✅ SEMANTIC" if sim > 0.65 else "⚠️ WEAK" if sim > 0.5 else "❌ NO"
    print(f"  -> Status: {status}\n")

if __name__ == "__main__":
    if torch.backends.mps.is_available(): DEVICE = "mps" 
    elif torch.cuda.is_available(): DEVICE = "cuda"
    else: DEVICE = "cpu"

    engine = WordPoolingEncoder("google/t5gemma-2-1b-1b", "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1", device=DEVICE)

    print("\n🔬 WORT-POOLING TEST (MULTI-CONTEXT)")
    print("=" * 60)

    # 1. Kraftwerk Test (Teilwort-Logik)
    compare(engine, 
            "Ein Atomkraftwerk hat einen großen Kühlturm.", "Atomkraftwerk", 
            "Ein Wasserkraftwerk erlaubt die Produktion von nachhaltigem Strom.", "Kraftwerk")

    # 2. Synonyme (Baby / Kind)
    compare(engine, "Ein Baby schläft im Bett.", "Baby", "Dort liegt ein Kind auf der Decke.", "Kind")

    # 3. Halluzination (Mietvertrag vs Atommüll)
    compare(engine, "Gefahr durch Atommüll.", "Atommüll", "Der Mietvertrag wurde gekündigt.", "Mietvertrag")

    # 4. Lexikalische Falle (Hase vs Haselnuss)
    compare(engine, "Ein Hase läuft über das Feld.", "Hase", "Die Haselnuss ist eine Frucht.", "Haselnuss")

    # 5. Kontextuelle Verschiebung (Gleiches Wort, andere Bedeutung)
    compare(engine, "Die Bank am See.", "Bank", "Geld auf der Bank.", "Bank")
    
    # 6. Abstrakt vs Konkret
    compare(engine, "Die Freiheit des Menschen.", "Freiheit", "Er wurde aus dem Gefängnis entlassen.", "Gefängnis")