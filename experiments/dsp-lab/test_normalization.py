import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
import re

class NormalizationTester:
    def __init__(self, t5_name, device="cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(t5_name, legacy=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(t5_name, dtype=torch.float32).to(self.device).eval()

    def get_pooled_words(self, text):
        # NORMALISIERUNG
        text = re.sub(r'[^\w\s]', ' ', text)
        text = " " + text
        text = " ".join(text.split())
        if not text.startswith(" "): text = " " + text

        t5_in = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            t5_emb = self.model.get_encoder()(input_ids=t5_in.input_ids).last_hidden_state[0]
        
        t5_tokens = self.tokenizer.convert_ids_to_tokens(t5_in.input_ids[0])
        
        words = []
        curr = None
        for i, tok in enumerate(t5_tokens):
            if tok in ["<bos>", "<eos>", "<pad>"]: continue
            
            # SentencePiece Unterstrich ist das Space-Symbol
            if tok.startswith("▁") or curr is None:
                if curr: words.append(curr)
                curr = {"text": tok.replace("▁", ""), "vecs": [t5_emb[i]], "tokens": [tok]}
            else:
                curr["text"] += tok.replace("▁", "")
                curr["vecs"].append(t5_emb[i])
                curr["tokens"].append(tok)
        if curr: words.append(curr)

        final = {}
        for w in words:
            mean_vec = torch.stack(w["vecs"]).mean(dim=0)
            norm_vec = F.normalize(mean_vec.unsqueeze(0), p=2, dim=1)[0]
            final[w["text"].lower()] = {"vec": norm_vec, "tokens": w["tokens"], "orig": w["text"]}
        return final

if __name__ == "__main__":
    if torch.backends.mps.is_available(): DEVICE = "mps"
    elif torch.cuda.is_available(): DEVICE = "cuda"
    else: DEVICE = "cpu"

    debug = NormalizationTester("google/t5gemma-2-1b-1b", device=DEVICE)

    print("\n🔬 TEST 1: BETTRUHE VS ATOMKRAFT (Bereinigt)")
    print("=" * 60)
    res1 = debug.get_pooled_words("Baby Bettruhe")
    res2 = debug.get_pooled_words("Radioaktiver Abfall ist ein Problem der Atomkraft.")
    
    w1 = res1.get("bettruhe")
    w2 = res2.get("atomkraft")
    
    if w1 and w2:
        sim = torch.dot(w1["vec"], w2["vec"]).item()
        print(f"'{w1['orig']}' vs '{w2['orig']}' -> Cosine: {sim:.4f}")
    else:
        print(f"Nicht gefunden. S1: {list(res1.keys())} | S2: {list(res2.keys())}")

    print("\n🔬 TEST 2: MIETVERTRAG (Mit Space-Trick)")
    print("=" * 60)
    res_q = debug.get_pooled_words("Mietvertrag")
    res_d = debug.get_pooled_words("Mietvertrag wurde gekündigt.")

    w_q = res_q.get("mietvertrag")
    w_d = res_d.get("mietvertrag")

    if w_q and w_d:
        sim = torch.dot(w_q["vec"], w_d["vec"])
        print(f"Query Tokens: {w_q['tokens']}")
        print(f"Doc Tokens:   {w_d['tokens']}")
        print(f"--> Cosine Similarity: {sim:.4f}")
    else:
        print(f"Nicht gefunden. Query: {list(res_q.keys())} | Doc: {list(res_d.keys())}")