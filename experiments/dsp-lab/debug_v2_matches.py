import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
import re

class FrankensteinV2Debug:
    def __init__(self, t5_name, device="cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(t5_name, legacy=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(t5_name, dtype=torch.float32).to(self.device).eval()

    def get_pooled_word(self, text, target_word_clean):
        t5_in = self.tokenizer(text, return_tensors="pt", return_offsets_mapping=True).to(self.device)
        with torch.no_grad():
            t5_emb = self.model.get_encoder()(input_ids=t5_in.input_ids).last_hidden_state[0]
        
        t5_tokens = self.tokenizer.convert_ids_to_tokens(t5_in.input_ids[0])
        t5_offsets = t5_in.offset_mapping[0].cpu().numpy()

        words = []
        curr = None
        for i, (tok, off) in enumerate(zip(t5_tokens, t5_offsets)):
            if off[0] == off[1]: continue
            if tok.startswith("▁") or curr is None:
                if curr: words.append(curr)
                curr = {"text": tok.replace("▁", ""), "vecs": [t5_emb[i]], "tokens": [tok]}
            else:
                curr["text"] += tok.replace("▁", "")
                curr["vecs"].append(t5_emb[i])
                curr["tokens"].append(tok)
        if curr: words.append(curr)

        for w in words:
            clean = re.sub(r'[^\w]', '', w["text"]).lower()
            if clean == target_word_clean.lower():
                mean_vec = torch.stack(w["vecs"]).mean(dim=0)
                norm_vec = F.normalize(mean_vec.unsqueeze(0), p=2, dim=1)[0]
                return {"vec": norm_vec, "text": w["text"], "tokens": w["tokens"]}
        return None

if __name__ == "__main__":
    if torch.backends.mps.is_available(): DEVICE = "mps"
    elif torch.cuda.is_available(): DEVICE = "cuda"
    else: DEVICE = "cpu"

    debug = FrankensteinV2Debug("google/t5gemma-2-1b-1b", device=DEVICE)

    print("\n🕵️ DIAGNOSE: BETTRUHE VS ATOMKRAFT")
    print("=" * 60)
    w1 = debug.get_pooled_word("Baby Bettruhe", "Bettruhe")
    w2 = debug.get_pooled_word("Radioaktiver Abfall ist ein Problem der Atomkraft.", "Atomkraft")
    
    if w1 and w2:
        sim = torch.dot(w1["vec"], w2["vec"]).item()
        print(f"Word 1: '{w1['text']}' | Tokens: {w1['tokens']}")
        print(f"Word 2: '{w2['text']}' | Tokens: {w2['tokens']}")
        print(f"--> Cosine Similarity: {sim:.4f}")
    else:
        print(f"Konnte Wörter nicht finden. W1: {w1 is not None}, W2: {w2 is not None}")

    print("\n🕵️ DIAGNOSE: MIETVERTRAG")
    print("=" * 60)
    # Wir prüfen, ob 'Mietvertrag' als Query das gleiche Vektor-Profil hat wie im Satz
    q_mv = debug.get_pooled_word("Mietvertrag", "Mietvertrag")
    d_mv = debug.get_pooled_word("Der Mietvertrag wurde fristgerecht gekündigt.", "Mietvertrag")

    if q_mv and d_mv:
        sim = torch.dot(q_mv["vec"], d_mv["vec"]).item()
        print(f"Query 'Mietvertrag' Tokens: {q_mv['tokens']}")
        print(f"Doc 'Mietvertrag' Tokens:   {d_mv['tokens']}")
        print(f"--> Cosine Similarity: {sim:.4f}")
        if sim < 0.55:
            print("❌ GEFUNDEN! Die Ähnlichkeit ist zu niedrig für das Noise Gate (0.55).")
    else:
        print(f"Konnte 'Mietvertrag' nicht finden. Query: {q_mv is not None}, Doc: {d_mv is not None}")
        # Wenn nicht gefunden, print alle Wörter im Doc
        t5_in = debug.tokenizer("Der Mietvertrag wurde fristgerecht gekündigt.", return_tensors="pt")
        tokens = debug.tokenizer.convert_ids_to_tokens(t5_in.input_ids[0])
        print(f"Doc Tokens: {tokens}")
