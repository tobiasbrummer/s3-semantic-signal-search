import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
import fitz  # PyMuPDF
import re

def slerp(v0, v1, t):
    """Spherical Linear Interpolation zwischen zwei normalisierten Vektoren."""
    dot = torch.dot(v0, v1)
    if dot > 0.9995:
        return F.normalize((1.0 - t) * v0 + t * v1, p=2, dim=-1)
    theta_0 = torch.acos(torch.clamp(dot, -1.0, 1.0))
    sin_theta_0 = torch.sin(theta_0)
    theta_t = theta_0 * t
    sin_theta_t = torch.sin(theta_t)
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    return s0 * v0 + s1 * v1

class AdvancedStitcher:
    def __init__(self, model_name="ibm-granite/granite-embedding-278m-multilingual", device="cpu"):
        self.device = device
        print(f"🏗️ Lade Modell: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    def read_pdf(self, path):
        doc = fitz.open(path)
        text = ""
        for page in doc: text += page.get_text()
        return text

    def get_stitched_embeddings(self, text, window_size=512, step=256, blend_size=64):
        print(f"🧵 Stitching (Window={window_size}, Step={step}, Blend={blend_size})...")
        inputs = self.tokenizer(text, return_tensors="pt")
        all_ids = inputs.input_ids[0]
        total_tokens = len(all_ids)
        
        stitched = torch.zeros((total_tokens, self.model.config.hidden_size), device=self.device)
        mask = torch.zeros(total_tokens, dtype=torch.bool)

        prev_emb = None
        prev_start = 0

        for start in range(0, total_tokens, step):
            end = min(start + window_size, total_tokens)
            window_ids = all_ids[start:end].unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                emb = self.model(input_ids=window_ids).last_hidden_state[0]
            emb = F.normalize(emb, p=2, dim=-1)

            # Junction bei start + 128 (Mitte des Overlaps)
            if prev_emb is not None:
                junction_abs = start + 128
                b_start = junction_abs - (blend_size // 2)
                b_end = junction_abs + (blend_size // 2)

                for abs_idx in range(b_start, b_end):
                    if abs_idx >= total_tokens: break
                    rel_old = abs_idx - prev_start
                    rel_curr = abs_idx - start
                    if rel_old < prev_emb.shape[0] and rel_curr < emb.shape[0]:
                        t = (abs_idx - b_start) / blend_size
                        stitched[abs_idx] = slerp(prev_emb[rel_old], emb[rel_curr], t)
                        mask[abs_idx] = True

            # Füllen
            f_start = start if prev_emb is None else (start + 128 + blend_size // 2)
            f_end = min(start + step + 128, total_tokens)
            if start + window_size >= total_tokens: f_end = total_tokens

            for abs_idx in range(f_start, f_end):
                if not mask[abs_idx]:
                    rel_idx = abs_idx - start
                    if rel_idx < emb.shape[0]:
                        stitched[abs_idx] = emb[rel_idx]
                        mask[abs_idx] = True

            prev_emb = emb
            prev_start = start
            if end == total_tokens: break

        return stitched, self.tokenizer.convert_ids_to_tokens(all_ids)

    def pool_to_words(self, token_embs, tokens):
        print("▁ Wort-Pooling...")
        words, curr = [], None
        SPACE_PREFIXES = ["Ġ", "▁", " ", " "]
        for i, tok in enumerate(tokens):
            if tok in ["<s>", "</s>", "<pad>", "[CLS]", "[SEP]"]: continue
            is_start = any(tok.startswith(p) for p in SPACE_PREFIXES) or curr is None
            if is_start:
                if curr: words.append(curr)
                clean_tok = tok
                for p in SPACE_PREFIXES: clean_tok = clean_tok.replace(p, "")
                curr = {"text": clean_tok, "vecs": [token_embs[i]]}
            else:
                curr["text"] += tok.replace("##", "")
                curr["vecs"].append(token_embs[i])
        if curr: words.append(curr)
        final = []
        for w in words:
            final.append({"text": w["text"], "vec": F.normalize(torch.stack(w["vecs"]).mean(dim=0).unsqueeze(0), p=2, dim=1)[0]})
        return final

if __name__ == "__main__":
    if torch.backends.mps.is_available(): DEVICE = "mps"
    elif torch.cuda.is_available(): DEVICE = "cuda"
    else: DEVICE = "cpu"

    stitcher = AdvancedStitcher(device=DEVICE)
    text = stitcher.read_pdf("contract.pdf")
    token_embs, tokens = stitcher.get_stitched_embeddings(text)
    word_data = stitcher.pool_to_words(token_embs, tokens)

    print(f"\n📊 STABILITÄTS-ANALYSE:")
    # Wir suchen nach markanten Wörtern
    targets = ["Mietvertrag", "Geltungsbereich", "Bankgeheimnis"]
    for t in targets:
        found = [w for w in word_data if t.lower() in w["text"].lower()]
        if len(found) >= 2:
            sim = torch.dot(found[0]["vec"], found[1]["vec"]).item()
            print(f"  Wort '{t}': Ähnlichkeit Vorkommen 1 vs 2 = {sim:.4f}")
        elif len(found) == 1:
            print(f"  Wort '{t}': Nur einmal gefunden.")
        else:
            print(f"  ⚠️ '{t}' nicht gefunden.")