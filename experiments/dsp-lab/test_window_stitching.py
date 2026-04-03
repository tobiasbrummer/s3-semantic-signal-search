import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np

class StitchingTester:
    def __init__(self, model_name, device="cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    def get_pooled_vectors(self, text):
        # Wir fügen KEIN extra Space hinzu, um die Tokenisierung nicht zu verfälschen
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            emb = self.model(input_ids=inputs.input_ids).last_hidden_state[0]
        tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        
        words = []
        curr = None
        # Wir suchen nach JEDEM Token-Start (Ġ, ▁,  , ...)
        SPACE_PREFIXES = ["Ġ", "▁", " "]

        for i, tok in enumerate(tokens):
            if tok in ["<s>", "</s>", "<pad>", "[CLS]", "[SEP]"]: continue
            
            is_start = any(tok.startswith(p) for p in SPACE_PREFIXES) or curr is None
            
            if is_start:
                if curr: words.append(curr)
                # Text bereinigen
                clean_tok = tok
                for p in SPACE_PREFIXES: clean_tok = clean_tok.replace(p, "")
                curr = {"text": clean_tok, "vecs": [emb[i]]}
            else:
                curr["text"] += tok.replace("##", "")
                curr["vecs"].append(emb[i])
        if curr: words.append(curr)
        
        for w in words:
            w["norm_vec"] = F.normalize(torch.stack(w["vecs"]).mean(dim=0).unsqueeze(0), p=2, dim=1)[0]
        return words

if __name__ == "__main__":
    if torch.backends.mps.is_available(): DEVICE = "mps"
    elif torch.cuda.is_available(): DEVICE = "cuda"
    else: DEVICE = "cpu"

    tester = StitchingTester("ibm-granite/granite-embedding-278m-multilingual", DEVICE)
    
    target = "Mietvertrag"
    s1 = "Dies ist ein langer Text am Anfang des Fensters. " * 20 + target
    s2 = target + " Dies ist ein langer Text am Ende des Fensters. " * 20

    res1 = tester.get_pooled_vectors(s1)
    res2 = tester.get_pooled_vectors(s2)

    # Suche das Wort 'Mietvertrag'
    def find_v(res_list, t):
        for w in res_list:
            if w["text"].lower() == t.lower():
                return w["norm_vec"]
        return None

    v1 = find_v(res1, target)
    v2 = find_v(res2, target)

    if v1 is not None and v2 is not None:
        sim = torch.dot(v1, v2).item()
        print(f"\n📊 STITCHING-STABILITÄT (Wort: '{target}')")
        print(f"  V1 (Ende) vs V2 (Anfang): Cosine = {sim:.4f}")
        if sim > 0.7: print("  ✅ Granite ist stabil genug für Stitching.")
        else: print("  ❌ Granite driftet zu stark.")
    else:
        print(f"❌ '{target}' nicht gefunden.")
        print(f"S1: {[w['text'] for w in res1[-3:]]}")
        print(f"S2: {[w['text'] for w in res2[:3]]}")
