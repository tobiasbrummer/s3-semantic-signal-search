import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np

class ContrastTester:
    def __init__(self, model_name="ibm-granite/granite-embedding-278m-multilingual", device="cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.device = device

    def get_word_vec(self, word, context=""):
        text = " " + (context if context else word)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            emb = self.model(input_ids=inputs.input_ids).last_hidden_state[0]
        
        tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        vecs = []
        for i, tok in enumerate(tokens):
            clean = tok.replace("Ġ", "").replace(" ", "").replace("##", "")
            if clean.lower() in word.lower() and clean != "":
                vecs.append(emb[i])
        
        if not vecs: return None
        v = torch.stack(vecs).mean(dim=0)
        return F.normalize(v.unsqueeze(0), p=2, dim=1)[0]

if __name__ == "__main__":
    if torch.backends.mps.is_available(): DEVICE = "mps"
    elif torch.cuda.is_available(): DEVICE = "cuda"
    else: DEVICE = "cpu"

    tester = ContrastTester(device=DEVICE)

    # Test-Paare
    pairs = [
        ("Zinsen", "Zinsen", "Identisch"),
        ("und", "auch", "Konjunktionen"),
        ("Bankgeheimnis", "hinweisen", "Halluzination"),
        ("Bankgeheimnis", "Hausmeister", "Völlig fremd"),
        ("Apfel", "Birne", "Obst"),
    ]

    print(f"\n📊 GRANITE KONTRAST-CHECK:")
    print(f"{ 'Wort 1':<15} | { 'Wort 2':<15} | { 'Typ':<15} | {'Cosine'}")
    print("-" * 65)

    for w1, w2, t in pairs:
        v1 = tester.get_word_vec(w1)
        v2 = tester.get_word_vec(w2)
        if v1 is not None and v2 is not None:
            sim = torch.dot(v1, v2).item()
            print(f"{w1:<15} | {w2:<15} | {t:<15} | {sim:.4f}")
