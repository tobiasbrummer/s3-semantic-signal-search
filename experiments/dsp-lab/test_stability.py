import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
import numpy as np
import re

class StabilityTester:
    def __init__(self, model_name, device="cpu", is_t5=False):
        self.device = device
        self.is_t5 = is_t5
        print(f"🏗️ Teste Stabilität von: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if is_t5:
            # Für T5/Gemma-2 nehmen wir nur den Encoder
            full_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device).eval()
            self.model = full_model.get_encoder()
        else:
            self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    def get_word_vector(self, text, target_word):
        # Space-Trick für konsistente Tokenisierung
        clean_text = " " + text
        inputs = self.tokenizer(clean_text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=inputs.input_ids)
            # Manche Modelle geben ein Tuple zurück
            if hasattr(outputs, "last_hidden_state"):
                emb = outputs.last_hidden_state[0]
            else:
                emb = outputs[0]
        
        tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        
        # Finde Tokens, die zum Wort gehören
        target_indices = []
        for i, tok in enumerate(tokens):
            # Wir säubern das Token von Sonderzeichen (Ġ, ▁,  , ##)
            clean_tok = tok.replace("Ġ", "").replace("▁", "").replace(" ", "").replace("##", "")
            if clean_tok.lower() in target_word.lower() and clean_tok != "":
                target_indices.append(i)
        
        if target_indices:
            # Wir nehmen alle Treffer und mitteln sie (Pooling)
            vecs = [emb[i] for i in target_indices]
            mean_vec = torch.stack(vecs).mean(dim=0)
            return F.normalize(mean_vec.unsqueeze(0), p=2, dim=1)[0]
        return None

if __name__ == "__main__":
    if torch.backends.mps.is_available(): DEVICE = "mps"
    elif torch.cuda.is_available(): DEVICE = "cuda"
    else: DEVICE = "cpu"

    # Wir testen Granite vs Gemma-2 (T5)
    config = [
        ("ibm-granite/granite-embedding-278m-multilingual", False),
        ("google/t5gemma-2-1b-1b", True)
    ]

    for m_name, is_t5 in config:
        tester = StabilityTester(m_name, DEVICE, is_t5)
        
        scenarios = [
            "Mietvertrag wurde heute unterschrieben.",
            "Nach langer Suche unterschrieben wir den Mietvertrag.",
            "Das Wetter ist schön und der Mietvertrag liegt im Auto auf dem Sitz.",
            "Mietvertrag."
        ]
        
        vectors = []
        for s in scenarios:
            v = tester.get_word_vector(s, "Mietvertrag")
            if v is not None:
                vectors.append(v)
            else:
                print(f"❌ Wort 'Mietvertrag' in Szenario '{s}' nicht gefunden.")

        if len(vectors) >= 2:
            print(f"\n📊 ERGEBNISSE FÜR {m_name}:")
            # Vergleich mit dem ersten Szenario (Baseline)
            for i in range(1, len(vectors)):
                sim = torch.dot(vectors[0], vectors[i]).item()
                desc = ["Ende", "Mitte/Lang", "Isoliert"][i-1]
                print(f"  Anfang vs {desc:<12}: Cosine = {sim:.4f}")
        print("-" * 60)