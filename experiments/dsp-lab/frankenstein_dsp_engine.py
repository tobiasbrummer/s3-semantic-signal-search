import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
import numpy as np
import fitz
import re
import matplotlib.pyplot as plt

class FrankensteinDSPEngine:
    def __init__(self, device="cpu"):
        self.device = device
        print(f"🏗️ Initialisiere DSP-Engine auf {device}...")
        
        # Wir bleiben bei Granite für die 512 "Frequenzen"
        model_name = "ibm-granite/granite-embedding-278m-multilingual"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.dense_model = AutoModel.from_pretrained(model_name).to(device).eval()
        
        # Sparse für die "Lautstärke" (Gain)
        sparse_name = "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1"
        self.sparse_tokenizer = AutoTokenizer.from_pretrained(sparse_name)
        self.sparse_model = AutoModelForMaskedLM.from_pretrained(sparse_name).to(device).eval()

    def _normalize_text(self, text):
        text = re.sub(r'[^\w\s]', ' ', text)
        return " ".join(text.split())

    def get_signal(self, text):
        """Wandelt Text in ein 2D-Signal um: [Wörter, Frequenzen]."""
        clean_text = self._normalize_text(text)
        processed_text = " " + clean_text
        
        # 1. Dense (Frequenzen)
        inputs = self.tokenizer(processed_text, return_tensors="pt")
        with torch.no_grad():
            # Wir nehmen den gesamten Text (bei langen PDFs müssten wir hier den Stitcher nutzen)
            # Für diesen Test nehmen wir den Standard-Pass (bis 512 tokens)
            emb = self.dense_model(input_ids=inputs.input_ids.to(self.device)).last_hidden_state[0]
        
        # 2. Sparse (Gain)
        s_in = self.sparse_tokenizer(processed_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            s_logits = self.sparse_model(s_in.input_ids).logits
            s_weights = torch.max(torch.log(1 + torch.relu(s_logits)), dim=-1)[0][0]

        # 3. Word Pooling (DSP-Style)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        signal = []
        curr_vecs = []
        curr_weights = []
        SPACE_PREFIXES = ["Ġ", "▁", " ", " "]

        for i, tok in enumerate(tokens):
            if tok in [self.tokenizer.bos_token, self.tokenizer.eos_token, "[CLS]", "[SEP]"]:
                continue
            
            is_start = any(tok.startswith(p) for p in SPACE_PREFIXES) or not curr_vecs
            if is_start and curr_vecs:
                # Frame finalisieren
                v = torch.stack(curr_vecs).mean(dim=0)
                w = max(curr_weights)
                # Das Signal eines Wortes: Vektor * Gain
                signal.append(F.normalize(v.unsqueeze(0), p=2, dim=1)[0] * w)
                curr_vecs, curr_weights = [], []
            
            curr_vecs.append(emb[i])
            curr_weights.append(s_weights[min(i, len(s_weights)-1)].item())
            
        if curr_vecs:
            v = torch.stack(curr_vecs).mean(dim=0)
            signal.append(F.normalize(v.unsqueeze(0), p=2, dim=1)[0] * max(curr_weights))
            
        return torch.stack(signal).float() # [Seq_Len, 512]

class DSPSearcher:
    def __init__(self, engine):
        self.engine = engine
        self.documents = []

    def add_document(self, text, name):
        print(f"📥 Erzeuge Signal für: {name}...")
        signal = self.engine.get_signal(text)
        self.documents.append({"name": name, "signal": signal, "text": text})

    def search(self, query):
        print(f"\n🔎 DSP-SCAN: '{query}'")
        q_signal = self.engine.get_signal(query) # [Q_len, 512]
        
        for doc in self.documents:
            d_signal = doc["signal"] # [D_len, 512]
            
            # --- CROSS-CORRELATION SCAN ---
            # Wir schieben die Query über das Dokument
            # Mathematisch: Gleitendes Skalarprodukt über alle Wörter
            
            scores = []
            for i in range(len(d_signal) - len(q_signal) + 1):
                window = d_signal[i : i + len(q_signal)]
                # Ähnlichkeit berechnen: Summe der Korrelationen aller Frequenzen
                # Dies entspricht der Energie des Übereinstimmungssignals
                match_energy = torch.sum(window * q_signal).item()
                scores.append(match_energy)
            
            if not scores: continue
            
            max_score = max(scores)
            best_idx = np.argmax(scores)
            
            print(f"[{max_score:.2f}] {doc['name']}")
            if max_score > 0.5:
                # Zeige den Kontext des besten Matches
                words = doc["text"].split()
                context = " ".join(words[max(0, best_idx-5) : min(len(words), best_idx+10)])
                print(f"      -> Best Match Area: ...{context}...")

            # Visualisierung des Scans
            plt.figure(figsize=(12, 3))
            plt.plot(scores)
            plt.title(f"DSP Scan Result: {query} in {doc['name']}")
            plt.xlabel("Wort-Position")
            plt.ylabel("Match Energie")
            plt.savefig(f"spectral_scan_{doc['name'].replace(' ', '_')}.png")
            plt.close()

if __name__ == "__main__":
    if torch.backends.mps.is_available(): DEVICE = "mps"
    elif torch.cuda.is_available(): DEVICE = "cuda"
    else: DEVICE = "cpu"

    engine = FrankensteinDSPEngine(device=DEVICE)
    searcher = DSPSearcher(engine)

    # Wir nehmen einen Teil der AGB (Seite 2)
    pdf_text = "Bankgeheimnis (1) Die Bank ist zur Verschwiegenheit über alle kundenbezogenen Tatsachen verpflichtet (Bankgeheimnis). Informationen über den Kunden darf die Bank nur weitergeben, wenn gesetzliche Bestimmungen dies gebieten oder der Kunde eingewilligt hat oder die Bank zur Erteilung einer Bankauskunft befugt ist."
    
    searcher.add_document(pdf_text, "C24_AGB_Snippet")
    searcher.add_document("Ein Baby schläft im Bettchen und träumt von Hasen.", "Baby_Story")

    searcher.search("Bankgeheimnis")
    searcher.search("Baby Bettruhe")
