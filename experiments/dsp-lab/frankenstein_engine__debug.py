import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForMaskedLM
import numpy as np

SPIECE_UNDERLINE = "▁"

class FrankensteinEncoder:
    def __init__(self, t5_name, sparse_name, device="cpu"):
        self.device = device
        print(f"🏗️ Lade Frankenstein Models auf {self.device}...")
        
        self.t5_tokenizer = AutoTokenizer.from_pretrained(t5_name, legacy=False)
        self.t5_model = AutoModelForSeq2SeqLM.from_pretrained(
            t5_name, dtype=torch.float32
        ).to(self.device).eval()

        self.sparse_tokenizer = AutoTokenizer.from_pretrained(sparse_name)
        self.sparse_model = AutoModelForMaskedLM.from_pretrained(sparse_name).to(self.device).eval()
        
    def get_sparse_details(self, text):
        """Holt Token, Offsets und Gewichte des Sparse-Modells."""
        inputs = self.sparse_tokenizer(
            text, return_tensors="pt", return_offsets_mapping=True, truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.sparse_model(inputs.input_ids)
        
        # SPLADE Gewichtsberechnung: log(1 + relu(logits))
        logits = outputs.logits
        values, _ = torch.max(torch.log(1 + torch.relu(logits)), dim=-1)
        
        tokens = self.sparse_tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        offsets = inputs.offset_mapping[0].cpu().numpy()
        weights = values[0].cpu().numpy()
        
        return list(zip(tokens, offsets, weights))

    def get_t5_details(self, text):
        """Holt Token, Offsets und Embeddings des T5-Modells."""
        t5_in = self.t5_tokenizer(text, return_tensors="pt", return_offsets_mapping=True).to(self.device)
        with torch.no_grad():
            t5_emb = self.t5_model.get_encoder()(
                input_ids=t5_in.input_ids, attention_mask=t5_in.attention_mask
            ).last_hidden_state

        tokens = self.t5_tokenizer.convert_ids_to_tokens(t5_in.input_ids[0])
        offsets = t5_in.offset_mapping[0].cpu().numpy()
        return tokens, offsets, t5_emb[0]

if __name__ == "__main__":
    if torch.backends.mps.is_available(): DEVICE = "mps" 
    elif torch.cuda.is_available(): DEVICE = "cuda"
    else: DEVICE = "cpu"
    
    T5_PATH = "google/t5gemma-2-1b-1b"
    SPARSE_PATH = "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1"
    
    engine = FrankensteinEncoder(T5_PATH, SPARSE_PATH, device=DEVICE)
    
    corpus = [
        "Radioaktiver Abfall ist ein Problem der Atomkraft.",
        "Der Mietvertrag wurde fristgerecht gekündigt."
    ]
    
    for txt in corpus:
        t5_tokens, t5_offsets, t5_embs = engine.get_t5_details(txt)
        sparse_details = engine.get_sparse_details(txt)

        print("\n" + "=" * 80)
        print(f"TEXT: {txt}")
        print("=" * 80)
        print(f"{ 'T5 Token':<15} | {'Offsets':<10} | {'Overlapping Sparse Tokens & Weights'}")
        print("-" * 80)

        for i, (t_tok, t_off) in enumerate(zip(t5_tokens, t5_offsets)):
            t_start, t_end = t_off
            
            # Finde überlappende Sparse-Tokens
            overlaps = []
            if t_start != t_end:
                for s_tok, s_off, s_weight in sparse_details:
                    s_start, s_end = s_off
                    # Überlappung prüfen
                    if max(t_start, s_start) < min(t_end, s_end):
                        overlaps.append(f"\033[93m{s_tok}\033[0m({s_weight:.2f})")
            
            overlap_str = ", ".join(overlaps) if overlaps else "-"
            
            # Embedding Magnitude als Indikator für "Base Energy"
            t5_mag = torch.norm(t5_embs[i]).item()
            
            print(f"\033[92m{t_tok:<15}\033[0m | {str(list(t_off)):<10} | {overlap_str}")
            # print(f"    [T5 L2-Norm: {t5_mag:.2f}]") # Optional für mehr Details