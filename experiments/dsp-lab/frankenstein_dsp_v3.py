import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
import numpy as np
import fitz
import re
import warnings
import logging

# Logger beruhigen
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

def slerp(v0, v1, t):
    dot = torch.dot(v0, v1)
    if dot > 0.9995: return F.normalize((1.0 - t) * v0 + t * v1, p=2, dim=-1)
    theta_0 = torch.acos(torch.clamp(dot, -1.0, 1.0))
    sin_theta_0 = torch.sin(theta_0)
    theta_t = theta_0 * t
    sin_theta_t = torch.sin(theta_t)
    return (torch.sin(theta_0 - theta_t) / sin_theta_0) * v0 + (sin_theta_t / sin_theta_0) * v1

class FrankensteinDSPV3:
    def __init__(self, device="cpu"):
        self.device = device
        print(f"🏗️ Initialisiere Jina-v3 Hybrid Engine (Standard Attention) auf {device}...")
        
        model_name = "jinaai/jina-embeddings-v3"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # WICHTIG: Flash-Attention deaktivieren, damit 'task' funktioniert
        self.dense_model = AutoModel.from_pretrained(
            model_name, 
            trust_remote_code=True, 
            attn_implementation="eager",
            use_flash_attn=False,
        ).to(device).eval()
        
        sparse_name = "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1"
        self.sparse_tokenizer = AutoTokenizer.from_pretrained(sparse_name)
        self.sparse_model = AutoModelForMaskedLM.from_pretrained(sparse_name).to(self.device).eval()
        print("✅ Jina-v3 (LoRA-ready) und SPLADE bereit.")

    def _normalize_text(self, text):
        text = re.sub(r'[^\w\s]', ' ', text)
        return " ".join(text.split())

    def _get_sparse_data_full(self, text):
        inputs = self.sparse_tokenizer(text, return_tensors="pt", truncation=True, max_length=512, 
                                       padding=True, return_offsets_mapping=True, stride=128, 
                                       return_overflowing_tokens=True).to(self.device)
        results = []
        with torch.no_grad():
            outputs = self.sparse_model(inputs.input_ids)
            values, indices = torch.topk(torch.log(1 + torch.relu(outputs.logits)), k=5, dim=-1)
        for i, offsets in enumerate(inputs.offset_mapping):
            v_chunk, idx_chunk, o_chunk = values[i].cpu().numpy(), indices[i].cpu().numpy(), offsets.cpu().numpy()
            for idx, (s, e) in enumerate(o_chunk):
                if s == e: continue
                results.append({"start": s, "end": e, "weight": v_chunk[idx, 0], "ids": set(idx_chunk[idx].tolist())})
        return results

    def get_signal(self, text, is_query=False):
        clean_text = self._normalize_text(text)
        processed_text = " " + clean_text
        sparse_data = self._get_sparse_data_full(processed_text)
        
        inputs = self.tokenizer(processed_text, return_tensors="pt", return_offsets_mapping=True, 
                                 padding=True, truncation=True, max_length=8192)
        model_inputs = {k: v.to(self.device) for k, v in inputs.items() if k in ["input_ids", "attention_mask"]}
        
        with torch.no_grad():
            # Da Flash-Attn aus ist, wird der task hier nun wirklich beachtet!
            self.dense_model.config.task = "classification"
            outputs = self.dense_model(**model_inputs)
            #outputs = self.dense_model(**model_inputs, task="text-matching")
            dense_emb = outputs.last_hidden_state[0]
        
        word_ids = inputs.word_ids(batch_index=0)
        offsets = inputs.offset_mapping[0].cpu().numpy()
        word_vecs, word_weights, words, word_fingerprints = [], [], [], []
        current_word_id, curr_v, curr_o = None, [], []
        
        for i, w_id in enumerate(word_ids):
            if w_id is None: continue
            if w_id != current_word_id:
                if curr_v:
                    word_vecs.append(F.normalize(torch.stack(curr_v).mean(dim=0).unsqueeze(0), p=2, dim=1).float()[0])
                    w_s, w_e = curr_o[0][0], curr_o[-1][1]
                    matches = [s for s in sparse_data if max(w_s, s["start"]) < min(w_e, s["end"])]
                    word_weights.append(max([m["weight"] for m in matches] + [0.1]))
                    word_fingerprints.append(set().union(*[m["ids"] for m in matches]) if matches else set())
                    words.append(processed_text[w_s:w_e].strip())
                current_word_id, curr_v, curr_o = w_id, [dense_emb[i]], [offsets[i]]
            else:
                curr_v.append(dense_emb[i])
                curr_o.append(offsets[i])
        if curr_v:
            word_vecs.append(F.normalize(torch.stack(curr_v).mean(dim=0).unsqueeze(0), p=2, dim=1).float()[0])
            w_s, w_e = curr_o[0][0], curr_o[-1][1]
            matches = [s for s in sparse_data if max(w_s, s["start"]) < min(w_e, s["end"])]
            word_weights.append(max([m["weight"] for m in matches] + [0.1]))
            word_fingerprints.append(set().union(*[m["ids"] for m in matches]) if matches else set())
            words.append(processed_text[w_s:w_e].strip())
            
        return torch.stack(word_vecs).cpu(), torch.tensor(word_weights).float(), words, word_fingerprints

class FrankensteinSearcherV3:
    def __init__(self, engine, threshold=0.40):
        self.engine = engine
        self.threshold = threshold
        self.documents = []

    def add_document(self, text, name):
        print(f"📥 Indexiere: {name}...")
        vecs, weights, words, ids = self.engine.get_signal(text)
        self.documents.append({"name": name, "vecs": vecs, "weights": weights, "words": words, "ids": ids})

    def search(self, query, top_k=3):
        print(f"\n🔎 JINA-V3 STABLE-DSP SCAN: '{query}'")
        print("-" * 60)
        q_vecs, q_weights, _, q_ids = self.engine.get_signal(query, is_query=True)
        q_len = len(q_vecs)
        results = []

        for doc in self.documents:
            d_vecs, d_weights, d_ids = doc["vecs"], doc["weights"], doc["ids"]
            if len(d_vecs) < q_len: continue
            
            semantic_radar = F.conv1d(d_vecs.T.unsqueeze(0), q_vecs.T.unsqueeze(0))[0, 0]
            amplitude_radar = F.conv1d(d_weights.unsqueeze(0).unsqueeze(0), q_weights.unsqueeze(0).unsqueeze(0))[0, 0]
            radar = semantic_radar + torch.log1p(amplitude_radar)
            
            top_peaks = torch.topk(radar, min(10, len(radar))).indices.tolist()
            for peak_idx in top_peaks:
                d_segment_vecs = d_vecs[peak_idx : peak_idx + q_len]
                d_segment_ids = d_ids[peak_idx : peak_idx + q_len]
                ccs = torch.sum(d_segment_vecs * q_vecs).item() / q_len
                
                overlap_score = 0
                for qw_ids in q_ids:
                    best_overlap = max([len(qw_ids.intersection(di)) for di in d_segment_ids] + [0])
                    overlap_score += (best_overlap / 5.0)
                lexical_bonus = 1.0 + (overlap_score / q_len)
                
                final_score = (ccs ** 3) * lexical_bonus * (radar[peak_idx].item() / q_len)
                
                if ccs > self.threshold:
                    results.append({
                        "doc": doc["name"], "score": final_score,
                        "text": " ".join(doc["words"][peak_idx : peak_idx + q_len + 3]),
                        "metrics": f"CCS:{ccs:.2f} | Bonus:{lexical_bonus:.2f}"
                    })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        seen = set()
        count = 0
        for res in results:
            if res["text"] not in seen and count < top_k:
                print(f"[{res['score']:.4f}] {res['doc']}")
                print(f"       -> Area: {res['text']}")
                print(f"       -> {res['metrics']}")
                seen.add(res["text"])
                count += 1

if __name__ == "__main__":
    if torch.backends.mps.is_available(): DEVICE = "mps"
    elif torch.cuda.is_available(): DEVICE = "cuda"
    else: DEVICE = "cpu"
    engine = FrankensteinDSPV3(device=DEVICE)
    searcher = FrankensteinSearcherV3(engine)
    
    import fitz
    doc = fitz.open("contract.pdf")
    full_text = " ".join([page.get_text() for page in doc])
    searcher.add_document(full_text, "AGB_FULL")
    searcher.add_document("Ein Baby schläft friedlich in seinem Bettchen.", "Baby_Story")
    
    searcher.search("Bankgeheimnis")
    searcher.search("Baby Bettruhe")
    searcher.search("Gefahr für Neugeborene")
