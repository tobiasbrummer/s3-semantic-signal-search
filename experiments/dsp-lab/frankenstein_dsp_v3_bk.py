import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
import numpy as np
import fitz
import re

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
        print(f"🏗️ Initialisiere Semantische DSP Engine auf {device}...")
        model_name = "ibm-granite/granite-embedding-278m-multilingual"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.dense_model = AutoModel.from_pretrained(model_name).to(device).eval()
        sparse_name = "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1"
        self.sparse_tokenizer = AutoTokenizer.from_pretrained(sparse_name)
        self.sparse_model = AutoModelForMaskedLM.from_pretrained(sparse_name).to(self.device).eval()

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
        inputs = self.tokenizer(processed_text, return_tensors="pt", return_offsets_mapping=True)
        all_ids, all_offsets = inputs.input_ids[0], inputs.offset_mapping[0].cpu().numpy()
        total_tokens = len(all_ids)
        sparse_data = self._get_sparse_data_full(processed_text)
        
        if is_query or total_tokens <= 512:
            with torch.no_grad():
                dense_emb = F.normalize(self.dense_model(input_ids=all_ids.unsqueeze(0).to(self.device)).last_hidden_state[0], p=2, dim=-1)
        else:
            window_size, step, blend = 512, 256, 64
            dense_emb = torch.zeros((total_tokens, self.dense_model.config.hidden_size), device=self.device)
            mask = torch.zeros(total_tokens, dtype=torch.bool)
            prev_d, prev_start = None, 0
            for start in range(0, total_tokens, step):
                end = min(start + window_size, total_tokens)
                win_ids = all_ids[start:end].unsqueeze(0).to(self.device)
                with torch.no_grad():
                    curr_d = F.normalize(self.dense_model(input_ids=win_ids).last_hidden_state[0], p=2, dim=-1)
                if prev_d is not None:
                    junc, b_s, b_e = start + 128, start + 128 - 32, start + 128 + 32
                    for i in range(b_s, b_e):
                        if i >= total_tokens: break
                        r_o, r_c = i - prev_start, i - start
                        if r_o < prev_d.shape[0] and r_c < curr_d.shape[0]:
                            dense_emb[i] = slerp(prev_d[r_o], curr_d[r_c], (i - b_s) / 64)
                            mask[i] = True
                f_s, f_e = (start if prev_d is None else start + 160), min(start + step + 128, total_tokens)
                if start + window_size >= total_tokens: f_e = total_tokens
                for i in range(f_s, f_e):
                    if not mask[i]:
                        r = i - start
                        if r < curr_d.shape[0]: dense_emb[i], mask[i] = curr_d[r], True
                prev_d, prev_start = curr_d, start

        tokens = self.tokenizer.convert_ids_to_tokens(all_ids)
        word_vecs, word_weights, words, word_ids = [], [], [], []
        curr_v, curr_t, curr_o = [], "", []
        SPACE_PREFIXES = ["Ġ", "▁", " ", " "]

        for i, tok in enumerate(tokens):
            if tok in [self.tokenizer.bos_token, self.tokenizer.eos_token, "[CLS]", "[SEP]"]: continue
            is_start = any(tok.startswith(p) for p in SPACE_PREFIXES) or not curr_v
            if is_start and curr_v:
                word_vecs.append(F.normalize(torch.stack(curr_v).mean(dim=0).unsqueeze(0), p=2, dim=1)[0])
                w_s, w_e = curr_o[0][0], curr_o[-1][1]
                matches = [s for s in sparse_data if max(w_s, s["start"]) < min(w_e, s["end"])]
                word_weights.append(max([m["weight"] for m in matches] + [0.1]))
                word_ids.append(set().union(*[m["ids"] for m in matches]) if matches else set())
                words.append(curr_t)
                curr_v, curr_t, curr_o = [], "", []
            curr_v.append(dense_emb[i])
            curr_t += tok.replace("Ġ", "").replace("▁", "").replace(" ", "").replace("##", "")
            curr_o.append(all_offsets[i])
        if curr_v:
            word_vecs.append(F.normalize(torch.stack(curr_v).mean(dim=0).unsqueeze(0), p=2, dim=1)[0])
            w_s, w_e = curr_o[0][0], curr_o[-1][1]
            matches = [s for s in sparse_data if max(w_s, s["start"]) < min(w_e, s["end"])]
            word_weights.append(max([m["weight"] for m in matches] + [0.1]))
            word_ids.append(set().union(*[m["ids"] for m in matches]) if matches else set())
            words.append(curr_t)
        return torch.stack(word_vecs).cpu().float(), torch.tensor(word_weights).float(), words, word_ids

class FrankensteinSearcherV3:
    def __init__(self, engine):
        self.engine = engine
        self.documents = []

    def add_document(self, text, name):
        print(f"📥 Indexiere Signal: {name}...")
        vecs, weights, words, ids = self.engine.get_signal(text)
        self.documents.append({"name": name, "vecs": vecs, "weights": weights, "words": words, "ids": ids})

    def search(self, query, top_k=3):
        print(f"\n🔎 SEMANTIC DSP SCAN: '{query}'")
        print("-" * 60)
        q_vecs, q_weights, _, q_ids = self.engine.get_signal(query, is_query=True)
        q_len = len(q_vecs)
        
        results = []
        for doc in self.documents:
            d_vecs, d_weights, d_ids = doc["vecs"], doc["weights"], doc["ids"]
            d_len = len(d_vecs)
            dim = d_vecs.shape[1] # Dynamisch (768)
            
            # --- DSP PADDING (Falls Dokument kürzer als Query) ---
            if d_len < q_len:
                pad_len = q_len - d_len + 1
                d_vecs_padded = torch.cat([d_vecs, torch.zeros((pad_len, dim))])
                d_weights_padded = torch.cat([d_weights, torch.zeros(pad_len)])
            else:
                d_vecs_padded = d_vecs
                d_weights_padded = d_weights

            # --- STAGE 1: ENERGY CORRELATION SCAN ---
            semantic_radar = F.conv1d(d_vecs_padded.T.unsqueeze(0), q_vecs.T.unsqueeze(0))[0, 0]
            amplitude_radar = F.conv1d(d_weights_padded.unsqueeze(0).unsqueeze(0), q_weights.unsqueeze(0).unsqueeze(0))[0, 0]
            
            if len(semantic_radar) == 0: continue
            
            radar = semantic_radar * torch.log1p(amplitude_radar)
            top_peaks = torch.topk(radar, min(20, len(radar))).indices.tolist()
            
            for peak_idx in top_peaks:
                actual_end = min(peak_idx + q_len, d_len)
                d_segment_vecs = d_vecs[peak_idx : actual_end]
                d_segment_ids = d_ids[peak_idx : actual_end]
                
                # Wenn das Segment zu kurz ist (am Ende des Dokuments), füllen wir für CCS auf
                if len(d_segment_vecs) < q_len:
                    short_q_vecs = q_vecs[:len(d_segment_vecs)]
                    effective_q_len = len(d_segment_vecs)
                else:
                    short_q_vecs = q_vecs
                    effective_q_len = q_len

                if effective_q_len == 0: continue

                # 1. Semantische Güte (CCS)
                ccs = torch.sum(d_segment_vecs * short_q_vecs).item() / effective_q_len
                
                # 2. Lexikalischer Bonus (Overlap der IDs)
                # KEIN Filter mehr, nur ein Bonus-Faktor
                overlap_score = 0
                for qw_ids in q_ids:
                    best_overlap = max([len(qw_ids.intersection(di)) for di in d_segment_ids] + [0])
                    overlap_score += (best_overlap / 5.0) # Normalisiert auf Top-5 IDs
                lexical_bonus = 1.0 + (overlap_score / q_len)
                
                # 3. Finaler Score
                # CCS (Qualität) hoch 3 mal Lexikalischer Bonus
                final_score = (ccs ** 3) * lexical_bonus * (radar[peak_idx].item() / q_len)
                
                if ccs > 0.60: # Minimales Noise Gate für Semantik
                    results.append({"doc": doc["name"], "score": final_score,
                                    "text": " ".join(doc["words"][peak_idx : peak_idx + q_len + 2]),
                                    "metrics": f"CCS:{ccs:.2f} | Bonus:{lexical_bonus:.2f} | Radar:{radar[peak_idx]:.2f}"})
        
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
    searcher.add_document(full_text, "AGB")
    searcher.add_document("Ein Baby schläft friedlich in seinem Bettchen.", "Baby_Story")
    
    searcher.search("Bankgeheimnis")
    searcher.search("Die Bank teilt dem Kunden Änderungen von Zinsen nicht mit. Bei einer Erhöhung kann der Kunde, sofern nichts anderes vereinbart ist, die davon betroffene Kreditvereinbarung innerhalb von sechs Minuten nach der Bekanntgabe der Änderung mit sofortiger Wirkung kündigen.")
    searcher.search("Baby Bettruhe")
    searcher.search("Gefahr für Neugeborene") # Semantischer Test
