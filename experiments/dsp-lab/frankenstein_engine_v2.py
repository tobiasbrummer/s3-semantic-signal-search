import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
import numpy as np
import re

def slerp(v0, v1, t):
    dot = torch.dot(v0, v1)
    if dot > 0.9995: return F.normalize((1.0 - t) * v0 + t * v1, p=2, dim=-1)
    theta_0 = torch.acos(torch.clamp(dot, -1.0, 1.0))
    sin_theta_0 = torch.sin(theta_0)
    theta_t = theta_0 * t
    sin_theta_t = torch.sin(theta_t)
    return (torch.sin(theta_0 - theta_t) / sin_theta_0) * v0 + (sin_theta_t / sin_theta_0) * v1

class FrankensteinEngineV2:
    def __init__(self, dense_name="ibm-granite/granite-embedding-278m-multilingual", 
                 sparse_name="opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1",
                 device="cpu"):
        self.device = device
        print(f"🏗️ Initialisiere Engine V2 auf {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(dense_name)
        self.dense_model = AutoModel.from_pretrained(dense_name).to(device).eval()
        self.sparse_tokenizer = AutoTokenizer.from_pretrained(sparse_name)
        self.sparse_model = AutoModelForMaskedLM.from_pretrained(sparse_name).to(self.device).eval()

    def _normalize_text(self, text):
        text = re.sub(r'[^\w\s]', ' ', text)
        return " ".join(text.split())

    def _get_sparse_weights_full(self, text):
        inputs = self.sparse_tokenizer(text, return_tensors="pt", truncation=True, max_length=512, 
                                       padding=True, return_offsets_mapping=True, stride=128, 
                                       return_overflowing_tokens=True).to(self.device)
        all_weights = []
        with torch.no_grad():
            outputs = self.sparse_model(inputs.input_ids)
            values = torch.max(torch.log(1 + torch.relu(outputs.logits)), dim=-1)[0]
        for i, offsets in enumerate(inputs.offset_mapping):
            w_chunk, o_chunk = values[i].cpu().numpy(), offsets.cpu().numpy()
            for idx, (s, e) in enumerate(o_chunk):
                if s == e: continue
                all_weights.append((s, e, w_chunk[idx]))
        return all_weights

    def encode(self, text, is_query=False):
        clean_text = self._normalize_text(text)
        processed_text = " " + clean_text
        inputs = self.tokenizer(processed_text, return_tensors="pt", return_offsets_mapping=True)
        all_ids, all_offsets = inputs.input_ids[0], inputs.offset_mapping[0].cpu().numpy()
        total_tokens = len(all_ids)

        sparse_data = self._get_sparse_weights_full(processed_text)
        dense_emb = torch.zeros((total_tokens, self.dense_model.config.hidden_size), device=self.device)
        
        if total_tokens <= 512:
            with torch.no_grad():
                dense_emb = F.normalize(self.dense_model(input_ids=all_ids.unsqueeze(0).to(self.device)).last_hidden_state[0], p=2, dim=-1)
        else:
            window_size, step, blend = 512, 256, 64
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
        word_data, curr_word = [], None
        SPACE_PREFIXES = ["Ġ", "▁", " ", " "]

        for i, tok in enumerate(tokens):
            if tok in [self.tokenizer.bos_token, self.tokenizer.eos_token, self.tokenizer.pad_token, "[CLS]", "[SEP]"]: continue
            is_start = any(tok.startswith(p) for p in SPACE_PREFIXES) or curr_word is None
            if is_start:
                if curr_word: word_data.append(self._finalize_word(curr_word, sparse_data))
                clean_tok = tok
                for p in SPACE_PREFIXES: clean_tok = clean_tok.replace(p, "")
                curr_word = {"text": clean_tok, "vecs": [dense_emb[i]], "offs": [all_offsets[i]]}
            else:
                curr_word["text"] += tok.replace("##", "")
                curr_word["vecs"].append(dense_emb[i])
                curr_word["offs"].append(all_offsets[i])
        if curr_word: word_data.append(self._finalize_word(curr_word, sparse_data))
        
        # --- Z-SCORE CALCULATION FOR WORDS ---
        weights = np.array([w["weight"] for w in word_data])
        mu, std = np.mean(weights), np.std(weights) + 1e-6
        for i, w in enumerate(word_data):
            w["z"] = (w["weight"] - mu) / std
            
        return word_data

    def _finalize_word(self, w, sparse_data):
        mean_vec = torch.stack(w["vecs"]).mean(dim=0).float()
        norm_vec = F.normalize(mean_vec.unsqueeze(0), p=2, dim=1)[0]
        w_s, w_e = w["offs"][0][0], w["offs"][-1][1]
        overlapping = [sw for ss, se, sw in sparse_data if max(w_s, ss) < min(w_e, se)]
        return {"text": w["text"], "vec": norm_vec, "weight": max(overlapping) if overlapping else 1.0}

class FrankensteinSearcherV2:
    def __init__(self, engine, threshold=0.80):
        self.engine = engine
        self.threshold = threshold
        self.index = []

    def add_document(self, text, doc_id=None):
        print(f"📥 Indexiere: {text[:50]}...")
        words = self.engine.encode(text)
        self.index.append({"id": doc_id or len(self.index), "text": text, "words": words})

    def search(self, query, top_k=3):
        print(f"\n🔎 SUCHE: '{query}'")
        q_words = self.engine.encode(query, is_query=True)
        results = []

        for doc in self.index:
            score, matches = 0, []
            
            # Matrix der Ähnlichkeiten vorbereiten [Q_len, D_len]
            q_vecs = torch.stack([qw["vec"] for qw in q_words])
            d_vecs = torch.stack([dw["vec"] for dw in doc["words"]])
            sim_matrix = torch.matmul(q_vecs, d_vecs.T) # [Q, D]

            for i, qw in enumerate(q_words):
                # 1. Vorwärts: Finde alle semantisch plausiblen Kandidaten (> 0.70)
                row = sim_matrix[i]
                candidate_mask = row > 0.70
                
                if not candidate_mask.any():
                    continue # Kein semantischer Treffer im Fenster
                
                # 2. SPARSE-FIRST: Aus den Kandidaten wählen wir das Wort mit dem höchsten Sparse-Gewicht
                candidate_indices = torch.where(candidate_mask)[0].cpu().numpy()
                best_d_idx = -1
                max_dw_weight = -1
                
                for d_idx in candidate_indices:
                    dw_weight = doc["words"][d_idx]["weight"]
                    if dw_weight > max_dw_weight:
                        max_dw_weight = dw_weight
                        best_d_idx = d_idx
                
                dw = doc["words"][best_d_idx]
                best_sim = row[best_d_idx].item()

                # 3. Rückwärts-Check für Stabilität (Mutual Nearest Neighbor)
                # Wir prüfen, ob das gewählte Dokument-Wort rückwärts die Query am besten matcht
                back_sim, q_idx_back = torch.max(sim_matrix[:, best_d_idx], dim=0)
                is_mutual = (q_idx_back.item() == i)

                # 4. Adaptives Noise Gate & Scoring
                adaptive_t = self.threshold - (qw["z"] * 0.05)
                adaptive_t = np.clip(adaptive_t, 0.70, 0.95)
                
                effective_threshold = 0.70 if is_mutual else adaptive_t

                if best_sim > effective_threshold:
                    importance = qw["weight"] * dw["weight"]
                    # Quality (Sim) ist wichtig, aber Sparse (Importance) dominiert nun die Wahl
                    quality = (best_sim ** 4) 
                    
                    z_delta = abs(qw["z"] - dw["z"])
                    z_penalty = np.exp(-z_delta)
                    
                    # Mutual Bonus
                    mutual_multiplier = 1.5 if is_mutual else 0.5
                    
                    score += quality * importance * z_penalty * mutual_multiplier
                    status = "💎" if is_mutual else "⚠️"
                    matches.append(f"{status}{qw['text']}->{dw['text']}({best_sim:.2f}/w:{dw['weight']:.1f})")
            
            if score > 0.1: results.append((score, doc["text"], matches))
        
        results.sort(key=lambda x: x[0], reverse=True)
        for s, t, m in results[:top_k]:
            print(f"[{s:.2f}] {t[:80]}...")
            print(f"      -> Matches: {m}")

if __name__ == "__main__":
    if torch.backends.mps.is_available(): DEVICE = "mps"
    elif torch.cuda.is_available(): DEVICE = "cuda"
    else: DEVICE = "cpu"
    engine = FrankensteinEngineV2(device=DEVICE)
    searcher = FrankensteinSearcherV2(engine)
    import fitz
    doc = fitz.open("contract.pdf")
    full_text = " ".join([page.get_text() for page in doc])
    searcher.add_document(full_text, doc_id="AGB")
    searcher.add_document("Ein Baby schläft friedlich in seinem Bettchen.")
    searcher.search("Bankgeheimnis und Zinsen")
    searcher.search("Baby Bettruhe")