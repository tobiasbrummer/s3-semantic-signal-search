#!/usr/bin/env python3
"""
Semantic Engine V2 - Hierarchische Embeddings mit GLiNER2.

Diese Engine baut auf den Kernkonzepten der V1 auf, strukturiert die Daten
aber hierarchisch (Dokument -> Sätze -> Wörter) und nutzt GLiNER2 für
hochwertige Entity-Erkennung.

Features:
- "Single Pass" Embedding: Der Text wird einmal am Stück verarbeitet.
- Strukturierung via Spacy: Satzgrenzen werden linguistisch korrekt erkannt.
- Hierarchische Vektoren:
  - Wort: Dense + Sparse (aus Kontext)
  - Satz: Aggregiert (Dense Mean, Sparse Max) aus Wörtern
- Präzise Positionierung: Jedes Wort kennt seinen Satz-Index und Wort-Index.
- Entity-Awareness: GLiNER2 taggt Personen, Orte, Fristen, Zahlen.

Abhängigkeiten:
- spacy (de_dep_news_trf)
- transformers (Jina v3, Opensearch Sparse)
- gliner2 (fastino/gliner2-large-v1)
- torch
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
import spacy
import re
import sys

try:
    from gliner2 import GLiNER2
    GLINER_AVAILABLE = True
except ImportError:
    print("[EngineV2] WARNUNG: gliner2 nicht gefunden. Entity-Erkennung deaktiviert.")
    GLINER_AVAILABLE = False

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

@dataclass
class TokenInfo:
    """Informationen zu einem einzelnen Wort."""
    text: str
    original_text: str
    dense_vec: np.ndarray
    sparse_weight: float
    global_idx: int
    sentence_idx: int
    in_sentence_idx: int
    char_offset: tuple[int, int]
    pos: str = ""              # Part-of-Speech
    ent_type: str = ""         # Entity Type
    dep: str = ""              # Dependency Tag (nsubj, obj...)
    lemma: str = ""            # Grundform

@dataclass
class SentenceInfo:
    """Informationen zu einem Satz."""
    idx: int
    text: str
    tokens: List[TokenInfo]
    dense_vec: np.ndarray
    char_offset: tuple[int, int]

@dataclass
class DocumentEmbedding:
    """Das komplette hierarchische Embedding."""
    sentences: List[SentenceInfo]
    full_text: str
    all_tokens: List[TokenInfo]

class HierarchicalEngine:
    # Modelle
    MODEL_PATH = "/models"
    DENSE_MODEL = "jinaai/jina-embeddings-v3"
    SPARSE_MODEL = "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1"
    GLINER_MODEL = "fastino/gliner2-large-v1"
    
    # Entity Labels für Vertrags-Kontext
    ENTITY_LABELS = [
        "person", "organization", "location", 
        "date", "time", "duration", # Zeitliches
        "money", "percentage", "number", # Zahlen
        "city", "country" # Spezifische Orte für PAWS-X
    ]
    
    # Sliding Window
    MAX_LEN = 8192
    STRIDE = 6000

    def __init__(self, device: str = None, spacy_model: str = "de_dep_news_trf"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[EngineV2] Init auf {self.device}...")

        # 1. Spacy laden
        try:
            self.nlp = spacy.load(spacy_model)
            print(f"[EngineV2] Spacy '{spacy_model}' geladen.")
        except OSError:
            print(f"[EngineV2] Spacy Modell '{spacy_model}' fehlt.")
            sys.exit(1)

        # 2. GLiNER laden
        self.gliner = None
        if GLINER_AVAILABLE:
            print(f"[EngineV2] Lade GLiNER2: {self.GLINER_MODEL}")
            try:
                self.gliner = GLiNER2.from_pretrained(self.GLINER_MODEL)
                self.gliner.to(self.device)
            except Exception as e:
                print(f"[EngineV2] Fehler beim Laden von GLiNER: {e}")

        # 3. Embedding Modelle laden
        print(f"[EngineV2] Lade Dense Model: {self.DENSE_MODEL}")
        self.dense_tokenizer = AutoTokenizer.from_pretrained(self.DENSE_MODEL, trust_remote_code=True)
        self.dense_tokenizer.save_pretrained(MODEL_PATH)
        self.dense_model = AutoModel.from_pretrained(
            self.DENSE_MODEL, trust_remote_code=True, attn_implementation="eager"
        ).to(self.device).eval()
        self.dense_model.save_pretrained(MODEL_PATH)

        print(f"[EngineV2] Lade Sparse Model: {self.SPARSE_MODEL}")
        self.sparse_tokenizer = AutoTokenizer.from_pretrained(self.SPARSE_MODEL)
        self.sparse_model = AutoModelForMaskedLM.from_pretrained(self.SPARSE_MODEL).to(self.device).eval()

    def embed_document(self, text: str) -> DocumentEmbedding:
        """Embeddet einen Text und strukturiert ihn hierarchisch."""
        
        # 1. Struktur (Spacy) & POS Tags
        doc = self.nlp(text)
        
        # Mapping: Char-Position -> (Satz-Index, Satz-Text)
        sentence_spans = []
        for i, sent in enumerate(doc.sents):
            sentence_spans.append((sent.start_char, sent.end_char, i, sent.text))

        # 2. Entity Detection (GLiNER)
        entity_spans = []
        if self.gliner and text.strip():
            try:
                # API Call gemäß Dokumentation
                result = self.gliner.extract_entities(
                    text, 
                    self.ENTITY_LABELS, 
                    include_spans=True
                )
                
                # Output parsen: {'entities': {'label': [{'start':..., 'end':...}]}}
                if 'entities' in result:
                    for label, ent_list in result['entities'].items():
                        for ent in ent_list:
                            # Sicherstellen dass start/end da sind
                            if 'start' in ent and 'end' in ent:
                                entity_spans.append((ent['start'], ent['end'], label))
            
            except Exception as e:
                print(f"[EngineV2] GLiNER Prediction Error: {e}")

        # 3. Low-Level Embeddings
        raw_tokens = self._embed_fulltext_flat(text)

        # 4. Hierarchie bauen
        def find_sentence(char_start, char_end):
            for s_start, s_end, idx, txt in sentence_spans:
                if max(char_start, s_start) < min(char_end, s_end):
                    return idx, txt, (s_start, s_end)
            return -1, "", (0, 0)

        def find_ent_tag(char_start, char_end):
            token_mid = (char_start + char_end) / 2
            for e_start, e_end, label in entity_spans:
                if e_start <= token_mid <= e_end:
                    return label
            return ""

        all_token_infos = []
        sent_info_map = {}

        for global_idx, (word_text, dense, sparse, offset) in enumerate(raw_tokens):
            s_idx, s_text, s_offset = find_sentence(offset[0], offset[1])
            
            if s_idx != -1:
                sent_info_map[s_idx] = (s_text, s_offset)

            clean_text = re.sub(r'[^\w\s]', '', word_text).strip()
            if not clean_text: clean_text = word_text

            pos = ""
            ent = ""
            dep = ""
            lemma = ""
            
            # Metadata via Spacy lookup
            span = doc.char_span(offset[0], offset[1], alignment_mode="contract")
            if span and len(span) > 0:
                token = span[0]
                pos = token.pos_
                dep = token.dep_
                lemma = token.lemma_
            
            # GLiNER Entity Check
            gliner_ent = find_ent_tag(offset[0], offset[1])
            if gliner_ent:
                ent = gliner_ent

            t_info = TokenInfo(
                text=clean_text,
                original_text=word_text,
                dense_vec=dense, 
                sparse_weight=sparse,
                global_idx=global_idx,
                sentence_idx=s_idx,
                in_sentence_idx=0, 
                char_offset=offset,
                pos=pos,
                ent_type=ent,
                dep=dep,
                lemma=lemma
            )
            all_token_infos.append(t_info)

        # Token in Sätze gruppieren
        sentences_dict = {}
        for t in all_token_infos:
            if t.sentence_idx == -1: continue
            if t.sentence_idx not in sentences_dict:
                sentences_dict[t.sentence_idx] = []
            t.in_sentence_idx = len(sentences_dict[t.sentence_idx])
            sentences_dict[t.sentence_idx].append(t)

        final_sentences = []
        for s_idx in sorted(sentences_dict.keys()):
            tokens = sentences_dict[s_idx]
            s_text, s_offset = sent_info_map[s_idx]
            
            valid_vecs = [t.dense_vec for t in tokens if t.text.isalnum()]
            if not valid_vecs: valid_vecs = [t.dense_vec for t in tokens]
            
            sent_vec = np.mean(np.stack(valid_vecs), axis=0)
            sent_vec = sent_vec / np.linalg.norm(sent_vec)

            final_sentences.append(SentenceInfo(
                idx=s_idx,
                text=s_text,
                tokens=tokens,
                dense_vec=sent_vec,
                char_offset=s_offset
            ))

        return DocumentEmbedding(
            sentences=final_sentences,
            full_text=text,
            all_tokens=all_token_infos
        )

    # --- Low Level Embedding Helpers (wie V1/V2alpha) ---
    def _embed_fulltext_flat(self, text: str):
        dense_tokens = self._embed_dense_sliding(text)
        sparse_tokens = self._embed_sparse_sliding(text)
        
        merged = []
        for d_word, d_vec, d_offset in dense_tokens:
            d_start, d_end = d_offset
            weights = []
            for s_word, s_weight, s_offset in sparse_tokens:
                s_start, s_end = s_offset
                if max(d_start, s_start) < min(d_end, s_end):
                    weights.append(s_weight)
            final_weight = max(weights) if weights else 0.0
            norm = np.linalg.norm(d_vec)
            if norm > 0: d_vec = d_vec / norm
            merged.append((d_word, d_vec, final_weight, d_offset))
        return merged

    def _embed_dense_sliding(self, text: str):
        full_encoding = self.dense_tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
        input_ids = full_encoding["input_ids"][0]
        offset_mapping = full_encoding["offset_mapping"][0].numpy()
        full_word_ids = full_encoding.word_ids(batch_index=0)
        total_tokens = len(input_ids)
        word_embeddings = {}
        word_infos = {}

        for start_idx in range(0, total_tokens, self.STRIDE):
            end_idx = min(start_idx + self.MAX_LEN, total_tokens)
            chunk_ids = input_ids[start_idx:end_idx].unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.dense_model(input_ids=chunk_ids)
                token_embeddings = outputs.last_hidden_state[0].cpu()
            
            chunk_word_ids = full_word_ids[start_idx:end_idx]
            
            current_wid = None
            current_vecs = []
            current_indices = []
            
            for i, wid in enumerate(chunk_word_ids):
                if wid is None: continue
                if wid != current_wid:
                    if current_vecs:
                        self._store_word(word_embeddings, word_infos, current_wid, current_vecs, current_indices, text, offset_mapping, start_idx, end_idx)
                    current_wid = wid
                    current_vecs = []
                    current_indices = []
                current_vecs.append(token_embeddings[i])
                current_indices.append(start_idx + i)
            
            if current_vecs:
                self._store_word(word_embeddings, word_infos, current_wid, current_vecs, current_indices, text, offset_mapping, start_idx, end_idx)
            if end_idx >= total_tokens: break

        final_tokens = []
        for wid in sorted(word_embeddings.keys()):
            vecs = word_embeddings[wid]
            text_str, offset = word_infos[wid]
            if not text_str.strip(): continue
            
            final_vec = vecs[0][0]
            total_w = vecs[0][1]
            for vec, w in vecs[1:]:
                final_vec = final_vec * total_w + vec * w
                total_w += w
                final_vec = final_vec / total_w
            final_tokens.append((text_str, final_vec.numpy(), offset))
        return final_tokens

    def _store_word(self, store, infos, wid, vecs, indices, text, offsets, chunk_start, chunk_end):
        start_global = indices[0]
        end_global = indices[-1]
        char_start = offsets[start_global][0]
        char_end = offsets[end_global][1]
        word_str = text[char_start:char_end]
        mean_vec = torch.stack(vecs).mean(dim=0)
        chunk_len = chunk_end - chunk_start
        rel_pos = (indices[len(indices)//2] - chunk_start) / chunk_len
        weight = 1.0 - 2.0 * abs(0.5 - rel_pos)
        weight = max(0.1, weight)
        if wid not in store:
            store[wid] = []
            infos[wid] = (word_str, (char_start, char_end))
        store[wid].append((mean_vec, weight))

    def _embed_sparse_sliding(self, text: str):
        encoding = self.sparse_tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512, 
            stride=128, return_overflowing_tokens=True, return_offsets_mapping=True, padding=True
        )
        input_ids = encoding["input_ids"].to(self.device)
        offset_mappings = encoding["offset_mapping"]
        with torch.no_grad():
            outputs = self.sparse_model(input_ids)
            logits = outputs.logits
            token_weights = torch.log(1 + torch.relu(logits)).max(dim=-1).values.cpu().numpy()
        results = []
        for i in range(len(input_ids)):
            offsets = offset_mappings[i].numpy()
            weights = token_weights[i]
            word_ids = encoding.word_ids(i)
            current_wid = None
            current_w = []
            current_off = []
            for j, wid in enumerate(word_ids):
                if wid is None: 
                    if current_w:
                        self._add_sparse(results, text, current_w, current_off)
                        current_w = []
                        current_off = []
                    continue
                if wid != current_wid:
                    if current_w:
                        self._add_sparse(results, text, current_w, current_off)
                    current_wid = wid
                    current_w = [weights[j]]
                    current_off = [offsets[j]]
                else:
                    current_w.append(weights[j])
                    current_off.append(offsets[j])
            if current_w:
                self._add_sparse(results, text, current_w, current_off)
        return results

    def _add_sparse(self, results, text, weights, offsets):
        start = offsets[0][0]
        end = offsets[-1][1]
        word = text[start:end]
        if not word.strip(): return
        w = max(weights)
        results.append((word, float(w), (start, end)))

if __name__ == "__main__":
    engine = HierarchicalEngine()
    text = "Das Mädchen bekommt ein Meerschweinchen. Die Frist beträgt 2 Wochen."
    doc = engine.embed_document(text)
    
    print(f"Sätze: {len(doc.sentences)}")
    for s in doc.sentences:
        print(f"[{s.idx}] {s.text}")
        for t in s.tokens:
            tag_info = f"[{t.ent_type}]" if t.ent_type else ""
            print(f"  - '{t.text}' (orig: '{t.original_text}') sparse={t.sparse_weight:.2f} pos={t.pos} dep={t.dep} {tag_info}")
