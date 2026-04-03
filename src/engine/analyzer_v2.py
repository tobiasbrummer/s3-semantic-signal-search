#!/usr/bin/env python3
"""
Semantic Engine - Anomalie-Erkennung V2 (Satz-basiert, Hierarchisch).

Nutzt engine_v2 für saubere Dokument-Strukturierung.
Findet:
1. Passende Sätze (Dense Matching).
2. Wort-Verschiebungen und Änderungen innerhalb der Sätze (Token Alignment).
3. Inhaltliche Widersprüche (LLM Critic).
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import re

try:
    from .engine_v2 import HierarchicalEngine, DocumentEmbedding, SentenceInfo, TokenInfo
    from .critic_v2 import CriticModelV2, LlamaCppCriticV2
except ImportError:
    from engine_v2 import HierarchicalEngine, DocumentEmbedding, SentenceInfo, TokenInfo
    from critic_v2 import CriticModelV2, LlamaCppCriticV2

@dataclass
class TokenMatch:
    query_token: TokenInfo
    match_type: str # EXACT, MOVED, REWORDED, CRITICAL, MISSING
    doc_token: Optional[TokenInfo] = None
    score: float = 0.0

@dataclass
class SentenceMatch:
    query_sentence: SentenceInfo
    doc_sentence: SentenceInfo
    dense_score: float
    # Signals (Ensemble)
    llm_verdict: bool = False
    has_entity_mismatch: bool = False
    has_structure_mismatch: bool = False
    has_keyword_mismatch: bool = False
    # Details
    reason: str = ""
    token_alignment: List[TokenMatch] = field(default_factory=list)
    missing_keywords: List[str] = field(default_factory=list)
    missing_entities: List[str] = field(default_factory=list)
    pos_mismatches: int = 0

    @property
    def contradiction(self) -> bool:
        # Ensemble Decision Logic
        if self.llm_verdict: return True
        if self.has_entity_mismatch: return True
        if self.has_structure_mismatch: return True
        # Keyword Mismatch allein reicht oft nicht (Synonyme), außer Score ist niedrig
        if self.has_keyword_mismatch and self.dense_score < 0.90: return True
        return False

@dataclass
class AnalysisResultV2:
    matches: List[SentenceMatch]
    missing_sentences: List[SentenceInfo]
    doc_similarity: float
    # Global Stats
    total_entities_found: int = 0
    total_entities_missing: int = 0

    def summary(self) -> str:
        lines = [f"Global Similarity: {self.doc_similarity:.1%}"]
        lines.append(f"Entities: {self.total_entities_found} found, {self.total_entities_missing} missing")
        
        if self.missing_sentences:
            lines.append(f"\nMISSING SENTENCES ({len(self.missing_sentences)}):")
            for s in self.missing_sentences:
                lines.append(f"  - '{s.text}'")
                
        matches_to_show = [m for m in self.matches if m.contradiction or any(t.match_type != "EXACT" for t in m.token_alignment)]
        
        for m in matches_to_show:
            title = "CONTRADICTION" if m.contradiction else "CHANGED / REWORDED"
            signals = []
            if m.llm_verdict: signals.append("LLM")
            if m.has_entity_mismatch: signals.append("ENTITY")
            if m.has_structure_mismatch: signals.append("STRUCT")
            if m.has_keyword_mismatch: signals.append("KEYWORD")
            
            lines.append(f"\n{title} [{'|'.join(signals)}]:")
            lines.append(f"  Score: {m.dense_score:.4f}")
            
            if m.reason:
                lines.append(f"  Reason: {m.reason}")
            
            if m.missing_entities:
                lines.append(f"  Missing Ents: {m.missing_entities}")

            # Inline Diff Construction
            diff_parts = []
            for t in m.token_alignment:
                text = t.query_token.text
                if t.match_type == "EXACT":
                    diff_parts.append(text)
                elif t.match_type == "MOVED":
                    diff_parts.append(f"<{text}>")
                elif t.match_type == "REWORDED":
                    diff_parts.append(f"~{text}~")
                elif t.match_type == "CRITICAL":
                    diff_parts.append(f"!!{text}!!")
                elif t.match_type == "MISSING":
                    diff_parts.append(f"-{text}-")
            
            lines.append(f"  Diff: {' '.join(diff_parts)}")
            lines.append(f"  Doc:  '{m.doc_sentence.text}'")
            
            # Detailed Word Map
            details = [t for t in m.token_alignment if t.match_type != "EXACT"]
            if details:
                lines.append("  Details:")
                for t in details:
                    target = f"'{t.doc_token.text}'" if t.doc_token else "(MISSING)"
                    pos_info = f" (@{t.doc_token.in_sentence_idx})" if t.match_type == "MOVED" else ""
                    lines.append(f"    {t.query_token.text:<15} -> {target:<15} [{t.match_type}]{pos_info} score={t.score:.2f}")

        if not self.missing_sentences and not matches_to_show:
             lines.append("\nNo anomalies found. Perfect match.")
                
        return "\n".join(lines)


class AnomalyAnalyzerV2:
    def __init__(self, engine: HierarchicalEngine, critic: CriticModelV2):
        self.engine = engine
        self.critic = critic

    def analyze(self, query_text: str, doc_text: str) -> AnalysisResultV2:
        # 1. Embed Documents (Hierarchisch)
        q_doc = self.engine.embed_document(query_text)
        d_doc = self.engine.embed_document(doc_text)
        
        matches = []
        missing = []
        
        if not q_doc.sentences: return AnalysisResultV2([], [], 0.0)
        if not d_doc.sentences: return AnalysisResultV2([], q_doc.sentences, 0.0)

        q_vecs = np.stack([s.dense_vec for s in q_doc.sentences])
        d_vecs = np.stack([s.dense_vec for s in d_doc.sentences])
        sim_matrix = q_vecs @ d_vecs.T
        
        for i, q_sent in enumerate(q_doc.sentences):
            best_doc_idx = np.argmax(sim_matrix[i])
            best_score = sim_matrix[i][best_doc_idx]
            
            # Rescue Logic
            if best_score < 0.40:
                missing.append(q_sent)
                continue
            
            d_sent = d_doc.sentences[best_doc_idx]
            match = SentenceMatch(q_sent, d_sent, float(best_score))
            
            # --- ENSEMBLE CHECKS (Execute ALL) ---
            
            # 1. Keywords (SPLADE)
            sorted_tokens = sorted(q_sent.tokens, key=lambda t: t.sparse_weight, reverse=True)
            important_tokens = [t for t in sorted_tokens if t.sparse_weight > 1.0][:15]
            
            missing_keys = []
            d_words_lower = set(t.text.lower() for t in d_sent.tokens)
            
            for t in important_tokens:
                clean_key = re.sub(r'[^\w]', '', t.text.lower())
                if not clean_key: continue
                found = False
                for dw in d_words_lower:
                    if clean_key in dw: 
                        found = True; break
                if not found: missing_keys.append(t.text)
            
            if missing_keys:
                match.has_keyword_mismatch = True
                match.missing_keywords = missing_keys

            # 2. Entities (GLiNER)
            entities = [t.text for t in q_sent.tokens if t.ent_type]
            missing_ents = []
            for ent in entities:
                clean_ent = re.sub(r'[^\w]', '', ent.lower())
                if not clean_ent: continue
                found = False
                for dw in d_words_lower:
                    if clean_ent in dw:
                        found = True; break
                if not found: missing_ents.append(ent)
            
            if missing_ents:
                match.has_entity_mismatch = True
                match.missing_entities = missing_ents

            # 3. Token Alignment & Structure (Spacy)
            match.token_alignment = self._align_tokens(q_sent, d_sent, False) # Contradiction noch unbekannt
            
            structural_hint = ""
            for t in match.token_alignment:
                if t.doc_token and t.query_token.dep:
                    q_dep = t.query_token.dep
                    d_dep = t.doc_token.dep
                    if "nsubj" in q_dep and "obj" in d_dep:
                        match.has_structure_mismatch = True
                        t.match_type = "MOVED"
                        structural_hint = "Achtung: Subjekt wurde zum Objekt."
                    elif "obj" in q_dep and "nsubj" in d_dep:
                        match.has_structure_mismatch = True
                        t.match_type = "MOVED"
                        structural_hint = "Achtung: Objekt wurde zum Subjekt."

            # 4. LLM Critic (IMMER)
            context = {
                "score": float(best_score),
                "missing_keywords": match.missing_keywords,
                "missing_entities": match.missing_entities,
                "hint": structural_hint
            }
            decision = self.critic.evaluate(q_sent.text, d_sent.text, context=context)
            
            if decision["contradiction"]:
                match.llm_verdict = True
                match.reason = decision["reason"]
                
            # Final Alignment Update (falls Contradiction detected)
            if match.contradiction:
                # Re-Run Alignment mit Contradiction Flag für CRITICAL Markierung
                match.token_alignment = self._align_tokens(q_sent, d_sent, True)
            
            matches.append(match)
            
        total_found = sum(len([t for t in s.tokens if t.ent_type]) for s in q_doc.sentences)
        total_missing = sum(len(m.missing_entities) for m in matches)

        return AnalysisResultV2(
            matches=matches,
            missing_sentences=missing,
            doc_similarity=float(np.mean(np.max(sim_matrix, axis=1))),
            total_entities_found=total_found,
            total_entities_missing=total_missing
        )

    def _align_tokens(self, q_sent: SentenceInfo, d_sent: SentenceInfo, is_contradiction: bool) -> List[TokenMatch]:
        """Aligniert die Tokens zweier gematchter Sätze."""
        alignment = []
        
        if not q_sent.tokens: return []
        if not d_sent.tokens: 
            return [TokenMatch(t, "MISSING") for t in q_sent.tokens]

        q_vecs = np.stack([t.dense_vec for t in q_sent.tokens])
        d_vecs = np.stack([t.dense_vec for t in d_sent.tokens])
        
        sim_matrix = q_vecs @ d_vecs.T
        
        for i, q_token in enumerate(q_sent.tokens):
            best_idx = np.argmax(sim_matrix[i])
            best_score = sim_matrix[i][best_idx]
            d_token = d_sent.tokens[best_idx]
            
            match_type = "MISSING"
            
            # Veto-Logik
            is_important = q_token.sparse_weight > 1.0
            is_entity = bool(q_token.ent_type)
            is_critical_pos = q_token.pos in ["ADJ", "ADV", "VERB", "NUM", "PROPN"]
            
            veto = False
            
            if is_entity:
                if q_token.text.lower() != d_token.text.lower():
                    if q_token.ent_type != d_token.ent_type:
                        veto = True
                    elif best_score < 0.85:
                        veto = True
            elif is_important and is_critical_pos:
                if q_token.text.lower() != d_token.text.lower():
                    if q_token.pos != d_token.pos:
                        veto = True
                    elif best_score < 0.82:
                        veto = True
            elif is_important:
                if q_token.text.lower() != d_token.text.lower():
                    if best_score < 0.80 and d_token.sparse_weight < 0.8:
                        veto = True
            
            if veto:
                d_token = None
                best_score = 0.0
            elif q_token.text == d_token.text:
                rel_q = i / len(q_sent.tokens)
                rel_d = best_idx / len(d_sent.tokens)
                if abs(rel_q - rel_d) > 0.3:
                    match_type = "MOVED"
                else:
                    match_type = "EXACT"
            elif best_score > 0.65:
                if is_contradiction:
                    match_type = "CRITICAL"
                else:
                    match_type = "REWORDED"
            else:
                d_token = None
            
            if match_type == "MISSING" and not q_token.text.isalnum():
                for dt in d_sent.tokens:
                    if dt.text == q_token.text:
                        match_type = "EXACT"
                        d_token = dt
                        break

            alignment.append(TokenMatch(
                query_token=q_token,
                match_type=match_type,
                doc_token=d_token,
                score=float(best_score) if d_token else 0.0
            ))
            
        return alignment

if __name__ == "__main__":
    print("=== Analyzer V2 (Hierarchical) Test ===\n")
    engine = HierarchicalEngine()
    critic = LlamaCppCriticV2()
    analyzer = AnomalyAnalyzerV2(engine, critic)
    
    q = "Das Mädchen bekommt ein Meerschweinchen. Die Frist beträgt 2 Wochen."
    d = "Die Frist beläuft sich auf 14 Tage. Ein Häschen wird dem Mädchen geschenkt."
    
    print(f"Query: {q}")
    print(f"Doc:   {d}\n")
    
    result = analyzer.analyze(q, d)
    print(result.summary())