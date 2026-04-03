#!/usr/bin/env python3
"""
Semantic Engine - Anomalie-Erkennung.

Dieses Modul analysiert die Unterschiede zwischen zwei Texten auf Wort-Ebene.
Es findet:
- **Missing**: Query-Wörter die im Dokument nicht vorkommen
- **Extra**: Dokument-Wörter die nicht zur Query passen

Die Kritikalität wird via LLM-Critic bewertet. Der Critic bekommt den
Satz-Kontext (vorheriger Satz + aktueller Satz + nächster Satz) mit
markiertem Anomalie-Wort und bewertet die semantische Bedeutung der Änderung.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Optional
import re
import numpy as np

try:
    from .engine import SemanticEngine, EmbeddingResult, adaptive_threshold
    from .critic import CriticModel, LlamaCppCritic
except ImportError:
    # Für direkte Ausführung als Skript
    from engine import SemanticEngine, EmbeddingResult, adaptive_threshold
    from critic import CriticModel, LlamaCppCritic


class AnomalyType(str, Enum):
    """Typ der Anomalie."""
    MISSING = "missing"  # Query-Wort fehlt im Dokument
    EXTRA = "extra"      # Dokument-Wort nicht in Query
    CHANGED = "changed"  # Semantisch ähnlich aber nicht identisch


@dataclass
class Anomaly:
    """
    Eine einzelne Anomalie zwischen Query und Dokument.

    Attributes
    ----------
    type : AnomalyType
        Art der Anomalie (missing, extra, changed).

    word : str
        Das betroffene Wort.

    position : int
        Position im jeweiligen Text (Index in words-Liste).

    score : float
        Similarity Score zum besten Match.
        Bei MISSING: Wie gut wurde das Query-Wort im Doc gefunden (niedrig = fehlt)
        Bei EXTRA: Wie gut passt das Doc-Wort zur Query (niedrig = fremd)

    criticality : float
        Semantische Kritikalität (0-1).
        Basiert auf Logic-Axis: Negationen, Zahlen, etc. haben hohe Werte.

    matched_to : str | None
        Bei CHANGED: Das Wort im anderen Text, zu dem gematcht wurde.
        Bei MISSING/EXTRA: None.

    context : tuple[int, int] | None
        Character-Offsets im Originaltext für Highlighting.
    """
    type: AnomalyType
    word: str
    position: int
    score: float
    criticality: float
    matched_to: Optional[str] = None
    context: Optional[tuple[int, int]] = None

    def __repr__(self) -> str:
        crit_str = f" CRITICAL" if self.criticality > 0.5 else ""
        match_str = f" -> '{self.matched_to}'" if self.matched_to else ""
        return f"Anomaly({self.type.value}: '{self.word}'{match_str} @{self.position} score={self.score:.2f}{crit_str})"


@dataclass
class AnalysisResult:
    """
    Ergebnis einer Anomalie-Analyse zwischen Query und Dokument.

    Attributes
    ----------
    anomalies : list[Anomaly]
        Liste aller gefundenen Anomalien, sortiert nach Kritikalität.

    similarity_score : float
        Gesamt-Ähnlichkeit zwischen Query und Dokument (0-1).
        Berechnet als gewichteter Durchschnitt der besten Matches.

    query_coverage : float
        Anteil der Query-Wörter, die im Dokument gefunden wurden (0-1).

    doc_coverage : float
        Anteil der Dokument-Wörter, die zur Query passen (0-1).

    threshold : float
        Der verwendete Similarity-Threshold.

    query_result : EmbeddingResult
        Das Embedding-Ergebnis der Query.

    doc_result : EmbeddingResult
        Das Embedding-Ergebnis des Dokuments.

    similarity_matrix : np.ndarray
        Die vollständige Similarity-Matrix (query x doc).
    """
    anomalies: list[Anomaly]
    similarity_score: float
    query_coverage: float
    doc_coverage: float
    threshold: float
    query_result: EmbeddingResult
    doc_result: EmbeddingResult
    similarity_matrix: np.ndarray

    def __len__(self) -> int:
        return len(self.anomalies)

    @property
    def has_critical(self) -> bool:
        """Gibt es kritische Anomalien?"""
        return any(a.criticality > 0.5 for a in self.anomalies)

    @property
    def missing(self) -> list[Anomaly]:
        """Nur MISSING Anomalien."""
        return [a for a in self.anomalies if a.type == AnomalyType.MISSING]

    @property
    def extra(self) -> list[Anomaly]:
        """Nur EXTRA Anomalien."""
        return [a for a in self.anomalies if a.type == AnomalyType.EXTRA]

    @property
    def changed(self) -> list[Anomaly]:
        """Nur CHANGED Anomalien."""
        return [a for a in self.anomalies if a.type == AnomalyType.CHANGED]

    def summary(self) -> str:
        """Kurze Zusammenfassung der Analyse."""
        lines = [
            f"Similarity: {self.similarity_score:.1%}",
            f"Query Coverage: {self.query_coverage:.1%}",
            f"Doc Coverage: {self.doc_coverage:.1%}",
            f"Anomalies: {len(self.anomalies)} ({len(self.missing)} missing, {len(self.extra)} extra, {len(self.changed)} changed)",
        ]
        if self.has_critical:
            crit = [a for a in self.anomalies if a.criticality > 0.5]
            lines.append(f"CRITICAL: {[a.word for a in crit]}")
        return "\n".join(lines)


class AnomalyAnalyzer:
    """
    Analysiert Unterschiede zwischen Query und Dokument.

    Diese Klasse nutzt die SemanticEngine um Wort-Level Embeddings zu erzeugen
    und vergleicht diese dann systematisch.

    Parameters
    ----------
    engine : SemanticEngine
        Die initialisierte Semantic Engine.

    criticality_model : str, optional
        Modell für Logic-Axis Berechnung.
        Default: Nutzt das Dense-Modell der Engine.

    Examples
    --------
    >>> engine = SemanticEngine()
    >>> analyzer = AnomalyAnalyzer(engine)
    >>> result = analyzer.analyze(
    ...     query="Die Bank teilt Zinsen mit.",
    ...     document="Die Bank teilt Zinsen nicht mit."
    ... )
    >>> print(result.summary())
    >>> for a in result.anomalies:
    ...     print(a)
    """

    # Punctuation und Stopwords die ignoriert werden
    SKIP_TOKENS = {
        ".", ",", ":", ";", "!", "?", "(", ")", "[", "]", "{", "}", '"', "'",
        "-", "–", "—", "/", "\\", "&", "+", "=", "*", "#", "@", "%", "^",
    }

    def __init__(self, engine: SemanticEngine, critic: Optional[CriticModel] = None):
        """
        Initialisiert den Analyzer.

        Parameters
        ----------
        engine : SemanticEngine
            Die Semantic Engine für Embeddings.

        critic : CriticModel
            LLM-basierter Critic für Criticality-Bewertung (Pflicht).
        """
        self.engine = engine
        self.critic = critic
        # Temporäre Speicherung für Kontext-Extraktion während analyze()
        self._current_query_text: Optional[str] = None
        self._current_doc_text: Optional[str] = None

    def _extract_sentence_context(
        self,
        text: str,
        word_offset: tuple[int, int],
        word: str,
        mark_word: bool = True,
    ) -> str:
        """
        Extrahiert den Satz mit dem Wort + Nachbarsätze als Kontext.

        Parameters
        ----------
        text : str
            Der vollständige Text.

        word_offset : tuple[int, int]
            Start- und End-Position des Wortes im Text.

        word : str
            Das Wort (zur Markierung).

        mark_word : bool
            Ob das Wort markiert werden soll (default: True).

        Returns
        -------
        str
            Kontext-String mit markiertem Wort.
            Format: "[vorheriger Satz] **Satz mit [WORT]**. [nächster Satz]"
        """
        start, end = word_offset

        # Satzgrenzen finden (. ! ? oder Zeilenumbruch)
        sentence_delimiters = re.compile(r'[.!?\n]+')

        # Finde alle Satzgrenzen
        boundaries = [0]
        for m in sentence_delimiters.finditer(text):
            boundaries.append(m.end())
        boundaries.append(len(text))

        # In welchem Satz liegt das Wort?
        current_sentence_idx = 0
        for i, (b_start, b_end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            if b_start <= start < b_end:
                current_sentence_idx = i
                break

        # Sätze extrahieren
        sentences = []
        for i, (b_start, b_end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            sent = text[b_start:b_end].strip()
            if sent:
                sentences.append((i, sent, b_start, b_end))

        if not sentences:
            return text

        # Finde den aktuellen Satz in der Liste
        current_idx_in_list = 0
        for idx, (orig_idx, sent, b_start, b_end) in enumerate(sentences):
            if b_start <= start < b_end:
                current_idx_in_list = idx
                break

        # Extrahiere: vorheriger Satz, aktueller Satz, nächster Satz
        result_parts = []

        # Vorheriger Satz (falls vorhanden)
        if current_idx_in_list > 0:
            result_parts.append(sentences[current_idx_in_list - 1][1])

        # Aktueller Satz mit Markierung
        _, current_sent, sent_start, _ = sentences[current_idx_in_list]
        if mark_word:
            # Position im Satz berechnen
            word_pos_in_sent = start - sent_start
            if 0 <= word_pos_in_sent < len(current_sent):
                # Wort im Satz markieren
                marked_sent = (
                    current_sent[:word_pos_in_sent]
                    + f"[{word}]"
                    + current_sent[word_pos_in_sent + len(word):]
                )
                result_parts.append(marked_sent)
            else:
                result_parts.append(f"[...] [{word}] [...]")
        else:
            result_parts.append(current_sent)

        # Nächster Satz (falls vorhanden)
        if current_idx_in_list < len(sentences) - 1:
            result_parts.append(sentences[current_idx_in_list + 1][1])

        return " ".join(result_parts)

    def analyze(
        self,
        query: str | EmbeddingResult,
        document: str | EmbeddingResult,
        threshold: Optional[float] = None,
        rel_factor: float = 2.0,
        min_threshold: float = 0.5,
        skip_punctuation: bool = True,
        z_score_threshold: float = -1.5,
        use_z_score: bool = True,
    ) -> AnalysisResult:
        """
        Analysiert Unterschiede zwischen Query und Dokument.

        Parameters
        ----------
        query : str | EmbeddingResult
            Die Query (Text oder bereits berechnetes Embedding).

        document : str | EmbeddingResult
            Das Dokument (Text oder bereits berechnetes Embedding).

        threshold : float, optional
            Fester Similarity-Threshold. Wenn None, wird adaptive_threshold verwendet.

        rel_factor : float
            Faktor für adaptive_threshold (default: 2.0).

        min_threshold : float
            Minimaler Threshold (default: 0.5).

        skip_punctuation : bool
            Ob Punctuation ignoriert werden soll (default: True).

        z_score_threshold : float
            Z-Score Threshold für Ausreißer-Erkennung (default: -1.5).
            Wörter mit Z-Score unter diesem Wert gelten als Anomalie.

        use_z_score : bool
            Ob Z-Score basierte Erkennung verwendet werden soll (default: True).
            Dies findet Wörter die statistisch schlechter matchen als der Rest.

        Returns
        -------
        AnalysisResult
            Vollständiges Analyseergebnis mit Anomalien und Metriken.
        """
        # Embeddings berechnen falls nötig
        if isinstance(query, str):
            self._current_query_text = query
            query_result = self.engine.embed_text(query)
        else:
            self._current_query_text = None
            query_result = query

        if isinstance(document, str):
            self._current_doc_text = document
            doc_result = self.engine.embed_text(document)
        else:
            self._current_doc_text = None
            doc_result = document

        # Similarity Matrix berechnen
        sim_matrix = self.engine.compute_similarity_matrix(query_result, doc_result)

        # Threshold bestimmen
        if threshold is None:
            threshold = adaptive_threshold(sim_matrix, rel_factor, min_threshold)

        # Anomalien finden
        anomalies = []

        # 1. Missing: Query-Wörter die im Dokument fehlen
        missing = self._find_missing(
            sim_matrix, query_result, doc_result, threshold, skip_punctuation
        )
        anomalies.extend(missing)

        # 2. Extra: Dokument-Wörter die nicht zur Query passen (Threshold-basiert)
        extra = self._find_extra(
            sim_matrix, query_result, doc_result, threshold, skip_punctuation
        )
        anomalies.extend(extra)

        # 3. Z-Score Outliers: Wörter die statistisch schlechter matchen
        if use_z_score:
            z_outliers = self._find_z_score_outliers(
                sim_matrix, query_result, doc_result, z_score_threshold, skip_punctuation
            )
            # Nur hinzufügen wenn nicht schon als extra markiert
            existing_positions = {(a.type, a.position) for a in anomalies}
            for outlier in z_outliers:
                if (outlier.type, outlier.position) not in existing_positions:
                    anomalies.append(outlier)

        # Nach Kritikalität sortieren (höchste zuerst)
        anomalies.sort(key=lambda a: a.criticality, reverse=True)

        # Metriken berechnen
        q_max_scores = np.max(sim_matrix, axis=1)
        d_max_scores = np.max(sim_matrix, axis=0)

        # Coverage: Anteil der Wörter über Threshold
        query_coverage = np.mean(q_max_scores >= threshold)
        doc_coverage = np.mean(d_max_scores >= threshold)

        # Similarity: Gewichteter Durchschnitt (Sparse-Weights)
        q_weights = query_result.sparse_weights
        if np.sum(q_weights) > 0:
            similarity_score = np.average(q_max_scores, weights=q_weights)
        else:
            similarity_score = np.mean(q_max_scores)

        return AnalysisResult(
            anomalies=anomalies,
            similarity_score=float(similarity_score),
            query_coverage=float(query_coverage),
            doc_coverage=float(doc_coverage),
            threshold=float(threshold),
            query_result=query_result,
            doc_result=doc_result,
            similarity_matrix=sim_matrix,
        )

    def _find_missing(
        self,
        sim_matrix: np.ndarray,
        query_result: EmbeddingResult,
        doc_result: EmbeddingResult,
        threshold: float,
        skip_punctuation: bool,
    ) -> list[Anomaly]:
        """
        Findet Query-Wörter die im Dokument nicht gefunden wurden.

        Ein Wort gilt als "missing" wenn sein bester Match-Score
        unter dem Threshold liegt.

        Returns
        -------
        list[Anomaly]
            Liste der MISSING Anomalien.
        """
        anomalies = []

        # Bester Match für jedes Query-Wort
        q_max_scores = np.max(sim_matrix, axis=1)
        q_best_idx = np.argmax(sim_matrix, axis=1)

        for i, (word, score) in enumerate(zip(query_result.words, q_max_scores)):
            # Skip Punctuation
            if skip_punctuation and word in self.SKIP_TOKENS:
                continue

            # Unter Threshold = Missing
            if score < threshold:
                best_match = doc_result.words[q_best_idx[i]]
                criticality = self._score_criticality(word, best_match)

                anomalies.append(Anomaly(
                    type=AnomalyType.MISSING,
                    word=word,
                    position=i,
                    score=float(score),
                    criticality=criticality,
                    matched_to=best_match if score > 0.3 else None,  # Nur bei halbwegs ähnlich
                    context=query_result.token_offsets[i] if i < len(query_result.token_offsets) else None,
                ))

        return anomalies

    def _find_extra(
        self,
        sim_matrix: np.ndarray,
        query_result: EmbeddingResult,
        doc_result: EmbeddingResult,
        threshold: float,
        skip_punctuation: bool,
    ) -> list[Anomaly]:
        """
        Findet Dokument-Wörter die nicht zur Query passen.

        Ein Wort gilt als "extra" wenn es zu keinem Query-Wort
        einen guten Match hat.

        Returns
        -------
        list[Anomaly]
            Liste der EXTRA Anomalien.
        """
        anomalies = []

        # Bester Match für jedes Doc-Wort
        d_max_scores = np.max(sim_matrix, axis=0)
        d_best_idx = np.argmax(sim_matrix, axis=0)

        for j, (word, score) in enumerate(zip(doc_result.words, d_max_scores)):
            # Skip Punctuation
            if skip_punctuation and word in self.SKIP_TOKENS:
                continue

            # Unter Threshold = Extra/Anomaly
            if score < threshold:
                best_match_idx = d_best_idx[j]
                best_match = query_result.words[best_match_idx]

                # Kontext für LLM-Critic extrahieren
                query_context = None
                doc_context = None
                if self.critic is not None and score > 0.3:  # Nur bei halbwegs ähnlich
                    if (self._current_query_text is not None
                        and best_match_idx < len(query_result.token_offsets)):
                        query_context = self._extract_sentence_context(
                            self._current_query_text,
                            query_result.token_offsets[best_match_idx],
                            best_match,
                        )
                    if (self._current_doc_text is not None
                        and j < len(doc_result.token_offsets)):
                        doc_context = self._extract_sentence_context(
                            self._current_doc_text,
                            doc_result.token_offsets[j],
                            word,
                        )

                criticality = self._score_criticality(
                    word, best_match,
                    query_context=query_context,
                    doc_context=doc_context,
                )

                anomalies.append(Anomaly(
                    type=AnomalyType.EXTRA,
                    word=word,
                    position=j,
                    score=float(score),
                    criticality=criticality,
                    matched_to=best_match if score > 0.3 else None,
                    context=doc_result.token_offsets[j] if j < len(doc_result.token_offsets) else None,
                ))

        return anomalies

    def _find_z_score_outliers(
        self,
        sim_matrix: np.ndarray,
        query_result: EmbeddingResult,
        doc_result: EmbeddingResult,
        z_threshold: float,
        skip_punctuation: bool,
    ) -> list[Anomaly]:
        """
        Findet Wörter die statistisch schlechter matchen als der Rest.

        Statt eines festen Thresholds berechnen wir den Z-Score:
        z = (score - mean) / std

        Ein Wort mit Z-Score < z_threshold (z.B. -1.5) ist ein Ausreißer -
        es matcht signifikant schlechter als die anderen Wörter.

        Dies ist besonders nützlich für Negationen wie "nicht", die
        zwar absolut hohe Scores haben (~0.93), aber relativ zur
        Verteilung der anderen Wörter (~0.97) ein Ausreißer sind.

        Parameters
        ----------
        sim_matrix : np.ndarray
            Similarity Matrix (query x doc).

        query_result : EmbeddingResult
            Query Embeddings.

        doc_result : EmbeddingResult
            Document Embeddings.

        z_threshold : float
            Z-Score Grenze für Ausreißer (default: -1.5).

        skip_punctuation : bool
            Ob Punctuation ignoriert werden soll.

        Returns
        -------
        list[Anomaly]
            Liste der statistischen Ausreißer als EXTRA Anomalien.
        """
        anomalies = []

        # Bester Match für jedes Doc-Wort
        d_max_scores = np.max(sim_matrix, axis=0)
        d_best_idx = np.argmax(sim_matrix, axis=0)

        # Z-Score berechnen
        mean_score = np.mean(d_max_scores)
        std_score = np.std(d_max_scores)

        # Wenn keine Varianz, keine Outliers möglich
        if std_score < 0.001:
            return anomalies

        z_scores = (d_max_scores - mean_score) / std_score

        for j, (word, score, z) in enumerate(zip(doc_result.words, d_max_scores, z_scores)):
            # Skip Punctuation
            if skip_punctuation and word in self.SKIP_TOKENS:
                continue

            # Z-Score unter Threshold = statistischer Ausreißer
            if z < z_threshold:
                best_match_idx = d_best_idx[j]
                best_match = query_result.words[best_match_idx]

                if word in best_match:
                    continue
                
                # Kontext für LLM-Critic extrahieren
                query_context = None
                doc_context = None
                if self.critic is not None and score > 0.3:
                    if (self._current_query_text is not None
                        and best_match_idx < len(query_result.token_offsets)):
                        query_context = self._extract_sentence_context(
                            self._current_query_text,
                            query_result.token_offsets[best_match_idx],
                            best_match,
                        )
                    if (self._current_doc_text is not None
                        and j < len(doc_result.token_offsets)):
                        doc_context = self._extract_sentence_context(
                            self._current_doc_text,
                            doc_result.token_offsets[j],
                            word,
                        )

                print(query_context, doc_context)
                criticality = self._score_criticality(
                    word, best_match,
                    query_context=query_context,
                    doc_context=doc_context,
                )

                # Kritikalität erhöhen wenn Sparse-Weight hoch ist
                # (wichtiges Wort das schlecht matcht = sehr kritisch)
                sparse_weight = doc_result.sparse_weights[j]
                if sparse_weight > 1.0:
                    criticality = min(1.0, criticality + 0.2)

                anomalies.append(Anomaly(
                    type=AnomalyType.EXTRA,
                    word=word,
                    position=j,
                    score=float(score),
                    criticality=criticality,
                    matched_to=best_match if score > 0.3 else None,
                    context=doc_result.token_offsets[j] if j < len(doc_result.token_offsets) else None,
                ))

        return anomalies

    def _score_criticality(
        self,
        word: str,
        matched_to: Optional[str] = None,
        query_context: Optional[str] = None,
        doc_context: Optional[str] = None,
    ) -> float:
        """
        Bewertet die semantische Kritikalität einer Wort-Änderung via LLM.

        Parameters
        ----------
        word : str
            Das neue/geänderte Wort (aus Dokument).

        matched_to : str, optional
            Das ursprüngliche Wort (aus Query) zu dem gematcht wurde.

        query_context : str, optional
            Satz-Kontext aus Query (mit markiertem Wort).

        doc_context : str, optional
            Satz-Kontext aus Dokument (mit markiertem Wort).

        Returns
        -------
        float
            Kritikalität zwischen 0 und 1.

        Raises
        ------
        RuntimeError
            Wenn kein Critic konfiguriert ist.
        """
        if self.critic is None:
            raise RuntimeError(
                "Kein LLM-Critic konfiguriert. "
                "Nutze: AnomalyAnalyzer(engine, critic=LlamaCppCritic(url))"
            )

        if query_context is None or doc_context is None:
            # Kein Kontext verfügbar (z.B. MISSING ohne Match)
            return 0.0

        return self.critic.score(old=query_context, new=doc_context)

if __name__ == "__main__":
    # Quick Test - benötigt llama.cpp Server auf localhost:8102
    import sys

    print("=== Anomaly Analyzer Test ===")
    print("Benötigt: llama.cpp Server auf http://localhost:8102\n")

    engine = SemanticEngine()
    critic = LlamaCppCritic(base_url="http://localhost:8102")
    analyzer = AnomalyAnalyzer(engine, critic=critic)

    test_cases = [
        ("Negation", "Die Bank teilt Zinsen mit.", "Die Bank teilt Zinsen nicht mit."),
        ("Zahlenänderung", "Zinssatz beträgt 5 Prozent.", "Zinssatz beträgt 50 Prozent."),
        ("Präfix-Negation", "Die Bank prüft Zahlungsfähigkeit.", "Die Bank prüft Zahlungsunfähigkeit."),
    ]

    for name, query, doc in test_cases:
        print(f"--- Test: {name} ---")
        try:
            result = analyzer.analyze(query=query, document=doc, z_score_threshold=-0.5)
            print(result.summary())
            print("\nAnomalien:")
            for a in result.anomalies:
                print(f"  {a.type.value}: '{a.word}' -> '{a.matched_to}' (crit={a.criticality:.2f})")
        except Exception as e:
            print(f"Fehler: {e}")
        print()
