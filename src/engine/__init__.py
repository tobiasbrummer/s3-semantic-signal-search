"""
Semantic Engine - Hybride semantische Textanalyse.

Dieses Paket kombiniert Dense Embeddings (Jina v3) mit Sparse Weights
(Opensearch SPLADE) für präzise semantische Textanalyse auf Wort-Ebene.

Hauptkomponenten:
    - SemanticEngine: Die Haupt-Engine für Text-Einbettung
    - EmbeddingResult: Datenstruktur für Einbettungsergebnisse
    - AnomalyAnalyzer: Findet Unterschiede zwischen Query und Dokument
    - Anomaly: Einzelne Anomalie (missing, extra, changed)
    - AnalysisResult: Vollständiges Analyseergebnis
    - adaptive_threshold: Hilfsfunktion für dynamische Thresholds

Schnellstart:
    >>> from semantic_engine import SemanticEngine
    >>> engine = SemanticEngine()
    >>> result = engine.embed_text("Die Bank erhöht die Zinsen.")
    >>> print(result.words)
    ['Die', 'Bank', 'erhöht', 'die', 'Zinsen', '.']
    >>> print(result.get_top_keywords(3))
    [('Zinsen', 3.1), ('Bank', 2.3), ('erhöht', 1.8)]

Für Similarity Matrix (Query vs Dokument):
    >>> query = engine.embed_text("Bank Zinsen erhöhen")
    >>> doc = engine.embed_text("Die Bank senkt die Zinsen nicht")
    >>> sim = engine.compute_similarity_matrix(query, doc)
    >>> # sim[i, j] = Ähnlichkeit zwischen query.words[i] und doc.words[j]

Für Anomalie-Erkennung:
    >>> from semantic_engine import AnomalyAnalyzer
    >>> analyzer = AnomalyAnalyzer(engine)
    >>> result = analyzer.analyze(
    ...     query="Die Bank teilt Zinsen mit.",
    ...     document="Die Bank teilt Zinsen nicht mit."
    ... )
    >>> print(result.summary())
    >>> for a in result.anomalies:
    ...     print(a.type, a.word, a.criticality)

Architektur:
    Das Paket ist in Phasen aufgebaut:
    - Phase 1: engine.py (Einbettung) + analyzer.py (Anomalie-Erkennung)
    - Phase 2: index.py (Dokumenten-Index)
    - Phase 3: Performance-Optimierungen

Siehe ROADMAP.org für Details zur Entwicklung.
"""

from .engine import SemanticEngine, EmbeddingResult, adaptive_threshold
from .analyzer import AnomalyAnalyzer, Anomaly, AnomalyType, AnalysisResult
from .critic import CriticModel, LlamaCppCritic, TransformersCritic, get_critic

__all__ = [
    # Engine
    "SemanticEngine",
    "EmbeddingResult",
    "adaptive_threshold",
    # Analyzer
    "AnomalyAnalyzer",
    "Anomaly",
    "AnomalyType",
    "AnalysisResult",
    # Critic
    "CriticModel",
    "LlamaCppCritic",
    "TransformersCritic",
    "get_critic",
]

__version__ = "0.3.0"
