#!/usr/bin/env python3
"""
Semantic Engine - Command Line Interface.

Dieses Modul stellt eine CLI für die Semantic Engine bereit.

Befehle:
    analyze     Vergleicht Query mit Dokument und findet Anomalien
    embed       Erzeugt Embeddings für einen Text
    keywords    Extrahiert Top-Keywords aus einem Text

Optionen:
    --device {cuda,mps,cpu}   Device für ML-Modelle (default: auto)

    analyze:
        --query, -q           Query-Text direkt
        --query-file, -qf     Query aus Datei laden
        --doc, -d             Dokument-Text direkt
        --doc-file, -df       Dokument aus Datei laden (PDF oder Text)
        --threshold, -t       Fester Similarity-Threshold (default: adaptiv)
        --min-threshold       Minimaler Threshold (default: 0.5)
        --verbose, -v         Mehr Details ausgeben

    embed:
        --text, -t            Text direkt
        --file, -f            Text aus Datei laden
        --limit, -l           Max. Anzahl Wörter in Ausgabe (default: 50)

    keywords:
        --text, -t            Text direkt
        --file, -f            Text aus Datei laden
        --top, -n             Anzahl Keywords (default: 10)

Beispiele:
    # Anomalie-Erkennung
    python -m semantic_engine analyze \\
        --query "Die Bank teilt Zinsen mit." \\
        --doc "Die Bank teilt Zinsen nicht mit."

    # Mit PDF
    python -m semantic_engine analyze \\
        --query "Kündigungsfrist sechs Wochen" \\
        --doc-file contract.pdf

    # Embeddings erzeugen
    python -m semantic_engine embed --text "Beispieltext"
    python -m semantic_engine embed --file document.pdf --limit 20

    # Keywords extrahieren
    python -m semantic_engine keywords --file document.pdf --top 10
    python -m semantic_engine keywords --text "Ein langer Text..." --top 5
"""

import argparse
import sys
import os
import re
from pathlib import Path
from typing import Optional

# PDF Support (optional)
try:
    import fitz  # PyMuPDF
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

try:
    from .engine import SemanticEngine
    from .analyzer import AnomalyAnalyzer, AnomalyType
    from .critic import LlamaCppCritic
except ImportError:
    from engine import SemanticEngine
    from analyzer import AnomalyAnalyzer, AnomalyType
    from critic import LlamaCppCritic


def extract_text_from_pdf(path: str) -> str:
    """
    Extrahiert Text aus einer PDF-Datei.

    Parameters
    ----------
    path : str
        Pfad zur PDF-Datei.

    Returns
    -------
    str
        Extrahierter Text.

    Raises
    ------
    RuntimeError
        Wenn PyMuPDF nicht installiert ist.
    FileNotFoundError
        Wenn die Datei nicht existiert.
    """
    if not HAS_PDF:
        raise RuntimeError(
            "PyMuPDF (fitz) ist nicht installiert. "
            "Installiere es via: pip install pymupdf"
        )

    if not os.path.exists(path):
        raise FileNotFoundError(f"Datei nicht gefunden: {path}")

    doc = fitz.open(path)
    text_parts = []

    for page in doc:
        text = page.get_text("text")
        # Einfaches Cleaning
        text = text.replace('-\n', '').replace('\n', ' ')
        text_parts.append(text)

    return " ".join(text_parts)


def read_text_file(path: str) -> str:
    """Liest eine Textdatei."""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def load_document(path: str) -> str:
    """
    Lädt ein Dokument (PDF oder Text).

    Parameters
    ----------
    path : str
        Pfad zur Datei.

    Returns
    -------
    str
        Extrahierter/gelesener Text.
    """
    path_obj = Path(path)

    if not path_obj.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {path}")

    if path_obj.suffix.lower() == '.pdf':
        return extract_text_from_pdf(path)
    else:
        return read_text_file(path)


def cmd_analyze(args, engine: SemanticEngine):
    """Führt Anomalie-Analyse durch."""
    # Critic initialisieren wenn angegeben
    critic = None
    if args.critic:
        print(f"[CLI] Nutze LLM-Critic: {args.critic}")
        critic = LlamaCppCritic(base_url=args.critic)

    analyzer = AnomalyAnalyzer(engine, critic=critic)

    # Query laden
    if args.query:
        query_text = args.query
    elif args.query_file:
        query_text = load_document(args.query_file)
    else:
        print("Fehler: --query oder --query-file erforderlich", file=sys.stderr)
        sys.exit(1)

    # Dokument laden
    if args.doc:
        doc_text = args.doc
    elif args.doc_file:
        doc_text = load_document(args.doc_file)
    else:
        print("Fehler: --doc oder --doc-file erforderlich", file=sys.stderr)
        sys.exit(1)

    # Analyse durchführen
    print(f"Query: {query_text[:100]}{'...' if len(query_text) > 100 else ''}")
    print(f"Dokument: {len(doc_text)} Zeichen")
    print()

    result = analyzer.analyze(
        query=query_text,
        document=doc_text,
        threshold=args.threshold,
        min_threshold=args.min_threshold,
    )

    # Ausgabe
    print("=" * 60)
    print("ANALYSE-ERGEBNIS")
    print("=" * 60)
    print(result.summary())
    print()

    if result.anomalies:
        print("-" * 60)
        print("ANOMALIEN")
        print("-" * 60)

        # Nach Typ gruppieren
        for atype in [AnomalyType.MISSING, AnomalyType.EXTRA, AnomalyType.CHANGED]:
            typed = [a for a in result.anomalies if a.type == atype]
            if typed:
                print(f"\n{atype.value.upper()} ({len(typed)}):")
                for a in typed:
                    crit_marker = " [CRITICAL]" if a.criticality > 0.5 else ""
                    match_info = f" -> '{a.matched_to}'" if a.matched_to else ""
                    print(f"  - '{a.word}'{match_info} (score={a.score:.2f}, crit={a.criticality:.2f}){crit_marker}")
    else:
        print("Keine Anomalien gefunden.")

    # Verbose: Similarity Matrix
    if args.verbose:
        print()
        print("-" * 60)
        print("SIMILARITY MATRIX")
        print("-" * 60)
        print(f"Query Words: {result.query_result.words}")
        print(f"Doc Words: {result.doc_result.words[:20]}{'...' if len(result.doc_result.words) > 20 else ''}")
        print(f"Matrix Shape: {result.similarity_matrix.shape}")
        print(f"Threshold: {result.threshold:.3f}")


def cmd_embed(args, engine: SemanticEngine):
    """Erzeugt Embeddings für einen Text."""
    # Text laden
    if args.text:
        text = args.text
    elif args.file:
        text = load_document(args.file)
    else:
        print("Fehler: --text oder --file erforderlich", file=sys.stderr)
        sys.exit(1)

    result = engine.embed_text(text)

    print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")
    print(f"Wörter: {len(result.words)}")
    print(f"Dense Shape: {result.dense_vectors.shape}")
    print()

    # Wörter mit Sparse Weights anzeigen
    print("Wörter (mit Sparse Weight):")
    for i, (word, weight) in enumerate(zip(result.words, result.sparse_weights)):
        if i >= args.limit:
            print(f"  ... ({len(result.words) - args.limit} weitere)")
            break
        print(f"  {i:3d}: {word:<20} sparse={weight:.2f}")


def cmd_keywords(args, engine: SemanticEngine):
    """Extrahiert Top-Keywords (aggregiert Duplikate)."""
    # Text laden
    if args.text:
        text = args.text
    elif args.file:
        text = load_document(args.file)
    else:
        print("Fehler: --text oder --file erforderlich", file=sys.stderr)
        sys.exit(1)

    result = engine.embed_text(text)

    # Alle Keywords sammeln und aggregieren
    word_stats = {}  # word -> [scores]
    for word, weight in zip(result.words, result.sparse_weights):
        # Normalisiere: Strip Punctuation und lowercase für Gruppierung
        key = re.sub(r'[^\w]', '', word.strip().lower())
        if not key or weight < 0.5:  # Filter niedrige Scores und leere Keys
            continue
        # Display: Punctuation entfernen, Großschreibung bevorzugen
        word_clean = re.sub(r'[^\w]', '', word)
        if key not in word_stats:
            word_stats[key] = {"display": word_clean, "scores": []}
        word_stats[key]["scores"].append(weight)
        # Großschreibung bevorzugen
        if word_clean and word_clean[0].isupper() and not word_stats[key]["display"][0].isupper():
            word_stats[key]["display"] = word_clean

    # Sortieren nach Max-Score
    sorted_keywords = sorted(
        word_stats.items(),
        key=lambda x: max(x[1]["scores"]),
        reverse=True
    )[:args.top]

    print(f"Top {args.top} Keywords:")
    for key, stats in sorted_keywords:
        scores = stats["scores"]
        display = stats["display"]
        count = len(scores)
        if count == 1:
            print(f"  {display:<20} {scores[0]:.2f}")
        else:
            print(f"  {display:<20} {min(scores):.2f}-{max(scores):.2f} ({count}x)")


def main():
    """CLI Einstiegspunkt."""
    parser = argparse.ArgumentParser(
        prog="semantic-engine",
        description="Semantic Engine - Hybride semantische Textanalyse",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "mps", "cpu"],
        default=None,
        help="Device für ML-Modelle (default: auto)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Verfügbare Befehle")

    # --- analyze ---
    p_analyze = subparsers.add_parser(
        "analyze",
        help="Vergleicht Query mit Dokument und findet Anomalien",
    )
    p_analyze.add_argument("--query", "-q", help="Query-Text direkt")
    p_analyze.add_argument("--query-file", "-qf", help="Query aus Datei laden")
    p_analyze.add_argument("--doc", "-d", help="Dokument-Text direkt")
    p_analyze.add_argument("--doc-file", "-df", help="Dokument aus Datei laden (PDF oder Text)")
    p_analyze.add_argument(
        "--threshold", "-t",
        type=float,
        default=None,
        help="Fester Similarity-Threshold (default: adaptiv)",
    )
    p_analyze.add_argument(
        "--min-threshold",
        type=float,
        default=0.5,
        help="Minimaler Threshold für adaptive Berechnung (default: 0.5)",
    )
    p_analyze.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Mehr Details ausgeben",
    )
    p_analyze.add_argument(
        "--critic",
        default=None,
        help="URL des LLM-Critic Servers für Criticality-Bewertung (z.B. http://localhost:8102)",
    )

    # --- embed ---
    p_embed = subparsers.add_parser(
        "embed",
        help="Erzeugt Embeddings für einen Text",
    )
    p_embed.add_argument("--text", "-t", help="Text direkt")
    p_embed.add_argument("--file", "-f", help="Text aus Datei laden")
    p_embed.add_argument(
        "--limit", "-l",
        type=int,
        default=50,
        help="Max. Anzahl Wörter in Ausgabe (default: 50)",
    )

    # --- keywords ---
    p_keywords = subparsers.add_parser(
        "keywords",
        help="Extrahiert Top-Keywords aus einem Text",
    )
    p_keywords.add_argument("--text", "-t", help="Text direkt")
    p_keywords.add_argument("--file", "-f", help="Text aus Datei laden")
    p_keywords.add_argument(
        "--top", "-n",
        type=int,
        default=10,
        help="Anzahl Keywords (default: 10)",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Engine initialisieren
    print("Initialisiere Semantic Engine...")
    engine = SemanticEngine(device=args.device)
    print()

    # Befehl ausführen
    if args.command == "analyze":
        cmd_analyze(args, engine)
    elif args.command == "embed":
        cmd_embed(args, engine)
    elif args.command == "keywords":
        cmd_keywords(args, engine)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
