#!/usr/bin/env python3
"""
Semantic Engine - Kern-Modul für semantische Textanalyse.

Dieses Modul bildet das Herzstück der Semantic Engine. Es kombiniert zwei
Embedding-Modelle zu einer hybriden Repräsentation:

1. **Jina v3 (Dense)**: Erzeugt 1024-dimensionale Vektoren mit Matryoshka-
   Unterstützung. Diese erfassen die semantische Bedeutung auf kontinuierlicher
   Ebene - ähnliche Konzepte haben ähnliche Vektoren.

2. **Opensearch Sparse (SPLADE-Style)**: Erzeugt sparse Gewichtungen pro Token.
   Diese zeigen an, welche Begriffe besonders wichtig/charakteristisch sind -
   ähnlich wie TF-IDF, aber kontextabhängig.

Die Kombination ermöglicht:
- **Dense**: Semantische Ähnlichkeit (auch bei Synonymen, Paraphrasen)
- **Sparse**: Lexikalische Präzision (wichtige Schlüsselwörter nicht übersehen)

Kernkonzept: Wort-Level statt Token-Level
------------------------------------------
Transformer-Tokenizer zerlegen Text in Subword-Tokens ("Kündigungsfrist" →
["Künd", "##igungs", "##frist"]). Für die Analyse ist das ungünstig, weil:

1. Ein Token allein hat oft keine klare Bedeutung
2. Scores auf Token-Ebene sind schwer zu interpretieren
3. Einzelne Tokens können den Gesamtscore stark verzerren

Lösung: Wir gruppieren Tokens zurück zu Wörtern und aggregieren:
- Dense: Mean-Pooling über alle Tokens eines Wortes
- Sparse: Max-Pooling (das wichtigste Token bestimmt die Wort-Wichtigkeit)

Jeder Tokenizer nutzt seine eigenen word_ids() für maximale Konsistenz.

Sliding Window mit SLERP
------------------------
Für lange Texte (>8192 Tokens bei Dense, >512 bei Sparse) nutzen wir
Sliding Windows mit Overlap. Überlappende Bereiche werden via SLERP
(Spherical Linear Interpolation) kombiniert - das ist besser als lineares
Mitteln, weil es auf der Hypersphere arbeitet und die Richtung erhält.

Vektorlänge als Signal
----------------------
Optional kann die L2-Norm (Länge) der Vektoren erhalten bleiben statt
zu normalisieren. Die Länge könnte Information enthalten:
- Längere Vektoren = stärkere semantische Aktivierung
- Kürzere Vektoren = weniger "confident" / mehr ambig

Referenzen:
    - frankenstein_dsp_v3.py: SLERP, Sliding Window für Sparse
    - S3_04_Robust_Audit.py: Token-Level Similarity Matrix
    - Jina v3: https://huggingface.co/jinaai/jina-embeddings-v3
    - Opensearch Sparse: https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
import warnings
import logging

# Logger beruhigen (Transformers ist sehr gesprächig)
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)


def slerp(v0: torch.Tensor, v1: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    """
    Spherical Linear Interpolation zwischen zwei Vektoren.

    SLERP interpoliert auf der Hypersphere statt linear im euklidischen Raum.
    Das ist besser für Embeddings, weil:
    - Die Richtung (semantische Bedeutung) wird besser erhalten
    - Kein "Shrinking" zur Mitte hin wie bei linearem Mitteln
    - Mathematisch korrekt für normalisierte Vektoren

    Formel:
        slerp(v0, v1, t) = sin((1-t)*θ)/sin(θ) * v0 + sin(t*θ)/sin(θ) * v1
        wobei θ = arccos(v0 · v1)

    Bei fast parallelen Vektoren (dot > 0.9995) fällt SLERP auf
    normalisiertes lineares Interpolieren zurück (numerisch stabiler).

    Parameters
    ----------
    v0 : torch.Tensor
        Erster Vektor (sollte normalisiert sein).

    v1 : torch.Tensor
        Zweiter Vektor (sollte normalisiert sein).

    t : float
        Interpolationsfaktor [0, 1]. 0 = v0, 1 = v1, 0.5 = Mitte.

    Returns
    -------
    torch.Tensor
        Interpolierter Vektor auf der Hypersphere.

    Examples
    --------
    >>> v0 = F.normalize(torch.randn(1024), dim=0)
    >>> v1 = F.normalize(torch.randn(1024), dim=0)
    >>> mid = slerp(v0, v1, 0.5)
    >>> torch.norm(mid)  # Ist auch normalisiert
    tensor(1.0000)

    References
    ----------
    - frankenstein_dsp_v3.py, Zeilen 14-21
    - https://en.wikipedia.org/wiki/Slerp
    """
    dot = torch.dot(v0, v1)

    # Fallback für fast parallele Vektoren
    if dot > 0.9995:
        return F.normalize((1.0 - t) * v0 + t * v1, p=2, dim=-1)

    theta_0 = torch.acos(torch.clamp(dot, -1.0, 1.0))
    sin_theta_0 = torch.sin(theta_0)
    theta_t = theta_0 * t
    sin_theta_t = torch.sin(theta_t)

    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0

    return s0 * v0 + s1 * v1


@dataclass
class EmbeddingResult:
    """
    Ergebnis einer Text-Einbettung auf Wort-Ebene.

    Diese Klasse kapselt alle Informationen, die aus der Einbettung eines
    Textes resultieren. Sie ist als Dataclass implementiert für einfachen
    Zugriff auf die Felder.

    Attributes
    ----------
    words : list[str]
        Die extrahierten Wörter aus dem Eingabetext, in Reihenfolge.
        Beispiel: ["Die", "Bank", "erhöht", "die", "Zinsen", "."]

    dense_vectors : np.ndarray
        Dense Embedding-Vektoren pro Wort.
        Shape: (n_words, embedding_dim), typisch (n_words, 1024) für Jina v3.

        WICHTIG: Diese sind standardmäßig NICHT normalisiert!
        Die Vektorlänge enthält Information (Aktivierungsstärke).
        Für Cosine Similarity manuell normalisieren oder normalized_dense nutzen.

    dense_norms : np.ndarray
        L2-Normen (Längen) der Dense Vektoren vor Normalisierung.
        Shape: (n_words,)
        Kann als Confidence/Stärke-Signal interpretiert werden.

    sparse_weights : np.ndarray
        SPLADE-artige Gewichtungen pro Wort.
        Shape: (n_words,)
        Höhere Werte = wichtigere/charakteristischere Wörter.
        Typischer Bereich: 0.0 (Stoppwörter) bis 5.0+ (Schlüsselbegriffe).

    token_offsets : list[tuple[int, int]]
        Character-Offsets für jedes Wort im Originaltext.
        Ermöglicht Rückverfolgung zur Originalposition.
        Beispiel: [(0, 3), (4, 8), ...] für "Die Bank ..."

    text : str
        Der ursprüngliche Eingabetext (für Referenz).

    Examples
    --------
    >>> result = engine.embed_text("Kurzer Test")
    >>> result.words
    ['Kurzer', 'Test']
    >>> result.dense_vectors.shape
    (2, 1024)
    >>> result.normalized_dense.shape  # Normalisiert für Cosine Sim
    (2, 1024)
    """

    words: list[str]
    dense_vectors: np.ndarray  # Nicht normalisiert!
    sparse_weights: np.ndarray
    token_offsets: list[tuple[int, int]]
    text: str

    @property
    def normalized_dense(self) -> np.ndarray:
        """
        Gibt L2-normalisierte Dense Vektoren zurück.

        Nützlich für Cosine Similarity Berechnungen, bei denen
        dot product = cosine similarity gilt.

        Returns
        -------
        np.ndarray
            Normalisierte Vektoren, Shape (n_words, dim).
        """
        norms = np.linalg.norm(self.dense_vectors, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)  # Avoid division by zero
        return self.dense_vectors / norms

    def __len__(self) -> int:
        """Anzahl der Wörter."""
        return len(self.words)

    def get_top_keywords(self, k: int = 5) -> list[tuple[str, float]]:
        """
        Gibt die k wichtigsten Wörter nach Sparse-Gewichtung zurück.

        Parameters
        ----------
        k : int
            Anzahl der Top-Keywords (default: 5)

        Returns
        -------
        list[tuple[str, float]]
            Liste von (Wort, Gewicht) Tupeln, absteigend sortiert.

        Examples
        --------
        >>> result.get_top_keywords(3)
        [('Zinsen', 3.1), ('erhöht', 1.8), ('Bank', 2.3)]
        """
        indices = np.argsort(self.sparse_weights)[::-1][:k]
        return [(self.words[i], float(self.sparse_weights[i])) for i in indices]


class SemanticEngine:
    """
    Haupt-Engine für semantische Textanalyse.

    Diese Klasse lädt und verwaltet die ML-Modelle und stellt die
    Hauptfunktionalität für Text-Einbettung bereit.

    Features:
    - Sliding Window für lange Texte (SLERP für Overlap-Kombinierung)
    - Wort-Level Aggregation (jeder Tokenizer nutzt eigene word_ids)
    - Hybrid: Dense (Jina v3) + Sparse (Opensearch SPLADE)

    Parameters
    ----------
    device : str, optional
        Das Gerät für die Modelle ("cuda", "mps", "cpu").
        Default: Automatische Erkennung (CUDA > MPS > CPU).

    dense_model : str, optional
        Name/Pfad des Dense-Embedding-Modells.
        Default: "jinaai/jina-embeddings-v3"

    sparse_model : str, optional
        Name/Pfad des Sparse-Embedding-Modells.
        Default: "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1"

    Examples
    --------
    >>> engine = SemanticEngine()
    >>> result = engine.embed_text("Die Bank erhöht die Zinsen um 0.5%.")
    >>> print(f"Wörter: {result.words}")
    >>> print(f"Top Keywords: {result.get_top_keywords(3)}")
    """

    # Standard-Modellnamen als Klassenkonstanten
    DEFAULT_DENSE_MODEL = "jinaai/jina-embeddings-v3"
    DEFAULT_SPARSE_MODEL = "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1"

    # Sliding Window Konfiguration
    DENSE_MAX_LENGTH = 8192
    DENSE_STRIDE = 7000  # ~85% Overlap vermeiden, aber genug für Kontext
    SPARSE_MAX_LENGTH = 512
    SPARSE_STRIDE = 128  # Wie in frankenstein_dsp_v3.py

    def __init__(
        self,
        device: Optional[str] = None,
        dense_model: Optional[str] = None,
        sparse_model: Optional[str] = None,
    ):
        """
        Initialisiert die Semantic Engine.

        Lädt beide Modelle (Dense + Sparse) und bereitet sie für Inference vor.
        Die Modelle werden in eval() Modus gesetzt und Gradientenberechnung
        wird deaktiviert.
        """
        # Device-Erkennung
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"[SemanticEngine] Initialisiere auf {self.device}...")

        # Dense Model (Jina v3)
        dense_name = dense_model or self.DEFAULT_DENSE_MODEL
        print(f"[SemanticEngine] Lade Dense Model: {dense_name}")
        self.dense_tokenizer = AutoTokenizer.from_pretrained(
            dense_name, trust_remote_code=True
        )
        self.dense_model = AutoModel.from_pretrained(
            dense_name,
            trust_remote_code=True,
            attn_implementation="eager",
            use_flash_attn=False,
        ).to(self.device).eval()

        # Sparse Model (Opensearch)
        sparse_name = sparse_model or self.DEFAULT_SPARSE_MODEL
        print(f"[SemanticEngine] Lade Sparse Model: {sparse_name}")
        self.sparse_tokenizer = AutoTokenizer.from_pretrained(sparse_name)
        self.sparse_model = AutoModelForMaskedLM.from_pretrained(sparse_name).to(
            self.device
        ).eval()

        # Embedding-Dimension ermitteln
        self.dense_dim = self.dense_model.config.hidden_size

        print(f"[SemanticEngine] Bereit. Dense dim: {self.dense_dim}")

    def embed_text(self, text: str) -> EmbeddingResult:
        """
        Erzeugt Wort-Level Embeddings für einen Text.

        Dies ist die Hauptmethode der Engine. Sie:
        1. Tokenisiert mit beiden Modellen (jeweils eigene word_ids)
        2. Sliding Window für lange Texte (SLERP für Overlap)
        3. Aggregiert zu Wörtern: Dense = Mean, Sparse = Max

        Parameters
        ----------
        text : str
            Der zu verarbeitende Text. Kann beliebig lang sein.
            Sliding Window wird automatisch angewendet wenn nötig.

        Returns
        -------
        EmbeddingResult
            Dataclass mit words, dense_vectors (nicht normalisiert!),
            sparse_weights, etc.

        Examples
        --------
        >>> result = engine.embed_text("Die Bank erhöht die Zinsen.")
        >>> print(result.words)
        ['Die', 'Bank', 'erhöht', 'die', 'Zinsen', '.']

        >>> # Für Cosine Similarity: normalisierte Version
        >>> normed = result.normalized_dense
        >>> np.linalg.norm(normed[0])
        1.0
        """
        # --- Dense Embeddings (Jina v3) mit Sliding Window ---
        dense_words, dense_vectors, dense_offsets = self._embed_dense(text)

        # --- Sparse Weights (Opensearch) mit Sliding Window ---
        sparse_words, sparse_weights, sparse_offsets = self._embed_sparse(text)

        # --- Wörter matchen (Dense ist primär, Sparse wird zugeordnet) ---
        # Beide Tokenizer können unterschiedliche Wörter produzieren.
        # Wir nutzen Dense als Basis und matchen Sparse via Character-Offsets.
        final_sparse_weights = self._match_sparse_to_dense(
            dense_offsets, sparse_offsets, sparse_weights
        )

        return EmbeddingResult(
            words=dense_words,
            dense_vectors=dense_vectors,
            sparse_weights=final_sparse_weights,
            token_offsets=dense_offsets,
            text=text,
        )

    def _embed_dense(self, text: str) -> tuple[list[str], np.ndarray, list]:
        """
        Erzeugt Dense Embeddings mit Sliding Window und SLERP.

        Für Texte länger als DENSE_MAX_LENGTH werden überlappende
        Fenster verarbeitet und via SLERP kombiniert.

        Returns
        -------
        tuple
            (words, vectors, offsets)
            - words: Liste der Wörter
            - vectors: (n_words, dim) NICHT normalisiert
            - offsets: [(start, end), ...] Character-Positionen
        """
        # Tokenisiere den gesamten Text
        full_encoding = self.dense_tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=False,
            return_offsets_mapping=True,
        )

        input_ids = full_encoding["input_ids"][0]
        offset_mapping = full_encoding["offset_mapping"][0].numpy()
        full_word_ids = full_encoding.word_ids(batch_index=0)
        total_tokens = len(input_ids)

        # Wenn kurz genug, kein Sliding Window nötig
        if total_tokens <= self.DENSE_MAX_LENGTH:
            return self._process_dense_chunk(
                text, input_ids, offset_mapping, full_word_ids
            )

        # Sliding Window für lange Texte
        print(f"[SemanticEngine] Dense Sliding Window: {total_tokens} Tokens")

        # Sammle Embeddings pro Wort-ID (für SLERP Kombinierung)
        word_embeddings = {}  # word_id -> list of (embedding, weight)
        word_infos = {}  # word_id -> (word_str, offset)

        stride = self.DENSE_STRIDE
        max_window_size = self.DENSE_MAX_LENGTH

        for start_idx in range(0, total_tokens, stride):
            end_idx = min(start_idx + max_window_size, total_tokens)
            window_size = end_idx - start_idx

            # Chunk extrahieren
            chunk_ids = input_ids[start_idx:end_idx].unsqueeze(0)
            chunk_offsets = offset_mapping[start_idx:end_idx]
            chunk_word_ids = full_word_ids[start_idx:end_idx]

            # Attention Mask
            attention_mask = torch.ones_like(chunk_ids)

            # Forward Pass
            with torch.no_grad():
                outputs = self.dense_model(
                    input_ids=chunk_ids.to(self.device),
                    attention_mask=attention_mask.to(self.device),
                )
                token_embeddings = outputs.last_hidden_state[0].cpu()

            # Wörter aggregieren
            current_word_id = None
            current_tokens = []
            current_token_indices = []
            current_offsets_list = []

            for i, wid in enumerate(chunk_word_ids):
                if wid is None:
                    if current_tokens:
                        self._accumulate_word_embedding(
                            text, current_word_id, current_tokens, current_token_indices,
                            current_offsets_list, word_embeddings, word_infos, window_size
                        )
                        current_tokens = []
                        current_token_indices = []
                        current_offsets_list = []
                        current_word_id = None
                    continue

                if wid != current_word_id:
                    if current_tokens:
                        self._accumulate_word_embedding(
                            text, current_word_id, current_tokens, current_token_indices,
                            current_offsets_list, word_embeddings, word_infos, window_size
                        )
                    current_word_id = wid
                    current_tokens = [token_embeddings[i]]
                    current_token_indices = [i]
                    current_offsets_list = [chunk_offsets[i]]
                else:
                    current_tokens.append(token_embeddings[i])
                    current_token_indices.append(i)
                    current_offsets_list.append(chunk_offsets[i])

            if current_tokens:
                self._accumulate_word_embedding(
                    text, current_word_id, current_tokens, current_token_indices,
                    current_offsets_list, word_embeddings, word_infos, window_size
                )

            if end_idx >= total_tokens:
                break

        # SLERP Kombinierung für überlappende Wörter
        words = []
        vectors = []
        offsets = []

        for word_id in sorted(word_embeddings.keys()):
            embeddings_list = word_embeddings[word_id]
            word_str, offset = word_infos[word_id]

            if len(embeddings_list) == 1:
                # Nur ein Embedding
                vec = embeddings_list[0][0]
            else:
                # Mehrere Embeddings (Overlap) - SLERP kombinieren
                # Gewichte basierend auf Position im Fenster (Mitte = höher)
                combined = embeddings_list[0][0]
                total_weight = embeddings_list[0][1]

                for emb, weight in embeddings_list[1:]:
                    # SLERP mit gewichtetem t
                    t = weight / (total_weight + weight)
                    # Normalisieren für SLERP
                    combined_norm = F.normalize(combined, dim=0)
                    emb_norm = F.normalize(emb, dim=0)
                    combined = slerp(combined_norm, emb_norm, t)
                    # Länge wiederherstellen (Durchschnitt)
                    avg_len = (torch.norm(embeddings_list[0][0]) + torch.norm(emb)) / 2
                    combined = combined * avg_len
                    total_weight += weight

                vec = combined

            vec_np = vec.numpy()

            words.append(word_str)
            vectors.append(vec_np)
            offsets.append(offset)

        return words, np.array(vectors), offsets

    def _accumulate_word_embedding(
        self,
        text: str,
        word_id: int,
        tokens: list,
        token_indices: list,
        offsets_list: list,
        word_embeddings: dict,
        word_infos: dict,
        window_size: int,
    ):
        """Akkumuliert Wort-Embeddings für SLERP Kombinierung."""
        start_char = int(offsets_list[0][0])
        end_char = int(offsets_list[-1][1])
        word_str = text[start_char:end_char].strip()

        if not word_str:
            return

        # Mean über Tokens
        stacked = torch.stack(tokens)
        mean_vec = stacked.mean(dim=0)

        # Gewicht basierend auf Position im Fenster (Mitte = höher)
        # Berechne durchschnittliche Position des Wortes im aktuellen Chunk
        if not token_indices:
            weight = 0.001
        else:
            avg_pos = sum(token_indices) / len(token_indices)
            chunk_center = window_size / 2.0
            distance_from_center = abs(avg_pos - chunk_center)
            # Normalisierte Distanz: 0 (Mitte) bis 1 (Rand)
            norm_distance = distance_from_center / (window_size / 2.0)
            weight = 1.0 / (1.0 + norm_distance)

        if word_id not in word_embeddings:
            word_embeddings[word_id] = []
            word_infos[word_id] = (word_str, (start_char, end_char))

        word_embeddings[word_id].append((mean_vec, weight))

    def _process_dense_chunk(
        self,
        text: str,
        input_ids: torch.Tensor,
        offset_mapping: np.ndarray,
        word_ids: list,
    ) -> tuple[list[str], np.ndarray, list]:
        """Verarbeitet einen einzelnen Dense Chunk (kein Sliding Window nötig)."""
        attention_mask = torch.ones(1, len(input_ids))

        with torch.no_grad():
            outputs = self.dense_model(
                input_ids=input_ids.unsqueeze(0).to(self.device),
                attention_mask=attention_mask.to(self.device),
            )
            token_embeddings = outputs.last_hidden_state[0].cpu()

        # Wörter aggregieren
        words = []
        vectors = []
        offsets = []

        current_word_id = None
        current_tokens = []
        current_offsets_list = []

        for i, wid in enumerate(word_ids):
            if wid is None:
                if current_tokens:
                    self._finalize_dense_word(
                        text, current_tokens, current_offsets_list,
                        words, vectors, offsets
                    )
                    current_tokens = []
                    current_offsets_list = []
                    current_word_id = None
                continue

            if wid != current_word_id:
                if current_tokens:
                    self._finalize_dense_word(
                        text, current_tokens, current_offsets_list,
                        words, vectors, offsets
                    )
                current_word_id = wid
                current_tokens = [token_embeddings[i]]
                current_offsets_list = [offset_mapping[i]]
            else:
                current_tokens.append(token_embeddings[i])
                current_offsets_list.append(offset_mapping[i])

        if current_tokens:
            self._finalize_dense_word(
                text, current_tokens, current_offsets_list,
                words, vectors, offsets
            )

        return words, np.array(vectors), offsets

    def _finalize_dense_word(
        self,
        text: str,
        tokens: list,
        offsets_list: list,
        words: list,
        vectors: list,
        offsets: list,
    ):
        """Finalisiert ein Dense Wort."""
        start_char = int(offsets_list[0][0])
        end_char = int(offsets_list[-1][1])
        word_str = text[start_char:end_char].strip()

        if not word_str:
            return

        stacked = torch.stack(tokens)
        mean_vec = stacked.mean(dim=0).numpy()

        words.append(word_str)
        vectors.append(mean_vec)
        offsets.append((start_char, end_char))

    def _embed_sparse(self, text: str) -> tuple[list[str], np.ndarray, list]:
        """
        Erzeugt Sparse Weights mit Sliding Window.

        Nutzt den Opensearch Sparse Tokenizer mit eigenem word_ids().
        Sliding Window ist eingebaut via return_overflowing_tokens.

        Returns
        -------
        tuple
            (words, weights, offsets)
        """
        # Tokenisiere mit Sliding Window (wie in frankenstein_dsp_v3.py)
        encoding = self.sparse_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.SPARSE_MAX_LENGTH,
            padding=True,
            return_offsets_mapping=True,
            stride=self.SPARSE_STRIDE,
            return_overflowing_tokens=True,
        )

        # Sammle Weights pro Wort über alle Chunks
        word_weights = {}  # (start, end) -> max_weight
        word_strings = {}  # (start, end) -> word_str

        input_ids = encoding["input_ids"].to(self.device)
        offset_mappings = encoding["offset_mapping"]

        with torch.no_grad():
            outputs = self.sparse_model(input_ids)
            # SPLADE-Style: log(1 + ReLU(logits)), max über Vokabular
            logits = outputs.logits
            token_weights = torch.log(1 + torch.relu(logits)).max(dim=-1).values
            token_weights = token_weights.cpu().numpy()

        # Über alle Chunks iterieren
        for chunk_idx in range(len(input_ids)):
            chunk_offsets = offset_mappings[chunk_idx].numpy()
            chunk_weights = token_weights[chunk_idx]

            # word_ids für diesen Chunk
            try:
                chunk_word_ids = encoding.word_ids(batch_index=chunk_idx)
            except:
                # Fallback wenn word_ids nicht verfügbar
                chunk_word_ids = [None] * len(chunk_offsets)

            current_word_id = None
            current_weights = []
            current_offsets_list = []

            for i, wid in enumerate(chunk_word_ids):
                if wid is None:
                    if current_weights:
                        self._finalize_sparse_word(
                            text, current_weights, current_offsets_list,
                            word_weights, word_strings
                        )
                        current_weights = []
                        current_offsets_list = []
                        current_word_id = None
                    continue

                if wid != current_word_id:
                    if current_weights:
                        self._finalize_sparse_word(
                            text, current_weights, current_offsets_list,
                            word_weights, word_strings
                        )
                    current_word_id = wid
                    current_weights = [chunk_weights[i]]
                    current_offsets_list = [chunk_offsets[i]]
                else:
                    current_weights.append(chunk_weights[i])
                    current_offsets_list.append(chunk_offsets[i])

            if current_weights:
                self._finalize_sparse_word(
                    text, current_weights, current_offsets_list,
                    word_weights, word_strings
                )

        # Sortiert nach Position zurückgeben
        sorted_offsets = sorted(word_weights.keys())
        words = [word_strings[off] for off in sorted_offsets]
        weights = np.array([word_weights[off] for off in sorted_offsets])

        return words, weights, sorted_offsets

    def _finalize_sparse_word(
        self,
        text: str,
        weights: list,
        offsets_list: list,
        word_weights: dict,
        word_strings: dict,
    ):
        """Finalisiert ein Sparse Wort (Max-Pool über Chunks)."""
        start_char = int(offsets_list[0][0])
        end_char = int(offsets_list[-1][1])

        if start_char == end_char:
            return

        word_str = text[start_char:end_char].strip()
        if not word_str:
            return

        key = (start_char, end_char)
        max_weight = max(weights)

        # Max-Pool über Chunks (bei Overlap)
        if key in word_weights:
            word_weights[key] = max(word_weights[key], max_weight)
        else:
            word_weights[key] = max_weight
            word_strings[key] = word_str

    def _match_sparse_to_dense(
        self,
        dense_offsets: list,
        sparse_offsets: list,
        sparse_weights: np.ndarray,
    ) -> np.ndarray:
        """
        Matcht Sparse Weights zu Dense Wörtern via Character-Offsets.

        Die beiden Tokenizer können unterschiedliche Wort-Grenzen haben.
        Wir nutzen Offset-Overlap um Sparse Weights den Dense Wörtern zuzuordnen.

        Returns
        -------
        np.ndarray
            Sparse Weights für jedes Dense Wort.
        """
        result = np.zeros(len(dense_offsets), dtype=np.float32)

        # Baue Lookup für Sparse
        sparse_lookup = list(zip(sparse_offsets, sparse_weights))

        for i, (d_start, d_end) in enumerate(dense_offsets):
            matching_weights = []

            for (s_start, s_end), weight in sparse_lookup:
                # Check Overlap
                if max(d_start, s_start) < min(d_end, s_end):
                    matching_weights.append(weight)

            if matching_weights:
                result[i] = max(matching_weights)  # Max-Pool

        return result

    def compute_similarity_matrix(
        self,
        result_a: EmbeddingResult,
        result_b: EmbeddingResult,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Berechnet die Cosine Similarity Matrix zwischen zwei Texten.

        Parameters
        ----------
        result_a : EmbeddingResult
            Embedding-Ergebnis des ersten Textes (z.B. Query).

        result_b : EmbeddingResult
            Embedding-Ergebnis des zweiten Textes (z.B. Dokument).

        normalize : bool
            Ob die Vektoren normalisiert werden sollen (default: True).
            True = Cosine Similarity, False = Dot Product (Länge zählt mit).

        Returns
        -------
        np.ndarray
            Similarity Matrix, Shape (len(result_a), len(result_b)).
        """
        if normalize:
            vecs_a = result_a.normalized_dense
            vecs_b = result_b.normalized_dense
        else:
            vecs_a = result_a.dense_vectors
            vecs_b = result_b.dense_vectors

        return vecs_a @ vecs_b.T


# --- Hilfsfunktionen ---

def adaptive_threshold(
    scores: np.ndarray,
    rel_factor: float = 2.0,
    min_threshold: float = 0.5,
) -> float:
    """
    Berechnet einen adaptiven Threshold basierend auf der Score-Verteilung.

    Statt eines fixen Thresholds (z.B. 0.85) passt sich dieser Ansatz
    an die tatsächliche Verteilung der Scores an.

    Formel: threshold = max(mean - rel_factor * std, min_threshold)

    Parameters
    ----------
    scores : np.ndarray
        Array von Similarity Scores.

    rel_factor : float
        Multiplikator für Standardabweichung (default: 2.0).

    min_threshold : float
        Absolutes Minimum (default: 0.5).

    Returns
    -------
    float
        Der berechnete Threshold.
    """
    flat_scores = scores.flatten()
    mean = np.mean(flat_scores)
    std = np.std(flat_scores)
    return max(mean - rel_factor * std, min_threshold)


if __name__ == "__main__":
    # Quick Test
    print("=== Semantic Engine Test ===\n")

    engine = SemanticEngine()

    test_text = "Die Bank erhöht die Zinsen um 0.5 Prozent."
    print(f"Input: {test_text}\n")

    result = engine.embed_text(test_text)

    print(f"Wörter ({len(result)}):")
    for i, (word, weight) in enumerate(
        zip(result.words, result.sparse_weights)
    ):
        print(f"  {i}: {word:<12} sparse={weight:.2f}")

    print(f"\nDense Shape: {result.dense_vectors.shape}")
    print(f"Top Keywords (Sparse): {result.get_top_keywords(3)}")

    # Similarity Test
    print("\n--- Similarity Test ---")
    query = engine.embed_text("Bank Zinsen erhöhen")
    doc = engine.embed_text("Die Bank senkt die Zinsen nicht")

    sim = engine.compute_similarity_matrix(query, doc)
    print(f"Query: {query.words}")
    print(f"Doc: {doc.words}")
    print(f"\nSimilarity Matrix (normalized):")

    # Formatierte Ausgabe
    print(" " * 12, end="")
    for w in doc.words:
        print(f"{w[:8]:>9}", end="")
    print()

    for i, qw in enumerate(query.words):
        print(f"{qw[:11]:<12}", end="")
        for j in range(len(doc.words)):
            print(f"{sim[i,j]:>9.2f}", end="")
        print()

    # Nicht-normalisiert (Dot Product)
    print("\nSimilarity Matrix (dot product, Länge zählt):")
    sim_dot = engine.compute_similarity_matrix(query, doc, normalize=False)

    print(" " * 12, end="")
    for w in doc.words:
        print(f"{w[:8]:>9}", end="")
    print()

    for i, qw in enumerate(query.words):
        print(f"{qw[:11]:<12}", end="")
        for j in range(len(doc.words)):
            print(f"{sim_dot[i,j]:>9.1f}", end="")
        print()
