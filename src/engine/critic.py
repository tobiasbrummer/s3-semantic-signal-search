#!/usr/bin/env python3
"""
Semantic Engine - LLM-basierte Criticality-Bewertung.

Dieses Modul stellt verschiedene Backends für die Bewertung der
semantischen Kritikalität von Textänderungen bereit.

Die Idee: Statt regelbasierter Listen (Logic-Axis) nutzen wir ein
kleines LLM das den Kontext versteht:
- "sechs Wochen" → "sechs Minuten" = Frist-Änderung (0.6)
- "mitteilen" → "nicht mit" = Negation (0.9)
- "Labrador" → "Golden Retriever" = Synonym (0.2)

Backends:
    - LlamaCppCritic: llama.cpp Server via HTTP (CPU-effizient)
    - TransformersCritic: HuggingFace transformers (für GPU)

Beispiel:
    >>> critic = LlamaCppCritic("http://localhost:8102")
    >>> score = critic.score(
    ...     old="Die Bank teilt Zinsen mit.",
    ...     new="Die Bank teilt Zinsen nicht mit."
    ... )
    >>> print(score)  # 0.9
"""

from abc import ABC, abstractmethod
from typing import Optional
import re


# System-Prompt für Criticality-Bewertung
CRITIC_SYSTEM_PROMPT = """Aufgabe: Bewerte die Kritikalität einer ERKANNTEN Textänderung für risiko-sensitive Dokumente (z.B. Verträge).
Gib NUR eine Zahl (0.0 bis 1.0) zurück.

WICHTIG: Hohe Werte sind selten. Standard ist niedrig.
Wenn keine klare risikorelevante Bedeutungsänderung vorliegt oder du unsicher bist: gib 0.0–0.2.

Bewertungsregeln (verwende die HÖCHSTE zutreffende):
1) 0.9–1.0 NUR wenn die Aussage logisch invertiert wird:
   - Negation hinzugefügt/entfernt (nicht/kein/ohne), Verbot↔Erlaubnis, muss↔darf nicht, immer↔nie.
2) 0.6–0.8 NUR wenn sich harte Vertragsparameter ändern:
   - Zahlen, Beträge, Fristen, Mengen, Prozente, Datumsangaben, Laufzeiten
   - Verantwortlichkeiten/Rechte/Pflichten (wer tut was), Bedingungen/Ausnahmen (außer/sofern/nur wenn)
3) 0.3–0.5 wenn der Scope/Präzision merklich geändert wird, aber keine harte Kategorie oben:
   - Einschränkung/Erweiterung, neue konkrete Eigenschaft, aber ohne Zahlen/Negation/Pflichtenwechsel
4) 0.0–0.2 wenn Bedeutung gleich bleibt:
   - Synonyme, Umformulierungen, Satzbau, gleiche Aussage mit anderer Wortwahl.

Zahlenänderung:
- Bestimme Zahltyp über Kontextwörter.
- Wenn Geld/Frist/Menge/Prozent/Haftungsgrenze/SLA => mindestens 0.6.
- Wenn ID/Telefon/PLZ/Referenz/Version => höchstens 0.2.
- Wenn unklar => 0.3.
- Extrem (>=0.9) nur bei logischer Umkehr (Negation/immer↔nie/unbegrenzt↔begrenzt) oder wenn Schwellenwerte/Bedingungen kippen.

Behandle Synonyme oder nahe Begriffe als niedrig, z.B. "Hase" vs "Kaninchen" => 0.0–0.2 (keine Negation, keine Zahl, keine Pflichtänderung). Verzichte auf Erklärungen, gib NUR die Bewertungszahl aus."""


class CriticModel(ABC):
    """
    Abstraktes Interface für Criticality-Bewertung.

    Subklassen implementieren verschiedene Backends (llama.cpp, transformers, etc.)
    """

    @abstractmethod
    def score(self, old: str, new: str) -> float:
        """
        Bewertet die Kritikalität einer Textänderung.

        Parameters
        ----------
        old : str
            Der ursprüngliche Text (aus Query).

        new : str
            Der geänderte Text (aus Dokument).

        Returns
        -------
        float
            Kritikalitäts-Score zwischen 0.0 und 1.0.
            - 0.0-0.2: Synonym/keine Änderung
            - 0.3-0.5: Scope-Änderung
            - 0.6-0.8: Harte Parameter (Zahlen, Fristen)
            - 0.9-1.0: Logische Inversion (Negation)
        """
        pass

    def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        """
        Bewertet mehrere Änderungen auf einmal.

        Default-Implementation ruft score() einzeln auf.
        Subklassen können das für Batch-Inference überschreiben.

        Parameters
        ----------
        pairs : list[tuple[str, str]]
            Liste von (old, new) Paaren.

        Returns
        -------
        list[float]
            Liste von Scores.
        """
        return [self.score(old, new) for old, new in pairs]


class LlamaCppCritic(CriticModel):
    """
    Criticality-Bewertung via llama.cpp HTTP Server.

    Erwartet einen llama.cpp Server mit OpenAI-kompatiblem API.

    Parameters
    ----------
    base_url : str
        URL des llama.cpp Servers (default: "http://localhost:8102").

    timeout : float
        Timeout für HTTP-Requests in Sekunden (default: 30).

    Examples
    --------
    >>> critic = LlamaCppCritic("http://localhost:8102")
    >>> critic.score("Frist: 6 Wochen", "Frist: 6 Minuten")
    0.6
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8102",
        model: str = "Huihui-granite-4.0-h-tiny-abliterated.i1-IQ4_NL",
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._session = None

    def _get_session(self):
        """Lazy-Init für HTTP Session."""
        if self._session is None:
            import requests
            self._session = requests.Session()
        return self._session

    def _build_prompt(self, old: str, new: str) -> str:
        """Baut den User-Prompt für eine Änderung."""
        return f'ALT: "{old}"\nNEU: "{new}"'

    def _parse_score(self, response_text: str) -> float:
        """Extrahiert den Score aus der LLM-Antwort."""
        # Suche nach einer Zahl zwischen 0 und 1
        match = re.search(r'([01]\.?\d*)', response_text.strip())
        if match:
            try:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))  # Clamp to [0, 1]
            except ValueError:
                pass
        # Fallback
        return 0.0

    def score(self, old: str, new: str) -> float:
        """Bewertet eine Textänderung via llama.cpp."""
        session = self._get_session()

        # OpenAI-kompatibles Chat-Format
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
                {"role": "user", "content": self._build_prompt(old, new)},
            ],
            "temperature": 0.0,  # Deterministisch
            "max_tokens": 10,    # Nur eine Zahl erwartet
        }

        try:
            response = session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

            # Antwort extrahieren
            content = data["choices"][0]["message"]["content"]
            return self._parse_score(content)

        except Exception as e:
            print(f"[Critic] Fehler bei LLM-Call: {e}")
            return 0.0  # Fallback: nicht kritisch


class TransformersCritic(CriticModel):
    """
    Criticality-Bewertung via HuggingFace transformers.

    TODO: Implementieren wenn CUDA verfügbar.

    Parameters
    ----------
    model_name : str
        HuggingFace Model ID (default: "ibm-granite/granite-3.3-2b-instruct").

    device : str
        Device für Inference (default: "auto").
    """

    def __init__(
        self,
        model_name: str = "ibm-granite/granite-3.3-2b-instruct",
        device: str = "auto",
    ):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Lazy-Loading des Modells."""
        if self._model is None:
            raise NotImplementedError(
                "TransformersCritic ist noch nicht implementiert. "
                "Nutze LlamaCppCritic für jetzt."
            )
            # TODO: Implementieren
            # from transformers import AutoModelForCausalLM, AutoTokenizer
            # self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # self._model = AutoModelForCausalLM.from_pretrained(
            #     self.model_name,
            #     device_map=self.device,
            # )

    def score(self, old: str, new: str) -> float:
        """Bewertet eine Textänderung via transformers."""
        self._load_model()
        # TODO: Implementieren
        return 0.0


# Convenience-Funktion
def get_critic(
    backend: str = "llama.cpp",
    url: Optional[str] = None,
    model: Optional[str] = None,
) -> CriticModel:
    """
    Factory-Funktion für Critic-Backends.

    Parameters
    ----------
    backend : str
        "llama.cpp" oder "transformers".

    url : str, optional
        URL für llama.cpp Server.

    model : str, optional
        Model-Name für transformers.

    Returns
    -------
    CriticModel
        Konfiguriertes Critic-Backend.

    Examples
    --------
    >>> critic = get_critic("llama.cpp", url="http://localhost:8102")
    >>> critic = get_critic("transformers", model="ibm-granite/granite-3.3-2b-instruct")
    """
    if backend == "llama.cpp":
        return LlamaCppCritic(base_url=url or "http://localhost:8102")
    elif backend == "transformers":
        return TransformersCritic(model_name=model or "ibm-granite/granite-3.3-2b-instruct")
    else:
        raise ValueError(f"Unbekanntes Backend: {backend}")


if __name__ == "__main__":
    # Quick Test
    print("=== Critic Test ===\n")

    critic = LlamaCppCritic()

    test_cases = [
        ("Die Bank teilt Zinsen mit.", "Die Bank teilt Zinsen nicht mit."),
        ("Frist: sechs Wochen", "Frist: sechs Minuten"),
        ("Opel Astra 2.2 L", "Opel Astra 1.4 L"),
        ("Labrador", "Golden Retriever"),
        ("Zahlungsfähigkeit", "Zahlungsunfähigkeit"),
    ]

    for old, new in test_cases:
        score = critic.score(old, new)
        print(f"'{old}' → '{new}'")
        print(f"  Score: {score:.1f}\n")
