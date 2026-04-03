#!/usr/bin/env python3
"""
Semantic Engine - LLM-basierte Criticality-Bewertung (V2).

Diese Version ist spezialisiert auf den Vergleich ganzer Sätze.
Statt eines abstrakten Scores liefert sie eine strukturierte Entscheidung:
- Widerspruch (Ja/Nein)
- Grund (Kurze Erklärung)

Benötigt einen OpenAI-kompatiblen Server (z.B. llama.cpp).
"""

from abc import ABC, abstractmethod
from typing import Optional, TypedDict
import json
import re


# System-Prompt für Satz-Vergleich (V3 - Context Aware)
CRITIC_SYSTEM_PROMPT_V2 = """Du bist ein forensischer Text-Analyst für Verträge.
Deine Aufgabe: Prüfe zwei Sätze auf inhaltliche WIDERSPRÜCHE oder KRITISCHE ÄNDERUNGEN.

Du erhältst den Text und maschinelle Analyse-Daten (Signale).
Nutze die Signale als Hinweis, aber vertraue im Zweifel deiner semantischen Analyse des Textes.

Eingabe-Format:
1. Satz A (Erwartung)
2. Satz B (Realität)
3. Analyse-Daten (Similarity Score, Fehlende Keywords, Struktur-Warnungen)

Entscheidungsregeln:
- WIDERSPRUCH (True): Wenn sich Fakten (Zahlen, Daten, Namen) ändern oder die Aussage logisch kippt (Verbot -> Erlaubnis).
- KEIN WIDERSPRUCH (False): Bei Paraphrasen, Zusammenfassungen oder wenn der Inhalt kompatibel ist.
- Hoher Score (0.9+) + Fehlendes Keyword -> Verdächtig (prüfe genau!).
- Niedriger Score (<0.6) -> Wahrscheinlich kein Match, aber prüfe ob sie sich direkt widersprechen.

Antworte IMMER im JSON-Format:
{
  "contradiction": true/false,
  "reason": "Präzise Begründung (nutze Analyse-Daten falls relevant)"
}
"""

class CriticDecision(TypedDict):
    contradiction: bool
    reason: str


class CriticModelV2(ABC):
    """Interface für V2 Critic."""
    
    @abstractmethod
    def evaluate(self, expected: str, actual: str, context: dict = None) -> CriticDecision:
        """Vergleicht zwei Sätze mit optionalem Kontext."""
        pass


class LlamaCppCriticV2(CriticModelV2):
    """
    LLM-Critic V2 via llama.cpp (JSON Mode empfohlen).
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
        if self._session is None:
            import requests
            self._session = requests.Session()
        return self._session

    def evaluate(self, expected: str, actual: str, context: dict = None) -> CriticDecision:
        session = self._get_session()
        
        # Prompt Building mit Kontext
        user_prompt = f'Satz A: "{expected}"\nSatz B: "{actual}"\n\nANALYSE-DATEN:'
        
        if context:
            if "score" in context:
                user_prompt += f'\n- Semantic Similarity: {context["score"]:.2f}'
            if "missing_keywords" in context and context["missing_keywords"]:
                user_prompt += f'\n- Fehlende wichtige Wörter: {", ".join(context["missing_keywords"])}'
            if "missing_entities" in context and context["missing_entities"]:
                user_prompt += f'\n- Fehlende Entitäten (Namen/Zahlen): {", ".join(context["missing_entities"])}'
            if "hint" in context and context["hint"]:
                user_prompt += f'\n- Struktur-Warnung: {context["hint"]}'
        else:
            user_prompt += "\n(Keine Daten verfügbar)"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": CRITIC_SYSTEM_PROMPT_V2},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"}, # Force JSON
            "max_tokens": 128,
        }

        try:
            response = session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            
            # Robustes JSON Parsing
            try:
                # Versuche direktes Parsing
                decision = json.loads(content)
            except json.JSONDecodeError:
                # Fallback: Suche erstes JSON-Objekt im String
                match = re.search(r'\{.*\}', content, re.DOTALL)
                if match:
                    decision = json.loads(match.group(0))
                else:
                    return {"contradiction": False, "reason": "Parse Error"}

            return {
                "contradiction": decision.get("contradiction", False),
                "reason": decision.get("reason", "Unklar"),
            }

        except Exception as e:
            print(f"[CriticV2] Fehler: {e}")
            return {"contradiction": False, "reason": "System Error"}

if __name__ == "__main__":
    print("=== Critic V2 Test ===\n")
    critic = LlamaCppCriticV2()
    
    cases = [
        ("Das Mädchen bekommt ein Meerschweinchen.", "Dem Mädchen wird ein Häschen geschenkt."),
        ("Die Frist beträgt 2 Wochen.", "Die Frist beläuft sich auf 14 Tage."),
        ("Die Bank haftet nicht.", "Die Bank haftet uneingeschränkt."),
    ]
    
    for exp, act in cases:
        print(f"Exp: {exp}")
        print(f"Act: {act}")
        print(f"Result: {critic.evaluate(exp, act)}")
        print("-" * 20)