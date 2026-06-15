"""
Obsluga modelu jezykowego (LLM) przez API OpenAI.

Ten modul to cienka "warstwa posrednia" miedzy nasza aplikacja a API OpenAI.
Udostepnia dwie rzeczy:
1. chat()       -> zwykla rozmowa (lista wiadomosci -> odpowiedz tekstowa).
2. chat_with_tools() -> rozmowa, w ktorej model moze poprosic o uzycie narzedzi
                        (to jest serce agenta AI - patrz agent.py).

Wybor modelu: domyslnie "gpt-4o-mini" - jest tani, szybki i wystarczajaco
"madry" do rozmowy o pielegnacji roslin oraz do podejmowania decyzji przez agenta.
Model mozna zmienic w pliku .env (zmienna OPENAI_MODEL).
"""

from __future__ import annotations

from .config import settings


class LLMClient:
    """Opakowanie na klienta OpenAI z prostymi metodami."""

    def __init__(self) -> None:
        self.model = settings.openai_model
        self._client = None

    def _ensure_client(self):
        """Tworzy klienta OpenAI przy pierwszym uzyciu."""
        if self._client is not None:
            return self._client

        if not settings.has_openai():
            raise RuntimeError(
                "Brak klucza OPENAI_API_KEY. Ustaw go w pliku .env lub jako "
                "zmienna srodowiskowa, aby korzystac z modelu jezykowego."
            )

        from openai import OpenAI

        kwargs = {"api_key": settings.openai_api_key}
        # Pozwalamy uzyc serwera zgodnego z OpenAI (np. lokalnego), jesli podano.
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url

        self._client = OpenAI(**kwargs)
        return self._client

    def chat(self, messages: list[dict], temperature: float = 0.3) -> str:
        """
        Zwykla rozmowa: podajemy liste wiadomosci, dostajemy odpowiedz tekstowa.

        Format wiadomosci (standard OpenAI):
            {"role": "system"|"user"|"assistant", "content": "..."}
        """
        client = self._ensure_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        temperature: float = 0.2,
    ):
        """
        Rozmowa z mozliwoscia wywolania narzedzi (tool calling).

        Zwraca caly obiekt "message" od modelu. Moze on zawierac:
        - zwykla odpowiedz tekstowa (content), LUB
        - prosbe o wywolanie jednego lub wielu narzedzi (tool_calls).

        Decyzje, ktore narzedzie wywolac, podejmuje SAM model - to wlasnie
        czyni z naszego systemu agenta AI.
        """
        client = self._ensure_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto",  # model sam decyduje, czy i czego uzyc
            temperature=temperature,
        )
        return response.choices[0].message


# Wspoldzielony klient dla calej aplikacji.
_llm: LLMClient | None = None


def get_llm() -> LLMClient:
    """Zwraca wspoldzielony obiekt klienta LLM."""
    global _llm
    if _llm is None:
        _llm = LLMClient()
    return _llm
