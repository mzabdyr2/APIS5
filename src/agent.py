"""
Agent AI - "mozg" calego systemu.

CZYM ROZNI SIE AGENT OD ZWYKLEGO PIPELINE'U?
W sztywnym pipeline kolejnosc krokow jest ustalona z gory: najpierw CLIP, potem
wyszukiwanie, potem odpowiedz. Agent dziala inaczej - to MODEL JEZYKOWY sam
decyduje, co zrobic w danym momencie:
- Czy uzytkownik wgral zdjecie? Moze warto rozpoznac rosline.
- Czy mamy juz informacje o tej roslinie? Jesli nie - poszukaj w internecie.
- Czy uzytkownik pyta o szczegol? Pobierz pasujacy fragment z bazy (RAG).

Agent ma do dyspozycji 3 NARZEDZIA (tools):
1. identify_plant         -> rozpoznaje rosline ze zdjecia (CLIP)
2. search_plant_care_info -> szuka informacji w internecie i buduje baze RAG
3. retrieve_care_context  -> pobiera pasujace fragmenty z bazy RAG

Mechanizm "tool calling" OpenAI dziala tak:
- Opisujemy narzedzia modelowi (nazwa, opis, parametry).
- Model, zamiast od razu odpowiadac, moze poprosic: "wywolaj narzedzie X".
- My wykonujemy narzedzie i oddajemy modelowi wynik.
- Model analizuje wynik i albo prosi o kolejne narzedzie, albo odpowiada
  uzytkownikowi. Ta petla powtarza sie, az agent uzna, ze ma komplet informacji.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from PIL import Image

from .clip_classifier import get_classifier
from .llm import get_llm
from .rag import RAGPipeline
from .web_search import search_plant_care

# ---------------------------------------------------------------------------
# Prompt systemowy - "instrukcja obslugi" dla modelu. Definiuje role agenta,
# jego sposob myslenia i zasady korzystania z narzedzi.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
Jestes pomocnym asystentem AI specjalizujacym sie w pielegnacji roslin domowych.
Rozmawiasz z uzytkownikiem po polsku, w sposob przyjazny i konkretny.

Masz do dyspozycji narzedzia. Korzystaj z nich madrze, wedlug zasad:

1. Jesli uzytkownik wgral zdjecie, a roslina nie zostala jeszcze rozpoznana -
   uzyj narzedzia `identify_plant`, aby ustalic gatunek.
2. Po rozpoznaniu (lub gdy uzytkownik sam poda nazwe rosliny), jesli nie masz
   jeszcze zebranych informacji o tej roslinie - uzyj `search_plant_care_info`,
   aby pobrac aktualne dane z internetu.
3. Gdy odpowiadasz na pytanie o pielegnacje - uzyj `retrieve_care_context`, aby
   znalezc najtrafniejsze fragmenty zebranych informacji i oprzec na nich odpowiedz.
4. Odpowiadaj na podstawie zebranego kontekstu. Jesli kontekst czegos nie
   zawiera, mozesz uzupelnic ogolna wiedza, ale zaznacz to uczciwie.
5. Nie zmyslaj zrodel. Badz praktyczny: podawaj konkrety (podlewanie, swiatlo,
   nawozenie, temperatura, czeste problemy).

Nie musisz uzywac wszystkich narzedzi przy kazdej wiadomosci - wybieraj tylko te,
ktore sa potrzebne do dobrej odpowiedzi.
"""

# Definicje narzedzi w formacie wymaganym przez API OpenAI (JSON Schema).
TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "identify_plant",
            "description": (
                "Rozpoznaje gatunek rosliny na podstawie zdjecia wgranego przez "
                "uzytkownika (model CLIP). Uzyj, gdy dostepne jest zdjecie, a "
                "roslina nie zostala jeszcze zidentyfikowana."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_plant_care_info",
            "description": (
                "Wyszukuje w internecie informacje o pielegnacji podanej rosliny "
                "i buduje z nich baze wiedzy (RAG). Uzyj raz dla danej rosliny, "
                "zanim zaczniesz odpowiadac na szczegolowe pytania."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "plant_name": {
                        "type": "string",
                        "description": "Angielska nazwa rosliny, np. 'monstera'.",
                    }
                },
                "required": ["plant_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_care_context",
            "description": (
                "Pobiera z bazy wiedzy (RAG) fragmenty najlepiej pasujace do "
                "pytania. Uzyj przed udzieleniem odpowiedzi o pielegnacji."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "O co pyta uzytkownik, np. 'jak czesto podlewac'.",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


@dataclass
class AgentState:
    """Pamiec agenta - przechowuje to, co dzieje sie w trakcie rozmowy."""

    image: Image.Image | None = None       # ostatnio wgrane zdjecie
    identified_plant: str | None = None     # rozpoznana roslina (klucz, np. "monstera")
    identified_label: str | None = None     # ladna nazwa rozpoznanej rosliny
    rag: RAGPipeline = field(default_factory=RAGPipeline)
    history: list[dict] = field(default_factory=list)  # historia rozmowy
    last_tools_used: list[str] = field(default_factory=list)  # log narzedzi (do GUI)


class PlantCareAgent:
    """Agent laczacy model jezykowy z narzedziami CLIP / wyszukiwarka / RAG."""

    def __init__(self) -> None:
        self.llm = get_llm()
        self.classifier = get_classifier()
        self.state = AgentState()

    # --- Ustawianie wgranego zdjecia ---
    def set_image(self, image: Image.Image | None) -> None:
        """Zapisuje wgrane zdjecie i resetuje wczesniejsze rozpoznanie."""
        self.state.image = image
        self.state.identified_plant = None
        self.state.identified_label = None

    # --- Implementacje poszczegolnych narzedzi ---
    def _tool_identify_plant(self) -> str:
        """Narzedzie 1: rozpoznanie rosliny ze zdjecia."""
        if self.state.image is None:
            return json.dumps(
                {"error": "Brak zdjecia. Popros uzytkownika o wgranie fotografii rosliny."},
                ensure_ascii=False,
            )
        predictions = self.classifier.classify(self.state.image, top_k=3)
        if predictions:
            best = predictions[0]
            self.state.identified_plant = best.key
            self.state.identified_label = best.label
        return json.dumps(
            {
                "predictions": [
                    {"name": p.key, "label": p.label, "confidence": round(p.confidence, 3)}
                    for p in predictions
                ]
            },
            ensure_ascii=False,
        )

    def _tool_search_plant_care_info(self, plant_name: str) -> str:
        """Narzedzie 2: wyszukanie informacji w internecie + budowa bazy RAG."""
        documents = search_plant_care(plant_name)
        if not documents:
            return json.dumps(
                {
                    "status": "brak_wynikow",
                    "message": (
                        "Nie udalo sie pobrac danych z internetu (brak klucza "
                        "SerpAPI lub brak wynikow). Odpowiedz na podstawie wiedzy ogolnej."
                    ),
                },
                ensure_ascii=False,
            )
        n_chunks = self.state.rag.build_index(documents, plant_name=plant_name)
        return json.dumps(
            {
                "status": "ok",
                "documents_found": len(documents),
                "chunks_indexed": n_chunks,
                "sources": [d.url for d in documents[:5]],
            },
            ensure_ascii=False,
        )

    def _tool_retrieve_care_context(self, query: str) -> str:
        """Narzedzie 3: pobranie pasujacych fragmentow z bazy RAG."""
        if not self.state.rag.has_data():
            return json.dumps(
                {"status": "pusta_baza", "context": ""}, ensure_ascii=False
            )
        context = self.state.rag.build_context(query)
        return json.dumps({"status": "ok", "context": context}, ensure_ascii=False)

    def _dispatch_tool(self, name: str, arguments: dict) -> str:
        """Uruchamia narzedzie o podanej nazwie z podanymi argumentami."""
        self.state.last_tools_used.append(name)
        if name == "identify_plant":
            return self._tool_identify_plant()
        if name == "search_plant_care_info":
            return self._tool_search_plant_care_info(arguments.get("plant_name", ""))
        if name == "retrieve_care_context":
            return self._tool_retrieve_care_context(arguments.get("query", ""))
        return json.dumps({"error": f"Nieznane narzedzie: {name}"}, ensure_ascii=False)

    # --- Glowna petla agenta ---
    def chat(self, user_message: str, max_iterations: int = 6) -> str:
        """
        Przetwarza wiadomosc uzytkownika, pozwalajac agentowi uzywac narzedzi.

        Argumenty:
            user_message: tekst od uzytkownika.
            max_iterations: zabezpieczenie przed nieskonczona petla narzedzi.

        Zwraca:
            Finalna odpowiedz tekstowa agenta.
        """
        self.state.last_tools_used = []

        # Podpowiadamy modelowi aktualny stan (czy jest zdjecie, co rozpoznano).
        context_hint = self._build_context_hint()

        # Budujemy liste wiadomosci: prompt systemowy + historia + nowa wiadomosc.
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        if context_hint:
            messages.append({"role": "system", "content": context_hint})
        messages.extend(self.state.history)
        messages.append({"role": "user", "content": user_message})

        final_answer = ""
        for _ in range(max_iterations):
            assistant_msg = self.llm.chat_with_tools(messages, TOOLS_SPEC)

            tool_calls = getattr(assistant_msg, "tool_calls", None)
            if not tool_calls:
                # Model nie chce wiecej narzedzi - to jest finalna odpowiedz.
                final_answer = assistant_msg.content or ""
                break

            # Dopisujemy wiadomosc asystenta (z prosbami o narzedzia) do historii.
            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )

            # Wykonujemy kazde zamowione narzedzie i oddajemy modelowi wynik.
            for tool_call in tool_calls:
                name = tool_call.function.name
                try:
                    arguments = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                result = self._dispatch_tool(name, arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )
        else:
            # Wykorzystano limit iteracji - prosimy o podsumowanie bez narzedzi.
            final_answer = self.llm.chat(
                messages + [{"role": "user", "content": "Podsumuj odpowiedz na podstawie zebranych informacji."}]
            )

        # Zapisujemy ture rozmowy do historii (tylko czysty tekst, bez narzedzi).
        self.state.history.append({"role": "user", "content": user_message})
        self.state.history.append({"role": "assistant", "content": final_answer})
        return final_answer

    def _build_context_hint(self) -> str:
        """Tworzy krotka notatke o aktualnym stanie dla modelu."""
        parts = []
        if self.state.image is not None:
            parts.append("Uzytkownik wgral zdjecie rosliny.")
        else:
            parts.append("Brak wgranego zdjecia.")
        if self.state.identified_plant:
            parts.append(
                f"Rozpoznana roslina: {self.state.identified_label} "
                f"(klucz: {self.state.identified_plant})."
            )
        if self.state.rag.has_data():
            parts.append(
                f"Baza RAG zawiera dane o roslinie: {self.state.rag.indexed_plant}."
            )
        return "STAN SYSTEMU: " + " ".join(parts)

    def reset_conversation(self) -> None:
        """Czysci historie rozmowy (np. przy nowej sesji)."""
        self.state.history = []
