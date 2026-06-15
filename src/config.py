"""
Modul konfiguracyjny.

Odpowiada za jedna prosta rzecz: wczytanie ustawien i kluczy API w jednym
miejscu, aby reszta programu nie musiala sie tym martwic.

Klucze sa wczytywane ze zmiennych srodowiskowych. Jesli w katalogu projektu
istnieje plik ".env", to jego zawartosc tez zostanie wczytana (dzieki bibliotece
python-dotenv).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

try:
    # Jesli istnieje plik .env, wczytaj z niego zmienne do srodowiska.
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    # Brak biblioteki python-dotenv nie jest bledem krytycznym -
    # mozna podac klucze bezposrednio przez zmienne srodowiskowe.
    pass


@dataclass
class Settings:
    """Zbior wszystkich ustawien aplikacji w jednym obiekcie."""

    # --- Model jezykowy (LLM) ---
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    # Pusty base_url oznacza oficjalny serwer OpenAI.
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "") or ""

    # --- Wyszukiwanie w internecie ---
    serpapi_api_key: str = os.getenv("SERPAPI_API_KEY", "")

    # --- Model CLIP (rozpoznawanie obrazow) ---
    clip_model_name: str = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")

    # --- Model embeddingow (do bazy wektorowej / RAG) ---
    embedding_model_name: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # Ile artykulow pobrac z wyszukiwarki dla rozpoznanej rosliny.
    search_results_count: int = int(os.getenv("SEARCH_RESULTS_COUNT", "8"))

    # Parametry dzielenia tekstu na fragmenty (chunking) w RAG.
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))

    # Ile najlepiej pasujacych fragmentow pobrac przy zapytaniu (retrieval).
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "4"))

    def has_openai(self) -> bool:
        """Czy mamy klucz do modelu jezykowego?"""
        return bool(self.openai_api_key)

    def has_serpapi(self) -> bool:
        """Czy mamy klucz do wyszukiwarki internetowej?"""
        return bool(self.serpapi_api_key)


# Jeden wspolny obiekt ustawien, importowany przez pozostale moduly.
settings = Settings()
