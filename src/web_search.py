"""
Wyszukiwanie informacji o pielegnacji rosliny w internecie.

JAK TO DZIALA:
1. Tworzymy zapytanie do Google, np. "how to take care of monstera plant".
2. Uzywamy SerpAPI (platnej uslugi, ktora zwraca wyniki Google w formie JSON).
3. Z wynikow bierzemy tytuly, linki i krotkie opisy (snippety).
4. Dodatkowo probujemy wejsc na kilka stron i pobrac wiecej tekstu artykulu,
   aby miec bogatszy kontekst dla modelu (RAG).

Jesli nie ma klucza SerpAPI, modul zwraca pusta liste - reszta aplikacji
nadal dziala (po prostu bez swiezych danych z internetu).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import requests

from .config import settings


@dataclass
class SearchDocument:
    """Pojedynczy wynik wyszukiwania zamieniony na dokument tekstowy."""

    title: str
    url: str
    text: str
    source: str = "web"


def _serpapi_search(query: str, num_results: int) -> list[dict]:
    """Wysyla zapytanie do SerpAPI i zwraca liste wynikow organicznych."""
    try:
        from serpapi import GoogleSearch
    except Exception:
        # Biblioteka 'google-search-results' udostepnia modul 'serpapi'.
        return []

    params = {
        "engine": "google",
        "q": query,
        "num": num_results,
        "api_key": settings.serpapi_api_key,
        "hl": "en",  # jezyk wynikow - angielski daje wiecej tresci o roslinach
    }
    try:
        search = GoogleSearch(params)
        data = search.get_dict()
    except Exception as exc:  # blad sieci / API
        print(f"[web_search] Blad SerpAPI: {exc}")
        return []

    return data.get("organic_results", []) or []


def _fetch_article_text(url: str, max_chars: int = 3000) -> str:
    """
    Probuje pobrac i oczyscic tekst artykulu spod podanego adresu.

    Uzywamy BeautifulSoup, aby wyciagnac sam tekst z akapitow <p>,
    pomijajac menu, reklamy itp. Jesli cos pojdzie nie tak (np. strona
    blokuje boty), po prostu zwracamy pusty tekst.
    """
    try:
        from bs4 import BeautifulSoup

        headers = {"User-Agent": "Mozilla/5.0 (compatible; PlantCareBot/1.0)"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        text = "\n".join(p for p in paragraphs if len(p) > 40)
        return text[:max_chars]
    except Exception:
        return ""


def search_plant_care(
    plant_name: str,
    num_results: int | None = None,
    fetch_full_text: bool = True,
) -> list[SearchDocument]:
    """
    Glowna funkcja modulu: szuka informacji o pielegnacji danej rosliny.

    Argumenty:
        plant_name: nazwa rosliny (np. "monstera").
        num_results: ile wynikow pobrac (domyslnie z konfiguracji).
        fetch_full_text: czy probowac pobrac pelny tekst artykulow.

    Zwraca:
        Liste dokumentow (SearchDocument) gotowych do wlozenia do bazy RAG.
    """
    if not settings.has_serpapi():
        print("[web_search] Brak klucza SERPAPI_API_KEY - pomijam wyszukiwanie.")
        return []

    num_results = num_results or settings.search_results_count
    query = f"how to take care of {plant_name} plant watering light soil"
    results = _serpapi_search(query, num_results)

    documents: list[SearchDocument] = []
    for item in results:
        title = item.get("title", "")
        url = item.get("link", "")
        snippet = item.get("snippet", "")

        # Bazowo uzywamy krotkiego opisu (snippetu) z wynikow Google.
        text = snippet

        # Jesli mozemy, dokladamy pelny tekst artykulu dla lepszego kontekstu.
        if fetch_full_text and url:
            full = _fetch_article_text(url)
            if full:
                text = f"{snippet}\n\n{full}"

        if text.strip():
            documents.append(
                SearchDocument(title=title, url=url, text=text.strip())
            )

    print(f"[web_search] Pobrano {len(documents)} dokumentow dla '{plant_name}'.")
    return documents
