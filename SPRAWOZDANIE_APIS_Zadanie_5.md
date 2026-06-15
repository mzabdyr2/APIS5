# Sprawozdanie - Zadanie 5: Agent AI

## 0. Opis wykonanych krokow i uzasadnienie decyzji

1. **Przygotowanie srodowiska**: utworzono strukture `src/`, `reports/`, `.env`, `requirements.txt`.
   - Decyzja: podzial na moduly upraszcza utrzymanie i testy.
2. **Implementacja NewsAPI**: dodano narzedzie `search_news(topic, page_size)`.
   - Decyzja: zwrot danych w JSON ulatwia przekazywanie ich miedzy narzedziami.
3. **Implementacja podsumowania**: dodano `summarize_news(news_json, topic)`.
   - Decyzja: oddzielenie pobierania danych od syntezy tekstu daje czytelny pipeline.
4. **Budowa agenta**: polaczono model i narzedzia w `agent_runner.py`.
   - Decyzja: agent sam dobiera narzedzia zamiast hard-coded workflow.
5. **Eksport do PDF (ocena 4.0)**: dodano `save_markdown_as_pdf(markdown_text, filename)`.
   - Decyzja: markdown -> html -> pdf jest prosty i stabilny.
6. **Ocena istotnosci (ocena 4.5)**: dodano `assess_importance(summary, domain)`.
   - Decyzja: klasy `low/medium/high` sa czytelne i zgodne z wymaganiami.
7. **Testy**: wykonano testy narzedzi osobno i test end-to-end.
   - Decyzja: testowanie etapowe przyspiesza diagnoze bledow.

## 1. Architektura Agenta AI

Przeplyw danych:
1. Uzytkownik podaje temat.
2. Agent wywoluje `search_news`.
3. Wynik trafia do `summarize_news`.
4. Agent wywoluje `assess_importance`.
5. Agent zapisuje raport przez `save_markdown_as_pdf`.
6. Uzytkownik dostaje podsumowanie i plik PDF.

Jesli narzedzie zwroci blad:
- blad jest przechwytywany i zwracany jako kontrolowany komunikat,
- agent moze ponowic probe albo zmienic strategie.

Agent moze uzywac narzedzi wielokrotnie:
- tak, np. moze wywolac ponownie `search_news`, gdy dane sa zbyt ubogie.

Przyklad self-correction:
- pierwsza odpowiedz narzedzia ma za malo danych,
- agent ponawia pobranie z innym zakresem i dopiero tworzy raport.

## 2. Model jezykowy

Uzyto lokalnego modelu **llama3.2** przez Ollama.

Dlaczego:
- brak zaleznosci od zewnetrznego API modelu,
- prostsze testy lokalne,
- wystarczajaca jakosc do podsumowan i klasyfikacji.

## 3. Prompt systemowy

Prompt definiuje:
- role agenta,
- dostepne narzedzia,
- cel i oczekiwany wynik,
- ograniczenia: bez halucynacji, tylko dane z narzedzi.

Techniki promptowania:
- jasna rola i cel,
- instrukcje warunkowe (np. zapis do PDF),
- ograniczenie formatu odpowiedzi (low/medium/high),
- nacisk na kompletne i wiarygodne dane.

Self-correction:
- agent ma poprawic strategie, gdy dane sa niepelne lub bledne.

Definicja celu:
- cel wynika z polecenia uzytkownika, np. "Przygotuj raport i zapisz do PDF".

## 4. Funkcje (narzedzia)

- `search_news(topic: str, page_size: int = 5) -> str` - pobiera artykuly z NewsAPI.
- `summarize_news(news_json: str, topic: str) -> str` - tworzy podsumowanie.
- `assess_importance(summary: str, domain: str) -> str` - ocenia istotnosc (`low/medium/high`).
- `save_markdown_as_pdf(markdown_text: str, filename: str) -> str` - zapisuje raport PDF.

Wszystkie narzedzia maja typowanie parametrow i docstring:
- typy wspieraja walidacje argumentow,
- docstring opisuje cel narzedzia i pomaga agentowi dobrac je poprawnie.

## 5. Podejmowanie decyzji przez agenta

Agent wybiera narzedzie na podstawie:
- celu uzytkownika,
- opisu narzedzi,
- wynikow poprzednich krokow.

Przyklad:
- brak danych -> `search_news`,
- sa dane -> `summarize_news`,
- potrzebny raport -> `assess_importance` i `save_markdown_as_pdf`.

## 6. Przyklady wynikow i ocena jakosci

Przykladowe polecenie:
"Przygotuj raport o AI w medycynie i zapisz do PDF."

Wynik:
- podsumowanie tematu,
- etykieta istotnosci,
- plik PDF w katalogu `reports/`.

Ocena jakosci:
- plusy: poprawny przeplyw end-to-end, automatyzacja raportu, modularnosc.
- ograniczenia: jakosc zalezy od danych z NewsAPI.
