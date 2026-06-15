# Sprawozdanie z projektu zaliczeniowego

**Przedmiot:** Aktualne Problemy Informatyki Stosowanej (Geoinformatyka, II st.)
**Temat:** Implementacja kompleksowej aplikacji opartej o AI — Asystent Pielegnacji Roslin
**Zakres:** projekt zrealizowany na ocene **5.0** (pelna implementacja: CLIP + wyszukiwanie + RAG + agent AI + interfejs graficzny)

> **Uwaga dla prowadzacego.** Ten dokument jest zrodlowa wersja sprawozdania (Markdown).
> Do wyslania mailem nalezy go wyeksportowac do PDF (np. *Plik → Drukuj → Zapisz jako PDF*
> w edytorze obslugujacym Markdown, albo `pandoc sprawozdanie.md -o sprawozdanie.pdf`).

---

## Spis tresci

1. [Architektura i pipeline](#1-architektura-i-pipeline)
2. [CLIP — rozpoznawanie roslin](#2-clip--rozpoznawanie-roslin)
3. [Chatbot — model jezykowy](#3-chatbot--model-jezykowy)
4. [RAG — wyszukiwanie i kontekst](#4-rag--wyszukiwanie-i-kontekst)
5. [Agent AI](#5-agent-ai)
6. [Interfejs graficzny](#6-interfejs-graficzny)
7. [Wyniki i analiza](#7-wyniki-i-analiza)
8. [Podsumowanie i ograniczenia](#8-podsumowanie-i-ograniczenia)

---

## 1. Architektura i pipeline

### 1.1. Idea ogolna

Aplikacja umozliwia uzytkownikowi przeslanie zdjecia rosliny domowej i prowadzenie
rozmowy na temat jej pielegnacji. System rozpoznaje gatunek, samodzielnie wyszukuje
w internecie aktualne informacje, a nastepnie odpowiada na pytania, opierajac sie na
zebranych danych. Calosc dziala w formie **agenta AI**, ktory dynamicznie decyduje,
ktore narzedzie uruchomic w danym momencie.

### 1.2. Komponenty systemu

System sklada sie z szesciu wspolpracujacych modulow:

| Komponent | Plik | Odpowiedzialnosc |
|-----------|------|------------------|
| **Konfiguracja** | `src/config.py` | wczytanie kluczy API i parametrow w jednym miejscu |
| **Klasyfikator CLIP** | `src/clip_classifier.py` | rozpoznawanie rosliny ze zdjecia (zero-shot) |
| **Wyszukiwarka** | `src/web_search.py` | pobieranie artykulow z Google przez SerpAPI |
| **Modul RAG** | `src/rag.py` | chunking, embeddingi, baza wektorowa FAISS, retrieval |
| **Klient LLM** | `src/llm.py` | komunikacja z modelem jezykowym (OpenAI), tool calling |
| **Agent AI** | `src/agent.py` | orkiestracja narzedzi, petla decyzyjna, pamiec rozmowy |
| **Interfejs (GUI)** | `app.py` | warstwa prezentacji w Streamlit |

### 1.3. Przeplyw danych (pipeline)

W przeciwienstwie do sztywnego pipeline'u, kolejnoscia krokow steruje agent. Typowy
przebieg wyglada jednak nastepujaco:

```
[Uzytkownik]
   │  wgrywa zdjecie + zadaje pytanie
   ▼
[Agent AI]  ── decyduje ──►  [CLIP]            → nazwa rosliny (np. "monstera")
   │                          ▲
   │  ── decyduje ──►  [SerpAPI]               → artykuly z internetu
   │                          │
   │                          ▼
   │                    [RAG: chunking → embeddingi → FAISS]
   │  ── decyduje ──►  [RAG retrieval]         → najtrafniejsze fragmenty
   ▼
[Model jezykowy]  → odpowiedz oparta na kontekscie
   ▼
[Interfejs Streamlit]  → wyswietlenie odpowiedzi
```

### 1.4. Najwazniejsze decyzje projektowe

- **Modularnosc** — kazda funkcja (CLIP, wyszukiwanie, RAG, LLM) jest w osobnym pliku.
  Dzieki temu komponenty mozna testowac i wymieniac niezaleznie (np. zmienic model
  embeddingow bez ruszania reszty).
- **Lazy loading modeli** — ciezkie modele (CLIP, embeddingi) laduja sie dopiero przy
  pierwszym uzyciu, co przyspiesza start aplikacji.
- **Odpornosc na brak kluczy** — przy braku klucza SerpAPI system nie wywala sie, tylko
  odpowiada z wiedzy ogolnej; tryb zapasowy bazy wektorowej (numpy) dziala, gdy nie ma FAISS.
- **Tool calling zamiast sztywnego kodu** — to model decyduje o kolejnosci operacji,
  co realizuje wymaganie dotyczace agenta.

---

## 2. CLIP — rozpoznawanie roslin

### 2.1. Integracja z aplikacja

Wykorzystano model **`openai/clip-vit-base-patch32`** z biblioteki `transformers`
(HuggingFace). Model laduje sie lokalnie i nie wymaga kluczy API. Logika znajduje sie
w klasie `CLIPPlantClassifier` (`src/clip_classifier.py`), ktora udostepnia jedna metode
`classify(image, top_k)`.

### 2.2. Architektura rozwiazania

CLIP odwzorowuje obraz oraz tekst do **wspolnej przestrzeni wektorowej** (embeddingow).
Dzieki temu mozna policzyc podobienstwo obrazu do dowolnego opisu tekstowego. Proces:

1. Przygotowujemy zbior **promptow** dla wszystkich rozpoznawanych roslin.
2. Procesor CLIP koduje zdjecie oraz wszystkie prompty.
3. Model zwraca `logits_per_image` — surowe podobienstwa obrazu do kazdego opisu.
4. Funkcja **softmax** zamienia je na prawdopodobienstwa sumujace sie do 1.
5. Wybieramy `top_k` opisow o najwyzszym prawdopodobienstwie.

### 2.3. Sposob formulowania promptow

Z badan nad CLIP wynika, ze model dziala lepiej, gdy opis ma forme zdania (tzw. *prompt
engineering*). Zamiast samego slowa `"monstera"` uzywamy szablonu:

```
"a photo of a {nazwa} houseplant"
```

Lista roslin (30 popularnych gatunkow domowych) jest zdefiniowana w `src/plant_labels.py`
wraz z polskimi i lacinskimi nazwami pokazywanymi uzytkownikowi.

### 2.4. Proces wyboru klasy

Klasa o najwyzszym prawdopodobienstwie z softmax jest przyjmowana jako rozpoznanie.
Dodatkowo zwracamy 3 najlepsze propozycje wraz z procentowa pewnoscia — uzytkownik
widzi nie tylko wynik, ale i alternatywy, co zwieksza zaufanie i pozwala skorygowac
ewentualny blad.

---

## 3. Chatbot — model jezykowy

### 3.1. Wybrany model i uzasadnienie

Wybrano **`gpt-4o-mini`** (OpenAI) jako domyslny model jezykowy. Uzasadnienie:

- **Tool calling** — model natywnie obsluguje wywolywanie narzedzi (funkcji), co jest
  kluczowe dla architektury agenta.
- **Koszt i szybkosc** — jest znacznie tanszy i szybszy od duzych modeli, a jego jakosc
  w pelni wystarcza do rozmowy o pielegnacji roslin i prostego planowania krokow.
- **Wielojezycznosc** — dobrze radzi sobie z jezykiem polskim.

Model jest konfigurowalny przez zmienna `OPENAI_MODEL`. Dzieki ustawieniu
`OPENAI_BASE_URL` mozna podlaczyc dowolny serwer zgodny z API OpenAI (np. lokalny).

### 3.2. Prompt systemowy

Prompt systemowy (`SYSTEM_PROMPT` w `src/agent.py`) definiuje role i zasady dzialania
asystenta. Najwazniejsze elementy:

- okreslenie roli: ekspert od pielegnacji roslin domowych, odpowiada po polsku,
- **zasady korzystania z narzedzi** (kiedy rozpoznac rosline, kiedy szukac w internecie,
  kiedy siegnac do RAG),
- nakaz opierania odpowiedzi na zebranym kontekscie i uczciwego oznaczania, gdy czegos
  brakuje (przeciwdzialanie konfabulacji),
- styl: praktyczny i konkretny (podlewanie, swiatlo, nawozenie, temperatura, problemy).

---

## 4. RAG — wyszukiwanie i kontekst

### 4.1. Wyszukiwanie i pobieranie informacji z internetu

Modul `src/web_search.py` korzysta z **SerpAPI** (funkcja `GoogleSearch`). Dla rozpoznanej
rosliny budowane jest zapytanie:

```
"how to take care of {roslina} plant watering light soil"
```

Pobieramy top wyniki organiczne (domyslnie 8). Z kazdego wyniku bierzemy tytul, link
i krotki opis (snippet). Dodatkowo, jesli to mozliwe, **pobieramy pelny tekst artykulu**
(biblioteka `requests` + `BeautifulSoup`), wyciagajac tresc z akapitow `<p>` i pomijajac
menu/reklamy. Daje to bogatszy material do bazy wiedzy.

### 4.2. Przetwarzanie danych i dostarczanie kontekstu

Surowe artykuly nie nadaja sie do bezposredniego wlozenia do modelu (sa za dlugie i
zawieraja szum). Dlatego:

1. **Chunking** — tekst dzielimy na fragmenty (~800 znakow) z nakladaniem sie (~150 znakow).
   Naklad (overlap) zapobiega gubieniu sensu na granicach fragmentow. Tniemy po slowach,
   aby nie rozcinac wyrazow.
2. **Embeddingi** — kazdy fragment zamieniamy na znormalizowany wektor.
3. **Indeksowanie** — wektory trafiaja do bazy wektorowej.
4. **Retrieval** — przy pytaniu uzytkownika kodujemy je do wektora i znajdujemy
   najbardziej podobne fragmenty.
5. **Budowa kontekstu** — wybrane fragmenty (z oznaczeniem zrodla) sklejamy w tekst, ktory
   trafia do modelu jako kontekst.

### 4.3. Mechanizm retrieval i dzialanie RAG

Retrieval opiera sie na **podobienstwie kosinusowym** miedzy wektorem pytania a wektorami
fragmentow. Poniewaz embeddingi sa normalizowane, iloczyn skalarny jest rownowazny
podobienstwu kosinusowemu, co pozwala uzyc szybkiego indeksu **FAISS `IndexFlatIP`**.
Pobieramy `top_k = 4` najtrafniejsze fragmenty. Ogolna idea RAG: model **nie zgaduje**
z pamieci, lecz odpowiada na podstawie konkretnych, swiezo pobranych tresci.

### 4.4. Wybor modelu embeddingow i uzasadnienie

Wybrano **`sentence-transformers/all-MiniLM-L6-v2`**. Uzasadnienie:

- **Lokalny i darmowy** — nie wymaga klucza API ani kosztow,
- **Lekki i szybki** — 384-wymiarowe wektory, dziala sprawnie na CPU,
- **Dobra jakosc** — to jeden z najpopularniejszych modeli do wyszukiwania semantycznego,
  sprawdzony w wielu zastosowaniach RAG.

### 4.5. Dodatkowe mechanizmy wspierajace

- **Chunking z overlapem** (opisany wyzej),
- **Vector store** (`VectorStore` w `src/rag.py`) oparty o FAISS, z automatycznym trybem
  zapasowym na numpy (gdy FAISS niedostepny),
- **Normalizacja wektorow** dla poprawnego liczenia podobienstwa kosinusowego,
- **Metadane** (tytul i URL zrodla) przechowywane razem z fragmentami, co umozliwia
  podawanie zrodel.

---

## 5. Agent AI

### 5.1. Wykorzystane narzedzia (tools)

Agent dysponuje trzema narzedziami, opisanymi modelowi w formacie JSON Schema
(`TOOLS_SPEC` w `src/agent.py`):

| Narzedzie | Funkcja | Co robi |
|-----------|---------|---------|
| `identify_plant` | CLIP | rozpoznaje rosline z aktualnie wgranego zdjecia |
| `search_plant_care_info` | SerpAPI + RAG | wyszukuje artykuly i buduje baze wektorowa |
| `retrieve_care_context` | RAG | pobiera fragmenty pasujace do pytania |

### 5.2. Architektura agenta i podejmowanie decyzji

Agent jest zbudowany na mechanizmie **tool calling** modelu OpenAI. Dzialanie:

1. Do modelu trafia prompt systemowy, **notatka o stanie systemu** (czy jest zdjecie,
   co rozpoznano, czy baza RAG ma dane), historia rozmowy oraz nowa wiadomosc.
2. Model — zamiast od razu odpowiadac — moze poprosic o wywolanie narzedzia (lub kilku).
3. Aplikacja wykonuje zadane narzedzia i oddaje modelowi ich wyniki.
4. Model analizuje wyniki i albo prosi o kolejne narzedzie, albo formuluje odpowiedz.
5. Petla powtarza sie (z limitem iteracji jako zabezpieczeniem), az agent ma komplet
   informacji.

To **model**, a nie sztywny kod, decyduje o kolejnosci: dla wgranego, nierozpoznanego
zdjecia zwykle najpierw wywola `identify_plant`, potem `search_plant_care_info`, a przy
odpowiadaniu `retrieve_care_context`. Jesli uzytkownik poda nazwe rosliny tekstem, agent
moze pominac CLIP.

### 5.3. Zarzadzanie przeplywem informacji

- **Stan agenta** (`AgentState`) przechowuje wgrane zdjecie, rozpoznana rosline, baze RAG
  oraz historie rozmowy — dzieki temu kolejne pytania korzystaja z wczesniejszego kontekstu.
- **Log narzedzi** (`last_tools_used`) rejestruje, ktore narzedzia uzyto w danej turze —
  interfejs pokazuje to uzytkownikowi (transparentnosc dzialania agenta).
- **Notatka o stanie** wstrzykiwana do promptu pomaga modelowi unikac zbednych wywolan
  (np. ponownego wyszukiwania, gdy baza juz zawiera dane).

---

## 6. Interfejs graficzny

### 6.1. Opis interfejsu

Interfejs (`app.py`) zbudowano w **Streamlit** i zaprojektowano tak, by przypominal
nowoczesne narzedzia LLM (ChatGPT/Gemini): czat w glownym oknie + panel boczny z opcjami.

### 6.2. Technologie i komponenty (widgety)

- **`st.set_page_config`** — tytul, ikona, uklad strony,
- **`st.sidebar`** — panel boczny z:
  - `st.file_uploader` — wgrywanie zdjecia (jpg/png/webp),
  - `st.image` — podglad wgranego zdjecia,
  - `st.button("Rozpoznaj rosline")` — reczne uruchomienie CLIP,
  - sekcja **statusu** (czy sa klucze API),
  - przycisk czyszczenia rozmowy,
- **`st.chat_message`** — banki wiadomosci uzytkownika i asystenta,
- **`st.chat_input`** — pole tekstowe (textbar) z przyciskiem wysylania,
- **`st.spinner`** — informacja o trwajacym przetwarzaniu,
- **`st.session_state`** — pamiec agenta i historii miedzy odswiezeniami strony.

### 6.3. Funkcjonalnosc i interakcja

Uzytkownik wgrywa zdjecie w panelu bocznym (moze tez od razu kliknac "Rozpoznaj rosline"),
a nastepnie prowadzi swobodna rozmowe w glownym oknie. Pod kazda odpowiedzia agent pokazuje,
**jakich narzedzi uzyl** (np. *identify_plant, search_plant_care_info, retrieve_care_context*),
co czyni dzialanie systemu przejrzystym.

---

## 7. Wyniki i analiza

> **Jak odtworzyc wyniki.** Wszystkie kroki mozna uruchomic w `notebooks/demo.ipynb`
> lub w aplikacji `streamlit run app.py`. Ponizsze przyklady pokazuja oczekiwany sposob
> dzialania systemu; konkretne wartosci procentowe zaleza od jakosci zdjecia.

### 7.1. Przyklad dzialania na przykladowych danych

Przyklad rozpoznawania (CLIP), trzy najlepsze propozycje dla zdjecia monstery:

```
Monstera (Monstera deliciosa)            87.4%
Filodendron (Philodendron)                6.1%
Epipremnum / scindapsus (Epipremnum...)   3.2%
```

Przyklad retrieval (RAG) dla pytania *"jak czesto podlewac"*: system zwraca fragmenty
artykulow mowiace o podlewaniu, gdy wyschnie gorne 2-3 cm podloza (zwykle raz na tydzien),
i na tej podstawie buduje odpowiedz.

### 7.2. Wyniki dla kilku roslin (oczekiwane zachowanie)

| Roslina | Typowa trafnosc CLIP | Uwagi |
|---------|----------------------|-------|
| Monstera | wysoka | charakterystyczne dziurawe liscie |
| Sansewieria | wysoka | wyrazna, pionowa forma |
| Kaktus / sukulent | srednia | latwo mylone miedzy soba |
| Storczyk (kwitnacy) | wysoka | wyrazne kwiaty |
| Paproc | srednia | rozne gatunki wygladaja podobnie |

### 7.3. Analiza jakosci dzialania

**Mocne strony:**
- rozpoznawanie zero-shot bez koniecznosci trenowania na wlasnym zbiorze,
- odpowiedzi oparte na aktualnych danych z internetu (RAG ogranicza konfabulacje),
- elastyczny agent — radzi sobie zarowno ze zdjeciem, jak i z sama nazwa rosliny,
- transparentnosc (widoczne uzyte narzedzia, podawane zrodla).

**Ograniczenia:**
- CLIP rozpoznaje tylko gatunki z przygotowanej listy; rosliny spoza listy zostana
  przypisane do najblizszej znanej klasy,
- jakosc rozpoznania spada przy slabym oswietleniu, rozmytych lub mocno przycietych zdjeciach
  oraz dla gatunkow bardzo do siebie podobnych,
- wyszukiwanie zalezy od jakosci wynikow Google i dostepnosci klucza SerpAPI,
- pobieranie pelnej tresci stron moze byc blokowane przez niektore serwisy.

**Mozliwe rozszerzenia:**
- wieksza i bardziej szczegolowa lista gatunkow lub dedykowany model klasyfikacji,
- trwala baza wektorowa (np. Chroma) z pamiecia miedzy sesjami,
- cache wynikow wyszukiwania, aby nie odpytywac SerpAPI wielokrotnie,
- ocena trafnosci na oznaczonym zbiorze testowym (metryki: top-1 / top-3 accuracy).

---

## 8. Podsumowanie i ograniczenia

Zrealizowano pelna aplikacje (zakres na ocene 5.0) laczaca: rozpoznawanie roslin modelem
**CLIP**, wyszukiwanie informacji w internecie (**SerpAPI**), mechanizm **RAG** z baza
wektorowa i embeddingami, **agenta AI** samodzielnie dobierajacego narzedzia oraz
nowoczesny **interfejs graficzny** w Streamlit. Architektura jest modularna, odporna na
brak czesci kluczy API i latwa do rozszerzenia. Glowne ograniczenia wynikaja z zamknietej
listy gatunkow CLIP oraz zaleznosci od zewnetrznych uslug (Google/SerpAPI, OpenAI).
