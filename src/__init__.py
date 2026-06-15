"""
Pakiet zrodlowy aplikacji "Asystent Pielegnacji Roslin".

Zawiera wszystkie moduly skladowe systemu:
- config           -> konfiguracja i klucze API
- plant_labels     -> lista nazw roslin (klas) dla modelu CLIP
- clip_classifier  -> rozpoznawanie roslin ze zdjec (model CLIP)
- web_search       -> wyszukiwanie informacji w internecie (SerpAPI)
- rag              -> baza wektorowa + embeddingi + retrieval (RAG)
- llm              -> obsluga modelu jezykowego (OpenAI)
- agent            -> agent AI laczacy powyzsze narzedzia w calosc
"""
