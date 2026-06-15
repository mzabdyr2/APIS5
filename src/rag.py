"""
RAG = Retrieval-Augmented Generation (generowanie wspomagane wyszukiwaniem).

PROBLEM: Model jezykowy zna ogolna wiedze, ale nie zna swiezych artykulow z
internetu o konkretnej roslinie. Moze tez "zmyslac".

ROZWIAZANIE (RAG):
1. Pobrane artykuly tniemy na male fragmenty (chunki).
2. Kazdy fragment zamieniamy na wektor liczb (embedding) - to "wspolrzedne"
   znaczenia tekstu w wielowymiarowej przestrzeni.
3. Wektory wrzucamy do bazy wektorowej (FAISS).
4. Gdy uzytkownik zada pytanie, zamieniamy je tez na wektor i szukamy w bazie
   fragmentow o najbardziej podobnym znaczeniu (retrieval).
5. Znalezione fragmenty doklejamy do zapytania jako KONTEKST dla modelu.

Dzieki temu model odpowiada na podstawie konkretnych, pobranych informacji.

Ten modul zawiera dwie klasy:
- VectorStore  -> sama baza wektorowa (przechowuje wektory i teksty).
- RAGPipeline  -> caly proces: chunking, embeddingi, dodawanie i wyszukiwanie.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import settings


@dataclass
class RetrievedChunk:
    """Fragment tekstu znaleziony w bazie wraz z metadanymi i ocena podobienstwa."""

    text: str
    source_title: str
    source_url: str
    score: float


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Dzieli dlugi tekst na mniejsze, czesciowo nakladajace sie fragmenty.

    Dlaczego nakladajace sie? Aby zdanie przeciete na granicy dwoch chunkow nie
    zgubilo sensu - "overlap" powtarza koncowke poprzedniego fragmentu na
    poczatku nastepnego.

    Tniemy po slowach (a nie po znakach), zeby nie rozcinac slow w polowie.
    """
    words = text.split()
    if not words:
        return []

    # Przeliczamy rozmiary z liczby znakow na przyblizona liczbe slow
    # (przyjmujemy, ze srednie slowo to ~5 znakow + spacja).
    words_per_chunk = max(1, chunk_size // 6)
    words_overlap = max(0, overlap // 6)
    step = max(1, words_per_chunk - words_overlap)

    chunks: list[str] = []
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + words_per_chunk])
        if chunk.strip():
            chunks.append(chunk.strip())
        if start + words_per_chunk >= len(words):
            break
    return chunks


class VectorStore:
    """
    Prosta baza wektorowa oparta o FAISS.

    Przechowuje:
    - wektory (embeddingi) fragmentow tekstu,
    - same teksty oraz informacje o ich zrodle.

    Jesli biblioteka FAISS nie jest dostepna, klasa automatycznie przelacza sie
    na zapasowy tryb oparty o numpy (wolniejszy, ale dziala tak samo logicznie).
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._texts: list[str] = []
        self._meta: list[dict] = []
        self._use_faiss = False
        self._index = None
        self._matrix: np.ndarray | None = None  # zapasowy magazyn dla trybu numpy

        try:
            import faiss

            # IndexFlatIP = wyszukiwanie po iloczynie skalarnym. Po znormalizowaniu
            # wektorow iloczyn skalarny = podobienstwo kosinusowe (cosine similarity).
            self._index = faiss.IndexFlatIP(dim)
            self._use_faiss = True
        except Exception:
            self._use_faiss = False

    def add(self, vectors: np.ndarray, texts: list[str], metas: list[dict]) -> None:
        """Dodaje nowe wektory wraz z odpowiadajacymi im tekstami do bazy."""
        if len(texts) == 0:
            return
        vectors = vectors.astype("float32")
        if self._use_faiss:
            self._index.add(vectors)
        else:
            if self._matrix is None:
                self._matrix = vectors
            else:
                self._matrix = np.vstack([self._matrix, vectors])
        self._texts.extend(texts)
        self._meta.extend(metas)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        """Zwraca indeksy i oceny podobienstwa najlepiej pasujacych fragmentow."""
        if len(self._texts) == 0:
            return []
        query_vector = query_vector.astype("float32").reshape(1, -1)
        top_k = min(top_k, len(self._texts))

        if self._use_faiss:
            scores, indices = self._index.search(query_vector, top_k)
            return list(zip(indices[0].tolist(), scores[0].tolist()))

        # Tryb zapasowy (numpy): liczymy podobienstwo kosinusowe recznie.
        sims = (self._matrix @ query_vector.T).ravel()
        top_indices = np.argsort(-sims)[:top_k]
        return [(int(i), float(sims[i])) for i in top_indices]

    def get(self, index: int) -> tuple[str, dict]:
        """Zwraca tekst i metadane fragmentu o danym indeksie."""
        return self._texts[index], self._meta[index]

    def __len__(self) -> int:
        return len(self._texts)


class RAGPipeline:
    """
    Pelny proces RAG: zamiana tekstu na wektory, budowa bazy i wyszukiwanie.
    """

    def __init__(self) -> None:
        self.embedding_model_name = settings.embedding_model_name
        self._embedder = None
        self.store: VectorStore | None = None
        self.indexed_plant: str | None = None

    def _ensure_embedder(self) -> None:
        """Laduje model do liczenia embeddingow (tylko raz, przy pierwszym uzyciu)."""
        if self._embedder is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._embedder = SentenceTransformer(self.embedding_model_name)

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Zamienia liste tekstow na znormalizowane wektory (embeddingi)."""
        self._ensure_embedder()
        # normalize_embeddings=True sprawia, ze iloczyn skalarny = cosine similarity.
        vectors = self._embedder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype="float32")

    def build_index(self, documents, plant_name: str = "") -> int:
        """
        Buduje baze wektorowa z listy dokumentow (np. artykulow z internetu).

        Argumenty:
            documents: lista obiektow z polami .text, .title, .url
                       (np. SearchDocument z modulu web_search).
            plant_name: nazwa rosliny, ktorej dotyczy baza (do informacji).

        Zwraca:
            Liczbe fragmentow (chunkow) dodanych do bazy.
        """
        all_chunks: list[str] = []
        all_meta: list[dict] = []

        for doc in documents:
            chunks = chunk_text(doc.text, settings.chunk_size, settings.chunk_overlap)
            for chunk in chunks:
                all_chunks.append(chunk)
                all_meta.append({"title": doc.title, "url": doc.url})

        if not all_chunks:
            return 0

        vectors = self._embed(all_chunks)
        self.store = VectorStore(dim=vectors.shape[1])
        self.store.add(vectors, all_chunks, all_meta)
        self.indexed_plant = plant_name
        return len(all_chunks)

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """
        Znajduje w bazie fragmenty najlepiej pasujace do zapytania uzytkownika.
        """
        if self.store is None or len(self.store) == 0:
            return []

        top_k = top_k or settings.retrieval_top_k
        query_vector = self._embed([query])[0]
        hits = self.store.search(query_vector, top_k)

        results: list[RetrievedChunk] = []
        for index, score in hits:
            text, meta = self.store.get(index)
            results.append(
                RetrievedChunk(
                    text=text,
                    source_title=meta.get("title", ""),
                    source_url=meta.get("url", ""),
                    score=score,
                )
            )
        return results

    def build_context(self, query: str, top_k: int | None = None) -> str:
        """
        Zwraca gotowy tekst kontekstu (sklejone fragmenty + zrodla),
        ktory mozna wkleic do promptu modelu jezykowego.
        """
        chunks = self.retrieve(query, top_k)
        if not chunks:
            return ""

        parts: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            source = chunk.source_title or chunk.source_url or "zrodlo internetowe"
            parts.append(f"[Fragment {i} | zrodlo: {source}]\n{chunk.text}")
        return "\n\n".join(parts)

    def has_data(self) -> bool:
        """Czy baza zawiera jakiekolwiek dane?"""
        return self.store is not None and len(self.store) > 0
