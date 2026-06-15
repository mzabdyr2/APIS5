"""
Rozpoznawanie roslin ze zdjecia za pomoca modelu CLIP.

JAK TO DZIALA (w skrocie):
CLIP to model, ktory potrafi umiescic OBRAZ i TEKST w tej samej "przestrzeni
liczb" (tzw. embeddingi). Dzieki temu mozna policzyc, jak bardzo obraz pasuje
do danego opisu tekstowego.

My robimy tak:
1. Przygotowujemy liste opisow, np. "a photo of a monstera houseplant",
   "a photo of a cactus houseplant", itd. (patrz plik plant_labels.py).
2. Wrzucamy do modelu zdjecie uzytkownika oraz wszystkie opisy.
3. Model liczy podobienstwo zdjecia do kazdego opisu.
4. Wybieramy opis (czyli rosline) o najwyzszym podobienstwie.

To jest tzw. klasyfikacja "zero-shot" - nie trzeba uczyc modelu na wlasnych
zdjeciach roslin; wystarczy podac mu nazwy w formie tekstu.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from .config import settings
from .plant_labels import get_plant_keys, get_plant_prompts, pretty_name


@dataclass
class PlantPrediction:
    """Pojedyncza propozycja rosliny wraz z pewnoscia modelu (0..1)."""

    key: str          # techniczna nazwa, np. "monstera"
    label: str        # ladna nazwa, np. "Monstera (Monstera deliciosa)"
    confidence: float # pewnosc modelu (prawdopodobienstwo), np. 0.87


class CLIPPlantClassifier:
    """
    Opakowanie na model CLIP, ktore udostepnia jedna prosta metode: classify().

    Model jest ladowany dopiero przy pierwszym uzyciu (tzw. lazy loading), aby
    nie spowalniac startu aplikacji, jesli akurat nie potrzebujemy rozpoznawania.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.clip_model_name
        self._model = None
        self._processor = None
        self._device = None

    def _ensure_loaded(self) -> None:
        """Laduje model i procesor tylko raz, przy pierwszym wywolaniu."""
        if self._model is not None:
            return

        # Importy sa w srodku metody, aby ciezkie biblioteki (torch) byly
        # wczytywane dopiero gdy faktycznie sa potrzebne.
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = CLIPModel.from_pretrained(self.model_name).to(self._device)
        self._processor = CLIPProcessor.from_pretrained(self.model_name)
        self._model.eval()  # tryb tylko do predykcji (bez uczenia)

    def classify(self, image: Image.Image, top_k: int = 3) -> list[PlantPrediction]:
        """
        Rozpoznaje rosline na zdjeciu.

        Argumenty:
            image: zdjecie rosliny (obiekt PIL.Image).
            top_k: ile najlepszych propozycji zwrocic.

        Zwraca:
            Liste propozycji posortowana od najbardziej do najmniej pewnej.
        """
        import torch

        self._ensure_loaded()

        # Upewniamy sie, ze obraz jest w formacie RGB (3 kanaly kolorow).
        if image.mode != "RGB":
            image = image.convert("RGB")

        prompts = get_plant_prompts()
        keys = get_plant_keys()

        # Procesor zamienia obraz i teksty na format zrozumialy dla modelu.
        inputs = self._processor(
            text=prompts,
            images=image,
            return_tensors="pt",
            padding=True,
        ).to(self._device)

        with torch.no_grad():  # nie liczymy gradientow - tylko predykcja
            outputs = self._model(**inputs)
            # logits_per_image to "surowe" podobienstwa obrazu do kazdego opisu.
            logits = outputs.logits_per_image
            # softmax zamienia je na prawdopodobienstwa sumujace sie do 1.
            probs = logits.softmax(dim=1).squeeze(0)

        # Wybieramy top_k najlepszych wynikow.
        top_k = min(top_k, len(keys))
        values, indices = probs.topk(top_k)

        predictions: list[PlantPrediction] = []
        for value, index in zip(values.tolist(), indices.tolist()):
            key = keys[index]
            predictions.append(
                PlantPrediction(
                    key=key,
                    label=pretty_name(key),
                    confidence=float(value),
                )
            )
        return predictions


# Jeden wspolny obiekt klasyfikatora dla calej aplikacji (oszczedza pamiec,
# bo model laduje sie tylko raz).
_classifier: CLIPPlantClassifier | None = None


def get_classifier() -> CLIPPlantClassifier:
    """Zwraca wspoldzielony obiekt klasyfikatora (tworzy go przy pierwszym uzyciu)."""
    global _classifier
    if _classifier is None:
        _classifier = CLIPPlantClassifier()
    return _classifier
