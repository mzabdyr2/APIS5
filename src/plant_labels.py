"""
Lista nazw popularnych roslin domowych.

Model CLIP rozpoznaje obraz przez porownanie go z opisami tekstowymi. Dlatego
musimy z gory podac liste "kandydatow" - czyli nazw roslin, sposrod ktorych
model wybierze najbardziej pasujaca do zdjecia.

Kazda nazwa to klucz w slowniku, a wartosc to ladna nazwa wyswietlana
uzytkownikowi (z nazwa lacinska dla pewnosci).
"""

# Klucz: nazwa angielska uzywana w promptcie CLIP (np. "monstera").
# Wartosc: czytelna nazwa pokazywana uzytkownikowi.
HOUSEPLANTS: dict[str, str] = {
    "monstera": "Monstera (Monstera deliciosa)",
    "snake plant": "Sansewieria / waz teściowej (Sansevieria trifasciata)",
    "pothos": "Epipremnum / scindapsus (Epipremnum aureum)",
    "spider plant": "Zielistka (Chlorophytum comosum)",
    "peace lily": "Skrzydlokwiat (Spathiphyllum)",
    "aloe vera": "Aloes (Aloe vera)",
    "fiddle leaf fig": "Fikus lirolistny (Ficus lyrata)",
    "rubber plant": "Fikus sprezysty / gumowiec (Ficus elastica)",
    "zz plant": "Zamiokulkas (Zamioculcas zamiifolia)",
    "dracaena": "Dracena / smokowiec (Dracaena)",
    "philodendron": "Filodendron (Philodendron)",
    "orchid": "Orchidea / storczyk (Orchidaceae)",
    "cactus": "Kaktus (Cactaceae)",
    "succulent": "Sukulent (Succulent)",
    "boston fern": "Paproc bostonska (Nephrolepis exaltata)",
    "calathea": "Kalatea (Calathea)",
    "english ivy": "Bluszcz pospolity (Hedera helix)",
    "jade plant": "Grubosz / drzewko szczescia (Crassula ovata)",
    "anthurium": "Anturium (Anthurium)",
    "begonia": "Begonia (Begonia)",
    "african violet": "Fiolek afrykanski (Saintpaulia)",
    "chinese money plant": "Pilea (Pilea peperomioides)",
    "areca palm": "Palma areka (Dypsis lutescens)",
    "bird of paradise": "Strelicja / rajski ptak (Strelitzia)",
    "croton": "Kroton (Codiaeum variegatum)",
    "geranium": "Pelargonia (Pelargonium)",
    "poinsettia": "Gwiazda betlejemska (Euphorbia pulcherrima)",
    "bamboo palm": "Palma bambusowa (Chamaedorea)",
    "ferns": "Paproc (Polypodiopsida)",
    "ivy": "Bluszcz (Hedera)",
}


def get_plant_prompts() -> list[str]:
    """
    Tworzy liste promptow tekstowych dla modelu CLIP.

    Z badan nad CLIP wiadomo, ze model dziala lepiej, gdy opis ma forme zdania,
    a nie pojedynczego slowa. Dlatego kazda nazwe rosliny wkladamy w szablon
    "a photo of a <nazwa> houseplant".
    """
    return [f"a photo of a {name} houseplant" for name in HOUSEPLANTS]


def get_plant_keys() -> list[str]:
    """Zwraca surowe nazwy roslin (klucze) w tej samej kolejnosci co prompty."""
    return list(HOUSEPLANTS.keys())


def pretty_name(plant_key: str) -> str:
    """Zamienia techniczna nazwe (np. 'monstera') na ladna nazwe dla uzytkownika."""
    return HOUSEPLANTS.get(plant_key, plant_key)
