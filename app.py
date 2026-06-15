"""
Interfejs graficzny aplikacji "Asystent Pielegnacji Roslin" (Streamlit).

Uruchomienie:
    streamlit run app.py

Co widzi uzytkownik:
- PANEL BOCZNY (sidebar): miejsce na wgranie zdjecia rosliny + przycisk
  "Rozpoznaj rosline" i status systemu (czy sa klucze API).
- GLOWNE OKNO: rozmowa w stylu ChatGPT - historia wiadomosci oraz pole tekstowe
  do zadawania pytan.

Aplikacja korzysta z agenta AI (src/agent.py), ktory sam decyduje, kiedy
rozpoznac rosline, kiedy szukac w internecie, a kiedy siegnac do bazy RAG.
"""

from __future__ import annotations

import io

import streamlit as st
from PIL import Image

from src.config import settings

st.set_page_config(
    page_title="Asystent Pielegnacji Roslin",
    page_icon="🌱",
    layout="centered",
)


# ---------------------------------------------------------------------------
# Inicjalizacja agenta. Trzymamy go w "session_state", aby przetrwal odswiezenia
# strony (Streamlit przy kazdej interakcji uruchamia skrypt od nowa).
# ---------------------------------------------------------------------------
def get_agent():
    """Tworzy agenta raz na sesje i zapamietuje go w stanie sesji."""
    if "agent" not in st.session_state:
        # Import wewnatrz funkcji, aby ewentualny blad (np. brak bibliotek)
        # pokazac w interfejsie, a nie wysypac caly skrypt.
        from src.agent import PlantCareAgent

        st.session_state.agent = PlantCareAgent()
        st.session_state.messages = []  # historia do wyswietlenia w GUI
    return st.session_state.agent


# ---------------------------------------------------------------------------
# PANEL BOCZNY: wgrywanie zdjecia + status
# ---------------------------------------------------------------------------
def render_sidebar(agent) -> None:
    with st.sidebar:
        st.header("🌿 Zdjecie rosliny")
        st.caption("Wgraj fotografie, a system rozpozna gatunek i pomoze w pielegnacji.")

        uploaded = st.file_uploader(
            "Wybierz zdjecie",
            type=["jpg", "jpeg", "png", "webp"],
            help="Najlepiej wyrazne zdjecie calej rosliny.",
        )

        if uploaded is not None:
            image = Image.open(io.BytesIO(uploaded.getvalue()))
            st.image(image, caption="Wgrane zdjecie", use_container_width=True)
            agent.set_image(image)

            if st.button("🔍 Rozpoznaj rosline", use_container_width=True):
                with st.spinner("Analizuje zdjecie modelem CLIP..."):
                    predictions = agent.classifier.classify(image, top_k=3)
                if predictions:
                    best = predictions[0]
                    agent.state.identified_plant = best.key
                    agent.state.identified_label = best.label
                    st.success(f"Rozpoznano: {best.label}")
                    st.write("**Inne mozliwosci:**")
                    for p in predictions:
                        st.write(f"- {p.label} — {p.confidence * 100:.1f}%")
                    # Dodajemy wiadomosc do rozmowy, aby agent znal kontekst.
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": (
                                f"Rozpoznalem na zdjeciu: **{best.label}** "
                                f"(pewnosc {best.confidence * 100:.1f}%). "
                                "Zapytaj mnie o cokolwiek zwiazanego z jej pielegnacja!"
                            ),
                        }
                    )

        st.divider()
        st.subheader("Status systemu")
        st.write("Model jezykowy (OpenAI):", "✅" if settings.has_openai() else "❌ brak klucza")
        st.write("Wyszukiwarka (SerpAPI):", "✅" if settings.has_serpapi() else "❌ brak klucza")
        if not settings.has_openai():
            st.warning(
                "Aby chatbot dzialal, ustaw OPENAI_API_KEY w pliku .env.",
                icon="⚠️",
            )

        st.divider()
        if st.button("🗑️ Wyczysc rozmowe", use_container_width=True):
            agent.reset_conversation()
            st.session_state.messages = []
            st.rerun()


# ---------------------------------------------------------------------------
# GLOWNE OKNO: rozmowa w stylu ChatGPT
# ---------------------------------------------------------------------------
def render_chat(agent) -> None:
    st.title("🌱 Asystent Pielegnacji Roslin")
    st.caption(
        "Wgraj zdjecie rosliny w panelu po lewej, a nastepnie zadawaj pytania "
        "o jej pielegnacje. System sam rozpozna gatunek i poszuka informacji."
    )

    # Wyswietlamy dotychczasowa historie rozmowy.
    for message in st.session_state.get("messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Pole tekstowe na dole (textbar + automatyczny przycisk wyslij).
    prompt = st.chat_input("Napisz wiadomosc, np. 'Jak czesto podlewac te rosline?'")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if not settings.has_openai():
                answer = (
                    "Nie moge odpowiedziec, bo brakuje klucza OPENAI_API_KEY. "
                    "Dodaj go do pliku .env i uruchom aplikacje ponownie."
                )
                st.markdown(answer)
            else:
                with st.spinner("Mysle i sprawdzam informacje..."):
                    answer = agent.chat(prompt)
                st.markdown(answer)
                if agent.state.last_tools_used:
                    used = ", ".join(dict.fromkeys(agent.state.last_tools_used))
                    st.caption(f"🛠️ Uzyte narzedzia: {used}")

        st.session_state.messages.append({"role": "assistant", "content": answer})


def main() -> None:
    try:
        agent = get_agent()
    except Exception as exc:
        st.error(f"Nie udalo sie uruchomic agenta: {exc}")
        st.stop()
        return

    render_sidebar(agent)
    render_chat(agent)


if __name__ == "__main__":
    main()
