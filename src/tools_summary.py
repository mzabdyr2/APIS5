import json

from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from src.config import OLLAMA_MODEL

llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)


@tool
def summarize_news(news_json: str, topic: str) -> str:
    """Create a concise Polish summary from news data in JSON format."""
    try:
        data = json.loads(news_json)
    except json.JSONDecodeError:
        return "Nie udało się odczytać danych wejściowych."

    if isinstance(data, dict) and data.get("error"):
        return f"Błąd danych: {data['error']}"

    if not data:
        return "Brak artykułów do podsumowania."

    lines = []
    for idx, article in enumerate(data, start=1):
        lines.append(
            f"{idx}. {article.get('title')} | źródło: {article.get('source')} | "
            f"data: {article.get('publishedAt')} | opis: {article.get('description')}"
        )

    prompt = (
        f"Temat: {topic}\n\n"
        f"Dane wejściowe:\n{'\n'.join(lines)}\n\n"
        "Napisz krótkie podsumowanie po polsku (6-10 zdań). "
        "Wskaż najważniejsze trendy i fakty."
    )
    return llm.invoke(prompt).content
