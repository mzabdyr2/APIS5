from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from src.config import OLLAMA_MODEL

llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)


@tool
def assess_importance(summary: str, domain: str) -> str:
    """Assess topic importance for a domain and return low, medium or high."""
    prompt = f"""
Oceń istotność informacji dla dziedziny: {domain}.

Podsumowanie:
{summary}

Zasady:
- high: duży wpływ społeczny, gospodarczy, prawny, zdrowotny lub bezpieczeństwa
- medium: istotne, ale wpływ ograniczony lub lokalny
- low: ciekawostkowe, mało wpływające na decyzje i praktykę

Odpowiedz jednym słowem: low albo medium albo high.
Bez dodatkowego tekstu.
"""

    answer = llm.invoke(prompt).content.strip().lower()
    if "high" in answer:
        return "high"
    if "medium" in answer:
        return "medium"
    return "low"
