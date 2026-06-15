from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from src.config import OLLAMA_MODEL
from src.tools_importance import assess_importance
from src.tools_news import search_news
from src.tools_pdf import save_markdown_as_pdf
from src.tools_summary import summarize_news

llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
tools = [search_news, summarize_news, assess_importance, save_markdown_as_pdf]

SYSTEM_PROMPT = """
Jesteś agentem analizującym newsy.
Masz narzędzia do:
- wyszukiwania wiadomości,
- tworzenia podsumowania,
- oceny istotności (low/medium/high),
- zapisu raportu do PDF.

Działaj samodzielnie i dobieraj narzędzia do celu użytkownika.
Nie wymyślaj faktów, opieraj się na danych z narzędzi.
Jeśli użytkownik prosi o raport, zapisz go jako PDF.
"""

agent = create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)


def _extract_text(result) -> str:
    """Extract final text from a create_agent invoke response."""
    if isinstance(result, dict):
        messages = result.get("messages", [])
        if messages:
            last = messages[-1]
            if hasattr(last, "content"):
                return last.content
            if isinstance(last, dict):
                return str(last.get("content", ""))
        if "output" in result:
            return str(result["output"])
    return str(result)


def run(user_query: str) -> str:
    """Run the agent for a user query and return text output."""
    result = agent.invoke({"messages": [{"role": "user", "content": user_query}]})
    return _extract_text(result)


if __name__ == "__main__":
    print(run("Przygotuj krótkie podsumowanie tematu: AI w medycynie."))
