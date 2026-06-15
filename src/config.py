import os

from dotenv import load_dotenv

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

if not NEWS_API_KEY:
    raise ValueError("Brak NEWS_API_KEY w pliku .env")
