import json

import requests
from langchain_core.tools import tool

from src.config import NEWS_API_KEY


@tool
def search_news(topic: str, page_size: int = 5) -> str:
    """Find recent articles for a topic using NewsAPI.

    Returns a JSON string with title, url, source, publishedAt and description.
    """
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": topic,
        "pageSize": page_size,
        "language": "en",
        "sortBy": "publishedAt",
        "apiKey": NEWS_API_KEY,
    }

    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:  # pragma: no cover - network path
        return json.dumps({"error": f"NewsAPI error: {exc}"}, ensure_ascii=False)

    articles = []
    for article in data.get("articles", [])[:page_size]:
        articles.append(
            {
                "title": article.get("title"),
                "url": article.get("url"),
                "source": article.get("source", {}).get("name"),
                "publishedAt": article.get("publishedAt"),
                "description": article.get("description"),
            }
        )

    return json.dumps(articles, ensure_ascii=False)
