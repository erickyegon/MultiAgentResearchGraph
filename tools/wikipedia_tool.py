import os
from dotenv import load_dotenv
load_dotenv()
import wikipedia
import requests
from utils.config import WIKI_API_URL
WIKI_API_URL = "https://en.wikipedia.org/w/api.php"

def search_wikipedia(query: str) -> list:
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "props": "extracts|info",
        "exintro": 1,
        "explaintext": 1,
        "inprop": "url",
        "titles": query
    }
    
    response = requests.get(WIKI_API_URL, params=params)
    page = response.json().get("query", {}).get("pages", {})
    if page:
        page = list(page.values())[0]
        return [page]
    else:
        return []
    return page.get("extract", "")