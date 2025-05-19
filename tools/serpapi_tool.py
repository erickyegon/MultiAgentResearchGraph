import os
from dotenv import load_dotenv
load_dotenv()
import requests

SERP_API_API_KEY = os.getenv("SERP_API_KEY")


# Google searech Query
def search_google(query: str) -> list: 
    url = f"https://serpapi.com/search?engine=google&q={query}&api_key={SERP_API_KEY}" 
    response = requests.get(url)
    return response.json().get("organic_results", [])



