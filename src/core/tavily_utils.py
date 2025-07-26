from tavily import TavilyClient
import os

def get_tavily_client():
    return TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def search_tavily(query: str, num_results: int = 5):
    client = get_tavily_client()
    return client.search(query, num_results=num_results)

