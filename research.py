"""
Looks up public info about a target company so the generator can personalize
scripts/emails to them. Uses Tavily (free tier, built for feeding LLMs -
returns clean summarized snippets instead of raw HTML).

Falls back to a manual text box in the UI if no Tavily key is set, so the
app still works with zero API keys configured for this part.
"""
from __future__ import annotations
from functools import lru_cache

from config import TAVILY_API_KEY


def research_company(company_name: str) -> str:
    """Return a short text blob describing the target company: industry,
    size, recent news, anything useful for personalizing outreach."""
    if not TAVILY_API_KEY:
        return ""

    from tavily import TavilyClient
    client = TavilyClient(api_key=TAVILY_API_KEY)

    query = f"{company_name} company overview industry size recent news"
    response = client.search(query=query, search_depth="basic", max_results=5)

    snippets = []
    for r in response.get("results", []):
        title = r.get("title", "")
        content = r.get("content", "")
        snippets.append(f"- {title}: {content}")

    return "\n".join(snippets) if snippets else ""


@lru_cache(maxsize=64)
def research_company_cached(company_name: str) -> str:
    """Cache repeated lookups within a process run so re-generating a script
    for the same company doesn't burn another API call."""
    return research_company(company_name)
