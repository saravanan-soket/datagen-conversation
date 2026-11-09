"""`web_search` tool — Tavily API. Needs `TAVILY_API_KEY`.

(DuckDuckGo's scrape endpoints were tried first in the original Rust engine
but CAPTCHA-block automated/datacenter traffic, so Tavily it is.)
"""

from __future__ import annotations

import os

import requests

TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the live web. Use this to ground advice in real, current "
                       "information rather than guessing.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The search query"}},
            "required": ["query"],
        },
    },
}


def enabled() -> bool:
    return bool(os.environ.get("TAVILY_API_KEY", "").strip())


def search(query: str, timeout: float = 30.0) -> tuple[str, dict]:
    """Returns (text handed back to the model, structured details for the trace).
    Raises RuntimeError on failure — the caller feeds the message back to the model."""
    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY not set")
    resp = requests.post(
        "https://api.tavily.com/search",
        json={"api_key": api_key, "query": query, "max_results": 5, "include_answer": True},
        timeout=timeout,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"web_search: HTTP {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    answer = data.get("answer")
    results = data.get("results") or []

    out = f'Search results for "{query}":\n\n'
    if answer and answer.strip():
        out += f"Summary: {answer}\n\n"
    for i, r in enumerate(results, 1):
        snippet = (r.get("content") or "")[:400]
        out += f"{i}. {r.get('title', '')}\n   {r.get('url', '')}\n   {snippet}\n\n"
    if not results and not answer:
        out += "No results found.\n"

    details = {
        "query": query,
        "answer": answer,
        "results": [{"title": r.get("title"), "url": r.get("url")} for r in results],
    }
    return out, details
