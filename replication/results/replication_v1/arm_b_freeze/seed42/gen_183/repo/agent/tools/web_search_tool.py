"""
Web search tool: search the internet for information.

Provides web search capabilities using search APIs.
Falls back to mock results if no API is configured.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.parse
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)


def tool_info() -> dict:
    return {
        "name": "web_search",
        "description": (
            "Search the web for information. "
            "Returns search results with titles, snippets, and URLs. "
            "Use this to find up-to-date information, documentation, or examples."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to execute.",
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (default: 5, max: 10).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    }


def _mock_search(query: str, num_results: int) -> list[dict[str, Any]]:
    """Return mock search results when no API is available."""
    return [
        {
            "title": f"Mock result for: {query}",
            "snippet": "This is a mock search result. Configure a search API for real results.",
            "url": "https://example.com/mock",
        }
    ]


def _duckduckgo_search(query: str, num_results: int = 5) -> list[dict[str, Any]]:
    """Search using DuckDuckGo HTML interface (no API key required)."""
    try:
        encoded_query = urllib.parse.quote(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
        
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.0"
            },
        )
        
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode("utf-8", errors="ignore")
        
        # Simple parsing of DuckDuckGo results
        results = []
        import re
        
        # Find result blocks
        result_blocks = re.findall(
            r'<a rel="nofollow" class="result__a" href="([^"]+)">([^<]+)</a>.*?<a class="result__snippet"[^>]*>([^<]+)</a>',
            html,
            re.DOTALL,
        )
        
        for i, (href, title, snippet) in enumerate(result_blocks[:num_results]):
            # Clean up HTML entities
            title = title.replace("&quot;", '"').replace("&amp;", "&").replace("&#x27;", "'")
            snippet = snippet.replace("&quot;", '"').replace("&amp;", "&").replace("&#x27;", "'")
            
            results.append({
                "title": title.strip(),
                "snippet": snippet.strip(),
                "url": href.strip(),
            })
        
        return results if results else _mock_search(query, num_results)
        
    except Exception as e:
        logger.warning(f"DuckDuckGo search failed: {e}")
        return _mock_search(query, num_results)


def tool_function(query: str, num_results: int = 5) -> str:
    """Execute a web search and return formatted results.
    
    Args:
        query: The search query
        num_results: Number of results to return (default: 5, max: 10)
    
    Returns:
        JSON string with search results
    """
    # Clamp num_results
    num_results = max(1, min(10, num_results))
    
    try:
        # Try DuckDuckGo search first (no API key needed)
        results = _duckduckgo_search(query, num_results)
        
        return json.dumps({
            "query": query,
            "num_results": len(results),
            "results": results,
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return json.dumps({
            "query": query,
            "error": str(e),
            "results": _mock_search(query, num_results),
        }, indent=2)
