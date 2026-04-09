"""
Web search tool for searching the internet.

Provides a simple interface for performing web searches and retrieving results.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.parse
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)


def tool_info() -> dict[str, Any]:
    """Return tool specification for web search."""
    return {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information. Returns search results with titles, URLs, and snippets. Uses DuckDuckGo HTML interface for searching.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to execute",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5, max: 10)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    }


def tool_function(query: str, num_results: int = 5) -> str:
    """Execute a web search and return results.
    
    Args:
        query: The search query
        num_results: Number of results to return (max 10)
        
    Returns:
        JSON string with search results
    """
    try:
        # Cap results at 10
        num_results = min(max(1, num_results), 10)
        
        # Use DuckDuckGo HTML interface
        encoded_query = urllib.parse.quote(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
        
        # Set headers to mimic a browser
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        
        req = urllib.request.Request(url, headers=headers)
        
        with urllib.request.urlopen(req, timeout=30) as response:
            html = response.read().decode("utf-8", errors="ignore")
        
        # Parse results from HTML
        results = []
        import re
        
        # Extract result blocks
        result_pattern = r'<div class="result__body">.*?<a class="result__a" href="([^"]+)"[^>]*>(.*?)</a>.*?<a class="result__snippet"[^>]*>(.*?)</a>.*?</div>'
        matches = re.findall(result_pattern, html, re.DOTALL)
        
        for i, (url, title, snippet) in enumerate(matches[:num_results]):
            # Clean up HTML entities and tags
            title = re.sub(r'<[^>]+>', '', title)
            title = title.replace('&quot;', '"').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
            
            snippet = re.sub(r'<[^>]+>', '', snippet)
            snippet = snippet.replace('&quot;', '"').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
            
            results.append({
                "title": title.strip(),
                "url": url.strip(),
                "snippet": snippet.strip(),
            })
        
        if not results:
            # Fallback: try alternative pattern
            alt_pattern = r'<a rel="nofollow" class="[^"]*result__a[^"]*" href="([^"]+)"[^>]*>(.*?)</a>'
            alt_matches = re.findall(alt_pattern, html, re.DOTALL)
            
            for i, (url, title) in enumerate(alt_matches[:num_results]):
                title = re.sub(r'<[^>]+>', '', title)
                title = title.replace('&quot;', '"').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
                
                results.append({
                    "title": title.strip(),
                    "url": url.strip(),
                    "snippet": "",
                })
        
        return json.dumps({
            "success": True,
            "query": query,
            "results": results,
            "count": len(results),
        }, indent=2)
        
    except Exception as e:
        logger.error("Web search failed: %s", e)
        return json.dumps({
            "success": False,
            "query": query,
            "error": str(e),
            "results": [],
            "count": 0,
        }, indent=2)
