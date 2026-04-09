"""
Web search tool for searching information online.

Provides a simple interface to search the web using DuckDuckGo.
"""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)


def tool_info() -> dict:
    """Return tool metadata."""
    return {
        "name": "web_search",
        "description": (
            "Search the web for information using DuckDuckGo. "
            "Returns search results with titles, URLs, and snippets. "
            "Useful for finding current information, documentation, "
            "or verifying facts."
        ),
        "input_schema": {
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
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
        },
    }


def _duckduckgo_search(query: str, num_results: int = 5) -> list[dict[str, Any]]:
    """Perform a DuckDuckGo search and return results."""
    try:
        # DuckDuckGo HTML search endpoint
        encoded_query = urllib.parse.quote(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
        
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        
        req = urllib.request.Request(url, headers=headers)
        
        with urllib.request.urlopen(req, timeout=30) as response:
            html = response.read().decode("utf-8", errors="ignore")
        
        # Parse results from HTML
        results = []
        import re
        
        # Find result blocks
        result_blocks = re.findall(
            r'<div class="result results_links[^"]*"[^>]*>.*?<\/div>\s*<\/div>',
            html,
            re.DOTALL
        )
        
        for block in result_blocks[:num_results]:
            # Extract title and URL
            title_match = re.search(
                r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)<\/a>',
                block,
                re.DOTALL
            )
            
            # Extract snippet
            snippet_match = re.search(
                r'<a[^>]*class="result__snippet"[^>]*>(.*?)<\/a>',
                block,
                re.DOTALL
            )
            
            if title_match:
                url = title_match.group(1)
                title = re.sub(r'<[^>]+>', '', title_match.group(2)).strip()
                
                snippet = ""
                if snippet_match:
                    snippet = re.sub(r'<[^>]+>', '', snippet_match.group(1)).strip()
                
                # Clean up URL (DuckDuckGo uses redirects)
                if url.startswith("//"):
                    url = "https:" + url
                elif url.startswith("/"):
                    url = "https://duckduckgo.com" + url
                
                results.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                })
        
        return results
        
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return []


def tool_function(query: str, num_results: int = 5) -> str:
    """Execute a web search and return formatted results."""
    logger.info(f"Web search: {query!r} (max {num_results} results)")
    
    try:
        results = _duckduckgo_search(query, num_results)
        
        if not results:
            return "No results found for the query."
        
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append(
                f"{i}. {result['title']}\n"
                f"   URL: {result['url']}\n"
                f"   {result['snippet']}\n"
            )
        
        return "\n".join(formatted_results)
        
    except Exception as e:
        error_msg = f"Error performing web search: {e}"
        logger.error(error_msg)
        return error_msg
