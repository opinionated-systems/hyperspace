"""
Web search tool: search for information on the internet.

Provides web search capabilities using search APIs to find relevant
information from the internet. Useful for gathering up-to-date information
that may not be available in the local codebase.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from typing import Any


def tool_info() -> dict:
    return {
        "name": "web_search",
        "description": (
            "Search for information on the internet. "
            "Uses web search APIs to find relevant web pages and information. "
            "Returns search results with titles, URLs, and snippets."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string.",
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (default: 5, max: 10).",
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
        },
    }


def _perform_web_search(query: str, num_results: int = 5) -> list[dict[str, Any]]:
    """Perform a web search using available search APIs.
    
    Tries multiple search providers in order of preference:
    1. SerpAPI (Google search)
    2. Bing Search API
    3. DuckDuckGo instant answer (fallback)
    
    Args:
        query: The search query
        num_results: Number of results to return
        
    Returns:
        List of search result dictionaries with title, url, and snippet
    """
    results: list[dict[str, Any]] = []
    
    # Try SerpAPI first (Google search)
    serpapi_key = os.environ.get("SERPAPI_KEY")
    if serpapi_key:
        try:
            params = urllib.parse.urlencode({
                "q": query,
                "api_key": serpapi_key,
                "num": min(num_results, 10),
                "engine": "google",
            })
            url = f"https://serpapi.com/search?{params}"
            
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "HyperAgents/1.0"},
            )
            
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
                
                organic_results = data.get("organic_results", [])
                for result in organic_results[:num_results]:
                    results.append({
                        "title": result.get("title", "No title"),
                        "url": result.get("link", "No URL"),
                        "snippet": result.get("snippet", "No snippet available"),
                    })
                
                if results:
                    return results
        except Exception:
            pass  # Fall through to next provider
    
    # Try Bing Search API
    bing_key = os.environ.get("BING_SEARCH_KEY")
    if bing_key:
        try:
            encoded_query = urllib.parse.quote(query)
            url = f"https://api.bing.microsoft.com/v7.0/search?q={encoded_query}&count={min(num_results, 10)}"
            
            req = urllib.request.Request(
                url,
                headers={
                    "Ocp-Apim-Subscription-Key": bing_key,
                    "User-Agent": "HyperAgents/1.0",
                },
            )
            
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
                
                web_pages = data.get("webPages", {}).get("value", [])
                for page in web_pages[:num_results]:
                    results.append({
                        "title": page.get("name", "No title"),
                        "url": page.get("url", "No URL"),
                        "snippet": page.get("snippet", "No snippet available"),
                    })
                
                if results:
                    return results
        except Exception:
            pass  # Fall through to next provider
    
    # Fallback: DuckDuckGo instant answer (limited results)
    try:
        encoded_query = urllib.parse.quote(query)
        url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&no_html=1&skip_disambig=1"
        
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "HyperAgents/1.0",
                "Accept": "application/json",
            },
        )
        
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
            
            # Add abstract if available
            abstract = data.get("Abstract", "")
            abstract_url = data.get("AbstractURL", "")
            abstract_text = data.get("AbstractText", "")
            
            if abstract_url and abstract_text:
                results.append({
                    "title": data.get("Heading", "Search Result"),
                    "url": abstract_url,
                    "snippet": abstract_text,
                })
            
            # Add related topics
            related = data.get("RelatedTopics", [])
            for topic in related[:num_results - len(results)]:
                if isinstance(topic, dict) and "FirstURL" in topic:
                    results.append({
                        "title": topic.get("Text", "Related Topic").split(" - ")[0] if " - " in topic.get("Text", "") else topic.get("Text", "Related Topic"),
                        "url": topic.get("FirstURL", "No URL"),
                        "snippet": topic.get("Text", "No snippet available"),
                    })
            
            if results:
                return results
    except Exception:
        pass
    
    return results


def tool_function(
    query: str,
    num_results: int = 5,
) -> str:
    """Search for information on the internet.
    
    Args:
        query: The search query string
        num_results: Number of results to return (default: 5, max: 10)
        
    Returns:
        Formatted search results with titles, URLs, and snippets
    """
    if not query or not query.strip():
        return "Error: Query cannot be empty"
    
    # Clamp num_results to valid range
    num_results = max(1, min(num_results, 10))
    
    try:
        results = _perform_web_search(query, num_results)
        
        if not results:
            # Check if any API keys are configured
            has_serpapi = bool(os.environ.get("SERPAPI_KEY"))
            has_bing = bool(os.environ.get("BING_SEARCH_KEY"))
            
            if not has_serpapi and not has_bing:
                return (
                    "No search results found. Note: No search API keys are configured. "
                    "Set SERPAPI_KEY or BING_SEARCH_KEY environment variable for better results. "
                    "Limited results may be available via DuckDuckGo."
                )
            
            return f"No search results found for query: '{query}'"
        
        # Format results
        output_lines = [f"Web search results for: '{query}'\n"]
        
        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            url = result.get("url", "No URL")
            snippet = result.get("snippet", "No snippet available")
            
            output_lines.append(f"{i}. {title}")
            output_lines.append(f"   URL: {url}")
            output_lines.append(f"   {snippet}\n")
        
        return "\n".join(output_lines)
        
    except Exception as e:
        return f"Error performing web search: {e}"
