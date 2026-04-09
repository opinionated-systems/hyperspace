"""
Web search tool: simulate web search for information retrieval.

Provides a simulated web search capability that can search for
information about programming concepts, documentation, and general knowledge.
This is a mock implementation for the agent's self-improvement process.
"""

from __future__ import annotations

from typing import Any


def tool_info() -> dict:
    return {
        "name": "web_search",
        "description": (
            "Simulate web search for information retrieval. "
            "Searches for programming concepts, documentation, and general knowledge. "
            "Returns relevant search results with summaries."
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
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    }


# Mock knowledge base for common programming topics
_KNOWLEDGE_BASE: dict[str, list[dict[str, Any]]] = {
    "python": [
        {"title": "Python Documentation", "url": "https://docs.python.org/3", "snippet": "Official Python documentation with tutorials and library references."},
        {"title": "Python Best Practices", "url": "https://realpython.com", "snippet": "Comprehensive guides on Python programming patterns and best practices."},
    ],
    "async": [
        {"title": "Asyncio Documentation", "url": "https://docs.python.org/3/library/asyncio.html", "snippet": "Python's asyncio library for writing concurrent code using async/await syntax."},
        {"title": "Async Python Patterns", "url": "https://superfastpython.com", "snippet": "Patterns and best practices for asynchronous programming in Python."},
    ],
    "error handling": [
        {"title": "Python Exception Handling", "url": "https://docs.python.org/3/tutorial/errors.html", "snippet": "How to handle exceptions and errors in Python programs."},
        {"title": "Best Practices for Error Handling", "url": "https://python-guide.readthedocs.io", "snippet": "Guidelines for robust error handling and logging in Python."},
    ],
    "testing": [
        {"title": "pytest Documentation", "url": "https://docs.pytest.org", "snippet": "Full-featured Python testing tool with fixtures and plugins."},
        {"title": "Unit Testing in Python", "url": "https://docs.python.org/3/library/unittest.html", "snippet": "Built-in unit testing framework for Python."},
    ],
    "logging": [
        {"title": "Python Logging HOWTO", "url": "https://docs.python.org/3/howto/logging.html", "snippet": "Comprehensive guide to Python's built-in logging module."},
        {"title": "Logging Best Practices", "url": "https://docs.python-guide.org", "snippet": "Best practices for structured logging in Python applications."},
    ],
    "performance": [
        {"title": "Python Performance Tips", "url": "https://wiki.python.org/moin/PythonSpeed", "snippet": "Tips and techniques for optimizing Python code performance."},
        {"title": "Profiling Python Code", "url": "https://docs.python.org/3/library/profile.html", "snippet": "Tools for profiling and analyzing Python program performance."},
    ],
    "type hints": [
        {"title": "Python Type Hints", "url": "https://docs.python.org/3/library/typing.html", "snippet": "Support for type hints as specified by PEP 484 and related PEPs."},
        {"title": "mypy Documentation", "url": "https://mypy.readthedocs.io", "snippet": "Static type checker for Python with comprehensive type inference."},
    ],
}


def _generate_generic_results(query: str, num_results: int) -> list[dict[str, Any]]:
    """Generate generic search results for any query."""
    results = []
    for i in range(min(num_results, 5)):
        results.append({
            "title": f"Search Result {i+1} for '{query}'",
            "url": f"https://example.com/search/{i+1}?q={query.replace(' ', '+')}",
            "snippet": f"This is a simulated search result about {query}. In a real implementation, this would contain actual content from web pages.",
        })
    return results


def _find_relevant_results(query: str, num_results: int) -> list[dict[str, Any]]:
    """Find relevant results from the knowledge base or generate generic ones."""
    query_lower = query.lower()
    
    # Check for matches in knowledge base
    all_matches = []
    for keyword, results in _KNOWLEDGE_BASE.items():
        if keyword in query_lower:
            all_matches.extend(results)
    
    # If we have matches, return them (up to num_results)
    if all_matches:
        return all_matches[:num_results]
    
    # Otherwise, generate generic results
    return _generate_generic_results(query, num_results)


def tool_function(query: str, num_results: int = 5) -> str:
    """Execute a web search query.
    
    Args:
        query: The search query string
        num_results: Number of results to return (default: 5, max: 10)
    
    Returns:
        Formatted search results
    """
    try:
        # Validate and cap num_results
        num_results = max(1, min(num_results, 10))
        
        # Find relevant results
        results = _find_relevant_results(query, num_results)
        
        # Format output
        lines = [
            f"Web Search Results for: '{query}'",
            "=" * 50,
            f"Found {len(results)} results:\n",
        ]
        
        for i, result in enumerate(results, 1):
            lines.append(f"{i}. {result['title']}")
            lines.append(f"   URL: {result['url']}")
            lines.append(f"   {result['snippet']}\n")
        
        lines.append("Note: This is a simulated search for the agent's self-improvement process.")
        
        return "\n".join(lines)
        
    except Exception as e:
        return f"Error performing web search: {e}"
