"""
Error analysis tool: analyze error messages and suggest fixes.

Provides intelligent error analysis with pattern matching and suggestions
for common Python and system errors.
"""

from __future__ import annotations

import re
from typing import Any


# Common error patterns and their explanations/fixes
ERROR_PATTERNS = {
    # Python syntax errors
    r"SyntaxError: invalid syntax": {
        "category": "Syntax Error",
        "explanation": "Python cannot parse the code due to incorrect syntax.",
        "common_causes": [
            "Missing colon (:) at the end of if/for/while/def statements",
            "Mismatched parentheses, brackets, or braces",
            "Incorrect indentation",
            "Using = instead of == for comparison",
        ],
        "suggestions": [
            "Check for missing colons after control statements",
            "Verify all parentheses and brackets are balanced",
            "Ensure consistent indentation (4 spaces recommended)",
        ],
    },
    r"IndentationError": {
        "category": "Indentation Error",
        "explanation": "Python code has inconsistent or incorrect indentation.",
        "common_causes": [
            "Mixing tabs and spaces",
            "Incorrect indentation level for a block",
            "Missing indentation after a control statement",
        ],
        "suggestions": [
            "Use 4 spaces for each indentation level",
            "Configure your editor to convert tabs to spaces",
            "Check that all lines in a block have the same indentation",
        ],
    },
    r"NameError: name '(\w+)' is not defined": {
        "category": "Name Error",
        "explanation": "A variable or function name is used before being defined.",
        "common_causes": [
            "Variable was never assigned a value",
            "Typo in the variable name",
            "Variable defined in a different scope",
            "Missing import statement",
        ],
        "suggestions": [
            "Check for typos in the variable name",
            "Ensure the variable is assigned before use",
            "Check if you need to import the name",
            "Verify the variable is in the correct scope",
        ],
    },
    r"TypeError: (.*)": {
        "category": "Type Error",
        "explanation": "An operation was performed on incompatible types.",
        "common_causes": [
            "Passing wrong type to a function",
            "Trying to concatenate different types (e.g., str + int)",
            "Calling a method on None",
            "Incorrect number of arguments",
        ],
        "suggestions": [
            "Check the types of all arguments",
            "Use type() or isinstance() to debug type issues",
            "Convert types explicitly if needed (str(), int(), etc.)",
            "Check function signature for correct argument count",
        ],
    },
    r"KeyError: (.*)": {
        "category": "Key Error",
        "explanation": "Attempted to access a dictionary key that doesn't exist.",
        "common_causes": [
            "Key was never added to the dictionary",
            "Typo in the key name",
            "Key was deleted",
        ],
        "suggestions": [
            "Use dict.get(key, default) to safely access keys",
            "Check if key exists with 'if key in dict:'",
            "Use dict.setdefault() to create missing keys",
        ],
    },
    r"IndexError: (.*)": {
        "category": "Index Error",
        "explanation": "Attempted to access a list index that doesn't exist.",
        "common_causes": [
            "Index is larger than list length",
            "Accessing index on an empty list",
            "Off-by-one error in loop bounds",
        ],
        "suggestions": [
            "Check list length with len() before accessing",
            "Use negative indices carefully (-1 is last element)",
            "Consider using list slicing for safe access",
        ],
    },
    r"AttributeError: (.*)": {
        "category": "Attribute Error",
        "explanation": "Attempted to access an attribute that doesn't exist.",
        "common_causes": [
            "Method or attribute name typo",
            "Object is of wrong type",
            "Module not properly imported",
        ],
        "suggestions": [
            "Check available attributes with dir(obj)",
            "Verify the object type with type(obj)",
            "Check if module was imported correctly",
        ],
    },
    r"ModuleNotFoundError: No module named '(\w+)'": {
        "category": "Import Error",
        "explanation": "Python cannot find the specified module.",
        "common_causes": [
            "Module is not installed",
            "Module name is misspelled",
            "Module is in a different Python environment",
        ],
        "suggestions": [
            "Install the module with pip install <module>",
            "Check the correct module name",
            "Verify you're in the correct virtual environment",
        ],
    },
    r"FileNotFoundError": {
        "category": "File Error",
        "explanation": "The specified file or directory does not exist.",
        "common_causes": [
            "File path is incorrect",
            "File was deleted or moved",
            "Relative path is wrong",
            "Permission denied",
        ],
        "suggestions": [
            "Verify the file path is correct",
            "Use absolute paths or check working directory",
            "Check file permissions",
            "Create the file if it doesn't exist",
        ],
    },
    r"PermissionError": {
        "category": "Permission Error",
        "explanation": "Insufficient permissions to perform the operation.",
        "common_causes": [
            "File is read-only",
            "Directory requires elevated permissions",
            "File is locked by another process",
        ],
        "suggestions": [
            "Check file permissions with ls -l",
            "Run with appropriate permissions",
            "Close other programs using the file",
        ],
    },
    r"ConnectionError|ConnectionRefusedError|ConnectionResetError": {
        "category": "Network Error",
        "explanation": "Failed to establish or maintain a network connection.",
        "common_causes": [
            "Server is not running",
            "Network is unreachable",
            "Firewall blocking connection",
            "Incorrect host or port",
        ],
        "suggestions": [
            "Check if the server is running",
            "Verify network connectivity",
            "Check firewall settings",
            "Verify host and port are correct",
        ],
    },
    r"TimeoutError|socket\.timeout": {
        "category": "Timeout Error",
        "explanation": "An operation took longer than the allowed time.",
        "common_causes": [
            "Server is slow or unresponsive",
            "Network latency is high",
            "Operation is too complex",
        ],
        "suggestions": [
            "Increase timeout duration",
            "Check server status",
            "Optimize the operation",
            "Implement retry logic with backoff",
        ],
    },
    r"JSONDecodeError": {
        "category": "JSON Error",
        "explanation": "Failed to parse a string as JSON.",
        "common_causes": [
            "String is not valid JSON",
            "Missing quotes around keys",
            "Trailing commas in objects/arrays",
            "Using single quotes instead of double quotes",
        ],
        "suggestions": [
            "Validate JSON with a linter",
            "Use double quotes for strings and keys",
            "Remove trailing commas",
            "Check for special characters that need escaping",
        ],
    },
    r"RecursionError": {
        "category": "Recursion Error",
        "explanation": "Maximum recursion depth exceeded.",
        "common_causes": [
            "Infinite recursion in a function",
            "Base case never reached",
            "Recursion depth too deep for the problem",
        ],
        "suggestions": [
            "Check base case condition in recursive function",
            "Convert to iterative approach",
            "Increase recursion limit if truly needed (sys.setrecursionlimit)",
        ],
    },
    r"MemoryError": {
        "category": "Memory Error",
        "explanation": "System ran out of memory.",
        "common_causes": [
            "Loading too much data at once",
            "Infinite loop creating objects",
            "Large data structures",
        ],
        "suggestions": [
            "Process data in chunks",
            "Use generators instead of lists",
            "Check for memory leaks",
            "Optimize data structures",
        ],
    },
}


def _analyze_error(error_message: str) -> dict[str, Any]:
    """Analyze an error message and return structured information."""
    error_message = error_message.strip()
    
    # Try to match against known patterns
    for pattern, info in ERROR_PATTERNS.items():
        match = re.search(pattern, error_message, re.IGNORECASE)
        if match:
            result = {
                "matched": True,
                "category": info["category"],
                "explanation": info["explanation"],
                "common_causes": info["common_causes"],
                "suggestions": info["suggestions"],
                "matched_pattern": pattern,
            }
            # Extract any captured groups
            if match.groups():
                result["extracted_info"] = match.groups()
            return result
    
    # No pattern matched
    return {
        "matched": False,
        "category": "Unknown Error",
        "explanation": "This error pattern is not recognized.",
        "common_causes": ["Unknown - manual investigation required"],
        "suggestions": [
            "Search for the error message online",
            "Check the full traceback for context",
            "Review recent code changes",
        ],
    }


def tool_info() -> dict:
    """Return tool metadata."""
    return {
        "name": "error_analysis",
        "description": "Analyze error messages and provide explanations, common causes, and suggestions for fixes. Supports Python errors, system errors, and common runtime issues.",
        "input_schema": {
            "type": "object",
            "properties": {
                "error_message": {
                    "type": "string",
                    "description": "The error message or traceback to analyze",
                },
            },
            "required": ["error_message"],
        },
    }


def tool_function(error_message: str) -> str:
    """Analyze an error message and provide helpful information."""
    if not error_message or not error_message.strip():
        return "Error: Please provide an error message to analyze."
    
    analysis = _analyze_error(error_message)
    
    lines = [
        f"Error Category: {analysis['category']}",
        "",
        f"Explanation: {analysis['explanation']}",
        "",
        "Common Causes:",
    ]
    
    for i, cause in enumerate(analysis['common_causes'], 1):
        lines.append(f"  {i}. {cause}")
    
    lines.extend(["", "Suggested Fixes:"])
    
    for i, suggestion in enumerate(analysis['suggestions'], 1):
        lines.append(f"  {i}. {suggestion}")
    
    if analysis.get('extracted_info'):
        lines.extend([
            "",
            f"Extracted Information: {analysis['extracted_info']}",
        ])
    
    if not analysis['matched']:
        lines.extend([
            "",
            "Note: This error pattern is not in the known database. Consider:",
            "  - Searching Stack Overflow or documentation",
            "  - Checking the full error traceback",
            "  - Verifying environment and dependencies",
        ])
    
    return "\n".join(lines)
