"""
Task agent: solves a given task with a single LLM call.

Reimplemented from facebookresearch/HyperAgents task_agent.py.
Same interface, same JSON output format, same extraction logic.

This is the INITIAL task agent. The meta agent modifies this file
during self-improvement. The evaluation harness loads whatever
task_agent.py exists at the agent's repo path.
"""

from __future__ import annotations

import json
import logging
import re

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks with enhanced recovery.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Implements a multi-layer recovery strategy for malformed JSON:
    1. Direct JSON parsing
    2. Common fix application (trailing commas, quote fixes)
    3. Outermost object extraction for nested structures
    4. Key-value pair extraction as final fallback
    Also searches markdown code blocks as secondary source.
    """
    results = []
    search_from = 0
    
    # Primary: Extract from <json> tags
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        # Attempt multi-layer JSON recovery
        parsed = _parse_json_with_recovery(inner)
        if parsed is not None:
            results.append(parsed)
    
    # Secondary: Extract from markdown code blocks
    if not results:
        code_block_pattern = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if code_block_pattern:
            json_content = code_block_pattern.group(1).strip()
            parsed = _parse_json_with_recovery(json_content)
            if parsed is not None:
                results.append(parsed)
    
    return results or None


def _parse_json_with_recovery(text: str) -> dict | None:
    """Attempt to parse JSON with multiple recovery strategies.
    
    Returns the parsed dict or None if all strategies fail.
    """
    # Strategy 1: Direct parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Apply common fixes
    try:
        fixed = _fix_json_string(text)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Extract outermost JSON object
    try:
        obj = _extract_outermost_json(text)
        if obj:
            return obj
    except Exception:
        pass
    
    # Strategy 4: Extract key-value pairs manually
    try:
        obj = _extract_key_value_pairs(text)
        if obj and ("response" in obj or "reasoning" in obj):
            return obj
    except Exception:
        pass
    
    return None


def _extract_key_value_pairs(text: str) -> dict | None:
    """Extract key-value pairs from malformed JSON as a last resort.
    
    Looks for "key": "value" patterns to reconstruct a minimal valid JSON object.
    Handles string values, booleans, null, and numeric values.
    """
    result = {}
    # Pattern to match "key": "value" or "key": value
    pattern = r'"(\w+)"\s*:\s*(?:"([^"]*)"|(\d+|true|false|null))'
    matches = re.findall(pattern, text)
    for match in matches:
        key = match[0]
        if match[1]:  # String value
            result[key] = match[1]
        elif match[2]:  # Non-string value
            val = match[2]
            if val == "true":
                result[key] = True
            elif val == "false":
                result[key] = False
            elif val == "null":
                result[key] = None
            elif val.isdigit():
                result[key] = int(val)
            else:
                result[key] = val
    return result if result else None


def _fix_json_string(text: str) -> str:
    """Apply common JSON fixes: remove trailing commas and normalize quotes."""
    # Remove trailing commas before closing braces/brackets
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    # Fix single quotes to double quotes (basic)
    text = text.replace("'", '"')
    return text


def _extract_outermost_json(text: str) -> dict | None:
    """Extract the outermost JSON object from text, handling nesting."""
    brace_count = 0
    start_idx = -1
    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                try:
                    return json.loads(text[start_idx:i+1])
                except json.JSONDecodeError:
                    continue
    return None


