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
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles markdown code blocks and common formatting issues.
    Includes robust fallback mechanisms for malformed JSON.
    """
    results = []
    search_from = 0

    # First, try to find <json>...</json> blocks
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7

        # Try to parse the JSON, handling common formatting issues
        parsed = _try_parse_json(inner)
        if parsed is not None:
            results.append(parsed)

    # If no <json> blocks found, try to find JSON in markdown code blocks
    if not results:
        search_from = 0
        while True:
            start = text.find("```json", search_from)
            if start == -1:
                # Try generic code blocks
                start = text.find("```", search_from)
                if start == -1:
                    break
                code_start = start + 3
            else:
                code_start = start + 7

            end = text.find("```", code_start)
            if end == -1:
                break

            inner = text[code_start:end].strip()
            search_from = end + 3

            parsed = _try_parse_json(inner)
            if parsed is not None:
                results.append(parsed)

    # Final fallback: try to find any JSON-like structure in the text
    if not results:
        parsed = _extract_json_heuristic(text)
        if parsed is not None:
            results.append(parsed)

    return results or None


def _try_parse_json(text: str) -> dict | None:
    """Try to parse JSON with various cleaning strategies.
    
    Args:
        text: The text to parse
        
    Returns:
        Parsed dict or None if parsing fails
    """
    # Remove markdown code block markers if present
    cleaned = text
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    
    # Try direct parsing first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Try to extract just the first valid JSON object
    try:
        json_start = cleaned.find("{")
        json_end = cleaned.rfind("}")
        if json_start != -1 and json_end != -1 and json_end > json_start:
            return json.loads(cleaned[json_start:json_end + 1])
    except json.JSONDecodeError:
        pass
    
    # Try to fix common JSON errors
    try:
        # Fix single quotes to double quotes
        fixed = cleaned.replace("'", '"')
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    return None


def _extract_json_heuristic(text: str) -> dict | None:
    """Extract JSON using heuristics when standard parsing fails.
    
    Looks for key-value patterns that might indicate a JSON structure.
    
    Args:
        text: The text to search
        
    Returns:
        Extracted dict or None
    """
    # Look for patterns like "response": "..." or 'response': '...'
    response_match = re.search(
        r'["\']?response["\']?\s*:\s*["\']([^"\']+)["\']',
        text,
        re.IGNORECASE
    )
    if response_match:
        return {"response": response_match.group(1)}
    
    # Look for any quoted string that might be the response
    # This is a last resort fallback
    lines = text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('<') and not line.startswith('```'):
            # Try to find a meaningful response
            if len(line) > 0 and len(line) < 1000:
                return {"response": line}
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with robust JSON extraction."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Build a more structured prompt for better responses
        instruction = self._build_prompt(inputs)

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with multiple fallback strategies
        prediction = self._extract_prediction(msg_history)

        return str(prediction), msg_history

    def _build_prompt(self, inputs: dict) -> str:
        """Build a structured prompt for the task.
        
        Args:
            inputs: Task inputs dictionary
            
        Returns:
            Formatted prompt string
        """
        # Extract key fields for better context
        domain = inputs.get("domain", "Unknown")
        problem = inputs.get("problem", "")
        
        prompt = f"""You are an expert grading agent for {domain} problems.

Your task is to evaluate a student's answer based on the provided problem, solution, and grading guidelines.

PROBLEM:
{problem}

INPUT DATA:
```json
{inputs}
```

You must respond in valid JSON format wrapped in <json> tags:
<json>
{{
    "response": "Your evaluation/grade here"
}}
</json>

Important:
- The response field should contain your complete evaluation
- Ensure valid JSON syntax with double quotes
- Wrap your entire JSON response in <json>...</json> tags"""
        
        return prompt

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history with robust error handling.
        
        Args:
            msg_history: List of message dictionaries
            
        Returns:
            Extracted prediction string or "None"
        """
        if not msg_history:
            self.log_fn("Warning: Empty message history")
            return "None"
        
        # Get the last assistant message
        last_msg = None
        for msg in reversed(msg_history):
            if msg.get("role") == "assistant":
                last_msg = msg
                break
        
        if last_msg is None:
            self.log_fn("Warning: No assistant message found in history")
            return "None"
        
        text = last_msg.get("text", "")
        if not text:
            self.log_fn("Warning: Empty text in assistant message")
            return "None"
        
        # Try to extract JSON
        try:
            extracted = _extract_jsons(text)
            if extracted:
                last_extracted = extracted[-1]
                if isinstance(last_extracted, dict) and "response" in last_extracted:
                    prediction = last_extracted["response"]
                    self.log_fn(f"Successfully extracted prediction: {str(prediction)[:100]}...")
                    return prediction
                else:
                    self.log_fn(f"Warning: Extracted JSON missing 'response' key: {last_extracted}")
                    # Try to use the entire extracted object as response
                    return str(last_extracted)
            else:
                self.log_fn(f"Warning: No JSON found in response, using raw text")
                # Fallback: return the raw text (truncated)
                return text[:1000] if len(text) > 1000 else text
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Last resort: return a portion of the raw text
            return text[:500] if len(text) > 500 else text
