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
    """
    results = []
    search_from = 0
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            continue
    return results or None


def _extract_json_with_retry(text: str, max_retries: int = 3) -> list[dict] | None:
    """Extract JSON with multiple fallback strategies.
    
    First tries standard extraction, then attempts to fix common
    JSON formatting issues like trailing commas, missing quotes,
    unescaped special characters, and malformed JSON blocks.
    """
    # First attempt: standard extraction
    result = _extract_jsons(text)
    if result:
        return result
    
    # Retry with progressively more aggressive fixes
    fixes = [
        # Fix 1: Basic fixes - trailing commas and comments
        lambda t: re.sub(r',(\s*[}\]])', r'\1', 
                re.sub(r'//.*?\n', '\n', 
                re.sub(r'/\*.*?\*/', '', t, flags=re.DOTALL)),
        # Fix 2: Quote normalization (handle single quotes)
        lambda t: re.sub(r"(?<!\\)'", '"', t),
        # Fix 3: Handle unescaped newlines in strings
        lambda t: re.sub(r'(?<=")([^"]*\n[^"]*)"', lambda m: m.group(0).replace('\n', '\\n'), t),
    ]
    
    for attempt, fix_func in enumerate(fixes[:max_retries]):
        try:
            fixed_text = fix_func(text)
            
            result = _extract_jsons(fixed_text)
            if result:
                logger.debug(f"JSON extraction succeeded after fix attempt {attempt + 1}")
                return result
        except Exception as e:
            logger.debug(f"Fix attempt {attempt + 1} failed: {e}")
            continue
    
    # Final attempt: Try to extract any JSON-like structure
    try:
        # Look for content between curly braces
        brace_pattern = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if brace_pattern:
            potential_json = brace_pattern.group(0)
            # Clean up common issues
            potential_json = re.sub(r',(\s*[}\]])', r'\1', potential_json)
            potential_json = re.sub(r'\n\s*}', '}', potential_json)
            parsed = json.loads(potential_json)
            logger.debug("JSON extraction succeeded with brace extraction fallback")
            return [parsed]
    except Exception as e:
        logger.debug(f"Final brace extraction fallback failed: {e}")
    
    # Log the problematic text for debugging (truncated)
    preview = text[:500] + "..." if len(text) > 500 else text
    logger.warning(f"Failed to extract JSON from text after {max_retries} attempts: {preview}")
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems."""

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
        # Validate required input fields
        required_fields = ["domain", "problem", "solution", "grading_guidelines", "student_answer"]
        missing_fields = [f for f in required_fields if f not in inputs]
        if missing_fields:
            logger.warning(f"Missing input fields: {missing_fields}")
        
        instruction = f"""You are an expert grading agent. Your task is to evaluate a student's answer to a problem.

Task input:
```
{inputs}
```

Respond in JSON format with the following schema:
<json>
{{
    "response": "Your evaluation result here"
}}
</json>

Important: Ensure your response is valid JSON with double quotes around keys and string values."""

        self.log_fn(f"Processing task with model: {self.model}")
        
        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
            
            # Log token usage if available
            usage = info.get("usage", {})
            if usage:
                self.log_fn(f"Token usage - Prompt: {usage.get('prompt_tokens', 'N/A')}, "
                          f"Completion: {usage.get('completion_tokens', 'N/A')}, "
                          f"Total: {usage.get('total_tokens', 'N/A')}")
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return "Error: LLM call failed", []

        # Extract prediction from JSON with retry mechanism
        prediction = "None"
        try:
            if msg_history and len(msg_history) > 0:
                last_message = msg_history[-1]
                text_content = last_message.get("text", "")
                
                extracted = _extract_json_with_retry(text_content)
                if extracted:
                    last_json = extracted[-1]
                    if "response" in last_json:
                        prediction = last_json["response"]
                        self.log_fn(f"Successfully extracted prediction: {str(prediction)[:100]}...")
                    else:
                        logger.warning(f"JSON missing 'response' key. Keys found: {list(last_json.keys())}")
                        prediction = str(last_json)
                else:
                    logger.warning("No valid JSON found in response")
                    # Fallback: return raw text if no JSON found
                    prediction = text_content[:500] if text_content else "None"
            else:
                logger.warning("Empty message history from LLM")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            logger.exception("Full traceback:")

        return str(prediction), msg_history
