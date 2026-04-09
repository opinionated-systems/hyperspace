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
from typing import Any

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles nested JSON objects within the content.
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
        
        # Try direct parsing first
        try:
            results.append(json.loads(inner))
            continue
        except json.JSONDecodeError:
            pass
        
        # Try to fix common JSON issues
        # 1. Remove trailing commas before closing braces/brackets
        fixed = re.sub(r',(\s*[}\]])', r'\1', inner)
        # 2. Fix single quotes to double quotes (carefully)
        fixed = re.sub(r"(?<!\\)'", '"', fixed)
        
        try:
            results.append(json.loads(fixed))
            logger.debug(f"Fixed and parsed JSON at position {start}")
        except json.JSONDecodeError as e:
            logger.debug(f"JSON decode error at position {start}: {e}")
            continue
    return results or None


def _extract_json_fallback(text: str) -> dict | None:
    """Fallback method to extract JSON using regex patterns.
    
    Used when primary extraction fails to find valid JSON.
    Includes multiple strategies and attempts to fix common JSON issues.
    """
    # Strategy 1: Try to find JSON in code blocks
    code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            # Try fixing common issues
            fixed = re.sub(r',(\s*[}\]])', r'\1', match.strip())
            fixed = re.sub(r"(?<!\\)'", '"', fixed)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                continue
    
    # Strategy 2: Try to find JSON between curly braces (non-greedy)
    brace_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(brace_pattern, text, re.DOTALL)
    
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            # Try fixing common issues
            fixed = re.sub(r',(\s*[}\]])', r'\1', match)
            fixed = re.sub(r"(?<!\\)'", '"', fixed)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                continue
    
    # Strategy 3: Look for the largest valid JSON object
    # This handles cases where the regex might have matched incomplete JSON
    start_idx = text.find('{')
    while start_idx != -1:
        # Try progressively larger substrings
        for end_idx in range(len(text), start_idx, -1):
            candidate = text[start_idx:end_idx]
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
        start_idx = text.find('{', start_idx + 1)
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.log_file = log_file

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Build a more structured prompt with clear instructions
        instruction = f"""You are an expert grading agent. Your task is to evaluate student answers based on the provided problem, solution, and grading guidelines.

Task input:
```
Domain: {inputs.get('domain', 'N/A')}
Problem: {inputs.get('problem', 'N/A')}
Solution: {inputs.get('solution', 'N/A')}
Grading Guidelines: {inputs.get('grading_guidelines', 'N/A')}
Student Answer: {inputs.get('student_answer', 'N/A')}
```

Your task is to evaluate the student's answer and provide a grade or assessment.

Respond in JSON format with the following schema:
<json>
{{
    "response": "Your detailed evaluation and grade here"
}}
</json>

Important: 
- Ensure your response is valid JSON wrapped in <json> tags
- The "response" field should contain your complete evaluation
- Be thorough and fair in your assessment"""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return "Error: LLM call failed", []

        # Extract prediction from JSON with improved error handling
        prediction = "None"
        extraction_method = "none"
        
        try:
            if msg_history and len(msg_history) > 0:
                last_message = msg_history[-1].get("text", "")
                
                # Primary extraction: <json> tags
                extracted = _extract_jsons(last_message)
                if extracted:
                    for item in extracted:
                        if isinstance(item, dict) and "response" in item:
                            prediction = item["response"]
                            extraction_method = "primary"
                            break
                
                # Fallback extraction if primary failed
                if extraction_method == "none":
                    fallback = _extract_json_fallback(last_message)
                    if fallback and isinstance(fallback, dict) and "response" in fallback:
                        prediction = fallback["response"]
                        extraction_method = "fallback"
                
                # Log extraction results
                if extraction_method != "none":
                    logger.info(f"Successfully extracted prediction using {extraction_method} method")
                else:
                    logger.warning(f"No valid JSON found in response")
                    # Try to extract any text as a last resort
                    if last_message:
                        # Remove common formatting
                        cleaned = re.sub(r'<json>|</json>|```json|```', '', last_message).strip()
                        if cleaned:
                            prediction = cleaned[:1000]  # Limit length
                            extraction_method = "raw_text"
                            logger.info("Used raw text extraction as last resort")
                    
                    if self.log_file:
                        self.log_fn(f"Raw response: {last_message[:500]}...")
            else:
                logger.warning("Empty message history received")
        except Exception as e:
            logger.error(f"Error extracting prediction: {e}")
            if self.log_file:
                self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
