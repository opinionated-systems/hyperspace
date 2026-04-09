"""
Task agent: solves a given task with a single LLM call.

Reimplemented from facebookresearch/HyperAgents task_agent.py.
Same interface, same JSON output format, same extraction logic.

This is the INITIAL task agent. The meta agent modifies this file
during self-improvement. The evaluation harness loads whatever
task_agent.py exists at the agent's repo path.

Enhanced with better JSON extraction and error handling.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from agent.llm_client import get_response_from_llm, EVAL_MODEL
from agent.utils import retry, truncate_string

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


def _extract_json_flexible(text: str) -> dict | None:
    """Extract JSON with more flexible parsing.
    
    Tries multiple strategies:
    1. Look for <json>...</json> blocks
    2. Look for ```json code blocks
    3. Look for JSON objects directly in the text
    """
    # Strategy 1: <json> tags
    results = _extract_jsons(text)
    if results:
        return results[-1]
    
    # Strategy 2: ```json code blocks
    json_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    for match in re.finditer(json_block_pattern, text, re.DOTALL):
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            continue
    
    # Strategy 3: Look for JSON objects directly
    # Match content between { and } (balanced)
    brace_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    for match in re.finditer(brace_pattern, text, re.DOTALL):
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self._call_count = 0
        self._success_count = 0

    def _build_prompt(self, inputs: dict) -> str:
        """Build the prompt for the task."""
        # Truncate long inputs with smart truncation
        truncated_inputs = {}
        for key, value in inputs.items():
            if isinstance(value, str) and len(value) > 5000:
                # Keep beginning and end, truncate middle
                truncated_inputs[key] = truncate_string(value, 5000)
            else:
                truncated_inputs[key] = value
        
        # Build domain-specific instructions
        domain = inputs.get("domain", "general")
        domain_instructions_map = {
            "math": "Focus on mathematical correctness, logical reasoning, and proper notation.",
            "physics": "Focus on physical principles, unit consistency, and dimensional analysis.",
            "chemistry": "Focus on chemical equations, stoichiometry, and reaction mechanisms.",
            "general": "Focus on accuracy, completeness, and clarity of the answer.",
        }
        domain_instruction = domain_instructions_map.get(domain, domain_instructions_map["general"])
        
        return f"""You are an expert grading agent for {domain} problems.

Your task is to evaluate a student's answer based on the provided problem, solution, and grading guidelines.

Task input:
```json
{json.dumps(truncated_inputs, indent=2)}
```

Instructions:
1. Carefully read the problem, solution, and grading guidelines
2. {domain_instruction}
3. Evaluate the student's answer against the criteria
4. Provide your assessment in the exact JSON format below

Respond in JSON format with the following schema:
<json>
{{
    "response": "Your detailed evaluation and grade here"
}}
</json>

Important: Your response MUST be valid JSON wrapped in <json> tags.
- Ensure the JSON is properly formatted with no syntax errors
- The response field should contain your complete evaluation"""

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        self._call_count += 1
        
        instruction = self._build_prompt(inputs)

        # Retry logic for LLM calls
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                break
            except Exception as e:
                logger.warning(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    logger.error(f"LLM call failed after {max_retries} attempts: {e}")
                    return f"Error: LLM call failed - {e}", []
                import time
                time.sleep(2 ** attempt)  # Exponential backoff

        # Extract prediction from JSON
        prediction = "None"
        try:
            # Try flexible extraction first
            extracted = _extract_json_flexible(msg_history[-1]["text"])
            if extracted and "response" in extracted:
                prediction = extracted["response"]
                self._success_count += 1
            else:
                # Fallback: try to extract any meaningful content
                text = msg_history[-1]["text"]
                # Remove common wrappers
                for wrapper in ["<json>", "</json>", "```json", "```"]:
                    text = text.replace(wrapper, "")
                prediction = text.strip()[:1000]  # Limit length
                
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Use raw response as fallback
            try:
                prediction = msg_history[-1]["text"][:1000]
            except (IndexError, KeyError):
                prediction = "Error: Could not extract prediction"

        return str(prediction), msg_history

    def get_stats(self) -> dict[str, Any]:
        """Get agent statistics."""
        return {
            "total_calls": self._call_count,
            "successful_extractions": self._success_count,
            "extraction_rate": self._success_count / self._call_count if self._call_count > 0 else 0,
        }

    def reset_stats(self) -> None:
        """Reset agent statistics."""
        self._call_count = 0
        self._success_count = 0
