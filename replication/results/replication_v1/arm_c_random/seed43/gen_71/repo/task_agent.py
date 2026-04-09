"""
Task agent: solves a given task with enhanced reasoning for IMO grading.

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
from agent.utils import validate_score, truncate_text

logger = logging.getLogger(__name__)


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles markdown code blocks (```json...```) as fallback.
    Includes additional heuristics for common LLM output patterns.
    """
    results = []
    search_from = 0
    
    # First try to find <json>...</json> blocks
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
            # Try to extract JSON from within the content if it's wrapped
            try:
                # Look for JSON object boundaries
                json_start = inner.find("{")
                json_end = inner.rfind("}")
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    results.append(json.loads(inner[json_start:json_end + 1]))
            except json.JSONDecodeError:
                continue
    
    # Second: try markdown code blocks with json specifier
    if not results:
        search_from = 0
        while True:
            start = text.find("```json", search_from)
            if start == -1:
                break
            end = text.find("```", start + 7)
            if end == -1:
                break
            inner = text[start + 7:end].strip()
            search_from = end + 3
            try:
                results.append(json.loads(inner))
            except json.JSONDecodeError:
                # Try to find JSON object boundaries
                json_start = inner.find("{")
                json_end = inner.rfind("}")
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    try:
                        results.append(json.loads(inner[json_start:json_end + 1]))
                    except json.JSONDecodeError:
                        continue
    
    # Third: try plain markdown code blocks
    if not results:
        search_from = 0
        while True:
            start = text.find("```", search_from)
            if start == -1:
                break
            end = text.find("```", start + 3)
            if end == -1:
                break
            inner = text[start + 3:end].strip()
            search_from = end + 3
            try:
                results.append(json.loads(inner))
            except json.JSONDecodeError:
                # Try to find JSON object boundaries
                json_start = inner.find("{")
                json_end = inner.rfind("}")
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    try:
                        results.append(json.loads(inner[json_start:json_end + 1]))
                    except json.JSONDecodeError:
                        continue
    
    # Final fallback: try to find any JSON object in the text using brace matching
    if not results:
        # Look for JSON-like patterns: {...} with proper brace counting
        json_start = text.find("{")
        while json_start != -1:
            # Try to find the matching closing brace by counting
            brace_count = 1
            pos = json_start + 1
            while pos < len(text) and brace_count > 0:
                if text[pos] == '{':
                    brace_count += 1
                elif text[pos] == '}':
                    brace_count -= 1
                pos += 1
            
            if brace_count == 0:
                json_end = pos - 1
                try:
                    candidate = text[json_start:json_end + 1]
                    results.append(json.loads(candidate))
                    break  # Found valid JSON, stop searching
                except json.JSONDecodeError:
                    pass
            
            json_start = text.find("{", json_start + 1)
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with chain-of-thought reasoning.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract fields for better structured prompting
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert IMO (International Mathematical Olympiad) grader. Your task is to evaluate a student's solution to a mathematical problem.

## Problem Domain
{domain}

## Problem Statement
{problem}

## Official Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Your Task

Please evaluate the student's answer following these steps:

1. **Understand the Problem**: Identify what the problem is asking and the key mathematical concepts involved.

2. **Analyze the Official Solution**: Understand the expected approach and the key steps required for a correct solution.

3. **Review Grading Guidelines**: Note the specific criteria for awarding points (partial or full).

4. **Evaluate Student's Answer**: 
   - Check if the student correctly identified the approach
   - Verify each step of their reasoning
   - Identify any errors, gaps, or incorrect assumptions
   - Note any creative or alternative valid approaches

5. **Assign Score**: Based on the grading guidelines, assign an appropriate score. Be precise and justify your decision.

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis here",
    "evaluation": "Summary of what the student did correctly/incorrectly",
    "response": "The final score/grade as a number or string"
}}
</json>

The "response" field should contain only the final score (e.g., "7", "3", "0", etc.) that will be used for evaluation."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        extraction_log = []
        try:
            # Try the last assistant message first
            last_msg = msg_history[-1]
            extracted = _extract_jsons(last_msg["text"])
            extraction_log.append(f"Last message extraction: {len(extracted) if extracted else 0} JSONs found")
            
            if not extracted and len(msg_history) >= 2:
                # Try previous messages if last one doesn't have JSON
                for i in range(len(msg_history) - 2, -1, -1):
                    if msg_history[i]["role"] == "assistant":
                        extracted = _extract_jsons(msg_history[i]["text"])
                        if extracted:
                            extraction_log.append(f"Found JSON in message {i}")
                            break
            
            if extracted:
                # Try to get response from the last valid JSON block
                last_json = extracted[-1]
                extraction_log.append(f"Using JSON with keys: {list(last_json.keys())}")
                
                # Priority order for score fields
                score_fields = ["response", "score", "grade", "evaluation", "result", "answer"]
                for field in score_fields:
                    if field in last_json:
                        raw_value = last_json[field]
                        validated = validate_score(raw_value)
                        if validated is not None:
                            prediction = str(validated)
                            extraction_log.append(f"Extracted and validated '{field}': {raw_value} -> {prediction}")
                        else:
                            prediction = str(raw_value)
                            extraction_log.append(f"Extracted '{field}': {prediction} (validation failed)")
                        break
                else:
                    # If no known field, use the first string/numeric value found
                    for key, value in last_json.items():
                        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
                            validated = validate_score(value)
                            if validated is not None:
                                prediction = str(validated)
                                extraction_log.append(f"Using validated value from '{key}': {value} -> {prediction}")
                            else:
                                prediction = str(value)
                                extraction_log.append(f"Using first valid value from '{key}': {prediction}")
                            break
            else:
                extraction_log.append("No JSON found in any message")
                
        except Exception as e:
            extraction_log.append(f"Error during extraction: {e}")
            self.log_fn(f"Error extracting prediction: {e}")

        # Log extraction details for debugging
        self.log_fn(f"Prediction extraction: {'; '.join(extraction_log)}")
        
        return str(prediction), msg_history
