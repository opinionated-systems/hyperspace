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
    Also handles nested JSON objects within the tags and multiple JSON objects.
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
        
        # Try to parse the inner content as JSON
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # Try to find JSON object boundaries by brace counting
            # This handles cases where there might be extra text before/after the JSON
            # Also handles multiple JSON objects in the same block
            brace_count = 0
            json_start = -1
            found_in_block = []
            for i, char in enumerate(inner):
                if char == '{':
                    if brace_count == 0:
                        json_start = i
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and json_start != -1:
                        try:
                            obj = json.loads(inner[json_start:i+1])
                            found_in_block.append(obj)
                        except json.JSONDecodeError:
                            continue
                        json_start = -1
            
            if found_in_block:
                results.extend(found_in_block)
            else:
                # If no valid JSON found via brace counting, skip this block
                continue
    return results or None


def _extract_jsons_fast(text: str) -> list[dict] | None:
    """Fast path JSON extraction for common cases.
    
    Optimized for single, well-formed JSON objects in <json> tags.
    Falls back to _extract_jsons for complex cases.
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
        
        # Fast path: try direct JSON parse first
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # Complex case: delegate to full parser
            return _extract_jsons(text)
    return results or None


def _extract_any_json(text: str) -> list[dict] | None:
    """Fallback JSON extraction that looks for any JSON objects in text."""
    results = []
    # Try to find JSON objects between curly braces
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
                    obj = json.loads(text[start_idx:i+1])
                    results.append(obj)
                except json.JSONDecodeError:
                    pass
                start_idx = -1
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

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
        # Extract task components for better prompting
        domain = inputs.get("domain", "Unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematical grader for International Mathematical Olympiad (IMO) problems.

Your task is to evaluate a student's answer to a mathematics problem and provide a grade with detailed reasoning.

DOMAIN: {domain}

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

Think step by step and provide a thorough analysis:
1. PROBLEM ANALYSIS: Identify the key mathematical concepts, theorems, and techniques required
2. SOLUTION REVIEW: Analyze the official solution's approach, key steps, and expected answer format
3. STUDENT WORK ANALYSIS: 
   - Identify what approach the student took
   - Note any correct steps or valid insights
   - Identify errors, gaps, or misconceptions
4. GRADING CRITERIA CHECK:
   - Verify if the student met each criterion in the grading guidelines
   - Note partial credit for incomplete but valid reasoning
5. FINAL DETERMINATION: Assign grade based on completeness, correctness, and adherence to guidelines

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis covering all 5 steps above",
    "response": "The final grade/prediction (e.g., 'Correct', 'Incorrect', 'Partial', or a numeric score)"
}}
</json>

Important: Be objective and consistent. Award partial credit when the student shows valid reasoning even if the final answer is incorrect."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with optimized fallback mechanisms
        prediction = "None"
        try:
            last_message = msg_history[-1]["text"]
            
            # Try fast path extraction first (optimized for common case)
            extracted = _extract_jsons_fast(last_message)
            if extracted:
                self.log_fn(f"Fast extraction found {len(extracted)} JSON objects")
            else:
                # Fallback to full extraction for complex cases
                extracted = _extract_jsons(last_message)
                self.log_fn(f"Full extraction found {len(extracted) if extracted else 0} JSON objects")
                
                # Final fallback: look for any JSON in text
                if extracted is None:
                    extracted = _extract_any_json(last_message)
                    self.log_fn(f"Generic extraction found {len(extracted) if extracted else 0} JSON objects")
            
            if extracted:
                # Prefer response field, but accept other common field names
                last_json = extracted[-1]
                self.log_fn(f"Using JSON with keys: {list(last_json.keys())}")
                
                # Priority order for prediction fields (most common first)
                priority_fields = ["response", "grade", "answer", "result", "evaluation", "prediction", "score"]
                found = False
                for field in priority_fields:
                    if field in last_json:
                        prediction = last_json[field]
                        found = True
                        break
                
                if not found:
                    # If no known field, use the first string or numeric value found
                    for key, value in last_json.items():
                        if isinstance(value, str):
                            prediction = value
                            self.log_fn(f"Using first string value from key '{key}'")
                            break
                        elif isinstance(value, (int, float)):
                            prediction = str(value)
                            self.log_fn(f"Using first numeric value from key '{key}'")
                            break
            else:
                self.log_fn("No JSON objects found in response")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
