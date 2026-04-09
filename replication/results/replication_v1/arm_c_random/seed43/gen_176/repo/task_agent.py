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


def _extract_json_flexible(text: str) -> list[dict] | None:
    """Extract JSON objects with flexible parsing for various formats.
    
    Tries multiple strategies:
    1. Standard <json>...</json> blocks
    2. Markdown code blocks with json
    3. Raw JSON objects in text
    """
    results = []
    
    # Strategy 1: Standard <json> tags
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
    
    # Strategy 2: Markdown code blocks
    md_pattern = r'```json\s*(.*?)\s*```'
    for match in re.finditer(md_pattern, text, re.DOTALL):
        try:
            results.append(json.loads(match.group(1).strip()))
        except json.JSONDecodeError:
            continue
    
    # Strategy 3: Raw JSON objects (look for { followed by "response" or "reasoning")
    # Find all potential JSON objects
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
                    potential_json = text[start_idx:i+1]
                    parsed = json.loads(potential_json)
                    # Only include if it has expected keys
                    if "response" in parsed or "reasoning" in parsed:
                        results.append(parsed)
                except (json.JSONDecodeError, ValueError):
                    pass
                start_idx = -1
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _build_prompt(self, inputs: dict) -> str:
        """Build a structured prompt for IMO grading with chain-of-thought reasoning."""
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        return f"""You are an expert IMO (International Mathematical Olympiad) grader.

Your task is to evaluate a student's solution to a mathematical problem.

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

## Instructions

1. First, analyze the student's answer step by step. Compare it against the official solution.
2. Identify any errors, missing steps, or creative alternative approaches.
3. Consider the grading guidelines carefully.
4. Provide your reasoning before giving the final grade.

## Output Format (IMPORTANT)

You MUST respond using EXACTLY this format with the <json> tags:

<json>
{{
    "reasoning": "Your detailed analysis of the student's answer...",
    "response": "Your final grade/evaluation (e.g., '7', '6', 'Partial credit: 3', 'Incorrect', etc.)"
}}
</json>

Requirements:
- The "response" field must contain ONLY the final grade/evaluation as a string
- The "reasoning" field contains your full analysis
- Use double quotes for all JSON strings
- Do not include any text outside the <json> tags
- The response will be automatically extracted, so format it exactly as shown"""

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with enhanced reasoning.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_prompt(inputs)

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with flexible extraction
        prediction = "None"
        reasoning = ""
        try:
            # Try to extract from the last assistant message
            last_msg = msg_history[-1]["text"] if msg_history else ""
            
            # First try flexible extraction for various JSON formats
            extracted = _extract_json_flexible(last_msg)
            
            if extracted:
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = last_json["response"]
                if "reasoning" in last_json:
                    reasoning = last_json["reasoning"]
                
                # Log the reasoning for debugging
                if reasoning:
                    self.log_fn(f"Reasoning: {reasoning[:200]}...")
            else:
                # Final fallback: try to extract any grade-like pattern
                # Look for patterns like "Grade: 7", "Score: 6", "Final grade: 5", etc.
                grade_patterns = [
                    r'(?:grade|score|final grade|mark|rating)["\']?\s*[:=]\s*["\']?([^"\'\n]+)',
                    r'(?:the grade is|grade is|score is)["\']?\s*[:=]?\s*["\']?([^"\'\n]+)',
                ]
                for pattern in grade_patterns:
                    match = re.search(pattern, last_msg, re.IGNORECASE)
                    if match:
                        prediction = match.group(1).strip()
                        break
                        
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
