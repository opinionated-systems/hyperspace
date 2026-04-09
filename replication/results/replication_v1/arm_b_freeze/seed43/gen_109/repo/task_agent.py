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


def _extract_brace_json(text: str) -> list[dict] | None:
    """Extract JSON objects by tracking brace balance.
    
    More robust than regex for nested structures.
    """
    results = []
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
                    if isinstance(obj, dict):
                        results.append(obj)
                except json.JSONDecodeError:
                    pass
                start_idx = -1
    
    return results or None


def _extract_markdown_json(text: str) -> list[dict] | None:
    """Extract JSON from markdown code blocks."""
    results = []
    # Match ```json or ``` blocks
    pattern = r'```(?:json)?\s*([\s\S]*?)```'
    matches = re.findall(pattern, text)
    
    for match in matches:
        # Try parsing the whole block first
        try:
            obj = json.loads(match.strip())
            if isinstance(obj, dict):
                results.append(obj)
                continue
        except json.JSONDecodeError:
            pass
        
        # If that fails, try extracting JSON objects from within
        nested = _extract_brace_json(match)
        if nested:
            results.extend(nested)
    
    return results or None


def _extract_all_json(text: str) -> list[dict]:
    """Extract all possible JSON objects using multiple strategies.
    
    Returns combined results from all extraction methods.
    """
    all_results = []
    seen = set()
    
    # Helper to add unique results
    def add_unique(results: list[dict] | None) -> None:
        if not results:
            return
        for obj in results:
            # Use string representation for deduplication
            obj_str = json.dumps(obj, sort_keys=True)
            if obj_str not in seen:
                seen.add(obj_str)
                all_results.append(obj)
    
    # Try all extraction methods
    add_unique(_extract_jsons(text))
    add_unique(_extract_markdown_json(text))
    add_unique(_extract_brace_json(text))
    
    return all_results


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    # Priority order for extracting prediction from JSON fields
    PREDICTION_FIELDS = [
        "response", "grade", "answer", "result", 
        "evaluation", "prediction", "score", "verdict"
    ]

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _extract_prediction(self, text: str) -> str:
        """Extract prediction from text using all available JSON extraction methods.
        
        Args:
            text: The text to extract prediction from
            
        Returns:
            The extracted prediction string, or "None" if extraction fails
        """
        # Try all extraction methods
        extracted = _extract_all_json(text)
        
        if not extracted:
            self.log_fn("No JSON objects found in response")
            return "None"
        
        # Use the last JSON object (most likely to be the final answer)
        last_json = extracted[-1]
        
        # Try known field names in priority order
        for field in self.PREDICTION_FIELDS:
            if field in last_json:
                value = last_json[field]
                if isinstance(value, str):
                    return value
                elif isinstance(value, (int, float, bool)):
                    return str(value)
        
        # If no known field, look for any string or numeric value
        for key, value in last_json.items():
            if isinstance(value, str):
                return value
            elif isinstance(value, (int, float)):
                return str(value)
        
        self.log_fn(f"Could not extract prediction from JSON: {last_json}")
        return "None"

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

        # Build a more structured and effective prompt
        instruction = self._build_grading_prompt(
            domain=domain,
            problem=problem,
            solution=solution,
            grading_guidelines=grading_guidelines,
            student_answer=student_answer
        )

        # Try up to 3 times to get a valid JSON response
        max_retries = 3
        prediction = "None"
        msg_history = []
        
        for attempt in range(max_retries):
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )

            # Extract prediction from the last assistant message
            try:
                if msg_history and len(msg_history) > 0:
                    last_message = msg_history[-1].get("text", "")
                    prediction = self._extract_prediction(last_message)
                    
                    # If we got a valid prediction (not "None"), break the retry loop
                    if prediction != "None":
                        break
                    
                    # If this wasn't the last attempt, add feedback for retry
                    if attempt < max_retries - 1:
                        self.log_fn(f"Attempt {attempt + 1}: Failed to extract valid JSON, retrying...")
                        instruction += "\n\nNOTE: Your previous response did not contain valid JSON in <json> tags. Please ensure your response follows the exact format specified above."
            except Exception as e:
                self.log_fn(f"Error extracting prediction on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    prediction = "None"

        return str(prediction), msg_history

    def _build_grading_prompt(
        self,
        domain: str,
        problem: str,
        solution: str,
        grading_guidelines: str,
        student_answer: str
    ) -> str:
        """Build a structured grading prompt with clear instructions."""
        return f"""You are an expert mathematical grader for International Mathematical Olympiad (IMO) problems.

Your task is to evaluate a student's answer to a mathematics problem and provide a grade.

=== PROBLEM INFORMATION ===
DOMAIN: {domain}

PROBLEM STATEMENT:
{problem}

=== REFERENCE MATERIALS ===
OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

=== STUDENT WORK ===
STUDENT'S ANSWER:
{student_answer}

=== EVALUATION INSTRUCTIONS ===
Think step by step and provide your analysis:

1. PROBLEM ANALYSIS
   - Identify the key mathematical concepts and techniques required
   - Note any critical theorems or methods that must be used

2. SOLUTION REVIEW
   - Understand the official solution's approach and key steps
   - Identify what constitutes a complete and correct solution

3. STUDENT ANSWER EVALUATION
   - Check mathematical correctness of each step
   - Verify logical flow and reasoning validity
   - Assess completeness (all cases covered? all steps justified?)
   - Evaluate clarity and rigor of presentation

4. GRADING DECISION
   - Apply the grading guidelines precisely
   - Consider partial credit where appropriate
   - Provide clear justification for the grade

=== RESPONSE FORMAT ===
IMPORTANT: Your response MUST be valid JSON wrapped in <json> tags.

<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis. Explain: (1) what the problem requires, (2) how the official solution works, (3) what the student did right/wrong, (4) why you assigned this grade.",
    "response": "The final grade (e.g., 'Correct', 'Incorrect', 'Partial', or a numeric score as specified in guidelines)"
}}
</json>

Do not include any text outside the JSON tags."""
