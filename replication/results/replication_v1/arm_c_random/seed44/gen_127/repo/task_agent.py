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
    Includes multiple fallback strategies for robust extraction.
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
        
        # Try multiple parsing strategies
        parsed = None
        
        # Strategy 1: Direct JSON parsing
        try:
            parsed = json.loads(inner)
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Fix trailing commas
        if parsed is None:
            try:
                fixed = re.sub(r',(\s*[}\]])', r'\1', inner)
                parsed = json.loads(fixed)
            except json.JSONDecodeError:
                pass
        
        # Strategy 3: Fix single quotes to double quotes
        if parsed is None:
            try:
                fixed = inner.replace("'", '"')
                parsed = json.loads(fixed)
            except json.JSONDecodeError:
                pass
        
        # Strategy 4: Extract just the value if it's a simple string response
        if parsed is None:
            try:
                # Look for "response": "..." pattern with flexible quoting
                match = re.search(r'["\']?response["\']?\s*:\s*["\']([^"\']+)["\']', inner)
                if match:
                    parsed = {"response": match.group(1)}
            except Exception:
                pass
        
        # Strategy 5: Try to find any key-value pair as fallback
        if parsed is None:
            try:
                # Look for the last number or quoted string in the content
                number_match = re.search(r'\b([0-7])\b', inner)
                if number_match:
                    parsed = {"response": number_match.group(1)}
            except Exception:
                pass
        
        if parsed is not None:
            results.append(parsed)
    
    return results or None


def _extract_grade_from_text(text: str) -> str | None:
    """Extract a grade from plain text when JSON extraction fails.
    
    Looks for common grade patterns like numbers 0-7, or phrases like
    'full credit', 'partial credit', 'no credit', etc.
    """
    # Look for IMO-style numeric grades (0-7)
    numeric_pattern = r'\b([0-7])\s*(?:/\s*7)?\b'
    matches = re.findall(numeric_pattern, text)
    if matches:
        return matches[-1]  # Return last match (usually the final grade)
    
    # Look for text-based grades
    text_patterns = [
        (r'\bfull\s+credit\b', '7'),
        (r'\bno\s+credit\b', '0'),
        (r'\bzero\b', '0'),
        (r'\bpartial\s+credit\b', 'Partial credit'),
        (r'\bincomplete\b', 'Partial credit'),
        (r'\bcorrect\b', '7'),
        (r'\bincorrect\b', '0'),
    ]
    
    for pattern, grade in text_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return grade
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _validate_inputs(self, inputs: dict) -> tuple[bool, str]:
        """Validate that all required fields are present in inputs.
        
        Returns:
            (is_valid, error_message)
        """
        required_fields = ["domain", "problem", "solution", "grading_guidelines", "student_answer"]
        missing = [f for f in required_fields if f not in inputs]
        if missing:
            return False, f"Missing required fields: {missing}"
        return True, ""

    def _build_grading_prompt(self, inputs: dict) -> str:
        """Build a structured prompt for the IMO grading task."""
        return f"""You are an expert mathematics grader for the International Mathematical Olympiad (IMO).

Your task is to evaluate a student's solution to a mathematical problem and provide a grade.

PROBLEM DOMAIN: {inputs['domain']}

PROBLEM STATEMENT:
{inputs['problem']}

OFFICIAL SOLUTION:
{inputs['solution']}

GRADING GUIDELINES:
{inputs['grading_guidelines']}

STUDENT'S ANSWER TO EVALUATE:
{inputs['student_answer']}

Instructions:
1. Carefully read the problem, official solution, and grading guidelines
2. Analyze the student's answer for correctness and completeness
3. Compare the student's approach with the official solution
4. Assign an appropriate grade based on the grading guidelines
5. Provide your grade in the JSON format below

IMO grades are typically integers from 0 to 7, where:
- 7 = Full credit (complete and correct solution)
- 6 = Minor flaw in an otherwise correct solution
- 1-5 = Partial credit based on progress made
- 0 = No credit (no meaningful progress or completely wrong)

Respond ONLY in JSON format with the following schema:
<json>
{{
    "response": "Your grade here (e.g., '7', '6', '0', 'Partial credit', etc.)"
}}
</json>

Important: 
- The response field must contain only the grade value, without additional explanation
- Use standard IMO numeric grades (0-7) when possible
- Wrap your entire JSON response in <json>...</json> tags"""

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Validate inputs
        is_valid, error_msg = self._validate_inputs(inputs)
        if not is_valid:
            self.log_fn(f"Input validation failed: {error_msg}")
            return f"Error: {error_msg}", []

        # Build structured prompt
        instruction = self._build_grading_prompt(inputs)

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with improved error handling
        prediction = "None"
        try:
            if msg_history and len(msg_history) > 0:
                response_text = msg_history[-1]["text"]
                extracted = _extract_jsons(response_text)
                if extracted:
                    last_extract = extracted[-1]
                    if isinstance(last_extract, dict) and "response" in last_extract:
                        prediction = last_extract["response"]
                        # Validate prediction is a string or number
                        if not isinstance(prediction, (str, int, float)):
                            prediction = str(prediction)
                    else:
                        self.log_fn(f"No 'response' key found in extracted JSON: {last_extract}")
                        # Try to extract from the raw text as fallback
                        text_grade = _extract_grade_from_text(response_text)
                        if text_grade:
                            prediction = text_grade
                            self.log_fn(f"Extracted grade from text fallback: {prediction}")
                else:
                    self.log_fn("No JSON blocks found in response, trying text extraction")
                    # Try to extract from the raw text as fallback
                    text_grade = _extract_grade_from_text(response_text)
                    if text_grade:
                        prediction = text_grade
                        self.log_fn(f"Extracted grade from text fallback: {prediction}")
            else:
                self.log_fn("Empty message history")
        except json.JSONDecodeError as e:
            self.log_fn(f"JSON decode error: {e}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
