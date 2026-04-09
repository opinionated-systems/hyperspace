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
    Case-insensitive tag matching for better robustness.
    """
    results = []
    search_from = 0
    text_lower = text.lower()
    
    while True:
        # Case-insensitive search for opening tag
        start = text_lower.find("<json>", search_from)
        if start == -1:
            break
        # Find closing tag (case-insensitive)
        end = text_lower.find("</json>", start)
        if end == -1:
            break
        
        # Extract from original text (not lowercased)
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        # Try to parse JSON
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # Try to fix common JSON issues
            try:
                # Remove trailing commas before closing braces/brackets
                fixed = re.sub(r',(\s*[}\]])', r'\1', inner)
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                continue
    
    return results or None


def _extract_json_with_regex(text: str) -> list[dict] | None:
    """Fallback JSON extraction using regex for malformed outputs."""
    results = []
    
    # Try to find JSON objects in code blocks (case-insensitive)
    code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(code_block_pattern, text, re.DOTALL | re.IGNORECASE)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            # Try fixing common issues
            try:
                fixed = re.sub(r',(\s*[}\]])', r'\1', match.strip())
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                continue
    
    # Try to find JSON objects with curly braces - improved pattern
    if not results:
        # More flexible pattern to catch nested structures with "response" field
        json_pattern = r'\{[\s\S]*?"response"[\s\S]*?\}'
        matches = re.findall(json_pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                results.append(json.loads(match))
            except json.JSONDecodeError:
                # Try fixing common issues
                try:
                    fixed = re.sub(r',(\s*[}\]])', r'\1', match)
                    results.append(json.loads(fixed))
                except json.JSONDecodeError:
                    continue
    
    # Try to find any valid JSON object as last resort
    if not results:
        # Find JSON objects by looking for balanced braces
        brace_pattern = r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}'
        matches = re.findall(brace_pattern, text)
        for match in matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, dict):
                    results.append(parsed)
            except json.JSONDecodeError:
                continue
    
    return results or None


def _extract_response_heuristic(text: str) -> Any | None:
    """Last resort: try to extract a response value using heuristics."""
    # Look for patterns like "response": 7 or "response": "correct"
    # Support both single and double quotes
    patterns = [
        (r'["\']response["\']\s*:\s*(-?\d+(?:\.\d+)?)', 'number'),  # Numbers
        (r'["\']response["\']\s*:\s*["\']([^"\']*)["\']', 'string'),  # Strings
        (r'["\']response["\']\s*:\s*(true|false|null)', 'bool_null'),  # Booleans/null
    ]
    
    for pattern, ptype in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1)
            if ptype == 'number':
                try:
                    if '.' in value:
                        return float(value)
                    return int(value)
                except ValueError:
                    return value
            elif ptype == 'bool_null':
                if value.lower() == 'true':
                    return True
                elif value.lower() == 'false':
                    return False
                elif value.lower() == 'null':
                    return None
            else:
                return value
    
    # Look for score patterns in text (common in grading)
    # Pattern: "score: X", "grade: X", "points: X", "X/10", "X out of 10"
    score_patterns = [
        r'(?:score|grade|points|mark)s?\s*[:=]\s*(\d+(?:\.\d+)?)',
        r'(\d+(?:\.\d+)?)\s*/\s*\d+',
        r'(\d+(?:\.\d+)?)\s+out\s+of\s+\d+',
        r'(?:score|grade|points|mark)s?\s+is\s+(\d+(?:\.\d+)?)',
    ]
    
    for pattern in score_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1)
            try:
                if '.' in value:
                    return float(value)
                return int(value)
            except ValueError:
                return value
    
    # Try to find any number that might be a score (0-10 range)
    # Look for standalone numbers that could be scores
    score_pattern = r'\b([0-9]|10)\b'
    matches = re.findall(score_pattern, text)
    if matches:
        # Return the last number found (often the final score)
        return int(matches[-1])
    
    # Try to find fraction patterns like "0/1" or "1/1"
    fraction_pattern = r'\b(\d+/\d+)\b'
    match = re.search(fraction_pattern, text)
    if match:
        return match.group(1)
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _build_grading_prompt(
        self,
        domain: str,
        problem: str,
        solution: str,
        grading_guidelines: str,
        student_answer: str,
    ) -> str:
        """Build the grading prompt with clear instructions."""
        return f"""You are an expert IMO (International Mathematical Olympiad) grader evaluating student solutions to competition mathematics problems.

Your task is to grade a student's answer to an IMO-level mathematics problem. You must carefully analyze:
1. The problem statement
2. The official solution
3. The grading guidelines (rubric)
4. The student's submitted answer

Then provide your evaluation in the specified JSON format.

PROBLEM DOMAIN: {domain}

PROBLEM STATEMENT:
```
{problem}
```

OFFICIAL SOLUTION:
```
{solution}
```

GRADING GUIDELINES (RUBRIC):
```
{grading_guidelines}
```

STUDENT'S ANSWER:
```
{student_answer}
```

INSTRUCTIONS:
1. Read the problem carefully and understand what is being asked.
2. Study the official solution to understand the correct approach.
3. Review the grading guidelines to understand how points are awarded.
4. Analyze the student's answer step by step:
   - Did they understand the problem correctly?
   - Did they use the right approach?
   - Are their calculations correct?
   - Did they provide a complete proof/solution?
   - Where did they make errors, if any?
5. Assign a score based EXACTLY on the grading guidelines provided.

IMO GRADING PRINCIPLES:
- IMO problems are typically worth 7 points maximum
- Partial credit is awarded for significant progress toward the solution
- A complete, correct proof receives full marks (7 points)
- Minor errors may result in point deductions
- The grading guidelines specify exactly how points should be allocated
- Follow the rubric precisely - do not deviate from the specified scoring scheme

IMPORTANT: Your response MUST be valid JSON wrapped in <json> tags.

Respond in JSON format with the following schema:
<json>
{{
    "response": <your numerical score or evaluation>
}}
</json>

The "response" field should contain your final grading decision (typically a number or specific evaluation string as specified in the grading guidelines).

Examples of valid responses:
<json>
{{
    "response": 7
}}
</json>

<json>
{{
    "response": "0/1"
}}
</json>

<json>
{{
    "response": 3
}}
</json>

<json>
{{
    "response": "1/7"
}}
</json>

Ensure your JSON is properly formatted with no trailing commas."""

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract key fields for better prompt construction
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = self._build_grading_prompt(
            domain, problem, solution, grading_guidelines, student_answer
        )

        # Retry logic for LLM calls
        max_retries = 3
        last_error = None
        msg_history = []
        
        for attempt in range(max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                break
            except Exception as e:
                last_error = e
                self.log_fn(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return "None", [{"role": "error", "text": str(last_error)}]
                continue

        # Extract prediction from JSON with multiple fallback strategies
        prediction = "None"
        raw_response = msg_history[-1]["text"] if msg_history else ""
        
        extraction_attempts = [
            ("json_tags", lambda: _extract_jsons(raw_response)),
            ("regex", lambda: _extract_json_with_regex(raw_response)),
            ("heuristic", lambda: [{"response": _extract_response_heuristic(raw_response)}] if _extract_response_heuristic(raw_response) is not None else None),
        ]
        
        for method_name, attempt_fn in extraction_attempts:
            try:
                extracted = attempt_fn()
                if extracted and isinstance(extracted, list) and len(extracted) > 0:
                    last_result = extracted[-1]
                    if isinstance(last_result, dict) and "response" in last_result:
                        prediction = last_result["response"]
                        self.log_fn(f"Successfully extracted prediction using {method_name}: {prediction}")
                        break
            except Exception as e:
                self.log_fn(f"Extraction method {method_name} failed: {e}")
                continue
        
        if prediction == "None":
            self.log_fn(f"Failed to extract prediction from response. Raw text (first 1000 chars): {raw_response[:1000]}")
            # Try to log more context for debugging
            if raw_response:
                self.log_fn(f"Response length: {len(raw_response)} chars")

        return str(prediction), msg_history
