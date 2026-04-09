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


def _extract_json_with_regex(text: str) -> list[dict] | None:
    """Fallback JSON extraction using regex for malformed outputs.
    
    Handles nested JSON objects by using a stack-based brace matching approach
    instead of simple regex that fails on nested structures.
    """
    results = []
    # Try to find JSON objects in code blocks
    code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            continue
    
    # Try to find JSON objects with curly braces using stack-based matching
    # This handles nested objects properly
    if not results:
        i = 0
        while i < len(text):
            # Find the start of a potential JSON object
            if text[i] == '{':
                start = i
                brace_count = 1
                i += 1
                # Track braces to find the matching closing brace
                while i < len(text) and brace_count > 0:
                    if text[i] == '{':
                        brace_count += 1
                    elif text[i] == '}':
                        brace_count -= 1
                    i += 1
                # If we found a complete object with "response" key, try to parse it
                if brace_count == 0:
                    candidate = text[start:i]
                    if '"response"' in candidate:
                        try:
                            results.append(json.loads(candidate))
                        except json.JSONDecodeError:
                            continue
            else:
                i += 1
    
    return results or None


def _extract_response_value(text: str) -> str | None:
    """Last-resort extraction: try to find a response value directly.
    
    This handles cases where the JSON is malformed but we can still
    extract the value after "response": using regex patterns.
    """
    # Pattern to match "response": followed by a value (number, string, or boolean)
    # Handles: "response": 7, "response": "7", "response": true, etc.
    patterns = [
        # Number (integer or float)
        r'"response"\s*:\s*(-?\d+(?:\.\d+)?)',
        # String in double quotes
        r'"response"\s*:\s*"([^"]*)"',
        # String in single quotes
        r"'response'\s*:\s*'([^']*)'",
        # Boolean or null
        r'"response"\s*:\s*(true|false|null)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1)
            # Try to convert to appropriate type
            try:
                # Try integer first
                return str(int(value))
            except ValueError:
                try:
                    # Try float
                    return str(float(value))
                except ValueError:
                    # Return as string
                    return value
    
    return None


def _validate_and_normalize_prediction(prediction: str, grading_guidelines: str) -> str:
    """Validate and normalize the prediction based on grading guidelines.
    
    This ensures the output matches expected format from the rubric.
    """
    if prediction is None or prediction == "None":
        return "None"
    
    # Convert to string and strip whitespace
    pred_str = str(prediction).strip()
    
    # Try to extract numeric score if grading guidelines suggest numeric scoring
    # Common patterns: "7 points", "score of 7", "0-7 scale"
    if any(pattern in grading_guidelines.lower() for pattern in ["points", "score", "mark", "0-", "1-", "2-", "3-", "4-", "5-", "6-", "7-", "8-", "9-", "10-"]):
        # Try to extract a number from the prediction
        try:
            # Direct numeric conversion
            if pred_str.replace('.', '', 1).replace('-', '', 1).isdigit():
                return pred_str
            # Extract number from text like "Score: 7" or "7 points"
            import re
            numbers = re.findall(r'-?\d+\.?\d*', pred_str)
            if numbers:
                return numbers[0]
        except Exception:
            pass
    
    return pred_str


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
        # Extract key fields for better prompt construction
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematical grader evaluating student solutions to competition mathematics problems.

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

IMPORTANT GRADING PRINCIPLES:
- Be objective and consistent with the official rubric
- Award partial credit when the rubric allows it
- Deduct points for missing steps, logical errors, or incomplete proofs
- Consider alternative correct approaches that differ from the official solution
- Check for common errors: calculation mistakes, missing cases, unstated assumptions

Respond ONLY in JSON format with the following schema:
<json>
{{
    "response": <your numerical score or evaluation>
}}
</json>

The "response" field must contain ONLY your final grading decision as a number or specific evaluation string exactly as specified in the grading guidelines. Do not include explanations or reasoning in the response field."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with multiple fallback strategies
        prediction = "None"
        raw_text = msg_history[-1]["text"]
        extraction_attempts = [
            lambda: _extract_jsons(raw_text),
            lambda: _extract_json_with_regex(raw_text),
        ]
        
        for attempt_fn in extraction_attempts:
            try:
                extracted = attempt_fn()
                if extracted and "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                    break
            except Exception as e:
                self.log_fn(f"Extraction attempt failed: {e}")
                continue
        
        # Last resort: try to extract response value directly from malformed JSON
        if prediction == "None":
            direct_value = _extract_response_value(raw_text)
            if direct_value is not None:
                prediction = direct_value
        
        # Validate and normalize the prediction
        prediction = _validate_and_normalize_prediction(prediction, grading_guidelines)
        
        if prediction == "None":
            self.log_fn(f"Failed to extract prediction from response. Raw text: {raw_text[:500]}")

        return str(prediction), msg_history
