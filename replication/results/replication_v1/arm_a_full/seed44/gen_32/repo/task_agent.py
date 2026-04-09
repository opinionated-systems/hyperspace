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
    Also handles nested JSON objects within the tags.
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
        
        # Try to parse the JSON, handling potential nested structures
        try:
            # First attempt: direct parse
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # Second attempt: try to find valid JSON substring
            # Handle cases where there might be extra text before/after the JSON
            try:
                # Find the first '{' and last '}'
                json_start = inner.find('{')
                json_end = inner.rfind('}')
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    results.append(json.loads(inner[json_start:json_end + 1]))
            except (json.JSONDecodeError, ValueError):
                continue
    return results or None


def _extract_json_with_regex(text: str) -> list[dict] | None:
    """Fallback JSON extraction using regex for malformed outputs."""
    results = []
    
    # Try to find JSON objects in code blocks
    code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            # Try to extract just the JSON object from the content
            try:
                json_start = match.find('{')
                json_end = match.rfind('}')
                if json_start != -1 and json_end != -1:
                    results.append(json.loads(match[json_start:json_end + 1]))
            except (json.JSONDecodeError, ValueError):
                continue
    
    # Try to find JSON objects with common grading-related keys
    if not results:
        # Pattern to match JSON objects containing grading-related fields
        json_patterns = [
            r'\{[^{}]*"response"[^{}]*\}',
            r'\{[^{}]*"score"[^{}]*\}',
            r'\{[^{}]*"grade"[^{}]*\}',
            r'\{[^{}]*"answer"[^{}]*\}',
            r'\{[^{}]*"result"[^{}]*\}',
        ]
        for pattern in json_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    results.append(json.loads(match))
                except json.JSONDecodeError:
                    continue
            if results:
                break
    
    # Last resort: try to find any JSON-like object
    if not results:
        # Match balanced braces for nested JSON
        stack = []
        start_idx = None
        for i, char in enumerate(text):
            if char == '{':
                if not stack:
                    start_idx = i
                stack.append(char)
            elif char == '}':
                if stack:
                    stack.pop()
                    if not stack and start_idx is not None:
                        try:
                            candidate = text[start_idx:i+1]
                            parsed = json.loads(candidate)
                            # Only keep if it has expected structure
                            if any(k in parsed for k in ["response", "score", "grade", "answer", "result"]):
                                results.append(parsed)
                        except json.JSONDecodeError:
                            pass
                        start_idx = None
    
    return results or None


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
5. Assign a score based on the grading guidelines.

Respond in JSON format with the following schema:
<json>
{{
    "response": <your numerical score or evaluation>
}}
</json>

The "response" field should contain your final grading decision (typically a number or specific evaluation string as specified in the grading guidelines)."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with multiple fallback strategies
        prediction = "None"
        raw_response = msg_history[-1]["text"] if msg_history else ""
        
        extraction_attempts = [
            lambda: _extract_jsons(raw_response),
            lambda: _extract_json_with_regex(raw_response),
        ]
        
        extracted_data = None
        for attempt_fn in extraction_attempts:
            try:
                extracted = attempt_fn()
                if extracted:
                    extracted_data = extracted[-1]
                    break
            except Exception as e:
                self.log_fn(f"Extraction attempt failed: {e}")
                continue
        
        # Try to get prediction from extracted data
        if extracted_data:
            # Look for "response" field first
            if "response" in extracted_data:
                prediction = extracted_data["response"]
            # Fallback: look for common alternative field names
            elif "answer" in extracted_data:
                prediction = extracted_data["answer"]
            elif "score" in extracted_data:
                prediction = extracted_data["score"]
            elif "grade" in extracted_data:
                prediction = extracted_data["grade"]
            elif "result" in extracted_data:
                prediction = extracted_data["result"]
            else:
                # If no recognized field, use the first value if only one key exists
                if len(extracted_data) == 1:
                    prediction = list(extracted_data.values())[0]
        
        if prediction == "None":
            self.log_fn(f"Failed to extract prediction from response. Raw text: {raw_response[:500]}")

        return str(prediction), msg_history
