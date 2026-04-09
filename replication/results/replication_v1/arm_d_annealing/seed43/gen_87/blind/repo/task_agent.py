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


def _extract_json_fuzzy(text: str) -> list[dict] | None:
    """Fallback JSON extraction for malformed responses.
    
    Tries to find JSON objects even without proper <json> tags.
    """
    results = []
    # Try to find JSON objects between curly braces
    depth = 0
    start_idx = None
    for i, char in enumerate(text):
        if char == '{':
            if depth == 0:
                start_idx = i
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0 and start_idx is not None:
                try:
                    obj = json.loads(text[start_idx:i+1])
                    results.append(obj)
                except json.JSONDecodeError:
                    pass
                start_idx = None
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced reasoning."""

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
        domain = inputs.get('domain', 'Mathematics')
        problem = inputs.get('problem', '')
        solution = inputs.get('solution', '')
        grading_guidelines = inputs.get('grading_guidelines', '')
        student_answer = inputs.get('student_answer', '')
        
        instruction = f"""You are an expert {domain} grader evaluating student solutions.

Your task is to grade a student's answer by comparing it against the official solution and grading guidelines.

PROBLEM:
```
{problem}
```

OFFICIAL SOLUTION:
```
{solution}
```

GRADING GUIDELINES:
```
{grading_guidelines}
```

STUDENT'S ANSWER:
```
{student_answer}
```

Think step by step:
1. Analyze what the problem is asking for
2. Review the official solution approach
3. Compare the student's answer to the official solution
4. Check if the student meets the criteria in the grading guidelines
5. Determine the appropriate grade/score

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis here",
    "response": "Your final grade/evaluation here"
}}
</json>"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with multiple fallback strategies
        prediction = "None"
        try:
            last_msg = msg_history[-1]["text"]
            
            # Try primary extraction method
            extracted = _extract_jsons(last_msg)
            
            # Fallback to fuzzy extraction if primary fails
            if not extracted:
                extracted = _extract_json_fuzzy(last_msg)
            
            if extracted:
                last_json = extracted[-1]
                # Prefer "response" field, but accept other common field names
                for key in ["response", "grade", "score", "evaluation", "answer", "result"]:
                    if key in last_json:
                        prediction = last_json[key]
                        break
                else:
                    # If no recognized field, use the whole JSON as string
                    prediction = json.dumps(last_json)
                    
                self.log_fn(f"Extracted prediction: {prediction}")
            else:
                self.log_fn("No JSON found in response, using raw text")
                # Last resort: use the raw response text
                prediction = last_msg[:500]  # Truncate to avoid huge outputs
                
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
