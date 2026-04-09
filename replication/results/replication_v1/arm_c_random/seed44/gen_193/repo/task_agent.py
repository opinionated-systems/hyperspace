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


def _extract_json_with_retry(text: str, max_retries: int = 2) -> list[dict] | None:
    """Extract JSON with multiple fallback strategies.
    
    First tries standard extraction, then attempts to fix common
    JSON formatting issues like trailing commas, missing quotes, etc.
    """
    # First attempt: standard extraction
    result = _extract_jsons(text)
    if result:
        return result
    
    # Common JSON fixes to try
    fixes = [
        # Fix trailing commas before closing braces/brackets
        (r',(\s*[}\]])', r'\1'),
        # Fix single quotes to double quotes
        (r"'([^']*?)'", r'"\1"'),
        # Fix unquoted keys (simple cases)
        (r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":'),
    ]
    
    # Retry with progressively more aggressive fixes
    fixed_text = text
    for attempt in range(min(max_retries, len(fixes))):
        try:
            pattern, replacement = fixes[attempt]
            fixed_text = re.sub(pattern, replacement, fixed_text)
            result = _extract_jsons(fixed_text)
            if result:
                logger.debug(f"JSON extraction succeeded after fix attempt {attempt + 1}")
                return result
        except Exception as e:
            logger.debug(f"Fix attempt {attempt + 1} failed: {e}")
            continue
    
    return None


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
        # Extract task-specific fields for better prompting
        domain = inputs.get("domain", "unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        instruction = f"""You are an expert grading agent for {domain} problems.

Your task is to evaluate a student's answer against the correct solution and grading guidelines.

Problem:
```
{problem}
```

Correct Solution:
```
{solution}
```

Grading Guidelines:
```
{grading_guidelines}
```

Student's Answer:
```
{student_answer}
```

Follow this structured evaluation process:

1. **Understanding**: First, identify the key concepts and steps required to solve this problem.

2. **Comparison**: Compare the student's answer to the correct solution:
   - What did the student get right?
   - What did the student get wrong or miss?
   - Are there any conceptual misunderstandings?

3. **Grading Decision**: Based on the grading guidelines, determine:
   - Is the answer correct, partially correct, or incorrect?
   - What score/grade would you assign?

4. **Feedback**: Provide specific, actionable feedback that would help the student improve.

Provide your evaluation in the following JSON format:
<json>
{{
    "evaluation": {{
        "is_correct": true/false,
        "confidence": 0.0-1.0,
        "score": "full/partial/none",
        "key_points_matched": ["point 1", "point 2"],
        "errors_identified": ["error 1", "error 2"]
    }},
    "response": "Your detailed evaluation here. Include: 1) Overall assessment, 2) Specific strengths, 3) Areas for improvement, 4) Final verdict."
}}
</json>

Important: 
- Ensure your response is valid JSON with proper double quotes.
- Confidence should reflect your certainty in the grading decision (1.0 = absolutely certain).
- Be objective and fair in your assessment."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with retry mechanism
        prediction = "None"
        try:
            extracted = _extract_json_with_retry(msg_history[-1]["text"])
            if extracted:
                data = extracted[-1]
                # Try to get structured evaluation first, fall back to simple response
                if "evaluation" in data and "response" in data:
                    eval_data = data["evaluation"]
                    response_text = data["response"]
                    # Build comprehensive prediction with structured info
                    parts = []
                    parts.append(f"Overall: {response_text}")
                    if "is_correct" in eval_data:
                        correctness = "Correct" if eval_data["is_correct"] else "Incorrect"
                        parts.append(f"Verdict: {correctness}")
                    if "confidence" in eval_data:
                        parts.append(f"Confidence: {eval_data['confidence']}")
                    if "score" in eval_data:
                        parts.append(f"Score: {eval_data['score']}")
                    if "key_points_matched" in eval_data and eval_data["key_points_matched"]:
                        parts.append(f"Strengths: {', '.join(eval_data['key_points_matched'])}")
                    if "errors_identified" in eval_data and eval_data["errors_identified"]:
                        parts.append(f"Issues: {', '.join(eval_data['errors_identified'])}")
                    prediction = " | ".join(parts)
                elif "response" in data:
                    prediction = data["response"]
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
