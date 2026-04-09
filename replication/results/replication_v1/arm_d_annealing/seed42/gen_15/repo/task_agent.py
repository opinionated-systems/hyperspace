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


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _build_grading_prompt(self, inputs: dict) -> str:
        """Build a structured prompt for IMO grading with chain-of-thought reasoning."""
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        return f"""You are an expert mathematical grader for the International Mathematical Olympiad (IMO).

Your task is to evaluate a student's solution to a mathematical problem with precision and consistency.

---

**Domain**: {domain}

**Problem**:
```
{problem}
```

**Official Solution**:
```
{solution}
```

**Grading Guidelines**:
```
{grading_guidelines}
```

**Student's Answer**:
```
{student_answer}
```

---

**Your Task**:

First, analyze the problem and solution carefully:
1. **Understand the Problem**: Identify what is being asked and the key mathematical concepts involved.
2. **Review the Official Solution**: Note the expected approach, key insights, and correct answer.
3. **Analyze the Student's Answer**: Examine the student's solution step by step for:
   - Mathematical correctness (are the calculations and proofs valid?)
   - Logical soundness (does the reasoning follow?)
   - Completeness (does it address all parts of the problem?)
   - Alignment with the official solution (is the approach valid even if different?)

4. **Apply Grading Guidelines**: Use the specific criteria provided to determine the assessment category.

**Assessment Categories** (choose exactly one):
- **"Correct"**: The solution is fully correct, complete, and well-justified. All mathematical steps are valid and the answer matches the official solution.
- **"Partially correct"**: The solution has some correct elements but is incomplete, has minor errors, or lacks proper justification. The student shows understanding but doesn't fully solve the problem.
- **"Incorrect"**: The solution is fundamentally wrong, contains major errors, or shows no meaningful understanding of the problem.

Then, provide your final assessment in the following JSON format:

<json>
{{
    "reasoning": "Your detailed step-by-step evaluation explaining what the student did right/wrong and why you chose the assessment category",
    "assessment": "Correct" or "Partially correct" or "Incorrect",
    "response": "The final answer/score as specified in the grading guidelines, or the assessment category if no specific score is given"
}}
</json>

**Critical Instructions**:
1. The "assessment" field MUST be exactly one of: "Correct", "Partially correct", or "Incorrect" (case-sensitive, no variations)
2. Ensure your response is valid JSON with proper escaping of quotes and newlines
3. Be consistent with the grading guidelines provided
4. If the student's answer is empty or nonsensical, mark as "Incorrect"
"""

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
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
                # Handle both "text" (paper format) and "content" (OpenAI format) fields
                last_message = msg_history[-1].get("text") or msg_history[-1].get("content", "")
                extracted = _extract_jsons(last_message)
                if extracted:
                    last_json = extracted[-1]
                    # Try assessment field first (contains categorical label like "Correct", "Partially correct", "Incorrect")
                    if "assessment" in last_json:
                        prediction = last_json["assessment"]
                        # Normalize assessment values to ensure consistency
                        assessment_str = str(prediction).strip()
                        # Map common variations to standard values
                        if assessment_str.lower() in ["correct", "right", "true", "valid", "full marks", "full credit"]:
                            prediction = "Correct"
                        elif assessment_str.lower() in ["partially correct", "partial", "partially", "partial credit", "some credit", "incomplete"]:
                            prediction = "Partially correct"
                        elif assessment_str.lower() in ["incorrect", "wrong", "false", "invalid", "no credit", "zero", "0"]:
                            prediction = "Incorrect"
                    # Fallback: try response field if assessment is not available
                    elif "response" in last_json:
                        prediction = last_json["response"]
                    # Fallback: try reasoning field if neither is available
                    elif "reasoning" in last_json:
                        prediction = last_json["reasoning"]
                else:
                    # If no JSON found, try to extract the last line as a fallback
                    lines = last_message.strip().split('\n')
                    for line in reversed(lines):
                        stripped = line.strip()
                        if stripped and not stripped.startswith('<') and not stripped.startswith('{'):
                            prediction = stripped
                            break
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
