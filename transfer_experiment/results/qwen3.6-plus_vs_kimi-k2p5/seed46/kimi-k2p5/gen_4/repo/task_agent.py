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


def _extract_json_from_markdown(text: str) -> list[dict] | None:
    """Extract JSON objects from markdown code blocks."""
    results = []
    # Match ```json ... ``` or just ``` ... ``` blocks
    pattern = r'```(?:json)?\s*(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            continue
    return results or None


def _extract_any_json(text: str) -> list[dict] | None:
    """Extract JSON objects using multiple strategies."""
    # Try <json> tags first
    results = _extract_jsons(text)
    if results:
        return results
    
    # Try markdown code blocks
    results = _extract_json_from_markdown(text)
    if results:
        return results
    
    # Try to find JSON objects directly
    results = []
    # Look for patterns that look like JSON objects
    pattern = r'\{[^{}]*"[^"]+"[^{}]*\}'
    matches = re.findall(pattern, text)
    for match in matches:
        try:
            results.append(json.loads(match))
        except json.JSONDecodeError:
            continue
    
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
        # Build a more detailed prompt that encourages structured output
        problem = inputs.get('problem', '')
        solution = inputs.get('solution', '')
        grading_guidelines = inputs.get('grading_guidelines', '')
        student_answer = inputs.get('student_answer', '')
        
        instruction = f"""You are an expert mathematical grader for IMO-style problems. Your task is to evaluate a student's answer and assign exactly one of four grades: Correct, Incorrect, Partial, or Almost.

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

GRADE DEFINITIONS AND DECISION CRITERIA:

**Correct**: The student's answer is fully correct, complete, and matches the official solution.
- ALL key steps and reasoning are present and valid
- The proof/solution is complete with no gaps
- All claims are properly justified
- Use this ONLY when the solution is 100% complete and correct

**Incorrect**: The student's answer is fundamentally wrong or makes no meaningful progress.
- The approach is completely wrong or misguided
- No key insights from the official solution are present
- The reasoning contains fatal flaws
- The student misunderstood the problem
- Use this when there is NO significant progress toward the solution

**Partial**: The student made MEANINGFUL progress but the solution is INCOMPLETE.
- Found a key invariant, lemma, or framework (significant insight)
- BUT: Missing the main proof, verification, or completion of the argument
- The student did NOT finish the solution
- Key components are missing even though some progress was made
- This is for "good start but didn't finish" cases

**Almost**: The solution is NEARLY COMPLETE with only MINOR issues.
- The core proof structure and reasoning are correct
- Only trivial errors: small calculation mistakes, typos, slight oversights
- The main argument is essentially complete and valid
- Issues are NEGLIGIBLE and don't affect the core logic
- This is for "essentially correct with tiny flaws" cases

CRITICAL DISTINCTIONS - USE THESE DECISION RULES:

1. Partial vs Incorrect:
   - Did the student find ANY key insight from the official solution? 
   - If YES (invariant, lemma, framework) → Partial
   - If NO meaningful insight → Incorrect

2. Partial vs Correct:
   - Is the solution COMPLETE (all steps present and verified)?
   - If YES (complete) → Correct
   - If NO (missing key steps) → Partial

3. Almost vs Correct:
   - Are there ANY non-trivial gaps or errors?
   - If YES → Almost
   - If NO (perfect) → Correct

4. Almost vs Partial:
   - Is the core argument essentially complete?
   - If YES (just minor issues) → Almost
   - If NO (significant work missing) → Partial

EVALUATION PROCESS:
Step 1: Identify what key elements from the official solution are present
Step 2: Identify what is MISSING (gaps, incomplete proofs, unverified claims)
Step 3: Identify any ERRORS (calculation mistakes, logical flaws)
Step 4: Classify errors as "fatal/fundamental" vs "minor/negligible"
Step 5: Determine if meaningful progress was made (key insights present?)
Step 6: Apply the decision rules above to select the grade

Respond in JSON format:
<json>
{{
    "response": "Correct" or "Incorrect" or "Partial" or "Almost"
}}
</json>

You must choose exactly ONE of these four labels. Be conservative and rigorous in your evaluation. When in doubt between two grades, choose the LOWER grade (more conservative)."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON using multiple strategies
        prediction = "None"
        try:
            extracted = _extract_any_json(response)
            if extracted:
                # Try to find a response field
                for item in extracted:
                    if isinstance(item, dict):
                        if "response" in item:
                            prediction = item["response"]
                            break
                        # Also check for common alternative field names
                        for key in ["grade", "label", "result", "evaluation", "answer"]:
                            if key in item:
                                prediction = item[key]
                                break
                        if prediction != "None":
                            break
            
            # If no JSON found, try to extract a simple label from the text
            if prediction == "None":
                # Look for common grade labels in the response
                text_lower = response.lower()
                # Check for exact matches first
                if "grade: correct" in text_lower or '"correct"' in text_lower:
                    prediction = "correct"
                elif "grade: incorrect" in text_lower or '"incorrect"' in text_lower:
                    prediction = "incorrect"
                elif "grade: partial" in text_lower or '"partial"' in text_lower:
                    prediction = "partial"
                elif "grade: almost" in text_lower or '"almost"' in text_lower:
                    prediction = "almost"
                elif "correct" in text_lower and "incorrect" not in text_lower:
                    prediction = "correct"
                elif "incorrect" in text_lower or "wrong" in text_lower:
                    prediction = "incorrect"
                elif "partial" in text_lower:
                    prediction = "partial"
                elif "almost" in text_lower:
                    prediction = "almost"
                else:
                    # Use the first line of the response as prediction
                    first_line = response.strip().split('\n')[0][:100]
                    if first_line:
                        prediction = first_line
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
