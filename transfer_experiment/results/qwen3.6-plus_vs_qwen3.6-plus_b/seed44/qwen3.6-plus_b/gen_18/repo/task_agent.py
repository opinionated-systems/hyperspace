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

__version__ = "2.0.0"


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also falls back to scanning for bare JSON objects if no tags are found.
    Handles nested braces within JSON values by tracking brace depth.
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

    # Fallback: if no tagged JSON found, try to extract bare JSON objects
    if not results:
        try:
            # Try parsing the entire text as JSON first
            parsed = json.loads(text.strip())
            if isinstance(parsed, dict):
                results.append(parsed)
            elif isinstance(parsed, list):
                results.extend(item for item in parsed if isinstance(item, dict))
        except json.JSONDecodeError:
            # Try to find JSON-like objects using regex as last resort
            # Use a pattern that handles one level of nested braces
            pattern = r'\{(?:[^{}]|\{[^{}]*\})*\}'
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    obj = json.loads(match)
                    if isinstance(obj, dict):
                        results.append(obj)
                except json.JSONDecodeError:
                    continue
            # If still no results, try the simpler pattern
            if not results:
                pattern = r'\{[^{}]*\}'
                matches = re.findall(pattern, text)
                for match in matches:
                    try:
                        obj = json.loads(match)
                        if isinstance(obj, dict):
                            results.append(obj)
                    except json.JSONDecodeError:
                        continue

    # Post-process: strip whitespace from string values to clean up LLM output
    for result in results:
        for key, value in result.items():
            if isinstance(value, str):
                result[key] = value.strip()

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
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        domain = inputs.get("domain", "")

        instruction = f"""You are an expert IMO (International Mathematical Olympiad) grader specializing in {domain if domain else "mathematics"}. Your task is to evaluate a student's answer to a math problem.

**Problem:**
{problem}

**Official Solution:**
{solution}

**Grading Guidelines:**
{grading_guidelines}

**Student's Answer:**
{student_answer}

Perform a structured analysis:

**Step 1 — Identify key ideas in the official solution:**
List the essential mathematical steps, lemmas, or insights required for a complete solution. Be specific about what mathematical content is needed.

**Step 2 — Map the student's work:**
For each key idea from Step 1, determine whether the student:
- Correctly established it with valid reasoning
- Attempted it but with errors or gaps
- Did not address it at all

**Step 3 — Evaluate mathematical validity:**
Check each step the student did attempt for logical correctness. Note any gaps, unjustified claims, computational errors, or fundamental misunderstandings. Be strict: a claim without justification does not count as established.

**Step 4 — Assign a grade using this rubric:**
- "correct" (7 points): All key steps present with valid reasoning, conclusion matches the official answer. No significant errors or gaps. The solution is essentially complete and correct.
- "almost" (6 points): The student has the right approach and nearly complete solution, but has 1-2 minor errors, omissions, or unjustified claims. The core mathematical insight is present and the solution path is correct. The student clearly understands how to solve the problem.
- "partial" (1 point): The student has made genuine mathematical progress — e.g., proven a correct non-trivial lemma, identified a useful invariant, set up the problem correctly with valid initial steps. The progress must be mathematically sound and relevant to solving the problem. Minor errors in otherwise valid work still count as partial. IMPORTANT: The student must have established at least one correct, non-trivial mathematical result that advances toward the solution.
- "incorrect" (0 points): The approach is fundamentally flawed, the conclusion is wrong with no valid intermediate work, or there is no substantive mathematical progress. This includes: wrong answers with no valid reasoning, attempts that contain only restatements of the problem, work with critical logical errors that invalidate the entire approach, or solutions that miss the key insight entirely. IMPORTANT: If the student's work contains significant errors that undermine their entire approach, or if they only restate the problem without making progress, grade as "incorrect".

**CRITICAL DISTINCTION — "partial" vs "incorrect":**
- Grade as "partial" ONLY if the student has produced at least one correct, non-trivial mathematical statement or derivation that genuinely advances toward solving the problem. Simply restating the problem, making vague observations, or writing down formulas without applying them does NOT count.
- Grade as "incorrect" if: the student's reasoning contains a critical flaw that invalidates their entire approach; the student only restates the problem or makes trivial observations; the student's work is incoherent or irrelevant; the student reaches a wrong conclusion with no valid intermediate steps.
- When in doubt between "partial" and "incorrect", carefully check: has the student actually PROVEN or DERIVED something correct and non-trivial? If not, it is "incorrect".

**Step 5 — Cross-check with grading guidelines:**
The grading guidelines above may specify exact criteria for partial credit. If the student's work matches any described partial-progress scenario, use that guidance. If the guidelines specify that certain errors reduce the score to 0, follow that.

**Step 6 — Final sanity check:**
Before finalizing your grade, ask: "Does the student's work contain at least one correct, non-trivial mathematical result?" If the answer is no, the grade should be "incorrect". If the student has the right idea but made a small error, consider "almost". If the student has made real progress but the solution is far from complete, consider "partial".

**Step 7 — Verify the student's final conclusion:**
Check whether the student's final answer/conclusion matches the expected result from the official solution. A correct conclusion with valid reasoning is "correct". A correct conclusion with minor gaps is "almost". An incorrect conclusion with valid partial work is "partial". An incorrect conclusion with no valid work is "incorrect".

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your structured analysis: list key ideas from the official solution, map what the student achieved for each, note errors, verify the final conclusion, then justify your grade",
    "response": "correct" or "incorrect" or "partial" or "almost"
}}
</json>"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted and "response" in extracted[-1]:
                prediction = extracted[-1]["response"]
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Normalize prediction to lowercase for consistent comparison
        if isinstance(prediction, str):
            prediction = prediction.strip().lower()
            # Map common variations to standard labels
            label_map = {
                "correct": "correct",
                "incorrect": "incorrect",
                "partial": "partial",
                "almost": "almost",
                "partially correct": "partial",
                "partially": "partial",
                "wrong": "incorrect",
                "right": "correct",
                "complete": "correct",
                "incomplete": "partial",
            }
            prediction = label_map.get(prediction, prediction)

        return str(prediction), msg_history
