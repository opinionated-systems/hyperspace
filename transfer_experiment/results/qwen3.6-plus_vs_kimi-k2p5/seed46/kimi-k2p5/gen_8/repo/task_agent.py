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

__version__ = "2.3.0"


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
        problem = inputs.get('problem', '')
        solution = inputs.get('solution', '')
        grading_guidelines = inputs.get('grading_guidelines', '')
        student_answer = inputs.get('student_answer', '')

        instruction = f"""You are an expert mathematical grader evaluating IMO-style competition problems. Your task is to carefully analyze a student's answer and assign exactly one of four grades.

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

---

GRADE DEFINITIONS WITH CONCRETE EXAMPLES:

**Correct**: The answer is 100% complete and correct with ZERO flaws.
- All key steps from the official solution are present
- The proof is complete with no gaps
- All reasoning is mathematically valid
- NO typos, NO calculation errors, NO missing steps, NO logical gaps
- Use ONLY when the solution is PERFECT - if you can find ANY flaw, it's not Correct
- EXAMPLE: A complete proof with all steps verified and no errors whatsoever

**Incorrect**: The answer is fundamentally wrong with NO meaningful progress.
- The approach is completely wrong or misguided
- No key insights from the official solution are present
- The reasoning contains fatal flaws
- Use when there is essentially ZERO progress toward the solution
- KEY TEST: If the student only restates the problem, states obvious facts, or provides a wrong approach → Incorrect
- EXAMPLE: Student claims "the answer is obvious" or uses a completely wrong method

**Partial**: The student made SIGNIFICANT progress but the solution is INCOMPLETE with MAJOR gaps.
- Found KEY insights, invariants, or lemmas from the official solution (not just trivial observations)
- Made MEANINGFUL progress toward the solution (e.g., proved a key lemma, found the right invariant)
- BUT: Missing critical steps, main proof, or completion - the core argument is NOT present
- The solution is unfinished despite good progress
- KEY TEST: The student found something IMPORTANT from the official solution but didn't finish the main proof
- EXAMPLE: Student proved a key lemma but never connected it to the main result; or found the right invariant but didn't complete the induction

**Almost**: The solution is NEARLY COMPLETE with the RIGHT approach but has MINOR issues.
- The core proof structure and main reasoning are CORRECT and COMPLETE
- The student understood the main approach and executed it properly
- Only trivial errors: small calculation mistakes, typos, minor oversights, one small step missing
- The main argument is essentially valid - fix the minor issues and it's Correct
- Issues are NEGLIGIBLE and don't affect the core logic
- KEY DISTINCTION: Unlike Partial, the main proof IS essentially complete
- KEY TEST: Can you fix the issues in 1-2 lines without changing the main argument? If yes → Almost
- EXAMPLE: Correct proof with one small calculation error; or correct approach with a typo in the final answer; or correct proof missing one trivial base case

---

DECISION FRAMEWORK - Answer these questions STRICTLY in order:

Q1: Does the student demonstrate ANY non-trivial insight from the official solution?
   - Check: Did they find a key invariant, prove a key lemma, or make substantive progress?
   - If NO (only restates problem, trivial observations, or wrong approach) → **Incorrect**
   - If YES → Continue to Q2

Q2: Is the main proof structure and core argument CORRECT and COMPLETE?
   - Check: Does the student have the RIGHT approach? Is the main proof essentially there?
   - If NO (missing main proof, wrong approach, only has preliminary work) → **Partial**
   - If YES (main proof is correct and complete) → Continue to Q3

Q3: Are there ANY errors, gaps, or issues at all?
   - Check: Look for typos, calculation errors, missing trivial steps, logical gaps
   - If NO (absolutely perfect) → **Correct**
   - If YES (has some issues) → Continue to Q4

Q4: Are the errors/gaps MINOR (typos, small calculation errors, one small step) or MAJOR?
   - Check: Can you fix the issues in 1-2 lines without changing the main argument?
   - If MINOR issues only (main proof is solid, just needs tiny fixes) → **Almost**
   - If MAJOR gaps (missing main proof technique, wrong approach) → **Partial** (revisit Q2)

---

CRITICAL DISTINCTIONS - READ CAREFULLY:

**Incorrect vs Partial**:
- "Incorrect" = NO meaningful progress (zero key insights from official solution)
- "Partial" = SIGNIFICANT progress (found key insights but didn't complete main proof)
- BE STRICT: Don't give Partial for trivial observations or restating the problem

**Partial vs Almost** (THIS IS THE HARDEST DISTINCTION):
- "Partial" = Significant progress but MAJOR gaps remain - the MAIN PROOF is NOT present
  * Student has good preliminary work but missing the core argument
  * Even if you fix minor issues, the solution is still incomplete
  * The student is maybe 30-70% done, but the critical 30% (main proof) is missing
- "Almost" = RIGHT approach and MAIN PROOF is CORRECT, only MINOR issues remain
  * The core argument is there and valid
  * Only small fixes needed (typos, calculation errors, one trivial step)
  * The student is 90-99% done, just needs tiny corrections
- KEY TEST: Can you imagine fixing the issues in 1-2 lines? If yes → Almost. If no → Partial.
- KEY TEST: Is the main proof structure correct and complete? If no → Partial. If yes → Almost.

**Almost vs Correct**:
- "Almost" = Some minor flaw exists (even a small typo or calculation error)
- "Correct" = PERFECT - absolutely no errors, no typos, no gaps, no calculation mistakes
- BE STRICT: If you can find ANY flaw, it's Almost, not Correct

---

WORK THROUGH THIS STEP BY STEP:

Step 1: List the KEY ELEMENTS required by the official solution
Step 2: Check which elements are PRESENT in the student's answer
Step 3: Identify the MAIN PROOF - is it correct and complete?
Step 4: Identify any ERRORS or GAPS in the student's reasoning
Step 5: Classify errors as "fatal/fundamental" vs "minor/negligible"
Step 6: Apply the Decision Framework above RIGOROUSLY
Step 7: Select the preliminary grade
Step 8: VERIFY your choice with these sanity checks:
   - If you chose "Correct": Are you SURE there are NO typos, NO calculation errors, NO gaps? If unsure, downgrade to "Almost"
   - If you chose "Almost": Is the main proof CORRECT and COMPLETE? Are issues truly minor (fixable in 1-2 lines)?
   - If you chose "Partial": Is the main proof MISSING or INCOMPLETE? Would fixing minor issues still leave major gaps?
   - If you chose "Incorrect": Did the student make ANY non-trivial progress? If yes, reconsider "Partial"
Step 9: Finalize your grade

---

Respond in this exact JSON format:
<json>
{{
    "response": "Correct" or "Incorrect" or "Partial" or "Almost"
}}
</json>

You must choose exactly ONE of these four labels. Be RIGOROUS and CONSERVATIVE:
- When in doubt between Incorrect and Partial, choose Incorrect
- When in doubt between Partial and Almost, choose Partial  
- When in doubt between Almost and Correct, choose Almost
- Only use Correct for PERFECT solutions with absolutely no errors whatsoever
- Only use Almost when the main proof is correct and complete, with only truly minor issues
- Remember: Almost = main proof is RIGHT but has tiny flaws; Partial = main proof is MISSING or wrong"""

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
                text_lower = response.lower()
                # Check for exact matches first (more specific patterns)
                if '"correct"' in text_lower or 'grade: correct' in text_lower:
                    prediction = "correct"
                elif '"incorrect"' in text_lower or 'grade: incorrect' in text_lower:
                    prediction = "incorrect"
                elif '"partial"' in text_lower or 'grade: partial' in text_lower:
                    prediction = "partial"
                elif '"almost"' in text_lower or 'grade: almost' in text_lower:
                    prediction = "almost"
                # Then check for standalone words (be careful with ordering)
                elif "incorrect" in text_lower or "wrong" in text_lower:
                    prediction = "incorrect"
                elif "partial" in text_lower:
                    prediction = "partial"
                elif "almost" in text_lower:
                    prediction = "almost"
                elif "correct" in text_lower:
                    prediction = "correct"
                else:
                    # Use the first line of the response as prediction
                    first_line = response.strip().split('\n')[0][:100]
                    if first_line:
                        prediction = first_line
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
