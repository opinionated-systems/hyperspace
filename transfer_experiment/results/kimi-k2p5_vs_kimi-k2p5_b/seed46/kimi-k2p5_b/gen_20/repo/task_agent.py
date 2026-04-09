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

# Valid grading labels for IMO evaluation
VALID_LABELS = ["correct", "almost", "partial", "incorrect"]


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks."""
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


def _extract_label_from_text(text: str) -> str | None:
    """Extract grading label from raw text using multiple strategies."""
    text_lower = text.lower()
    
    # Look for explicit label mentions
    # Check for "correct" (but not "incorrect")
    if re.search(r'\bcorrect\b', text_lower) and not re.search(r'\bincorrect\b', text_lower):
        return "correct"
    
    # Check for "incorrect"
    if re.search(r'\bincorrect\b', text_lower):
        return "incorrect"
    
    # Check for "partial"
    if re.search(r'\bpartial\b', text_lower):
        return "partial"
    
    # Check for "almost"
    if re.search(r'\balmost\b', text_lower):
        return "almost"
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with improved accuracy."""

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
        
        instruction = f"""You are an expert mathematics grader for International Mathematical Olympiad (IMO) problems. Your task is to evaluate student answers and assign exactly one of four grades with high precision.

## GRADE DEFINITIONS (STRICT):

**"correct"** (90-100%, 7/7 marks): 
- Complete, rigorous proof with all necessary steps
- Valid mathematical reasoning throughout
- May have minor typos but no logical gaps
- The solution would receive full marks in a real competition
- NO significant errors or missing steps

**"almost"** (70-89%, 6/7 marks):
- Correct approach and main ideas are right
- Nearly complete solution with only MINOR technical gaps
- Small errors that don't invalidate the core argument
- The student clearly understood the solution but made small mistakes
- KEY DISTINCTION: The solution is SUBSTANTIALLY COMPLETE (85%+ done) - missing only minor details
- Examples: missing a base case in induction, small calculation error, missing one inequality check, minor notation issues
- The core proof structure is sound and complete

**"partial"** (30-69%, 2-5/7 marks):
- Some correct insights and meaningful progress
- Significant gaps or missing key steps
- Incomplete proof or major technical issues
- Right direction but substantially unfinished
- KEY DISTINCTION: The solution has SIGNIFICANT GAPS (30-70% complete) - not just minor issues
- Examples: missing the main lemma, only solving a special case, major logical flaw, incomplete proof of key claim
- The student understood PART of the problem but not the full solution

**"incorrect"** (0-29%, 0-1/7 marks):
- Fundamentally wrong approach or invalid reasoning
- No valid mathematical progress
- Completely misses the point of the problem
- The solution is essentially wrong or trivial
- KEY DISTINCTION: The student made NO meaningful progress toward the solution
- Examples: completely wrong method, trivial observations only, misunderstanding the problem statement

## CRITICAL DECISION RULES:

1. "almost" vs "partial": 
   - "almost" = solution is NEARLY COMPLETE (85%+ done), only minor technical issues
   - "partial" = SIGNIFICANT PORTIONS are missing or wrong (30-70% complete)
   - When in doubt, prefer "partial" over "almost"
   - Ask: "Could this solution get 6/7 marks?" If yes -> "almost", if no -> "partial"
   - "almost" means the proof is essentially complete with only cosmetic issues

2. "partial" vs "incorrect":
   - "partial" = student made MEANINGFUL PROGRESS with correct insights
   - "incorrect" = NO meaningful progress, fundamentally wrong approach
   - If the student has ANY valid mathematical insight (even if incomplete), prefer "partial" over "incorrect"
   - Ask: "Did the student understand any part of the problem?" If yes -> "partial", if no -> "incorrect"
   - "partial" requires at least one correct non-trivial step toward the solution

3. "almost" vs "correct":
   - "correct" = fully rigorous, would get full marks (7/7)
   - "almost" = small but noticeable gaps that would lose exactly 1 mark (6/7)
   - Ask: "Are there ANY gaps that would cause mark deduction?" If yes -> "almost", if no -> "correct"

## GRADING GUIDELINES CONTEXT:
{grading_guidelines}

## PROBLEM:
{problem}

## OFFICIAL SOLUTION:
{solution}

## STUDENT'S ANSWER TO EVALUATE:
{student_answer}

## YOUR TASK:

Step 1: Analyze what the student got right and what they got wrong.
Step 2: Estimate the percentage of completeness (0-100%).
Step 3: Check if the student made ANY meaningful progress toward the solution.
Step 4: Determine if the solution is substantially complete (85%+) or has significant gaps.
Step 5: Apply the decision rules above to select the appropriate grade.

IMPORTANT: Be conservative in assigning "almost" - only use it when the solution is truly nearly complete with just minor issues. Most incomplete solutions should be "partial".

Respond with ONLY a JSON object in <json> tags:

<json>
{{
    "reasoning": "Detailed analysis: (1) What did the student get right? (2) What are the gaps/errors? (3) Estimated completeness percentage. (4) Why this specific grade was chosen based on the definitions and decision rules.",
    "response": "correct" | "almost" | "partial" | "incorrect"
}}
</json>

The "response" field MUST be exactly one of: "correct", "almost", "partial", or "incorrect" (all lowercase)."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction using multiple strategies
        prediction = "None"
        try:
            raw_text = msg_history[-1]["text"] if msg_history else ""
            
            # Strategy 1: Try to extract from JSON tags
            extracted = _extract_jsons(raw_text)
            if extracted:
                for key in ["response", "label", "grade", "evaluation"]:
                    val = extracted[-1].get(key, "")
                    if isinstance(val, str):
                        val_lower = val.strip().lower()
                        if val_lower in VALID_LABELS:
                            prediction = val_lower.capitalize()
                            break
            
            # Strategy 2: If JSON extraction failed, try text extraction
            if prediction == "None":
                text_pred = _extract_label_from_text(raw_text)
                if text_pred:
                    prediction = text_pred.capitalize()
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
