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
    """Extract JSON objects from <json>...</json> blocks with improved robustness."""
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
        
        # Try multiple parsing strategies
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # Strategy 1: Try to find JSON object boundaries
            try:
                json_start = inner.find("{")
                json_end = inner.rfind("}")
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    results.append(json.loads(inner[json_start:json_end+1]))
            except json.JSONDecodeError:
                # Strategy 2: Try to fix common JSON issues
                try:
                    # Remove trailing commas before closing braces
                    fixed = re.sub(r',\s*}', '}', inner)
                    fixed = re.sub(r',\s*]', ']', fixed)
                    # Try to extract again
                    json_start = fixed.find("{")
                    json_end = fixed.rfind("}")
                    if json_start != -1 and json_end != -1 and json_end > json_start:
                        results.append(json.loads(fixed[json_start:json_end+1]))
                except json.JSONDecodeError:
                    continue
    return results or None


def _extract_label_from_text(text: str) -> str | None:
    """Extract grading label from raw text with improved pattern matching."""
    text_lower = text.lower()
    
    # Priority order: check for explicit JSON-like patterns first
    # These are more reliable than standalone words
    patterns = [
        # JSON format patterns
        r'"response"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"label"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"grade"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"evaluation"\s*:\s*"(correct|incorrect|partial|almost)"',
        # Single quote variants
        r"'response'\s*:\s*'(correct|incorrect|partial|almost)'",
        r"'label'\s*:\s*'(correct|incorrect|partial|almost)'",
        r"'grade'\s*:\s*'(correct|incorrect|partial|almost)'",
        r"'evaluation'\s*:\s*'(correct|incorrect|partial|almost)'",
        # Assignment patterns
        r'response\s*=\s*"(correct|incorrect|partial|almost)"',
        r'label\s*=\s*"(correct|incorrect|partial|almost)"',
        r'grade\s*=\s*"(correct|incorrect|partial|almost)"',
        r'evaluation\s*=\s*"(correct|incorrect|partial|almost)"',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1)
    
    # Check for standalone labels with word boundaries
    # Use a scoring system to handle multiple matches
    scores = {
        "almost": 0,
        "incorrect": 0,
        "partial": 0,
        "correct": 0
    }
    
    # Count occurrences with word boundaries
    for label in scores.keys():
        matches = re.findall(rf'\b{label}\b', text_lower)
        scores[label] = len(matches)
    
    # Return the label with highest score that appears at least once
    max_score = 0
    best_label = None
    for label, score in scores.items():
        if score > max_score:
            max_score = score
            best_label = label
    
    return best_label


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
        
        instruction = f"""You are an expert mathematics grader for International Mathematical Olympiad (IMO) problems. Your task is to evaluate student answers and assign exactly one of four grades.

## GRADE DEFINITIONS (STRICT):

**"correct"** (90-100%, 7/7 marks): 
- Complete, rigorous proof with all necessary steps
- Valid mathematical reasoning throughout
- May have minor typos but no logical gaps
- The solution would receive full marks in a real competition

**"almost"** (70-89%, 6/7 marks):
- Correct approach and main ideas are right
- Nearly complete solution with only MINOR technical gaps
- Small errors that don't invalidate the core argument
- The student clearly understood the solution but made small mistakes
- KEY DISTINCTION: The solution is SUBSTANTIALLY COMPLETE - missing only minor details
- Examples: missing a base case in induction, small calculation error, missing one inequality check

**"partial"** (30-69%, 2-5/7 marks):
- Some correct insights and meaningful progress
- Significant gaps or missing key steps
- Incomplete proof or major technical issues
- Right direction but substantially unfinished
- KEY DISTINCTION: The solution has SIGNIFICANT GAPS - not just minor issues
- Examples: missing the main lemma, only solving a special case, major logical flaw

**"incorrect"** (0-29%, 0-1/7 marks):
- Fundamentally wrong approach or invalid reasoning
- No valid mathematical progress
- Completely misses the point of the problem
- The solution is essentially wrong or trivial
- KEY DISTINCTION: The student made NO meaningful progress toward the solution

## CRITICAL DECISION RULES:

1. "almost" vs "partial": 
   - "almost" = solution is NEARLY COMPLETE (85%+ done), only minor technical issues
   - "partial" = SIGNIFICANT PORTIONS are missing or wrong (30-70% complete)
   - When in doubt, prefer "partial" over "almost"
   - Ask: "Could this solution get 6/7 marks?" If yes -> "almost", if no -> "partial"

2. "partial" vs "incorrect":
   - "partial" = student made MEANINGFUL PROGRESS with correct insights
   - "incorrect" = NO meaningful progress, fundamentally wrong approach
   - If the student has ANY valid mathematical insight (even if incomplete), prefer "partial" over "incorrect"
   - Ask: "Did the student understand any part of the problem?" If yes -> "partial", if no -> "incorrect"

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

        # Extract prediction with improved logic
        prediction = "None"
        raw_text = msg_history[-1]["text"] if msg_history else ""
        
        try:
            # Strategy 1: Extract from <json> tags
            extracted = _extract_jsons(raw_text)
            if extracted:
                for json_obj in extracted:
                    if isinstance(json_obj, dict):
                        for key in ["response", "label", "grade", "evaluation"]:
                            val = json_obj.get(key, "")
                            if isinstance(val, str):
                                val_lower = val.strip().lower()
                                if val_lower in ["correct", "incorrect", "partial", "almost"]:
                                    prediction = val_lower.capitalize()
                                    break
                        if prediction != "None":
                            break
            
            # Strategy 2: Direct text extraction as fallback
            if prediction == "None":
                text_pred = _extract_label_from_text(raw_text)
                if text_pred:
                    prediction = text_pred.capitalize()
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        self.log_fn(f"Final prediction: {prediction}")
        return str(prediction), msg_history
