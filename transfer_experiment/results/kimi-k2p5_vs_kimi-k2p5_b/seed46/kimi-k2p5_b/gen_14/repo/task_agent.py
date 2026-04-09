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
            # Try to extract JSON object boundaries
            try:
                json_start = inner.find("{")
                json_end = inner.rfind("}")
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    results.append(json.loads(inner[json_start:json_end+1]))
            except json.JSONDecodeError:
                continue
    return results or None


def _extract_label_from_text(text: str) -> str | None:
    """Extract grading label from raw text with improved "almost" detection."""
    text_lower = text.lower()
    
    # Priority order: check for more specific patterns first
    # Pattern: key: "value" or key: 'value' or key: value
    patterns = [
        r'["\']?response["\']?\s*[:=]\s*["\']?(correct|incorrect|partial|almost)["\']?',
        r'["\']?label["\']?\s*[:=]\s*["\']?(correct|incorrect|partial|almost)["\']?',
        r'["\']?grade["\']?\s*[:=]\s*["\']?(correct|incorrect|partial|almost)["\']?',
        r'["\']?evaluation["\']?\s*[:=]\s*["\']?(correct|incorrect|partial|almost)["\']?',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1)
    
    # Check for standalone labels with word boundaries
    # Priority: almost > incorrect > partial > correct (most specific first)
    if re.search(r'\balmost\b', text_lower):
        return "almost"
    if re.search(r'\bincorrect\b', text_lower):
        return "incorrect"
    if re.search(r'\bpartial\b', text_lower):
        return "partial"
    if re.search(r'\bcorrect\b', text_lower):
        return "correct"
    
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

**"almost"** (70-89%, 6/7 marks):
- Correct approach and main ideas are right
- Nearly complete solution with only MINOR technical gaps
- Small errors that don't invalidate the core argument
- More complete than partial, but not fully rigorous

**"partial"** (30-69%, 2-5/7 marks):
- Some correct insights and meaningful progress
- Significant gaps or missing key steps
- Incomplete proof or major technical issues
- Right direction but substantially unfinished

**"incorrect"** (0-29%, 0-1/7 marks):
- Fundamentally wrong approach or invalid reasoning
- No valid mathematical progress
- Completely misses the point of the problem

## CRITICAL DISTINCTIONS:

- "almost" vs "partial": "almost" means the solution is NEARLY COMPLETE with only minor issues. "partial" means SIGNIFICANT GAPS remain.
- When in doubt between "almost" and "partial", prefer "partial" if there are any substantial gaps.
- "almost" should be used sparingly - only when the student clearly understood the solution but made small mistakes.

## GRADING GUIDELINES CONTEXT:
{grading_guidelines}

## PROBLEM:
{problem}

## OFFICIAL SOLUTION:
{solution}

## STUDENT'S ANSWER TO EVALUATE:
{student_answer}

## YOUR TASK:

Analyze the student's answer carefully. Compare against the official solution and grading guidelines.

Respond with ONLY a JSON object in <json> tags:

<json>
{{
    "reasoning": "Detailed analysis: What did the student get right? What are the gaps? Be specific about why this grade was chosen.",
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
                                val = val.strip().lower()
                                if val in ["correct", "incorrect", "partial", "almost"]:
                                    prediction = val
                                    break
                        if prediction != "None":
                            break
            
            # Strategy 2: Direct text extraction as fallback
            if prediction == "None":
                text_pred = _extract_label_from_text(raw_text)
                if text_pred:
                    prediction = text_pred
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        self.log_fn(f"Final prediction: {prediction}")
        return str(prediction), msg_history
