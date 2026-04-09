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
from typing import Optional

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
            continue
    return results or None


def _normalize_prediction(prediction: str) -> str:
    """Normalize prediction to one of the four valid labels."""
    if not prediction:
        return "incorrect"
    
    cleaned = prediction.strip().lower().strip(".!?,:;\"'").strip()
    valid_labels = ["correct", "almost", "partial", "incorrect"]
    
    # Check for exact match
    if cleaned in valid_labels:
        return cleaned
    
    # Check for negation patterns
    negation_patterns = ["not ", "no ", "isn't ", "isnt ", "is not ", "not a "]
    for neg in negation_patterns:
        if neg in cleaned and ("correct" in cleaned or "right" in cleaned):
            return "incorrect"
    
    # Word boundary matching (priority: almost > partial > incorrect > correct)
    if re.search(r'\balmost\b', cleaned):
        return "almost"
    if re.search(r'\bpartial\b', cleaned):
        return "partial"
    if re.search(r'\bincorrect\b', cleaned) or re.search(r'\bwrong\b', cleaned):
        return "incorrect"
    
    # Substring matching fallback
    if "almost" in cleaned:
        return "almost"
    if "partial" in cleaned:
        return "partial"
    if "incorrect" in cleaned or "wrong" in cleaned or "error" in cleaned:
        return "incorrect"
    
    # Split by separators and check parts
    parts = re.split(r'[\s,;|]+', cleaned)
    for part in parts:
        part = part.strip(".!?,:;\"'")
        if part in valid_labels:
            return part
    
    # Check for "correct" as substring (last resort)
    if "correct" in cleaned:
        return "correct"
    
    logger.warning(f"Could not normalize prediction '{prediction}', defaulting to 'incorrect'")
    return "incorrect"


def _extract_label_from_text(text: str) -> str:
    """Extract a valid label from raw text when JSON parsing fails."""
    if not text:
        return "incorrect"
    
    text_lower = text.lower()
    
    # Look for explicit labels in JSON-like format
    json_patterns = [
        ('"almost"', "almost"), ("'almost'", "almost"),
        ('"partial"', "partial"), ("'partial'", "partial"),
        ('"incorrect"', "incorrect"), ("'incorrect'", "incorrect"),
        ('"correct"', "correct"), ("'correct'", "correct"),
    ]
    for pattern, label in json_patterns:
        if pattern in text_lower:
            return label
    
    # Look for labels after common indicators
    indicators = ["label:", "grade:", "rating:", "assessment:", "verdict:", 
                "final answer:", "conclusion:", "decision:", "evaluation:"]
    for indicator in indicators:
        idx = text_lower.find(indicator)
        if idx != -1:
            after = text_lower[idx + len(indicator):].strip()
            for label in ["correct", "almost", "partial", "incorrect"]:
                if after.startswith(label):
                    remainder = after[len(label):]
                    if not remainder or not remainder[0].isalpha():
                        return label
    
    # Regex word boundary matching
    priority_labels = ["almost", "partial", "incorrect", "correct"]
    for label in priority_labels:
        pattern = r'\b' + label + r'\b'
        if re.search(pattern, text_lower):
            if label == "correct":
                if re.search(r'\bincorrect\b', text_lower):
                    return "incorrect"
            return label
    
    # Partial word matches fallback
    if "almost" in text_lower:
        return "almost"
    if "partial" in text_lower:
        return "partial"
    if "incorrect" in text_lower or "wrong" in text_lower or "error" in text_lower:
        return "incorrect"
    if "correct" in text_lower:
        if "incorrect" not in text_lower and "partial" not in text_lower and "almost" not in text_lower:
            return "correct"
    
    logger.warning(f"Could not extract label from text, defaulting to 'incorrect'")
    return "incorrect"


# Few-shot examples for better grading accuracy
FEW_SHOT_EXAMPLES = """
Example 1 - CORRECT (Complete and perfect):
Problem: Find the sum of 2 + 3.
Solution: The sum of 2 + 3 is 5.
Student Answer: 5
Reasoning: The answer is exactly correct with no errors or omissions.
<json>
{"response": "correct"}
</json>

Example 2 - ALMOST (Minor calculation error):
Problem: Compute 15 * 12.
Solution: 15 * 12 = 180.
Student Answer: 15 * 12 = 150 + 30 = 190. (Student made a small addition error)
Reasoning: The student used the correct method but made a tiny arithmetic error.
<json>
{"response": "almost"}
</json>

Example 3 - PARTIAL (Missing one of multiple solutions):
Problem: Solve x^2 = 4 for x.
Solution: x = 2 or x = -2.
Student Answer: x = 2
Reasoning: The student found one correct solution but missed the other. Significant omission.
<json>
{"response": "partial"}
</json>

Example 4 - INCORRECT (Wrong approach):
Problem: Find the area of a circle with radius 3.
Solution: A = πr² = 9π
Student Answer: A = 2πr = 6π
Reasoning: The student confused area formula with circumference formula.
<json>
{"response": "incorrect"}
</json>

Example 5 - ALMOST vs PARTIAL distinction:
Problem: Solve the system: x + y = 10, x - y = 2.
Solution: Adding equations: 2x = 12, so x = 6. Then y = 4.
Student Answer: x = 6 (found x correctly but didn't find y).
Reasoning: This is PARTIAL not ALMOST because finding only one variable is a significant omission.
<json>
{"response": "partial"}
</json>
"""


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.call_count = 0
        self.error_count = 0

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        self.call_count += 1
        logger.info(f"TaskAgent forward call #{self.call_count}")
        
        # Extract fields from inputs
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        instruction = f"""You are an expert grading agent for {domain}. Your task is to carefully analyze the student's answer and provide a detailed evaluation.

## Problem Statement:
{problem}

## Correct Solution:
{solution}

## Grading Guidelines (CRITICAL - FOLLOW THESE EXACTLY):
{grading_guidelines}

## Student's Answer:
{student_answer}

## Grading Rubric:

**CORRECT**: The answer is fully correct, complete, and demonstrates thorough understanding.
- All required steps are present and correct
- Final answer matches the solution exactly
- No errors, omissions, or issues of any significance

**ALMOST**: The answer is nearly correct with ONLY minor issues.
- Core approach and all major steps are correct
- Only tiny calculation errors or trivial notation issues
- Missing only the most trivial final step
- KEY: The error is so minor it barely affects the solution quality

**PARTIAL**: The answer has SIGNIFICANT gaps or errors, but shows some valid work.
- Some correct steps or partial understanding shown
- Missing KEY components (e.g., only found 1 of 2 solutions)
- Has significant errors that affect the result
- KEY: There's real missing content or significant errors, not just tiny slips

**INCORRECT**: The answer is wrong or shows no valid mathematical reasoning.
- No valid mathematical reasoning or approach
- Completely wrong method or nonsensical answer

## Critical Decision Guidelines:

**ALMOST vs PARTIAL - This is the most important distinction:**

Use **ALMOST** when ALL of these are true:
- The student clearly demonstrates understanding of the core concept
- The error is truly minor (small arithmetic slip, trivial notation issue)
- The solution is essentially complete with only a tiny gap
- Examples: 2+2=5, forgot to write "x=" before the answer
- **From guidelines: "verification contains minor mistakes only" → ALMOST**

Use **PARTIAL** when ANY of these are true:
- Missing significant portions of the answer (e.g., only 1 of 2 solutions)
- Missing key reasoning steps or explanations
- Examples: "x^2=4, student says x=2" (missing -2 is significant)
- **From guidelines: "Found a correct invariant" (without more) → PARTIAL**

**Quick Test:** If the student fixed their error in 10 seconds with a quick check, it's ALMOST. If they would need to redo significant work, it's PARTIAL.

## Few-Shot Examples:
{FEW_SHOT_EXAMPLES}

## Grading Instructions:
1. Carefully read the problem and understand what is being asked.
2. **Study the grading guidelines first** - they define what each label means for THIS specific problem.
3. Compare the student's answer against the correct solution step by step.
4. Check if the student shows valid mathematical work and reasoning.
5. Identify any errors, omissions, or misconceptions.
6. **Map to guideline criteria:**
   - Does the student meet the "Partial" criteria in the guidelines?
   - Does the student meet the "Almost" criteria?
   - Does the student meet the "Correct" criteria?
7. **Classify the error severity:**
   - Is it a tiny arithmetic slip or trivial omission? → Consider ALMOST
   - Is it missing significant content or has major errors? → Consider PARTIAL
8. Select the label that best matches the quality of the student's work.

## Response Format:
You must respond with EXACTLY ONE of these four labels in JSON format:
- "correct" - The answer is fully correct and complete
- "almost" - The answer is nearly correct with ONLY truly minor issues  
- "partial" - The answer has SIGNIFICANT gaps or errors (but some valid work)
- "incorrect" - The answer is wrong or shows no valid reasoning

<json>
{{
    "response": "correct|almost|partial|incorrect"
}}
</json>

IMPORTANT: Your response field must contain ONLY one of these four exact words: correct, almost, partial, or incorrect."""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            logger.error(f"Failed to get LLM response: {e}")
            self.error_count += 1
            return "incorrect", []

        # Extract prediction from JSON
        prediction = "incorrect"  # Default to incorrect if extraction fails
        try:
            if msg_history and len(msg_history) > 0:
                extracted = _extract_jsons(msg_history[-1]["text"])
                if extracted and len(extracted) > 0 and "response" in extracted[-1]:
                    raw_prediction = extracted[-1]["response"]
                    prediction = _normalize_prediction(raw_prediction)
                    logger.info(f"Successfully extracted prediction: {prediction}")
                else:
                    # Try to extract label from raw text if JSON parsing fails
                    text = msg_history[-1].get("text", "")
                    prediction = _extract_label_from_text(text)
                    logger.info(f"Extracted label from text: {prediction}")
            else:
                logger.warning("Empty message history from LLM")
        except Exception as e:
            self.error_count += 1
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history

    def get_stats(self) -> dict:
        """Return agent statistics for monitoring."""
        return {
            "call_count": self.call_count,
            "error_count": self.error_count,
            "success_rate": (self.call_count - self.error_count) / max(self.call_count, 1),
        }
