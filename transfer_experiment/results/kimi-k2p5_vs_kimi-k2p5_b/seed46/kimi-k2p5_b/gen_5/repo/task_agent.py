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
            continue
    return results or None


def _extract_json_from_markdown(text: str) -> list[dict] | None:
    """Extract JSON objects from markdown code blocks."""
    results = []
    pattern = r'```(?:json)?\s*(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            continue
    return results or None


def _extract_json_braces(text: str) -> list[dict] | None:
    """Extract JSON objects directly from curly braces."""
    results = []
    i = 0
    while i < len(text):
        if text[i] == '{':
            brace_count = 1
            j = i + 1
            while j < len(text) and brace_count > 0:
                if text[j] == '{':
                    brace_count += 1
                elif text[j] == '}':
                    brace_count -= 1
                j += 1
            if brace_count == 0:
                try:
                    obj = json.loads(text[i:j])
                    results.append(obj)
                except json.JSONDecodeError:
                    pass
            i = j if j > i else i + 1
        else:
            i += 1
    return results or None


def _extract_label_from_text(text: str) -> str | None:
    """Extract grading label from raw text using multiple strategies."""
    text_lower = text.lower()
    
    # Look for explicit patterns like "grade: correct" or "evaluation: partial"
    # Include "almost" as a valid label
    patterns = [
        r'grade[d]?\s*[:=]\s*(correct|incorrect|partial|almost)',
        r'evaluation\s*[:=]\s*(correct|incorrect|partial|almost)',
        r'label\s*[:=]\s*(correct|incorrect|partial|almost)',
        r'response\s*[:=]\s*(correct|incorrect|partial|almost)',
        r'classification\s*[:=]\s*(correct|incorrect|partial|almost)',
        r'final\s+(?:grade|evaluation|label|response)\s*[:=]?\s*(correct|incorrect|partial|almost)',
        r'(?:the\s+)?(?:answer|result)\s+(?:is\s+)?["\']?(correct|incorrect|partial|almost)["\']?',
        r'(?:i\s+)?(?:would\s+)?(?:classify|grade|label)\s+(?:this\s+)?(?:as\s+)?["\']?(correct|incorrect|partial|almost)["\']?',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1)
    
    # Check for standalone labels with word boundaries in priority order
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


def _normalize_prediction(pred: str | None) -> str:
    """Normalize prediction to one of the valid labels."""
    if not pred or not isinstance(pred, str):
        return "None"
    
    pred = pred.strip().lower()
    
    # Direct match - include "almost" as valid label
    if pred in ["correct", "incorrect", "partial", "almost"]:
        return pred
    
    # Check for substring matches - be careful with ordering
    if "almost" in pred:
        return "almost"
    if "incorrect" in pred:
        return "incorrect"
    if "partial" in pred:
        return "partial"
    if "correct" in pred:
        return "correct"
    
    return "None"


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
        
        instruction = f"""You are an expert mathematics grader for International Mathematical Olympiad (IMO) style problems.

Your task is to evaluate a student's answer to a mathematical problem and classify it into exactly one of these four categories.

## LABEL DEFINITIONS (READ CAREFULLY):

**"correct"** - The student's answer is fully correct and complete:
- All key steps and reasoning match the official solution
- The proof is rigorous with no gaps or errors
- Final answer is correct and properly justified
- All required elements from the grading guidelines are present

**"almost"** - The student was very close to a complete solution:
- Found the correct overall strategy/approach
- Has the right key insights and main lemmas
- Made substantial progress toward the complete solution
- Contains only MINOR gaps, small errors, or incomplete verification
- The solution is "nearly there" - just needs small fixes
- Award this when the student clearly understood the problem but made minor mistakes

**"partial"** - The student made some progress but has significant gaps:
- Demonstrates meaningful understanding of SOME parts
- Found some correct insights or intermediate results
- BUT: Missing key elements, major gaps in reasoning, or far from complete
- The solution has correct pieces but doesn't form a coherent path to the answer
- Award this when the student started correctly but didn't get far

**"incorrect"** - The student's answer is fundamentally wrong:
- Contains major errors in reasoning
- Shows no meaningful understanding of the problem
- Completely wrong approach or strategy
- No correct key insights or progress toward the solution

## KEY DISTINCTIONS:
- "almost" vs "partial": "almost" means the student had the RIGHT approach and was CLOSE to finishing (minor issues only). "partial" means the student made SOME correct progress but is FAR from a complete solution.
- When in doubt: If the student had the right strategy but small errors → "almost". If the student only got some isolated correct pieces → "partial".

## Problem:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Instructions:
1. Read the problem, official solution, and grading guidelines carefully
2. Analyze the student's answer step by step against the official solution
3. Identify what the student got RIGHT (key insights, correct lemmas, right strategy)
4. Identify what is MISSING or WRONG (gaps, errors, incomplete parts)
5. Determine the label based on:
   - Did they have the right overall strategy? (almost/correct vs partial/incorrect)
   - How close are they to the complete solution? (correct/almost vs partial)
   - Are the errors minor or major? (almost vs partial)
6. Make your final determination

IMPORTANT: You must respond with a valid JSON object in this exact format:
<json>
{{
    "reasoning": "Your detailed analysis here...",
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
        raw_text = msg_history[-1]["text"] if msg_history else ""
        
        # Keys to check in JSON objects (in priority order)
        json_keys = ["response", "label", "grade", "evaluation", "classification", "result", "answer", "prediction"]
        
        try:
            # Strategy 1: Try to extract from <json> tags
            extracted = _extract_jsons(raw_text)
            if extracted:
                for json_obj in extracted:
                    for key in json_keys:
                        if key in json_obj:
                            prediction = _normalize_prediction(str(json_obj[key]))
                            if prediction != "None":
                                break
                    if prediction != "None":
                        break
            
            # Strategy 2: Try markdown code blocks
            if prediction == "None":
                extracted = _extract_json_from_markdown(raw_text)
                if extracted:
                    for json_obj in extracted:
                        for key in json_keys:
                            if key in json_obj:
                                prediction = _normalize_prediction(str(json_obj[key]))
                                if prediction != "None":
                                    break
                        if prediction != "None":
                            break
            
            # Strategy 3: Try raw JSON braces
            if prediction == "None":
                extracted = _extract_json_braces(raw_text)
                if extracted:
                    for json_obj in extracted:
                        for key in json_keys:
                            if key in json_obj:
                                prediction = _normalize_prediction(str(json_obj[key]))
                                if prediction != "None":
                                    break
                        if prediction != "None":
                            break
            
            # Strategy 4: Direct text extraction as fallback
            if prediction == "None":
                text_pred = _extract_label_from_text(raw_text)
                if text_pred:
                    prediction = text_pred
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        self.log_fn(f"Final prediction: {prediction}")
        return str(prediction), msg_history
