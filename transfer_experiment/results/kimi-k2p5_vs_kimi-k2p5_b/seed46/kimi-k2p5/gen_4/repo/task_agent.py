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
    # Match ```json ... ``` or ``` ... ``` blocks
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
    # Try to find JSON objects by looking for balanced braces
    i = 0
    while i < len(text):
        if text[i] == '{':
            # Try to find matching closing brace
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
    
    # Look for explicit label mentions with word boundaries
    # Priority order matters - check more specific terms first
    
    # Check for "almost" first (before "correct" to avoid confusion)
    if re.search(r'\balmost\b', text_lower):
        return "almost"
    
    # Check for "incorrect" (more specific than "correct")
    if re.search(r'\bincorrect\b', text_lower):
        return "incorrect"
    
    # Check for "partial"
    if re.search(r'\bpartial\b', text_lower):
        return "partial"
    
    # Check for "correct" (but not "incorrect" - already checked above)
    if re.search(r'\bcorrect\b', text_lower):
        return "correct"
    
    return None


def _normalize_prediction(pred: str | None) -> str:
    """Normalize prediction to one of the valid labels."""
    if not pred or not isinstance(pred, str):
        return "None"
    
    pred = pred.strip().lower()
    
    # Direct match
    if pred in ["correct", "almost", "partial", "incorrect"]:
        return pred
    
    # Check for substring matches - priority order matters
    # "almost" must be checked before "correct"
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

    def _check_grading_guidelines_for_hint(self, grading_guidelines: str) -> str | None:
        """Check if grading guidelines contain explicit hints about the expected label.
        
        This helps guide the model when the guidelines explicitly mention categories.
        """
        if not grading_guidelines:
            return None
            
        guidelines_lower = grading_guidelines.lower()
        
        # Check for explicit "(Almost)" marker in guidelines
        if "(almost)" in guidelines_lower:
            return "almost"
        
        return None

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract fields from inputs
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        instruction = f"""You are an expert mathematics grader for International Mathematical Olympiad (IMO) style problems.

Your task is to evaluate a student's answer to a mathematical problem and classify it into EXACTLY ONE of these four categories:

1. "correct" - The student's answer is fully correct, complete, and matches the official solution. All reasoning is sound, all claims are properly justified, and the final answer is correct.

2. "almost" - The answer demonstrates a complete or nearly complete understanding of the solution approach, with the main ideas and strategy being correct. However, there are minor verification mistakes, small computational errors, or slight gaps in justification that don't invalidate the core reasoning. The student has essentially solved the problem but made minor mistakes in execution.

3. "partial" - The answer has some correct elements and shows understanding of key concepts, but is incomplete, missing significant components, or has substantial gaps in the reasoning. The student has made meaningful progress but hasn't reached a complete or nearly complete solution.

4. "incorrect" - The answer is wrong, fundamentally flawed, or shows no meaningful understanding of the problem. The approach is misguided or the reasoning contains critical errors.

## Problem:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Instructions:
1. Carefully read the problem, official solution, and grading guidelines
2. Analyze the student's answer step by step
3. Compare the student's reasoning and conclusion with the official solution
4. Consider the grading guidelines for partial credit
5. Make a final determination using EXACTLY one of: "correct", "almost", "partial", or "incorrect"

## Key Distinctions:
- "correct" vs "almost": "correct" means everything is right; "almost" means the main solution strategy is correct but there are minor execution errors or small gaps.
- "almost" vs "partial": "almost" means the solution is nearly complete (just minor issues); "partial" means significant work remains to be done.
- "partial" vs "incorrect": "partial" shows meaningful understanding; "incorrect" shows fundamental misunderstanding.

## Examples of "almost" (for reference):
- The student found the correct invariant and approach but made a minor calculation error in verification
- The student proved the main theorem but missed a small case or made a minor logical slip that doesn't affect the core argument
- The solution is essentially complete but contains minor mistakes in the final steps

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed analysis of the student's answer...",
    "response": "correct" | "almost" | "partial" | "incorrect"
}}
</json>

The "response" field MUST be exactly one of: "correct", "almost", "partial", or "incorrect" (lowercase, no extra text)."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction using multiple strategies
        prediction = "None"
        raw_text = msg_history[-1]["text"] if msg_history else ""
        
        try:
            # Strategy 1: Try to extract from <json> tags
            extracted = _extract_jsons(raw_text)
            if extracted:
                for json_obj in extracted:
                    for key in ["response", "answer", "label", "grade", "evaluation", "result"]:
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
                        for key in ["response", "answer", "label", "grade", "evaluation", "result"]:
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
                        for key in ["response", "answer", "label", "grade", "evaluation", "result"]:
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

        # Check if grading guidelines provide a hint for uncertain predictions
        if prediction == "None" or prediction not in ["correct", "almost", "partial", "incorrect"]:
            hint = self._check_grading_guidelines_for_hint(grading_guidelines)
            if hint:
                prediction = hint
                self.log_fn(f"Using hint from grading guidelines: {prediction}")
        
        # Final fallback: if prediction is still None, try to use text extraction on full response
        if prediction == "None":
            text_pred = _extract_label_from_text(raw_text)
            if text_pred:
                prediction = text_pred
        
        self.log_fn(f"Final prediction: {prediction}")
        return str(prediction), msg_history
