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


def _extract_json_flexible(text: str) -> list[dict] | None:
    """Extract JSON objects using multiple strategies.
    
    Tries multiple extraction methods in order of preference:
    1. <json>...</json> tags
    2. ```json...``` code blocks
    3. Bare JSON objects (starting with { and ending with })
    """
    # Try <json> tags first
    results = _extract_jsons(text)
    if results:
        return results
    
    # Try ```json code blocks
    results = []
    pattern = r'```json\s*(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            continue
    if results:
        return results
    
    # Try bare JSON objects using balanced brace matching
    results = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == '{':
            depth = 1
            j = i + 1
            in_string = False
            escape_next = False
            while j < n and depth > 0:
                char = text[j]
                if escape_next:
                    escape_next = False
                elif char == '\\':
                    escape_next = True
                elif char == '"' and not in_string:
                    in_string = True
                elif char == '"' and in_string:
                    in_string = False
                elif not in_string:
                    if char == '{':
                        depth += 1
                    elif char == '}':
                        depth -= 1
                j += 1
            if depth == 0:
                candidate = text[i:j]
                try:
                    results.append(json.loads(candidate))
                except json.JSONDecodeError:
                    pass
                i = j
                continue
        i += 1
    
    return results or None


def _normalize_prediction(prediction: str) -> str:
    """Normalize prediction to one of the four valid categories.
    
    Args:
        prediction: Raw prediction string from LLM
        
    Returns:
        Normalized prediction: "correct", "almost", "partial", or "incorrect"
    """
    if not isinstance(prediction, str):
        return "incorrect"
    
    prediction_lower = prediction.lower().strip()
    
    # Direct matches (exact match takes highest priority)
    valid_categories = ["correct", "almost", "partial", "incorrect"]
    for cat in valid_categories:
        if prediction_lower == cat:
            return cat
    
    # Check for exact word matches with word boundaries
    words = re.findall(r'\b\w+\b', prediction_lower)
    for word in words:
        if word in valid_categories:
            return word
    
    # Priority-based phrase matching (order matters!)
    # Priority: almost > partial > incorrect > correct
    
    # Check for "almost" indicators first (highest priority after exact match)
    almost_indicators = [
        "almost", "nearly", "minor mistake", "minor error", "small error",
        "slight error", "tiny mistake", "nearly correct", "mostly correct",
        "minor issue", "small issue", "trivial mistake", "insignificant error",
        "minor flaw", "small flaw", "tiny error", "slight mistake",
        "essentially correct", "fundamentally correct", "correct approach",
        "correct method", "right approach", "right method", "small slip",
        "minor slip", "tiny slip", "nearly right", "mostly right",
        "minor miscalculation", "small miscalculation", "tiny miscalculation",
        "just a small", "only a minor", "minor typo", "small typo",
        "calculation error", "arithmetic error", "minor arithmetic",
        "slight imprecision", "minor imprecision", "nearly perfect",
        "almost perfect", "very close", "minor omission", "small omission",
        "90-99%", "90%", "95%", "99%", "essentially complete"
    ]
    for indicator in almost_indicators:
        if indicator in prediction_lower:
            return "almost"
    
    # Check for "partial" indicators
    partial_indicators = [
        "partial", "incomplete", "partially", "some progress", "meaningful progress",
        "significant gaps", "major gaps", "incomplete solution", "partial solution",
        "partial credit", "some understanding", "partially correct", "half correct",
        "missing parts", "incomplete answer", "fragmented", "some valid",
        "incomplete proof", "partial proof", "started correctly", "on the right track",
        "good start", "some correct", "partly correct", "halfway", "significant portion",
        "major portion missing", "incomplete reasoning", "missing steps",
        "10-50%", "10%", "20%", "30%", "40%", "50%", "in progress"
    ]
    for indicator in partial_indicators:
        if indicator in prediction_lower:
            return "partial"
    
    # Check for "incorrect" indicators (before "correct" to catch negations)
    incorrect_indicators = [
        "incorrect", "wrong", "error", "not correct", "not right", "false",
        "invalid", "no progress", "meaningless", "fundamentally wrong",
        "completely wrong", "totally wrong", "essentially wrong", "no understanding",
        "fundamental misunderstanding", "empty", "trivial", "nonsense",
        "not valid", "invalid approach", "wrong approach", "incorrect approach",
        "no valid", "no meaningful", "failed to", "does not understand",
        "did not understand", "no attempt", "blank", "no answer",
        "0%", "zero", "none", "nothing correct"
    ]
    for indicator in incorrect_indicators:
        if indicator in prediction_lower:
            return "incorrect"
    
    # Check for "correct" indicators (lowest priority for substrings)
    correct_indicators = [
        "fully correct", "completely correct", "entirely correct", "totally correct",
        "correct solution", "correct answer", "full marks", "full credit",
        "complete solution", "perfect", "excellent", "full understanding",
        "all correct", "everything correct", "correct throughout", "valid solution",
        "valid proof", "correct proof", "sound reasoning", "correct reasoning",
        "100%", "complete and correct"
    ]
    for indicator in correct_indicators:
        if indicator in prediction_lower:
            return "correct"
    
    # Fallback: if just "correct" appears without negation
    if "correct" in prediction_lower and "not " not in prediction_lower:
        return "correct"
    
    # Default fallback to incorrect for safety
    return "incorrect"


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
        # Build a more structured prompt with clear instructions
        domain = inputs.get('domain', 'Mathematics')
        problem = inputs.get('problem', '')
        solution = inputs.get('solution', '')
        grading_guidelines = inputs.get('grading_guidelines', '')
        student_answer = inputs.get('student_answer', '')
        
        instruction = f"""You are an expert grader for {domain} problems. Evaluate the student's answer and classify it into exactly ONE category.

## Problem:
{problem}

## Correct Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Classification Categories (choose exactly one):

**"correct"** - Complete, correct solution with sound mathematical reasoning. May have negligible typos.

**"almost"** - Nearly complete (90-99%) with correct approach. Only MINOR errors (calculation mistakes, small typos, missing one trivial case). KEY TEST: If errors were fixed, would this be perfect? If YES → "almost".

**"partial"** - Meaningful progress (10-50%) with SOME valid understanding (proved a lemma, identified key insight, correct approach but incomplete). Has SIGNIFICANT gaps. KEY TEST: Is there valid mathematical content despite gaps? If YES → "partial".

**"incorrect"** - No valid mathematical progress. Wrong approach, nonsense, or empty response.

## CRITICAL DECISION RULES:
1. First check: Is it fully correct? → "correct"
2. If not, ask: Is it 90-99% complete with only tiny fixable errors? → "almost"
3. If not, ask: Is there ANY valid mathematical progress (lemma, insight, partial proof)? → "partial"
4. Otherwise → "incorrect"

## Output Format:
Respond ONLY in this JSON format:
<json>
{{
    "response": "correct"
}}</json>

The response value MUST be exactly one of: "correct", "almost", "partial", or "incorrect" (lowercase)."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON using flexible extraction
        prediction = "incorrect"  # Default to incorrect for safety
        try:
            extracted = _extract_json_flexible(msg_history[-1]["text"])
            if extracted:
                last_json = extracted[-1]
                if isinstance(last_json, dict) and "response" in last_json:
                    raw_prediction = last_json["response"]
                    prediction = _normalize_prediction(raw_prediction)
                    self.log_fn(f"Raw prediction: {raw_prediction} -> Normalized: {prediction}")
                else:
                    self.log_fn(f"JSON missing 'response' key: {last_json}")
            else:
                # Try to extract directly from text if no JSON found
                text = msg_history[-1]["text"].lower()
                prediction = _normalize_prediction(text)
                self.log_fn(f"No JSON found, extracted from text: {prediction}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            prediction = "incorrect"

        return str(prediction), msg_history
