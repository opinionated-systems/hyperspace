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
    
    # Check for exact word matches with word boundaries (but be careful about context)
    words = re.findall(r'\b\w+\b', prediction_lower)
    # Only use word match if it's a standalone category word
    for word in words:
        if word in valid_categories:
            # Check for negation context
            idx = prediction_lower.find(word)
            if idx != -1:
                before = prediction_lower[max(0, idx-10):idx]
                if "not " not in before and "n't " not in before:
                    return word
    
    # Priority-based phrase matching with STRICT ordering
    # The key insight: "almost" should be very conservative - only when solution is nearly perfect
    # "partial" is for solutions with meaningful progress but significant gaps
    # "incorrect" is for solutions with no valid mathematical progress
    
    # First, check for clear "incorrect" indicators (highest priority for wrong answers)
    incorrect_indicators = [
        "no valid mathematical", "no progress", "fundamentally wrong", "completely wrong",
        "totally wrong", "nonsense", "gibberish", "invalid approach", "wrong approach",
        "no understanding", "no solution", "failed", "failure", "no attempt",
        "blank", "empty", "no answer", "missing", "irrelevant", "off topic",
        "fundamental misunderstanding", "trivial", "not valid", "does not understand",
        "did not understand", "0%", "zero", "nothing correct", "no correct",
        "completely incorrect", "totally incorrect", "absolutely wrong",
        "fundamental error", "critical error", "serious mistake", "major error"
    ]
    for indicator in incorrect_indicators:
        if indicator in prediction_lower:
            return "incorrect"
    
    # Check for "almost" indicators - be VERY conservative
    # "Almost" means: 90-99% complete, tiny fixable errors, would be perfect if errors fixed
    almost_indicators = [
        "almost correct", "nearly correct", "almost complete", "nearly complete",
        "minor mistake only", "minor error only", "small error only",
        "tiny mistake", "slight error only", "essentially correct",
        "fundamentally correct", "correct approach with minor",
        "correct method with minor", "nearly perfect", "almost perfect",
        "mostly correct with minor", "correct except for minor",
        "correct except for small", "correct except for tiny",
        "would be perfect if", "would be correct if", "only a minor",
        "just a minor", "just a small", "only a small", "minor typo only",
        "small typo only", "minor calculation error", "small calculation error",
        "minor arithmetic error", "slight miscalculation", "tiny miscalculation"
    ]
    for indicator in almost_indicators:
        if indicator in prediction_lower:
            return "almost"
    
    # Check for "partial" indicators - broader than "almost" but requires valid progress
    partial_indicators = [
        "partial credit", "partially correct", "incomplete solution",
        "incomplete proof", "partial solution", "partial progress",
        "significant progress", "meaningful progress", "valid insight",
        "key insight", "correct lemma", "proved a lemma", "identified key",
        "some understanding", "partial proof", "partial result",
        "some correct", "partially valid", "partial understanding",
        "correct framework", "valid reasoning", "partial success",
        "some valid", "partially worked", "partial attempt",
        "partially successful", "some correct steps", "valid partial",
        "partially complete", "not fully correct", "not completely correct",
        "partial marks", "some credit", "partially right", "incomplete but",
        "partial understanding demonstrated", "some valid reasoning",
        "made progress", "good start", "on the right track",
        "correct direction", "valid approach", "substantial progress",
        "important insight", "major progress", "significant portion"
    ]
    for indicator in partial_indicators:
        if indicator in prediction_lower:
            return "partial"
    
    # Check for "correct" indicators - must be very clear
    correct_indicators = [
        "fully correct", "completely correct", "entirely correct", "totally correct",
        "correct solution", "correct answer", "full marks", "full credit",
        "complete solution", "perfect solution", "excellent solution",
        "full understanding", "all correct", "everything correct",
        "correct throughout", "valid solution", "valid proof",
        "correct proof", "sound reasoning", "correct reasoning",
        "100% correct", "complete and correct", "fully solved"
    ]
    for indicator in correct_indicators:
        if indicator in prediction_lower:
            return "correct"
    
    # Check for negated correct (which means incorrect)
    if ("not correct" in prediction_lower or 
        "not fully" in prediction_lower or
        "not completely" in prediction_lower or
        "not entirely" in prediction_lower or
        "not totally" in prediction_lower):
        return "incorrect"
    
    # Check for "wrong" or "error" without "minor" or "small" qualifier
    if ("wrong" in prediction_lower or "error" in prediction_lower or "mistake" in prediction_lower):
        # Check if it's qualified as minor/small/tiny
        if not any(q in prediction_lower for q in ["minor", "small", "tiny", "slight", "trivial"]):
            return "incorrect"
    
    # Fallback: if just "correct" appears without negation or qualification
    if "correct" in prediction_lower:
        # Check for partial qualifiers
        if any(q in prediction_lower for q in ["partially", "not fully", "incomplete"]):
            return "partial"
        # Check for almost qualifiers  
        if any(q in prediction_lower for q in ["almost", "nearly", "mostly"]):
            return "almost"
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

**"correct"** - Complete, correct solution with sound mathematical reasoning. May have negligible typos. The solution demonstrates full understanding and arrives at the correct answer with valid proof.

**"almost"** - Nearly complete solution (90-99%) with correct approach. Only MINOR errors (calculation mistakes, small typos, missing one trivial case). The solution is fundamentally sound and would be perfect if minor errors were fixed. KEY TEST: If the small errors were corrected, would this be a complete correct solution? If YES → "almost".

**"partial"** - Meaningful progress (10-50%) with SOME valid mathematical content (proved a lemma, identified key insight, correct approach but incomplete execution). Has SIGNIFICANT gaps or missing major steps. KEY TEST: Is there valid mathematical content that contributes to the solution, despite significant gaps? If YES → "partial".

**"incorrect"** - No valid mathematical progress. Wrong approach, fundamental misunderstanding, nonsense, or empty response. KEY TEST: Is there any valid mathematical contribution toward solving the problem? If NO → "incorrect".

## CRITICAL DECISION RULES - APPLY IN ORDER:
1. **Is it fully correct?** (Complete proof, correct answer, sound reasoning) → "correct"
2. **Is it nearly perfect with only tiny fixable errors?** (90-99% complete, minor calculation errors, small typos, one trivial case missing - would be perfect if fixed) → "almost"
3. **Is there meaningful valid progress despite significant gaps?** (Proved a lemma, identified key insight, correct approach started but incomplete, some valid reasoning) → "partial"
4. **Otherwise** (wrong approach, no valid progress, nonsense) → "incorrect"

## IMPORTANT DISTINCTIONS:
- "almost" vs "partial": "almost" means the solution is fundamentally correct but has minor flaws. "partial" means there's valid mathematical work but significant portions are missing or wrong.
- "partial" vs "incorrect": "partial" requires at least some valid mathematical contribution (a lemma, insight, or partial proof). "incorrect" means no valid progress at all.

## Output Format:
Respond ONLY in this JSON format:
<json>
{{
    "response": "correct"
}}</json>

The response value MUST be exactly one of: "correct", "almost", "partial", or "incorrect" (lowercase). Be conservative: when in doubt between two categories, choose the lower one."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON using flexible extraction
        prediction = "incorrect"  # Default to incorrect for safety
        try:
            # Validate msg_history structure
            if not msg_history or not isinstance(msg_history, list):
                self.log_fn("Invalid msg_history structure")
                return str(prediction), msg_history
            
            last_message = msg_history[-1]
            if not isinstance(last_message, dict) or "text" not in last_message:
                self.log_fn("Last message missing 'text' key")
                return str(prediction), msg_history
            
            text_content = last_message["text"]
            if not isinstance(text_content, str):
                self.log_fn(f"Text content is not a string: {type(text_content)}")
                return str(prediction), msg_history
            
            extracted = _extract_json_flexible(text_content)
            if extracted:
                last_json = extracted[-1]
                if isinstance(last_json, dict):
                    # Try multiple possible keys for the response
                    response_keys = ["response", "answer", "prediction", "grade", "result", "category"]
                    raw_prediction = None
                    for key in response_keys:
                        if key in last_json:
                            raw_prediction = last_json[key]
                            break
                    
                    if raw_prediction is not None:
                        prediction = _normalize_prediction(str(raw_prediction))
                        self.log_fn(f"Raw prediction: {raw_prediction} -> Normalized: {prediction}")
                    else:
                        self.log_fn(f"JSON missing response key. Available keys: {list(last_json.keys())}")
                else:
                    self.log_fn(f"Last JSON is not a dict: {type(last_json)}")
            else:
                # Try to extract directly from text if no JSON found
                text_lower = text_content.lower()
                prediction = _normalize_prediction(text_lower)
                self.log_fn(f"No JSON found, extracted from text: {prediction}")
        except json.JSONDecodeError as e:
            self.log_fn(f"JSON decode error: {e}")
            # Try to extract from raw text as fallback
            try:
                if msg_history and isinstance(msg_history, list) and isinstance(msg_history[-1], dict):
                    text_content = msg_history[-1].get("text", "")
                    prediction = _normalize_prediction(text_content.lower())
            except Exception:
                pass
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            prediction = "incorrect"

        return str(prediction), msg_history
