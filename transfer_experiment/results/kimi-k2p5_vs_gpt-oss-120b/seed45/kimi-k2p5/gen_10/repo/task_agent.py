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
    if not isinstance(text, str) or not text:
        return None
    
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
    if not isinstance(text, str) or not text:
        return None
    
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
        "fundamental error", "critical error", "serious mistake", "major error",
        "not correct", "is wrong", "are wrong", "was wrong", "were wrong",
        "incorrect solution", "incorrect answer", "incorrect approach",
        "no valid", "invalid reasoning", "flawed reasoning", "unsound"
    ]
    for indicator in incorrect_indicators:
        if indicator in prediction_lower:
            return "incorrect"
    
    # Check for "almost" indicators - be VERY conservative
    # "Almost" means: 90-99% complete, tiny fixable errors, would be perfect if errors fixed
    # Key distinction: "almost" requires the solution to be fundamentally correct
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
        "minor arithmetic error", "slight miscalculation", "tiny miscalculation",
        "minor oversight", "small oversight", "trivial error", "insignificant error",
        "cosmetic error", "formatting error", "notation error", "sign error",
        "minor gap", "small gap", "trivial gap", "nearly solved",
        "almost solved", "essentially solved", "practically correct",
        "one minor", "a minor", "the minor", "single minor",
        "one small", "a small", "the small", "single small",
        "one tiny", "a tiny", "the tiny", "single tiny"
    ]
    for indicator in almost_indicators:
        if indicator in prediction_lower:
            return "almost"
    
    # Check for "partial" indicators - broader than "almost" but requires valid progress
    # "Partial" means: 10-50% complete, has valid mathematical content but significant gaps
    # Key distinction: "partial" has meaningful progress but is NOT nearly complete
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
        "important insight", "major progress", "significant portion",
        "half correct", "partially worked out", "some progress",
        "incomplete execution", "started correctly", "began correctly",
        "correct beginning", "valid start", "some valid steps",
        "partially valid solution", "incomplete but valid",
        "missing steps", "incomplete reasoning", "unfinished",
        "needs more work", "requires completion", "incomplete answer",
        "partial answer", "some right", "partly correct", "partly right",
        "50%", "halfway", "incomplete work", "unfinished solution"
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
        "100% correct", "complete and correct", "fully solved",
        "correctly solved", "properly solved", "well done",
        "excellent work", "perfect work", "flawless", "no errors",
        "no mistakes", "accurate", "precise", "exactly right"
    ]
    for indicator in correct_indicators:
        if indicator in prediction_lower:
            return "correct"
    
    # Check for negated correct (which means incorrect)
    if ("not correct" in prediction_lower or 
        "not fully" in prediction_lower or
        "not completely" in prediction_lower or
        "not entirely" in prediction_lower or
        "not totally" in prediction_lower or
        "not accurate" in prediction_lower or
        "not right" in prediction_lower):
        return "incorrect"
    
    # Check for "wrong" or "error" without "minor" or "small" qualifier
    if ("wrong" in prediction_lower or "error" in prediction_lower or "mistake" in prediction_lower):
        # Check if it's qualified as minor/small/tiny
        if not any(q in prediction_lower for q in ["minor", "small", "tiny", "slight", "trivial", "cosmetic", "insignificant"]):
            return "incorrect"
    
    # Check for "mostly" - this is tricky, need context
    if "mostly" in prediction_lower:
        # If "mostly correct" or "mostly right" -> almost
        if "mostly correct" in prediction_lower or "mostly right" in prediction_lower:
            return "almost"
        # If "mostly wrong" or "mostly incorrect" -> incorrect
        if "mostly wrong" in prediction_lower or "mostly incorrect" in prediction_lower:
            return "incorrect"
        # If "mostly incomplete" or "mostly partial" -> partial
        if "mostly incomplete" in prediction_lower or "mostly partial" in prediction_lower:
            return "partial"
    
    # Check for "nearly" - similar to "almost"
    if "nearly" in prediction_lower:
        if "nearly correct" in prediction_lower or "nearly right" in prediction_lower or "nearly complete" in prediction_lower:
            return "almost"
        if "nearly wrong" in prediction_lower or "nearly incorrect" in prediction_lower:
            return "incorrect"
    
    # Fallback: if just "correct" appears without negation or qualification
    if "correct" in prediction_lower:
        # Check for strong partial qualifiers first
        if any(q in prediction_lower for q in ["partially", "not fully", "incomplete", "unfinished", "missing"]):
            return "partial"
        # Check for almost qualifiers  
        if any(q in prediction_lower for q in ["almost", "nearly", "mostly", "essentially", "practically"]):
            return "almost"
        return "correct"
    
    # Check for "right" as synonym for correct
    if "right" in prediction_lower:
        if any(q in prediction_lower for q in ["partially", "not fully", "incomplete"]):
            return "partial"
        if any(q in prediction_lower for q in ["almost", "nearly", "mostly"]):
            return "almost"
        if "not right" in prediction_lower or "wrong" in prediction_lower:
            return "incorrect"
        return "correct"
    
    # Check for "solved" indicators
    if "solved" in prediction_lower:
        if "not solved" in prediction_lower or "unsolved" in prediction_lower:
            return "incorrect"
        if "partially solved" in prediction_lower or "incompletely solved" in prediction_lower:
            return "partial"
        if "almost solved" in prediction_lower or "nearly solved" in prediction_lower:
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

**"correct"** - Complete, correct solution with sound mathematical reasoning. May have negligible typos. The solution demonstrates full understanding and arrives at the correct answer with valid proof. The solution is 100% complete.

**"almost"** - Nearly complete solution (85-99%) with fundamentally correct approach. Only MINOR errors that are easily fixable:
- Small calculation errors (arithmetic mistakes, sign errors)
- Minor typos or notation issues
- Missing one trivial case that doesn't affect the main argument
- Small gaps that are obvious to fill

KEY TEST for "almost": If the minor errors were corrected, would this be a complete correct solution? If YES and the errors are truly minor → "almost".

IMPORTANT: "almost" requires the solution to be fundamentally sound. The core logic must be correct. Do NOT use "almost" for solutions with major gaps, wrong approaches, or significant missing steps.

**"partial"** - Meaningful progress (10-75%) with SOME valid mathematical content but SIGNIFICANT gaps:
- Proved a useful lemma or sub-result
- Identified a key insight or correct approach but didn't complete it
- Started with valid reasoning but stopped prematurely
- Has correct framework but missing major execution steps
- Some valid mathematical work but substantial portions missing or incorrect

KEY TEST for "partial": Is there valid mathematical content that contributes meaningfully to the solution, despite significant gaps? If YES → "partial".

IMPORTANT: "partial" is for incomplete solutions with genuine progress. The student demonstrated some understanding but didn't get close to a complete solution.

**"incorrect"** - No valid mathematical progress. Wrong approach, fundamental misunderstanding, nonsense, or empty response. KEY TEST: Is there any valid mathematical contribution toward solving the problem? If NO → "incorrect".

## CRITICAL DECISION RULES - APPLY IN ORDER:
1. **Is it fully correct?** (Complete proof, correct answer, sound reasoning, 100% complete) → "correct"
2. **Is it nearly perfect with only tiny fixable errors?** (85-99% complete, minor calculation errors, small typos, trivial case missing - would be perfect if fixed) → "almost"
3. **Is there meaningful valid progress despite significant gaps?** (10-75% complete, proved a lemma, identified key insight, correct approach started but incomplete) → "partial"
4. **Otherwise** (wrong approach, no valid progress, nonsense) → "incorrect"

## CRITICAL DISTINCTIONS - READ CAREFULLY:

**"almost" vs "partial" - THIS IS THE HARDEST DISTINCTION:**
- "almost": The solution is 85-99% complete. The core logic is correct. Only minor, easily fixable errors remain. The student clearly understands the problem and would solve it with minimal fixes.
- "partial": The solution is 10-75% complete. There's valid mathematical work (a lemma, insight, or partial proof) but major gaps remain. The student made progress but is far from a complete solution.

**DECISION HEURISTIC:**
- If fixing small errors would give a complete solution → "almost"
- If the solution has good ideas but needs substantial additional work → "partial"
- If the solution is more than 80% complete with minor issues → "almost"
- If the solution is less than 75% complete with significant gaps → "partial"

**"partial" vs "incorrect":**
- "partial": At least some valid mathematical contribution (proved something useful, identified a key insight, correct approach started)
- "incorrect": No valid progress at all (wrong approach, nonsense, empty)

## Output Format:
Respond ONLY in this JSON format:
<json>
{{
    "response": "correct"
}}</json>

The response value MUST be exactly one of: "correct", "almost", "partial", or "incorrect" (lowercase).

## FINAL INSTRUCTION:
Be precise in your classification. When deciding between "almost" and "partial", ask yourself: "Is this solution nearly complete with only minor issues, or does it have significant gaps despite some valid work?" If truly uncertain, prefer "partial" over "almost" to be conservative."""

        # Initialize msg_history as empty list in case of early failure
        msg_history = []
        
        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"Error calling LLM: {e}")
            return "incorrect", [{"role": "error", "text": f"LLM call failed: {e}"}]

        # Extract prediction from JSON using flexible extraction
        prediction = "incorrect"  # Default to incorrect for safety
        
        # Handle edge case: msg_history is None
        if msg_history is None:
            self.log_fn("msg_history is None")
            return str(prediction), [{"role": "error", "text": "No response from LLM"}]
        
        # Handle edge case: msg_history is not a list
        if not isinstance(msg_history, list):
            self.log_fn(f"Invalid msg_history type: {type(msg_history)}")
            return str(prediction), [{"role": "error", "text": f"Invalid response type: {type(msg_history)}"}]
        
        # Handle edge case: empty msg_history
        if len(msg_history) == 0:
            self.log_fn("msg_history is empty")
            return str(prediction), msg_history
        
        try:
            last_message = msg_history[-1]
            
            # Handle edge case: last message is not a dict
            if not isinstance(last_message, dict):
                self.log_fn(f"Last message is not a dict: {type(last_message)}")
                return str(prediction), msg_history
            
            # Handle edge case: missing 'text' key
            if "text" not in last_message:
                self.log_fn(f"Last message missing 'text' key. Keys: {list(last_message.keys())}")
                # Try to find any string value that might be the response
                for key, value in last_message.items():
                    if isinstance(value, str) and len(value) > 10:
                        prediction = _normalize_prediction(value.lower())
                        self.log_fn(f"Extracted from alternative key '{key}': {prediction}")
                        return str(prediction), msg_history
                return str(prediction), msg_history
            
            text_content = last_message["text"]
            
            # Handle edge case: text is not a string
            if not isinstance(text_content, str):
                self.log_fn(f"Text content is not a string: {type(text_content)}")
                # Try to convert to string if possible
                try:
                    text_content = str(text_content)
                except Exception:
                    return str(prediction), msg_history
            
            # Handle edge case: empty text
            if not text_content or not text_content.strip():
                self.log_fn("Text content is empty")
                return str(prediction), msg_history
            
            # Try to extract JSON
            extracted = _extract_json_flexible(text_content)
            
            if extracted and len(extracted) > 0:
                last_json = extracted[-1]
                
                if isinstance(last_json, dict):
                    # Try multiple possible keys for the response
                    response_keys = ["response", "answer", "prediction", "grade", "result", "category", "evaluation", "classification"]
                    raw_prediction = None
                    matched_key = None
                    
                    for key in response_keys:
                        if key in last_json:
                            raw_prediction = last_json[key]
                            matched_key = key
                            break
                    
                    # If no standard key found, try any key that might contain the prediction
                    if raw_prediction is None:
                        for key, value in last_json.items():
                            if isinstance(value, str):
                                val_lower = value.lower().strip()
                                if val_lower in ["correct", "almost", "partial", "incorrect"]:
                                    raw_prediction = value
                                    matched_key = key
                                    break
                    
                    if raw_prediction is not None:
                        # Handle different types of raw_prediction
                        if isinstance(raw_prediction, str):
                            prediction = _normalize_prediction(raw_prediction)
                        elif isinstance(raw_prediction, (int, float)):
                            # Handle numeric predictions (unlikely but possible)
                            prediction = "incorrect"
                        elif isinstance(raw_prediction, bool):
                            prediction = "correct" if raw_prediction else "incorrect"
                        else:
                            prediction = _normalize_prediction(str(raw_prediction))
                        
                        self.log_fn(f"Raw prediction from key '{matched_key}': {raw_prediction} -> Normalized: {prediction}")
                    else:
                        self.log_fn(f"JSON missing response key. Available keys: {list(last_json.keys())}")
                        # Try to extract from the entire JSON as string
                        try:
                            json_str = json.dumps(last_json).lower()
                            prediction = _normalize_prediction(json_str)
                            self.log_fn(f"Extracted from JSON string: {prediction}")
                        except Exception:
                            pass
                else:
                    self.log_fn(f"Last JSON is not a dict: {type(last_json)}")
                    # If it's a string, try to normalize it directly
                    if isinstance(last_json, str):
                        prediction = _normalize_prediction(last_json)
                        self.log_fn(f"Extracted from JSON string value: {prediction}")
            else:
                # Try to extract directly from text if no JSON found
                text_lower = text_content.lower()
                prediction = _normalize_prediction(text_lower)
                self.log_fn(f"No JSON found, extracted from text: {prediction}")
                
        except json.JSONDecodeError as e:
            self.log_fn(f"JSON decode error: {e}")
            # Try to extract from raw text as fallback
            try:
                if msg_history and isinstance(msg_history, list) and len(msg_history) > 0:
                    last_msg = msg_history[-1]
                    if isinstance(last_msg, dict):
                        text_content = last_msg.get("text", "")
                        if isinstance(text_content, str):
                            prediction = _normalize_prediction(text_content.lower())
                            self.log_fn(f"Fallback extraction after JSON error: {prediction}")
            except Exception as fallback_e:
                self.log_fn(f"Fallback extraction failed: {fallback_e}")
                
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            prediction = "incorrect"

        # Final validation: ensure prediction is one of the valid categories
        valid_categories = ["correct", "almost", "partial", "incorrect"]
        if prediction not in valid_categories:
            self.log_fn(f"Invalid prediction '{prediction}', defaulting to 'incorrect'")
            prediction = "incorrect"

        return str(prediction), msg_history
