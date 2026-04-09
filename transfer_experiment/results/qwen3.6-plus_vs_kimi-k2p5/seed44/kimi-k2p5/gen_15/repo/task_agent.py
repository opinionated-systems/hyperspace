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

__version__ = "2.5.0"

# Valid grade labels for normalization
_VALID_GRADES = {"correct", "almost", "partial", "incorrect"}


def _normalize_prediction(raw: str) -> str:
    """Normalize a raw prediction string to one of the valid grade labels.
    
    Uses word boundary matching to avoid false substring matches
    (e.g., "incorrect" should NOT match "correct").
    
    Args:
        raw: Raw prediction string
        
    Returns:
        Capitalized grade label or "None" if not recognized
    """
    if not raw or not isinstance(raw, str):
        return "None"
    
    cleaned = raw.strip().strip('"').strip("'").lower()
    
    # Strip common prefixes like "grade:", "prediction:", etc.
    cleaned = re.sub(r'^(grade|prediction|result|answer|verdict|evaluation|response|label)\s*[:=]\s*', '', cleaned).strip()
    
    # Remove trailing punctuation
    cleaned = re.sub(r'[.,;:!?]+$', '', cleaned).strip()
    
    # Direct match first (exact match after cleaning)
    if cleaned in _VALID_GRADES:
        return cleaned.capitalize()
    
    # Handle common variations
    if cleaned in ["almost correct", "nearly correct", "mostly correct"]:
        return "Almost"
    if cleaned in ["partly correct", "some understanding", "partial credit"]:
        return "Partial"
    if cleaned in ["wrong", "false", "not correct", "not correct"]:
        return "Incorrect"
    if cleaned in ["right", "true", "valid", "full marks"]:
        return "Correct"
    
    # Use word-boundary regex to avoid false substring matches
    # Check in order: almost, partial, incorrect, correct (to avoid "incorrect" matching "correct")
    for grade in ["almost", "partial", "incorrect", "correct"]:
        if re.search(r'(?<![a-z])' + re.escape(grade) + r'(?![a-z])', cleaned):
            return grade.capitalize()
    
    return "None"


def _extract_grade_from_reasoning(text: str) -> str | None:
    """Extract grade from reasoning text when JSON extraction fails.
    
    Looks for explicit grade statements in the reasoning/analysis section.
    Uses word boundaries to avoid false matches.
    """
    text_lower = text.lower()
    
    # Look for explicit grade statements in reasoning with word boundaries
    # More specific patterns first to avoid false matches
    patterns = [
        r'grade\s*[:=]\s*["\']?(correct|almost|partial|incorrect)["\']?',
        r'classification\s*[:=]\s*["\']?(correct|almost|partial|incorrect)["\']?',
        r'evaluation\s*[:=]\s*["\']?(correct|almost|partial|incorrect)["\']?',
        r'verdict\s*[:=]\s*["\']?(correct|almost|partial|incorrect)["\']?',
        r'\bthis answer is\b["\']?(correct|almost|partial|incorrect)["\']?',
        r'\bthe answer is\b["\']?(correct|almost|partial|incorrect)["\']?',
        r'\banswer is\b["\']?(correct|almost|partial|incorrect)["\']?',
        r'should be graded as["\']?(correct|almost|partial|incorrect)["\']?',
        r'qualifies as["\']?(correct|almost|partial|incorrect)["\']?',
        r'falls under["\']?(correct|almost|partial|incorrect)["\']?',
        r'belongs to["\']?(correct|almost|partial|incorrect)["\']?',
        r'\bfinal grade\b["\']?(correct|almost|partial|incorrect)["\']?',
        r'\bresult\b["\']?(correct|almost|partial|incorrect)["\']?',
        r'\bi would grade\b["\']?(correct|almost|partial|incorrect)["\']?',
        r'\bthe grade is\b["\']?(correct|almost|partial|incorrect)["\']?',
        r'\bassigned grade\b["\']?(correct|almost|partial|incorrect)["\']?',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            grade = match.group(1).lower()
            if grade in _VALID_GRADES:
                return grade.capitalize()
    
    # Last resort: look for standalone grade words with word boundaries
    # But be careful about "incorrect" containing "correct"
    # Check in order: almost, partial, incorrect, correct (to avoid "incorrect" matching "correct")
    for grade in ["almost", "partial", "incorrect", "correct"]:
        # Use negative lookbehind/lookahead to ensure standalone words
        pattern = r'(?<![a-z])' + re.escape(grade) + r'(?![a-z])'
        if re.search(pattern, text_lower):
            return grade.capitalize()
    
    return None


def _repair_json(json_str: str) -> str:
    """Repair common JSON formatting issues in LLM outputs.
    
    Args:
        json_str: Raw JSON string that may have formatting issues
        
    Returns:
        Repaired JSON string
    """
    cleaned = json_str.strip()
    
    # Remove trailing commas before } or ]
    cleaned = re.sub(r',\s*}', '}', cleaned)
    cleaned = re.sub(r',\s*]', ']', cleaned)
    
    # Fix single quotes to double quotes (but be careful with apostrophes in text)
    # Only replace quotes that appear to be JSON delimiters
    cleaned = re.sub(r"(?<=[{\[,\s])'([^']+)'(?=\s*[:}\],])", r'"\1"', cleaned)
    
    # Fix common JSON issues: unquoted keys
    cleaned = re.sub(r'(\{|,\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', cleaned)
    
    # Fix double curly braces from f-string escaping
    cleaned = cleaned.replace('{{', '{').replace('}}', '}')
    
    # Remove control characters
    cleaned = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', cleaned)
    
    # Fix escaped newlines that might break parsing
    cleaned = cleaned.replace('\\n', ' ').replace('\\t', ' ')
    
    return cleaned


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
        
        # Skip empty blocks
        if not inner:
            continue
            
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # Try to clean up common issues
            try:
                cleaned = _repair_json(inner)
                results.append(json.loads(cleaned))
            except json.JSONDecodeError:
                continue
    return results or None


def _extract_json_from_markdown(text: str) -> list[dict] | None:
    """Extract JSON objects from markdown code blocks as a fallback."""
    results = []
    pattern = r'```(?:json)?\s*(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        match = match.strip()
        if not match:
            continue
        try:
            results.append(json.loads(match))
        except json.JSONDecodeError:
            try:
                cleaned = _repair_json(match)
                results.append(json.loads(cleaned))
            except json.JSONDecodeError:
                continue
    return results or None


def _extract_json_braces(text: str) -> list[dict] | None:
    """Extract JSON objects by finding matching braces as a last resort."""
    results = []
    i = 0
    while i < len(text):
        if text[i] == '{':
            count = 1
            j = i + 1
            while j < len(text) and count > 0:
                if text[j] == '{':
                    count += 1
                elif text[j] == '}':
                    count -= 1
                j += 1
            if count == 0:
                json_str = text[i:j]
                try:
                    results.append(json.loads(json_str))
                except json.JSONDecodeError:
                    try:
                        cleaned = _repair_json(json_str)
                        results.append(json.loads(cleaned))
                    except json.JSONDecodeError:
                        pass
            i = j
        else:
            i += 1
    return results or None


def _validate_grade_consistency(prediction: str, reasoning: str) -> tuple[str, bool]:
    """Validate that the grade is consistent with the reasoning text.
    
    Args:
        prediction: The predicted grade
        reasoning: The reasoning text from the LLM
        
    Returns:
        Tuple of (validated_grade, is_consistent)
    """
    if prediction == "None" or not reasoning:
        return prediction, True
    
    reasoning_lower = reasoning.lower()
    pred_lower = prediction.lower()
    
    # Check for contradictory language in reasoning
    contradictions = {
        "correct": ["wrong", "incorrect", "not right", "not correct", "error"],
        "almost": ["completely wrong", "totally incorrect", "no understanding"],
        "partial": ["complete solution", "fully correct", "perfect"],
        "incorrect": ["correct", "right answer", "perfect solution"]
    }
    
    # Check if reasoning contradicts the prediction
    contradiction_words = contradictions.get(pred_lower, [])
    for word in contradiction_words:
        if word in reasoning_lower:
            # Look for negation that might negate the contradiction
            negation_pattern = r'\b(not|no|never|hardly|barely)\s+\w*\s*' + re.escape(word)
            if not re.search(negation_pattern, reasoning_lower):
                logger.warning(f"Potential inconsistency: grade '{prediction}' but reasoning contains '{word}'")
                return prediction, False
    
    return prediction, True


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
        # Build a structured prompt with clearly labeled fields
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematical grader for International Mathematical Olympiad (IMO) problems.

Your task is to evaluate a student's answer and assign exactly one of four grades: "Correct", "Almost", "Partial", or "Incorrect".

## Problem Statement:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Grading Rubric - READ CAREFULLY:

1. "Correct" - The answer is fully correct, complete, and matches the official solution. All key steps and reasoning are present and valid. The solution would receive full marks.

2. "Almost" - The answer demonstrates COMPLETE understanding with only MINOR errors:
   - Small arithmetic/calculation errors in an otherwise correct derivation
   - Typos in notation or variable names
   - Missing trivial final simplification steps
   - The student clearly knows how to solve the problem but made careless mistakes
   - Key insight: If you fixed these minor errors, the solution would be "Correct"
   - The core approach and reasoning are sound throughout

3. "Partial" - The answer shows SOME understanding but has SIGNIFICANT issues:
   - Missing critical proof steps or logical gaps
   - Incomplete solution (stopped halfway or missed key cases)
   - Significant errors in reasoning that affect the conclusion
   - Correct approach but major calculation errors that invalidate the result
   - Key insight: The student is on the right track but needs substantial work

4. "Incorrect" - The answer is fundamentally wrong:
   - Wrong approach or method entirely
   - Fundamental misunderstanding of the problem
   - No meaningful progress toward the solution

## Critical Distinctions - PAY CLOSE ATTENTION:

**"Almost" vs "Partial" - THIS IS THE MOST COMMON ERROR:**
- "Almost" = The solution is essentially complete with only tiny, fixable flaws
  * The student demonstrates full mastery of the solution method
  * Errors are superficial (typos, arithmetic slips, minor notation issues)
  * If you fixed the errors, you'd have a "Correct" answer
  * Example: Correct proof with one small calculation error

- "Partial" = Significant portions are missing or wrong
  * The student shows some understanding but has major gaps
  * Missing critical steps, incomplete reasoning, or major errors
  * Would need substantial work to become correct
  * Example: Correct approach but stopped halfway, or major logical flaw

**"Partial" vs "Incorrect":**
- "Partial" = Genuine understanding of some key concepts (on the right track)
- "Incorrect" = Little to no understanding (wrong approach or no progress)

## Decision Process:
1. First, check if the answer matches the official solution perfectly → "Correct"
2. If not perfect, ask: "Is the approach completely right with only minor errors?" → "Almost"
3. If not "Almost", ask: "Does the student show genuine understanding of key concepts?" → "Partial"
4. If none of the above → "Incorrect"

## How to Identify a CORRECT Answer:
A "Correct" answer MUST have:
- A complete logical structure from start to finish
- All key steps present and valid
- Correct final conclusion
- Valid reasoning throughout

IMPORTANT: A complete proof with proper reasoning that arrives at the correct conclusion should be graded as "Correct", even if the wording differs from the official solution. The key is whether the mathematical logic is sound and complete, not whether it matches the official solution word-for-word.

Signs of a CORRECT answer:
- States the correct final result
- Provides a complete proof with multiple logical steps
- Uses proper mathematical reasoning
- Has a clear beginning, middle, and end to the argument
- All critical steps in the reasoning chain are present

## IMPORTANT REMINDERS:
- "Almost" is for NEARLY CORRECT answers with minor flaws - the student clearly knows the solution
- "Partial" is for answers with SIGNIFICANT gaps or errors - the student is on the right track but missing key pieces
- When in doubt between "Almost" and "Partial", choose "Partial" if there are missing critical steps
- Be conservative: if you're unsure, it's better to grade lower than higher
- Pay special attention to the grading guidelines provided above - they contain specific criteria for this problem

## Output Format - CRITICAL - FOLLOW EXACTLY:
You MUST output ONLY a JSON object wrapped in <json> tags. Do not include any other text, explanations, or markdown formatting outside the JSON tags.

Your response MUST follow this EXACT format (replace X with your grade):
<json>
{{"grade": "X"}}
</json>

Valid values for "grade" are exactly: "Correct", "Almost", "Partial", or "Incorrect" (case-sensitive, with first letter capitalized).

Example 1 - For a correct answer:
<json>
{{"grade": "Correct"}}
</json>

Example 2 - For an almost correct answer:
<json>
{{"grade": "Almost"}}
</json>

Example 3 - For a partial answer:
<json>
{{"grade": "Partial"}}
</json>

Example 4 - For an incorrect answer:
<json>
{{"grade": "Incorrect"}}
</json>

DO NOT include any text before or after the <json> tags. Output ONLY the JSON block."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON using multiple methods
        prediction = "None"
        try:
            if msg_history and len(msg_history) > 0:
                text = msg_history[-1].get("text", "")
                
                if not text:
                    self.log_fn("Empty response from LLM")
                    return "None", msg_history

                # Try multiple extraction methods in order of reliability
                extracted = _extract_jsons(text)
                extraction_method = "<json> tags" if extracted else None
                
                if not extracted:
                    extracted = _extract_json_from_markdown(text)
                    extraction_method = "markdown code blocks" if extracted else None
                    
                if not extracted:
                    extracted = _extract_json_braces(text)
                    extraction_method = "brace matching" if extracted else None

                if extracted:
                    last_extract = extracted[-1]
                    self.log_fn(f"Extracted JSON using {extraction_method}: {last_extract}")

                    # Try multiple field names in order of preference
                    # Prioritize "grade" since that's what the prompt asks for
                    pred_value = None
                    for field in ["grade", "response", "evaluation", "result", "answer", "verdict", "prediction"]:
                        if field in last_extract:
                            pred_value = last_extract[field]
                            self.log_fn(f"Found grade in field '{field}': {pred_value}")
                            break

                    if pred_value is not None:
                        prediction = _normalize_prediction(str(pred_value))
                        self.log_fn(f"Normalized prediction: {prediction}")
                    else:
                        # Try to find any value that looks like a valid grade
                        for key, value in last_extract.items():
                            if isinstance(value, str):
                                normalized = _normalize_prediction(value)
                                if normalized != "None":
                                    prediction = normalized
                                    self.log_fn(f"Found valid grade in field '{key}': {prediction}")
                                    break
                        if prediction == "None":
                            self.log_fn(f"No valid grade found in JSON fields: {list(last_extract.keys())}")
                else:
                    # No JSON found, try to extract grade from reasoning text first
                    self.log_fn("No JSON found, trying to extract from reasoning text")
                    grade_from_reasoning = _extract_grade_from_reasoning(text)
                    if grade_from_reasoning:
                        prediction = grade_from_reasoning
                        self.log_fn(f"Extracted grade from reasoning: {prediction}")
                    else:
                        # Fall back to normalizing raw text
                        prediction = _normalize_prediction(text)
                        self.log_fn(f"No JSON found, normalized from text: {prediction}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            import traceback
            self.log_fn(f"Traceback: {traceback.format_exc()}")

        return str(prediction), msg_history
