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

# Compile regex patterns for grade extraction
_GRADE_PATTERNS = {
    "correct": re.compile(r'\bcorrect\b', re.IGNORECASE),
    "almost": re.compile(r'\balmost\b', re.IGNORECASE),
    "partial": re.compile(r'\bpartial\b', re.IGNORECASE),
    "incorrect": re.compile(r'\bincorrect\b', re.IGNORECASE),
}

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
            # Try to clean up common issues using the helper function
            cleaned = _clean_json_string(inner)
            if cleaned:
                try:
                    results.append(json.loads(cleaned))
                except json.JSONDecodeError:
                    continue
    # If no results from <json> tags, try broader patterns
    if not results:
        # Try to find JSON-like content between any xml-style tags
        tag_pattern = re.compile(r'<(\w+)>\s*(.*?)\s*</\1>', re.DOTALL)
        for match in tag_pattern.finditer(text):
            inner = match.group(2).strip()
            if inner.startswith('{'):
                try:
                    results.append(json.loads(inner))
                except json.JSONDecodeError:
                    cleaned = _clean_json_string(inner)
                    if cleaned:
                        try:
                            results.append(json.loads(cleaned))
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
            # Try to clean up common issues using the helper function
            cleaned = _clean_json_string(match.strip())
            if cleaned:
                try:
                    results.append(json.loads(cleaned))
                except json.JSONDecodeError:
                    continue
    # Fallback: try to find JSON in any code block or inline code
    if not results:
        # Try single backtick code blocks
        inline_pattern = r'`([^`]+)`'
        for match in re.finditer(inline_pattern, text):
            content = match.group(1).strip()
            if content.startswith('{'):
                try:
                    results.append(json.loads(content))
                except json.JSONDecodeError:
                    cleaned = _clean_json_string(content)
                    if cleaned:
                        try:
                            results.append(json.loads(cleaned))
                        except json.JSONDecodeError:
                            continue
    return results or None


def _extract_json_braces(text: str) -> list[dict] | None:
    """Extract JSON objects by finding matching braces with improved handling."""
    results = []
    i = 0
    while i < len(text):
        if text[i] == '{':
            # Find matching closing brace with improved tracking
            count = 1
            j = i + 1
            in_string = False
            escape_next = False
            
            while j < len(text) and count > 0:
                char = text[j]
                
                if escape_next:
                    escape_next = False
                elif char == '\\' and in_string:
                    escape_next = True
                elif char == '"' and not in_string:
                    in_string = True
                elif char == '"' and in_string:
                    in_string = False
                elif not in_string:
                    if char == '{':
                        count += 1
                    elif char == '}':
                        count -= 1
                j += 1
                
            if count == 0:
                json_str = text[i:j]
                try:
                    results.append(json.loads(json_str))
                except json.JSONDecodeError:
                    # Try multiple cleanup strategies
                    cleaned = _clean_json_string(json_str)
                    if cleaned:
                        try:
                            results.append(json.loads(cleaned))
                        except json.JSONDecodeError:
                            pass
            i = j
        else:
            i += 1
    return results or None


def _clean_json_string(json_str: str) -> str | None:
    """Clean up common JSON formatting issues.
    
    Args:
        json_str: Raw JSON string that failed to parse
        
    Returns:
        Cleaned JSON string or None if cleaning failed
    """
    cleaned = json_str.strip()
    
    # Strip leading/trailing non-JSON text (e.g., "Here is the JSON:" or "```json")
    first_brace = cleaned.find('{')
    last_brace = cleaned.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        cleaned = cleaned[first_brace:last_brace + 1]
    
    # Remove trailing commas before closing braces/brackets
    cleaned = re.sub(r',\s*}', '}', cleaned)
    cleaned = re.sub(r',\s*]', ']', cleaned)
    
    # Fix single quotes to double quotes (but not within strings)
    # This is a simplified approach - handle simple cases
    cleaned = cleaned.replace("'", '"')
    
    # Fix common Python-style boolean/null values
    cleaned = re.sub(r'\bTrue\b', 'true', cleaned)
    cleaned = re.sub(r'\bFalse\b', 'false', cleaned)
    cleaned = re.sub(r'\bNone\b', 'null', cleaned)
    
    # Remove comments (both // and /* */ styles)
    cleaned = re.sub(r'//[^\n]*', '', cleaned)
    cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
    
    # Remove control characters
    cleaned = ''.join(char for char in cleaned if ord(char) >= 32 or char in '\n\r\t')
    
    return cleaned if cleaned else None


def _extract_grade_directly(text: str) -> str | None:
    """Extract grade directly from text by looking for grade keywords.
    
    This is a last-resort fallback when JSON extraction fails.
    Uses a priority-based approach to handle ambiguous cases.
    """
    text_lower = text.lower()
    
    # Priority 1: Check for explicit grade declarations
    # Look for patterns like "Grade: X", "My grade is X", "I would grade this as X"
    grade_patterns = [
        r'(?:grade|grading|classification|verdict|evaluation|result|answer)\s*(?:is|:|=|was|should be|would be)\s*(correct|almost|partial|incorrect)',
        r'(?:i\s*(?:would|will|think|believe|consider|classify|rate|score|grade))\s*(?:this|it|as|the answer)?\s*(?:as|is|:|=)?\s*(correct|almost|partial|incorrect)',
        r'(?:the\s*(?:answer|response|solution|work|attempt)\s*is\s*)(correct|almost|partial|incorrect)',
    ]
    for pattern in grade_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).capitalize()
    
    # Priority 2: Check for negated forms (these are strong signals)
    if re.search(r'\bnot\s+(?:incorrect|wrong|false|invalid)\b', text, re.IGNORECASE):
        return "Correct"
    if re.search(r'\bnot\s+(?:correct|right|accurate|valid)\b', text, re.IGNORECASE):
        return "Incorrect"
    
    # Priority 3: Check for "almost" - indicates minor mistakes
    if re.search(r'\balmost\b', text, re.IGNORECASE):
        return "Almost"
    
    # Priority 4: Check for "partial" or "partially"
    if re.search(r'\bpartial(ly)?\b', text, re.IGNORECASE):
        return "Partial"
    
    # Priority 5: Check for "incorrect", "wrong", "false" (but not "not incorrect")
    if re.search(r'\b(incorrect|wrong|false|invalid)\b', text, re.IGNORECASE):
        # Make sure it's not negated
        if not re.search(r'\bnot\s+(incorrect|wrong|false|invalid)\b', text, re.IGNORECASE):
            return "Incorrect"
    
    # Priority 6: Check for "correct" (but not "incorrect")
    if re.search(r'\bcorrect\b', text, re.IGNORECASE):
        if not re.search(r'\bincorrect\b', text, re.IGNORECASE):
            return "Correct"
    
    return None


def _normalize_prediction(prediction: str) -> str:
    """Normalize prediction to one of the expected values.
    
    Handles a wide variety of LLM output formats including:
    - Direct grade names (Correct, Almost, Partial, Incorrect)
    - Synonym phrases (mostly correct, partially right, etc.)
    - Negated forms (not correct, not wrong, etc.)
    - JSON field values with extra text
    """
    prediction_str = str(prediction).strip().lower()
    
    # Strip common prefixes like "grade:", "prediction:", etc.
    prediction_str = re.sub(r'^(grade|prediction|result|answer|verdict|evaluation)\s*[:=]\s*', '', prediction_str).strip()
    
    # Check for exact matches first
    if prediction_str in ["correct", "incorrect", "partial", "almost"]:
        return prediction_str.capitalize()
    
    # Comprehensive synonym mappings - check multi-word phrases first
    # "Almost" synonyms (minor issues, mostly right)
    almost_phrases = [
        "almost correct", "nearly correct", "mostly correct", "largely correct",
        "essentially correct", "basically correct", "substantially correct",
        "reasonably correct", "fairly correct", "quite correct",
        "almost right", "nearly right", "mostly right",
        "minor error", "small error", "tiny mistake", "minor mistake",
        "minor issue", "small issue", "slight error", "slight mistake",
        "not entirely correct", "not completely correct", "not fully correct",
        "not quite correct", "not entirely right", "not completely right",
        "almost entirely correct", "nearly entirely correct",
        "almost perfect", "nearly perfect", "nearly flawless",
        "very close", "very nearly", "almost there",
        "mostly right with minor", "largely right with minor",
    ]
    if any(phrase in prediction_str for phrase in almost_phrases):
        return "Almost"
    
    # "Partial" synonyms (some understanding, incomplete)
    partial_phrases = [
        "partial", "partially", "partly", "somewhat",
        "partially correct", "partly correct", "somewhat correct",
        "partially right", "partly right", "somewhat right",
        "partially incorrect", "partly incorrect", "somewhat incorrect",
        "partially wrong", "partly wrong", "somewhat wrong",
        "some correct", "partially right", "partly correct",
        "some understanding", "on the right track",
        "incomplete", "missing", "lacking", "insufficient",
        "not complete", "not finished", "not fully developed",
        "partially complete", "partially done", "partially solved",
        "some progress", "partial progress", "some merit",
        "partially valid", "partially sound", "partially justified",
        "not entirely wrong", "not completely wrong", "not fully wrong",
        "not quite wrong", "not entirely incorrect", "not completely incorrect",
        "not fully incorrect", "not quite incorrect",
        "partially understood", "partially grasped",
        "some correct steps", "some correct reasoning",
        "correct approach but", "right idea but",
        "good start but", "promising but",
    ]
    if any(phrase in prediction_str for phrase in partial_phrases):
        return "Partial"
    
    # "Correct" synonyms (fully right) - checked BEFORE incorrect to handle negations
    # like "not incorrect", "not wrong" which contain the word "incorrect"/"wrong"
    # but should map to "Correct"
    correct_phrases = [
        "correct", "right", "true", "valid", "accurate", "complete", "full",
        "completely correct", "fully correct", "entirely correct",
        "totally correct", "absolutely correct", "perfectly correct",
        "completely right", "fully right", "entirely right",
        "totally right", "absolutely right", "perfectly right",
        "fully valid", "completely valid", "entirely valid",
        "fully accurate", "completely accurate", "entirely accurate",
        "fully justified", "completely justified", "fully proven",
        "completely proven", "fully sound", "completely sound",
        "no errors", "no issues", "no problems", "no mistakes",
        "flawless", "perfect", "impeccable",
        "fully satisfies", "completely satisfies",
        "meets all criteria", "satisfies all requirements",
        "not incorrect", "not wrong", "not inaccurate",
        "not invalid", "not unsound",
    ]
    if any(phrase in prediction_str for phrase in correct_phrases):
        return "Correct"

    # "Incorrect" synonyms (fundamentally wrong)
    incorrect_phrases = [
        "incorrect", "wrong", "false", "error", "mistake", "flawed", "invalid",
        "completely wrong", "totally incorrect", "fundamentally wrong",
        "entirely incorrect", "completely incorrect", "totally wrong",
        "entirely wrong", "wholly incorrect", "wholly wrong",
        "not correct", "not right", "not accurate", "not valid",
        "not sound", "not justified", "not proven",
        "no merit", "no validity", "no basis",
        "fundamentally flawed", "fundamentally incorrect",
        "completely off", "totally off", "entirely off",
        "misses the point", "misses the mark",
        "does not address", "fails to address",
        "unrelated", "irrelevant", "inapplicable",
        "not applicable", "not relevant",
        "completely misses", "totally misses",
        "no understanding", "no grasp", "no comprehension",
        "entirely incorrect", "completely incorrect",
        "largely incorrect", "mostly incorrect",
        "largely wrong", "mostly wrong",
        "essentially wrong", "basically wrong",
        "substantially wrong", "substantially incorrect",
    ]
    if any(phrase in prediction_str for phrase in incorrect_phrases):
        return "Incorrect"
    
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
        # Extract fields from inputs for better prompt construction
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        instruction = f"""You are an expert mathematical grader for International Mathematical Olympiad (IMO) problems.

Your task is to evaluate a student's answer to a mathematical problem and assign a grade based on the provided grading guidelines.

## Problem Statement:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Grading Rubric:
Carefully analyze the student's answer against the official solution and grading guidelines. Assign ONE of these grades:

- "Correct" - The answer is fully correct, complete, and matches the official solution. All key steps and reasoning are present and valid.

- "Almost" - The answer is essentially correct with only minor mistakes (e.g., small calculation errors, typos, or notation issues). The core reasoning and approach are sound.

- "Partial" - The answer has some correct elements and shows understanding of key concepts, but is incomplete, missing critical steps, or has significant errors that don't invalidate the entire approach.

- "Incorrect" - The answer is fundamentally wrong, uses incorrect methods, or shows a fundamental misunderstanding of the problem.

## Examples:

Example 1 - Correct answer:
<json>
{{"response": "Correct"}}
</json>

Example 2 - Answer with minor errors:
<json>
{{"response": "Almost"}}
</json>

Example 3 - Partial progress:
<json>
{{"response": "Partial"}}
</json>

Example 4 - Fundamentally wrong:
<json>
{{"response": "Incorrect"}}
</json>

## Your Task:
1. First, identify what the problem is asking and what the official solution provides.
2. Check if the student's answer matches the official solution's approach and conclusion.
3. Look for any grading guidelines that specifically apply to this answer.
4. Determine the appropriate grade based on the rubric above.

## CRITICAL OUTPUT INSTRUCTIONS:
You MUST output ONLY a JSON object wrapped in <json> tags. Do not include any other text, explanations, or markdown formatting outside the JSON tags.

Your response MUST follow this EXACT format:
<json>
{{"response": "Correct"}}
</json>

OR

<json>
{{"response": "Almost"}}
</json>

OR

<json>
{{"response": "Partial"}}
</json>

OR

<json>
{{"response": "Incorrect"}}
</json>

The value for "response" must be exactly one of: "Correct", "Almost", "Partial", or "Incorrect" (case-sensitive, with first letter capitalized). Choose the grade that best matches the quality of the student's answer."""

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
                
                # Try multiple extraction methods
                extracted = None
                
                # Method 1: <json> tags (most reliable)
                extracted = _extract_jsons(text)
                
                # Method 2: Markdown code blocks
                if not extracted:
                    extracted = _extract_json_from_markdown(text)
                
                # Method 3: Raw JSON with braces
                if not extracted:
                    extracted = _extract_json_braces(text)
                
                if extracted:
                    last_extract = extracted[-1]
                    
                    # Try multiple field names in order of preference
                    pred_value = None
                    field_used = None
                    for field in ["response", "grade", "evaluation", "result", "answer", "verdict", "prediction"]:
                        if field in last_extract:
                            pred_value = last_extract[field]
                            field_used = field
                            break
                    
                    if pred_value is not None:
                        prediction = _normalize_prediction(str(pred_value))
                        self.log_fn(f"Extracted prediction: {prediction} from field: {field_used}")
                    else:
                        # If no recognized field, try to use the whole response if it's a simple string
                        if isinstance(last_extract, str):
                            prediction = _normalize_prediction(last_extract)
                            self.log_fn(f"Extracted prediction from string JSON: {prediction}")
                        else:
                            # Log available keys for debugging
                            self.log_fn(f"No recognized field in JSON. Available keys: {list(last_extract.keys())}")
                            # Try to find any value that looks like a valid grade
                            for key, value in last_extract.items():
                                if isinstance(value, str):
                                    normalized = _normalize_prediction(value)
                                    if normalized != "None":
                                        prediction = normalized
                                        self.log_fn(f"Found valid grade in field '{key}': {prediction}")
                                        break
                else:
                    # No JSON found, try to extract from raw text by looking for grade keywords
                    prediction = _normalize_prediction(text)
                    self.log_fn(f"No JSON found, normalized from text: {prediction}")
                    
                    # If still None, try the direct grade extraction
                    if prediction == "None":
                        direct_grade = _extract_grade_directly(text)
                        if direct_grade:
                            prediction = direct_grade
                            self.log_fn(f"Found grade via direct extraction: {prediction}")
                    
                    # If still None, try to find exact grade words in the text
                    if prediction == "None":
                        for grade_name, pattern in _GRADE_PATTERNS.items():
                            if pattern.search(text):
                                prediction = grade_name.capitalize()
                                self.log_fn(f"Found grade keyword in text: {prediction}")
                                break
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            import traceback
            self.log_fn(traceback.format_exc())

        return str(prediction), msg_history
