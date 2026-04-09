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
    """
    text_lower = text.lower()
    
    # Look for exact grade words with word boundaries
    # Order matters: check for more specific patterns first
    
    # Check for "almost" - indicates minor mistakes
    if re.search(r'\balmost\b', text, re.IGNORECASE):
        return "Almost"
    
    # Check for "partial" or "partially"
    if re.search(r'\bpartial(ly)?\b', text, re.IGNORECASE):
        return "Partial"
    
    # Check for "incorrect", "wrong", "false" (but not "not incorrect")
    if re.search(r'\b(incorrect|wrong|false|invalid)\b', text, re.IGNORECASE):
        # Make sure it's not negated
        if not re.search(r'\b(not\s+(incorrect|wrong|false)|correct)\b', text, re.IGNORECASE):
            return "Incorrect"
    
    # Check for "correct" (but not "incorrect")
    if re.search(r'\bcorrect\b', text, re.IGNORECASE):
        if not re.search(r'\bincorrect\b', text, re.IGNORECASE):
            return "Correct"
    
    return None


def _normalize_prediction(prediction: str) -> str:
    """Normalize prediction to one of the expected values."""
    prediction_str = str(prediction).strip().lower()
    
    # Remove common prefixes and clean up
    prediction_str = re.sub(r'^(grade|prediction|result|answer|verdict|evaluation|response)\s*[:=]\s*["\']?', '', prediction_str).strip()
    prediction_str = prediction_str.strip('"\'').strip()
    
    # Check for exact matches first (including "almost")
    if prediction_str in ["correct", "incorrect", "partial", "almost"]:
        return prediction_str.capitalize()
    
    # Check for "almost" first - indicates minor mistakes only
    # Use word boundary check to avoid matching inside other words
    if re.search(r'\balmost\b', prediction_str):
        return "Almost"
    
    # Check for "nearly", "minor", "small error" -> Almost
    if any(phrase in prediction_str for phrase in ["nearly", "minor mistake", "minor error", "small error", "tiny mistake", "mostly correct", "nearly correct"]):
        return "Almost"
    
    # Check for partial first (to avoid matching "partially correct" as "correct")
    if re.search(r'\bpartial\b', prediction_str):
        return "Partial"
    
    # Check for "partially" or "partly" -> Partial
    if any(phrase in prediction_str for phrase in ["partially", "partly", "some correct", "partially right", "partly correct", "some understanding", "on the right track"]):
        return "Partial"
    
    # Check for incorrect/wrong/false with word boundaries
    if re.search(r'\b(incorrect|wrong|false|invalid)\b', prediction_str):
        return "Incorrect"
    
    # Check for error/mistake/flawed
    if any(word in prediction_str for word in ["error", "mistake", "flawed"]):
        return "Incorrect"
    
    # Check for "completely wrong" or "totally incorrect" -> Incorrect
    if any(phrase in prediction_str for phrase in ["completely wrong", "totally incorrect", "fundamentally wrong", "entirely incorrect"]):
        return "Incorrect"
    
    # Check for correct (but not incorrect) - use word boundary
    if re.search(r'\bcorrect\b', prediction_str) and not re.search(r'\bincorrect\b', prediction_str):
        return "Correct"
    
    # Check for other positive indicators
    if any(word in prediction_str for word in ["right", "true", "valid", "accurate", "complete", "full"]):
        return "Correct"
    
    # Check for negative indicators suggesting fundamental issues
    if any(word in prediction_str for word in ["incomplete", "missing", "lacking", "insufficient"]):
        return "Partial"
    
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
Carefully analyze the student's answer against the official solution and grading guidelines. Assign EXACTLY ONE of these four grades:

1. "Correct" - The answer is fully correct, complete, and matches the official solution. All key steps and reasoning are present and valid.

2. "Almost" - The answer is essentially correct with only minor mistakes (e.g., small calculation errors, typos, or notation issues). The core reasoning and approach are sound. Use this when the student clearly understands the solution but made small errors.

3. "Partial" - The answer has some correct elements and shows understanding of key concepts, but is incomplete, missing critical steps, or has significant errors that don't invalidate the entire approach.

4. "Incorrect" - The answer is fundamentally wrong, uses incorrect methods, or shows a fundamental misunderstanding of the problem.

## Decision Guidelines:
- If the answer is perfect → "Correct"
- If the answer has the right approach but small errors (typos, minor calculation mistakes) → "Almost"
- If the answer shows some good ideas but is incomplete or has significant gaps → "Partial"
- If the answer is completely wrong or uses wrong methods → "Incorrect"

## Examples:

Example 1 - Perfect answer:
<json>
{{"response": "Correct"}}
</json>

Example 2 - Right approach, minor calculation error:
<json>
{{"response": "Almost"}}
</json>

Example 3 - Some progress but incomplete:
<json>
{{"response": "Partial"}}
</json>

Example 4 - Wrong approach:
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
                    # Prioritize "response" since that's what the prompt asks for
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
                        # Check for "almost" first (highest priority for this category)
                        if re.search(r'\balmost\b', text, re.IGNORECASE):
                            prediction = "Almost"
                            self.log_fn(f"Found 'almost' keyword in text: {prediction}")
                        else:
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
