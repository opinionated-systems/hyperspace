"""
Task agent: solves a given task with chain-of-thought reasoning.

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
    Also handles nested JSON objects, markdown code blocks, and common LLM formatting errors.
    
    Args:
        text: The text containing <json>...</json> blocks.
        
    Returns:
        A list of parsed JSON dicts, or None if no valid JSON found.
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
        
        # Try to parse the inner content as JSON
        try:
            results.append(json.loads(inner))
            continue
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code blocks within the content
        code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', inner, re.DOTALL)
        if code_block_match:
            try:
                results.append(json.loads(code_block_match.group(1)))
                continue
            except json.JSONDecodeError:
                pass
        
        # Try to find raw JSON object with proper brace matching
        try:
            json_str = _extract_json_with_brace_matching(inner)
            if json_str:
                results.append(json.loads(json_str))
                continue
        except (json.JSONDecodeError, ValueError):
            pass
                
        # Try to clean common LLM formatting errors and re-parse
        try:
            cleaned = _clean_json_string(inner)
            if cleaned:
                results.append(json.loads(cleaned))
                continue
        except (json.JSONDecodeError, ValueError):
            pass
                
    return results or None


def _extract_json_with_brace_matching(text: str) -> str | None:
    """Extract JSON object using proper brace matching.
    
    This handles nested braces correctly by counting open/close braces.
    
    Args:
        text: Text that may contain a JSON object.
        
    Returns:
        The extracted JSON string, or None if no valid object found.
    """
    # Find the first opening brace
    start = text.find('{')
    if start == -1:
        return None
    
    brace_count = 0
    in_string = False
    escape_next = False
    end = start
    
    for i in range(start, len(text)):
        char = text[i]
        
        if escape_next:
            escape_next = False
            continue
            
        if char == '\\':
            escape_next = True
            continue
            
        if char == '"' and not in_string:
            in_string = True
        elif char == '"' and in_string:
            in_string = False
        elif not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end = i + 1
                    break
    
    if brace_count == 0 and end > start:
        return text[start:end]
    return None


def _clean_json_string(text: str) -> str | None:
    """Clean common LLM formatting errors from JSON strings.
    
    Args:
        text: Potentially malformed JSON string.
        
    Returns:
        Cleaned JSON string, or None if cleaning failed.
    """
    # Remove leading/trailing whitespace and newlines
    cleaned = text.strip()
    
    # Remove trailing commas before closing braces/brackets
    cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
    
    # Fix single quotes to double quotes (common LLM error)
    # Only replace quotes that are not inside strings
    result = []
    in_string = False
    escape_next = False
    for char in cleaned:
        if escape_next:
            result.append(char)
            escape_next = False
            continue
        if char == '\\':
            result.append(char)
            escape_next = True
            continue
        if char == '"' and not in_string:
            in_string = True
            result.append(char)
        elif char == '"' and in_string:
            in_string = False
            result.append(char)
        elif char == "'" and not in_string:
            result.append('"')
        else:
            result.append(char)
    cleaned = ''.join(result)
    
    # Remove comments (// style)
    cleaned = re.sub(r'//[^\n]*', '', cleaned)
    
    # Remove comments (/* */ style)
    cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
    
    # Validate by trying to parse
    try:
        json.loads(cleaned)
        return cleaned
    except json.JSONDecodeError:
        return None


def _extract_json_fallback(text: str) -> dict | None:
    """Fallback JSON extraction for non-tagged JSON or markdown code blocks.

    Tries to find JSON in markdown code blocks or raw JSON objects.
    Uses robust brace-matching and cleaning to handle nested structures and LLM errors.
    
    Args:
        text: The text to search for JSON objects.
        
    Returns:
        A parsed JSON dict if found, None otherwise.
        
    Priority:
        1. Markdown code blocks with JSON
        2. Raw JSON objects with expected keys (response, grade, score, etc.)
        3. Any valid JSON object
        4. Cleaned JSON with common LLM errors fixed
    """
    # First try markdown code blocks
    patterns = [
        r'```json\s*(.*?)\s*```',  # Markdown JSON blocks
        r'```\s*(\{.*?\})\s*```',  # Generic code blocks with JSON
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                # Try cleaning the match
                cleaned = _clean_json_string(match)
                if cleaned:
                    return json.loads(cleaned)
                continue
    
    # Try to find JSON objects by matching braces with proper nesting
    json_candidates = []
    
    # Find all potential JSON object starts
    for match in re.finditer(r'\{\s*"', text):
        start = match.start()
        json_str = _extract_json_with_brace_matching(text[start:])
        if json_str:
            try:
                parsed = json.loads(json_str)
                json_candidates.append(parsed)
            except json.JSONDecodeError:
                # Try cleaning
                cleaned = _clean_json_string(json_str)
                if cleaned:
                    try:
                        json_candidates.append(json.loads(cleaned))
                    except json.JSONDecodeError:
                        pass
    
    # Also try to find and clean any JSON-like structures
    # Look for patterns that might be JSON with errors
    potential_json = re.search(r'\{[\s\S]{10,500}\}', text)
    if potential_json:
        cleaned = _clean_json_string(potential_json.group(0))
        if cleaned:
            try:
                parsed = json.loads(cleaned)
                json_candidates.append(parsed)
            except json.JSONDecodeError:
                pass
    
    # Prioritize candidates with expected keys
    priority_keys = ['response', 'grade', 'score', 'answer', 'result', 'value']
    for key in priority_keys:
        for candidate in json_candidates:
            if key in candidate:
                return candidate
    
    # Return first valid candidate if any
    if json_candidates:
        return json_candidates[0]
    
    return None


def _extract_grade_from_text(text: str, grading_guidelines: str) -> str:
    """Extract grade label from text using simple, fast strategies.
    
    Args:
        text: The text to search for grade labels.
        grading_guidelines: The grading guidelines (unused but kept for API compatibility).
        
    Returns:
        The extracted grade label, or "None" if not found.
    """
    if not text or not text.strip():
        return "None"
    
    text = text.strip()
    text_lower = text.lower()
    
    # Valid labels in priority order
    valid_labels = ["Correct", "Incorrect", "Partial", "Almost"]
    
    # Strategy 1: Exact match
    for label in valid_labels:
        if text_lower == label.lower():
            return label
    
    # Strategy 2: Look for "response": "X" or similar patterns
    patterns = [
        rf'"response"\s*:\s*"?\s*({"|".join(valid_labels)})\s*"?',
        rf'"grade"\s*:\s*"?\s*({"|".join(valid_labels)})\s*"?',
        rf'"answer"\s*:\s*"?\s*({"|".join(valid_labels)})\s*"?',
        rf'"score"\s*:\s*"?\s*({"|".join(valid_labels)})\s*"?',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            matched = match.group(1)
            for label in valid_labels:
                if matched.lower() == label.lower():
                    return label
    
    # Strategy 3: Look for standalone labels on their own lines (last 5 lines)
    lines = text.split('\n')
    for line in lines[-5:]:
        line_stripped = line.strip().lower()
        if not line_stripped or line_stripped.startswith('```') or line_stripped.startswith('<'):
            continue
        for label in valid_labels:
            if label.lower() == line_stripped or label.lower() == line_stripped.rstrip('.;:,!?()[]{}'):
                return label
    
    # Strategy 4: Look for any valid label as a whole word
    for label in valid_labels:
        if re.search(rf'\b{label}\b', text, re.IGNORECASE):
            return label
    
    return "None"


def _validate_and_normalize_prediction(prediction: str, grading_guidelines: str) -> str:
    """Validate and normalize the prediction.
    
    Args:
        prediction: The raw prediction string.
        grading_guidelines: The grading guidelines (unused but kept for API compatibility).
        
    Returns:
        The normalized prediction, or "None" if invalid.
    """
    if not prediction or prediction.strip() == "":
        return "None"
    
    prediction = prediction.strip()
    
    # Remove surrounding quotes
    if (prediction.startswith('"') and prediction.endswith('"')) or \
       (prediction.startswith("'") and prediction.endswith("'")):
        prediction = prediction[1:-1].strip()
    
    # Remove common prefixes
    prefixes = ["the answer is", "answer:", "score:", "grade:", 
                "final answer:", "prediction:", "result:", "output:",
                "the grade is", "the score is", "i would give",
                "therefore,", "thus,", "so,", "conclusion:",
                "evaluation:", "assessment:", "verdict:", "decision:"]
    pred_lower = prediction.lower()
    for prefix in prefixes:
        if pred_lower.startswith(prefix):
            prediction = prediction[len(prefix):].strip()
            pred_lower = prediction.lower()
            break
    
    # Remove trailing punctuation
    prediction = prediction.rstrip('.;:,!?')
    pred_lower = prediction.lower()
    
    # Reject if too long or contains LaTeX
    if len(prediction) > 50:
        extracted = _extract_grade_from_text(prediction, grading_guidelines)
        return extracted if extracted != "None" else "None"
    
    latex_indicators = ['\\[', '\\]', '\\begin', '\\end', '\\frac', '\\sum', 
                       '\\int', '\\prod', '\\sqrt', '\\left', '\\right', '$', '\\mathbf']
    for indicator in latex_indicators:
        if indicator in prediction:
            extracted = _extract_grade_from_text(prediction, grading_guidelines)
            return extracted if extracted != "None" else "None"
    
    # Check for exact matches
    valid_labels = ["Correct", "Incorrect", "Partial", "Almost"]
    for label in valid_labels:
        if pred_lower == label.lower():
            return label
    
    # Check for partial matches
    for label in valid_labels:
        if label.lower() in pred_lower:
            return label
    
    # If just a number, return it
    if re.match(r'^-?\d+(\.\d+)?$', prediction):
        return prediction
    
    return "None"


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

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
        # Extract fields for structured prompting
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert IMO grader. Evaluate the student's solution and assign one grade: Correct, Incorrect, Partial, or Almost.

## Problem
{problem}

## Official Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Grade Definitions
- **Correct**: Complete, correct solution. All key steps present and logically sound.
- **Almost**: Nearly complete with only minor errors/omissions. Main ideas correct.
- **Partial**: Meaningful progress with correct steps, but significant gaps/errors.
- **Incorrect**: Fundamentally wrong, critical errors, or no meaningful progress.

## Your Task
1. Compare the student's answer to the official solution
2. Identify what they did correctly and where they erred
3. Assign the appropriate grade

## Response Format
Respond ONLY in this JSON format:

<json>
{{
    "reasoning": "Your analysis of what the student did right and wrong, and why this grade applies.",
    "response": "GRADE"
}}
</json>

The "response" field must contain ONLY one of: Correct, Incorrect, Partial, or Almost (exact spelling, no quotes)."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        last_text = msg_history[-1]["text"] if msg_history else ""
        extraction_method = "none"
        
        try:
            # First try: extract from <json> tags
            extracted = _extract_jsons(last_text)
            if extracted:
                extraction_method = "json_tags"
                # Try to get response field, fall back to other common fields
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = str(last_json["response"])
                elif "grade" in last_json:
                    prediction = str(last_json["grade"])
                elif "score" in last_json:
                    prediction = str(last_json["score"])
                elif "answer" in last_json:
                    prediction = str(last_json["answer"])
                elif "result" in last_json:
                    prediction = str(last_json["result"])
                elif "value" in last_json:
                    prediction = str(last_json["value"])
                else:
                    # If no recognized field, use the whole JSON as string
                    prediction = json.dumps(last_json)
            else:
                # Second try: fallback extraction for non-tagged JSON
                fallback = _extract_json_fallback(last_text)
                if fallback:
                    extraction_method = "fallback"
                    if "response" in fallback:
                        prediction = str(fallback["response"])
                    elif "grade" in fallback:
                        prediction = str(fallback["grade"])
                    elif "score" in fallback:
                        prediction = str(fallback["score"])
                    elif "answer" in fallback:
                        prediction = str(fallback["answer"])
                    elif "result" in fallback:
                        prediction = str(fallback["result"])
                    elif "value" in fallback:
                        prediction = str(fallback["value"])
                    else:
                        prediction = json.dumps(fallback)
                    self.log_fn(f"Used fallback JSON extraction: {prediction}")
                else:
                    # Third try: look for grade labels directly in the text
                    extraction_method = "direct"
                    prediction = _extract_grade_from_text(last_text, grading_guidelines)
                    if prediction != "None":
                        self.log_fn(f"Used direct grade extraction: {prediction}")
            
            # Validate and normalize the prediction
            original_prediction = prediction
            prediction = _validate_and_normalize_prediction(prediction, grading_guidelines)
            
            if original_prediction != prediction:
                self.log_fn(f"Normalized prediction from '{original_prediction}' to '{prediction}'")
            
            self.log_fn(f"Extraction method: {extraction_method}, initial prediction: {prediction}")
            
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Last resort: try to find any numeric or keyword answer in the text
            try:
                # Look for IMO scores (0-7)
                score_match = re.search(r'\b([0-7])\b', last_text)
                if score_match:
                    prediction = score_match.group(1)
                    self.log_fn(f"Used regex extraction for score: {prediction}")
            except Exception:
                pass

        # Self-verification step: if we have a valid prediction, verify it
        if prediction not in ["None", ""]:
            verified_prediction = self._verify_prediction(
                prediction, last_text, problem, solution, grading_guidelines, student_answer
            )
            if verified_prediction != prediction:
                self.log_fn(f"Prediction changed after verification: {prediction} -> {verified_prediction}")
                prediction = verified_prediction

        return str(prediction), msg_history

    def _verify_prediction(
        self, 
        initial_prediction: str, 
        reasoning_text: str,
        problem: str,
        solution: str,
        grading_guidelines: str,
        student_answer: str
    ) -> str:
        """Quick verification of the prediction by checking consistency."""
        # Skip verification if no valid reasoning
        if not reasoning_text or len(reasoning_text) < 30:
            return initial_prediction
        
        # Simple consistency check: look for contradictions in reasoning
        reasoning_lower = reasoning_text.lower()
        pred_lower = initial_prediction.lower()
        
        # Check for obvious mismatches
        contradiction_patterns = {
            "correct": ["wrong", "incorrect", "error", "mistake", "flawed"],
            "incorrect": ["correct", "right", "valid", "properly", "complete"],
        }
        
        # If prediction is "Correct" but reasoning mentions many errors
        if pred_lower == "correct":
            error_count = sum(1 for word in ["error", "wrong", "incorrect", "mistake"] 
                            if word in reasoning_lower)
            if error_count >= 2:
                # Likely should be Partial or Almost
                if "minor" in reasoning_lower or "small" in reasoning_lower:
                    return "Almost"
                return "Partial"
        
        # If prediction is "Incorrect" but reasoning praises the solution
        if pred_lower == "incorrect":
            positive_count = sum(1 for word in ["correct", "right", "valid", "good", "proper"] 
                               if word in reasoning_lower)
            if positive_count >= 3:
                return "Partial"
        
        return initial_prediction
