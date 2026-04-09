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


def _validate_and_normalize_prediction(prediction: str, grading_guidelines: str) -> str:
    """Validate and normalize the prediction based on grading guidelines.
    
    Simplified version that focuses on the four main grading categories:
    Correct, Partial, Almost, Incorrect.
    """
    if not prediction or prediction.strip() == "":
        return "None"
    
    prediction = prediction.strip()
    
    # Remove surrounding quotes if present
    if (prediction.startswith('"') and prediction.endswith('"')) or \
       (prediction.startswith("'") and prediction.endswith("'")):
        prediction = prediction[1:-1].strip()
    
    # Remove common prefixes that LLMs sometimes add
    prefixes_to_remove = [
        "the answer is", "answer:", "score:", "grade:", 
        "final answer:", "prediction:", "result:", "output:",
        "the grade is", "the score is", "i would give",
        "therefore,", "thus,", "so,", "conclusion:",
        "evaluation:", "assessment:", "verdict:", "decision:"
    ]
    pred_lower = prediction.lower()
    for prefix in prefixes_to_remove:
        if pred_lower.startswith(prefix):
            prediction = prediction[len(prefix):].strip()
            pred_lower = prediction.lower()
            break
    
    # Remove trailing punctuation
    prediction = prediction.rstrip('.;:,!?')
    pred_lower = prediction.lower()
    
    # Define the four standard grading labels
    standard_labels = ["Correct", "Partial", "Almost", "Incorrect"]
    
    # Check for exact match first (case-insensitive)
    for label in standard_labels:
        if pred_lower == label.lower():
            return label
    
    # Check for match with parentheses
    for label in standard_labels:
        if pred_lower == f"({label.lower()})":
            return label
    
    # Check for partial matches - be more specific to avoid false matches
    # Order matters: check for more specific terms first
    # Check if any of the negative indicator words appear in the prediction
    negative_indicators = ["wrong", "false", "error", "invalid", "none", "completely wrong", "totally wrong"]
    if "incorrect" in pred_lower or any(indicator in pred_lower for indicator in negative_indicators):
        return "Incorrect"
    if "almost" in pred_lower or "nearly" in pred_lower or "close" in pred_lower:
        return "Almost"
    if "partial" in pred_lower or "partially" in pred_lower or "incomplete" in pred_lower or "some progress" in pred_lower:
        return "Partial"
    # Check if any of the positive indicator words appear in the prediction
    positive_indicators = ["right", "true", "valid", "full", "complete", "perfect"]
    if "correct" in pred_lower or any(indicator in pred_lower for indicator in positive_indicators):
        return "Correct"
    
    # If no match found, return the original prediction (may be numeric)
    return prediction


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

        instruction = f"""You are an expert grader for the International Mathematical Olympiad (IMO). Your task is to evaluate the student's answer and assign the appropriate grade.

## Domain
{domain}

## Problem Statement
{problem}

## Official Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Your Task

Carefully analyze the student's answer:
1. Read the problem and understand what is being asked
2. Review the official solution to understand the correct approach
3. Compare the student's work to the official solution
4. Identify what the student got right, what they got wrong, and what is missing
5. Apply the grading guidelines to determine the appropriate grade

## Grading Categories

The grade must be one of these four categories:

- **Correct**: The solution is complete and correct. The student has provided a full, valid solution that correctly solves the problem. All key steps are present and correct.

- **Partial**: The solution has significant progress but is incomplete or has minor errors. The student has made meaningful progress toward the solution, perhaps solving part of the problem or providing a mostly correct approach with some gaps or small mistakes.

- **Almost**: The solution is nearly complete but has a significant error or gap. The student has a good understanding and approach but made a critical mistake that prevents the solution from being fully correct. The solution is close but fundamentally flawed in an important way.

- **Incorrect**: The solution is wrong or makes no meaningful progress. The student either provided no valid work, completely misunderstood the problem, or made fundamental errors that show no real understanding of the correct approach.

Respond in this exact JSON format wrapped in <json> tags:
<json>
{{
    "reasoning": "Your detailed analysis here...",
    "response": "GRADE_HERE"
}}
</json>

CRITICAL INSTRUCTIONS:
- The "response" field must contain ONLY one of: "Correct", "Partial", "Almost", or "Incorrect"
- Do NOT add any extra text, explanations, or formatting in the "response" field
- Do NOT use quotes around the grade in the response field
- Use ONLY the exact grade values: Correct, Partial, Almost, or Incorrect
- Be decisive in your grading - choose the single best category that matches the student's work"""

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
                    # Third try: direct text extraction for simple responses
                    extraction_method = "direct"
                    # Look for the last line that might be the answer
                    lines = [line.strip() for line in last_text.split('\n') if line.strip()]
                    if lines:
                        # Check if the last non-empty line looks like a simple answer
                        last_line = lines[-1]
                        if len(last_line) < 50 and not last_line.startswith('{') and not last_line.startswith('<'):
                            prediction = last_line
                            self.log_fn(f"Used direct text extraction: {prediction}")
            
            # Validate and normalize the prediction
            original_prediction = prediction
            prediction = _validate_and_normalize_prediction(prediction, grading_guidelines)
            
            if original_prediction != prediction:
                self.log_fn(f"Normalized prediction from '{original_prediction}' to '{prediction}'")
            
            self.log_fn(f"Extraction method: {extraction_method}, final prediction: {prediction}")
            
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

        return str(prediction), msg_history
