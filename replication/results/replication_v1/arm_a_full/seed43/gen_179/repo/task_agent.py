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
    
    Ensures the prediction matches expected format from grading guidelines.
    Handles various edge cases and normalizes common variations.
    Prioritizes extracting the exact label format from grading guidelines.
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
    
    # Remove trailing punctuation (but preserve parentheses for labels)
    prediction = prediction.rstrip('.;:,!?')
    pred_lower = prediction.lower()
    
    # IMO GRADE LABELS - Primary focus for this task
    # These are the four standard IMO grade labels
    imo_labels = ["Correct", "Partial", "Almost", "Incorrect"]
    
    # Check for exact match with IMO labels (case-insensitive)
    for label in imo_labels:
        if pred_lower == label.lower():
            return label
    
    # Check for IMO labels with parentheses
    for label in imo_labels:
        if pred_lower == f"({label.lower()})":
            return label
    
    # Check for partial match with IMO labels
    for label in imo_labels:
        if label.lower() in pred_lower:
            return label
    
    # IMO POINTS TO LABEL MAPPING
    # Map numeric point values to grade labels
    point_to_label = {
        "7": "Correct",
        "6": "Partial",
        "1": "Almost", 
        "0": "Incorrect"
    }
    
    # Check if prediction is a numeric point value
    if prediction in point_to_label:
        return point_to_label[prediction]
    
    # Check for spelled-out point values
    number_words = {
        "zero": "0", "one": "1", "two": "2", "three": "3", 
        "four": "4", "five": "5", "six": "6", "seven": "7",
        "0 points": "0", "1 point": "1", "2 points": "2",
        "3 points": "3", "4 points": "4", "5 points": "5",
        "6 points": "6", "7 points": "7"
    }
    for word, digit in number_words.items():
        if re.search(rf'\b{word}\b', pred_lower):
            if digit in point_to_label:
                return point_to_label[digit]
            return digit
    
    # Extract expected score patterns from grading guidelines
    if re.search(r'\b[0-7]\b', grading_guidelines):
        # IMO-style 0-7 scoring - look for single digit 0-7
        match = re.search(r'\b([0-7])\b', prediction)
        if match:
            digit = match.group(1)
            if digit in point_to_label:
                return point_to_label[digit]
            return digit
    
    # Check for synonyms that map to IMO labels
    synonym_map = {
        "Correct": ["full", "complete", "right", "true", "valid", "perfect", "solved", "success"],
        "Incorrect": ["wrong", "false", "error", "invalid", "unsolved", "failed", "fail", "failure"],
        "Partial": ["partly", "incomplete", "some", "partially"],
        "Almost": ["nearly", "close", "mostly"]
    }
    
    for label, synonyms in synonym_map.items():
        for synonym in synonyms:
            if synonym in pred_lower:
                return label
    
    # Check for "Correct"/"Incorrect" format with better precision
    if "correct" in grading_guidelines.lower() or "incorrect" in grading_guidelines.lower():
        # Check for exact matches first
        if pred_lower == "correct":
            return "Correct"
        if pred_lower == "incorrect":
            return "Incorrect"
        # Then check for partial matches
        if "incorrect" in pred_lower:
            return "Incorrect"
        if "correct" in pred_lower:
            return "Correct"
    
    # Check for "Partial" format
    if "partial" in grading_guidelines.lower():
        if pred_lower == "partial":
            return "Partial"
        if "partial" in pred_lower:
            return "Partial"
    
    # Check for "Almost" format
    if "almost" in grading_guidelines.lower():
        if pred_lower == "almost":
            return "Almost"
        if "almost" in pred_lower:
            return "Almost"
    
    # Check for Yes/No format
    if re.search(r'\b(yes|no)\b', grading_guidelines, re.IGNORECASE):
        if pred_lower == "yes":
            return "Yes"
        if pred_lower == "no":
            return "No"
        if re.search(r'\byes\b', pred_lower):
            return "Yes"
        elif re.search(r'\bno\b', pred_lower):
            return "No"
    
    # Check for Pass/Fail format
    if re.search(r'\b(pass|fail)\b', grading_guidelines, re.IGNORECASE):
        if pred_lower == "pass":
            return "Pass"
        if pred_lower == "fail":
            return "Fail"
        if re.search(r'\bpass\b', pred_lower):
            return "Pass"
        elif re.search(r'\bfail\b', pred_lower):
            return "Fail"
    
    # Check for numeric ranges in guidelines (e.g., "0-100", "1-10")
    range_match = re.search(r'(\d+)\s*-\s*(\d+)', grading_guidelines)
    if range_match:
        min_val, max_val = int(range_match.group(1)), int(range_match.group(2))
        # Look for a number in the prediction within the range
        num_match = re.search(r'\b(\d+)\b', prediction)
        if num_match:
            val = int(num_match.group(1))
            if min_val <= val <= max_val:
                return str(val)
    
    # If prediction is just a number, return it as-is
    if re.match(r'^-?\d+(\.\d+)?$', prediction):
        return prediction
    
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

        # Extract point values from grading guidelines to provide explicit mapping
        point_to_label = {
            "7": "Correct",
            "6": "Partial", 
            "1": "Almost",
            "0": "Incorrect"
        }
        
        # Build point mapping explanation
        point_mapping_text = ""
        for points, label in point_to_label.items():
            if points in grading_guidelines:
                point_mapping_text += f"- {points} points = {label}\n"
        
        instruction = f"""You are an expert grader for the International Mathematical Olympiad (IMO). Your task is to evaluate the student's answer and assign the appropriate grade label.

## Domain
{domain}

## Problem Statement
{problem}

## Official Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## IMO Point-to-Label Mapping
The IMO uses a 0-7 point scoring system. Map points to grade labels as follows:
- 7 points → Correct (complete, correct solution)
- 6 points → Partial (substantial progress with minor gaps)
- 1 point → Almost (significant progress but incomplete)
- 0 points → Incorrect (no meaningful progress or wrong approach)

{point_mapping_text}
## Student's Answer
{student_answer}

## Your Task

Analyze the student's answer systematically:

1. **Understand the Problem**: Identify what the problem is asking and the key requirements.

2. **Review the Official Solution**: Note the correct approach, key insights, and expected steps.

3. **Analyze the Student's Work**:
   - What correct steps did the student take?
   - What errors or misconceptions are present?
   - How much progress did they make toward the full solution?
   - Did they reach the final answer or get stuck partway?

4. **Apply IMO Scoring Logic**:
   - **Correct (7 points)**: Student provided a complete, correct solution with proper reasoning.
   - **Partial (6 points)**: Student made substantial progress, solved most of the problem, but has minor gaps or errors.
   - **Almost (1 point)**: Student made significant progress on key parts but the solution is incomplete or has major gaps.
   - **Incorrect (0 points)**: Student made no meaningful progress, used wrong approach, or solution is fundamentally flawed.

5. **Determine the Grade**: Based on your analysis, select the grade label that best matches the student's work.

Respond in this exact JSON format wrapped in <json> tags:
<json>
{{
    "reasoning": "Your detailed analysis here...",
    "response": "GRADE_HERE"
}}
</json>

IMPORTANT:
- The "response" field must contain ONLY one of these four labels: "Correct", "Partial", "Almost", or "Incorrect".
- Do NOT add any extra text, explanations, or formatting in the "response" field.
- Choose the label that best matches the student's level of progress based on IMO scoring standards."""

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
