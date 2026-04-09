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
    
    # Remove surrounding quotes if present (handle nested quotes)
    while (len(prediction) >= 2 and 
           ((prediction[0] == '"' and prediction[-1] == '"') or
            (prediction[0] == "'" and prediction[-1] == "'"))):
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
    
    # FIRST: Extract exact label formats from grading guidelines
    # Look for categorical labels like (Correct), (Partial), (Almost), (Incorrect)
    guideline_labels = []
    
    # Pattern for parenthesized labels: (Correct), (Partial), etc.
    parenthesized_pattern = r'\(([A-Za-z]+)\)'
    for match in re.finditer(parenthesized_pattern, grading_guidelines):
        label = match.group(1)
        if label not in guideline_labels:
            guideline_labels.append(label)
    
    # Pattern for labels without parentheses: Correct, Partial, etc.
    # Match standalone capitalized words that appear to be labels
    standalone_pattern = r'(?:^|\n|\s)\s*([A-Z][a-z]+)\s*(?:\n|$|\d+\.|\s)'
    for match in re.finditer(standalone_pattern, grading_guidelines):
        label = match.group(1)
        if label not in guideline_labels and label not in ["The", "A", "An", "This", "That"]:
            guideline_labels.append(label)
    
    # If we found categorical labels in guidelines, use them for validation
    if guideline_labels:
        # Check for exact match (case-insensitive) - highest priority
        for label in guideline_labels:
            if pred_lower == label.lower():
                return label
        
        # Check for exact match with parentheses
        for label in guideline_labels:
            if pred_lower == f"({label.lower()})":
                return label
        
        # Check for partial match (prediction contains label or vice versa)
        for label in guideline_labels:
            label_lower = label.lower()
            if label_lower in pred_lower or pred_lower in label_lower:
                return label
    
    # SECOND: Handle specific known grading formats with improved logic
    
    # Check for "Correct"/"Incorrect" format with better precision
    if "correct" in grading_guidelines.lower() or "incorrect" in grading_guidelines.lower():
        # Check for exact matches first - highest priority
        if pred_lower == "correct":
            return "Correct"
        if pred_lower == "incorrect":
            return "Incorrect"
        # Check for exact match with "not correct" - should be Incorrect
        if pred_lower == "not correct":
            return "Incorrect"
        
        # Check for phrases that clearly indicate "Partial" before checking Incorrect/Correct
        # This helps distinguish Partial from Incorrect
        partial_indicators = ["some progress", "partially", "incomplete but", "partial credit",
                             "meaningful progress", "valid intermediate", "some correct steps",
                             "on the right track", "good start", "correct approach"]
        for indicator in partial_indicators:
            if indicator in pred_lower:
                # Don't return Correct/Incorrect yet - let Partial check handle it
                break
        else:
            # No partial indicators found, check for Incorrect/Correct synonyms
            # Check for synonyms that mean "Incorrect" - check BEFORE "correct" synonyms
            # This prevents over-classification as "Correct"
            incorrect_synonyms = ["wrong", "false", "error", "invalid", "unsolved", "failed", "fail", 
                                  "not correct", "not right", "not valid", "not true", "bad", "rejected",
                                  "does not prove", "doesn't prove", "failed to", "unable to",
                                  "not solved", "fundamentally wrong", "no real progress", "no progress"]
            for synonym in incorrect_synonyms:
                if synonym in pred_lower:
                    return "Incorrect"
            # Check for synonyms that mean "Correct"
            correct_synonyms = ["fully correct", "completely correct", "perfect", "solved",
                               "fully rigorous", "complete proof", "valid proof", "correct proof"]
            for synonym in correct_synonyms:
                if synonym in pred_lower:
                    return "Correct"
        
        # Then check for partial matches - check "incorrect" BEFORE "correct"
        # to avoid misclassifying "not correct" or similar as "Correct"
        # But be careful not to misclassify "partially correct" as "Incorrect"
        if "incorrect" in pred_lower and "partially" not in pred_lower:
            return "Incorrect"
        if pred_lower == "correct":  # Exact match only after other checks
            return "Correct"
    
    # Check for "Partial" format
    if "partial" in grading_guidelines.lower():
        if pred_lower == "partial":
            return "Partial"
        # Check for partial synonyms - be more specific to avoid misclassification
        partial_synonyms = ["partially correct", "partial credit", "partial solution", 
                          "partial proof", "partial result", "incomplete proof",
                          "some progress made", "meaningful progress", "valid intermediate results",
                          "significant progress", "substantial progress"]
        for synonym in partial_synonyms:
            if synonym in pred_lower:
                return "Partial"
        # Check for "partial" but be careful not to override "Almost" or "Incorrect"
        # Only return Partial if the prediction seems to be indicating partial credit
        if pred_lower == "partial" or pred_lower.startswith("partial "):
            return "Partial"
    
    # Check for "Almost" format
    if "almost" in grading_guidelines.lower():
        if pred_lower == "almost":
            return "Almost"
        # Check for almost synonyms - be specific to distinguish from Partial
        almost_synonyms = ["nearly complete", "almost correct", "almost complete", 
                          "mostly correct", "minor error", "small gap", "tiny error",
                          "nearly there", "very close", "small mistake", "minor gap",
                          "essentially correct", "minor technical gap"]
        for synonym in almost_synonyms:
            if synonym in pred_lower:
                return "Almost"
        # Only return Almost for exact match or clear indicators
        if pred_lower == "almost" or pred_lower.startswith("almost "):
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
    
    # THIRD: Handle numeric scoring formats
    
    # Extract expected score patterns from grading guidelines
    # Common IMO patterns: "7" (full score), "0" (no score), "1-6" (partial)
    if re.search(r'\b[0-7]\b', grading_guidelines):
        # IMO-style 0-7 scoring - look for single digit 0-7
        match = re.search(r'\b([0-7])\b', prediction)
        if match:
            return match.group(1)
        # Also check for spelled-out numbers
        number_words = {"zero": "0", "one": "1", "two": "2", "three": "3", 
                       "four": "4", "five": "5", "six": "6", "seven": "7",
                       "0 points": "0", "1 point": "1", "2 points": "2",
                       "3 points": "3", "4 points": "4", "5 points": "5",
                       "6 points": "6", "7 points": "7"}
        for word, digit in number_words.items():
            if re.search(rf'\b{word}\b', pred_lower):
                return digit
    
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

        instruction = f"""You are an expert grader for the International Mathematical Olympiad (IMO). Your task is to critically evaluate the student's answer and assign the appropriate grade based on the grading guidelines.

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

Analyze the student's answer step by step:
1. Understand what the problem requires and what constitutes a complete, correct solution
2. Compare the student's work to the official solution - identify ALL errors, gaps, and incorrect reasoning
3. Check if the student has proven ALL required claims or if key steps are missing
4. Look for logical errors, calculation mistakes, incomplete proofs, and unjustified assertions
5. Apply the grading guidelines strictly to determine the appropriate grade

IMPORTANT GRADING PRINCIPLES:
- "Correct" means the solution is complete, rigorous, and fully proves all required claims with no gaps or errors
- "Incorrect" means the solution has fundamental errors, completely wrong approach, or fails to prove the main claim
- "Partial" means the solution has some correct progress and valid intermediate results, but significant gaps remain before a complete proof
- "Almost" means the solution is nearly complete with valid overall approach, but has minor gaps or small errors that don't invalidate the main result
- Be CRITICAL and THOROUGH: carefully check every step against the official solution
- "Partial" is for solutions with meaningful progress (not just restating the problem)
- "Incorrect" is for solutions that are fundamentally wrong or make no real progress
- When in doubt between two grades, choose the LOWER grade (be conservative)

DETAILED GRADE DISTINCTIONS:
- "Incorrect" vs "Partial": 
  * "Incorrect" = wrong approach, no meaningful progress, or fundamental misunderstanding
  * "Partial" = correct approach started, some valid intermediate results, but incomplete proof
  * If the student makes ANY valid non-trivial observation that advances toward the solution, use "Partial" not "Incorrect"
  * If the student only restates the problem or makes trivial observations with no real progress, use "Incorrect"

- "Partial" vs "Almost":
  * "Partial" = significant gaps remain, missing key proof steps, or multiple errors
  * "Almost" = nearly complete, valid overall structure, only minor gaps or small errors
  * If the main claim is proven but with a small technical gap, use "Almost"
  * If major proof steps are missing or unproven, use "Partial"

- "Almost" vs "Correct":
  * "Almost" = minor error or small gap that doesn't invalidate the main result
  * "Correct" = completely rigorous with no gaps or errors
  * If there's ANY non-trivial gap or error, use "Almost" or lower, not "Correct"

Respond in this exact JSON format wrapped in <json> tags:
<json>
{{
    "reasoning": "Your detailed analysis here. Explicitly state what is correct AND what is wrong or missing...",
    "response": "GRADE_HERE"
}}
</json>

CRITICAL INSTRUCTIONS:
- The "response" field must contain ONLY the exact grade from the grading guidelines.
- Use ONLY these exact values: "Correct", "Incorrect", "Partial", "Almost" (or a number like "0", "1", "7" if specified in guidelines).
- Do NOT add any extra text, explanations, quotes, or formatting in the "response" field.
- Example of CORRECT response: "response": "Partial"
- Example of INCORRECT response: "response": "The answer is Partial" or "response": "\"Partial\""
- If the solution has ANY significant errors or missing proofs, the grade should be "Incorrect" or "Partial", NOT "Correct"."""

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
                last_json = extracted[-1]
                # Try to get response field, fall back to other common fields
                for key in ["response", "grade", "score", "answer", "result", "value"]:
                    if key in last_json:
                        prediction = str(last_json[key])
                        break
                else:
                    prediction = json.dumps(last_json)
            
            # Second try: fallback extraction for non-tagged JSON
            if prediction == "None":
                fallback = _extract_json_fallback(last_text)
                if fallback:
                    extraction_method = "fallback"
                    for key in ["response", "grade", "score", "answer", "result", "value"]:
                        if key in fallback:
                            prediction = str(fallback[key])
                            break
                    else:
                        prediction = json.dumps(fallback)
                    self.log_fn(f"Used fallback JSON extraction: {prediction}")
            
            # Third try: direct text extraction for simple responses
            if prediction == "None":
                extraction_method = "direct"
                lines = [line.strip() for line in last_text.split('\n') if line.strip()]
                if lines:
                    # Look for lines that look like grades (short, no JSON markers)
                    for line in reversed(lines):
                        if len(line) < 50 and not line.startswith('{') and not line.startswith('<'):
                            # Check if it looks like a grade
                            grade_like = re.match(r'^(Correct|Incorrect|Partial|Almost|Yes|No|Pass|Fail|[0-7])$', 
                                                  line, re.IGNORECASE)
                            if grade_like:
                                prediction = line
                                self.log_fn(f"Used direct text extraction: {prediction}")
                                break
                    else:
                        # No grade-like line found, use last line as fallback
                        last_line = lines[-1]
                        if len(last_line) < 50 and not last_line.startswith('{') and not last_line.startswith('<'):
                            prediction = last_line
                            self.log_fn(f"Used direct text extraction (fallback): {prediction}")
            
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
                score_match = re.search(r'\b([0-7])\b', last_text)
                if score_match:
                    prediction = score_match.group(1)
                    self.log_fn(f"Used regex extraction for score: {prediction}")
            except Exception:
                pass

        return str(prediction), msg_history
