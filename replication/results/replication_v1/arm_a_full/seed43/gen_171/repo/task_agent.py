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
    """Extract grade label directly from text when JSON extraction fails.
    
    This function looks for valid grade labels in the text using multiple strategies.
    Returns the extracted grade or "None" if no valid grade found.
    """
    if not text or not text.strip():
        return "None"
    
    text_lower = text.lower()
    
    # Define valid grade labels (from most to least specific)
    valid_labels = ["Correct", "Incorrect", "Partial", "Almost"]
    
    # Strategy 1: Look for labels in the "response" field context
    # This is the most common format from the LLM
    response_match = re.search(r'"response"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
    if response_match:
        value = response_match.group(1).strip()
        for label in valid_labels:
            if value.lower() == label.lower():
                return label
    
    # Strategy 2: Look for labels in single quotes in response field
    response_match2 = re.search(r"'response'\s*:\s*'([^']+)'", text, re.IGNORECASE)
    if response_match2:
        value = response_match2.group(1).strip()
        for label in valid_labels:
            if value.lower() == label.lower():
                return label
    
    # Strategy 3: Look for labels in quotes (e.g., "Correct", 'Partial')
    for label in valid_labels:
        pattern = rf'["\']\s*{label}\s*["\']'
        if re.search(pattern, text, re.IGNORECASE):
            return label
    
    # Strategy 4: Look for labels preceded by common keywords
    keyword_patterns = [
        rf'grade\s*(?:is|:)\s*["\']?\s*({"|".join(valid_labels)})\s*["\']?',
        rf'(?:the\s+)?answer\s*(?:is|:)\s*["\']?\s*({"|".join(valid_labels)})\s*["\']?',
        rf'(?:the\s+)?score\s*(?:is|:)\s*["\']?\s*({"|".join(valid_labels)})\s*["\']?',
        rf'response\s*(?:is|:)\s*["\']?\s*({"|".join(valid_labels)})\s*["\']?',
        rf'(?:i\s+(?:would\s+)?(?:say|give|assign|rate))\s*["\']?\s*({"|".join(valid_labels)})',
        rf'(?:this\s+(?:is|should\s+be))\s*["\']?\s*({"|".join(valid_labels)})',
    ]
    
    for pattern in keyword_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            matched = match.group(1)
            for label in valid_labels:
                if matched.lower() == label.lower():
                    return label
    
    # Strategy 5: Look for standalone labels on their own lines
    lines = text.split('\n')
    for line in lines:
        line_stripped = line.strip()
        line_lower = line_stripped.lower()
        for label in valid_labels:
            # Match: "Correct", "Correct.", "(Correct)", "- Correct", etc.
            pattern = rf'^[\s\(\)\[\]\-\*]*{label}[\s\(\)\[\]\-\*\.]*$'
            if re.match(pattern, line_lower, re.IGNORECASE):
                return label
    
    # Strategy 6: Look for labels at the end of the text (last meaningful line)
    for line in reversed(lines):
        line_stripped = line.strip()
        if not line_stripped:
            continue
        if line_stripped.startswith('```') or line_stripped.startswith('<json>'):
            continue
        for label in valid_labels:
            pattern = rf'\b{label}\b'
            if re.search(pattern, line_stripped, re.IGNORECASE):
                if len(line_stripped) < 30:
                    return label
                if re.search(rf'[:\-]\s*{label}\s*$', line_stripped, re.IGNORECASE):
                    return label
    
    # Strategy 7: Look for parenthesized labels like (Correct), (Partial), etc.
    for label in valid_labels:
        pattern = rf'\(\s*{label}\s*\)'
        if re.search(pattern, text, re.IGNORECASE):
            return label
    
    # Strategy 8: Check for numeric grades (0-7) and map to categorical labels
    numeric_match = re.search(r'\b([0-7])\b', text)
    if numeric_match:
        score = int(numeric_match.group(1))
        if score == 7:
            return "Correct"
        elif score == 6:
            return "Almost"
        elif score == 1:
            return "Partial"
        elif score == 0:
            return "Incorrect"
        elif score >= 5:
            return "Almost"
        elif score >= 2:
            return "Partial"
        else:
            return "Incorrect"
    
    # Strategy 9: Look for any occurrence of the valid labels as whole words
    # But be careful not to match partial words like "In" from "Incorrect"
    for label in valid_labels:
        if re.search(rf'\b{label}\b', text, re.IGNORECASE):
            return label
    
    # Strategy 10: Handle common variations and misspellings
    variation_map = {
        "correctly": "Correct",
        "right": "Correct",
        "true": "Correct",
        "valid": "Correct",
        "yes": "Correct",
        "wrong": "Incorrect",
        "false": "Incorrect",
        "error": "Incorrect",
        "invalid": "Incorrect",
        "no": "Incorrect",
        "part": "Partial",
        "partially": "Partial",
        "some": "Partial",
        "nearly": "Almost",
        "close": "Almost",
        "mostly": "Almost",
    }
    
    for variation, mapped_label in variation_map.items():
        if re.search(rf'\b{variation}\b', text_lower):
            return mapped_label
    
    return "None"


def _validate_and_normalize_prediction(prediction: str, grading_guidelines: str) -> str:
    """Validate and normalize the prediction based on grading guidelines.
    
    Ensures the prediction matches expected format from grading guidelines.
    Handles various edge cases and normalizes common variations.
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
    
    # Define valid grade labels
    valid_labels = ["Correct", "Incorrect", "Partial", "Almost"]
    
    # Check for exact match first (case-insensitive)
    for label in valid_labels:
        if pred_lower == label.lower():
            return label
    
    # Handle common word variations and misspellings
    variation_map = {
        # Correct variations
        "correctly": "Correct",
        "right": "Correct",
        "true": "Correct",
        "valid": "Correct",
        "yes": "Correct",
        # Incorrect variations
        "wrong": "Incorrect",
        "false": "Incorrect",
        "error": "Incorrect",
        "invalid": "Incorrect",
        "no": "Incorrect",
        # Partial variations
        "part": "Partial",
        "partially": "Partial",
        "some": "Partial",
        # Almost variations
        "nearly": "Almost",
        "close": "Almost",
        "mostly": "Almost",
    }
    
    # Check for exact variation match
    if pred_lower in variation_map:
        return variation_map[pred_lower]
    
    # If prediction is too short (single char) and not a valid label, reject it
    if len(prediction) == 1:
        # Only accept single digits 0-7 which map to grades
        if prediction.isdigit():
            score = int(prediction)
            if score == 7:
                return "Correct"
            elif score == 6:
                return "Almost"
            elif score == 1:
                return "Partial"
            elif score == 0:
                return "Incorrect"
        return "None"
    
    # If prediction is too long or contains LaTeX, try to extract grade from it
    if len(prediction) > 100:
        extracted = _extract_grade_from_text(prediction, grading_guidelines)
        if extracted != "None":
            return extracted
        return "None"
    
    # Reject predictions that look like LaTeX fragments
    latex_indicators = ['\\[', '\\]', '\\begin', '\\end', '\\frac', '\\sum', 
                       '\\int', '\\prod', '\\sqrt', '\\left', '\\right', '$']
    for indicator in latex_indicators:
        if indicator in prediction:
            extracted = _extract_grade_from_text(prediction, grading_guidelines)
            if extracted != "None":
                return extracted
            return "None"
    
    # Handle numeric IMO scores (0-7) and map to categorical labels
    if prediction.isdigit() or (prediction.startswith('-') and prediction[1:].isdigit()):
        try:
            score = int(prediction)
            if score == 7:
                return "Correct"
            elif score == 6:
                return "Almost"
            elif score == 1:
                return "Partial"
            elif score == 0:
                return "Incorrect"
            elif score >= 5:
                return "Almost"
            elif score >= 2:
                return "Partial"
            else:
                return "Incorrect"
        except ValueError:
            pass
    
    # Check for partial matches with valid labels - but require word boundaries
    # to avoid matching "In" as "Incorrect" or "Cor" as "Correct"
    for label in valid_labels:
        # Use word boundary check to avoid partial word matches
        if re.search(rf'\b{label.lower()}\b', pred_lower):
            return label
    
    # Check for variation matches with word boundaries
    for variation, mapped_label in variation_map.items():
        if re.search(rf'\b{variation}\b', pred_lower):
            return mapped_label
    
    # Try to extract grade from the prediction text
    extracted = _extract_grade_from_text(prediction, grading_guidelines)
    if extracted != "None":
        return extracted
    
    # SECOND: Handle specific known grading formats with improved logic
    
    # Check for "Correct"/"Incorrect" format with better precision
    if "correct" in grading_guidelines.lower() or "incorrect" in grading_guidelines.lower():
        # Check for exact matches first
        if pred_lower == "correct":
            return "Correct"
        if pred_lower == "incorrect":
            return "Incorrect"
        # Then check for partial matches with word boundaries
        if re.search(r'\bincorrect\b', pred_lower) or pred_lower in ["wrong", "false", "error", "invalid"]:
            return "Incorrect"
        if re.search(r'\bcorrect\b', pred_lower) or pred_lower in ["right", "true", "valid"]:
            return "Correct"
    
    # Check for "Partial" format - prioritize exact match
    if "partial" in grading_guidelines.lower():
        if pred_lower == "partial":
            return "Partial"
        # Check for "partial" as a standalone word
        if re.search(r'\bpartial\b', pred_lower):
            return "Partial"
    
    # Check for "Almost" format - prioritize exact match
    if "almost" in grading_guidelines.lower():
        if pred_lower == "almost":
            return "Almost"
        # Check for "almost" as a standalone word
        if re.search(r'\balmost\b', pred_lower):
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

        instruction = f"""You are an expert grader for the International Mathematical Olympiad (IMO). Your task is to evaluate a student's mathematical solution and assign the appropriate grade.

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

## Grade Category Definitions

Use these precise definitions when evaluating. Pay special attention to the boundary cases:

### **Correct** (7 points)
The solution is complete, correct, and would receive full marks. ALL of the following must be true:
- All key steps are present and logically sound
- The final answer is correct
- The reasoning is valid throughout
- Minor notation issues that don't affect correctness are acceptable

**CRITICAL**: Only assign "Correct" if the solution is truly complete with no significant gaps or errors.

### **Almost** (6 points) 
The solution is nearly complete and correct, with only minor errors or omissions. This is DIFFERENT from "Correct" - look for:
- The main ideas and approach are correct
- Most of the solution is well-executed
- There are small mistakes, minor gaps in reasoning, or slight omissions
- The error(s) are not fundamental - they're at the detail level
- The student clearly understood the problem but made minor execution errors

**EXAMPLES of "Almost":**
- Correct approach and most steps, but a small calculation error at the end
- Correct solution but missing one minor case or edge condition
- Correct main argument but with a small logical gap that's easily fixable
- Correct solution but presented with unclear notation that causes minor confusion

### **Partial** (1 point)
The solution shows meaningful progress with some correct steps, but has significant gaps or errors. This is DIFFERENT from "Incorrect" - look for:
- The student understood PART of the problem (not just the statement)
- There are some correct, non-trivial steps toward the solution
- The approach may be partially correct or incomplete
- There may be major errors, but also some valid progress

**EXAMPLES of "Partial":**
- Correctly identified what needs to be proved but couldn't complete the proof
- Made some valid observations or simplifications but got stuck
- Started with a reasonable approach but made a critical error partway through
- Has some correct intermediate results but couldn't connect them to the final answer
- Shows understanding of relevant concepts but couldn't apply them correctly

### **Incorrect** (0 points)
The solution is fundamentally wrong or shows no meaningful progress. ALL of the following indicate "Incorrect":
- The approach is completely wrong or irrelevant
- No meaningful progress toward the solution
- The answer is wrong with no valid reasoning
- Only restates the problem or makes trivial observations
- Contains fundamental conceptual errors

**CRITICAL**: Only assign "Incorrect" if there is truly NO meaningful progress. If the student showed any understanding or made any non-trivial correct steps, consider "Partial" instead.

## Your Task

1. Carefully read the problem and official solution
2. Analyze the student's answer step by step
3. Compare the student's work to the official solution
4. Identify what the student did correctly and where they made errors
5. **CRITICAL**: Before deciding between boundary categories (Almost/Correct or Partial/Incorrect), re-read the definitions above carefully
6. Determine the appropriate grade based on the definitions above and the grading guidelines

## Decision Framework for Boundary Cases

When deciding between two categories, ask yourself these specific questions:

**Correct vs Almost:**
- Is the solution COMPLETE with ALL key steps? If yes → Correct
- Are there ANY significant gaps or errors? If yes → Almost
- Would this solution receive full marks (7/7)? If yes → Correct, else → Almost

**Almost vs Partial:**
- Did the student understand the MAIN IDEAS of the problem? If yes → Almost
- Is the error at the DETAIL level (calculation, minor gap) or CONCEPTUAL level? Detail → Almost, Conceptual → Partial
- Could a small fix make this solution correct? If yes → Almost, else → Partial

**Partial vs Incorrect:**
- Did the student make ANY non-trivial correct progress? If yes → Partial
- Are there ANY valid intermediate results or correct observations? If yes → Partial
- Is there more than just restating the problem or trivial observations? If yes → Partial
- Did the student show understanding of relevant concepts, even if misapplied? If yes → Partial

## Response Format

You MUST respond in this exact JSON format wrapped in <json> tags:

<json>
{{
    "reasoning": "Your detailed analysis here. First analyze what the student did right and wrong. Then explicitly state which boundary case decision framework you used (Correct vs Almost, Almost vs Partial, or Partial vs Incorrect) and why you chose your final grade.",
    "response": "Correct"
}}
</json>

OR

<json>
{{
    "reasoning": "Your detailed analysis here. First analyze what the student did right and wrong. Then explicitly state which boundary case decision framework you used (Correct vs Almost, Almost vs Partial, or Partial vs Incorrect) and why you chose your final grade.",
    "response": "Incorrect"
}}
</json>

OR

<json>
{{
    "reasoning": "Your detailed analysis here. First analyze what the student did right and wrong. Then explicitly state which boundary case decision framework you used (Correct vs Almost, Almost vs Partial, or Partial vs Incorrect) and why you chose your final grade.",
    "response": "Partial"
}}
</json>

OR

<json>
{{
    "reasoning": "Your detailed analysis here. First analyze what the student did right and wrong. Then explicitly state which boundary case decision framework you used (Correct vs Almost, Almost vs Partial, or Partial vs Incorrect) and why you chose your final grade.",
    "response": "Almost"
}}
</json>

CRITICAL INSTRUCTIONS - READ CAREFULLY:
1. The "response" field MUST contain ONLY one of these exact values: "Correct", "Incorrect", "Partial", or "Almost"
2. Do NOT add any extra text, quotes, or explanations in the "response" field
3. Do NOT use LaTeX, markdown, or any formatting in the "response" field
4. The response should be a single word: Correct, Incorrect, Partial, or Almost
5. Do NOT use variations like "Correctly", "Partially", "Almost correct", etc.
6. Do NOT use abbreviations like "Corr", "Incorr", "Part", etc.
7. Note: The grading guidelines may reference numeric scores (0, 1, 6, 7). These map to the categories as: 7=Correct, 6=Almost, 1=Partial, 0=Incorrect

EXAMPLES OF CORRECT RESPONSES:
- "response": "Correct"
- "response": "Incorrect"  
- "response": "Partial"
- "response": "Almost"

EXAMPLES OF INCORRECT RESPONSES (DO NOT USE):
- "response": "Correctly" (wrong word)
- "response": "In" (incomplete)
- "response": "Partially" (wrong word)
- "response": "Almost correct" (extra words)
- "response": "7" (use "Correct" instead)
- "response": "0" (use "Incorrect" instead)"""

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
