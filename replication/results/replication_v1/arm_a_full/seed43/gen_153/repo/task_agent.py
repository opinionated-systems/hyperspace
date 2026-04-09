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
    
    This function looks for valid grade labels in the text using multiple strategies:
    1. Look for labels in quotes (e.g., "Correct", "Partial")
    2. Look for labels at the end of sentences
    3. Look for labels preceded by keywords like "grade is", "the answer is", etc.
    4. Look for standalone labels on their own lines
    
    Returns the extracted grade or "None" if no valid grade found.
    """
    if not text or not text.strip():
        return "None"
    
    text_lower = text.lower()
    
    # Define valid grade labels (from most to least specific)
    valid_labels = ["Correct", "Incorrect", "Partial", "Almost"]
    
    # Strategy 1: Look for labels in quotes (e.g., "Correct", 'Partial')
    for label in valid_labels:
        # Match quoted labels
        pattern = rf'["\']\s*{label}\s*["\']'
        if re.search(pattern, text, re.IGNORECASE):
            return label
    
    # Strategy 2: Look for labels preceded by common keywords
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
            # Return the matched label with proper capitalization
            matched = match.group(1)
            for label in valid_labels:
                if matched.lower() == label.lower():
                    return label
    
    # Strategy 3: Look for standalone labels on their own lines
    lines = text.split('\n')
    for line in lines:
        line_stripped = line.strip()
        line_lower = line_stripped.lower()
        # Check if the line is just a valid label (possibly with punctuation)
        for label in valid_labels:
            # Match: "Correct", "Correct.", "(Correct)", "- Correct", etc.
            pattern = rf'^[\s\(\)\[\]\-\*]*{label}[\s\(\)\[\]\-\*\.]*$'
            if re.match(pattern, line_lower, re.IGNORECASE):
                return label
    
    # Strategy 4: Look for labels at the end of the text (last meaningful line)
    # Work backwards from the end
    for line in reversed(lines):
        line_stripped = line.strip()
        if not line_stripped:
            continue
        # Skip lines that are clearly not grades
        if line_stripped.startswith('```') or line_stripped.startswith('<json>'):
            continue
        # Check if this line contains a valid label
        for label in valid_labels:
            # Match the label as a whole word
            pattern = rf'\b{label}\b'
            if re.search(pattern, line_stripped, re.IGNORECASE):
                # Make sure it's not part of a larger word or sentence fragment
                # by checking if the line is short or the label is prominent
                if len(line_stripped) < 30:
                    return label
                # Or if it's preceded/followed by specific markers
                if re.search(rf'[:\-]\s*{label}\s*$', line_stripped, re.IGNORECASE):
                    return label
    
    # Strategy 5: Look for parenthesized labels like (Correct), (Partial), etc.
    for label in valid_labels:
        pattern = rf'\(\s*{label}\s*\)'
        if re.search(pattern, text, re.IGNORECASE):
            return label
    
    # Strategy 6: Check for numeric grades (0-7) and map to categorical labels
    # IMO scoring: 7=Correct, 6=Almost, 1=Partial, 0=Incorrect
    # Use findall to get all matches and pick the most confident one
    numeric_matches = re.findall(r'\b([0-7])\b', text)
    if numeric_matches:
        # If multiple scores found, prefer the last one (usually the final answer)
        score = int(numeric_matches[-1])
        if score == 7:
            return "Correct"
        elif score == 6:
            return "Almost"
        elif score == 1:
            return "Partial"
        elif score == 0:
            return "Incorrect"
        # For intermediate scores, use best judgment
        elif score >= 5:
            return "Almost"
        elif score >= 2:
            return "Partial"
        else:
            return "Incorrect"
    
    return "None"


def _validate_and_normalize_prediction(prediction: str, grading_guidelines: str) -> str:
    """Validate and normalize the prediction based on grading guidelines.
    
    Ensures the prediction matches expected format from grading guidelines.
    Handles various edge cases and normalizes common variations.
    Prioritizes extracting the exact label format from grading guidelines.
    
    Maps numeric IMO scores to categorical labels:
    - 7 -> Correct
    - 6 -> Almost  
    - 1 -> Partial
    - 0 -> Incorrect
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
    
    # Reject single-character predictions that aren't valid numeric scores (0-7)
    # This prevents partial text extraction like 'n' from "Incorrect" or 'I' from "Incorrect"
    if len(prediction) == 1 and prediction not in '01234567':
        # Try to extract a valid grade from the text
        extracted = _extract_grade_from_text(prediction, grading_guidelines)
        if extracted != "None":
            return extracted
        return "None"
    
    # Reject two-character predictions that are likely partial matches
    # (like "In" from "Incorrect", "Co" from "Correct", "Pa" from "Partial", "Al" from "Almost")
    if len(prediction) == 2:
        partial_prefixes = ['in', 'co', 'pa', 'al']
        if pred_lower in partial_prefixes:
            # Try to extract a valid grade from the text
            extracted = _extract_grade_from_text(prediction, grading_guidelines)
            if extracted != "None":
                return extracted
            return "None"
    
    # Early rejection: if prediction is too long or contains LaTeX/math, it's likely not a grade
    if len(prediction) > 100:
        # Try to extract a grade from within the long text
        extracted = _extract_grade_from_text(prediction, grading_guidelines)
        if extracted != "None":
            return extracted
        return "None"
    
    # Reject predictions that look like LaTeX fragments or math expressions
    latex_indicators = ['\\[', '\\]', '\\begin', '\\end', '\\frac', '\\sum', 
                       '\\int', '\\prod', '\\sqrt', '\\left', '\\right', '$']
    for indicator in latex_indicators:
        if indicator in prediction:
            # Try to extract a valid grade from the text
            extracted = _extract_grade_from_text(prediction, grading_guidelines)
            if extracted != "None":
                return extracted
            return "None"
    
    # Handle numeric IMO scores (0-7) and map to categorical labels
    # This is critical for the IMO grading task where points map to categories
    if prediction.isdigit() or (prediction.startswith('-') and prediction[1:].isdigit()):
        try:
            score = int(prediction)
            # IMO score mapping: 7=Correct, 6=Almost, 1=Partial, 0=Incorrect
            if score == 7:
                return "Correct"
            elif score == 6:
                return "Almost"
            elif score == 1:
                return "Partial"
            elif score == 0:
                return "Incorrect"
            # For other scores, try to infer based on guidelines
            elif score >= 5:
                return "Almost"
            elif score >= 2:
                return "Partial"
            else:
                return "Incorrect"
        except ValueError:
            pass
    
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
        # Check for exact matches first
        if pred_lower == "correct":
            return "Correct"
        if pred_lower == "incorrect":
            return "Incorrect"
        # Then check for partial matches
        if "incorrect" in pred_lower or pred_lower in ["wrong", "false", "error", "invalid"]:
            return "Incorrect"
        if "correct" in pred_lower or pred_lower in ["right", "true", "valid"]:
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
    
    # FINAL VALIDATION: For IMO grading task, only return valid categorical labels
    # If the prediction doesn't match any expected format, try one more extraction
    valid_categories = ["Correct", "Incorrect", "Partial", "Almost"]
    pred_lower = prediction.lower()
    
    # Check if it's a valid category
    for category in valid_categories:
        if pred_lower == category.lower():
            return category
    
    # If prediction is short (1-3 chars) and not a valid category, it's likely garbage
    if len(prediction) <= 3:
        return "None"
    
    # Last attempt: try to find a valid category in the prediction text
    for category in valid_categories:
        if category.lower() in pred_lower:
            return category
    
    # If all else fails, return "None" to indicate extraction failure
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

## Grade Category Definitions - BE STRICT AND CONSERVATIVE

Use these precise definitions when evaluating. When in doubt, choose the LOWER grade:

- **Correct** (7 points): The solution is COMPLETE and FULLY CORRECT. ALL key steps are present, logically sound, and lead to the correct answer. The proof is rigorous with no gaps in reasoning. Only award this if you are confident the solution would receive full marks.
  - Requirements: Complete proof/correct answer + all key steps + rigorous reasoning + no significant gaps
  - Do NOT award if: missing key steps, logical gaps, incomplete proof, or minor errors that affect correctness

- **Incorrect** (0 points): The solution is FUNDAMENTALLY WRONG or shows NO MEANINGFUL PROGRESS. The approach is flawed, the answer is wrong with invalid reasoning, or the student made no substantial progress toward the solution.
  - Award this if: wrong approach, critical errors, no valid reasoning, or answer is incorrect with no salvageable work
  - Do NOT award "Partial" just because the student wrote something - there must be genuine mathematical progress
  - **KEY RULE**: If the final answer is wrong AND the reasoning has significant flaws, the grade is "Incorrect"
  - **KEY RULE**: If the student made only trivial observations (e.g., restating the problem, basic algebraic manipulations that don't advance toward the solution), the grade is "Incorrect"

- **Partial** (1 point): The solution shows MEANINGFUL PROGRESS with SOME CORRECT STEPS, but has SIGNIFICANT GAPS or MAJOR ERRORS. The student understood PART of the problem and made NON-TRIVIAL progress, but the solution is incomplete or contains substantial errors.
  - Requirements: Some correct steps + genuine understanding of part of the problem + non-trivial progress
  - Do NOT award if: only trivial observations, incorrect approach with minor correct calculations, or mostly wrong with tiny correct fragments
  - Do NOT award "Correct" unless the solution is truly complete
  - **KEY RULE**: Only award "Partial" if the student made SUBSTANTIAL progress toward the solution (e.g., correctly solved a significant sub-problem, or made meaningful progress on the main approach)

- **Almost** (6 points): The solution is NEARLY COMPLETE with only MINOR ERRORS or SMALL OMISSIONS. The main ideas are correct and well-executed, but there are small mistakes that don't invalidate the overall approach. This is significantly better than "Partial" but not quite "Correct".
  - Requirements: Nearly complete proof + correct main ideas + minor errors only + small gaps in reasoning
  - The errors must be truly minor and easily fixable

## Decision Framework - FOLLOW THIS EXACTLY

When grading, use this step-by-step decision process:

**Step 1: Is the solution COMPLETE and FULLY CORRECT?**
- Does it have ALL key steps from the official solution?
- Is the reasoning rigorous with no gaps?
- Is the final answer correct?
- If YES to all → **Correct**
- If NO → Continue to Step 2

**Step 2: Is the solution NEARLY COMPLETE with only MINOR errors?**
- Are the main ideas correct?
- Are errors truly minor and easily fixable?
- If YES → **Almost**
- If NO → Continue to Step 3

**Step 3: Does the solution show MEANINGFUL PROGRESS?**
- Did the student make SUBSTANTIAL progress (not just trivial observations)?
- Did they correctly solve a significant sub-problem?
- Did they demonstrate genuine understanding of PART of the problem?
- If YES → **Partial**
- If NO → **Incorrect**

**CRITICAL**: If the student's answer is mostly wrong with only minor correct fragments, the grade is **Incorrect**, NOT Partial.

## Critical Grading Principles

1. **BE CONSERVATIVE**: When uncertain, choose the LOWER grade. It's better to under-grade than over-grade.
2. **"Correct" is rare**: Only award "Correct" for truly complete, rigorous solutions. Most student solutions will be "Partial" or "Incorrect".
3. **"Partial" requires genuine progress**: Don't award "Partial" for solutions that are mostly wrong with only trivial correct observations.
4. **Compare to official solution**: The official solution shows the expected standard. Student solutions should match this rigor to be "Correct".
5. **Watch for false progress**: Some solutions appear to make progress but contain fundamental flaws. These should be "Incorrect", not "Partial".
6. **Wrong answer = Incorrect**: If the final answer is wrong and there's no substantial correct work, the grade is "Incorrect".

## Your Task

1. Carefully read the problem and official solution
2. Analyze the student's answer step by step
3. Compare the student's work to the official solution - note ALL missing steps or errors
4. Identify what the student did correctly and where they made errors
5. Apply the Decision Framework above step-by-step
6. Determine the appropriate grade based on the STRICT definitions
7. **When in doubt, choose the more conservative (lower) grade**

## Response Format

You MUST respond in this exact JSON format wrapped in <json> tags:

<json>
{{
    "reasoning": "Your detailed analysis here. Explain what the student did right and wrong, and why you chose this grade. Be explicit about any gaps or errors found.",
    "response": "GRADE"
}}
</json>

CRITICAL INSTRUCTIONS:
- The "response" field MUST contain ONLY one of these exact values: "Correct", "Incorrect", "Partial", or "Almost"
- Do NOT add any extra text, quotes, or explanations in the "response" field
- Do NOT use LaTeX, markdown, or any formatting in the "response" field
- The response should be a single word: Correct, Incorrect, Partial, or Almost
- Note: The grading guidelines may reference numeric scores (0, 1, 6, 7). These map to the categories as: 7=Correct, 6=Almost, 1=Partial, 0=Incorrect
- **BE STRICT AND CONSERVATIVE IN YOUR GRADING**"""

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
