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
    
    # SECOND: Handle specific known grading formats with better priority
    
    # IMPORTANT: Check for more specific grades BEFORE general "Correct"/"Incorrect"
    # This prevents "partially correct" from being matched as "Correct"
    
    # First, check for compound phrases that include "partial" or "almost"
    # These should be treated as the more specific grade, not "Correct"
    if 'partially correct' in pred_lower or 'partial credit' in pred_lower:
        return "Partial"
    if 'almost correct' in pred_lower:
        return "Almost"
    
    # Check for "Partial" format FIRST (before Correct/Incorrect)
    if "partial" in grading_guidelines.lower():
        if pred_lower == "partial":
            return "Partial"
        # Check for exact match with parentheses
        if pred_lower == "(partial)":
            return "Partial"
        # Only match "partial" as a standalone word, not as part of "partially"
        if re.search(r'\bpartial\b', pred_lower):
            return "Partial"
    
    # Check for "Almost" format FIRST (before Correct/Incorrect)
    if "almost" in grading_guidelines.lower():
        if pred_lower == "almost":
            return "Almost"
        if pred_lower == "(almost)":
            return "Almost"
        if re.search(r'\balmost\b', pred_lower):
            return "Almost"
    
    # Check for "Correct"/"Incorrect" format with better precision
    if "correct" in grading_guidelines.lower() or "incorrect" in grading_guidelines.lower():
        # Check for exact matches first
        if pred_lower == "correct":
            return "Correct"
        if pred_lower == "incorrect":
            return "Incorrect"
        # Check for synonyms that mean "Correct"
        if pred_lower in ["full", "complete", "right", "true", "valid", "perfect", "solved"]:
            return "Correct"
        # Check for synonyms that mean "Incorrect"
        if pred_lower in ["wrong", "false", "error", "invalid", "unsolved", "failed", "fail"]:
            return "Incorrect"
        # Then check for partial matches - use word boundaries to avoid partial matches
        if re.search(r'\bincorrect\b', pred_lower):
            return "Incorrect"
        # Only match "correct" if it's not preceded by "partial" or "almost"
        # This prevents "partially correct" or "almost correct" from matching as "Correct"
        if re.search(r'\bcorrect\b', pred_lower):
            # Check if "partial" or "almost" appears anywhere in the prediction
            # If so, don't match as "Correct" - the prediction is ambiguous
            if 'partial' in pred_lower or 'almost' in pred_lower:
                # Don't return "Correct" - let it fall through to other checks
                pass
            else:
                return "Correct"
    
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


def _extract_grade_from_text(text: str, grading_guidelines: str) -> str | None:
    """Extract grade from raw text using grading guidelines as reference.
    
    This function looks for valid grade values in the text, avoiding
    fragments from student answers or other content.
    
    Args:
        text: The text to search for grades.
        grading_guidelines: The grading guidelines to determine valid grades.
        
    Returns:
        The extracted grade or None if no valid grade found.
    """
    text_lower = text.lower()
    
    # Extract valid grade labels from grading guidelines
    valid_grades = set()
    
    # Look for parenthesized labels: (Correct), (Partial), etc.
    for match in re.finditer(r'\(([A-Za-z]+)\)', grading_guidelines):
        label = match.group(1)
        valid_grades.add(label.lower())
    
    # Look for standalone capitalized labels
    for match in re.finditer(r'(?:^|\n|\s)\s*([A-Z][a-z]+)\s*(?:\n|$|\d+\.|\s)', grading_guidelines):
        label = match.group(1)
        if label not in ["The", "A", "An", "This", "That"]:
            valid_grades.add(label.lower())
    
    # Add common grade formats
    valid_grades.update(['correct', 'incorrect', 'partial', 'almost', 'none'])
    valid_grades.update(['0', '1', '2', '3', '4', '5', '6', '7'])  # IMO scores
    valid_grades.update(['yes', 'no', 'pass', 'fail'])
    
    # Search for valid grades in the text
    # Look for the grade in specific contexts to avoid matching student answer content
    
    # IMPORTANT: Check for more specific grades BEFORE general "Correct"/"Incorrect"
    # This prevents "partially correct" from being matched as "Correct"
    
    # Pattern 1: Look for grade after common prefixes (highest priority)
    prefix_patterns = [
        r'["\']?response["\']?\s*[:=]\s*["\']?([^"\'\n]{1,30})["\']?',
        r'["\']?grade["\']?\s*[:=]\s*["\']?([^"\'\n]{1,30})["\']?',
        r'["\']?score["\']?\s*[:=]\s*["\']?([^"\'\n]{1,30})["\']?',
        r'grade\s*(?:is|:)?\s*["\']?([^"\'\n]{1,30})["\']?',
        r'score\s*(?:is|:)?\s*["\']?([^"\'\n]{1,30})["\']?',
        r'response\s*(?:is|:)?\s*["\']?([^"\'\n]{1,30})["\']?',
    ]
    
    for pattern in prefix_patterns:
        match = re.search(pattern, text_lower)
        if match:
            candidate = match.group(1).strip().rstrip('",\'})').strip()
            # Clean up the candidate
            candidate_clean = candidate.lower()
            # Remove parentheses if present
            if candidate_clean.startswith('(') and candidate_clean.endswith(')'):
                candidate_clean = candidate_clean[1:-1]
            
            # Check for exact matches first
            if candidate_clean in valid_grades:
                if candidate_clean in ['correct', 'incorrect', 'partial', 'almost', 'none']:
                    return candidate_clean.capitalize()
                return candidate_clean
            
            # Check for partial matches (e.g., "partial" in "partial credit")
            for grade in ['partial', 'almost', 'correct', 'incorrect']:
                if grade in candidate_clean:
                    return grade.capitalize()
    
    # Pattern 2: Look for standalone valid grades in the last few lines
    # (grades are often at the end of reasoning)
    lines = text.split('\n')
    # Check last 10 lines first (most likely location for final grade)
    for line in reversed(lines[-10:]):
        line = line.strip()
        line_lower = line.lower()
        
        # Skip lines that are clearly not grades
        if len(line) > 30:
            continue
        if any(c in line for c in ['$', '\\', '_', '^', '{', '}', '`']):
            continue
        if line.startswith('//') or line.startswith('/*'):
            continue
            
        # Check for exact match
        for grade in valid_grades:
            if line_lower == grade or line_lower == f'({grade})':
                if grade in ['correct', 'incorrect', 'partial', 'almost', 'none']:
                    return grade.capitalize()
                return grade
    
    # Pattern 3: Look for "Partial" and "Almost" FIRST (before Correct/Incorrect)
    # This prevents "partially correct" from being matched as "Correct"
    if 'partial' in valid_grades:
        # Look for standalone "partial" with word boundaries
        if re.search(r'\bpartial\b', text_lower):
            return "Partial"
    
    if 'almost' in valid_grades:
        if re.search(r'\balmost\b', text_lower):
            return "Almost"
    
    # Pattern 4: Look for "Correct" and "Incorrect" with word boundaries
    # Only match "correct" if it's not preceded by "partial" or "almost"
    if 'incorrect' in valid_grades:
        if re.search(r'\bincorrect\b', text_lower):
            return "Incorrect"
    
    if 'correct' in valid_grades:
        # Check for "correct" but avoid "partially correct" or "almost correct"
        correct_match = re.search(r'\bcorrect\b', text_lower)
        if correct_match:
            # Check surrounding context
            pos = correct_match.start()
            context_before = text_lower[max(0, pos-30):pos]
            context_after = text_lower[pos+7:pos+30]
            if 'partial' not in context_before and 'partial' not in context_after and \
               'almost' not in context_before and 'almost' not in context_after:
                return "Correct"
    
    # Pattern 5: Look for IMO numeric scores (0-7) as standalone values
    # Search from the end of the text (more likely to be the final grade)
    score_matches = list(re.finditer(r'\b([0-7])\b', text))
    if score_matches:
        # Use the last match (more likely to be the final grade decision)
        for score_match in reversed(score_matches):
            pos = score_match.start()
            context = text[max(0, pos-30):min(len(text), pos+30)]
            context_lower = context.lower()
            # Avoid matching if surrounded by math notation or part of larger number
            if '$' not in context and '\\' not in context[:20]:
                # Check if this looks like a grade context
                if any(word in context_lower for word in ['grade', 'score', 'mark', 'point', 'response']):
                    return score_match.group(1)
        # If no context match, return the last score found
        return score_matches[-1].group(1)
    
    return None


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

        instruction = f"""You are an expert grader for the International Mathematical Olympiad (IMO). Your task is to evaluate the student's answer and provide a grade.

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

Analyze the student's answer using this structured approach:

**Step 1: Problem Analysis**
- What is the problem asking for?
- What are the key theorems/techniques needed?
- What would a complete solution look like?

**Step 2: Student Answer Analysis**
- What approach did the student take?
- Which steps are correct?
- Which steps are incorrect or missing?
- Is there meaningful mathematical progress?

**Step 3: Grade Decision**
Apply the decision tree below to determine the grade.

## Grading Decision Tree - FOLLOW THIS EXACTLY

**START HERE: Is the solution complete and rigorous?**
→ If YES (all key steps present, no gaps, would get full marks): **"Correct"**
→ If NO, continue to next question.

**Does the solution have significant correct progress toward the answer?**
→ If NO (wrong approach, no progress, trivial/empty, fundamental misunderstanding): **"Incorrect"**
→ If YES (some correct steps, meaningful progress made), continue to next question.

**Is the solution nearly complete with only minor issues?**
→ If YES (small gaps that don't affect main argument, minor errors with correct approach): **"Almost"**
→ If NO (significant gaps remain, incomplete but has correct ideas): **"Partial"**

## Grade Definitions with Examples

**"Correct"** - Complete, rigorous solution
- Example: Full proof with all steps correct, would receive 7/7 in competition
- Example: Correct approach with all key insights, properly justified

**"Almost"** - Nearly complete, minor issues only
- Example: Correct proof with one small computational error
- Example: Complete solution missing one minor justification
- Example: Correct approach with small gap that doesn't affect conclusion

**"Partial"** - Meaningful progress but significant gaps
- Example: Correct initial setup and first few steps, then stops
- Example: Has right idea but proof is incomplete or has significant errors
- Example: Partial case analysis with some correct cases
- DEFAULT for incomplete solutions with some correct work

**"Incorrect"** - No meaningful progress
- Example: Completely wrong approach to the problem
- Example: Trivial or empty answer (just restates problem)
- Example: Fundamental misunderstanding of key concepts
- Example: Answer is nonsense or unrelated to problem

## IMO Numeric Scoring (if applicable)
- "7" = Complete solution (equivalent to "Correct")
- "5-6" = Almost complete (equivalent to "Almost")
- "1-2" = Partial progress (equivalent to "Partial")
- "0" = No progress (equivalent to "Incorrect")

## Response Format

Respond in this exact JSON format wrapped in <json> tags:
<json>
{{
    "reasoning": "Your detailed analysis here...",
    "response": "GRADE_HERE"
}}
</json>

## CRITICAL RULES - FOLLOW EXACTLY

1. The "response" field must contain ONLY one exact grade: "Correct", "Almost", "Partial", "Incorrect", or "0"-"7"
2. Do NOT add explanations in the response field
3. Be CONSERVATIVE - when in doubt, choose the LOWER grade
4. "Incorrect" is for answers with NO meaningful progress - be honest!
5. "Partial" requires SOME correct progress - not just random attempts
6. Only "Correct" for truly complete solutions with no significant issues
7. Use the decision tree above - do not skip steps
"""

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
                    # Third try: use grade extraction with guidelines
                    extraction_method = "grade_extraction"
                    grade = _extract_grade_from_text(last_text, grading_guidelines)
                    if grade:
                        prediction = grade
                        self.log_fn(f"Used grade extraction: {prediction}")
            
            # Validate and normalize the prediction
            original_prediction = prediction
            prediction = _validate_and_normalize_prediction(prediction, grading_guidelines)
            
            if original_prediction != prediction:
                self.log_fn(f"Normalized prediction from '{original_prediction}' to '{prediction}'")
            
            self.log_fn(f"Extraction method: {extraction_method}, final prediction: {prediction}")
            
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Last resort: try grade extraction
            try:
                grade = _extract_grade_from_text(last_text, grading_guidelines)
                if grade:
                    prediction = grade
                    self.log_fn(f"Used emergency grade extraction: {prediction}")
            except Exception:
                pass

        return str(prediction), msg_history
