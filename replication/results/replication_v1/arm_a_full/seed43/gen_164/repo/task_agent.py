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
    Also handles nested JSON objects within the content.
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
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks within the content
            code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', inner, re.DOTALL)
            if code_block_match:
                try:
                    results.append(json.loads(code_block_match.group(1)))
                except json.JSONDecodeError:
                    continue
            else:
                continue
    return results or None


def _extract_json_fallback(text: str) -> dict | None:
    """Fallback JSON extraction for non-tagged JSON or markdown code blocks.

    Tries to find JSON in markdown code blocks or raw JSON objects.
    Uses a robust approach to handle nested braces.
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
                continue
    
    # Use a more efficient approach: find all JSON-like substrings and validate
    # Look for patterns that look like JSON objects with expected keys
    json_candidates = []
    
    # Find all potential JSON object starts (looking for {" or {\n")
    for match in re.finditer(r'\{\s*"', text):
        start = match.start()
        # Use a stack-based approach to find the matching end brace
        stack = []
        in_string = False
        escape_next = False
        
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
                    stack.append('{')
                elif char == '}':
                    if stack:
                        stack.pop()
                        if not stack:
                            # Found a complete JSON object
                            candidate = text[start:i+1]
                            json_candidates.append(candidate)
                            break
                    else:
                        # Unbalanced braces, skip
                        break
    
    # Try to parse each candidate, preferring ones with expected keys
    best_match = None
    for candidate in json_candidates:
        try:
            parsed = json.loads(candidate)
            # Prioritize candidates with expected keys
            if any(key in parsed for key in ["response", "grade", "score", "answer", "reasoning"]):
                return parsed
            # Keep the first valid JSON as fallback
            if best_match is None:
                best_match = parsed
        except json.JSONDecodeError:
            continue
    
    return best_match


def _validate_and_normalize_prediction(prediction: str, grading_guidelines: str) -> str:
    """Validate and normalize the prediction based on grading guidelines.
    
    Ensures the prediction matches expected format from grading guidelines.
    Handles various edge cases and normalizes common variations with improved
    robustness for IMO grading scenarios.
    
    Args:
        prediction: The raw prediction string from the model
        grading_guidelines: The grading guidelines to validate against
        
    Returns:
        A normalized prediction string that conforms to expected format
    """
    if not prediction or prediction == "None":
        return "Incorrect"  # Default to Incorrect for empty/None predictions
    
    # Strip whitespace and common punctuation artifacts
    prediction = prediction.strip().strip('"\'.,;:')
    
    # Normalize common variations to standard IMO grading labels
    prediction_lower = prediction.lower()
    
    # Map common variations to standard labels - expanded list with more variations
    correct_variations = [
        'correct', 'correctly', 'full', 'complete', 'right', 'true', 'valid', 
        '7', 'seven', 'full credit', 'full marks', 'solved', 'success', 'pass', 'perfect'
    ]
    almost_variations = [
        'almost', 'nearly', 'nearly complete', 'minor errors', '6', 'six', 
        'mostly correct', 'mostly', 'minor', 'small errors'
    ]
    partial_variations = [
        'partial', 'partially correct', 'some progress', 'partial credit', 
        '1', '2', '3', '4', '5', 'one', 'two', 'three', 'four', 'five', 
        'incomplete', 'some credit', 'progress', 'partially'
    ]
    incorrect_variations = [
        'incorrect', 'wrong', 'false', 'invalid', '0', 'zero', 'none', 'no', 
        'no credit', 'no marks', 'fail', 'failed', 'error', 'n', 'in'
    ]
    
    if prediction_lower in correct_variations:
        return 'Correct'
    elif prediction_lower in almost_variations:
        return 'Almost'
    elif prediction_lower in partial_variations:
        return 'Partial'
    elif prediction_lower in incorrect_variations:
        return 'Incorrect'
    
    # Handle numeric IMO scores (0-7 scale)
    if prediction.isdigit():
        score = int(prediction)
        if 0 <= score <= 7:
            # Map numeric scores to labels for consistency
            if score == 7:
                return 'Correct'
            elif score == 6:
                return 'Almost'
            elif score == 0:
                return 'Incorrect'
            else:
                # For scores 1-5, use numeric representation
                return str(score)
    
    # Check for numeric at the start of the string
    num_match = re.match(r'^(\d+)\b', prediction)
    if num_match:
        score = int(num_match.group(1))
        if 0 <= score <= 7:
            if score == 7:
                return 'Correct'
            elif score == 6:
                return 'Almost'
            elif score == 0:
                return 'Incorrect'
            else:
                return str(score)
    
    # If grading guidelines contain specific labels, try to match them
    if grading_guidelines:
        guideline_lower = grading_guidelines.lower()
        # Check if guidelines use specific label format
        if '(correct)' in guideline_lower and prediction_lower == 'correct':
            return 'Correct'
        if '(almost)' in guideline_lower and prediction_lower == 'almost':
            return 'Almost'
        if '(partial)' in guideline_lower and prediction_lower == 'partial':
            return 'Partial'
        if '(incorrect)' in guideline_lower and prediction_lower == 'incorrect':
            return 'Incorrect'
    
    # Check for label as substring with word boundaries
    for label in ['Correct', 'Almost', 'Partial', 'Incorrect']:
        if re.search(rf'\b{label.lower()}\b', prediction_lower):
            return label
    
    # Default to Incorrect for unrecognized predictions
    return 'Incorrect'


def _extract_confidence_score(text: str) -> float | None:
    """Extract confidence score from text if present.
    
    Looks for patterns like "confidence: 0.85" or "confidence score: 85%"
    Returns a float between 0 and 1, or None if not found.
    """
    # Look for confidence patterns
    patterns = [
        r'confidence[:\s]+(\d+\.?\d*)\s*%?',
        r'confidence score[:\s]+(\d+\.?\d*)\s*%?',
        r'confidence[:\s]+(\d+)%',
    ]
    
    text_lower = text.lower()
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            try:
                value = float(match.group(1))
                # Normalize percentage to 0-1 range
                if value > 1.0:
                    value = value / 100.0
                return max(0.0, min(1.0, value))  # Clamp to [0, 1]
            except ValueError:
                continue
    
    return None


def calibrate_confidence(raw_confidence: float, extraction_method: str) -> float:
    """Calibrate confidence score based on extraction reliability.
    
    Different extraction methods have different reliability levels.
    This function adjusts raw confidence scores to account for method reliability.
    
    Args:
        raw_confidence: The original confidence score from extraction.
        extraction_method: The method used to extract the prediction.
        
    Returns:
        Calibrated confidence score in range [0.0, 1.0].
    """
    # Reliability factors for different extraction methods
    reliability_factors = {
        "json_tags": 1.0,      # Most reliable - structured JSON output
        "fallback": 0.9,       # Still reliable - found valid JSON
        "direct": 0.7,         # Less reliable - direct text extraction
        "label_regex": 0.65,   # Regex for specific labels
        "regex": 0.6,          # Least reliable - generic regex pattern matching
        "none": 0.5,           # No extraction method identified
    }
    
    factor = reliability_factors.get(extraction_method, 0.8)
    calibrated = raw_confidence * factor
    
    # Ensure minimum confidence to avoid overconfidence in low-reliability methods
    if factor < 0.8 and calibrated > 0.8:
        calibrated = 0.8
    
    return max(0.0, min(1.0, calibrated))


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

        instruction = f"""You are an expert grader for the International Mathematical Olympiad (IMO). Your task is to evaluate a student's answer to a mathematical problem with precision and consistency.

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

Please evaluate the student's answer following this structured approach:

### Step 1: Problem Analysis
- Summarize what the problem is asking
- Identify key mathematical concepts and required techniques
- Note any critical assumptions or constraints

### Step 2: Solution Mapping
- Break down the official solution into key milestones
- Identify which milestones are essential vs. optional
- Note common alternative valid approaches

### Step 3: Student Answer Evaluation
- Map the student's work to solution milestones
- Identify correct steps with clear justification
- Flag any errors, gaps, or logical flaws
- Check for partial credit opportunities per guidelines

### Step 4: Grade Determination
- Apply grading guidelines systematically
- Consider: correctness, completeness, and clarity
- Assign the most appropriate grade/score

Respond ONLY in JSON format wrapped in <json> tags with the following exact schema:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis following the 4 steps above...",
    "response": "The final grade/score as specified in the grading guidelines",
    "confidence": 0.85
}}
</json>

CRITICAL INSTRUCTIONS FOR IMO GRADING:
1. The "response" field must contain ONLY the final grade/score value (e.g., "7", "2", "0", "Correct", "Incorrect", "Partial", "Almost", etc.) exactly as specified in the grading guidelines.
2. The "confidence" field is OPTIONAL. If provided, it should be a number between 0 and 1 indicating your confidence in the grade.
3. Do NOT add explanations, reasoning, or extra text in the "response" field.
4. Do NOT use markdown formatting, code blocks, or any other formatting inside the JSON.
5. The JSON must be valid and properly escaped.
6. Wrap your entire JSON response in <json>...</json> tags.

GRADING GUIDELINES INTERPRETATION - READ CAREFULLY:
The grading guidelines describe what the student has achieved in their solution. They typically contain sections marked with grade labels like:
- "(Correct)" or "(Full)" - for complete, correct solutions
- "(Almost)" or "(Nearly complete)" - for solutions with minor errors only
- "(Partial)" - for solutions with some progress but significant gaps
- "(Incorrect)" or "(No credit)" - for solutions with no meaningful progress or completely wrong approaches

GRADE DETERMINATION RULES - FOLLOW STRICTLY:
1. First, check if the student has achieved items listed under "(Correct)" or has a complete correct solution → grade "Correct" or "7"
2. If not correct, check if achieved items under "(Almost)" → grade "Almost" or "6"
3. If not almost, check if achieved items under "(Partial)" → grade "Partial" or the appropriate numeric score (1-5)
4. If none of the above, the grade should be "Incorrect" or "0"

CRITICAL - STRICT GRADING STANDARDS:

**CORRECT (7 points):**
- The solution must be COMPLETE and ESSENTIALLY CORRECT
- All major steps must be justified
- Minor calculation errors or missing trivial justifications are acceptable
- ANY significant logical flaw or missing critical step makes it NOT "Correct"

**ALMOST (6 points):**
- The solution is NEARLY COMPLETE with ONLY MINOR ERRORS
- Examples: small arithmetic mistakes, missing trivial justifications, slight oversights
- The core approach and main arguments must be correct
- Significant gaps or major errors disqualify this grade

**PARTIAL (1-5 points):**
- The student made MEANINGFUL PROGRESS toward the solution
- Must have at least ONE substantive correct step or insight
- Examples: proved a useful lemma, set up the problem correctly with some progress, identified key approach
- Simply restating the problem, making trivial observations, or having only incorrect work = NOT Partial
- The work must contribute meaningfully toward a solution

**INCORRECT (0 points):**
- NO MEANINGFUL PROGRESS toward the solution
- Examples: completely wrong approach, fundamental misconceptions, only trivial observations, no substantive work
- If the student only restates the problem or makes obvious statements → "Incorrect"
- If all claimed results are wrong or unsupported → "Incorrect"
- When in doubt between "Incorrect" and "Partial", choose "Incorrect" unless there's clear meaningful progress

**KEY PRINCIPLE: When uncertain between two grades, choose the LOWER grade. Be conservative in awarding partial credit.**

Pay close attention to the specific point values mentioned in the guidelines (IMO typically uses 0-7 scale)."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        confidence = None
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
                else:
                    # If no recognized field, use the whole JSON as string
                    prediction = json.dumps(last_json)
                
                # Extract confidence if available
                if "confidence" in last_json:
                    try:
                        confidence = float(last_json["confidence"])
                    except (ValueError, TypeError):
                        pass
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
                    else:
                        prediction = json.dumps(fallback)
                    
                    # Extract confidence if available
                    if "confidence" in fallback:
                        try:
                            confidence = float(fallback["confidence"])
                        except (ValueError, TypeError):
                            pass
                    
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
            
            # Fourth try: Look for IMO grading labels in the text if still no prediction
            if prediction == "None" or prediction == json.dumps({}):
                # Search for common IMO grading labels in the text - expanded patterns
                label_patterns = [
                    (r'\b(Correct|Full|Complete|Full credit|Full marks)\b', 'Correct'),
                    (r'\b(Almost|Nearly complete|Minor errors|Mostly correct)\b', 'Almost'),
                    (r'\b(Partial|Partially correct|Some progress|Partial credit|Incomplete)\b', 'Partial'),
                    (r'\b(Incorrect|Wrong|No credit|No marks|False|Invalid|Fail|Failed)\b', 'Incorrect'),
                ]
                for pattern, label in label_patterns:
                    match = re.search(pattern, last_text, re.IGNORECASE)
                    if match:
                        prediction = label
                        extraction_method = "label_regex"
                        self.log_fn(f"Used label regex extraction: {prediction}")
                        break
            
            # Try to extract confidence from reasoning text if not in JSON
            if confidence is None:
                confidence = _extract_confidence_score(last_text)
            
            # Calibrate confidence based on extraction method reliability
            if confidence is not None:
                raw_confidence = confidence
                confidence = calibrate_confidence(raw_confidence, extraction_method)
                if raw_confidence != confidence:
                    self.log_fn(f"Calibrated confidence from {raw_confidence:.2f} to {confidence:.2f} (method: {extraction_method})")
            
            # Validate and normalize the prediction
            original_prediction = prediction
            prediction = _validate_and_normalize_prediction(prediction, grading_guidelines)
            
            if original_prediction != prediction:
                self.log_fn(f"Normalized prediction from '{original_prediction}' to '{prediction}'")
            
            # Additional validation: ensure prediction matches expected format from guidelines
            # If guidelines use specific labels like "Correct", "Almost", "Partial", "Incorrect"
            # and our prediction is a numeric score, try to map it appropriately
            if grading_guidelines and prediction.isdigit():
                score = int(prediction)
                # Check if guidelines suggest using label-based grading
                has_label_format = any(label in grading_guidelines for label in ['(Correct)', '(Almost)', '(Partial)', '(Incorrect)'])
                if has_label_format:
                    # Map numeric scores to labels
                    if score == 7:
                        prediction = 'Correct'
                        self.log_fn(f"Mapped score 7 to 'Correct' based on guideline format")
                    elif score == 6:
                        prediction = 'Almost'
                        self.log_fn(f"Mapped score 6 to 'Almost' based on guideline format")
                    elif score == 0:
                        prediction = 'Incorrect'
                        self.log_fn(f"Mapped score 0 to 'Incorrect' based on guideline format")
                    elif 1 <= score <= 5:
                        prediction = 'Partial'
                        self.log_fn(f"Mapped score {score} to 'Partial' based on guideline format")
            
            # Also check if guidelines use numeric format but prediction is a label
            if grading_guidelines and prediction in ['Correct', 'Almost', 'Partial', 'Incorrect']:
                has_numeric_format = any(str(i) in grading_guidelines for i in range(8))
                if has_numeric_format and not any(label in grading_guidelines for label in ['(Correct)', '(Almost)', '(Partial)', '(Incorrect)']):
                    # Map labels back to numeric scores
                    label_to_score = {
                        'Correct': '7',
                        'Almost': '6',
                        'Partial': '3',  # Middle of 1-5 range
                        'Incorrect': '0'
                    }
                    original_prediction = prediction
                    prediction = label_to_score.get(prediction, prediction)
                    self.log_fn(f"Mapped label '{original_prediction}' to score '{prediction}' based on numeric guideline format")
            
            # Log confidence if extracted
            if confidence is not None:
                self.log_fn(f"Extraction method: {extraction_method}, final prediction: {prediction}, confidence: {confidence:.2f}")
            else:
                self.log_fn(f"Extraction method: {extraction_method}, final prediction: {prediction}")
            
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Last resort: try to find any numeric or keyword answer in the text
            try:
                # Look for IMO scores (0-7) - be more specific to avoid false positives
                score_match = re.search(r'["\']?([0-7])["\']?\s*$', last_text.strip().split('\n')[-1])
                if not score_match:
                    score_match = re.search(r'grade.*?["\']?([0-7])["\']?', last_text, re.IGNORECASE)
                if score_match:
                    prediction = score_match.group(1)
                    self.log_fn(f"Used regex extraction for score: {prediction}")
                else:
                    # Try to find grading labels as last resort - expanded patterns
                    label_patterns = [
                        (r'\b(Correct|Full|Complete|Full credit|Full marks)\b', 'Correct'),
                        (r'\b(Almost|Nearly complete|Minor errors|Mostly correct)\b', 'Almost'),
                        (r'\b(Partial|Partially correct|Some progress|Partial credit|Incomplete)\b', 'Partial'),
                        (r'\b(Incorrect|Wrong|No credit|No marks|False|Invalid|Fail|Failed)\b', 'Incorrect'),
                    ]
                    for pattern, label in label_patterns:
                        match = re.search(pattern, last_text, re.IGNORECASE)
                        if match:
                            prediction = label
                            self.log_fn(f"Used emergency label extraction: {prediction}")
                            break
            except Exception:
                pass
        
        # Final validation - ensure we have a valid prediction
        prediction = _validate_and_normalize_prediction(str(prediction), grading_guidelines)

        return str(prediction), msg_history
