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
    if not prediction:
        return prediction
    
    # Strip whitespace and common punctuation artifacts
    prediction = prediction.strip().strip('"\'.,;:')
    
    # Determine if guidelines use label-based or numeric grading
    uses_labels = False
    if grading_guidelines:
        if any(label in grading_guidelines for label in ['(Correct)', '(Almost)', '(Partial)', '(Incorrect)']):
            uses_labels = True
        elif any(label in grading_guidelines for label in ['Correct', 'Almost', 'Partial', 'Incorrect']):
            uses_labels = True
    
    # Normalize common variations to standard IMO grading labels
    prediction_lower = prediction.lower()
    
    # Map common variations to standard labels - be strict about matching
    if prediction_lower in ['correct', 'full', 'complete', 'right', 'true', 'valid', 'correctly']:
        return 'Correct'
    elif prediction_lower in ['almost', 'nearly', 'nearly complete', 'minor errors', 'almost correct']:
        return 'Almost'
    elif prediction_lower in ['partial', 'partially correct', 'some progress', 'partial credit', 'partially']:
        return 'Partial'
    elif prediction_lower in ['incorrect', 'wrong', 'false', 'invalid', 'none', 'no', 'incorrectly']:
        return 'Incorrect'
    
    # Handle numeric IMO scores (0-7 scale)
    if prediction.isdigit():
        score = int(prediction)
        if 0 <= score <= 7:
            # If guidelines use labels, map numeric to labels
            if uses_labels:
                if score == 7:
                    return 'Correct'
                elif score == 6:
                    return 'Almost'
                elif score == 0:
                    return 'Incorrect'
                else:
                    # Scores 1-5 map to Partial
                    return 'Partial'
            else:
                # For numeric grading, return the score
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
    
    return prediction


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

        # Determine the grading format from guidelines
        grading_format = "numeric"  # default
        if grading_guidelines:
            if any(label in grading_guidelines for label in ['(Correct)', '(Almost)', '(Partial)', '(Incorrect)']):
                grading_format = "labels"
            elif any(label in grading_guidelines for label in ['Correct', 'Almost', 'Partial', 'Incorrect']):
                grading_format = "labels"
        
        # Build format-specific instructions
        if grading_format == "labels":
            format_instructions = """CRITICAL: The grading guidelines use LABEL-BASED grading (Correct, Almost, Partial, Incorrect).
- You MUST output one of these EXACT labels: "Correct", "Almost", "Partial", or "Incorrect"
- Do NOT use numeric scores (0, 1, 2, 3, 4, 5, 6, 7) when the guidelines use labels
- The "response" field must contain ONLY one of: "Correct", "Almost", "Partial", or "Incorrect"
- Match the highest level achieved by the student based on the guidelines"""
            expected_response_examples = '"Correct", "Almost", "Partial", or "Incorrect"'
        else:
            format_instructions = """CRITICAL: The grading guidelines use NUMERIC scoring (IMO 0-7 scale).
- You MUST output a numeric score from 0 to 7
- The "response" field must contain ONLY a number: 0, 1, 2, 3, 4, 5, 6, or 7
- Match the highest score achieved by the student based on the guidelines"""
            expected_response_examples = '"0", "1", "2", "3", "4", "5", "6", or "7"'

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
1. The "response" field must contain ONLY the final grade/score value ({expected_response_examples}) exactly as specified in the grading guidelines.
2. The "confidence" field is OPTIONAL. If provided, it should be a number between 0 and 1 indicating your confidence in the grade.
3. Do NOT add explanations, reasoning, or extra text in the "response" field.
4. Do NOT use markdown formatting, code blocks, or any other formatting inside the JSON.
5. The JSON must be valid and properly escaped.
6. Wrap your entire JSON response in <json>...</json> tags.

{format_instructions}

GRADING GUIDELINES INTERPRETATION - READ CAREFULLY:

The grading guidelines describe what the student has achieved in their solution. The guidelines typically have sections marked with grade labels like (Correct), (Almost), (Partial), or (Incorrect).

**IMPORTANT: How to Determine the Correct Grade:**

1. **"Correct"** - Award ONLY when:
   - The student has a complete, correct solution with all necessary steps
   - The solution matches the official solution or a valid alternative approach
   - There are no significant errors or gaps in reasoning
   - The student has achieved items listed under "(Correct)" in the guidelines

2. **"Almost"** - Award when:
   - The student has made substantial progress but the solution is incomplete or has minor errors
   - The student has achieved items listed under "(Almost)" in the guidelines
   - The solution is nearly correct but missing some details or has small gaps
   - DO NOT award "Correct" if the solution has any significant gaps or errors

3. **"Partial"** - Award when:
   - The student has made some meaningful progress but the solution is incomplete
   - The student has achieved items listed under "(Partial)" in the guidelines
   - The student has shown understanding of some key concepts but not solved the problem
   - Be CONSERVATIVE with "Partial" - only award if there's genuine mathematical progress

4. **"Incorrect"** - Award when:
   - The student has not made meaningful progress toward the solution
   - The answer is wrong or contains fundamental errors
   - The student has not achieved items listed under any higher grade level

**CRITICAL RULE:**
- The final grade is determined by the HIGHEST level achieved by the student
- BUT be CONSERVATIVE - do not upgrade a grade unless the student clearly meets the criteria
- When in doubt between two grades, choose the LOWER grade
- "Correct" should be awarded ONLY for truly complete and correct solutions
- "Almost" is for solutions that are very close to correct but have minor issues
- "Partial" is for solutions with some correct elements but significant gaps

Pay close attention to the specific format used in the guidelines and match it exactly."""

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
                # Search for common IMO grading labels in the text
                label_patterns = [
                    (r'\b(Correct|Full|Complete)\b', 'Correct'),
                    (r'\b(Almost|Nearly complete)\b', 'Almost'),
                    (r'\b(Partial|Partially correct)\b', 'Partial'),
                    (r'\b(Incorrect|Wrong|No credit)\b', 'Incorrect'),
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
            
            # Final validation: if guidelines use labels but prediction is numeric, force mapping
            if grading_guidelines:
                has_label_format = any(label in grading_guidelines for label in ['(Correct)', '(Almost)', '(Partial)', '(Incorrect)'])
                if has_label_format and prediction.isdigit():
                    score = int(prediction)
                    if score == 7:
                        prediction = 'Correct'
                    elif score == 6:
                        prediction = 'Almost'
                    elif score == 0:
                        prediction = 'Incorrect'
                    else:
                        prediction = 'Partial'
                    self.log_fn(f"Final mapping: numeric score {score} -> '{prediction}' for label-based guidelines")
            
            # Log confidence if extracted
            if confidence is not None:
                self.log_fn(f"Extraction method: {extraction_method}, final prediction: {prediction}, confidence: {confidence:.2f}")
            else:
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
                else:
                    # Try to find grading labels as last resort
                    label_patterns = [
                        (r'\b(Correct|Full|Complete)\b', 'Correct'),
                        (r'\b(Almost|Nearly complete)\b', 'Almost'),
                        (r'\b(Partial|Partially correct)\b', 'Partial'),
                        (r'\b(Incorrect|Wrong|No credit)\b', 'Incorrect'),
                    ]
                    for pattern, label in label_patterns:
                        match = re.search(pattern, last_text, re.IGNORECASE)
                        if match:
                            prediction = label
                            self.log_fn(f"Used emergency label extraction: {prediction}")
                            break
            except Exception:
                pass

        return str(prediction), msg_history
