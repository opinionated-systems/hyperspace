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
from typing import Any

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles nested JSON objects within the tags and common JSON errors.
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
            # Try to clean and fix common JSON issues
            cleaned = inner
            
            # Remove trailing commas before closing braces/brackets
            cleaned = re.sub(r',\s*}', '}', cleaned)
            cleaned = re.sub(r',\s*]', ']', cleaned)
            
            # Fix single quotes to double quotes (but not within strings)
            # This is a simple heuristic - replace 'key': with "key":
            cleaned = re.sub(r"'([^']+)':\s*", r'"\1": ', cleaned)
            
            # Try to parse cleaned JSON
            try:
                results.append(json.loads(cleaned))
                continue
            except json.JSONDecodeError:
                pass
            
            # Try to extract JSON object pattern from within the content
            try:
                # Look for JSON object pattern with nested brace handling
                json_match = re.search(r'\{[\s\S]*\}', inner)
                if json_match:
                    json_str = json_match.group()
                    # Clean the extracted JSON too
                    json_str = re.sub(r',\s*}', '}', json_str)
                    json_str = re.sub(r',\s*]', ']', json_str)
                    results.append(json.loads(json_str))
            except (json.JSONDecodeError, AttributeError):
                continue
    return results or None


def _extract_json_from_markdown(text: str) -> dict | None:
    """Extract JSON from markdown code blocks (```json ... ```).
    
    Also handles common JSON errors like trailing commas.
    """
    pattern = r'```json\s*(.*?)\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        cleaned = match.strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to clean and fix common JSON issues
            # Remove trailing commas before closing braces/brackets
            cleaned = re.sub(r',\s*}', '}', cleaned)
            cleaned = re.sub(r',\s*]', ']', cleaned)
            
            # Fix single quotes to double quotes for keys
            cleaned = re.sub(r"'([^']+)':\s*", r'"\1": ', cleaned)
            
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                continue
    return None


def _validate_grade(prediction: str, grading_guidelines: str) -> tuple[str, bool]:
    """Validate that the extracted grade is reasonable.
    
    Enhanced validation with support for:
    - IMO 0-7 point scale
    - Partial credit notation
    - Fractional grades (e.g., 3/7)
    - Descriptive evaluations
    - Range-based grades (e.g., "5-6", "4 to 5")
    - Common grading phrases and variations
    
    Returns:
        (validated_grade, is_valid)
    """
    if not prediction or prediction == "None":
        return "None", False
    
    pred_clean = prediction.strip()
    pred_lower = pred_clean.lower()
    
    # First, try to extract a single digit 0-7 that's clearly a grade
    # Look for patterns like "Grade: 5" or "Score: 3" or just "5"
    explicit_grade_patterns = [
        r'(?:grade|score|mark|points?)\s*:?\s*["\']?([0-7])["\']?\b',
        r'(?:award|assign|give)\s*:?\s*["\']?([0-7])["\']?\b',
        r'\b([0-7])\s*(?:points?|out\s+of\s+7)\b',
        r'(?:^|\s)["\']?([0-7])["\']?(?:\s*$|\s+points?)',
    ]
    for pattern in explicit_grade_patterns:
        match = re.search(pattern, pred_lower)
        if match:
            return match.group(1), True
    
    # Check for range-based grades like "5-6" or "4 to 5" - take the average
    range_match = re.search(r'\b([0-7])\s*(?:-|to|–|—|~|\.\.\.)\s*([0-7])\b', pred_clean)
    if range_match:
        low, high = int(range_match.group(1)), int(range_match.group(2))
        avg = round((low + high) / 2)
        return str(avg), True
    
    # Check for fractional grades like "3/7" or "5 / 7"
    fractional_match = re.search(r'\b([0-7])\s*/\s*7\b', pred_clean)
    if fractional_match:
        return fractional_match.group(1), True
    
    # Check for numeric grades (0-7 for IMO problems) - standalone digit
    numeric_match = re.search(r'\b([0-7])\b', pred_clean)
    if numeric_match:
        return numeric_match.group(1), True
    
    # Check for written-out numbers
    written_numbers = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3',
        'four': '4', 'five': '5', 'six': '6', 'seven': '7'
    }
    for word, digit in written_numbers.items():
        if re.search(rf'\b{word}\b', pred_lower):
            return digit, True
    
    # Check for partial credit patterns with more variations
    partial_patterns = [
        (r'partial\s*(?:credit)?\s*:?\s*([0-7])', 1),
        (r'partial\s*([0-7])\s*(?:points?)?', 1),
        (r'([0-7])\s*(?:points?)?\s*partial', 1),
        (r'partially\s*(?:correct)?\s*:?\s*([0-7])', 1),
        (r'(?:some|limited|minimal)\s*(?:progress|credit)\s*:?\s*([0-7])', 1),
        (r'(?:incomplete|unfinished)\s*(?:solution)?\s*:?\s*([0-7])', 1),
    ]
    for pattern, group in partial_patterns:
        partial_match = re.search(pattern, pred_lower)
        if partial_match:
            grade = partial_match.group(group)
            return grade, True
    
    # Check for full credit patterns
    full_patterns = [
        r'\bfull\s*(?:credit|points?|score|marks?)?\b',
        r'\bcomplete\s*(?:solution|answer|credit|proof)?\b',
        r'\ball\s*(?:points?|credit|marks?)?\b',
        r'\bperfect\s*(?:score|solution|answer)?\b',
        r'\bcorrect\s*(?:solution|answer|proof|approach)?\b',
        r'\bfully\s*(?:correct|solved)?\b',
        r'\bsolved\s*(?:completely|correctly|fully)?\b',
        r'\bvalid\s*(?:solution|proof|approach)?\b',
        r'\bexcellent\s*(?:solution|work|answer)?\b',
    ]
    for pattern in full_patterns:
        if re.search(pattern, pred_lower):
            return "7", True
    
    # Check for zero/no credit patterns
    zero_patterns = [
        r'\bno\s*(?:credit|points?|score|marks?|progress)?\b',
        r'\bzero\s*(?:credit|points?|score|marks?)?\b',
        r'\b0\s*(?:points?|credit|score|marks?)?\b',
        r'\bincorrect\s*(?:solution|answer|approach|proof)?\b',
        r'\bwrong\s*(?:solution|answer|approach|proof)?\b',
        r'\bno\s*solution\b',
        r'\bempty\s*(?:answer|solution|response)?\b',
        r'\bno\s*progress\b',
        r'\bno\s*meaningful\s*(?:progress|work)?\b',
        r'\bblank\s*(?:answer|response)?\b',
        r'\bno\s*attempt\b',
    ]
    for pattern in zero_patterns:
        if re.search(pattern, pred_lower):
            return "0", True
    
    # Check for specific grade level patterns
    grade_patterns = {
        '6': [r'\bminor\s*(?:gap|issue|error|flaw)', r'\bsmall\s*(?:error|mistake)'],
        '5': [r'\bsignificant\s*(?:gap|issue)', r'\bmajor\s*step\s*correct'],
        '4': [r'\bpartial\s*solution', r'\bsome\s*correct\s*steps'],
        '3': [r'\bmeaningful\s*progress', r'\bsome\s*progress'],
        '2': [r'\blimited\s*progress', r'\bfew\s*correct\s*ideas'],
        '1': [r'\bminimal\s*progress', r'\brelevant\s*observation'],
    }
    for grade, patterns in grade_patterns.items():
        for pattern in patterns:
            if re.search(pattern, pred_lower):
                return grade, True
    
    # Check for other valid grade keywords
    valid_keywords = ['correct', 'partial', 'n/a', 'not applicable', 'incomplete']
    for keyword in valid_keywords:
        if keyword in pred_lower:
            return pred_clean, True
    
    # If prediction is very short (1-2 chars), it might be a grade
    if len(pred_clean) <= 2 and pred_clean.isdigit():
        grade_val = int(pred_clean)
        if 0 <= grade_val <= 7:
            return pred_clean, True
    
    # If no clear grade found, mark as invalid but return cleaned prediction
    return pred_clean, False


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced reasoning and validation."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _build_prompt(self, inputs: dict) -> str:
        """Build a structured prompt for IMO grading with chain-of-thought reasoning."""
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        return f"""You are an expert IMO (International Mathematical Olympiad) grader with extensive experience in evaluating mathematical proofs and solutions.

Your task is to evaluate a student's solution to a mathematical problem with precision and consistency.

## Problem Domain
{domain}

## Problem Statement
{problem}

## Official Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## IMO Grading Scale Reference
- 7 points: Complete, correct solution with clear reasoning
- 6 points: Correct solution with minor gaps or presentation issues
- 5 points: Correct approach with significant gaps but major steps correct
- 4 points: Partial solution with some correct key steps
- 3 points: Some meaningful progress toward solution
- 2 points: Limited progress with some correct ideas
- 1 point: Minimal progress or relevant observations
- 0 points: No meaningful progress or completely incorrect

## Instructions

1. **Analyze the student's approach**: Identify the method/technique used and compare it to the official solution.
2. **Check each step**: Verify correctness of calculations, logic, and mathematical reasoning.
3. **Identify gaps/errors**: Note any missing steps, logical flaws, or computational errors.
4. **Consider alternative approaches**: Valid alternative methods should receive full credit if correct.
5. **Apply grading guidelines**: Use the specific rubric provided in the grading guidelines.
6. **Determine final grade**: Assign a numeric score from 0-7 based on the IMO scale above.

## Response Format

You MUST respond in valid JSON format wrapped in <json> tags. The JSON must be valid and parseable.

<json>
{{
    "reasoning": "Your detailed step-by-step analysis of the student's answer, including: (1) approach identification, (2) correctness verification, (3) gaps/errors found, (4) comparison to official solution, (5) justification for the assigned grade",
    "response": "X"
}}
</json>

Where X is EXACTLY ONE digit: 0, 1, 2, 3, 4, 5, 6, or 7.

CRITICAL RULES:
- The "response" field MUST contain ONLY a single digit from 0-7
- Do NOT include any text, explanations, or ranges in the response field
- Do NOT use quotes around the digit in the response field
- Do NOT write things like "7 points" or "grade: 5" - just the digit
- Put ALL analysis and justification in the "reasoning" field
- Ensure the JSON is valid (no trailing commas, proper quotes)

Example of CORRECT response:
<json>
{{
    "reasoning": "The student correctly identified the approach but made a calculation error in step 3. The overall structure is sound but the final answer is incorrect due to this error.",
    "response": "5"
}}
</json>

Example of INCORRECT response:
<json>
{{
    "reasoning": "Good solution",
    "response": "5 points"  // WRONG - should be just "5"
}}
</json>"""

    def _extract_prediction(self, msg_history: list[dict]) -> tuple[str, str]:
        """Extract prediction and reasoning from message history.
        
        Enhanced extraction with multiple fallback strategies and better
        handling of various response formats.
        
        Returns:
            (prediction, reasoning)
        """
        prediction = "None"
        reasoning = ""
        
        try:
            # Combine all assistant messages for extraction
            full_text = ""
            for msg in msg_history:
                if msg.get("role") == "assistant":
                    full_text += msg.get("text", "") + "\n"
            
            if not full_text and msg_history:
                full_text = msg_history[-1].get("text", "")
            
            if not full_text:
                return "None", ""
            
            # Strategy 1: Try <json> tags first (most reliable)
            extracted = _extract_jsons(full_text)
            if extracted:
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = str(last_json["response"]).strip()
                if "reasoning" in last_json:
                    reasoning = str(last_json["reasoning"])
                if prediction != "None":
                    return prediction, reasoning
            
            # Strategy 2: Try markdown code blocks
            md_json = _extract_json_from_markdown(full_text)
            if md_json:
                if "response" in md_json:
                    prediction = str(md_json["response"]).strip()
                if "reasoning" in md_json:
                    reasoning = str(md_json["reasoning"])
                if prediction != "None":
                    return prediction, reasoning
            
            # Strategy 3: Try to find any JSON-like structure with response field
            # Use a more flexible pattern that handles nested braces
            json_pattern = r'\{[\s\S]*?"response"[\s\S]*?\}'
            json_matches = re.findall(json_pattern, full_text)
            for json_match in reversed(json_matches):  # Try from the end
                try:
                    # Clean up common issues
                    cleaned = json_match.replace('\n', ' ').replace('\t', ' ')
                    # Remove trailing commas before closing braces
                    cleaned = re.sub(r',\s*}', '}', cleaned)
                    fallback = json.loads(cleaned)
                    pred = str(fallback.get("response", "None")).strip()
                    if pred and pred != "None":
                        prediction = pred
                        reasoning = str(fallback.get("reasoning", ""))
                        return prediction, reasoning
                except json.JSONDecodeError:
                    continue
            
            # Strategy 4: Look for explicit grade patterns in the last 1000 chars
            # (grades are often at the end of the response)
            last_part = full_text[-1000:] if len(full_text) > 1000 else full_text
            
            # Pattern: "Grade: X" or "Score: X" or "Final grade: X"
            explicit_patterns = [
                r'(?:final\s+)?(?:grade|score|mark)\s*[:=]\s*["\']?([0-7])["\']?\b',
                r'\bgrade\s+(?:is|of|equals?)\s*[:=]?\s*["\']?([0-7])["\']?\b',
                r'\b(?:award|assign|give)\s*[:=]?\s*["\']?([0-7])["\']?\s*(?:points?)?\b',
                r'\b([0-7])\s*(?:points?|out\s+of\s+7)\b',
                r'["\']?response["\']?\s*[:=]\s*["\']?([0-7])["\']?\b',
            ]
            for pattern in explicit_patterns:
                grade_match = re.search(pattern, last_part, re.IGNORECASE)
                if grade_match:
                    prediction = grade_match.group(1)
                    return prediction, reasoning
            
            # Strategy 5: Look for standalone digits 0-7 in the last 200 chars
            # (often the grade is the very last thing)
            very_last = full_text[-200:] if len(full_text) > 200 else full_text
            standalone_match = re.search(r'\b([0-7])\b', very_last)
            if standalone_match:
                prediction = standalone_match.group(1)
                return prediction, reasoning
            
            # Strategy 6: Look for grade in reasoning section if present
            reasoning_match = re.search(r'["\']?reasoning["\']?\s*[:=]\s*["\']?([^"\']+)', full_text, re.IGNORECASE)
            if reasoning_match:
                reasoning_text = reasoning_match.group(1)
                grade_in_reasoning = re.search(r'\b([0-7])\b', reasoning_text)
                if grade_in_reasoning:
                    prediction = grade_in_reasoning.group(1)
                    return prediction, reasoning
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
        
        return prediction, reasoning

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with enhanced reasoning and validation.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_prompt(inputs)
        grading_guidelines = inputs.get("grading_guidelines", "")

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction with enhanced extraction
        prediction, reasoning = self._extract_prediction(msg_history)
        
        # Validate the grade
        validated_grade, is_valid = _validate_grade(prediction, grading_guidelines)
        
        # Log the reasoning and validation result
        if reasoning:
            self.log_fn(f"Reasoning: {reasoning[:200]}...")
        self.log_fn(f"Extracted grade: {prediction}, Validated: {validated_grade}, Is valid: {is_valid}")
        
        # If grade is invalid, try multiple fallback strategies
        if not is_valid:
            # Strategy 1: Try to find any numeric grade 0-7 in the full response
            if response:
                # Look for explicit grade mentions first
                explicit_patterns = [
                    r'(?:grade|score|mark|final)\s*[:=]\s*([0-7])\b',
                    r'(?:award|assign)\s+([0-7])\s*(?:points?)?',
                    r'\b([0-7])\s*(?:points?|out\s+of\s+7)\b',
                ]
                for pattern in explicit_patterns:
                    match = re.search(pattern, response, re.IGNORECASE)
                    if match:
                        validated_grade = match.group(1)
                        is_valid = True
                        self.log_fn(f"Fallback 1 (explicit) found grade: {validated_grade}")
                        break
                
                # If still not valid, look for any standalone digit 0-7
                if not is_valid:
                    numeric_match = re.search(r'\b([0-7])\b', response)
                    if numeric_match:
                        validated_grade = numeric_match.group(1)
                        is_valid = True
                        self.log_fn(f"Fallback 1 (numeric) found grade: {validated_grade}")
            
            # Strategy 2: Look for grade at the very end of the response
            if not is_valid and response:
                end_patterns = [
                    r'(?:grade|score|mark)\s*[:=]?\s*([0-7])\s*$',
                    r'\b([0-7])\s*$',
                ]
                for pattern in end_patterns:
                    end_match = re.search(pattern, response.strip(), re.IGNORECASE | re.MULTILINE)
                    if end_match:
                        validated_grade = end_match.group(1)
                        is_valid = True
                        self.log_fn(f"Fallback 2 found grade: {validated_grade}")
                        break
            
            # Strategy 3: Check for common grade indicators in the last 1000 chars
            if not is_valid and response:
                last_part = response[-1000:] if len(response) > 1000 else response
                indicators = {
                    '7': ['full credit', 'complete solution', 'perfect', 'all points', 'fully correct', 'excellent'],
                    '6': ['minor gap', 'minor issue', 'small error', 'almost complete'],
                    '5': ['significant gap', 'major step correct', 'mostly correct'],
                    '4': ['partial solution', 'some correct steps', 'half correct'],
                    '3': ['meaningful progress', 'some progress', 'on the right track'],
                    '2': ['limited progress', 'few correct ideas', 'some understanding'],
                    '1': ['minimal progress', 'relevant observation', 'little progress'],
                    '0': ['no credit', 'zero', 'incorrect', 'wrong', 'no solution', 'empty', 'no progress', 'blank'],
                }
                for grade, keywords in indicators.items():
                    if any(kw in last_part.lower() for kw in keywords):
                        validated_grade = grade
                        is_valid = True
                        self.log_fn(f"Fallback 3 found grade: {validated_grade}")
                        break
            
            # Strategy 4: Check for written-out numbers
            if not is_valid and response:
                written_numbers = {
                    'zero': '0', 'one': '1', 'two': '2', 'three': '3',
                    'four': '4', 'five': '5', 'six': '6', 'seven': '7'
                }
                for word, digit in written_numbers.items():
                    if re.search(rf'\b{word}\b', response.lower()):
                        validated_grade = digit
                        is_valid = True
                        self.log_fn(f"Fallback 4 found grade: {validated_grade}")
                        break
        
        # Final validation: ensure grade is within valid range and is a clean digit
        if is_valid:
            try:
                grade_val = int(validated_grade)
                if 0 <= grade_val <= 7:
                    validated_grade = str(grade_val)  # Normalize to clean string
                    is_valid = True
                else:
                    is_valid = False
            except (ValueError, TypeError):
                is_valid = False
        
        if not is_valid:
            self.log_fn(f"Could not extract valid grade, returning 'None'")
            validated_grade = "None"

        return str(validated_grade), msg_history
