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

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles nested JSON objects within the tags and common formatting issues.
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
            # Try multiple recovery strategies
            
            # Strategy 1: Look for JSON object pattern
            try:
                json_match = re.search(r'\{[\s\S]*\}', inner)
                if json_match:
                    results.append(json.loads(json_match.group()))
                    continue
            except (json.JSONDecodeError, AttributeError):
                pass
            
            # Strategy 2: Try to fix common JSON issues
            try:
                # Replace single quotes with double quotes (but be careful with apostrophes in text)
                fixed = inner
                # First, try to handle the case where keys use single quotes
                fixed = re.sub(r"'([a-zA-Z_][a-zA-Z0-9_]*)'\s*:", r'"\1":', fixed)
                # Fix trailing commas
                fixed = re.sub(r',\s*}', '}', fixed)
                fixed = re.sub(r',\s*]', ']', fixed)
                # Try parsing again
                results.append(json.loads(fixed))
                continue
            except json.JSONDecodeError:
                pass
            
            # Strategy 3: Extract just the response field if present (handles malformed JSON)
            try:
                # Look for response field with various quote styles
                response_match = re.search(r'["\']?response["\']?\s*:\s*["\']?([0-7])["\']?', inner)
                if response_match:
                    grade = response_match.group(1)
                    # Try to extract reasoning too
                    reasoning_match = re.search(r'["\']?reasoning["\']?\s*:\s*["\']([^"\']*)["\']', inner)
                    reasoning = reasoning_match.group(1) if reasoning_match else ""
                    results.append({"response": grade, "reasoning": reasoning})
                    continue
            except (AttributeError, IndexError):
                pass
            
            continue
    return results or None


def _extract_json_from_markdown(text: str) -> dict | None:
    """Extract JSON from markdown code blocks (```json ... ``` or ``` ... ```)."""
    # Try json-specific blocks first
    pattern = r'```json\s*(.*?)\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    # Also try generic code blocks that might contain JSON
    generic_pattern = r'```\s*(\{[\s\S]*?\})\s*```'
    matches.extend(re.findall(generic_pattern, text, re.DOTALL))
    
    for match in matches:
        match = match.strip()
        # Try parsing as-is first
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            pass
        
        # Try fixing common issues
        try:
            # Replace single quotes with double quotes (but be careful with apostrophes in text)
            fixed = match
            # First, try to handle the case where keys use single quotes
            fixed = re.sub(r"'([a-zA-Z_][a-zA-Z0-9_]*)'\s*:", r'"\1":', fixed)
            # Fix trailing commas
            fixed = re.sub(r',\s*}', '}', fixed)
            fixed = re.sub(r',\s*]', ']', fixed)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        
        # Try to extract just the response field
        try:
            # Look for response field with various quote styles
            response_match = re.search(r'["\']?response["\']?\s*:\s*["\']?([0-7])["\']?', match)
            if response_match:
                grade = response_match.group(1)
                reasoning_match = re.search(r'["\']?reasoning["\']?\s*:\s*["\']([^"\']*)["\']', match)
                reasoning = reasoning_match.group(1) if reasoning_match else ""
                return {"response": grade, "reasoning": reasoning}
        except (AttributeError, IndexError):
            pass
        
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
    - Verbal descriptions mapped to numeric grades
    
    Returns:
        (validated_grade, is_valid)
    """
    if not prediction or prediction == "None":
        return "None", False
    
    pred_clean = prediction.strip()
    pred_lower = pred_clean.lower()
    
    # First, check if the entire response is just a single digit (most common case)
    if pred_clean.isdigit() and len(pred_clean) == 1:
        grade_val = int(pred_clean)
        if 0 <= grade_val <= 7:
            return pred_clean, True
    
    # Clean the prediction - extract only digits 0-7
    digits_only = re.sub(r'[^0-7]', '', pred_clean)
    if len(digits_only) == 1:
        return digits_only, True
    
    # Check for explicit "Grade: X" or "Score: X" patterns (strong indicators)
    explicit_patterns = [
        r'(?:grade|score|mark|points?)\s*[:=]\s*([0-7])\b',
        r'(?:grade|score|mark|points?)\s+(?:is|of|equals?)\s*[:=]?\s*([0-7])\b',
        r'(?:award|assign|give)\s*[:=]?\s*([0-7])\s*(?:points?)?\b',
        r'\bfinal\s+(?:grade|score|mark)\s*[:=]?\s*([0-7])\b',
        r'\b(?:grade|score)\s*[:=]?\s*([0-7])\s*(?:out\s+of\s+7)?\b',
        r'["\']?response["\']?\s*:\s*["\']?([0-7])["\']?',
    ]
    for pattern in explicit_patterns:
        match = re.search(pattern, pred_lower)
        if match:
            return match.group(1), True
    
    # Check for range-based grades like "5-6" or "4 to 5" - take the average
    range_match = re.search(r'\b([0-7])\s*(?:-|to|–|—)\s*([0-7])\b', pred_clean)
    if range_match:
        low, high = int(range_match.group(1)), int(range_match.group(2))
        avg = round((low + high) / 2)
        return str(avg), True
    
    # Check for fractional grades like "3/7" or "5 / 7"
    fractional_match = re.search(r'\b([0-7])\s*/\s*7\b', pred_clean)
    if fractional_match:
        return fractional_match.group(1), True
    
    # Check for numeric grades (0-7 for IMO problems) - word boundary check
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
        r'partial\s*(?:credit)?\s*:?\s*([0-7])',
        r'partial\s*([0-7])\s*(?:points?)?',
        r'([0-7])\s*(?:points?)?\s*partial',
        r'partially\s*(?:correct)?\s*:?\s*([0-7])',
        r'some\s*(?:credit|points?)\s*:?\s*([0-7])',
        r'incomplete\s*(?:solution)?\s*:?\s*([0-7])',
    ]
    for pattern in partial_patterns:
        partial_match = re.search(pattern, pred_lower)
        if partial_match:
            return partial_match.group(1), True
    
    # Check for full credit patterns (7 points)
    full_patterns = [
        r'\bfull\s*(?:credit|points?|score|marks?)?\b',
        r'\bcomplete\s*(?:solution|answer|credit)?\b',
        r'\ball\s*(?:points?|credit|marks?|score)?\b',
        r'\bperfect\s*(?:score|solution|answer)?\b',
        r'\bcorrect\s*(?:solution|answer|proof)?\b',
        r'\bfully\s*(?:correct|solved)?\b',
        r'\bexcellent\s*(?:solution|work|answer)?\b',
        r'\bcomplete\s*proof\b',
        r'\bsolved\s*(?:correctly|completely)?\b',
    ]
    for pattern in full_patterns:
        if re.search(pattern, pred_lower):
            return "7", True
    
    # Check for zero/no credit patterns
    zero_patterns = [
        r'\bno\s*(?:credit|points?|score|marks?)?\b',
        r'\bzero\s*(?:credit|points?|score|marks?)?\b',
        r'\b0\s*(?:points?|credit|score|marks?)?\b',
        r'\bincorrect\s*(?:solution|answer|proof)?\b',
        r'\bwrong\s*(?:solution|answer|approach)?\b',
        r'\bno\s*solution\b',
        r'\bempty\s*(?:answer|solution)?\b',
        r'\bno\s*progress\b',
        r'\bno\s*meaningful\s*(?:progress|work)?\b',
        r'\bcompletely\s*(?:wrong|incorrect)\b',
    ]
    for pattern in zero_patterns:
        if re.search(pattern, pred_lower):
            return "0", True
    
    # Check for specific grade level descriptions
    grade_descriptions = {
        '6': [r'\bminor\s*(?:gap|error|issue)', r'\balmost\s*complete', r'\bsmall\s*(?:error|mistake)'],
        '5': [r'\bsignificant\s*gap', r'\bmajor\s*step\s*correct', r'\bmostly\s*correct'],
        '4': [r'\bsome\s*correct\s*key\s*steps', r'\bpartial\s*solution', r'\bgood\s*progress'],
        '3': [r'\bmeaningful\s*progress', r'\bsome\s*progress', r'\bpartial\s*approach'],
        '2': [r'\blimited\s*progress', r'\bsome\s*correct\s*ideas', r'\bfew\s*correct\s*steps'],
        '1': [r'\bminimal\s*progress', r'\brelevant\s*observation', r'\bminor\s*insight'],
    }
    
    for grade, patterns in grade_descriptions.items():
        for pattern in patterns:
            if re.search(pattern, pred_lower):
                return grade, True
    
    # Check for other valid grade keywords
    valid_keywords = ['correct', 'partial', 'n/a', 'not applicable']
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

## Response Format (CRITICAL)

You MUST respond in valid JSON format wrapped in <json> tags. The JSON must be properly formatted with double quotes around keys and string values.

<json>
{{
    "reasoning": "Your detailed step-by-step analysis of the student's answer, including: (1) approach identification, (2) correctness verification, (3) gaps/errors found, (4) comparison to official solution, (5) justification for the assigned grade",
    "response": "7"
}}
</json>

CRITICAL REQUIREMENTS:
1. The "response" field MUST contain ONLY a single digit from 0-7 (e.g., "7", "5", "2", "0")
2. Do NOT include any text, explanations, or ranges in the response field
3. Do NOT use quotes around the number in the response field - just the digit
4. Put ALL analysis and justification in the "reasoning" field
5. Ensure the JSON is valid - use double quotes for all strings
6. The response field should be the LAST field in the JSON object

Example of CORRECT response:
<json>
{{
    "reasoning": "The student provided a complete and correct solution with clear reasoning. All steps are valid and the proof is rigorous.",
    "response": "7"
}}
</json>

Example of INCORRECT response:
<json>
{{
    "reasoning": "Good solution",
    "response": "Grade: 7 points"  // WRONG - should be just "7"
}}
</json>

## Grading Decision Framework

When determining the grade, follow this systematic approach:

**For 7 points (Full Credit):**
- The solution is complete and correct
- All logical steps are valid
- The proof is rigorous and well-structured
- Alternative valid approaches are equally acceptable

**For 6 points (Minor Issues):**
- The solution is essentially correct
- Minor gaps in reasoning that don't affect validity
- Small computational errors with correct approach
- Presentation issues that don't obscure the logic

**For 5 points (Significant Gaps):**
- Correct approach but missing some key justifications
- Major steps are correct but some details omitted
- Significant progress toward complete solution

**For 4 points (Partial Solution):**
- Some correct key steps identified
- Good progress but incomplete
- Partial understanding demonstrated

**For 3 points (Meaningful Progress):**
- Some meaningful progress toward solution
- Partial approach with correct elements
- Understanding of problem structure shown

**For 2 points (Limited Progress):**
- Limited progress with some correct ideas
- Few correct steps identified
- Some understanding of problem

**For 1 point (Minimal Progress):**
- Minimal progress or relevant observations
- Minor insight into problem
- Relevant observation made

**For 0 points (No Credit):**
- No meaningful progress
- Completely incorrect approach
- Empty or irrelevant answer"""

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
                return prediction, reasoning
            
            # Strategy 1: Try <json> tags first (most reliable)
            extracted = _extract_jsons(full_text)
            if extracted:
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = str(last_json["response"]).strip()
                    # Clean up the prediction - remove any non-digit characters
                    prediction = re.sub(r'[^0-7]', '', prediction)
                    if prediction:  # If we have a valid digit
                        prediction = prediction[0]  # Take first valid digit
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
            json_pattern = r'\{[^{}]*"response"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            for match in re.finditer(json_pattern, full_text):
                try:
                    fallback = json.loads(match.group())
                    if "response" in fallback:
                        prediction = str(fallback["response"]).strip()
                        if "reasoning" in fallback:
                            reasoning = str(fallback["reasoning"])
                        if prediction != "None":
                            return prediction, reasoning
                except json.JSONDecodeError:
                    continue
            
            # Strategy 4: Look for explicit grade/score patterns in the last 1000 chars
            # (grades are often stated at the end)
            last_part = full_text[-1000:] if len(full_text) > 1000 else full_text
            
            # Strong patterns first
            strong_patterns = [
                r'"response"\s*:\s*"([0-7])"',
                r'"response"\s*:\s*([0-7])\b',
                r'(?:final\s+)?(?:grade|score|mark)\s*[:=]\s*([0-7])\b',
                r'\bgrade\s+(?:is|of|equals?)\s*[:=]?\s*([0-7])\b',
                r'\b(?:award|assign|give)\s*[:=]?\s*([0-7])\s*(?:points?)?\b',
                r'\b([0-7])\s*(?:points?|out\s+of\s+7)\b',
            ]
            for pattern in strong_patterns:
                grade_match = re.search(pattern, last_part, re.IGNORECASE)
                if grade_match:
                    prediction = grade_match.group(1)
                    return prediction, reasoning
            
            # Strategy 5: Look for standalone digits 0-7 in the response field context
            # This handles cases where the JSON might be malformed
            response_context = re.search(r'response["\']?\s*[:=]\s*["\']?([0-7])', full_text, re.IGNORECASE)
            if response_context:
                prediction = response_context.group(1)
                return prediction, reasoning
            
            # Strategy 6: Look for any single digit 0-7 that appears to be a grade
            # Only if it's on its own line or clearly separated
            standalone_match = re.search(r'(?:^|\n)\s*([0-7])\s*(?:\n|$)', full_text)
            if standalone_match:
                prediction = standalone_match.group(1)
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
            # Strategy 1: Try to find any numeric grade in the full response
            if response:
                numeric_match = re.search(r'\b([0-7])\b', response)
                if numeric_match:
                    validated_grade = numeric_match.group(1)
                    is_valid = True
                    self.log_fn(f"Fallback 1 found grade: {validated_grade}")
            
            # Strategy 2: Look for grade at the very end of the response
            if not is_valid and response:
                end_match = re.search(r'(?:grade|score)\s*:?\s*([0-7])\s*$', response.strip(), re.IGNORECASE)
                if end_match:
                    validated_grade = end_match.group(1)
                    is_valid = True
                    self.log_fn(f"Fallback 2 found grade: {validated_grade}")
            
            # Strategy 3: Check for common grade indicators in the last 500 chars
            if not is_valid and response:
                last_part = response[-500:] if len(response) > 500 else response
                indicators = {
                    '7': ['full credit', 'complete solution', 'perfect', 'all points', 'fully correct'],
                    '0': ['no credit', 'zero', 'incorrect', 'wrong', 'no solution', 'empty'],
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
            
            # Strategy 5: Look for grade descriptions in the response
            if not is_valid and response:
                grade_desc_patterns = {
                    '6': [r'\bminor\s*(?:gap|error)', r'\balmost\s*complete', r'\bsmall\s*error'],
                    '5': [r'\bsignificant\s*gap', r'\bmostly\s*correct'],
                    '4': [r'\bpartial\s*solution', r'\bsome\s*correct\s*steps'],
                    '3': [r'\bmeaningful\s*progress', r'\bsome\s*progress'],
                    '2': [r'\blimited\s*progress', r'\bsome\s*correct\s*ideas'],
                    '1': [r'\bminimal\s*progress', r'\brelevant\s*observation'],
                }
                for grade, patterns in grade_desc_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, response.lower()):
                            validated_grade = grade
                            is_valid = True
                            self.log_fn(f"Fallback 5 found grade: {validated_grade}")
                            break
                    if is_valid:
                        break
        
        # Final validation: ensure grade is within valid range
        if is_valid:
            try:
                grade_val = int(validated_grade)
                if grade_val < 0 or grade_val > 7:
                    is_valid = False
            except (ValueError, TypeError):
                is_valid = False
        
        # Retry with a clearer prompt if grade is still invalid
        if not is_valid:
            self.log_fn(f"Initial grading failed, attempting retry with clearer instructions...")
            retry_prompt = self._build_retry_prompt(inputs, response)
            
            retry_response, retry_msg_history, retry_info = get_response_from_llm(
                msg=retry_prompt,
                model=self.model,
                msg_history=[],
            )
            
            # Extract from retry response
            retry_prediction, retry_reasoning = self._extract_prediction(retry_msg_history)
            validated_grade, is_valid = _validate_grade(retry_prediction, grading_guidelines)
            
            if retry_reasoning:
                self.log_fn(f"Retry reasoning: {retry_reasoning[:200]}...")
            self.log_fn(f"Retry extracted grade: {retry_prediction}, Validated: {validated_grade}, Is valid: {is_valid}")
            
            # If retry succeeded, use the new history
            if is_valid:
                msg_history = retry_msg_history
                response = retry_response
        
        if not is_valid:
            self.log_fn(f"Could not extract valid grade, returning 'None'")
            validated_grade = "None"

        return str(validated_grade), msg_history

    def _build_retry_prompt(self, inputs: dict, previous_response: str) -> str:
        """Build a simplified retry prompt when initial grading fails."""
        problem = inputs.get("problem", "")
        student_answer = inputs.get("student_answer", "")
        
        return f"""You are an expert IMO grader. Your previous response could not be parsed correctly.

Please provide ONLY a valid JSON response in the exact format below. The response field must contain ONLY a single digit from 0-7.

## Problem
{problem[:500]}{"..." if len(problem) > 500 else ""}

## Student's Answer
{student_answer[:1000]}{"..." if len(student_answer) > 1000 else ""}

## Your Previous Response (unparseable)
{previous_response[:500]}{"..." if len(previous_response) > 500 else ""}

## Required Response Format
You MUST respond with ONLY this JSON format:

<json>
{{
    "reasoning": "Brief analysis of the student's solution",
    "response": "7"
}}
</json>

CRITICAL: The "response" field must contain ONLY a single digit: 0, 1, 2, 3, 4, 5, 6, or 7.
No other text, no quotes around the number, no explanations. Just the digit."""
