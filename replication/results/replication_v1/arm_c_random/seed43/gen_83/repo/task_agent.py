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
    Enhanced to handle nested JSON structures and edge cases.
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
        
        # Try direct parsing first
        try:
            results.append(json.loads(inner))
            continue
        except json.JSONDecodeError:
            pass
        
        # Try to find valid JSON within the content using brace counting
        try:
            json_start = inner.find('{')
            if json_start == -1:
                continue
                
            # Use brace counting to find the matching closing brace
            brace_count = 0
            json_end = -1
            for i, char in enumerate(inner[json_start:], start=json_start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i
                        break
            
            if json_end != -1:
                potential_json = inner[json_start:json_end+1]
                results.append(json.loads(potential_json))
                continue
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Last resort: try simple find/rfind approach
        try:
            json_start = inner.find('{')
            json_end = inner.rfind('}')
            if json_start != -1 and json_end != -1 and json_end > json_start:
                results.append(json.loads(inner[json_start:json_end+1]))
        except json.JSONDecodeError:
            continue
            
    return results or None


def _extract_json_from_markdown(text: str) -> dict | None:
    """Extract JSON from markdown code blocks (```json ... ```).
    
    Enhanced to handle nested JSON structures and edge cases.
    """
    # Try json-specific blocks first
    pattern = r'```json\s*(.*?)\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    # Also try generic code blocks
    generic_pattern = r'```\s*(.*?)\s*```'
    generic_matches = re.findall(generic_pattern, text, re.DOTALL)
    all_matches = matches + [m for m in generic_matches if m not in matches]
    
    for match in all_matches:
        match = match.strip()
        
        # Try direct parsing first
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            pass
        
        # Try brace counting for nested structures
        try:
            json_start = match.find('{')
            if json_start == -1:
                continue
                
            brace_count = 0
            json_end = -1
            for i, char in enumerate(match[json_start:], start=json_start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i
                        break
            
            if json_end != -1:
                potential_json = match[json_start:json_end+1]
                return json.loads(potential_json)
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Last resort: simple find/rfind
        try:
            json_start = match.find('{')
            json_end = match.rfind('}')
            if json_start != -1 and json_end != -1 and json_end > json_start:
                return json.loads(match[json_start:json_end+1].strip())
        except json.JSONDecodeError:
            continue
            
    return None


def _validate_grade(prediction: str, grading_guidelines: str) -> tuple[str, bool]:
    """Validate that the extracted grade is reasonable.
    
    Enhanced validation with strict support for IMO 0-7 point scale.
    Only accepts single digit grades 0-7 for consistency.
    
    Returns:
        (validated_grade, is_valid)
    """
    if not prediction or prediction == "None":
        return "None", False
    
    pred_clean = prediction.strip()
    pred_lower = pred_clean.lower()
    
    # Strict validation: only accept single digit 0-7
    # This ensures consistent output format
    if pred_clean in ["0", "1", "2", "3", "4", "5", "6", "7"]:
        return pred_clean, True
    
    # Check for fractional grades like "3/7" or "5 / 7" - extract just the numerator
    fractional_match = re.search(r'\b([0-7])\s*/\s*7\b', pred_clean)
    if fractional_match:
        return fractional_match.group(1), True
    
    # Check for numeric grades embedded in text (0-7 for IMO problems)
    numeric_match = re.search(r'\b([0-7])\b', pred_clean)
    if numeric_match:
        return numeric_match.group(1), True
    
    # Check for full credit patterns -> 7
    full_patterns = [
        r'\bfull\s*(?:credit|points?|score)?\b',
        r'\bcomplete\s*(?:solution|answer|credit)?\b',
        r'\ball\s*(?:points?|credit|marks?)?\b',
        r'\bperfect\s*(?:score|solution)?\b',
        r'\bcorrect\s*(?:solution|answer)?\b',
        r'\bvalid\s*(?:solution|proof|answer)?\b',
        r'\bsound\s*(?:solution|proof|reasoning)?\b',
        r'\bwell\s*justified\b',
    ]
    for pattern in full_patterns:
        if re.search(pattern, pred_lower):
            return "7", True
    
    # Check for zero/no credit patterns -> 0
    zero_patterns = [
        r'\bno\s*(?:credit|points?|score|marks?)?\b',
        r'\bzero\s*(?:credit|points?|score|marks?)?\b',
        r'\b0\s*(?:points?|credit|score|marks?)?\b',
        r'\bincorrect\s*(?:solution|answer)?\b',
        r'\bwrong\s*(?:solution|answer)?\b',
        r'\bnone\b',
        r'\bno\s*solution\b',
        r'\bno\s*progress\b',
        r'\bblank\b',
        r'\bempty\b',
    ]
    for pattern in zero_patterns:
        if re.search(pattern, pred_lower):
            return "0", True
    
    # Check for intermediate grades based on common patterns
    # "partial credit" or "some credit" -> try to find a number
    partial_patterns = [
        r'\bpartial\s*(?:credit|points?|score)?\b',
        r'\bsome\s*(?:credit|points?|score)?\b',
        r'\bpartial\s*progress\b',
        r'\bsome\s*progress\b',
    ]
    for pattern in partial_patterns:
        if re.search(pattern, pred_lower):
            # Look for a number 1-6 in the text
            partial_match = re.search(r'\b([1-6])\b', pred_clean)
            if partial_match:
                return partial_match.group(1), True
            # Default to 3 for partial credit if no specific number found
            return "3", True
    
    # If no clear grade found, mark as invalid
    return pred_clean, False


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced reasoning and validation."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _build_prompt(self, inputs: dict) -> str:
        """Build a structured prompt for IMO grading with chain-of-thought reasoning.
        
        This method constructs a detailed prompt that guides the LLM to evaluate
        student solutions according to IMO grading standards (0-7 point scale).
        
        Args:
            inputs: Dictionary containing problem data with keys:
                - domain: Problem domain (e.g., "Mathematics")
                - problem: The problem statement
                - solution: The official solution
                - grading_guidelines: Specific grading criteria
                - student_answer: The student's submitted answer
                
        Returns:
            A formatted prompt string ready for LLM consumption
        """
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        return f"""You are an expert IMO (International Mathematical Olympiad) grader with extensive experience in mathematical problem evaluation.

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

## Instructions for Evaluation

1. **Initial Assessment**: Read the student's answer completely before forming any conclusions.

2. **Step-by-Step Analysis**: 
   - Compare each step of the student's solution to the official solution
   - Identify any logical errors, computational mistakes, or gaps in reasoning
   - Note any creative alternative approaches that may be valid

3. **Partial Credit Evaluation**:
   - IMO problems use a 0-7 point scale
   - Award points for meaningful progress toward the solution
   - Consider partial results, correct methodology with errors, or incomplete proofs

4. **Final Grade Determination**:
   - 7 points: Complete, correct solution with proper justification
   - 6 points: Minor flaw in an otherwise complete solution
   - 5 points: Significant progress with substantial solution elements
   - 4 points: Good partial progress with several correct elements
   - 3 points: Some partial progress with correct elements
   - 2 points: Minimal progress with some relevant ideas
   - 1 point: Very minimal progress or relevant ideas only
   - 0 points: No meaningful progress or completely incorrect

## Common Grading Scenarios

**Full Credit (7 points)**: The solution is complete, correct, and well-justified. All steps are logically sound and the final answer matches the official solution.

**Near-Complete (6 points)**: The solution is essentially correct but has a minor flaw (e.g., a small computational error, missing edge case, or slight gap in reasoning that doesn't affect the main result).

**Substantial Progress (5 points)**: The student made significant progress, perhaps solving a major part of the problem or providing a mostly correct approach with some significant gaps.

**Good Partial Progress (4 points)**: The student has several correct elements, perhaps a good initial approach, some correct calculations, or partial results that show understanding.

**Some Partial Progress (3 points)**: The student has some correct elements but limited progress toward the full solution.

**Minimal Progress (2 points)**: The student shows some relevant ideas or understanding of the problem, but little concrete progress toward the solution.

**Very Minimal Progress (1 point)**: The student shows very minimal understanding or only mentions concepts related to the solution.

**No Credit (0 points)**: The answer is blank, completely irrelevant, or shows no understanding of the problem.

## Partial Credit Guidelines

When assigning partial credit, consider:
- **Correct approach with minor errors (5-6 points)**: Student understands the method but makes small execution errors
- **Correct approach with significant errors (4-5 points)**: Student understands the method but makes substantial execution errors
- **Correct intermediate results (3-4 points)**: Student gets some key lemmas or partial results correct
- **Good setup but limited progress (2-3 points)**: Student correctly interprets the problem but makes limited progress
- **Relevant ideas only (1-2 points)**: Student mentions concepts related to the solution but cannot apply them
- **Wrong approach but some correct math (1-2 points)**: Method is wrong but calculations are correct

## IMO-Specific Grading Considerations

**Proof Structure**: IMO problems require rigorous proofs. A correct answer without proper justification receives at most 1 point.

**Alternative Solutions**: Valid alternative approaches that correctly solve the problem should receive full credit (7 points), even if different from the official solution.

**Partial Results**: Award points for:
- Correctly proving a significant lemma (2-4 points depending on importance)
- Correctly solving a special case (1-3 points depending on how special)
- Meaningful progress on the main problem (1-5 points depending on extent)

**Common Errors**:
- Arithmetic errors in an otherwise correct solution: -1 point (6 points total)
- Logical gaps that can be filled: -1 to -2 points
- Wrong approach with some correct calculations: 1-2 points
- Correct answer, no work shown: 0-1 points

**Important**: Always provide the specific numeric grade (0-7) that best matches the student's work based on the official solution and grading guidelines. Be precise with your grade - use the full 0-7 scale appropriately.

## Output Format (STRICT REQUIREMENT)

You MUST respond with a valid JSON object wrapped in <json> tags. The JSON must have exactly these two fields:

<json>
{{
    "reasoning": "Your detailed analysis here. Explain the student's approach, identify errors or gaps, compare to official solution, and justify your grade...",
    "response": "X"
}}
</json>

CRITICAL RULES:
- The "response" field MUST contain ONLY a single digit: 0, 1, 2, 3, 4, 5, 6, or 7
- Do NOT include quotes around the number, explanations, or any other text in the response field
- The "reasoning" field should contain your complete analysis (can be multiple sentences)
- Ensure the JSON is valid and properly formatted
- Double-check that your grade matches your reasoning (e.g., don't give 7 if you mention major errors)

EXAMPLES OF VALID RESPONSES:
<json>
{{
    "reasoning": "The student provided a complete and correct solution with proper justification. All steps are logically sound.",
    "response": "7"
}}
</json>

<json>
{{
    "reasoning": "The student made good progress with the initial approach but had a computational error in the final step.",
    "response": "4"
}}
</json>

<json>
{{
    "reasoning": "The answer is blank with no meaningful content or mathematical work shown.",
    "response": "0"
}}
</json>"""

    def _extract_prediction(self, msg_history: list[dict]) -> tuple[str, str]:
        """Extract prediction and reasoning from message history with enhanced extraction.
        
        Uses multiple extraction strategies to handle various response formats.
        
        Returns:
            (prediction, reasoning)
        """
        prediction = "None"
        reasoning = ""
        
        try:
            # Handle different message formats
            last_msg = ""
            if msg_history:
                last_entry = msg_history[-1]
                if isinstance(last_entry, dict):
                    # Try common keys for message content
                    last_msg = last_entry.get("text") or last_entry.get("content", "")
                    if not last_msg and "message" in last_entry:
                        msg_obj = last_entry["message"]
                        if isinstance(msg_obj, dict):
                            last_msg = msg_obj.get("content", "")
            
            if not last_msg:
                return prediction, reasoning
            
            # Strategy 1: Try <json> tags first (most reliable)
            extracted = _extract_jsons(last_msg)
            if extracted:
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = str(last_json["response"]).strip()
                if "reasoning" in last_json:
                    reasoning = str(last_json["reasoning"])
                return prediction, reasoning
            
            # Strategy 2: Try markdown code blocks
            md_json = _extract_json_from_markdown(last_msg)
            if md_json:
                if "response" in md_json:
                    prediction = str(md_json["response"]).strip()
                if "reasoning" in md_json:
                    reasoning = str(md_json["reasoning"])
                return prediction, reasoning
            
            # Strategy 3: Try to find JSON with brace counting for nested structures
            start = last_msg.find('{')
            while start != -1:
                # Use brace counting to find matching closing brace
                brace_count = 0
                json_end = -1
                for i, char in enumerate(last_msg[start:], start=start):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i
                            break
                
                if json_end != -1:
                    try:
                        potential_json = last_msg[start:json_end+1]
                        fallback = json.loads(potential_json)
                        if "response" in fallback:
                            prediction = str(fallback["response"]).strip()
                            if "reasoning" in fallback:
                                reasoning = str(fallback["reasoning"])
                            return prediction, reasoning
                    except json.JSONDecodeError:
                        pass
                
                # Look for next opening brace
                start = last_msg.find('{', start + 1)
            
            # Strategy 4: Look for grade patterns in text
            # Look for "Grade: X" or "Final grade: X" patterns
            grade_patterns = [
                r'(?:grade|score|mark|final grade|final score)\s*:?\s*([0-7])\b',
                r'(?:grade|score|mark)\s+is\s+([0-7])\b',
                r'(?:grade|score|mark)\s+of\s+([0-7])\b',
                r'\bgives?\s+(?:a\s+)?([0-7])\s*(?:points?)?\b',
                r'\baward\s+(?:a\s+)?([0-7])\b',
                r'\bassign\s+(?:a\s+)?([0-7])\b',
                r'\b(?:the\s+)?(?:grade|score)\s+(?:should\s+be|is)\s+([0-7])\b',
                r'\b(?:deserves?|earns?)\s+(?:a\s+)?([0-7])\b',
                r'\b(?:assigning?|giving?)\s+(?:a\s+)?([0-7])\b',
                r'\b(?:worth|value)\s+(?:a\s+)?([0-7])\b',
            ]
            for pattern in grade_patterns:
                grade_match = re.search(pattern, last_msg, re.IGNORECASE)
                if grade_match:
                    prediction = grade_match.group(1)
                    # Try to extract reasoning from surrounding text
                    reasoning_match = re.search(r'(?:reasoning|analysis|explanation|justification|assessment)[:\s]+(.+?)(?:\n\n|\Z)', last_msg, re.IGNORECASE | re.DOTALL)
                    if reasoning_match:
                        reasoning = reasoning_match.group(1).strip()
                    break
            
            # Strategy 5: Look for standalone digits 0-7 in the last few lines
            if prediction == "None":
                lines = last_msg.strip().split('\n')
                # Check last 5 lines for standalone digits
                for line in reversed(lines[-5:]):
                    digit_match = re.search(r'\b([0-7])\b', line)
                    if digit_match:
                        prediction = digit_match.group(1)
                        break
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
        
        return prediction, reasoning

    def _build_retry_prompt(self, inputs: dict, previous_response: str, error_reason: str) -> str:
        """Build a retry prompt when the initial response was invalid.
        
        Args:
            inputs: Original problem inputs
            previous_response: The previous LLM response that failed validation
            error_reason: Why the previous response was invalid
            
        Returns:
            A formatted retry prompt
        """
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        return f"""You are an expert IMO (International Mathematical Olympiad) grader. Your previous response was invalid.

## Previous Response (INVALID)
{previous_response}

## Error
{error_reason}

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

## Instructions for Evaluation (RETRY)

Your previous response could not be parsed correctly. Please provide a valid JSON response following these strict rules:

1. The "response" field MUST contain ONLY a single digit: 0, 1, 2, 3, 4, 5, 6, or 7
2. Do NOT include quotes around the number, explanations, or any other text in the response field
3. The "reasoning" field should contain your complete analysis
4. Ensure the JSON is valid and properly formatted
5. The grade in "response" must match your reasoning - if you say the solution has errors, don't give it 7 points

## IMO Grading Scale Reference

- 7 points: Complete, correct solution with proper justification
- 6 points: Minor flaw in otherwise complete solution  
- 5 points: Significant progress with substantial elements
- 4 points: Good partial progress with several correct elements
- 3 points: Some partial progress with correct elements
- 2 points: Minimal progress with some relevant ideas
- 1 point: Very minimal progress or relevant ideas only
- 0 points: No meaningful progress

## Output Format (STRICT REQUIREMENT)

You MUST respond with a valid JSON object wrapped in <json> tags:

<json>
{{
    "reasoning": "Your detailed analysis here...",
    "response": "X"
}}
</json>

Where X is exactly one digit: 0, 1, 2, 3, 4, 5, 6, or 7

## Examples of Valid Responses

<json>
{{
    "reasoning": "The student provided a complete and correct solution with proper justification. All steps are logically sound.",
    "response": "7"
}}
</json>

<json>
{{
    "reasoning": "The student made good progress with the initial approach but had a computational error in the final step.",
    "response": "4"
}}
</json>

<json>
{{
    "reasoning": "The answer is blank with no meaningful content.",
    "response": "0"
}}
</json>"""

    def _verify_grade_consistency(self, grade: str, reasoning: str) -> tuple[str, bool]:
        """Verify that the grade is consistent with the reasoning.
        
        Uses a balanced approach to detect only clear contradictions between
        the assigned grade and the reasoning provided.
        
        Args:
            grade: The extracted grade
            reasoning: The reasoning text
            
        Returns:
            (verified_grade, is_consistent)
        """
        if not reasoning or grade == "None":
            return grade, True
            
        reasoning_lower = reasoning.lower()
        
        # Only flag severe contradictions that clearly indicate a parsing error
        
        # If grade is 7 but reasoning explicitly states the solution is wrong
        if grade == "7":
            severe_negative = [
                "completely wrong", "totally incorrect", "fundamentally flawed",
                "does not solve", "fails to prove", "incorrect solution",
                "no solution", "no progress", "blank", "empty answer",
                "does not address", "irrelevant", "no meaningful",
                "no credit", "zero points", "0 points", "worth 0",
                "deserves 0", "should receive 0", "assign 0"
            ]
            for indicator in severe_negative:
                if indicator in reasoning_lower:
                    self.log_fn(f"Grade 7 but reasoning contains '{indicator}' - potential inconsistency")
                    return grade, False
                    
        # If grade is 0 but reasoning explicitly states the solution is correct
        if grade == "0":
            severe_positive = [
                "completely correct", "fully correct", "perfect solution",
                "correct solution", "valid proof", "correctly proves",
                "complete solution", "all steps correct", "fully justified",
                "correctly solved", "valid solution", "full credit",
                "full marks", "7 points", "deserves 7", "worth 7",
                "should receive 7", "assign 7"
            ]
            for indicator in severe_positive:
                if indicator in reasoning_lower:
                    self.log_fn(f"Grade 0 but reasoning contains '{indicator}' - potential inconsistency")
                    return grade, False
                    
        return grade, True

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with enhanced reasoning, validation, and retry.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_prompt(inputs)
        grading_guidelines = inputs.get("grading_guidelines", "")
        max_retries = 2
        
        for attempt in range(max_retries + 1):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
            except Exception as e:
                self.log_fn(f"LLM call failed (attempt {attempt + 1}): {e}")
                if attempt == max_retries:
                    return "0", []  # Return 0 instead of "None" on complete failure
                continue

            # Extract prediction with enhanced extraction
            prediction, reasoning = self._extract_prediction(msg_history)
            
            # Validate the grade
            validated_grade, is_valid = _validate_grade(prediction, grading_guidelines)
            
            # Log the reasoning and validation result
            if reasoning:
                self.log_fn(f"Reasoning: {reasoning[:200]}...")
            self.log_fn(f"Attempt {attempt + 1}: Extracted grade: {prediction}, Validated: {validated_grade}, Is valid: {is_valid}")
            
            # If grade is invalid, try to extract from the full response text
            if not is_valid and response:
                # Try to find any numeric grade in the response with word boundaries
                numeric_match = re.search(r'\b([0-7])\b', response)
                if numeric_match:
                    validated_grade = numeric_match.group(1)
                    is_valid = True
                    self.log_fn(f"Fallback extraction found grade: {validated_grade}")
                else:
                    # Try to find grade patterns in the full response
                    grade_patterns = [
                        (r'(?:grade|score|mark|final grade|final score)\s*:?\s*([0-7])\b', 'numeric'),
                        (r'\bfull\s*(?:credit|points?|score)?\b', 'full'),
                        (r'\bcomplete\s*(?:solution|answer|credit)?\b', 'full'),
                        (r'\ball\s*(?:points?|credit|marks?)?\b', 'full'),
                        (r'\bperfect\s*(?:score|solution)?\b', 'full'),
                        (r'\bcorrect\s*(?:solution|answer)?\b', 'full'),
                        (r'\bno\s*(?:credit|points?|score|marks?)?\b', 'zero'),
                        (r'\bzero\s*(?:credit|points?|score|marks?)?\b', 'zero'),
                        (r'\bincorrect\s*(?:solution|answer)?\b', 'zero'),
                        (r'\bwrong\s*(?:solution|answer)?\b', 'zero'),
                        (r'\bnone\b', 'zero'),
                        (r'\bblank\b', 'zero'),
                        (r'\bempty\b', 'zero'),
                    ]
                    for pattern, pattern_type in grade_patterns:
                        match = re.search(pattern, response, re.IGNORECASE)
                        if match:
                            if pattern_type == 'full':
                                validated_grade = "7"
                            elif pattern_type == 'zero':
                                validated_grade = "0"
                            else:
                                validated_grade = match.group(1)
                            is_valid = True
                            self.log_fn(f"Pattern-based fallback found grade: {validated_grade}")
                            break
            
            # Verify grade-reasoning consistency (only for severe contradictions)
            if is_valid and reasoning:
                validated_grade, is_consistent = self._verify_grade_consistency(validated_grade, reasoning)
                if not is_consistent and attempt < max_retries:
                    self.log_fn(f"Grade-reasoning inconsistency detected, retrying...")
                    error_reason = f"The grade '{validated_grade}' contradicts your reasoning. Your reasoning states the solution is {'correct' if validated_grade == '0' else 'incorrect'}, but you assigned {'0' if validated_grade == '0' else '7'} points. Please provide a consistent grade."
                    instruction = self._build_retry_prompt(inputs, response, error_reason)
                    continue
            
            # If we have a valid grade, return it
            if is_valid:
                return str(validated_grade), msg_history
            
            # If invalid and we have retries left, build retry prompt
            if attempt < max_retries:
                error_reason = f"Could not extract a valid grade from the response. The response field contained: '{prediction}'. Please provide a valid single digit (0-7) in the response field."
                instruction = self._build_retry_prompt(inputs, response, error_reason)
                self.log_fn(f"Invalid grade, retrying with corrected prompt...")
        
        # All retries exhausted, return best effort (default to 0 for safety)
        self.log_fn(f"All retries exhausted, returning best effort: {validated_grade}")
        return str(validated_grade) if validated_grade not in ["None", ""] else "0", msg_history
