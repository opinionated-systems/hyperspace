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
    
    # Check for "X out of 7" or "X points" patterns
    out_of_match = re.search(r'\b([0-7])\s*(?:out\s+of\s+7|points?|marks?)\b', pred_lower)
    if out_of_match:
        return out_of_match.group(1), True
    
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
    ]
    for pattern in zero_patterns:
        if re.search(pattern, pred_lower):
            return "0", True
    
    # Check for partial credit patterns to extract numeric grades
    partial_patterns = [
        r'\b([0-7])\s*(?:points?|marks?|credit)\b',
        r'\bgrade\s+(?:of\s+)?([0-7])\b',
        r'\bscore\s+(?:of\s+)?([0-7])\b',
        r'\baward\s+(?:of\s+)?([0-7])\b',
    ]
    for pattern in partial_patterns:
        match = re.search(pattern, pred_lower)
        if match:
            return match.group(1), True
    
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
   - Check if the student used correct theorems or formulas

3. **Partial Credit Evaluation**:
   - IMO problems use a 0-7 point scale
   - Award points for meaningful progress toward the solution
   - Consider partial results, correct methodology with errors, or incomplete proofs
   - Even small correct steps deserve some credit (1-2 points)

4. **Final Grade Determination**:
   - 7 points: Complete, correct solution with proper justification
   - 6 points: Minor flaw in an otherwise complete solution
   - 5 points: Significant progress with substantial solution elements
   - 3-4 points: Partial progress with some correct elements
   - 1-2 points: Minimal progress or some relevant ideas
   - 0 points: No meaningful progress or completely incorrect

## Detailed Grading Rubric

**7 points - Complete Solution**:
- All steps are logically sound and correct
- Proper mathematical justification provided
- Final answer matches the official solution
- No significant gaps or errors
- All necessary cases/conditions considered

**6 points - Near-Complete**:
- Solution is essentially correct
- Minor flaw: small computational error, missing edge case, or slight gap
- Main result is correct and well-justified
- Error does not affect the core solution
- Could be fixed with minor correction

**5 points - Substantial Progress**:
- Solved a major part of the problem correctly
- OR provided a mostly correct approach with significant gaps
- Shows strong understanding of the problem structure
- Missing some elements but core insight is present
- Good attempt at the main proof/argument

**4 points - Good Partial Progress**:
- Multiple correct elements or steps
- Good initial approach with some correct calculations
- Partial results that demonstrate understanding
- Some meaningful progress beyond just understanding the problem
- Correct setup but execution had issues

**3 points - Partial Progress**:
- Some correct elements or initial approach
- Perhaps correct setup but failed to execute
- Shows understanding but limited concrete progress
- May have correct ideas but significant gaps
- Some valid intermediate results

**2 points - Minimal Progress**:
- Some relevant ideas or understanding of the problem
- Little concrete progress toward solution
- May have started correctly but quickly went wrong
- Shows engagement but limited success
- Correctly identified what needs to be proved

**1 point - Very Minimal Progress**:
- Shows some understanding of what the problem asks
- Very little progress beyond restating the problem
- May have written something relevant but no real solution attempt
- Essentially no meaningful mathematical progress
- Attempted to use relevant concepts

**0 points - No Credit**:
- Answer is blank or completely irrelevant
- Shows no understanding of the problem
- Completely incorrect approach with no redeeming elements
- No meaningful progress whatsoever
- Wrong problem or nonsense answer

## Common Grading Scenarios

**Full Credit (7 points)**: The solution is complete, correct, and well-justified. All steps are logically sound and the final answer matches the official solution.

**Near-Complete (6 points)**: The solution is essentially correct but has a minor flaw (e.g., a small computational error, missing edge case, or slight gap in reasoning that doesn't affect the main result).

**Substantial Progress (5 points)**: The student made significant progress, perhaps solving a major part of the problem or providing a mostly correct approach with some significant gaps.

**Partial Progress (3-4 points)**: The student has some correct elements, perhaps a good initial approach, some correct calculations, or partial results that show understanding.

**Minimal Progress (1-2 points)**: The student shows some relevant ideas or understanding of the problem, but little concrete progress toward the solution.

**No Credit (0 points)**: The answer is blank, completely irrelevant, or shows no understanding of the problem.

## Partial Credit Guidelines

When awarding partial credit, consider:
- **Correct setup/approach** (even if execution fails): 2-3 points
- **Correct intermediate results**: 1-2 points per significant result
- **Correct final answer with poor justification**: 4-5 points (depending on gap)
- **Correct method with computational error**: 4-6 points (depending on severity)
- **Alternative valid approach**: Full credit if correct, partial if incomplete
- **Partial proof of a multi-part problem**: Proportional credit

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
- Be generous with partial credit - even small correct steps deserve recognition

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
    "reasoning": "The student correctly identified the problem type and set up the approach, but made errors in execution. Awarding 2 points for correct setup and partial understanding.",
    "response": "2"
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
                if prediction != "None":
                    return prediction, reasoning
            
            # Strategy 2: Try markdown code blocks
            md_json = _extract_json_from_markdown(last_msg)
            if md_json:
                if "response" in md_json:
                    prediction = str(md_json["response"]).strip()
                if "reasoning" in md_json:
                    reasoning = str(md_json["reasoning"])
                if prediction != "None":
                    return prediction, reasoning
            
            # Strategy 3: Try to find any JSON-like structure with response field
            # Use a more robust pattern that handles nested braces
            json_pattern = r'\{[^{}]*"response"[^{}]*\}'
            json_matches = re.findall(json_pattern, last_msg)
            for json_match in json_matches:
                try:
                    fallback = json.loads(json_match)
                    pred = str(fallback.get("response", "None")).strip()
                    if pred and pred != "None":
                        prediction = pred
                        if "reasoning" in fallback:
                            reasoning = str(fallback["reasoning"])
                        return prediction, reasoning
                except json.JSONDecodeError:
                    continue
            
            # Strategy 4: Try to find JSON with nested structure using brace counting
            start = last_msg.find('{')
            end = last_msg.rfind('}')
            if start != -1 and end != -1 and end > start:
                try:
                    # Use brace counting to find the outermost valid JSON
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
                        potential_json = last_msg[start:json_end+1]
                        fallback = json.loads(potential_json)
                        if "response" in fallback:
                            prediction = str(fallback["response"]).strip()
                            if "reasoning" in fallback:
                                reasoning = str(fallback["reasoning"])
                            if prediction != "None":
                                return prediction, reasoning
                except (json.JSONDecodeError, ValueError):
                    pass
            
            # Strategy 5: Last resort - look for grade patterns in text
            if prediction == "None":
                # Look for "Grade: X" or "Final grade: X" patterns
                grade_patterns = [
                    r'(?:grade|score|mark|final grade|final score)\s*:?\s*([0-7])',
                    r'(?:grade|score|mark)\s+is\s+([0-7])',
                    r'(?:grade|score|mark)\s+of\s+([0-7])',
                    r'\bgives?\s+(?:a\s+)?([0-7])\s*(?:points?)?',
                    r'\baward\s+(?:a\s+)?([0-7])',
                    r'\bassign\s+(?:a\s+)?([0-7])',
                    r'\b([0-7])\s*(?:points?|marks?)\b',
                    r'\bgrade\s+(?:of\s+)?([0-7])\b',
                ]
                for pattern in grade_patterns:
                    grade_match = re.search(pattern, last_msg, re.IGNORECASE)
                    if grade_match:
                        prediction = grade_match.group(1)
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
5. Make sure your grade is consistent with your reasoning
6. Be generous with partial credit - even small correct steps deserve recognition

## Quick Reference: IMO 0-7 Point Scale

- **7**: Complete, correct solution with proper justification
- **6**: Minor flaw in an otherwise complete solution  
- **5**: Significant progress with substantial solution elements
- **4**: Good partial progress with multiple correct elements
- **3**: Partial progress with some correct elements
- **2**: Minimal progress, some relevant ideas
- **1**: Very minimal progress, little beyond understanding the problem
- **0**: No meaningful progress or completely incorrect

## Partial Credit Guidelines

When awarding partial credit, consider:
- **Correct setup/approach** (even if execution fails): 2-3 points
- **Correct intermediate results**: 1-2 points per significant result
- **Correct final answer with poor justification**: 4-5 points (depending on gap)
- **Correct method with computational error**: 4-6 points (depending on severity)

## Output Format (STRICT REQUIREMENT)

You MUST respond with a valid JSON object wrapped in <json> tags:

<json>
{{
    "reasoning": "Your detailed analysis here...",
    "response": "X"
}}
</json>

Where X is exactly one digit: 0, 1, 2, 3, 4, 5, 6, or 7"""

    def _verify_grade_consistency(self, grade: str, reasoning: str) -> tuple[str, bool]:
        """Verify that the grade is consistent with the reasoning.
        
        Uses a comprehensive keyword-based approach to detect inconsistencies
        between the assigned grade and the reasoning provided.
        
        Args:
            grade: The extracted grade
            reasoning: The reasoning text
            
        Returns:
            (verified_grade, is_consistent)
        """
        if not reasoning or grade == "None":
            return grade, True
            
        reasoning_lower = reasoning.lower()
        
        # Define grade-appropriate keywords with context awareness
        # High grades (6-7) should have positive indicators
        # Low grades (0-2) should have negative indicators
        # Middle grades (3-5) can have mixed indicators
        
        # Context-aware positive indicators (check for "not" prefix)
        positive_indicators = [
            "correct", "valid", "proper", "complete", "sound", "well", "good",
            "right approach", "appropriate", "justified", "proven", "shown",
            "correctly", "valid approach", "properly justified"
        ]
        
        # Context-aware negative indicators
        negative_indicators = [
            "incorrect", "error", "mistake", "wrong", "flaw", "missing",
            "incomplete", "not correct", "not valid", "invalid", "failed",
            "does not work", "cannot", "unable", "no progress", "not proven",
            "not justified", "not complete"
        ]
        
        # Strong positive indicators for grade 7
        strong_positive = [
            "complete solution", "fully correct", "perfect", "excellent",
            "all steps correct", "correctly solved", "complete and correct",
            "fully justified", "entirely correct"
        ]
        
        # Strong negative indicators for low grades
        strong_negative = [
            "completely wrong", "totally incorrect", "major error", "critical flaw",
            "no meaningful progress", "fundamentally wrong", "does not solve",
            "completely incorrect", "entirely wrong"
        ]
        
        # Count indicators with context awareness
        def count_with_context(indicators, text):
            """Count indicators but exclude those preceded by negation."""
            count = 0
            for ind in indicators:
                # Find all occurrences
                idx = 0
                while True:
                    idx = text.find(ind, idx)
                    if idx == -1:
                        break
                    # Check if preceded by negation (within 20 chars before)
                    start = max(0, idx - 20)
                    context = text[start:idx]
                    negation_words = ["not ", "no ", "never ", "hardly ", "barely ", "scarcely ", "without "]
                    if not any(neg in context[-5:] for neg in negation_words):
                        count += 1
                    idx += len(ind)
            return count
        
        positive_count = count_with_context(positive_indicators, reasoning_lower)
        negative_count = count_with_context(negative_indicators, reasoning_lower)
        strong_pos_count = count_with_context(strong_positive, reasoning_lower)
        strong_neg_count = count_with_context(strong_negative, reasoning_lower)
        
        # Grade-specific consistency checks with refined thresholds
        if grade == "7":
            # Grade 7 should have strong positive indicators and no strong negative
            if strong_neg_count > 0:
                self.log_fn(f"Grade 7 but reasoning contains strong negative indicators - inconsistency detected")
                return grade, False
            # Should have at least some positive indicators
            if positive_count == 0 and strong_pos_count == 0:
                self.log_fn(f"Grade 7 but no positive indicators in reasoning - potential inconsistency")
                return grade, False
                
        elif grade == "6":
            # Grade 6 should be mostly positive with minor issues
            if strong_neg_count > 1:
                self.log_fn(f"Grade 6 but multiple strong negative indicators - potential inconsistency")
                return grade, False
            if negative_count > positive_count + 3:  # Relaxed threshold
                self.log_fn(f"Grade 6 but significantly more negative than positive indicators")
                return grade, False
                
        elif grade in ["0", "1"]:
            # Low grades should have negative indicators
            if strong_pos_count > 0:
                self.log_fn(f"Grade {grade} but reasoning contains strong positive indicators - inconsistency detected")
                return grade, False
            if positive_count > negative_count + 3:  # Relaxed threshold
                self.log_fn(f"Grade {grade} but significantly more positive than negative indicators")
                return grade, False
                
        # For grades 2-5, mixed indicators are expected, so be more lenient
        # Only flag if there's a severe mismatch
        elif grade in ["2", "3", "4", "5"]:
            if strong_pos_count > 0 and strong_neg_count > 0:
                # Both strong positive and negative - this is actually good for partial credit
                pass
            elif grade in ["4", "5"] and strong_neg_count > 3:  # Relaxed threshold
                self.log_fn(f"Grade {grade} but multiple strong negative indicators - potential inconsistency")
                return grade, False
            elif grade in ["2", "3"] and strong_pos_count > 3:  # Relaxed threshold
                self.log_fn(f"Grade {grade} but multiple strong positive indicators - potential inconsistency")
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
                    return "None", []
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
                # Try to find any numeric grade in the response
                numeric_match = re.search(r'\b([0-7])\b', response)
                if numeric_match:
                    validated_grade = numeric_match.group(1)
                    is_valid = True
                    self.log_fn(f"Fallback extraction found grade: {validated_grade}")
                else:
                    # Try to find grade patterns in the full response
                    grade_patterns = [
                        (r'(?:grade|score|mark|final grade|final score)\s*:?\s*([0-7])', 'numeric'),
                        (r'\bfull\s*(?:credit|points?|score)?\b', 'full'),
                        (r'\bcorrect\s*(?:solution|answer)?\b', 'correct'),
                        (r'\bno\s*(?:credit|points?|score|marks?)?\b', 'zero'),
                        (r'\bzero\s*(?:credit|points?|score|marks?)?\b', 'zero'),
                        (r'\bincorrect\s*(?:solution|answer)?\b', 'zero'),
                    ]
                    for pattern, pattern_type in grade_patterns:
                        match = re.search(pattern, response, re.IGNORECASE)
                        if match:
                            if pattern_type == 'full' or pattern_type == 'correct':
                                validated_grade = "7"
                            elif pattern_type == 'zero':
                                validated_grade = "0"
                            else:
                                validated_grade = match.group(1)
                            is_valid = True
                            self.log_fn(f"Pattern-based fallback found grade: {validated_grade}")
                            break
            
            # Verify grade-reasoning consistency
            if is_valid and reasoning:
                validated_grade, is_consistent = self._verify_grade_consistency(validated_grade, reasoning)
                if not is_consistent and attempt < max_retries:
                    self.log_fn(f"Grade-reasoning inconsistency detected, retrying...")
                    error_reason = f"The grade '{validated_grade}' may not match the reasoning provided. The reasoning suggests a different grade might be appropriate."
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
        
        # All retries exhausted, return best effort
        self.log_fn(f"All retries exhausted, returning best effort: {validated_grade}")
        return str(validated_grade) if validated_grade != "None" else "0", msg_history
