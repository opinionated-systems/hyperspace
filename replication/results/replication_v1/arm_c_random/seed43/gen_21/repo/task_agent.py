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
    Also handles nested JSON objects within the tags.
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
            # Try to extract JSON from within the content if it's wrapped in other text
            try:
                # Look for JSON object pattern
                json_match = re.search(r'\{[\s\S]*\}', inner)
                if json_match:
                    results.append(json.loads(json_match.group()))
            except (json.JSONDecodeError, AttributeError):
                continue
    return results or None


def _extract_json_from_markdown(text: str) -> dict | None:
    """Extract JSON from markdown code blocks (```json ... ```)."""
    pattern = r'```json\s*(.*?)\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match.strip())
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
    
    Returns:
        (validated_grade, is_valid)
    """
    if not prediction or prediction == "None":
        return "None", False
    
    pred_clean = prediction.strip()
    pred_lower = pred_clean.lower()
    
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
    
    # Check for numeric grades (0-7 for IMO problems)
    numeric_match = re.search(r'\b([0-7])\b', pred_clean)
    if numeric_match:
        return numeric_match.group(1), True
    
    # Check for partial credit patterns with more variations
    partial_patterns = [
        r'partial\s*(?:credit)?\s*:?\s*([0-7])',
        r'partial\s*([0-7])\s*(?:points?)?',
        r'([0-7])\s*(?:points?)?\s*partial',
        r'partially\s*(?:correct)?\s*:?\s*([0-7])',
    ]
    for pattern in partial_patterns:
        partial_match = re.search(pattern, pred_lower)
        if partial_match:
            return f"Partial credit: {partial_match.group(1)}", True
    
    # Check for full credit patterns
    full_patterns = [
        r'\bfull\s*(?:credit|points?|score)?\b',
        r'\bcomplete\s*(?:solution|answer|credit)?\b',
        r'\ball\s*(?:points?|credit|marks?)?\b',
        r'\bperfect\s*(?:score|solution)?\b',
        r'\bcorrect\s*(?:solution|answer)?\b',
        r'\bfully\s*(?:correct)?\b',
    ]
    for pattern in full_patterns:
        if re.search(pattern, pred_lower):
            return "7", True
    
    # Check for zero/no credit patterns
    zero_patterns = [
        r'\bno\s*(?:credit|points?|score|marks?)?\b',
        r'\bzero\s*(?:credit|points?|score|marks?)?\b',
        r'\b0\s*(?:points?|credit|score|marks?)?\b',
        r'\bincorrect\s*(?:solution|answer)?\b',
        r'\bwrong\s*(?:solution|answer)?\b',
        r'\bno\s*solution\b',
        r'\bempty\s*(?:answer|solution)?\b',
    ]
    for pattern in zero_patterns:
        if re.search(pattern, pred_lower):
            return "0", True
    
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

## Response Format

You MUST respond in valid JSON format wrapped in <json> tags:

<json>
{{
    "reasoning": "Your detailed step-by-step analysis of the student's answer, including: (1) approach identification, (2) correctness verification, (3) gaps/errors found, (4) comparison to official solution, (5) justification for the assigned grade",
    "response": "A single numeric grade from 0-7 (e.g., '7', '5', '2', '0')"
}}
</json>

IMPORTANT: 
- The "response" field must contain ONLY a single digit from 0-7
- Do not include explanations, text, or ranges in the response field
- Put all analysis and justification in the "reasoning" field"""

    def _extract_prediction(self, msg_history: list[dict]) -> tuple[str, str]:
        """Extract prediction and reasoning from message history.
        
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
            
            # Try <json> tags first
            extracted = _extract_jsons(full_text)
            if extracted:
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = str(last_json["response"]).strip()
                if "reasoning" in last_json:
                    reasoning = last_json["reasoning"]
                return prediction, reasoning
            
            # Try markdown code blocks
            md_json = _extract_json_from_markdown(full_text)
            if md_json:
                if "response" in md_json:
                    prediction = str(md_json["response"]).strip()
                if "reasoning" in md_json:
                    reasoning = md_json["reasoning"]
                return prediction, reasoning
            
            # Fallback: try to find any JSON-like structure with response field
            json_match = re.search(r'\{[^{}]*"response"[^{}]*\}', full_text)
            if json_match:
                try:
                    fallback = json.loads(json_match.group())
                    prediction = str(fallback.get("response", "None")).strip()
                except json.JSONDecodeError:
                    pass
            
            # Last resort: look for grade patterns in text
            if prediction == "None":
                # Look for explicit grade patterns
                grade_patterns = [
                    r'(?:final\s+)?(?:grade|score|mark)\s*:?\s*([0-7])\b',
                    r'\bgrade\s+(?:is|of|equals?)\s*:?\s*([0-7])\b',
                    r'\b(?:award|assign|give)\s*:?\s*([0-7])\s*(?:points?)?\b',
                    r'\b([0-7])\s*(?:points?|out\s+of\s+7)\b',
                ]
                for pattern in grade_patterns:
                    grade_match = re.search(pattern, full_text, re.IGNORECASE)
                    if grade_match:
                        prediction = grade_match.group(1)
                        break
                    
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
        
        # Final validation: ensure grade is within valid range
        if is_valid:
            try:
                grade_val = int(validated_grade)
                if grade_val < 0 or grade_val > 7:
                    is_valid = False
            except (ValueError, TypeError):
                is_valid = False
        
        if not is_valid:
            self.log_fn(f"Could not extract valid grade, returning 'None'")
            validated_grade = "None"

        return str(validated_grade), msg_history
