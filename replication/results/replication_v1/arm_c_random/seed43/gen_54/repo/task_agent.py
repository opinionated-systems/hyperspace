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
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
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
    """Validate that the extracted grade is a valid IMO grade (0-7).
    
    Strict validation for IMO 0-7 point scale.
    
    Returns:
        (validated_grade, is_valid)
    """
    if not prediction or prediction == "None":
        return "None", False
    
    pred_clean = prediction.strip()
    
    # Direct single digit check (most common case with new prompt)
    if pred_clean in "01234567" and len(pred_clean) == 1:
        return pred_clean, True
    
    # Check for numeric grades (0-7 for IMO problems) anywhere in the text
    numeric_match = re.search(r'\b([0-7])\b', pred_clean)
    if numeric_match:
        return numeric_match.group(1), True
    
    # Check for fractional grades like "3/7" or "5 / 7"
    fractional_match = re.search(r'\b([0-7])\s*/\s*7\b', pred_clean)
    if fractional_match:
        return fractional_match.group(1), True
    
    pred_lower = pred_clean.lower()
    
    # Check for full credit patterns -> 7
    full_patterns = [
        r'\bfull\s*(?:credit|points?|score|mark)?\b',
        r'\bcomplete\s*(?:solution|answer|credit)?\b',
        r'\ball\s*(?:points?|credit|marks?|score)?\b',
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
        r'\bno\s*progress\b',
    ]
    for pattern in zero_patterns:
        if re.search(pattern, pred_lower):
            return "0", True
    
    # Check for partial credit patterns
    partial_match = re.search(r'partial\s*(?:credit)?\s*:?\s*([0-7])', pred_lower)
    if partial_match:
        return partial_match.group(1), True
    
    # If prediction is a digit but outside 0-7 range, it's invalid
    if pred_clean.isdigit():
        grade_val = int(pred_clean)
        if 0 <= grade_val <= 7:
            return pred_clean, True
        else:
            return pred_clean, False
    
    # If no clear grade found, mark as invalid
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

## Instructions for Evaluation

1. **Initial Assessment**: Read the student's answer completely before forming any judgments.

2. **Step-by-Step Analysis**: 
   - Compare each step of the student's solution against the official solution
   - Identify correct mathematical statements and valid reasoning
   - Note any errors, gaps, or logical flaws
   - Recognize creative alternative approaches that may differ from the official solution but are mathematically valid

3. **IMO Grading Scale (0-7 points)**:
   - **7 points**: Complete, correct solution with clear reasoning
   - **6 points**: Minor flaw or omission in an otherwise correct solution
   - **5 points**: Significant progress with one major gap or error
   - **3-4 points**: Partial progress with multiple gaps or errors
   - **1-2 points**: Limited progress, some relevant ideas
   - **0 points**: No meaningful progress or completely incorrect

4. **Considerations**:
   - IMO problems often have multiple valid solution paths
   - Partial credit should reflect the mathematical value of the work shown
   - Computational errors may receive partial credit if the approach is correct
   - Missing crucial steps or logical gaps reduce the grade proportionally

5. **Output Format**:
   - Provide detailed reasoning in the "reasoning" field
   - Give ONLY the numeric grade (0-7) in the "response" field
   - Do not include explanations in the response field

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis of the student's answer, including comparison with official solution, identification of errors/gaps, and justification for the grade...",
    "response": "X"
}}
</json>

Where X is a single digit from 0 to 7 representing the IMO grade. The "response" field must contain ONLY this digit, nothing else."""

    def _extract_prediction(self, msg_history: list[dict]) -> tuple[str, str]:
        """Extract prediction and reasoning from message history.
        
        Returns:
            (prediction, reasoning)
        """
        prediction = "None"
        reasoning = ""
        
        try:
            if not msg_history:
                return prediction, reasoning
                
            last_msg = msg_history[-1].get("text", "") if isinstance(msg_history[-1], dict) else ""
            
            # Try <json> tags first (primary format)
            extracted = _extract_jsons(last_msg)
            if extracted:
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = str(last_json["response"]).strip()
                if "reasoning" in last_json:
                    reasoning = str(last_json["reasoning"])
                return prediction, reasoning
            
            # Try markdown code blocks
            md_json = _extract_json_from_markdown(last_msg)
            if md_json:
                if "response" in md_json:
                    prediction = str(md_json["response"]).strip()
                if "reasoning" in md_json:
                    reasoning = str(md_json["reasoning"])
                return prediction, reasoning
            
            # Fallback: try to find any JSON-like structure with response field
            # Use a more robust pattern that handles nested braces
            json_pattern = r'\{[^{}]*"response"[^{}]*\}'
            json_matches = re.findall(json_pattern, last_msg)
            for json_match in json_matches:
                try:
                    fallback = json.loads(json_match)
                    if "response" in fallback:
                        prediction = str(fallback["response"]).strip()
                        if "reasoning" in fallback:
                            reasoning = str(fallback["reasoning"])
                        break
                except json.JSONDecodeError:
                    continue
            
            # Last resort: look for grade patterns in text
            if prediction == "None":
                # Look for "Grade: X" or "Final grade: X" patterns
                grade_patterns = [
                    r'(?:grade|score|mark|final grade)\s*:?\s*([0-7])',
                    r'(?:grade|score|mark)\s+of\s+([0-7])',
                    r'(?:grade|score|mark)\s+is\s+([0-7])',
                    r'(?:^|\s)([0-7])\s*(?:points?|/\s*7)',
                ]
                for pattern in grade_patterns:
                    grade_match = re.search(pattern, last_msg, re.IGNORECASE)
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
            # Strategy 1: Try to find any numeric grade 0-7 in the full response
            if response:
                numeric_match = re.search(r'\b([0-7])\b', response)
                if numeric_match:
                    validated_grade = numeric_match.group(1)
                    is_valid = True
                    self.log_fn(f"Fallback 1: Found grade in full response: {validated_grade}")
            
            # Strategy 2: Look for grade patterns in reasoning if available
            if not is_valid and reasoning:
                numeric_match = re.search(r'\b([0-7])\b', reasoning)
                if numeric_match:
                    validated_grade = numeric_match.group(1)
                    is_valid = True
                    self.log_fn(f"Fallback 2: Found grade in reasoning: {validated_grade}")
            
            # Strategy 3: Check for text patterns indicating grade
            if not is_valid and response:
                text_lower = response.lower()
                if any(word in text_lower for word in ['full credit', 'complete', 'perfect', 'correct solution']):
                    validated_grade = "7"
                    is_valid = True
                    self.log_fn(f"Fallback 3: Detected full credit pattern -> 7")
                elif any(word in text_lower for word in ['no credit', 'zero', 'incorrect', 'wrong', 'no progress']):
                    validated_grade = "0"
                    is_valid = True
                    self.log_fn(f"Fallback 3: Detected no credit pattern -> 0")
        
        # Final validation check
        if not is_valid:
            self.log_fn(f"Warning: Could not extract valid grade. Raw prediction: {prediction}")
            # Return the original prediction even if invalid, to avoid losing information
            validated_grade = prediction if prediction != "None" else "0"

        return str(validated_grade), msg_history
