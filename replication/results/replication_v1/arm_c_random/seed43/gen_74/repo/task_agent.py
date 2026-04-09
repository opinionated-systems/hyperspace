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
            # Try to find valid JSON within the content
            try:
                # Look for JSON object pattern
                json_start = inner.find('{')
                json_end = inner.rfind('}')
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    results.append(json.loads(inner[json_start:json_end+1]))
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
            # Try to find valid JSON within the content
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
    # Look for standalone digits or digits at start/end of string
    numeric_match = re.search(r'(?:^|\s)([0-7])(?:\s|$|[,;])', pred_clean)
    if numeric_match:
        return numeric_match.group(1), True
    
    # Also try simple digit search as fallback
    simple_match = re.search(r'\b([0-7])\b', pred_clean)
    if simple_match:
        return simple_match.group(1), True
    
    # Check for full credit patterns -> 7
    full_patterns = [
        r'\bfull\s*(?:credit|points?|score|marks?)?\b',
        r'\bcomplete\s*(?:solution|answer|credit)?\b',
        r'\ball\s*(?:points?|credit|marks?)?\b',
        r'\bperfect\s*(?:score|solution)?\b',
        r'\bcorrect\s*(?:solution|answer|complete)?\b',
        r'\bseven\s*(?:points?)?\b',
        r'\bmaximum\s*(?:score|points?|credit)?\b',
    ]
    for pattern in full_patterns:
        if re.search(pattern, pred_lower):
            return "7", True
    
    # Check for zero/no credit patterns -> 0
    zero_patterns = [
        r'\bno\s*(?:credit|points?|score|marks?|progress)?\b',
        r'\bzero\s*(?:credit|points?|score|marks?)?\b',
        r'\b0\s*(?:points?|credit|score|marks?)?\b',
        r'\bincorrect\s*(?:solution|answer|approach)?\b',
        r'\bwrong\s*(?:solution|answer|approach)?\b',
        r'\bnone\b',
        r'\bno\s*solution\b',
        r'\bempty\s*(?:answer|response)?\b',
    ]
    for pattern in zero_patterns:
        if re.search(pattern, pred_lower):
            return "0", True
    
    # Check for partial credit patterns with specific numbers
    partial_patterns = [
        (r'\bone\s*(?:point|mark|credit)?\b', "1"),
        (r'\btwo\s*(?:points?|marks?|credit)?\b', "2"),
        (r'\bthree\s*(?:points?|marks?|credit)?\b', "3"),
        (r'\bfour\s*(?:points?|marks?|credit)?\b', "4"),
        (r'\bfive\s*(?:points?|marks?|credit)?\b', "5"),
        (r'\bsix\s*(?:points?|marks?|credit)?\b', "6"),
    ]
    for pattern, grade in partial_patterns:
        if re.search(pattern, pred_lower):
            return grade, True
    
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
   - 3-4 points: Partial progress with some correct elements
   - 1-2 points: Minimal progress or some relevant ideas
   - 0 points: No meaningful progress or completely incorrect

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
- Example of CORRECT output: "response": "5"
- Example of INCORRECT output: "response": "Grade: 5 points" or "response": "5/7"
"""

    def _extract_prediction(self, msg_history: list[dict]) -> tuple[str, str]:
        """Extract prediction and reasoning from message history.
        
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
            
            # Try <json> tags first
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
            json_match = re.search(r'\{[^{}]*"response"[^{}]*\}', last_msg)
            if json_match:
                try:
                    fallback = json.loads(json_match.group())
                    prediction = str(fallback.get("response", "None")).strip()
                    if "reasoning" in fallback:
                        reasoning = str(fallback["reasoning"])
                except json.JSONDecodeError:
                    pass
            
            # Try to find JSON with nested content
            if prediction == "None":
                # Look for JSON blocks that might have nested structures
                json_blocks = re.findall(r'\{.*"response".*\}', last_msg, re.DOTALL)
                for block in json_blocks:
                    try:
                        parsed = json.loads(block)
                        if "response" in parsed:
                            prediction = str(parsed["response"]).strip()
                            if "reasoning" in parsed:
                                reasoning = str(parsed["reasoning"])
                            break
                    except json.JSONDecodeError:
                        continue
            
            # Last resort: look for grade patterns in text
            if prediction == "None":
                # Look for "Grade: X" or "Final grade: X" patterns
                grade_match = re.search(r'(?:grade|score|mark|final grade|final score)\s*:?\s*([0-7])', last_msg, re.IGNORECASE)
                if grade_match:
                    prediction = grade_match.group(1)
                    
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

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"LLM call failed: {e}")
            return "None", []

        # Extract prediction with enhanced extraction
        prediction, reasoning = self._extract_prediction(msg_history)
        
        # Validate the grade
        validated_grade, is_valid = _validate_grade(prediction, grading_guidelines)
        
        # Log the reasoning and validation result
        if reasoning:
            self.log_fn(f"Reasoning: {reasoning[:200]}...")
        self.log_fn(f"Extracted grade: {prediction}, Validated: {validated_grade}, Is valid: {is_valid}")
        
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
                    r'(?:grade|score|mark|final grade|final score)\s*:?\s*([0-7])',
                    r'\bfull\s*(?:credit|points?|score)?\b',
                    r'\bcorrect\s*(?:solution|answer)?\b',
                    r'\bno\s*(?:credit|points?|score|marks?)?\b',
                    r'\bzero\s*(?:credit|points?|score|marks?)?\b',
                    r'\bincorrect\s*(?:solution|answer)?\b',
                ]
                for pattern in grade_patterns:
                    match = re.search(pattern, response, re.IGNORECASE)
                    if match:
                        if 'full' in pattern or 'correct' in pattern:
                            validated_grade = "7"
                        elif 'no' in pattern or 'zero' in pattern or 'incorrect' in pattern:
                            validated_grade = "0"
                        else:
                            validated_grade = match.group(1)
                        is_valid = True
                        self.log_fn(f"Pattern-based fallback found grade: {validated_grade}")
                        break

        return str(validated_grade), msg_history
