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
    Also handles markdown-style ```json blocks.
    """
    results = []
    search_from = 0
    
    # First try <json>...</json> tags
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
            # Try to fix common JSON issues
            try:
                # Remove trailing commas before closing braces/brackets
                fixed = re.sub(r',(\s*[}\]])', r'\1', inner)
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                continue
    
    # Also try markdown code blocks with json tag
    md_search_from = 0
    while True:
        start = text.find("```json", md_search_from)
        if start == -1:
            break
        end = text.find("```", start + 7)
        if end == -1:
            break
        inner = text[start + 7:end].strip()
        md_search_from = end + 3
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # Try to fix common JSON issues
            try:
                fixed = re.sub(r',(\s*[}\]])', r'\1', inner)
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                continue
    
    return results or None


def _extract_json_from_markdown(text: str) -> dict | None:
    """Extract JSON from markdown code blocks (```json ... ```).
    
    Enhanced to handle various code block formats and common JSON errors.
    """
    # Try ```json ... ``` blocks
    pattern = r'```json\s*(.*?)\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            # Try to fix common JSON issues
            try:
                fixed = re.sub(r',(\s*[}\]])', r'\1', match.strip())
                return json.loads(fixed)
            except json.JSONDecodeError:
                continue
    
    # Also try plain ``` blocks (without json tag)
    plain_pattern = r'```\s*(\{[\s\S]*?\})\s*```'
    plain_matches = re.findall(plain_pattern, text, re.DOTALL)
    for match in plain_matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            try:
                fixed = re.sub(r',(\s*[}\]])', r'\1', match.strip())
                return json.loads(fixed)
            except json.JSONDecodeError:
                continue
    
    return None


def _validate_grade(prediction: str, grading_guidelines: str) -> tuple[str, bool]:
    """Validate that the extracted grade is reasonable.
    
    Simplified validation that always returns a clean numeric grade (0-7)
    or "None" if invalid. Removes verbose text prefixes for cleaner output.
    
    Returns:
        (validated_grade, is_valid) where validated_grade is "0"-"7" or "None"
    """
    if not prediction or prediction == "None":
        return "None", False
    
    pred_clean = prediction.strip()
    pred_lower = pred_clean.lower()
    
    # Direct numeric match (0-7 for IMO problems) - most common case
    if pred_clean.isdigit():
        grade = int(pred_clean)
        if 0 <= grade <= 7:
            return str(grade), True
        return "None", False
    
    # Extract numeric grade from text patterns
    # Check for "X out of 7" or "X/7" patterns
    out_of_match = re.search(r'([0-7])\s*(?:out\s+of|/)\s*7', pred_lower)
    if out_of_match:
        return out_of_match.group(1), True
    
    # Check for partial credit with explicit number
    partial_match = re.search(r'partial.*?([0-7])', pred_lower)
    if partial_match:
        return partial_match.group(1), True
    
    # Check for standalone numeric grades in text
    numeric_match = re.search(r'\b([0-7])\b', pred_clean)
    if numeric_match:
        return numeric_match.group(1), True
    
    # Check for full credit patterns -> 7
    full_patterns = ['full credit', 'full marks', 'complete', 'perfect', '7/7', 'full score']
    if any(p in pred_lower for p in full_patterns):
        return "7", True
    
    # Check for "correct" without modifiers -> 7
    if 'correct' in pred_lower and 'partial' not in pred_lower and 'incorrect' not in pred_lower:
        return "7", True
    
    # Check for zero/incorrect patterns -> 0
    zero_patterns = ['zero', 'no credit', '0/7', 'none', 'incorrect', 'wrong', 'invalid', 'empty']
    if any(p in pred_lower for p in zero_patterns):
        return "0", True
    
    # Check for partial without number -> default to 3
    if 'partial' in pred_lower:
        return "3", True
    
    # If no clear grade found, return None
    return "None", False


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
        
        # Calculate approximate length for context
        student_len = len(student_answer) if student_answer else 0
        solution_len = len(solution) if solution else 0

        return f"""You are an expert IMO (International Mathematical Olympiad) grader with deep knowledge of mathematical problem-solving and competition grading standards.

Your task is to evaluate a student's solution to a mathematical problem and assign an appropriate grade.

## Problem Domain
{domain}

## Problem Statement
{problem}

## Official Solution (Length: ~{solution_len} chars)
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer (Length: ~{student_len} chars)
{student_answer}

## IMO Grading Scale Reference
- 7 points: Complete, correct solution with proper reasoning
- 6 points: Minor flaw in an otherwise correct solution
- 5 points: Significant progress with one gap or error
- 4 points: Multiple gaps but substantial progress
- 3 points: Some meaningful progress toward solution
- 2 points: Limited progress, some correct ideas
- 1 point: Minimal progress, minor relevant observation
- 0 points: No meaningful progress or completely wrong

## Instructions

1. **Analyze**: Carefully read the student's answer and compare it to the official solution.
2. **Identify**: Note any errors, missing steps, creative alternative approaches, or partial progress.
3. **Evaluate**: Consider the grading guidelines and the IMO scale above.
4. **Decide**: Assign a grade from 0-7 based on the student's demonstrated understanding and progress.
5. **Format**: Provide your detailed reasoning, then give the final grade as a single number (0-7).

Respond ONLY in the following JSON format:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis of the student's answer, comparing to the official solution, identifying errors or gaps, and explaining your evaluation...",
    "response": "X"
}}
</json>

Where X is a single digit from 0 to 7 representing the final grade. The "response" field must contain ONLY the numeric grade (0-7), nothing else."""

    def _extract_prediction(self, msg_history: list[dict]) -> tuple[str, str]:
        """Extract prediction and reasoning from message history.
        
        Simplified extraction focusing on the most reliable patterns.
        
        Returns:
            (prediction, reasoning)
        """
        prediction = "None"
        reasoning = ""
        
        try:
            if not msg_history:
                return prediction, reasoning
                
            last_msg = msg_history[-1].get("text", "") if isinstance(msg_history[-1], dict) else ""
            
            # Try <json> tags first (most reliable)
            extracted = _extract_jsons(last_msg)
            if extracted:
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = str(last_json["response"]).strip()
                if "reasoning" in last_json:
                    reasoning = str(last_json["reasoning"])[:500]
                return prediction, reasoning
            
            # Try markdown code blocks
            md_json = _extract_json_from_markdown(last_msg)
            if md_json:
                if "response" in md_json:
                    prediction = str(md_json["response"]).strip()
                if "reasoning" in md_json:
                    reasoning = str(md_json["reasoning"])[:500]
                return prediction, reasoning
            
            # Try to find any JSON-like object with grade/response/score
            json_match = re.search(r'\{[^{}]*"(?:response|grade|score)"[^{}]*\}', last_msg, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    for key in ["response", "grade", "score"]:
                        if key in parsed:
                            prediction = str(parsed[key]).strip()
                            break
                    if "reasoning" in parsed:
                        reasoning = str(parsed["reasoning"])[:500]
                    if prediction != "None":
                        return prediction, reasoning
                except json.JSONDecodeError:
                    pass
            
            # Look for explicit grade declarations in text
            text_patterns = [
                r'(?:final\s+)?(?:grade|score)\s*:?\s*["\']?([0-7])["\']?',
                r'(?:award|assign)\s*:?\s*["\']?([0-7])["\']?\s*(?:points?)?',
            ]
            for pattern in text_patterns:
                match = re.search(pattern, last_msg, re.IGNORECASE)
                if match:
                    prediction = match.group(1).strip()
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

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"LLM call failed: {e}")
            return "None", []

        # Extract prediction from message history
        prediction, reasoning = self._extract_prediction(msg_history)
        
        # Validate the grade
        validated_grade, is_valid = _validate_grade(prediction, "")
        
        # Log extraction results
        if reasoning:
            self.log_fn(f"Reasoning: {reasoning[:200]}...")
        self.log_fn(f"Extracted: {prediction}, Validated: {validated_grade}, Valid: {is_valid}")
        
        # Fallback extraction from full response if needed
        if not is_valid and response:
            # Try to find any numeric grade (0-7) in the response
            numeric_matches = re.findall(r'\b([0-7])\b', response)
            if numeric_matches:
                validated_grade = numeric_matches[-1]
                is_valid = True
                self.log_fn(f"Fallback: Found grade {validated_grade} in response")
            else:
                # Use content-based heuristic
                student_answer = inputs.get("student_answer", "")
                if not student_answer or len(student_answer.strip()) < 10:
                    validated_grade = "0"
                    self.log_fn("Fallback: Empty answer, defaulting to 0")
                else:
                    validated_grade = "3"
                    self.log_fn("Fallback: Uncertain grade, defaulting to 3")
                is_valid = True

        return str(validated_grade), msg_history
