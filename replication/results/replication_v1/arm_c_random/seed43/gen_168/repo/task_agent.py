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

# IMO grading scale constants
MIN_GRADE = 0
MAX_GRADE = 7
VALID_GRADES = [str(i) for i in range(MIN_GRADE, MAX_GRADE + 1)]


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
        
        # Try direct JSON parsing first
        try:
            results.append(json.loads(inner))
            continue
        except json.JSONDecodeError:
            pass
        
        # Try to find valid JSON object within the content
        json_start = inner.find('{')
        json_end = inner.rfind('}')
        if json_start != -1 and json_end != -1 and json_end > json_start:
            try:
                results.append(json.loads(inner[json_start:json_end+1]))
                continue
            except json.JSONDecodeError:
                pass
        
        # Try to find valid JSON array within the content
        arr_start = inner.find('[')
        arr_end = inner.rfind(']')
        if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
            try:
                results.append(json.loads(inner[arr_start:arr_end+1]))
            except json.JSONDecodeError:
                continue
    
    return results or None


def _extract_json_from_markdown(text: str) -> dict | None:
    """Extract JSON from markdown code blocks (```json ... ``` or ``` ... ```)."""
    # Try ```json ... ``` blocks first
    pattern_json = r'```json\s*(.*?)\s*```'
    matches = re.findall(pattern_json, text, re.DOTALL)
    
    # If no json-specific blocks, try generic code blocks
    if not matches:
        pattern_generic = r'```\s*(.*?)\s*```'
        matches = re.findall(pattern_generic, text, re.DOTALL)
    
    for match in matches:
        match_stripped = match.strip()
        
        # Try direct JSON parsing first
        try:
            parsed = json.loads(match_stripped)
            if isinstance(parsed, dict):
                return parsed
            continue
        except json.JSONDecodeError:
            pass
        
        # Try to find valid JSON object within the content
        json_start = match_stripped.find('{')
        json_end = match_stripped.rfind('}')
        if json_start != -1 and json_end != -1 and json_end > json_start:
            try:
                parsed = json.loads(match_stripped[json_start:json_end+1])
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
    
    return None


def _extract_grade_from_text(text: str) -> str | None:
    """Extract a grade (0-7) from free-form text using various patterns.
    
    Args:
        text: The text to extract a grade from
    
    Returns:
        The grade as a string if found, None otherwise
    """
    if not text:
        return None
    
    text_lower = text.lower()
    grade_range = f"{MIN_GRADE}-{MAX_GRADE}"
    
    # Pattern 1: Explicit grade/score statements
    grade_patterns = [
        rf'(?:grade|score|mark|final grade|final score)\s*[:=]\s*([{grade_range}])',
        rf'(?:grade|score|mark)\s+of\s+([{grade_range}])',
        rf'(?:grade|score|mark)\s+is\s+([{grade_range}])',
        rf'(?:assigned|given|awarded)\s+(?:a\s+)?(?:grade|score|mark)\s+of\s+([{grade_range}])',
        rf'(?:worth|deserves|earns?)\s+([{grade_range}])\s*(?:points?)?',
    ]
    
    for pattern in grade_patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1)
    
    # Pattern 2: Full credit patterns -> MAX_GRADE (7)
    full_credit_patterns = [
        r'\bfull\s*(?:credit|points?|score|marks?)\b',
        r'\bcomplete\s*(?:solution|answer|credit)\b',
        r'\ball\s*(?:points?|credit|marks?)\b',
        r'\bperfect\s*(?:score|solution|answer)\b',
        r'\bcorrect\s*(?:solution|answer)\b',
        r'\bsolved\s+correctly\b',
        r'\bvalid\s+(?:solution|proof)\b',
    ]
    
    for pattern in full_credit_patterns:
        if re.search(pattern, text_lower):
            return str(MAX_GRADE)
    
    # Pattern 3: Zero/no credit patterns -> MIN_GRADE (0)
    zero_credit_patterns = [
        r'\bno\s*(?:credit|points?|score|marks?)\b',
        r'\bzero\s*(?:credit|points?|score|marks?)\b',
        r'\bincorrect\s*(?:solution|answer)\b',
        r'\bwrong\s*(?:solution|answer)\b',
        r'\bno\s+solution\b',
        r'\bempty\s*(?:answer|response)?\b',
        r'\bblank\b',
    ]
    
    for pattern in zero_credit_patterns:
        if re.search(pattern, text_lower):
            return str(MIN_GRADE)
    
    # Pattern 4: Partial credit indicators
    partial_patterns = [
        r'\bpartial\s*(?:credit|points?|score)?\b',
        r'\bsome\s*(?:credit|points?|progress)\b',
        r'\bincomplete\s*(?:solution|answer)\b',
    ]
    
    has_partial = any(re.search(p, text_lower) for p in partial_patterns)
    
    # Pattern 5: Look for standalone valid grades
    # Only if we have some context that suggests it's a grade
    if has_partial or re.search(r'\b(?:grade|score|point|mark)\b', text_lower):
        # Find all valid grades in the text
        digits = re.findall(rf'\b([{grade_range}])\b', text)
        if digits:
            # Return the last digit found (usually the final grade)
            return digits[-1]
    
    return None


def _validate_grade(prediction: str, grading_guidelines: str) -> tuple[str, bool]:
    """Validate that the extracted grade is reasonable.
    
    Enhanced validation with strict support for IMO 0-7 point scale.
    Only accepts single digit grades 0-7 for consistency.
    
    Args:
        prediction: The predicted grade string to validate
        grading_guidelines: The grading guidelines (for context, currently unused)
    
    Returns:
        Tuple of (validated_grade, is_valid)
    """
    if not prediction or prediction == "None":
        return "None", False
    
    pred_clean = prediction.strip()
    pred_lower = pred_clean.lower()
    
    # Strict validation: only accept valid IMO grades
    if pred_clean in VALID_GRADES:
        return pred_clean, True
    
    # Check for fractional grades like "3/7" or "5 / 7" - extract just the numerator
    fractional_match = re.search(rf'\b([{MIN_GRADE}-{MAX_GRADE}])\s*/\s*7\b', pred_lower)
    if fractional_match:
        return fractional_match.group(1), True
    
    # Check for "X out of 7" or "X out of seven" patterns
    out_of_match = re.search(rf'\b([{MIN_GRADE}-{MAX_GRADE}])\s+out\s+of\s+(?:7|seven)\b', pred_lower)
    if out_of_match:
        return out_of_match.group(1), True
    
    # Check for numeric grades embedded in text
    numeric_match = re.search(rf'\b([{MIN_GRADE}-{MAX_GRADE}])\b', pred_clean)
    if numeric_match:
        return numeric_match.group(1), True
    
    # Check for full credit patterns -> MAX_GRADE (7)
    full_patterns = [
        r'\bfull\s*(?:credit|points?|score|marks?)?\b',
        r'\bcomplete\s*(?:solution|answer|credit)?\b',
        r'\ball\s*(?:points?|credit|marks?)?\b',
        r'\bperfect\s*(?:score|solution)?\b',
        r'\bcorrect\s*(?:solution|answer)?\b',
    ]
    for pattern in full_patterns:
        if re.search(pattern, pred_lower):
            return str(MAX_GRADE), True
    
    # Check for zero/no credit patterns -> MIN_GRADE (0)
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
            return str(MIN_GRADE), True
    
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

        return f"""You are an expert IMO (International Mathematical Olympiad) grader.

Your task is to evaluate a student's solution to a mathematical problem.

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

## Instructions

1. First, analyze the student's answer step by step. Compare it against the official solution.
2. Identify any errors, missing steps, or creative alternative approaches.
3. Consider the grading guidelines carefully - IMO problems are typically graded {MIN_GRADE}-{MAX_GRADE} points.
4. Provide your reasoning before giving the final grade.
5. The final grade must be a single numeric value from {MIN_GRADE} to {MAX_GRADE} (inclusive).

## Output Format

You MUST respond with a valid JSON object wrapped in <json> tags. The JSON must have exactly these two fields:

<json>
{{
    "reasoning": "Your detailed analysis of the student's answer, comparing it to the official solution and explaining your evaluation...",
    "response": "X"
}}
</json>

IMPORTANT:
- The "response" field MUST contain ONLY a single digit from {MIN_GRADE} to {MAX_GRADE} (e.g., "{MAX_GRADE}", "5", "{MIN_GRADE}")
- Do NOT include any other text, explanations, or formatting in the response field
- The "reasoning" field should contain your full analysis
- Valid grades are: {', '.join(VALID_GRADES)}
- {MAX_GRADE} = full credit (complete correct solution)
- {MIN_GRADE} = no credit (completely wrong or blank)
- 1-6 = partial credit based on progress made"""

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
            json_match = re.search(r'\{[^}]*"response"[^}]*\}', last_msg)
            if json_match:
                try:
                    fallback = json.loads(json_match.group())
                    prediction = str(fallback.get("response", "None")).strip()
                    if "reasoning" in fallback:
                        reasoning = str(fallback["reasoning"])
                except json.JSONDecodeError:
                    pass
            
            # Last resort: use the enhanced grade extraction from text
            if prediction == "None":
                grade = _extract_grade_from_text(last_msg)
                if grade:
                    prediction = grade
                    
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
            # Use the enhanced grade extraction function
            grade = _extract_grade_from_text(response)
            if grade:
                validated_grade = grade
                is_valid = True
                self.log_fn(f"Enhanced fallback extraction found grade: {validated_grade}")
            else:
                # Last resort: try simple numeric extraction
                numeric_match = re.search(r'\b([0-7])\b', response)
                if numeric_match:
                    validated_grade = numeric_match.group(1)
                    is_valid = True
                    self.log_fn(f"Simple numeric fallback found grade: {validated_grade}")

        return str(validated_grade), msg_history
