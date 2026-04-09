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
    """Validate that the extracted grade is reasonable.
    
    Enhanced validation with support for:
    - IMO 0-7 point scale
    - Partial credit notation
    - Fractional grades (e.g., 3/7)
    - Descriptive evaluations
    - Multi-language grade descriptions
    - Range notation (e.g., "5-6 points")
    
    Returns:
        (validated_grade, is_valid)
    """
    if not prediction or prediction == "None":
        return "None", False
    
    pred_clean = prediction.strip()
    pred_lower = pred_clean.lower()
    
    # Check for fractional grades like "3/7" or "5 / 7"
    fractional_match = re.search(r'\b([0-7])\s*/\s*7\b', pred_clean)
    if fractional_match:
        return fractional_match.group(1), True
    
    # Check for range notation like "5-6" or "5 to 6" - take the average or lower bound
    range_match = re.search(r'\b([0-6])\s*(?:-|to|~|–)\s*([0-7])\b', pred_clean)
    if range_match:
        lower = int(range_match.group(1))
        upper = int(range_match.group(2))
        # Return the lower bound to be conservative
        return str(lower), True
    
    # Check for numeric grades (0-7 for IMO problems)
    numeric_match = re.search(r'\b([0-7])\b', pred_clean)
    if numeric_match:
        return numeric_match.group(1), True
    
    # Check for spelled-out numbers
    number_words = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7',
        '0': '0', '1': '1', '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7'
    }
    for word, digit in number_words.items():
        if re.search(rf'\b{word}\b', pred_lower):
            return digit, True
    
    # Check for partial credit patterns with more variations
    partial_patterns = [
        r'partial\s*(?:credit)?\s*:?\s*([0-7])',
        r'partial\s*([0-7])\s*(?:points?)?',
        r'([0-7])\s*(?:points?)?\s*partial',
        r'partially\s*(?:correct|right)\s*(?:worth\s*)?([0-7])?',
        r'some\s*(?:credit|points?)\s*:?\s*([0-7])?',
    ]
    for pattern in partial_patterns:
        partial_match = re.search(pattern, pred_lower)
        if partial_match:
            grade = partial_match.group(1) if partial_match.group(1) else "3"  # Default partial
            return f"Partial credit: {grade}", True
    
    # Check for full credit patterns
    full_patterns = [
        r'\bfull\s*(?:credit|points?|score|marks?)?\b',
        r'\bcomplete\s*(?:solution|answer|credit)?\b',
        r'\ball\s*(?:points?|credit|marks?)?\b',
        r'\bperfect\s*(?:score|solution|answer)?\b',
        r'\bmax(?:imum)?\s*(?:score|points?|credit)?\b',
        r'\bentirely\s*(?:correct|right)\b',
        r'\bfully\s*(?:correct|solved)\b',
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
        r'\bno\s*(?:solution|progress|work)\b',
        r'\bempty\s*(?:answer|response)?\b',
        r'\bno\s*marks?\b',
    ]
    for pattern in zero_patterns:
        if re.search(pattern, pred_lower):
            return "0", True
    
    # Check for high grades (6-7)
    high_patterns = [
        r'\b(?:almost|nearly)\s*(?:perfect|full)\b',
        r'\bminor\s*(?:error|mistake)\b',
        r'\bsmall\s*(?:issue|problem)\b',
    ]
    for pattern in high_patterns:
        if re.search(pattern, pred_lower):
            return "6", True
    
    # Check for medium grades (3-5)
    medium_patterns = [
        r'\b(?:significant|substantial)\s*(?:progress|work)\b',
        r'\b(?:half|some)\s*(?:correct|right)\b',
        r'\bincomplete\s*(?:solution|proof)\b',
        r'\b(?:good|decent)\s*(?:attempt|try)\b',
    ]
    for pattern in medium_patterns:
        if re.search(pattern, pred_lower):
            return "4", True
    
    # Check for low grades (1-2)
    low_patterns = [
        r'\b(?:minimal|little)\s*(?:progress|work)\b',
        r'\b(?:few|minor)\s*(?:steps|ideas)\b',
        r'\b(?:slight|small)\s*(?:insight|idea)\b',
    ]
    for pattern in low_patterns:
        if re.search(pattern, pred_lower):
            return "2", True
    
    # Check for other valid grade keywords
    valid_keywords = ['correct', 'partial', 'n/a', 'not applicable', 'incomplete']
    for keyword in valid_keywords:
        if keyword in pred_lower:
            if keyword == 'correct':
                return "7", True
            elif keyword == 'partial':
                return "3", True
            elif keyword == 'incomplete':
                return "2", True
            elif keyword in ['n/a', 'not applicable']:
                return "0", True  # N/A typically means no credit
            return pred_clean, True
    
    # Check for explicit "no credit" or "zero" patterns
    no_credit_patterns = [
        r'\bno\s*credit\b',
        r'\bno\s*points?\b',
        r'\bzero\s*(?:points?|credit|score)?\b',
        r'\b0\s*(?:points?|credit|score|marks?)?\b',
    ]
    for pattern in no_credit_patterns:
        if re.search(pattern, pred_lower):
            return "0", True
    
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
3. Consider the grading guidelines carefully - IMO problems are typically graded 0-7 points.
4. Provide your reasoning before giving the final grade.
5. The final grade should be a clear numeric value (0-7) or descriptive evaluation.

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed analysis of the student's answer...",
    "response": "Your final grade/evaluation (e.g., '7', '6', 'Partial credit: 3', 'Incorrect', etc.)"
}}
</json>

The "response" field should contain only the final grade/evaluation, while "reasoning" contains your full analysis."""

    def _extract_prediction(self, msg_history: list[dict]) -> tuple[str, str]:
        """Extract prediction and reasoning from message history.
        
        Enhanced extraction with multiple fallback strategies and better
        handling of edge cases like nested JSON, malformed responses,
        and various grade formats. Includes improved logging for debugging.
        
        Returns:
            (prediction, reasoning)
        """
        prediction = "None"
        reasoning = ""
        
        try:
            last_msg = msg_history[-1]["text"] if msg_history else ""
            
            # Strategy 1: Try <json> tags first (most reliable)
            extracted = _extract_jsons(last_msg)
            if extracted:
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = str(last_json["response"]).strip()
                if "reasoning" in last_json:
                    reasoning = last_json["reasoning"]
                if prediction != "None":
                    return prediction, reasoning
            
            # Strategy 2: Try markdown code blocks
            md_json = _extract_json_from_markdown(last_msg)
            if md_json:
                if "response" in md_json:
                    prediction = str(md_json["response"]).strip()
                if "reasoning" in md_json:
                    reasoning = md_json["reasoning"]
                if prediction != "None":
                    return prediction, reasoning
            
            # Strategy 3: Try to find JSON with nested braces using balanced brace counting
            # This handles cases where the regex \{[^}]*"response"[^}]*\} fails due to nested braces
            def find_json_with_field(text: str, field: str) -> str | None:
                """Find JSON object containing specific field using brace balancing."""
                field_pattern = f'"{field}"'
                idx = text.find(field_pattern)
                while idx != -1:
                    # Find the opening brace before this field
                    brace_start = text.rfind('{', 0, idx)
                    if brace_start == -1:
                        idx = text.find(field_pattern, idx + 1)
                        continue
                    
                    # Count braces to find matching closing brace
                    brace_count = 0
                    for i, char in enumerate(text[brace_start:], start=brace_start):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                return text[brace_start:i+1]
                    idx = text.find(field_pattern, idx + 1)
                return None
            
            json_str = find_json_with_field(last_msg, "response")
            if json_str:
                try:
                    fallback = json.loads(json_str)
                    prediction = str(fallback.get("response", "None")).strip()
                    reasoning = fallback.get("reasoning", "")
                    if prediction != "None":
                        return prediction, reasoning
                except json.JSONDecodeError:
                    pass
            
            # Strategy 4: Look for explicit grade/score patterns in text
            if prediction == "None":
                # Pattern: "Grade: X", "Final grade: X", "Score: X"
                grade_patterns = [
                    r'(?:final\s+)?(?:grade|score|mark)s?\s*:?\s*([0-7])\s*(?:/\s*7)?\b',
                    r'(?:grade|score|mark)s?\s*:?\s*(full|partial|zero|none|incorrect|correct)',
                    r'\b(grade|score)\s*:?\s*([0-7])\b',
                ]
                for pattern in grade_patterns:
                    grade_match = re.search(pattern, last_msg, re.IGNORECASE)
                    if grade_match:
                        # Get the last group that contains the value
                        prediction = grade_match.group(grade_match.lastindex)
                        break
            
            # Strategy 5: Look for standalone numeric grades (0-7) with context
            if prediction == "None":
                # Look for numbers 0-7 that appear to be grades (near keywords)
                context_pattern = r'(?:points?|credit|score|grade|mark|worth|value).*?\b([0-7])\b|\b([0-7])\b.*?(?:points?|credit|score|grade|mark)'
                context_match = re.search(context_pattern, last_msg, re.IGNORECASE | re.DOTALL)
                if context_match:
                    prediction = context_match.group(1) or context_match.group(2)
            
            # Strategy 6: Extract any reasoning found even if grade extraction failed
            if not reasoning:
                # Look for reasoning field in various formats
                reasoning_patterns = [
                    r'"reasoning"\s*:\s*"([^"]*)"',
                    r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"',
                    r'reasoning[\s\w]*:\s*(.+?)(?:\n\n|\Z|grade|score|mark)',
                ]
                for pattern in reasoning_patterns:
                    reasoning_match = re.search(pattern, last_msg, re.IGNORECASE | re.DOTALL)
                    if reasoning_match:
                        reasoning = reasoning_match.group(1).strip()
                        break
            
            # Strategy 7: Fallback - take the last numeric grade 0-7 in the response
            # This is often the final conclusion grade when other strategies fail
            if prediction == "None":
                all_grades = re.findall(r'\b([0-7])\b', last_msg)
                if all_grades:
                    prediction = all_grades[-1]
                        
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
        
        # If grade is invalid, try multiple fallback extraction strategies
        if not is_valid and response:
            # Fallback 1: Try to find any numeric grade in the response
            numeric_match = re.search(r'\b([0-7])\b', response)
            if numeric_match:
                validated_grade = numeric_match.group(1)
                is_valid = True
                self.log_fn(f"Fallback 1 found grade: {validated_grade}")
            
            # Fallback 2: Look for grade in conclusion/summary sections
            if not is_valid:
                conclusion_patterns = [
                    r'(?:conclusion|summary|final|overall|therefore).*?\b([0-7])\b',
                    r'(?:in\s+conclusion|to\s+summarize|overall).*?\b([0-7])\b',
                ]
                for pattern in conclusion_patterns:
                    match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
                    if match:
                        validated_grade = match.group(1)
                        is_valid = True
                        self.log_fn(f"Fallback 2 found grade: {validated_grade}")
                        break
            
            # Fallback 3: Look for the last number 0-7 in the response (often the final grade)
            if not is_valid:
                all_grades = re.findall(r'\b([0-7])\b', response)
                if all_grades:
                    validated_grade = all_grades[-1]  # Take the last one
                    is_valid = True
                    self.log_fn(f"Fallback 3 found grade: {validated_grade}")
            
            # Fallback 4: Try to validate the raw response text
            if not is_valid:
                raw_grade, raw_valid = _validate_grade(response, grading_guidelines)
                if raw_valid:
                    validated_grade = raw_grade
                    is_valid = True
                    self.log_fn(f"Fallback 4 found grade: {validated_grade}")
        
        # Final fallback: if still invalid, return "0" (no credit) as conservative default
        if not is_valid:
            self.log_fn(f"All extraction methods failed. Using conservative default grade: 0")
            validated_grade = "0"

        return str(validated_grade), msg_history
