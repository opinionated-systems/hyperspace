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
    
    Returns:
        (validated_grade, is_valid)
    """
    if not prediction or prediction == "None":
        return "None", False
    
    pred_clean = prediction.strip()
    pred_lower = pred_clean.lower()
    
    # Direct single digit check (most reliable for IMO 0-7 scale)
    if len(pred_clean) == 1 and pred_clean.isdigit():
        grade_val = int(pred_clean)
        if 0 <= grade_val <= 7:
            return pred_clean, True
    
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
    ]
    for pattern in partial_patterns:
        partial_match = re.search(pattern, pred_lower)
        if partial_match:
            return partial_match.group(1), True
    
    # Check for full credit patterns
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
    
    # Check for zero/no credit patterns
    zero_patterns = [
        r'\bno\s*(?:credit|points?|score|marks?)?\b',
        r'\bzero\s*(?:credit|points?|score|marks?)?\b',
        r'\b0\s*(?:points?|credit|score|marks?)?\b',
        r'\bincorrect\s*(?:solution|answer)?\b',
        r'\bwrong\s*(?:solution|answer)?\b',
        r'\bno\s*progress\b',
        r'\bno\s*meaningful\b',
    ]
    for pattern in zero_patterns:
        if re.search(pattern, pred_lower):
            return "0", True
    
    # Check for other valid grade keywords
    valid_keywords = ['partial', 'n/a', 'not applicable']
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

        # Extract key grading criteria if available
        criteria_section = ""
        if grading_guidelines:
            # Try to extract point values from grading guidelines
            point_hints = []
            for line in grading_guidelines.split('\n'):
                line = line.strip()
                if any(x in line.lower() for x in ['point', 'mark', 'score', 'credit', 'grade']):
                    if any(c.isdigit() for c in line):
                        point_hints.append(line)
            if point_hints:
                criteria_section = "\n## Key Grading Criteria\n" + '\n'.join(point_hints[:5])

        return f"""You are an expert IMO (International Mathematical Olympiad) grader with years of experience evaluating mathematical proofs.

Your task is to evaluate a student's solution to a mathematical problem with precision and consistency.

## Problem Domain
{domain}

## Problem Statement
{problem}

## Official Solution
{solution}

## Grading Guidelines
{grading_guidelines}{criteria_section}

## Student's Answer
{student_answer}

## Instructions

1. **Analysis Phase**: Carefully analyze the student's answer step by step. Compare it against the official solution.
   - Check if the student understood the problem correctly
   - Verify each mathematical claim and step
   - Identify any logical gaps or errors
   - Note any creative alternative approaches

2. **Grading Phase**: Based on the grading guidelines, assign a numeric grade from 0-7.
   - IMO problems use a 0-7 point scale
   - 7 = Complete, correct solution
   - 6 = Minor flaw in an otherwise correct solution
   - 3-5 = Partial progress with significant gaps
   - 1-2 = Minor progress
   - 0 = No meaningful progress or completely wrong

3. **Output Format**: Respond ONLY in the following JSON format:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis of the student's answer, including specific errors found and positive aspects noted...",
    "response": "X"
}}
</json>

IMPORTANT:
- The "response" field MUST contain ONLY a single digit from 0-7 (no quotes, no explanations)
- The "reasoning" field should contain your full analysis
- Do not include any text outside the JSON block
- Be strict but fair in your grading"""

    def _extract_prediction(self, msg_history: list[dict]) -> tuple[str, str]:
        """Extract prediction and reasoning from message history.
        
        Enhanced extraction with multiple fallback strategies and better
        handling of edge cases like nested JSON, malformed responses,
        and various grade formats.
        
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
            
            # Strategy 3: Try to find any JSON-like structure with response field
            # Use a more robust pattern that handles nested braces
            json_pattern = r'\{[^{}]*"response"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            json_matches = re.findall(json_pattern, last_msg, re.DOTALL)
            for match in json_matches:
                try:
                    fallback = json.loads(match)
                    if "response" in fallback:
                        prediction = str(fallback["response"]).strip()
                        if "reasoning" in fallback:
                            reasoning = fallback["reasoning"]
                        if prediction != "None":
                            return prediction, reasoning
                except json.JSONDecodeError:
                    continue
            
            # Strategy 4: Look for explicit grade/score patterns
            grade_patterns = [
                # "Grade: X" or "Final grade: X"
                r'(?:final\s+)?(?:grade|score|mark)s?\s*:?\s*["\']?([0-7]|partial|full|complete|incorrect|correct|zero|none)["\']?',
                # "The grade is X" or "I give a grade of X"
                r'(?:grade|score|mark)\s+(?:is|of|equals?|:\s*)["\']?([0-7])["\']?',
                # "X points" or "X out of 7"
                r'\b([0-7])\s*(?:points?|/\s*7|out\s+of\s+7)',
                # "awarded X" or "deserves X"
                r'(?:awarded|deserves|earns?|receives?)\s*["\']?([0-7])["\']?',
            ]
            
            for pattern in grade_patterns:
                grade_match = re.search(pattern, last_msg, re.IGNORECASE)
                if grade_match:
                    prediction = grade_match.group(1)
                    break
            
            # Strategy 5: Look for standalone numbers 0-7 in context
            if prediction == "None":
                # Find numbers that appear near grade-related words
                context_pattern = r'(?:grade|score|mark|point|evaluation|assessment).{0,30}\b([0-7])\b'
                context_match = re.search(context_pattern, last_msg, re.IGNORECASE | re.DOTALL)
                if context_match:
                    prediction = context_match.group(1)
            
            # Strategy 6: Last resort - look for any single digit 0-7 in the response
            if prediction == "None":
                # Try to find a standalone digit that could be the grade
                standalone_pattern = r'(?:^|\s|"|\')([0-7])(?:\s|$|"|\')'
                standalone_match = re.search(standalone_pattern, last_msg)
                if standalone_match:
                    prediction = standalone_match.group(1)
            
            # Strategy 7: Extract reasoning from text if no JSON found
            if not reasoning and last_msg:
                # Look for reasoning sections
                reasoning_patterns = [
                    r'(?:reasoning|analysis|explanation|thoughts?)[:\s]+(.{50,500})',
                    r'(?:step\s+by\s+step|first|analysis)[:\s]+(.{50,500})',
                ]
                for pattern in reasoning_patterns:
                    reasoning_match = re.search(pattern, last_msg, re.IGNORECASE | re.DOTALL)
                    if reasoning_match:
                        reasoning = reasoning_match.group(1).strip()
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
        
        # If grade is invalid, try to extract from the full response text
        if not is_valid and response:
            # Try to find any numeric grade in the response
            numeric_match = re.search(r'\b([0-7])\b', response)
            if numeric_match:
                validated_grade = numeric_match.group(1)
                is_valid = True
                self.log_fn(f"Fallback extraction found grade: {validated_grade}")
        
        # If still invalid, try one more time with a clearer prompt
        if not is_valid:
            self.log_fn("Grade extraction failed, attempting retry with simplified prompt...")
            retry_prompt = f"""Based on your previous analysis, provide ONLY the final numeric grade (0-7) for this IMO problem.

Respond in this exact format:
<json>
{{
    "response": "X"
}}
</json>

Where X is a single digit from 0 to 7."""
            
            try:
                retry_response, retry_history, _ = get_response_from_llm(
                    msg=retry_prompt,
                    model=self.model,
                    msg_history=msg_history,
                )
                retry_prediction, _ = self._extract_prediction(retry_history)
                retry_validated, retry_is_valid = _validate_grade(retry_prediction, grading_guidelines)
                
                if retry_is_valid:
                    validated_grade = retry_validated
                    is_valid = True
                    msg_history = retry_history
                    self.log_fn(f"Retry successful, grade: {validated_grade}")
            except Exception as e:
                self.log_fn(f"Retry failed: {e}")

        return str(validated_grade), msg_history
