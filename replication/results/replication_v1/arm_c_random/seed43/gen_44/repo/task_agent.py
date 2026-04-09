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
from collections import Counter

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
                # Look for JSON object pattern with balanced braces
                json_start = inner.find('{')
                if json_start != -1:
                    # Find the matching closing brace
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
                        results.append(json.loads(inner[json_start:json_end+1]))
            except json.JSONDecodeError:
                continue
    return results or None


def _extract_json_from_markdown(text: str) -> dict | None:
    """Extract JSON from markdown code blocks (```json ... ```)."""
    # Try both ```json and ``` patterns
    patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(\{.*?\})\s*```',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                # Try to find valid JSON within the content using balanced braces
                try:
                    json_start = match.find('{')
                    if json_start != -1:
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
                            return json.loads(match[json_start:json_end+1].strip())
                except json.JSONDecodeError:
                    continue
    return None


def _validate_grade(prediction: str, grading_guidelines: str) -> tuple[str, bool, float]:
    """Validate that the extracted grade is reasonable.
    
    Enhanced validation with strict support for IMO 0-7 point scale.
    Only accepts single digit grades 0-7 for consistency.
    
    Returns:
        (validated_grade, is_valid, confidence_score)
    """
    if not prediction or prediction == "None":
        return "None", False, 0.0
    
    pred_clean = prediction.strip()
    pred_lower = pred_clean.lower()
    
    # Strict validation: only accept single digit 0-7 (highest confidence)
    if pred_clean in ["0", "1", "2", "3", "4", "5", "6", "7"]:
        return pred_clean, True, 1.0
    
    # Check for fractional grades like "3/7" or "5 / 7" - extract just the numerator
    fractional_match = re.search(r'\b([0-7])\s*/\s*7\b', pred_clean)
    if fractional_match:
        return fractional_match.group(1), True, 0.95
    
    # Check for "X out of 7" patterns
    out_of_match = re.search(r'\b([0-7])\s+out\s+of\s+7\b', pred_lower)
    if out_of_match:
        return out_of_match.group(1), True, 0.95
    
    # Check for numeric grades embedded in text (0-7 for IMO problems)
    # Look for standalone digits or digits at word boundaries
    numeric_match = re.search(r'(?:^|\s|[^0-9])([0-7])(?:\s|[^0-9]|$)', pred_clean)
    if numeric_match:
        return numeric_match.group(1), True, 0.85
    
    # Check for full credit patterns -> 7
    full_patterns = [
        r'\bfull\s*(?:credit|points?|score|marks?)?\b',
        r'\bcomplete\s*(?:solution|answer|credit)?\b',
        r'\ball\s*(?:points?|credit|marks?)?\b',
        r'\bperfect\s*(?:score|solution)?\b',
        r'\bcorrect\s*(?:solution|answer)?\b',
        r'\bseven\s*(?:points?)?\b',
        r'\bmax(?:imum)?\s*(?:credit|points?|score)?\b',
    ]
    for pattern in full_patterns:
        if re.search(pattern, pred_lower):
            return "7", True, 0.75
    
    # Check for zero/no credit patterns -> 0
    zero_patterns = [
        r'\bno\s*(?:credit|points?|score|marks?)?\b',
        r'\bzero\s*(?:credit|points?|score|marks?)?\b',
        r'\b0\s*(?:points?|credit|score|marks?)?\b',
        r'\bincorrect\s*(?:solution|answer)?\b',
        r'\bwrong\s*(?:solution|answer)?\b',
        r'\bnone\b',
        r'\bblank\b',
        r'\bempty\b',
    ]
    for pattern in zero_patterns:
        if re.search(pattern, pred_lower):
            return "0", True, 0.75
    
    # Check for spelled-out numbers
    number_words = {
        "zero": "0", "one": "1", "two": "2", "three": "3",
        "four": "4", "five": "5", "six": "6", "seven": "7"
    }
    for word, digit in number_words.items():
        if re.search(rf'\b{word}\b', pred_lower):
            return digit, True, 0.7
    
    # Check for partial credit indicators with specific numbers
    partial_patterns = [
        (r'\b(?:partial|some)\s+(?:credit|points?)\s*:?\s*([0-7])\b', 0.6),
        (r'\b(?:award|assign|give)\s*:?\s*([0-7])\b', 0.6),
        (r'\bgrade\s*(?:is|of|=|:)\s*([0-7])\b', 0.65),
        (r'\bscore\s*(?:is|of|=|:)\s*([0-7])\b', 0.65),
    ]
    for pattern, conf in partial_patterns:
        match = re.search(pattern, pred_lower)
        if match:
            return match.group(1), True, conf
    
    # If no clear grade found, mark as invalid
    return pred_clean, False, 0.0


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

        return f"""You are an expert IMO (International Mathematical Olympiad) grader. Your task is to evaluate the student's solution and assign a grade from 0-7.

## Problem
{problem}

## Official Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## IMO Grading Scale (0-7) - BE PRECISE AND CONSISTENT
- 7: Complete, correct solution with proper justification. All steps are valid and complete. No gaps or errors.
- 6: Minor flaw in an otherwise complete solution (e.g., small gap in reasoning, minor calculation error, slight imprecision).
- 5: Significant progress with substantial solution elements. Most key ideas present but with notable gaps or one significant error.
- 4: Good partial progress. Several correct key steps but incomplete solution. Multiple gaps or errors in reasoning.
- 3: Some meaningful progress. At least one key idea or correct step demonstrated. Partial solution with significant gaps.
- 2: Minimal progress. Some relevant ideas but little concrete progress toward solution. Few correct steps.
- 1: Very minimal progress. Barely relevant to the problem. Only trivial observations or incorrect attempts.
- 0: No meaningful progress, completely incorrect, blank, or irrelevant.

## Detailed Grading Principles
1. **Award partial credit generously** for correct mathematical ideas, even if incomplete
2. **A correct key insight** (e.g., crucial lemma, correct approach) deserves at least 2-3 points
3. **Multiple correct steps** with good reasoning deserve 4-5 points
4. **Only deduct for actual errors**, not for missing "elegant" steps or different notation
5. **Check alternative approaches**: The student's approach may be valid even if different from the official solution
6. **Consider partial credit for**: correct setup, valid intermediate results, correct final answer with minor errors
7. **Be consistent**: Similar levels of progress should receive similar grades

## Step-by-Step Evaluation Process
1. **Read the problem carefully** - understand what is being asked
2. **Analyze the official solution** - identify key steps, critical insights, and proof structure
3. **Read the student's answer** - identify what they actually did
4. **Identify correct elements**: 
   - Correct setup/initialization
   - Valid mathematical statements
   - Correct key insights or lemmas
   - Valid proof techniques
   - Correct final answer (if any)
5. **Identify errors or gaps**:
   - Mathematical errors
   - Logical gaps in reasoning
   - Missing steps
   - Invalid assumptions
6. **Compare to official solution** - assess how much of the solution path was covered
7. **Assign grade** based on the rubric above

## Output Format (CRITICAL - FOLLOW EXACTLY)
Respond with ONLY this JSON format:

<json>
{{
    "reasoning": "Detailed analysis: (1) Summary of student's approach, (2) All correct elements identified, (3) All errors/gaps noted, (4) Comparison to official solution, (5) Specific justification for the grade assigned based on the 0-7 scale",
    "response": "X"
}}
</json>

CRITICAL RULES:
- "response" field: ONLY a single digit 0-7 (no quotes, no text, no explanation, no fractions)
- "reasoning" field: Your complete analysis (be thorough, specific, and reference the rubric)
- Ensure valid JSON with proper formatting
- The response field must contain exactly one digit: 0, 1, 2, 3, 4, 5, 6, or 7
- Do NOT write "7/7" or "5 points" - only the single digit
- Example correct response: "response": "5"
- Example incorrect response: "response": "5/7" or "response": "five points"
"""

    def _extract_prediction(self, msg_history: list[dict]) -> tuple[str, str]:
        """Extract prediction and reasoning from message history with enhanced robustness.
        
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
            
            # Try <json> tags first (most reliable)
            extracted = _extract_jsons(last_msg)
            if extracted:
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = str(last_json["response"]).strip()
                if "reasoning" in last_json:
                    reasoning = str(last_json["reasoning"])
                # If we got a valid prediction, return it
                if prediction != "None" and prediction:
                    return prediction, reasoning
            
            # Try markdown code blocks
            md_json = _extract_json_from_markdown(last_msg)
            if md_json:
                if "response" in md_json:
                    prediction = str(md_json["response"]).strip()
                if "reasoning" in md_json:
                    reasoning = str(md_json["reasoning"])
                if prediction != "None" and prediction:
                    return prediction, reasoning
            
            # Fallback: try to find any JSON-like structure with response field
            # Use a more flexible pattern that handles nested braces
            json_pattern = r'\{[^{}]*"response"[^{}]*\}'
            json_matches = re.findall(json_pattern, last_msg)
            for json_match in json_matches:
                try:
                    fallback = json.loads(json_match)
                    pred = str(fallback.get("response", "")).strip()
                    if pred and pred != "None":
                        prediction = pred
                        if "reasoning" in fallback:
                            reasoning = str(fallback["reasoning"])
                        return prediction, reasoning
                except json.JSONDecodeError:
                    continue
            
            # Try to find JSON with balanced braces (more complex structures)
            brace_start = last_msg.find('{')
            if brace_start != -1:
                brace_count = 0
                for i, char in enumerate(last_msg[brace_start:], start=brace_start):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            try:
                                json_str = last_msg[brace_start:i+1]
                                parsed = json.loads(json_str)
                                if "response" in parsed:
                                    pred = str(parsed["response"]).strip()
                                    if pred and pred != "None":
                                        prediction = pred
                                        if "reasoning" in parsed:
                                            reasoning = str(parsed["reasoning"])
                                        return prediction, reasoning
                            except json.JSONDecodeError:
                                pass
                            break
            
            # Try to find all JSON objects in the message and check each one
            # This handles cases where there might be multiple JSON blocks
            json_blocks = re.findall(r'\{.*?\}', last_msg, re.DOTALL)
            for block in json_blocks:
                try:
                    parsed = json.loads(block)
                    if "response" in parsed:
                        pred = str(parsed["response"]).strip()
                        if pred and pred != "None":
                            prediction = pred
                            if "reasoning" in parsed:
                                reasoning = str(parsed["reasoning"])
                            return prediction, reasoning
                except json.JSONDecodeError:
                    continue
            
            # Last resort: look for grade patterns in text
            if prediction == "None":
                # Look for "Grade: X" or "Final grade: X" patterns
                grade_match = re.search(r'(?:grade|score|mark|final grade|final score)\s*:?\s*([0-7])', last_msg, re.IGNORECASE)
                if grade_match:
                    prediction = grade_match.group(1)
                
                # Look for "response": "X" pattern (even if not valid JSON)
                response_match = re.search(r'"response"\s*:\s*"([0-7])"', last_msg)
                if response_match:
                    prediction = response_match.group(1)
                    
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
        
        # Validate the grade with confidence score
        validated_grade, is_valid, confidence = _validate_grade(prediction, grading_guidelines)
        
        # Log the reasoning and validation result
        if reasoning:
            self.log_fn(f"Reasoning: {reasoning[:200]}...")
        self.log_fn(f"Extracted grade: {prediction}, Validated: {validated_grade}, Is valid: {is_valid}, Confidence: {confidence}")
        
        # If grade is invalid or low confidence, try to extract from the full response text
        if (not is_valid or confidence < 0.8) and response:
            # Try to find any numeric grade in the response
            numeric_match = re.search(r'\b([0-7])\b', response)
            if numeric_match:
                validated_grade = numeric_match.group(1)
                is_valid = True
                confidence = 0.85
                self.log_fn(f"Fallback extraction found grade: {validated_grade}")
            else:
                # Try to find grade patterns in the full response
                grade_patterns = [
                    (r'(?:grade|score|mark|final grade|final score)\s*:?\s*([0-7])', 'numeric'),
                    (r'\bfull\s*(?:credit|points?|score|marks?)?\b', 'full'),
                    (r'\bcorrect\s*(?:solution|answer)?\b', 'full'),
                    (r'\bno\s*(?:credit|points?|score|marks?)?\b', 'zero'),
                    (r'\bzero\s*(?:credit|points?|score|marks?)?\b', 'zero'),
                    (r'\bincorrect\s*(?:solution|answer)?\b', 'zero'),
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
                        confidence = 0.75
                        self.log_fn(f"Pattern-based fallback found grade: {validated_grade}")
                        break
        
        # If still invalid or low confidence, retry with a clearer prompt
        if not is_valid or confidence < 0.5:
            self.log_fn(f"Grade validation failed or low confidence ({confidence}), retrying with clearer prompt...")
            retry_instruction = instruction + "\n\nCRITICAL REMINDER: Your response field MUST contain ONLY a single digit from 0-7. No other text allowed. Examples: \"5\", \"7\", \"0\". Do NOT write \"5/7\" or \"five points\"."
            try:
                retry_response, retry_msg_history, retry_info = get_response_from_llm(
                    msg=retry_instruction,
                    model=self.model,
                    msg_history=[],
                )
                retry_prediction, retry_reasoning = self._extract_prediction(retry_msg_history)
                validated_grade, is_valid, confidence = _validate_grade(retry_prediction, grading_guidelines)
                self.log_fn(f"Retry result: grade={validated_grade}, valid={is_valid}, confidence={confidence}")
                if is_valid:
                    msg_history = retry_msg_history
            except Exception as e:
                self.log_fn(f"Retry LLM call failed: {e}")
        
        # Final fallback: if we still don't have a valid grade, try one more aggressive extraction
        if not is_valid and response:
            self.log_fn(f"Final fallback: searching entire response for any valid grade indicator...")
            # Look for any digit 0-7 in the response
            all_digits = re.findall(r'\b([0-7])\b', response)
            if all_digits:
                # Use the most common digit or the last one mentioned
                digit_counts = Counter(all_digits)
                most_common = digit_counts.most_common(1)[0][0]
                validated_grade = most_common
                is_valid = True
                confidence = 0.5
                self.log_fn(f"Final fallback found grade: {validated_grade} (most common digit)")

        return str(validated_grade), msg_history
