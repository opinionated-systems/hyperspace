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
    # Try both ```json and ``` patterns with more flexible matching
    patterns = [
        r'```json\s*\n?(.*?)\n?```',
        r'```\s*\n?(\{[\s\S]*?\})\n?```',
        r'```\s*\n?(.*?)\n?```',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            match_clean = match.strip()
            try:
                return json.loads(match_clean)
            except json.JSONDecodeError:
                # Try to find valid JSON within the content using balanced braces
                try:
                    json_start = match_clean.find('{')
                    if json_start != -1:
                        brace_count = 0
                        json_end = -1
                        for i, char in enumerate(match_clean[json_start:], start=json_start):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    json_end = i
                                    break
                        if json_end != -1:
                            return json.loads(match_clean[json_start:json_end+1].strip())
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
    
    # Remove common prefixes/suffixes that might be attached
    pred_clean = re.sub(r'^(?:grade|score|mark|points?)\s*[:=]?\s*', '', pred_lower, flags=re.IGNORECASE)
    pred_clean = re.sub(r'\s*(?:points?|marks?|score|grade)?$', '', pred_clean, flags=re.IGNORECASE)
    pred_clean = pred_clean.strip()
    
    # Strict validation: only accept single digit 0-7
    if pred_clean in ["0", "1", "2", "3", "4", "5", "6", "7"]:
        return pred_clean, True
    
    # Check for fractional grades like "3/7" or "5 / 7" - extract just the numerator
    fractional_match = re.search(r'\b([0-7])\s*/\s*7\b', pred_clean)
    if fractional_match:
        return fractional_match.group(1), True
    
    # Check for numeric grades embedded in text (0-7 for IMO problems)
    # Look for standalone digits or digits at word boundaries
    numeric_match = re.search(r'(?:^|\s|[^0-9])([0-7])(?:\s|[^0-9]|$)', pred_clean)
    if numeric_match:
        return numeric_match.group(1), True
    
    # Check for full credit patterns -> 7
    full_patterns = [
        r'\bfull\s*(?:credit|points?|score|marks?)?\b',
        r'\bcomplete\s*(?:solution|answer|credit)?\b',
        r'\ball\s*(?:points?|credit|marks?)?\b',
        r'\bperfect\s*(?:score|solution)?\b',
        r'\bcorrect\s*(?:solution|answer)?\b',
        r'\bseven\s*(?:points?)?\b',
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
        r'\bblank\b',
        r'\bempty\b',
    ]
    for pattern in zero_patterns:
        if re.search(pattern, pred_lower):
            return "0", True
    
    # Check for spelled-out numbers
    number_words = {
        "zero": "0", "one": "1", "two": "2", "three": "3",
        "four": "4", "five": "5", "six": "6", "seven": "7"
    }
    for word, digit in number_words.items():
        if re.search(rf'\b{word}\b', pred_lower):
            return digit, True
    
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

        return f"""You are an expert IMO (International Mathematical Olympiad) grader with years of experience evaluating mathematical proofs. Your task is to evaluate the student's solution and assign a precise grade from 0-7.

## Problem
{problem}

## Official Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## IMO Grading Scale (0-7) - USE THESE CRITERIA EXACTLY
- **7 points**: Complete, correct solution with rigorous proof and proper justification. All steps are logically sound.
- **6 points**: Minor flaw (e.g., small gap in reasoning, typo in calculation) in an otherwise complete and correct solution.
- **5 points**: Significant progress with substantial solution elements. Most key ideas present but some gaps remain.
- **4 points**: Good partial progress. Multiple correct key steps but solution incomplete or has notable errors.
- **3 points**: Some genuine progress. At least one key idea or meaningful step toward solution.
- **2 points**: Minimal progress. Some relevant ideas or observations but little substantive work.
- **1 point**: Very minimal progress. Some awareness of problem structure but essentially no useful work.
- **0 points**: No meaningful progress, completely incorrect approach, or blank submission.

## Grading Instructions
1. **Read carefully**: First understand what the problem asks and what the official solution demonstrates.
2. **Analyze the student's approach**: Identify their strategy and key mathematical ideas.
3. **Check correctness**: Verify each claim and calculation in the student's work.
4. **Compare to official solution**: Note which parts they completed correctly vs. where they went wrong.
5. **Assess partial credit**: Award points for each correct key insight or meaningful step, even if final answer is wrong.
6. **Be precise**: The grade must reflect the actual quality of work, not your guess at their intent.

## Common Grading Pitfalls to Avoid
- Don't give full credit just for correct final answer without proper proof
- Don't penalize for minor notational issues if logic is sound
- Don't ignore partial progress - award points for correct intermediate steps
- Don't be swayed by length; short correct solutions deserve full credit
- Don't assume unstated steps are obvious; check if they're actually justified

## Output Format (CRITICAL - FOLLOW EXACTLY)
Respond with ONLY this JSON format. No other text before or after:

<json>
{{
    "reasoning": "Your detailed analysis here. Structure as: 1) Student's approach summary, 2) Key correct elements found, 3) Errors or gaps identified, 4) Comparison to official solution, 5) Justification for the specific grade assigned",
    "response": "X"
}}
</json>

STRICT RULES:
- "response" field: ONLY a single digit 0-7. No quotes, no spaces, no explanation, no punctuation.
- "reasoning" field: Complete analysis as described above.
- Ensure valid JSON with proper quotes and commas.
- Example correct response field: "response": "5"
- Example incorrect response field: "response": "5 points" or "response": "grade 5"
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
            # Use a more flexible pattern that handles nested braces
            json_pattern = r'\{[^{}]*"response"[^{}]*\}'
            json_matches = re.findall(json_pattern, last_msg, re.DOTALL)
            for json_match in reversed(json_matches):  # Try last match first
                try:
                    fallback = json.loads(json_match)
                    pred = str(fallback.get("response", "None")).strip()
                    if pred and pred != "None":
                        prediction = pred
                        if "reasoning" in fallback:
                            reasoning = str(fallback["reasoning"])
                        break
                except json.JSONDecodeError:
                    continue
            
            # If still no prediction, try broader JSON extraction
            if prediction == "None":
                # Look for any JSON object in the text
                brace_pattern = r'\{[\s\S]*?\}'
                for match in re.finditer(brace_pattern, last_msg):
                    try:
                        data = json.loads(match.group())
                        if "response" in data:
                            prediction = str(data["response"]).strip()
                            if "reasoning" in data:
                                reasoning = str(data["reasoning"])
                            break
                    except (json.JSONDecodeError, ValueError):
                        continue
            
            # Last resort: look for grade patterns in text
            if prediction == "None":
                # Look for "Grade: X" or "Final grade: X" patterns
                grade_patterns = [
                    r'(?:grade|score|mark|final grade|final score)\s*[:=]\s*([0-7])\b',
                    r'(?:grade|score|mark|final grade|final score)\s+is\s+([0-7])\b',
                    r'["\']response["\']\s*:\s*["\']?([0-7])["\']?',
                    r'\bgrade\s+([0-7])\s*(?:points?)?\b',
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
                    (r'(?:grade|score|mark|final grade|final score)\s*[:=]\s*([0-7])\b', 'numeric'),
                    (r'["\']response["\']\s*:\s*["\']?([0-7])["\']?', 'json_field'),
                    (r'\bfull\s*(?:credit|points?|score|marks?)?\b', 'full'),
                    (r'\bcorrect\s*(?:solution|answer)?\b', 'correct'),
                    (r'\bno\s*(?:credit|points?|score|marks?)?\b', 'zero'),
                    (r'\bzero\s*(?:credit|points?|score|marks?)?\b', 'zero'),
                    (r'\bincorrect\s*(?:solution|answer)?\b', 'zero'),
                    (r'\bwrong\s*(?:solution|answer)?\b', 'zero'),
                ]
                for pattern, pattern_type in grade_patterns:
                    match = re.search(pattern, response, re.IGNORECASE)
                    if match:
                        if pattern_type in ['full', 'correct']:
                            validated_grade = "7"
                        elif pattern_type == 'zero':
                            validated_grade = "0"
                        else:
                            validated_grade = match.group(1)
                        is_valid = True
                        self.log_fn(f"Pattern-based fallback found grade: {validated_grade}")
                        break
        
        # If still invalid, retry with a clearer prompt
        if not is_valid:
            self.log_fn(f"Grade validation failed, retrying with clearer prompt...")
            retry_instruction = instruction + """

CRITICAL REMINDER: Your response field MUST contain ONLY a single digit from 0-7. 
- Correct: "response": "5"
- Incorrect: "response": "5 points" or "response": "Grade: 5"
- The response field should be ONLY the digit, nothing else.

Please respond with valid JSON in the exact format specified."""
            try:
                retry_response, retry_msg_history, retry_info = get_response_from_llm(
                    msg=retry_instruction,
                    model=self.model,
                    msg_history=[],
                )
                retry_prediction, retry_reasoning = self._extract_prediction(retry_msg_history)
                validated_grade, is_valid = _validate_grade(retry_prediction, grading_guidelines)
                self.log_fn(f"Retry result: grade={validated_grade}, valid={is_valid}")
                if is_valid:
                    msg_history = retry_msg_history
                    if retry_reasoning:
                        self.log_fn(f"Retry reasoning: {retry_reasoning[:200]}...")
            except Exception as e:
                self.log_fn(f"Retry LLM call failed: {e}")

        # Final fallback: if still invalid, use the best guess from the original prediction
        if not is_valid and prediction != "None":
            # Try to extract any digit from the invalid prediction
            digit_match = re.search(r'[0-7]', prediction)
            if digit_match:
                validated_grade = digit_match.group(0)
                is_valid = True
                self.log_fn(f"Final fallback: extracted digit {validated_grade} from prediction")

        return str(validated_grade), msg_history
