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
    # Look for standalone digits or digits at word boundaries
    numeric_match = re.search(r'(?:^|\s|\b)([0-7])(?:\s*(?:points?|/|out\s+of)|\b|$)', pred_clean, re.IGNORECASE)
    if numeric_match:
        return numeric_match.group(1), True
    
    # Additional check: look for any digit 0-7 as a standalone number
    standalone_match = re.search(r'\b([0-7])\b', pred_clean)
    if standalone_match:
        return standalone_match.group(1), True
    
    # Check for full credit patterns -> 7
    full_patterns = [
        r'\bfull\s*(?:credit|points?|score|marks?)?\b',
        r'\bcomplete\s*(?:solution|answer|credit|proof)?\b',
        r'\ball\s*(?:points?|credit|marks?)?\b',
        r'\bperfect\s*(?:score|solution|answer)?\b',
        r'\bcorrect\s*(?:solution|answer|proof)?\b',
        r'\bseven\s*(?:points?)?\b',
        r'\bmaximum\s*(?:score|points?|credit)?\b',
    ]
    for pattern in full_patterns:
        if re.search(pattern, pred_lower):
            return "7", True
    
    # Check for zero/no credit patterns -> 0
    zero_patterns = [
        r'\bno\s*(?:credit|points?|score|marks?)?\b',
        r'\bzero\s*(?:credit|points?|score|marks?)?\b',
        r'\b0\s*(?:points?|credit|score|marks?)?\b',
        r'\bincorrect\s*(?:solution|answer|proof)?\b',
        r'\bwrong\s*(?:solution|answer|approach)?\b',
        r'\binvalid\s*(?:solution|proof|argument)?\b',
        r'\bnone\b',
        r'\bno\s+progress\b',
        r'\bno\s+meaningful\b',
    ]
    for pattern in zero_patterns:
        if re.search(pattern, pred_lower):
            return "0", True
    
    # Check for partial credit patterns that might indicate specific grades
    # These are less reliable but can help in ambiguous cases
    partial_patterns = [
        (r'\b(?:almost|nearly)\s*(?:complete|correct)\b', "6"),
        (r'\bminor\s*(?:error|flaw|mistake)\b', "6"),
        (r'\bsignificant\s*progress\b', "5"),
        (r'\bone\s*(?:major|significant)\s*(?:gap|error|flaw)\b', "5"),
        (r'\bpartial\s*(?:progress|credit|solution)\b', "3"),
        (r'\bmultiple\s*(?:gaps|errors|flaws)\b', "3"),
        (r'\bsome\s*(?:progress|ideas|insight)\b', "2"),
        (r'\bminimal\s*(?:progress|credit)\b', "1"),
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
        """Build a structured prompt for IMO grading with chain-of-thought reasoning."""
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        return f"""You are an expert IMO (International Mathematical Olympiad) grader with deep expertise in mathematical problem evaluation.

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

## IMO Grading Standards Reference

The IMO uses a 0-7 point scale with these specific criteria:

**7 points (Full marks)**: Complete, correct solution with full mathematical rigor. All steps are justified, no gaps in logic, and the conclusion is correct.

**6 points (Near complete)**: Minor flaw in an otherwise complete solution. This could be a small calculation error, a missing trivial case, or a slight lack of clarity that doesn't affect the core argument.

**5 points (Significant progress)**: Significant progress with one major gap. The student has the right approach and has made substantial progress, but there's one significant missing piece or error.

**3-4 points (Partial progress)**: Partial progress with multiple gaps. The student has some correct ideas and made meaningful progress, but there are several gaps or errors preventing a complete solution.

**1-2 points (Some progress)**: Some meaningful progress or ideas. The student has at least one correct idea or has made some progress toward the solution, even if incomplete.

**0 points (No progress)**: No meaningful progress or completely wrong approach. The student either didn't attempt the problem, wrote something irrelevant, or their approach is fundamentally flawed.

## Grading Examples

**Example 1 - Grade 7**: Student provides a complete proof with all steps justified, matching or exceeding the official solution in rigor.

**Example 2 - Grade 6**: Student has a correct approach and nearly complete solution, but has a minor calculation error in one step that doesn't affect the overall structure.

**Example 3 - Grade 4**: Student correctly identifies the key insight but fails to fully develop the proof, leaving significant gaps in the argument.

**Example 4 - Grade 1**: Student writes down a relevant formula or makes a single correct observation but cannot proceed further.

## Instructions

1. **Analyze the student's answer step by step** - Compare each step against the official solution.
2. **Identify key elements**: correct approach, logical flow, mathematical rigor, and completeness.
3. **Note any errors**: calculation mistakes, logical gaps, missing cases, or incorrect assumptions.
4. **Consider alternative approaches**: Valid alternative methods should receive appropriate credit if they are mathematically sound.
5. **Be conservative**: When in doubt between two grades, choose the lower one. IMO grading rewards correctness over partial progress.
6. **Check for rigor**: IMO requires rigorous proofs, not just correct answers. Ensure all claims are justified.

## Output Format

You MUST respond with a valid JSON object wrapped in <json> tags. The JSON must have exactly these two fields:

<json>
{{
    "reasoning": "Your detailed analysis of the student's answer, comparing it to the official solution and explaining your evaluation...",
    "response": "X"
}}
</json>

CRITICAL REQUIREMENTS:
- The "response" field MUST contain ONLY a single digit from 0 to 7 (e.g., "7", "5", "0")
- Do NOT include any other text, explanations, or formatting in the response field
- The "reasoning" field should contain your full analysis with specific references to the solution
- Valid grades are: 0, 1, 2, 3, 4, 5, 6, 7
- Be consistent with IMO grading standards"""

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
                # Look for JSON objects that might contain nested structures
                json_pattern = re.search(r'\{.*"response".*\}', last_msg, re.DOTALL)
                if json_pattern:
                    try:
                        # Try to extract just the response value
                        response_match = re.search(r'"response"\s*:\s*"?([0-7])"?', last_msg)
                        if response_match:
                            prediction = response_match.group(1)
                        # Try to extract reasoning
                        reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', last_msg)
                        if reasoning_match:
                            reasoning = reasoning_match.group(1)
                    except Exception:
                        pass
            
            # Last resort: look for grade patterns in text
            if prediction == "None":
                # Look for "Grade: X" or "Final grade: X" patterns
                grade_match = re.search(r'(?:grade|score|mark|final grade|final score)\s*:?\s*([0-7])', last_msg, re.IGNORECASE)
                if grade_match:
                    prediction = grade_match.group(1)
                
                # Look for standalone grade mentions
                if prediction == "None":
                    standalone_match = re.search(r'\bgrade\s+(?:is\s+)?([0-7])\b', last_msg, re.IGNORECASE)
                    if standalone_match:
                        prediction = standalone_match.group(1)
                    
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
        
        max_retries = 2
        best_grade = "None"
        best_history = []
        
        for attempt in range(max_retries + 1):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                best_history = msg_history
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
            
            # Track the best valid grade we've seen
            if is_valid and validated_grade in ["0", "1", "2", "3", "4", "5", "6", "7"]:
                best_grade = validated_grade
                # If we have a high-confidence grade (7 or 0), return immediately
                if validated_grade in ["0", "7"]:
                    return str(validated_grade), msg_history
            
            # If grade is invalid, try to extract from the full response text
            if not is_valid and response:
                # Try to find any numeric grade in the response
                numeric_match = re.search(r'\b([0-7])\b', response)
                if numeric_match:
                    validated_grade = numeric_match.group(1)
                    is_valid = True
                    self.log_fn(f"Fallback extraction found grade: {validated_grade}")
                    best_grade = validated_grade
                    if validated_grade in ["0", "7"]:
                        return str(validated_grade), msg_history
                else:
                    # Try to find grade patterns in the full response
                    grade_patterns = [
                        (r'(?:grade|score|mark|final grade|final score)\s*:?\s*([0-7])', 'numeric'),
                        (r'\bfull\s*(?:credit|points?|score|marks?)?\b', 'full'),
                        (r'\bcomplete\s*(?:solution|answer|proof)?\b', 'full'),
                        (r'\bcorrect\s*(?:solution|answer|proof)?\b', 'full'),
                        (r'\bno\s*(?:credit|points?|score|marks?)?\b', 'zero'),
                        (r'\bzero\s*(?:credit|points?|score|marks?)?\b', 'zero'),
                        (r'\bincorrect\s*(?:solution|answer|proof)?\b', 'zero'),
                        (r'\bwrong\s*(?:solution|answer|approach)?\b', 'zero'),
                    ]
                    for pattern, ptype in grade_patterns:
                        match = re.search(pattern, response, re.IGNORECASE)
                        if match:
                            if ptype == 'full':
                                validated_grade = "7"
                            elif ptype == 'zero':
                                validated_grade = "0"
                            else:
                                validated_grade = match.group(1)
                            is_valid = True
                            self.log_fn(f"Pattern-based fallback found grade: {validated_grade}")
                            best_grade = validated_grade
                            if validated_grade in ["0", "7"]:
                                return str(validated_grade), msg_history
                            break
            
            # If we have a valid grade now, return it
            if is_valid and validated_grade in ["0", "1", "2", "3", "4", "5", "6", "7"]:
                return str(validated_grade), msg_history
            
            # If we still don't have a valid grade and have retries left, try again
            if not is_valid and attempt < max_retries:
                self.log_fn(f"Invalid grade on attempt {attempt + 1}, retrying...")
                continue
        
        # Return the best grade we found, or "None" if nothing worked
        return str(best_grade) if best_grade != "None" else "None", best_history
