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
    
    # Clean the prediction
    pred_clean = prediction.strip().strip('"\'')
    pred_lower = pred_clean.lower()
    
    # Direct match: single digit 0-7 (most common case)
    if pred_clean in ["0", "1", "2", "3", "4", "5", "6", "7"]:
        return pred_clean, True
    
    # Check for fractional grades like "3/7" or "5 / 7"
    fractional_match = re.search(r'\b([0-7])\s*/\s*7\b', pred_clean)
    if fractional_match:
        return fractional_match.group(1), True
    
    # Extract first digit 0-7 from the string
    numeric_match = re.search(r'\b([0-7])\b', pred_clean)
    if numeric_match:
        return numeric_match.group(1), True
    
    # Text pattern matching for semantic grades
    # Full credit patterns -> 7
    full_patterns = [
        r'\bfull\s*(?:credit|points?|score|marks?)?\b',
        r'\bcomplete\s*(?:solution|answer|credit)?\b',
        r'\ball\s*(?:points?|credit|marks?)?\b',
        r'\bperfect\s*(?:score|solution)?\b',
        r'\bcorrect\s*(?:solution|answer|work)?\b',
        r'\bexcellent\b',
        r'\bfully\s*correct\b',
    ]
    for pattern in full_patterns:
        if re.search(pattern, pred_lower):
            return "7", True
    
    # Zero/no credit patterns -> 0
    zero_patterns = [
        r'\bno\s*(?:credit|points?|score|marks?)?\b',
        r'\bzero\s*(?:credit|points?|score|marks?)?\b',
        r'\b0\s*(?:points?|credit|score|marks?)?\b',
        r'\bincorrect\s*(?:solution|answer|work)?\b',
        r'\bwrong\s*(?:solution|answer|work)?\b',
        r'\bnone\b',
        r'\bno\s*score\b',
        r'\bno\s*marks?\b',
    ]
    for pattern in zero_patterns:
        if re.search(pattern, pred_lower):
            return "0", True
    
    # If no clear grade found, mark as invalid
    return pred_clean, False


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced reasoning and validation."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _build_prompt(self, inputs: dict) -> str:
        """Build a structured prompt for IMO grading with chain-of-thought reasoning.
        
        Enhanced with input validation and default handling for robustness.
        """
        # Validate inputs with defaults and warnings
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Log warnings for missing critical inputs
        if not problem:
            self.log_fn("Warning: Empty problem statement provided")
        if not student_answer:
            self.log_fn("Warning: Empty student answer provided")
        if not solution:
            self.log_fn("Warning: Empty official solution provided - grading may be less accurate")
        if not grading_guidelines:
            self.log_fn("Warning: Empty grading guidelines provided - using default IMO 0-7 scale")
        
        # Build the prompt with validated inputs - more concise and structured
        return f"""You are an expert IMO (International Mathematical Olympiad) grader. Evaluate the student's solution below.

[PROBLEM - {domain}]
{problem}

[OFFICIAL SOLUTION]
{solution}

[GRADING GUIDELINES]
{grading_guidelines}

[STUDENT'S ANSWER]
{student_answer}

[EVALUATION INSTRUCTIONS]
1. Analyze the student's answer step-by-step against the official solution
2. Identify errors, missing steps, or valid alternative approaches
3. Apply the grading guidelines (IMO scale: 0-7 points)
4. Provide detailed reasoning, then give the final grade

[OUTPUT FORMAT - STRICT JSON REQUIRED]
Respond ONLY with a JSON object in <json> tags:

<json>
{{
    "reasoning": "Detailed analysis comparing student answer to official solution...",
    "response": "N"
}}
</json>

CRITICAL RULES:
- "response" must be EXACTLY one digit: 0, 1, 2, 3, 4, 5, 6, or 7
- No extra text, spaces, or formatting in the response field
- 7 = full credit, 0 = no credit, 1-6 = partial credit"""

    def _extract_prediction(self, msg_history: list[dict]) -> tuple[str, str]:
        """Extract prediction and reasoning from message history.
        
        Enhanced extraction with multiple fallback strategies for robustness.
        
        Returns:
            (prediction, reasoning)
        """
        prediction = "None"
        reasoning = ""
        
        if not msg_history:
            return prediction, reasoning
        
        try:
            # Get the last assistant message
            last_msg = ""
            for msg in reversed(msg_history):
                if msg.get("role") == "assistant" or "text" in msg:
                    last_msg = msg.get("text", msg.get("content", ""))
                    break
            
            if not last_msg:
                return prediction, reasoning
            
            # Strategy 1: Try <json> tags first (primary format)
            extracted = _extract_jsons(last_msg)
            if extracted:
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = str(last_json["response"]).strip()
                if "reasoning" in last_json:
                    reasoning = str(last_json["reasoning"])
                if prediction != "None":
                    return prediction, reasoning
            
            # Strategy 2: Try markdown code blocks
            md_json = _extract_json_from_markdown(last_msg)
            if md_json:
                if "response" in md_json:
                    prediction = str(md_json["response"]).strip()
                if "reasoning" in md_json:
                    reasoning = str(md_json["reasoning"])
                if prediction != "None":
                    return prediction, reasoning
            
            # Strategy 3: Look for standalone JSON with response field
            json_match = re.search(r'\{[^{}]*"response"\s*:\s*"([^"{}]+)"[^{}]*\}', last_msg)
            if json_match:
                prediction = json_match.group(1).strip()
                if prediction in ["0", "1", "2", "3", "4", "5", "6", "7"]:
                    return prediction, reasoning
            
            # Strategy 4: Direct digit extraction from specific patterns
            patterns = [
                r'"response"\s*:\s*"([0-7])"',
                r'"response"\s*:\s*([0-7])\b',
                r'response[:\s]+([0-7])\b',
                r'grade[:\s]+([0-7])\b',
                r'score[:\s]+([0-7])\b',
                r'\bgrade\s+is\s+([0-7])\b',
                r'\bscore\s+is\s+([0-7])\b',
            ]
            for pattern in patterns:
                match = re.search(pattern, last_msg, re.IGNORECASE)
                if match:
                    prediction = match.group(1)
                    return prediction, reasoning
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
        
        return prediction, reasoning

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with enhanced reasoning, validation, and retry logic.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_prompt(inputs)
        grading_guidelines = inputs.get("grading_guidelines", "")
        
        # Retry configuration
        max_retries = 3
        retry_count = 0
        last_error = None
        
        while retry_count < max_retries:
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                break  # Success, exit retry loop
            except Exception as e:
                last_error = e
                retry_count += 1
                self.log_fn(f"LLM call failed (attempt {retry_count}/{max_retries}): {e}")
                if retry_count >= max_retries:
                    self.log_fn(f"All {max_retries} LLM call attempts failed. Returning 'None'.")
                    return "None", []
                # Brief pause before retry to allow transient issues to resolve
                import time
                time.sleep(0.5 * retry_count)
        
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
        
        # Final validation: ensure we return a valid IMO grade (0-7)
        if not is_valid or validated_grade not in ["0", "1", "2", "3", "4", "5", "6", "7"]:
            self.log_fn(f"Warning: Could not extract valid grade. Defaulting to '0'.")
            validated_grade = "0"

        return str(validated_grade), msg_history
