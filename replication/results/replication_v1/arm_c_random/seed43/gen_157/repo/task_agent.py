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


def _fix_json(json_str: str) -> str | None:
    """Fix common JSON formatting issues from LLM outputs.
    
    Fixes:
    - Trailing commas before closing braces/brackets
    - Single quotes instead of double quotes
    
    Returns fixed string or None if unfixable.
    """
    try:
        # Remove trailing commas before closing braces/brackets
        fixed = re.sub(r',(\s*[}\]])', r'\1', json_str)
        # Fix single quotes to double quotes (common LLM error)
        fixed = re.sub(r"'([^']*?)'", r'"\1"', fixed)
        return fixed
    except Exception:
        return None


def _try_parse_json(json_str: str) -> dict | None:
    """Try to parse JSON string with automatic fixing.
    
    Returns parsed dict or None if parsing fails.
    """
    try:
        return json.loads(json_str.strip())
    except json.JSONDecodeError:
        fixed = _fix_json(json_str)
        if fixed:
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass
    return None


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract all JSON objects from text using multiple strategies.
    
    Searches for JSON in:
    1. <json>...</json> tags (most reliable)
    2. ```json ... ``` markdown blocks
    3. Plain ``` ... ``` blocks containing JSON-like content
    4. Inline JSON-like objects in text
    
    Uses index-based search to avoid regex truncation issues with nested braces.
    Includes automatic fixing of common LLM JSON errors (trailing commas, single quotes).
    """
    results = []
    
    # Strategy 1: <json>...</json> tags
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
        parsed = _try_parse_json(inner)
        if parsed:
            results.append(parsed)
    
    # Strategy 2: ```json ... ``` blocks
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
        parsed = _try_parse_json(inner)
        if parsed:
            results.append(parsed)
    
    # Strategy 3: Plain ``` blocks with JSON content
    plain_search_from = 0
    while True:
        start = text.find("```", plain_search_from)
        if start == -1:
            break
        end = text.find("```", start + 3)
        if end == -1:
            break
        inner = text[start + 3:end].strip()
        plain_search_from = end + 3
        # Only try if it looks like JSON (starts with {)
        if inner.startswith('{'):
            parsed = _try_parse_json(inner)
            if parsed:
                results.append(parsed)
    
    return results or None


def _extract_json_from_markdown(text: str) -> dict | None:
    """Extract first JSON object from markdown code blocks or inline text.
    
    Searches in order of reliability:
    1. ```json ... ``` blocks
    2. Plain ``` ... ``` blocks with JSON
    3. Inline JSON-like objects
    
    Returns first successfully parsed JSON dict or None.
    """
    # Strategy 1: ```json ... ``` blocks
    pattern = r'```json\s*(.*?)\s*```'
    for match in re.findall(pattern, text, re.DOTALL):
        parsed = _try_parse_json(match)
        if parsed:
            return parsed
    
    # Strategy 2: Plain ``` blocks with JSON
    plain_pattern = r'```\s*(\{[\s\S]*?\})\s*```'
    for match in re.findall(plain_pattern, text, re.DOTALL):
        parsed = _try_parse_json(match)
        if parsed:
            return parsed
    
    # Strategy 3: Inline JSON-like objects
    json_like_pattern = r'\{[^{}]*"[^"]+"[^{}]*\}'
    for match in re.findall(json_like_pattern, text, re.DOTALL):
        parsed = _try_parse_json(match)
        if parsed:
            return parsed
    
    return None


def _validate_grade(prediction: str, grading_guidelines: str) -> tuple[str, bool]:
    """Validate that the extracted grade is reasonable.
    
    Enhanced validation with support for more grade formats and
    better handling of edge cases. Includes improved pattern matching
    for various grading expressions and better handling of edge cases.
    
    Returns:
        (validated_grade, is_valid)
    """
    if not prediction or prediction == "None":
        return "None", False
    
    pred_clean = prediction.strip()
    pred_lower = pred_clean.lower()
    
    # Direct numeric match (0-7 for IMO problems)
    if pred_clean.isdigit():
        grade = int(pred_clean)
        if 0 <= grade <= 7:
            return str(grade), True
        return pred_clean, False
    
    # Check for numeric grades embedded in text (word boundaries)
    numeric_match = re.search(r'\b([0-7])\b', pred_clean)
    if numeric_match:
        return numeric_match.group(1), True
    
    # Check for "X out of 7" or "X/7" patterns with more variations
    out_of_patterns = [
        r'([0-7])\s*(?:out\s+of|/)\s*7',
        r'grade\s*:?\s*([0-7])\s*/\s*7',
        r'score\s*:?\s*([0-7])\s*/\s*7',
        r'([0-7])\s*points?\s*/\s*7',
    ]
    for pattern in out_of_patterns:
        match = re.search(pattern, pred_lower)
        if match:
            return match.group(1), True
    
    # Check for partial credit patterns with more variations
    partial_patterns = [
        r'partial\s*(?:credit)?\s*:?\s*([0-7])',
        r'partially\s*(?:correct)?\s*:?\s*([0-7])',
        r'partial\s*score\s*:?\s*([0-7])',
        r'([0-7])\s*points?\s*(?:partial)?',
        r'partial\s*mark\s*:?\s*([0-7])',
        r'partial\s*grade\s*:?\s*([0-7])',
    ]
    for pattern in partial_patterns:
        partial_match = re.search(pattern, pred_lower)
        if partial_match:
            return partial_match.group(1), True
    
    # Check for full credit patterns (expanded list)
    full_patterns = [
        'full credit', 'full marks', 'complete', 'perfect', '7/7', 'full score',
        'maximum', 'max points', 'all points', 'full solution', 'entirely correct',
        'completely correct', 'totally correct', '100%'
    ]
    for pattern in full_patterns:
        if pattern in pred_lower:
            return "7", True
    
    # Check for zero/incorrect patterns (expanded list)
    zero_patterns = [
        'zero', 'no credit', '0/7', 'none', 'incorrect', 'wrong', 'invalid', 'empty',
        'no marks', 'no points', '0 points', 'nothing', 'blank', 'no solution',
        'completely wrong', 'totally wrong', 'entirely wrong', '0%'
    ]
    for pattern in zero_patterns:
        if pattern in pred_lower:
            return "0", True
    
    # Check for "correct" (implies full marks unless specified otherwise)
    if 'correct' in pred_lower and 'partial' not in pred_lower and 'incorrect' not in pred_lower:
        return "7", True
    
    # Check for N/A or not applicable
    if any(x in pred_lower for x in ['n/a', 'not applicable', 'ungradable', 'cannot grade']):
        return "N/A", True
    
    # Check for written-out numbers that might indicate grades
    written_numbers = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7'
    }
    for word, num in written_numbers.items():
        # Match as whole word to avoid partial matches
        if re.search(rf'\b{word}\b', pred_lower):
            return num, True
    
    # If no clear grade found, mark as invalid but return the original
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
        
        Enhanced extraction with multiple fallback strategies and
        better handling of various response formats. Includes improved
        pattern matching and better handling of edge cases.
        
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
            
            # Try to find JSON objects with response/grade fields
            # Look for patterns like {"response": "...", "reasoning": "..."}
            json_patterns = [
                r'\{[^}]*"response"[^}]*"reasoning"[^}]*\}',
                r'\{[^}]*"reasoning"[^}]*"response"[^}]*\}',
                r'\{[^}]*"grade"[^}]*\}',
                r'\{[^}]*"score"[^}]*\}',
                r'\{[^}]*"evaluation"[^}]*\}',
            ]
            for pattern in json_patterns:
                for match in re.finditer(pattern, last_msg, re.DOTALL):
                    try:
                        parsed = json.loads(match.group())
                        if "response" in parsed:
                            prediction = str(parsed["response"]).strip()
                        elif "grade" in parsed:
                            prediction = str(parsed["grade"]).strip()
                        elif "score" in parsed:
                            prediction = str(parsed["score"]).strip()
                        elif "evaluation" in parsed:
                            prediction = str(parsed["evaluation"]).strip()
                        if "reasoning" in parsed:
                            reasoning = str(parsed["reasoning"])
                        if prediction != "None":
                            return prediction, reasoning
                    except json.JSONDecodeError:
                        continue
            
            # Look for explicit grade/score declarations in text (expanded patterns)
            text_patterns = [
                r'(?:final\s+)?(?:grade|score|mark|evaluation)\s*:?\s*["\']?([0-7]|partial\s*(?:credit)?\s*:?\s*[0-7]|full|complete|correct|incorrect|zero|none)["\']?',
                r'(?:the\s+)?(?:student\s+)?(?:received|got|earned|deserves)\s*:?\s*["\']?([0-7])["\']?',
                r'(?:award|assign|give)\s*:?\s*["\']?([0-7])["\']?\s*(?:points?)?',
                r'(?:grade|score)\s+(?:is|of|equals?)\s*:?\s*["\']?([0-7])["\']?',
                r'(?:i\s+(?:would\s+)?(?:assign|give|award))\s*:?\s*["\']?([0-7])["\']?',
                r'(?:this\s+(?:deserves|warrants|merits))\s*:?\s*["\']?([0-7])["\']?',
            ]
            for pattern in text_patterns:
                match = re.search(pattern, last_msg, re.IGNORECASE)
                if match:
                    prediction = match.group(1).strip()
                    break
            
            # Extract reasoning if available (look for "Reasoning:" or "Analysis:" sections)
            if not reasoning:
                reasoning_patterns = [
                    r'(?:reasoning|analysis|explanation|rationale)\s*:?\s*(.*?)(?:\n\n|\Z)',
                    r'(?:step\s+by\s+step|detailed)\s+(?:analysis|reasoning)\s*:?\s*(.*?)(?:\n\n|\Z)',
                    r'(?:evaluation|assessment)\s*:?\s*(.*?)(?:\n\n|\Z)',
                    r'(?:justification|explanation)\s*:?\s*(.*?)(?:\n\n|\Z)',
                ]
                for pattern in reasoning_patterns:
                    reasoning_match = re.search(pattern, last_msg, re.IGNORECASE | re.DOTALL)
                    if reasoning_match:
                        reasoning = reasoning_match.group(1).strip()[:500]  # Limit length
                        break
            
            # If still no prediction, try to find any standalone number 0-7
            if prediction == "None":
                # Look for standalone digits at the end of the message
                end_match = re.search(r'\b([0-7])\b\s*$', last_msg.strip())
                if end_match:
                    prediction = end_match.group(1)
                    
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
        
        # Multi-level fallback extraction if grade is invalid
        if not is_valid:
            # Level 1: Try to extract from the full response text
            if response:
                # Try to find any numeric grade (0-7) in the response
                numeric_matches = re.findall(r'\b([0-7])\b', response)
                if numeric_matches:
                    # Use the last numeric grade found (usually the final decision)
                    validated_grade = numeric_matches[-1]
                    is_valid = True
                    self.log_fn(f"Fallback L1: Found grade {validated_grade} in response text")
            
            # Level 2: Try to find grade keywords in the response
            if not is_valid and response:
                response_lower = response.lower()
                if any(x in response_lower for x in ['full credit', 'full marks', 'perfect', 'complete solution']):
                    validated_grade = "7"
                    is_valid = True
                    self.log_fn("Fallback L2: Detected full credit")
                elif any(x in response_lower for x in ['zero', 'no credit', 'completely wrong', 'no solution']):
                    validated_grade = "0"
                    is_valid = True
                    self.log_fn("Fallback L2: Detected zero/no credit")
                elif 'partial' in response_lower:
                    # Try to find a number near "partial"
                    partial_match = re.search(r'partial.*?([0-7])', response_lower)
                    if partial_match:
                        validated_grade = partial_match.group(1)
                    else:
                        validated_grade = "3"  # Default partial credit
                    is_valid = True
                    self.log_fn(f"Fallback L2: Detected partial credit: {validated_grade}")
            
            # Level 3: Use grading guidelines to make a heuristic decision
            if not is_valid and grading_guidelines:
                # If we can't determine the grade, default to a conservative estimate
                # based on whether the student answer has content
                student_answer = inputs.get("student_answer", "")
                if not student_answer or len(student_answer.strip()) < 10:
                    validated_grade = "0"
                    self.log_fn("Fallback L3: Empty/very short answer, defaulting to 0")
                else:
                    # Default to middle grade when uncertain
                    validated_grade = "3"
                    self.log_fn("Fallback L3: Uncertain grade, defaulting to 3")
                is_valid = True

        return str(validated_grade), msg_history
