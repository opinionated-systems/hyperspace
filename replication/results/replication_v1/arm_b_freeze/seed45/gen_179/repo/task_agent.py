"""
Task agent: solves a given task with chain-of-thought reasoning.

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
    Also handles markdown code blocks and bare JSON objects.
    Includes robust handling for nested braces and common JSON formatting issues.
    """
    results = []
    search_from = 0
    
    # First try to find <json>...</json> blocks
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
            # Try to extract JSON from within the content using brace matching
            try:
                json_str = _extract_json_with_brace_matching(inner)
                if json_str:
                    results.append(json.loads(json_str))
            except (json.JSONDecodeError, ValueError):
                continue
    
    # If no <json> blocks found, try markdown code blocks
    if not results:
        # Try ```json ... ``` blocks
        json_blocks = re.findall(r'```json\s*(.*?)```', text, re.DOTALL)
        for block in json_blocks:
            try:
                results.append(json.loads(block.strip()))
            except json.JSONDecodeError:
                # Try brace matching as fallback
                try:
                    json_str = _extract_json_with_brace_matching(block)
                    if json_str:
                        results.append(json.loads(json_str))
                except (json.JSONDecodeError, ValueError):
                    continue
        
        # Try bare JSON objects as fallback
        if not results:
            # Use brace matching to find JSON objects
            json_str = _extract_json_with_brace_matching(text)
            if json_str:
                try:
                    results.append(json.loads(json_str))
                except json.JSONDecodeError:
                    pass
    
    # Try to find all JSON objects in the text using brace matching
    if not results:
        json_objects = _extract_all_json_objects(text)
        for obj in json_objects:
            try:
                results.append(json.loads(obj))
            except json.JSONDecodeError:
                continue
    
    return results or None


def _extract_all_json_objects(text: str) -> list[str]:
    """Extract all JSON objects from text using brace counting.
    
    This handles nested braces correctly by counting open/close braces.
    Returns a list of all valid JSON object strings found.
    """
    results = []
    i = 0
    
    while i < len(text):
        # Find the next opening brace
        start = text.find("{", i)
        if start == -1:
            break
        
        brace_count = 0
        in_string = False
        escape_next = False
        
        for j, char in enumerate(text[start:], start):
            if escape_next:
                escape_next = False
                continue
            
            if char == "\\" and in_string:
                escape_next = True
                continue
            
            if char == '"' and not in_string:
                in_string = True
            elif char == '"' and in_string:
                in_string = False
            elif not in_string:
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        # Found a complete JSON object
                        results.append(text[start:j + 1])
                        i = j + 1
                        break
        else:
            # No complete JSON object found, move past this brace
            i = start + 1
    
    return results


def _extract_json_with_brace_matching(text: str) -> str | None:
    """Extract a JSON object from text using brace counting.
    
    This handles nested braces correctly by counting open/close braces.
    Returns the first valid JSON object found, or None if no valid JSON.
    """
    start = text.find("{")
    if start == -1:
        return None
    
    brace_count = 0
    in_string = False
    escape_next = False
    
    for i, char in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        
        if char == "\\" and in_string:
            escape_next = True
            continue
        
        if char == '"' and not in_string:
            in_string = True
        elif char == '"' and in_string:
            in_string = False
        elif not in_string:
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    # Found a complete JSON object
                    return text[start:i + 1]
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 2  # Max retries for JSON extraction failures
        self._response_keys = ["response", "grade", "answer", "result", "assessment", "evaluation", "score"]
        self._reasoning_keys = ["reasoning", "analysis", "thought", "explanation", "rationale", "thinking"]

    def _extract_value(self, result: dict, keys: list[str], default: str) -> str:
        """Extract a value from result dict using multiple possible keys.
        
        Args:
            result: The JSON result dictionary
            keys: List of possible keys to try
            default: Default value if no key is found
            
        Returns:
            The extracted value or default
        """
        for key in keys:
            if key in result:
                value = result[key]
                # Handle both string and non-string values
                if isinstance(value, str):
                    return value.strip()
                else:
                    return str(value)
        return default

    def _normalize_grade(self, grade: str) -> str:
        """Normalize grade to a standard format for consistency.
        
        Args:
            grade: The raw grade string
            
        Returns:
            Normalized grade string (Correct, Incorrect, Partially Correct, or original)
        """
        if not grade or grade == "None":
            return grade
            
        grade_lower = grade.lower().strip()
        
        # Map common variations to standard forms
        correct_variants = [
            'correct', 'right', 'true', 'yes', 'full', 'full credit', '100%', 'pass', 
            '1/1', '1', 'full marks', 'complete', 'accurate', 'valid', 'accepted',
            'satisfactory', 'excellent', 'good', 'perfect', 'all correct'
        ]
        incorrect_variants = [
            'incorrect', 'wrong', 'false', 'no', 'none', 'fail', 'zero', '0/1', '0',
            'unsatisfactory', 'invalid', 'rejected', 'error', 'mistake', 'bad',
            'not correct', 'not right', 'not valid', 'not acceptable'
        ]
        partial_variants = [
            'partial', 'partially correct', 'partial credit', 'half', '0.5', '50%', '1/2',
            'incomplete', 'mostly correct', 'some correct', 'partially right',
            'partial marks', 'partial success', 'partially valid', 'partially acceptable'
        ]
        
        # Check for exact matches first
        if grade_lower in correct_variants:
            return 'Correct'
        if grade_lower in incorrect_variants:
            return 'Incorrect'
        if grade_lower in partial_variants:
            return 'Partially Correct'
            
        # Check for partial matches (contains or starts/ends with)
        if any(v in grade_lower for v in correct_variants):
            return 'Correct'
        elif any(v in grade_lower for v in incorrect_variants):
            return 'Incorrect'
        elif any(v in grade_lower for v in partial_variants):
            return 'Partially Correct'
        
        # Check for numeric grades
        try:
            # Try to extract numeric value
            import re
            numeric_match = re.search(r'(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)', grade)
            if numeric_match:
                numerator = float(numeric_match.group(1))
                denominator = float(numeric_match.group(2))
                if denominator > 0:
                    ratio = numerator / denominator
                    if ratio >= 0.9:
                        return 'Correct'
                    elif ratio <= 0.1:
                        return 'Incorrect'
                    else:
                        return 'Partially Correct'
            
            # Check for percentage
            percent_match = re.search(r'(\d+(?:\.\d+)?)\s*%', grade)
            if percent_match:
                percent = float(percent_match.group(1))
                if percent >= 90:
                    return 'Correct'
                elif percent <= 10:
                    return 'Incorrect'
                else:
                    return 'Partially Correct'
        except (ValueError, ZeroDivisionError):
            pass
        
        # Return original if no match
        return grade

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with chain-of-thought reasoning.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract fields for better prompting
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert {domain} grader evaluating student solutions.

Your task is to grade a student's answer to a problem by comparing it against the official solution and following the grading guidelines.

## Problem Statement:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Instructions:
1. First, analyze the student's answer step by step. Identify what they did correctly and incorrectly.
2. Compare their approach to the official solution.
3. Consider the grading guidelines carefully - these are the specific criteria for grading.
4. Provide your reasoning before giving the final grade.
5. Respond ONLY in JSON format wrapped in <json>...</json> tags with the following exact schema:

<json>
{{
    "reasoning": "Your detailed analysis and reasoning about the student's answer...",
    "response": "The final grade/assessment (e.g., 'Correct', 'Partially Correct', 'Incorrect', or a numeric score)"
}}
</json>

## Grading Standards:
- **Correct**: The student's answer is fully correct, matches the official solution, and follows all requirements.
- **Partially Correct**: The student's answer has some correct elements but is incomplete, has minor errors, or partially matches the official solution.
- **Incorrect**: The student's answer is wrong, does not match the official solution, or violates the grading guidelines.

## Important Notes:
- Be lenient with minor formatting differences (e.g., "2x" vs "2*x" vs "2 * x")
- Consider alternative valid approaches that arrive at the correct answer
- Check if the student's reasoning is sound even if the final answer format differs
- For partial credit, consider what percentage of the solution is correct

IMPORTANT:
- Your entire response must be valid JSON inside <json>...</json> tags
- Do not include any text outside the JSON tags
- The "reasoning" field should contain your detailed analysis
- The "response" field should contain only the final grade/assessment
- Ensure the JSON is properly formatted with no syntax errors
- Use standard grade values: "Correct", "Partially Correct", or "Incorrect"

Think carefully and provide a fair assessment based on the official solution and grading guidelines."""

        msg_history = []
        prediction = "None"
        reasoning = ""
        
        for attempt in range(self.max_retries + 1):
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=msg_history,
            )

            # Extract prediction from JSON
            extracted = None
            try:
                # Try the last assistant message
                last_msg = msg_history[-1]["text"] if msg_history else ""
                extracted = _extract_jsons(last_msg)
                
                # If not found, try searching all messages
                if not extracted:
                    for msg in reversed(msg_history):
                        text = msg.get("text", "")
                        extracted = _extract_jsons(text)
                        if extracted:
                            break
                
                if extracted:
                    result = extracted[-1]
                    
                    # Try multiple possible keys for the response
                    prediction = self._extract_value(result, self._response_keys, "None")
                    
                    # Normalize the grade for consistency
                    prediction = self._normalize_grade(prediction)
                    
                    # Extract reasoning if available
                    reasoning = self._extract_value(result, self._reasoning_keys, "")
                    
                    # Log reasoning if available
                    if reasoning:
                        self.log_fn(f"Reasoning: {reasoning[:200]}...")
                    
                    # Log the normalized prediction
                    self.log_fn(f"Prediction: {prediction}")
                    
                    # Success - break out of retry loop
                    break
                elif attempt < self.max_retries:
                    # No JSON found - add feedback and retry
                    self.log_fn(f"No JSON found in attempt {attempt + 1}, retrying with feedback...")
                    feedback = (
                        "ERROR: Your previous response did not contain valid JSON in the required format.\n\n"
                        "You MUST respond with a JSON object wrapped in <json>...</json> tags.\n\n"
                        "Example of correct format:\n"
                        "<json>\n"
                        '{\n'
                        '    "reasoning": "The student correctly solved the equation by factoring...",\n'
                        '    "response": "Correct"\n'
                        '}\n'
                        "</json>\n\n"
                        "Requirements:\n"
                        "1. Use <json>...</json> tags around the entire JSON object\n"
                        "2. The JSON must have exactly two fields: 'reasoning' and 'response'\n"
                        "3. 'reasoning' should contain your detailed analysis\n"
                        "4. 'response' should be one of: 'Correct', 'Partially Correct', or 'Incorrect'\n"
                        "5. Do not include any text outside the <json> tags\n"
                        "6. Ensure proper JSON syntax with quotes around keys and string values"
                    )
                    msg_history.append({"role": "user", "text": feedback})
                    instruction = feedback  # Update instruction for next iteration
                else:
                    self.log_fn(f"No JSON found after {self.max_retries + 1} attempts")
                    
            except Exception as e:
                self.log_fn(f"Error extracting prediction (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries:
                    feedback = (
                        f"ERROR: Error parsing your response: {e}\n\n"
                        "Please ensure your response contains valid JSON wrapped in <json>...</json> tags.\n\n"
                        "Example of correct format:\n"
                        "<json>\n"
                        '{\n'
                        '    "reasoning": "The student correctly solved the equation by factoring...",\n'
                        '    "response": "Correct"\n'
                        '}\n'
                        "</json>\n\n"
                        "Common errors to avoid:\n"
                        "- Missing quotes around keys or string values\n"
                        "- Trailing commas in JSON objects\n"
                        "- Unescaped quotes within string values\n"
                        "- Text outside the <json> tags"
                    )
                    msg_history.append({"role": "user", "text": feedback})
                    instruction = feedback

        return str(prediction), msg_history
