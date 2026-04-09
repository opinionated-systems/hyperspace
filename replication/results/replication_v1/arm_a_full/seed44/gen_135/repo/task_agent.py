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
from typing import Any

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Case-insensitive tag matching for better robustness.
    Also handles nested JSON objects and common formatting issues.
    """
    results = []
    search_from = 0
    text_lower = text.lower()
    
    while True:
        # Case-insensitive search for opening tag
        start = text_lower.find("<json>", search_from)
        if start == -1:
            break
        # Find closing tag (case-insensitive)
        end = text_lower.find("</json>", start)
        if end == -1:
            break
        
        # Extract from original text (not lowercased)
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        # Try to parse JSON with multiple fallback strategies
        parsed = _try_parse_json(inner)
        if parsed is not None:
            results.append(parsed)
    
    return results or None


def _try_parse_json(text: str) -> dict | None:
    """Try to parse JSON with multiple fallback strategies.
    
    Returns the parsed dict or None if all strategies fail.
    """
    # Strategy 1: Direct parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Remove trailing commas before closing braces/brackets
    try:
        fixed = re.sub(r',(\s*[}\]])', r'\1', text)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Fix single quotes (replace with double quotes)
    try:
        # Replace single quotes with double quotes, but be careful with apostrophes in text
        fixed = re.sub(r"(?<!\\)'", '"', text)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Strategy 4: Extract just the response field if present
    response_match = re.search(r'["\']response["\']\s*:\s*([^,}\]]+)', text, re.IGNORECASE)
    if response_match:
        value_str = response_match.group(1).strip()
        # Try to parse the value
        try:
            # Try as number
            if value_str.replace('.', '', 1).replace('-', '', 1).isdigit():
                if '.' in value_str:
                    value = float(value_str)
                else:
                    value = int(value_str)
            elif value_str.lower() == 'true':
                value = True
            elif value_str.lower() == 'false':
                value = False
            elif value_str.lower() == 'null':
                value = None
            elif (value_str.startswith('"') and value_str.endswith('"')) or \
                 (value_str.startswith("'") and value_str.endswith("'")):
                value = value_str[1:-1]
            else:
                value = value_str
            return {"response": value}
        except Exception:
            pass
    
    return None


def _extract_json_with_regex(text: str) -> list[dict] | None:
    """Fallback JSON extraction using regex for malformed outputs.
    
    Uses multiple strategies to find valid JSON objects in text.
    """
    results = []
    
    # Strategy 1: Try to find JSON objects in code blocks (case-insensitive)
    code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(code_block_pattern, text, re.DOTALL | re.IGNORECASE)
    for match in matches:
        parsed = _try_parse_json(match.strip())
        if parsed is not None:
            results.append(parsed)
    
    # Strategy 2: Try to find JSON objects with "response" field
    if not results:
        # More flexible pattern to catch nested structures with "response" field
        json_pattern = r'\{[\s\S]*?["\']response["\'][\s\S]*?\}'
        matches = re.findall(json_pattern, text, re.IGNORECASE)
        for match in matches:
            parsed = _try_parse_json(match)
            if parsed is not None:
                results.append(parsed)
    
    # Strategy 3: Try to find any valid JSON object by looking for balanced braces
    if not results:
        # Use a more robust approach: find all { and } positions, then try to match them
        def find_balanced_braces(s: str) -> list[str]:
            """Find all balanced brace pairs in the string."""
            brace_pairs = []
            stack = []
            for i, char in enumerate(s):
                if char == '{':
                    stack.append(i)
                elif char == '}':
                    if stack:
                        start = stack.pop()
                        brace_pairs.append((start, i))
            # Return the substrings from outermost braces
            return [s[start:end+1] for start, end in brace_pairs if s[start:end+1].count('{') == s[start:end+1].count('}')]
        
        brace_matches = find_balanced_braces(text)
        for match in brace_matches:
            parsed = _try_parse_json(match)
            if parsed is not None and isinstance(parsed, dict):
                results.append(parsed)
    
    # Strategy 4: Look for simple key-value patterns like {"response": 7}
    if not results:
        simple_pattern = r'\{\s*["\']?response["\']?\s*:\s*([^}]+)\}'
        matches = re.findall(simple_pattern, text, re.IGNORECASE)
        for match in matches:
            value_str = match.strip()
            # Try to parse the value
            try:
                # Try as number first
                if value_str.replace('.', '', 1).replace('-', '', 1).isdigit():
                    if '.' in value_str:
                        value = float(value_str)
                    else:
                        value = int(value_str)
                elif value_str.lower() == 'true':
                    value = True
                elif value_str.lower() == 'false':
                    value = False
                elif value_str.lower() == 'null':
                    value = None
                elif (value_str.startswith('"') and value_str.endswith('"')) or \
                     (value_str.startswith("'") and value_str.endswith("'")):
                    value = value_str[1:-1]
                else:
                    value = value_str
                results.append({"response": value})
            except Exception:
                continue
    
    return results or None


def _extract_response_heuristic(text: str) -> Any | None:
    """Last resort: try to extract a response value using heuristics.
    
    Optimized for IMO grading tasks where scores are typically:
    - Integers 0-7 (IMO problems are worth 7 points)
    - Fractions like "3/7" or "0/1"
    - Specific strings like "correct", "incorrect", "partial"
    """
    # Look for patterns like "response": 7 or "response": "correct"
    # Support both single and double quotes, with flexible whitespace
    patterns = [
        (r'["\']response["\']\s*:\s*(-?\d+(?:\.\d+)?)', 'number'),  # Numbers
        (r'["\']response["\']\s*:\s*["\']([^"\']*)["\']', 'string'),  # Strings
        (r'["\']response["\']\s*:\s*(true|false|null)', 'bool_null'),  # Booleans/null
    ]
    
    for pattern, ptype in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1)
            if ptype == 'number':
                try:
                    if '.' in value:
                        return float(value)
                    return int(value)
                except ValueError:
                    return value
            elif ptype == 'bool_null':
                if value.lower() == 'true':
                    return True
                elif value.lower() == 'false':
                    return False
                elif value.lower() == 'null':
                    return None
            else:
                return value
    
    # Look for score patterns in text (common in grading)
    # Pattern: "score: X", "grade: X", "points: X", "X/7", "X out of 7"
    score_patterns = [
        r'(?:score|grade|points|mark)s?\s*[:=]\s*(\d+(?:\.\d+)?)',
        r'(\d+(?:\.\d+)?)\s*/\s*\d+',  # Fractions like 3/7
        r'(\d+(?:\.\d+)?)\s+out\s+of\s+\d+',
        r'(?:score|grade|points|mark)s?\s+is\s+(\d+(?:\.\d+)?)',
        r'(?:assigned|awarded|given)\s+(?:a\s+)?(?:score|grade|points)?\s*(?:of\s+)?(\d+(?:\.\d+)?)',
        r'(?:final|total)\s+(?:score|grade|points)\s*(?:is\s+)?[:=]?\s*(\d+(?:\.\d+)?)',
    ]
    
    for pattern in score_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1)
            try:
                if '.' in value:
                    return float(value)
                return int(value)
            except ValueError:
                return value
    
    # Try to find fraction patterns like "0/1", "1/1", "3/7", "7/7"
    fraction_pattern = r'\b(\d+/\d+)\b'
    match = re.search(fraction_pattern, text)
    if match:
        return match.group(1)
    
    # Look for IMO-specific score indicators in the last few sentences
    # (often the conclusion contains the final score)
    lines = text.split('\n')
    for line in reversed(lines[-10:]):  # Check last 10 lines
        # Look for standalone numbers in conclusion lines
        conclusion_patterns = [
            r'(?:therefore|thus|hence|conclusion|final|score|grade)[,:]?\s+(\d+)\s*(?:points?)?',
            r'(?:student|answer)\s+(?:gets?|receives?|earns?|deserves?)\s+(\d+)\s*(?:points?)?',
            r'(?:worth|value)\s+(\d+)\s*(?:points?)?',
        ]
        for pattern in conclusion_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                return int(match.group(1))
    
    # Try to find any number that might be a score (0-7 range for IMO)
    # Look for standalone numbers that could be scores
    score_pattern = r'\b([0-7])\b'
    matches = re.findall(score_pattern, text)
    if matches:
        # Return the last number found in 0-7 range (often the final score)
        return int(matches[-1])
    
    # Broader search for any number if no 0-7 found
    any_number_pattern = r'\b(\d+)\b'
    matches = re.findall(any_number_pattern, text)
    if matches:
        # Return the last number found (often the final score in conclusions)
        return int(matches[-1])
    
    # Look for text-based evaluations
    text_patterns = [
        (r'\b(correct|incorrect|partial|full|zero|none)\b', 'string'),
        (r'\b(complete|incomplete|valid|invalid)\b', 'string'),
    ]
    for pattern, ptype in text_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).lower()
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _validate_prediction(self, prediction: Any) -> Any:
        """Validate and normalize the prediction value.
        
        Ensures the prediction is in a valid format for IMO grading.
        """
        if prediction is None or prediction == "None":
            return "None"
        
        # If it's already a string, check if it's a valid format
        if isinstance(prediction, str):
            # Check for fraction format like "3/7" or "0/1"
            if '/' in prediction:
                parts = prediction.split('/')
                if len(parts) == 2:
                    try:
                        numerator = float(parts[0])
                        denominator = float(parts[1])
                        return prediction  # Keep as string
                    except ValueError:
                        pass
            # Try to convert to number
            try:
                if '.' in prediction:
                    return float(prediction)
                return int(prediction)
            except ValueError:
                # Keep as string if not convertible
                return prediction
        
        # Numbers are valid
        if isinstance(prediction, (int, float)):
            return prediction
        
        # Booleans
        if isinstance(prediction, bool):
            return prediction
        
        # Default to string representation
        return str(prediction)

    def _attempt_self_correction(self, raw_response: str, msg_history: list[dict]) -> Any:
        """Attempt to extract a valid response using a follow-up LLM call.
        
        This is used when the initial response doesn't contain valid JSON.
        """
        correction_prompt = f"""The previous response did not contain valid JSON. Please extract or provide the final grading score.

Previous response:
```
{raw_response[:2000]}
```

Based on the above response, what is the final score or evaluation? 

Respond ONLY with valid JSON in this exact format:
<json>
{{
    "response": <score or evaluation>
}}
</json>

The response value should be a number (typically 0-7 for IMO problems) or a string like "3/7" or "0/1".
"""
        try:
            correction_response, correction_history, _ = get_response_from_llm(
                msg=correction_prompt,
                model=self.model,
                msg_history=[],
            )
            
            # Try to extract from the correction response
            extracted = _extract_jsons(correction_response)
            if extracted and len(extracted) > 0:
                last_result = extracted[-1]
                if isinstance(last_result, dict) and "response" in last_result:
                    return last_result["response"]
            
            # Try regex extraction
            extracted = _extract_json_with_regex(correction_response)
            if extracted and len(extracted) > 0:
                last_result = extracted[-1]
                if isinstance(last_result, dict) and "response" in last_result:
                    return last_result["response"]
            
            # Try heuristic extraction
            heuristic_value = _extract_response_heuristic(correction_response)
            if heuristic_value is not None:
                return heuristic_value
                
        except Exception as e:
            self.log_fn(f"Self-correction attempt failed: {e}")
        
        return "None"

    def _build_grading_prompt(
        self,
        domain: str,
        problem: str,
        solution: str,
        grading_guidelines: str,
        student_answer: str,
    ) -> str:
        """Build the grading prompt with clear instructions."""
        return f"""You are an expert IMO (International Mathematical Olympiad) grader evaluating student solutions to competition mathematics problems.

Your task is to grade a student's answer to an IMO-level mathematics problem. You must carefully analyze:
1. The problem statement
2. The official solution
3. The grading guidelines (rubric)
4. The student's submitted answer

Then provide your evaluation in the specified JSON format.

PROBLEM DOMAIN: {domain}

PROBLEM STATEMENT:
```
{problem}
```

OFFICIAL SOLUTION:
```
{solution}
```

GRADING GUIDELINES (RUBRIC):
```
{grading_guidelines}
```

STUDENT'S ANSWER:
```
{student_answer}
```

INSTRUCTIONS:
1. Read the problem carefully and understand what is being asked.
2. Study the official solution to understand the correct approach.
3. Review the grading guidelines to understand how points are awarded.
4. Analyze the student's answer step by step:
   - Did they understand the problem correctly?
   - Did they use the right approach?
   - Are their calculations correct?
   - Did they provide a complete proof/solution?
   - Where did they make errors, if any?
5. Assign a score based EXACTLY on the grading guidelines provided.

IMO GRADING PRINCIPLES:
- IMO problems are typically worth 7 points maximum
- Partial credit is awarded for significant progress toward the solution
- A complete, correct proof receives full marks (7 points)
- Minor errors may result in point deductions
- The grading guidelines specify exactly how points should be allocated
- Follow the rubric precisely - do not deviate from the specified scoring scheme
- Be consistent with standard IMO grading practices
- Look for key insights and partial progress that merit partial credit
- Consider both the correctness of the final answer and the quality of the reasoning

IMPORTANT: Your response MUST be valid JSON wrapped in <json> tags.

Respond in JSON format with the following schema:
<json>
{{
    "response": <your numerical score or evaluation>
}}
</json>

The "response" field should contain your final grading decision (typically a number or specific evaluation string as specified in the grading guidelines).

Examples of valid responses:
<json>
{{
    "response": 7
}}
</json>

<json>
{{
    "response": "0/1"
}}
</json>

<json>
{{
    "response": 3
}}
</json>

<json>
{{
    "response": "1/7"
}}
</json>

CRITICAL: Ensure your JSON is properly formatted with:
- Double quotes around keys and string values (not single quotes)
- No trailing commas
- Valid JSON syntax
- The value can be a number, string, boolean, or null

Wrap your entire JSON response in <json>...</json> tags.

Remember: Your goal is to provide an accurate, fair grade based on the official grading guidelines."""

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Validate inputs
        if not isinstance(inputs, dict):
            self.log_fn(f"Invalid inputs type: {type(inputs)}")
            return "None", [{"role": "error", "text": f"Invalid inputs type: {type(inputs)}"}]
        
        # Extract key fields for better prompt construction
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Check for empty required fields
        if not problem or not student_answer:
            self.log_fn("Missing required fields: problem or student_answer")
            return "None", [{"role": "error", "text": "Missing required fields"}]

        instruction = self._build_grading_prompt(
            domain, problem, solution, grading_guidelines, student_answer
        )

        # Retry logic for LLM calls
        max_retries = 3
        last_error = None
        msg_history = []
        
        for attempt in range(max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                break
            except Exception as e:
                last_error = e
                self.log_fn(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return "None", [{"role": "error", "text": str(last_error)}]
                continue

        # Extract prediction from JSON with multiple fallback strategies
        prediction = "None"
        raw_response = msg_history[-1]["text"] if msg_history else ""
        
        # Log the raw response for debugging
        self.log_fn(f"Raw response length: {len(raw_response)} chars")
        
        extraction_attempts = [
            ("json_tags", lambda: _extract_jsons(raw_response)),
            ("regex", lambda: _extract_json_with_regex(raw_response)),
            ("heuristic", lambda: [{"response": _extract_response_heuristic(raw_response)}] if _extract_response_heuristic(raw_response) is not None else None),
        ]
        
        for method_name, attempt_fn in extraction_attempts:
            try:
                extracted = attempt_fn()
                if extracted and isinstance(extracted, list) and len(extracted) > 0:
                    last_result = extracted[-1]
                    if isinstance(last_result, dict) and "response" in last_result:
                        prediction = last_result["response"]
                        self.log_fn(f"Successfully extracted prediction using {method_name}: {prediction}")
                        break
            except Exception as e:
                self.log_fn(f"Extraction method {method_name} failed: {e}")
                continue
        
        # If extraction failed, try self-correction with the LLM
        if prediction == "None" and raw_response:
            self.log_fn(f"Initial extraction failed. Attempting self-correction...")
            correction_prediction = self._attempt_self_correction(raw_response, msg_history)
            if correction_prediction != "None":
                prediction = correction_prediction
                self.log_fn(f"Self-correction successful: {prediction}")
        
        # Validate the prediction
        prediction = self._validate_prediction(prediction)
        
        if prediction == "None":
            self.log_fn(f"Failed to extract prediction from response. Raw text (first 1000 chars): {raw_response[:1000]}")
            # Try to log more context for debugging
            if raw_response:
                self.log_fn(f"Response length: {len(raw_response)} chars")

        return str(prediction), msg_history
