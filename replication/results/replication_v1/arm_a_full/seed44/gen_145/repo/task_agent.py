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
import time

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)

# Maximum retries for LLM calls
MAX_RETRIES = 3
# Delay between retries (seconds)
RETRY_DELAY = 1.0


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles nested JSON objects within the content.
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
        
        # Try to parse the inner content as JSON
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # If direct parsing fails, try to extract JSON using brace matching
            # This handles cases where there might be nested structures
            try:
                json_obj = _extract_json_with_brace_matching(inner)
                if json_obj:
                    results.append(json_obj)
            except Exception:
                continue
    return results or None


def _extract_json_with_brace_matching(text: str) -> dict | None:
    """Extract a JSON object from text using brace counting.
    
    This handles nested JSON objects properly by tracking brace depth.
    Also handles strings to avoid counting braces inside string literals.
    """
    start_idx = text.find('{')
    if start_idx == -1:
        return None
    
    brace_count = 1
    i = start_idx + 1
    in_string = False
    escape_next = False
    
    while i < len(text) and brace_count > 0:
        char = text[i]
        
        if escape_next:
            escape_next = False
        elif char == '\\' and in_string:
            escape_next = True
        elif char == '"' and not in_string:
            in_string = True
        elif char == '"' and in_string:
            in_string = False
        elif not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
        
        i += 1
    
    if brace_count == 0:
        candidate = text[start_idx:i]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # Try cleaning up common JSON issues
            try:
                # Remove trailing commas before } or ]
                cleaned = re.sub(r',(\s*[}\]])', r'\1', candidate)
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass
    return None


def _extract_json_with_regex(text: str) -> list[dict] | None:
    """Fallback JSON extraction using regex for malformed outputs.
    
    Handles nested JSON objects by using a stack-based brace matching approach
    instead of simple regex that fails on nested structures.
    Also handles strings to avoid counting braces inside string literals.
    """
    results = []
    # Try to find JSON objects in code blocks
    code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            # Try with trailing comma cleanup
            try:
                cleaned = re.sub(r',(\s*[}\]])', r'\1', match.strip())
                results.append(json.loads(cleaned))
            except json.JSONDecodeError:
                continue
    
    # Try to find JSON objects with curly braces using stack-based matching
    # This handles nested objects and strings properly
    if not results:
        i = 0
        while i < len(text):
            # Find the start of a potential JSON object
            if text[i] == '{':
                start = i
                brace_count = 1
                i += 1
                in_string = False
                escape_next = False
                
                # Track braces to find the matching closing brace
                # Handle strings to avoid counting braces inside string literals
                while i < len(text) and brace_count > 0:
                    char = text[i]
                    
                    if escape_next:
                        escape_next = False
                    elif char == '\\' and in_string:
                        escape_next = True
                    elif char == '"' and not in_string:
                        in_string = True
                    elif char == '"' and in_string:
                        in_string = False
                    elif not in_string:
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                    i += 1
                
                # If we found a complete object with "response" key, try to parse it
                if brace_count == 0:
                    candidate = text[start:i]
                    if '"response"' in candidate:
                        try:
                            results.append(json.loads(candidate))
                        except json.JSONDecodeError:
                            # Try with trailing comma cleanup
                            try:
                                cleaned = re.sub(r',(\s*[}\]])', r'\1', candidate)
                                results.append(json.loads(cleaned))
                            except json.JSONDecodeError:
                                continue
            else:
                i += 1
    
    return results or None


def _extract_response_value(text: str) -> str | None:
    """Last-resort extraction: try to find a response value directly.
    
    This handles cases where the JSON is malformed but we can still
    extract the value after "response": using regex patterns.
    Includes enhanced handling for nested quotes and special characters.
    """
    # Pattern to match "response": followed by a value (number, string, or boolean)
    # Handles: "response": 7, "response": "7", "response": true, etc.
    # Enhanced patterns to handle more edge cases
    patterns = [
        # Number (integer or float) - most specific first
        r'"response"\s*:\s*(-?\d+(?:\.\d+)?)(?:\s*[,}\]])',
        # String in double quotes (handles escaped quotes)
        r'"response"\s*:\s*"((?:[^"\\]|\\.)*)"',
        # String in single quotes
        r"'response'\s*:\s*'([^']*)'",
        # Boolean or null
        r'"response"\s*:\s*(true|false|null)(?:\s*[,}\]])',
        # Fallback: number without following delimiter
        r'"response"\s*:\s*(-?\d+(?:\.\d+)?)',
        # Fallback: boolean/null without following delimiter
        r'"response"\s*:\s*(true|false|null)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1)
            # Unescape any escaped characters in string values
            if '"' in text[match.start():match.start() + 20]:
                value = value.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
            # Try to convert to appropriate type
            try:
                # Try integer first
                return str(int(value))
            except ValueError:
                try:
                    # Try float
                    return str(float(value))
                except ValueError:
                    # Return as string (stripped of surrounding whitespace)
                    return value.strip()
    
    return None


def _validate_prediction(prediction: str, grading_guidelines: str) -> bool:
    """Validate that the prediction is reasonable based on grading guidelines.
    
    Checks if the prediction is a valid numeric score or matches expected
    evaluation strings from the grading guidelines.
    
    Args:
        prediction: The extracted prediction value
        grading_guidelines: The grading guidelines text
        
    Returns:
        True if prediction appears valid, False otherwise
    """
    if prediction is None or prediction == "None":
        return False
    
    # Check if it's a valid number
    try:
        float(prediction)
        return True
    except ValueError:
        pass
    
    # Check if it's a boolean-like value
    if prediction.lower() in ("true", "false", "correct", "incorrect", "yes", "no"):
        return True
    
    # Check if it appears in the grading guidelines (case-insensitive)
    pred_lower = prediction.lower()
    guidelines_lower = grading_guidelines.lower()
    
    # Look for common evaluation terms in guidelines
    if pred_lower in guidelines_lower:
        return True
    
    # Check for partial matches with common grading terms
    common_terms = ["correct", "incorrect", "partial", "full", "zero", "none", "pass", "fail"]
    for term in common_terms:
        if term in pred_lower:
            return True
    
    return False


def _get_retry_prompt(original_instruction: str, previous_response: str) -> str:
    """Generate a retry prompt when the previous response was malformed.
    
    Args:
        original_instruction: The original grading instruction
        previous_response: The malformed response from the previous attempt
        
    Returns:
        A new prompt asking the model to retry with proper formatting
    """
    return f"""{original_instruction}

IMPORTANT: Your previous response could not be parsed correctly. The response was:
```
{previous_response[:500]}
```

Please ensure your response follows the EXACT format specified above with valid JSON wrapped in <json> tags."""


class TaskAgent:
    """Task agent that solves IMO grading problems with retry logic and validation."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = MAX_RETRIES
        self.retry_delay = RETRY_DELAY

    def _extract_prediction(self, raw_text: str, grading_guidelines: str) -> tuple[str, bool]:
        """Extract prediction from raw LLM response.
        
        Args:
            raw_text: The raw text response from the LLM
            grading_guidelines: The grading guidelines for validation
            
        Returns:
            Tuple of (prediction, is_valid)
        """
        prediction = "None"
        
        # Try multiple extraction strategies
        extraction_attempts = [
            ("_extract_jsons", lambda: _extract_jsons(raw_text)),
            ("_extract_json_with_regex", lambda: _extract_json_with_regex(raw_text)),
        ]
        
        for attempt_name, attempt_fn in extraction_attempts:
            try:
                extracted = attempt_fn()
                if extracted and len(extracted) > 0:
                    # Check the last extracted JSON object for "response" key
                    last_obj = extracted[-1]
                    if isinstance(last_obj, dict) and "response" in last_obj:
                        prediction = last_obj["response"]
                        self.log_fn(f"Successfully extracted prediction using {attempt_name}: {prediction}")
                        break
            except Exception as e:
                self.log_fn(f"Extraction attempt {attempt_name} failed: {e}")
                continue
        
        # Last resort: try to extract response value directly from malformed JSON
        if prediction == "None":
            direct_value = _extract_response_value(raw_text)
            if direct_value is not None:
                prediction = direct_value
                self.log_fn(f"Extracted prediction using direct value extraction: {prediction}")
        
        # Validate the prediction
        is_valid = _validate_prediction(prediction, grading_guidelines)
        
        return str(prediction), is_valid

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with retry logic.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract key fields for better prompt construction
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematical grader evaluating student solutions to competition mathematics problems.

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
5. Assign a score based on the grading guidelines.

IMPORTANT: Your response MUST be a valid JSON object wrapped in <json> tags. Do not include any text outside the JSON tags.

Respond in the following exact format:
<json>
{{
    "response": <your numerical score or evaluation>
}}
</json>

The "response" field should contain your final grading decision (typically a number or specific evaluation string as specified in the grading guidelines).

Examples of valid responses:
- For a numeric score: <json>{{"response": 7}}</json>
- For a string evaluation: <json>{{"response": "Correct"}}</json>
- For a boolean: <json>{{"response": true}}</json>

Ensure your JSON is properly formatted with no trailing commas or syntax errors."""

        # Retry loop with validation
        prediction = "None"
        msg_history = []
        current_instruction = instruction
        
        for attempt in range(self.max_retries + 1):
            response, msg_history, info = get_response_from_llm(
                msg=current_instruction,
                model=self.model,
                msg_history=msg_history if attempt > 0 else [],
            )
            
            raw_text = msg_history[-1]["text"]
            self.log_fn(f"Attempt {attempt + 1}: Raw LLM response (first 500 chars): {raw_text[:500]}")
            
            prediction, is_valid = self._extract_prediction(raw_text, grading_guidelines)
            
            if is_valid:
                self.log_fn(f"Successfully obtained valid prediction on attempt {attempt + 1}: {prediction}")
                break
            
            if attempt < self.max_retries:
                self.log_fn(f"Invalid prediction on attempt {attempt + 1}: '{prediction}'. Retrying...")
                # Prepare retry prompt with feedback
                current_instruction = _get_retry_prompt(instruction, raw_text)
                if self.retry_delay > 0:
                    time.sleep(self.retry_delay)
            else:
                self.log_fn(f"Failed to get valid prediction after {self.max_retries + 1} attempts. Final prediction: {prediction}")
        
        return str(prediction), msg_history
