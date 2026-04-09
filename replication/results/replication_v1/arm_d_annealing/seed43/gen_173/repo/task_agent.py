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


def _extract_json_with_regex(text: str) -> list[dict] | None:
    """Fallback JSON extraction using regex for malformed responses.
    
    Handles nested braces, code blocks, and various formatting edge cases.
    Improved to handle escaped quotes, nested structures, and common LLM output patterns.
    """
    results = []
    
    # Try to find JSON objects in code blocks with proper brace balancing
    code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    for match in re.finditer(code_block_pattern, text, re.DOTALL):
        content = match.group(1).strip()
        # Try to find balanced JSON objects within the code block
        brace_count = 0
        start_idx = -1
        in_string = False
        escape_next = False
        for i, char in enumerate(content):
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"' and not in_string:
                in_string = True
            elif char == '"' and in_string:
                in_string = False
            elif not in_string:
                if char == '{':
                    if brace_count == 0:
                        start_idx = i
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and start_idx != -1:
                        try:
                            json_obj = json.loads(content[start_idx:i+1])
                            if isinstance(json_obj, dict):
                                results.append(json_obj)
                        except json.JSONDecodeError:
                            pass
                        start_idx = -1
    
    # If no results from code blocks, try to find any balanced JSON object in text
    if not results:
        brace_count = 0
        start_idx = -1
        in_string = False
        escape_next = False
        for i, char in enumerate(text):
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"' and not in_string:
                in_string = True
            elif char == '"' and in_string:
                in_string = False
            elif not in_string:
                if char == '{':
                    if brace_count == 0:
                        start_idx = i
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and start_idx != -1:
                        try:
                            json_obj = json.loads(text[start_idx:i+1])
                            if isinstance(json_obj, dict):
                                results.append(json_obj)
                        except json.JSONDecodeError:
                            pass
                        start_idx = -1
    
    # Try to repair common JSON errors and re-parse
    if not results:
        # Look for JSON-like structures with common errors (trailing commas, unquoted keys)
        repaired = _attempt_json_repair(text)
        if repaired:
            results.extend(repaired)
    
    # Final fallback: try to find any response-like pattern
    if not results:
        # Look for "response": "value" pattern with flexible whitespace and quotes
        response_patterns = [
            r'["\']?response["\']?\s*:\s*["\']([^"\']+)["\']',
            r'["\']?response["\']?\s*:\s*(\w+)',
            r'(?:^|\n)\s*response\s*[:=]\s*["\']?([^"\'\n]+)["\']?',
        ]
        for pattern in response_patterns:
            response_match = re.search(pattern, text, re.IGNORECASE)
            if response_match:
                results.append({"response": response_match.group(1).strip()})
                break
        
        # Look for reasoning pattern too
        reasoning_patterns = [
            r'["\']?reasoning["\']?\s*:\s*["\']([^"\']*(?:\.[^"\']*)*)["\']',
            r'["\']?reasoning["\']?\s*:\s*(.+?)(?:\n\s*["\']?response|\Z)',
        ]
        for pattern in reasoning_patterns:
            reasoning_match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if reasoning_match:
                reasoning_text = reasoning_match.group(1).strip()
                if results:
                    results[0]["reasoning"] = reasoning_text
                else:
                    results.append({"reasoning": reasoning_text})
                break
    
    return results or None


def _attempt_json_repair(text: str) -> list[dict]:
    """Attempt to repair common JSON formatting errors.
    
    Handles:
    - Trailing commas before closing braces/brackets
    - Unquoted keys (converting to quoted)
    - Single quotes instead of double quotes
    - Missing quotes around string values
    
    Returns a list of successfully repaired JSON objects.
    """
    repaired_results = []
    
    # Find potential JSON objects
    potential_jsons = []
    brace_count = 0
    start_idx = -1
    in_string = False
    escape_next = False
    
    for i, char in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if char == '\\':
            escape_next = True
            continue
        if char == '"' and not in_string:
            in_string = True
        elif char == '"' and in_string:
            in_string = False
        elif not in_string:
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    potential_jsons.append(text[start_idx:i+1])
                    start_idx = -1
    
    for json_str in potential_jsons:
        repaired = json_str
        
        # Remove trailing commas before } or ]
        repaired = re.sub(r',(\s*[}\]])', r'\1', repaired)
        
        # Try to parse after basic repair
        try:
            parsed = json.loads(repaired)
            if isinstance(parsed, dict):
                repaired_results.append(parsed)
                continue
        except json.JSONDecodeError:
            pass
        
        # Try converting unquoted keys to quoted keys
        # Match word characters followed by colon at start of object or after comma
        repaired = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', repaired)
        
        # Try converting single quotes to double quotes (carefully)
        # Only convert if it looks like JSON (has colons and braces)
        if ':' in repaired and '{' in repaired:
            # Replace single quotes that are likely JSON delimiters
            # This is a heuristic - we look for 'key': or :'value' patterns
            repaired = re.sub(r"'([^']+)'\s*:", r'"\1":', repaired)
            repaired = re.sub(r":\s*'([^']+)'", r': "\1"', repaired)
        
        try:
            parsed = json.loads(repaired)
            if isinstance(parsed, dict):
                repaired_results.append(parsed)
        except json.JSONDecodeError:
            pass
    
    return repaired_results


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3

    def _build_grading_prompt(self, inputs: dict) -> str:
        """Build a structured prompt for the grading task with chain-of-thought."""
        domain = inputs.get("domain", "unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        return f"""You are an expert grader for {domain} problems. Your task is to evaluate a student's answer against the correct solution and grading guidelines.

## Problem Statement
{problem}

## Correct Solution
{solution}

## Grading Guidelines
{guidelines}

## Student's Answer
{student_answer}

## Instructions
1. First, analyze the student's answer step by step. Compare it against the correct solution.
2. Identify any errors, omissions, or alternative valid approaches.
3. Consider the grading guidelines carefully.
4. Provide your reasoning for the grade you will assign.
5. Finally, provide your grade/assessment in the JSON format below.

## Grading Rubric
When assigning grades, consider:
- **Correct**: The answer matches the solution or uses an equivalent valid approach with correct reasoning and final result.
- **Partial**: The answer shows some correct reasoning but has minor errors, incomplete steps, or partially correct results.
- **Incorrect**: The answer contains fundamental errors, wrong approach, or completely wrong results.

## Response Format (REQUIRED)
You MUST respond with valid JSON wrapped in <json>...</json> tags. Do not include any text outside the JSON tags.

<json>
{{
    "reasoning": "Your detailed step-by-step analysis and reasoning...",
    "response": "Your final grade/assessment (e.g., 'Correct', 'Partial', 'Incorrect', or a numeric score)"
}}
</json>

IMPORTANT: Ensure your JSON is valid and properly formatted. The 'response' field should contain only the grade, not the reasoning."""

    def _extract_prediction(self, text: str) -> tuple[str, str]:
        """Extract prediction and reasoning from response text.
        
        Uses multiple extraction strategies:
        1. Primary: Extract JSON from <json> tags
        2. Fallback: Use regex to find JSON objects
        3. Final fallback: Look for response/reasoning patterns
        
        Args:
            text: Raw LLM response text
            
        Returns:
            (prediction, reasoning) tuple. prediction is normalized to
            'Correct', 'Incorrect', 'Partial', or 'None' if extraction fails.
        """
        prediction = "None"
        reasoning = ""
        
        if not text or not text.strip():
            self.log_fn("Warning: Empty response text received")
            return prediction, reasoning
        
        # Try primary extraction method
        extracted = _extract_jsons(text)
        if extracted is None:
            # Fallback to regex extraction with repair
            extracted = _extract_json_with_regex(text)
            if extracted:
                self.log_fn(f"Used regex fallback for JSON extraction, found {len(extracted)} objects")
                # Check if repair was used
                repaired = _attempt_json_repair(text)
                if repaired:
                    self.log_fn(f"JSON repair successful for {len(repaired)} objects")
        
        if extracted:
            last_json = extracted[-1]
            if isinstance(last_json, dict):
                if "response" in last_json:
                    prediction = str(last_json["response"]).strip()
                if "reasoning" in last_json:
                    reasoning = str(last_json["reasoning"]).strip()
            else:
                self.log_fn(f"Warning: Extracted object is not a dict: {type(last_json)}")
        else:
            self.log_fn("Warning: No JSON objects found in response")
        
        # Normalize common grade variations
        prediction_lower = prediction.lower()
        if prediction_lower in ["correct", "right", "true", "yes", "1", "100%"]:
            prediction = "Correct"
        elif prediction_lower in ["incorrect", "wrong", "false", "no", "0", "0%"]:
            prediction = "Incorrect"
        elif prediction_lower in ["partial", "partially correct", "partial credit", "half"]:
            prediction = "Partial"
        
        return prediction, reasoning

    def _validate_inputs(self, inputs: dict) -> tuple[bool, str]:
        """Validate that all required input fields are present.
        
        Args:
            inputs: dict with input fields
            
        Returns:
            (is_valid, error_message) tuple
        """
        required_fields = ["domain", "problem", "solution", "grading_guidelines", "student_answer"]
        missing_fields = []
        
        for field in required_fields:
            if field not in inputs:
                missing_fields.append(field)
            elif not inputs[field] or (isinstance(inputs[field], str) and not inputs[field].strip()):
                missing_fields.append(f"{field} (empty)")
        
        if missing_fields:
            return False, f"Missing or empty required fields: {', '.join(missing_fields)}"
        return True, ""

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Validate inputs before processing
        is_valid, error_msg = self._validate_inputs(inputs)
        if not is_valid:
            self.log_fn(f"Input validation failed: {error_msg}")
            return f"Error: {error_msg}", []
        
        instruction = self._build_grading_prompt(inputs)
        
        msg_history = []
        prediction = "None"
        reasoning = ""
        
        # Retry loop for robust extraction
        for attempt in range(self.max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=msg_history if attempt > 0 else [],
                )
                
                last_text = msg_history[-1]["text"] if msg_history else ""
                prediction, reasoning = self._extract_prediction(last_text)
                
                if prediction != "None":
                    self.log_fn(f"Successfully extracted prediction: {prediction}")
                    if reasoning:
                        self.log_fn(f"Reasoning: {reasoning[:200]}...")
                    break
                else:
                    self.log_fn(f"Attempt {attempt + 1}: Failed to extract prediction, retrying...")
                    # Add feedback for retry
                    if attempt < self.max_retries - 1:
                        instruction = f"""ERROR: Your previous response did not contain valid JSON with a 'response' field.

You MUST respond with ONLY valid JSON wrapped in <json>...</json> tags. Do not include any text before or after the JSON tags.

Correct format:
<json>
{{
    "reasoning": "Your detailed analysis here...",
    "response": "Correct"
}}
</json>

Now try again with the original task:

{self._build_grading_prompt(inputs)}"""
                    
            except Exception as e:
                self.log_fn(f"Error in attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    prediction = f"Error: {e}"
        
        return str(prediction), msg_history
