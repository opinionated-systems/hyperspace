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
    Improved with better brace balancing, more robust pattern matching,
    and additional heuristics for common LLM output formats.
    """
    results = []
    
    def extract_balanced_json(content: str) -> list[dict]:
        """Extract all balanced JSON objects from content using stack-based parsing."""
        objects = []
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
                continue
            if char == '"' and in_string:
                in_string = False
                continue
            if in_string:
                continue
            
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
                            objects.append(json_obj)
                    except json.JSONDecodeError:
                        # Try to fix common JSON issues
                        json_str = content[start_idx:i+1]
                        fixed = _fix_common_json_issues(json_str)
                        if fixed:
                            try:
                                json_obj = json.loads(fixed)
                                if isinstance(json_obj, dict):
                                    objects.append(json_obj)
                            except json.JSONDecodeError:
                                pass
                    start_idx = -1
        return objects
    
    # Try to find JSON objects in code blocks first
    code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    for match in re.finditer(code_block_pattern, text, re.DOTALL):
        content = match.group(1).strip()
        objects = extract_balanced_json(content)
        results.extend(objects)
    
    # If no results from code blocks, try to find any balanced JSON object in text
    if not results:
        results = extract_balanced_json(text)
    
    # Final fallback: try to find key-value patterns for response and reasoning
    if not results:
        result = {}
        # Look for response pattern with more flexible matching
        response_patterns = [
            r'["\']response["\']\s*:\s*["\']([^"\']+)["\']',
            r'["\']response["\']\s*:\s*(\d+)',
            r'response\s*:\s*([\w\s-]+)(?:\n|$)',
            r'(?:grade|assessment|score)\s*[:=]\s*["\']?([^"\'\n]+)["\']?',
        ]
        for pattern in response_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result["response"] = match.group(1).strip()
                break
        
        # Look for reasoning pattern
        reasoning_patterns = [
            r'["\']reasoning["\']\s*:\s*["\']([^"\']*(?:\.[^"\']*)*)["\']',
            r'["\']reasoning["\']\s*:\s*"([^"]*)"',
            r'reasoning\s*:\s*(.+?)(?:\n\s*["\']|$)',
            r'(?:analysis|explanation|thoughts?)\s*[:=]\s*["\']?(.+?)(?:\n\s*(?:response|grade)|$)',
        ]
        for pattern in reasoning_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                result["reasoning"] = match.group(1).strip()
                break
        
        if result:
            results.append(result)
    
    return results or None


def _fix_common_json_issues(json_str: str) -> str | None:
    """Attempt to fix common JSON formatting issues from LLM outputs.
    
    Returns fixed JSON string or None if unfixable.
    """
    fixed = json_str.strip()
    
    # Fix single quotes used instead of double quotes for keys/values
    # This is a simplified fix - only handles simple cases
    if "'" in fixed and '"' not in fixed:
        # Replace single quotes with double quotes for keys and string values
        # This is heuristic and may not work for all cases
        fixed = re.sub(r"'([^']+)'\s*:", r'"\1":', fixed)
        fixed = re.sub(r":\s*'([^']+)'", r': "\1"', fixed)
    
    # Fix trailing commas in objects/arrays
    fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
    
    # Fix missing quotes around keys
    fixed = re.sub(r'(\{|,\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', fixed)
    
    # Fix newlines in string values (replace with escaped newlines)
    # This is complex - we'll try a simpler approach
    
    return fixed if fixed != json_str else None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3
        self._validation_cache: dict[str, bool] = {}

    def _validate_inputs(self, inputs: dict) -> tuple[bool, str]:
        """Validate that all required fields are present and non-empty.
        
        Returns:
            (is_valid, error_message) tuple
        """
        required_fields = ["problem", "solution", "grading_guidelines", "student_answer"]
        
        for field in required_fields:
            if field not in inputs:
                return False, f"Missing required field: {field}"
            value = inputs.get(field)
            if not value or not isinstance(value, str):
                return False, f"Field '{field}' must be a non-empty string"
            if len(value.strip()) < 3:
                return False, f"Field '{field}' is too short (min 3 characters)"
        
        return True, ""

    def _build_retry_prompt(self, inputs: dict, previous_error: str, attempt: int) -> str:
        """Build an enhanced retry prompt with specific error guidance."""
        base_prompt = self._build_grading_prompt(inputs)
        
        error_guidance = {
            "json": "Your previous response had invalid JSON. Ensure proper syntax: use double quotes, no trailing commas, and valid escape sequences.",
            "response": "The 'response' field must contain ONLY one of: 'Correct', 'Partial', or 'Incorrect' (exactly as written, no extra text).",
            "format": "Do not include any text outside the <json>...</json> tags. Only output the JSON block.",
            "reasoning": "The 'reasoning' field should contain your detailed step-by-step analysis.",
            "extract": "Failed to extract a valid grade from your response. Please follow the format exactly.",
            "tag": "Missing or malformed <json> tags. Wrap your entire JSON response in <json>...</json> tags.",
        }
        
        # Determine which guidance to provide based on the error
        guidance = []
        error_lower = previous_error.lower()
        if "json" in error_lower or "decode" in error_lower or "parse" in error_lower:
            guidance.append(error_guidance["json"])
        if "response" in error_lower and "field" in error_lower:
            guidance.append(error_guidance["response"])
        if "format" in error_lower or "tag" in error_lower or "extract" in error_lower:
            guidance.append(error_guidance["format"])
            guidance.append(error_guidance["tag"])
        if "extract" in error_lower or "none" in error_lower:
            guidance.append(error_guidance["extract"])
        
        guidance_text = "\n".join(f"- {g}" for g in guidance) if guidance else "- Ensure valid JSON format with proper <json> tags."
        
        return f"""⚠️ RETRY ATTEMPT {attempt + 1}/{self.max_retries}

PREVIOUS ERROR: {previous_error}

CORRECTION GUIDANCE:
{guidance_text}

---

{base_prompt}"""

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
When assigning grades, use EXACTLY one of these values:
- **Correct**: The answer matches the solution or uses an equivalent valid approach with correct reasoning and final result.
- **Partial**: The answer shows some correct reasoning but has minor errors, incomplete steps, or partially correct results.
- **Incorrect**: The answer contains fundamental errors, wrong approach, or completely wrong results.

## Response Format (CRITICAL - FOLLOW EXACTLY)
You MUST respond with valid JSON wrapped in <json>...</json> tags. Do not include any text outside the JSON tags.

<json>
{{
    "reasoning": "Your detailed step-by-step analysis and reasoning here...",
    "response": "Correct"
}}
</json>

### Formatting Rules:
1. The JSON must be valid - use double quotes for all strings
2. The "response" field must contain ONLY: "Correct", "Partial", or "Incorrect" (exactly as written)
3. The "reasoning" field should contain your detailed analysis
4. Do not include any text before or after the <json> tags
5. Do not use markdown formatting inside the JSON values

### Example Response:
<json>
{{
    "reasoning": "The student correctly identified the approach and arrived at the right answer through valid reasoning.",
    "response": "Correct"
}}
</json>"""

    def _extract_prediction(self, text: str) -> tuple[str, str]:
        """Extract prediction and reasoning from response text.
        
        Returns:
            (prediction, reasoning) tuple
        """
        prediction = "None"
        reasoning = ""
        
        # Try primary extraction method first
        extracted = _extract_jsons(text)
        
        # If primary fails or returns empty, try regex fallback
        if not extracted:
            extracted = _extract_json_with_regex(text)
        
        if extracted:
            # Use the last valid JSON object (most likely to be the final answer)
            last_json = extracted[-1]
            
            # Extract response/grade with multiple key name support
            response_keys = ["response", "grade", "assessment", "score", "answer", "result"]
            for key in response_keys:
                if key in last_json:
                    value = last_json[key]
                    # Handle both string and numeric values
                    if isinstance(value, (int, float)):
                        prediction = str(value)
                    elif isinstance(value, str):
                        prediction = value.strip()
                    else:
                        prediction = str(value)
                    break
            
            # Extract reasoning with multiple key name support
            reasoning_keys = ["reasoning", "analysis", "explanation", "thought", "thinking"]
            for key in reasoning_keys:
                if key in last_json:
                    reasoning = str(last_json[key]).strip()
                    break
        
        # Clean up prediction - remove extra whitespace and normalize
        if prediction != "None":
            prediction = prediction.strip()
            # Handle common grade variations
            prediction_lower = prediction.lower()
            if prediction_lower in ["correct", "right", "true", "yes", "1", "full"]:
                prediction = "Correct"
            elif prediction_lower in ["partial", "partially correct", "half", "0.5"]:
                prediction = "Partial"
            elif prediction_lower in ["incorrect", "wrong", "false", "no", "0", "none"]:
                prediction = "Incorrect"
        
        return prediction, reasoning

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
        last_error = ""
        
        # Retry loop for robust extraction with exponential backoff
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
                    last_error = "Failed to extract prediction from response"
                    self.log_fn(f"Attempt {attempt + 1}/{self.max_retries}: {last_error}, retrying...")
                    # Use enhanced retry prompt with specific guidance
                    if attempt < self.max_retries - 1:
                        instruction = self._build_retry_prompt(inputs, last_error, attempt)
                    
            except Exception as e:
                last_error = str(e)
                self.log_fn(f"Error in attempt {attempt + 1}/{self.max_retries}: {e}")
                if attempt < self.max_retries - 1:
                    instruction = self._build_retry_prompt(inputs, last_error, attempt)
                else:
                    prediction = f"Error: {e}"
        
        return str(prediction), msg_history
