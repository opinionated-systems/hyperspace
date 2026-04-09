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


def _attempt_json_repair(text: str) -> str:
    """Attempt to repair common JSON errors in LLM output.
    
    Returns the repaired text if changes were made, otherwise original.
    """
    original = text
    
    # Fix 1: Replace "1 or 0" or "0 or 1" with just the first number
    text = re.sub(r'"response":\s*1\s+or\s+0', '"response": 1', text)
    text = re.sub(r'"response":\s*0\s+or\s+1', '"response": 0', text)
    text = re.sub(r'"response":\s*1\s+or\s+1', '"response": 1', text)
    text = re.sub(r'"response":\s*0\s+or\s+0', '"response": 0', text)
    
    # Fix 2: Remove trailing commas before closing braces/brackets
    text = re.sub(r',(\s*})', r'\1', text)
    text = re.sub(r',(\s*])', r'\1', text)
    
    # Fix 3: Fix single quotes to double quotes (common LLM error)
    # Only fix if it looks like JSON structure
    if '"' not in text and "'" in text and ('{' in text or '[' in text):
        text = text.replace("'", '"')
    
    # Fix 4: Fix unquoted keys (e.g., {key: "value"} -> {"key": "value"})
    text = re.sub(r'(\{|,\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', text)
    
    # Fix 5: Remove comments (// style and /* */ style)
    text = re.sub(r'//.*?\n', '\n', text)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    
    return text if text != original else original


def _extract_json_flexible(text: str) -> dict | None:
    """Extract JSON with multiple fallback strategies.
    
    Tries multiple approaches in order of reliability:
    1. Standard <json>...</json> blocks (most reliable)
    2. Markdown code blocks with json
    3. Bracket-balanced JSON extraction with scoring
    4. Relaxed parsing for common LLM output errors
    """
    # Strategy 1: Standard <json> tags (most reliable)
    results = _extract_jsons(text)
    if results:
        return results[-1]
    
    # Strategy 2: Markdown code blocks
    json_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    for match in re.finditer(json_block_pattern, text, re.DOTALL):
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            continue
    
    # Strategy 3: Find all potential JSON objects with proper brace balancing
    # This is the most robust method for finding JSON in arbitrary text
    potential_jsons = []
    for match in re.finditer(r'\{', text):
        start = match.start()
        brace_count = 0
        in_string = False
        escape_next = False
        for i in range(start, len(text)):
            char = text[i]
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"' and (i == 0 or text[i-1] != '\\'):
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    candidate = text[start:i+1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict) and "response" in parsed:
                            # Score based on having both required fields
                            score = 0
                            if "reasoning" in parsed:
                                score += 2
                            if "response" in parsed:
                                score += 1
                            potential_jsons.append((score, parsed))
                    except json.JSONDecodeError:
                        pass
                    break
    
    # Return the highest-scoring valid JSON (prefer ones with both fields)
    if potential_jsons:
        potential_jsons.sort(reverse=True, key=lambda x: x[0])
        return potential_jsons[0][1]
    
    # Strategy 4: Try to fix common LLM JSON errors and re-parse
    # Look for patterns like {"response": 1 or 0} and fix them
    fixed_text = _attempt_json_repair(text)
    if fixed_text != text:
        return _extract_json_flexible(fixed_text)
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning.
    
    This agent evaluates student solutions by comparing them to correct solutions
    and following grading guidelines. It uses structured JSON output with reasoning
    and response fields, and includes validation to ensure output quality.
    
    Attributes:
        model: The LLM model to use for grading
        max_retries: Maximum number of retry attempts for failed extractions
        validation_enabled: Whether to validate grading output quality
    """

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3
        self.validation_enabled = True
        self._stats = {
            "total_calls": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "validation_failures": 0,
            "retry_attempts": 0,
        }

    def get_stats(self) -> dict:
        """Get agent performance statistics.
        
        Returns:
            Dictionary with call statistics
        """
        return self._stats.copy()

    def reset_stats(self) -> None:
        """Reset all performance statistics to zero."""
        for key in self._stats:
            self._stats[key] = 0

    def set_validation(self, enabled: bool) -> None:
        """Enable or disable output validation.
        
        Args:
            enabled: True to enable validation, False to disable
        """
        self.validation_enabled = enabled

    def _validate_grading(self, extracted: dict, inputs: dict) -> tuple[bool, str]:
        """Validate the grading output for consistency.
        
        Args:
            extracted: The extracted JSON response from the model
            inputs: The original problem inputs
            
        Returns:
            (is_valid, reason) tuple
        """
        if not isinstance(extracted, dict):
            return False, "Extracted response is not a dictionary"
        
        if "response" not in extracted:
            return False, "Missing 'response' field"
        
        response = extracted["response"]
        # Normalize response to integer for validation
        if isinstance(response, str):
            response = response.strip().lower()
            if response in ["0", "1", "true", "false", "correct", "incorrect"]:
                # Convert boolean-like strings
                if response in ["1", "true", "correct"]:
                    extracted["response"] = 1
                else:
                    extracted["response"] = 0
            else:
                return False, f"Invalid response value: {response}"
        elif isinstance(response, bool):
            # Convert boolean to int
            extracted["response"] = 1 if response else 0
        elif response not in [0, 1]:
            return False, f"Invalid response value: {response}"
        
        # Check for reasoning field
        if "reasoning" not in extracted:
            return False, "Missing 'reasoning' field"
        
        reasoning = extracted.get("reasoning", "")
        if not reasoning or len(reasoning.strip()) < 5:
            return False, "Reasoning is too short or empty"
        
        # Additional validation: check reasoning mentions key elements from problem
        reasoning_lower = reasoning.lower()
        problem = inputs.get("problem", "")
        
        # Basic sanity check: reasoning should be substantive
        if len(reasoning.strip()) < 20:
            return False, "Reasoning is too short to be meaningful"
        
        return True, "Valid grading output"

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract key fields for better prompting
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        # Build a more structured and clear prompt
        instruction = f"""You are an expert {domain} grader evaluating student solutions.

Your task is to grade a student's answer by comparing it to the correct solution and following the grading guidelines.

PROBLEM:
{problem}

CORRECT SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

Think step by step:
1. Analyze what the problem is asking for
2. Review the correct solution approach
3. Compare the student's answer to the correct solution
4. Check if the student followed the grading guidelines
5. Determine if the student's answer is correct (1) or incorrect (0)

IMPORTANT: You must respond with ONLY a JSON object wrapped in <json> tags. Do not include any other text outside the JSON tags.

<json>
{{
    "reasoning": "Your detailed step-by-step analysis here. Explain your reasoning clearly.",
    "response": 0
}}
</json>

The "response" field must be either 1 (correct) or 0 (incorrect). Use 1 if the student's answer is correct, 0 if incorrect."""

        self._stats["total_calls"] += 1
        msg_history = []
        last_error = None
        
        # Try with retries for robustness
        for attempt in range(self.max_retries):
            try:
                # On retry, add error context to help the model correct itself
                if attempt > 0 and last_error:
                    retry_msg = f"""Your previous response had an issue: {last_error}

Please provide a corrected response following the same format:

<json>
{{
    "reasoning": "Your detailed step-by-step analysis here",
    "response": 0
}}
</json>"""
                    response, msg_history, info = get_response_from_llm(
                        msg=retry_msg,
                        model=self.model,
                        msg_history=msg_history,
                    )
                else:
                    response, msg_history, info = get_response_from_llm(
                        msg=instruction,
                        model=self.model,
                        msg_history=[],
                    )
                
                # Extract prediction from JSON using flexible extraction
                extracted = _extract_json_flexible(msg_history[-1]["text"])
                
                if extracted:
                    self._stats["successful_extractions"] += 1
                    
                    # Validate the grading output if enabled
                    if self.validation_enabled:
                        is_valid, validation_msg = self._validate_grading(extracted, inputs)
                        if not is_valid:
                            self._stats["validation_failures"] += 1
                            self.log_fn(f"Validation failed: {validation_msg}, retrying...")
                            self._stats["retry_attempts"] += 1
                            last_error = validation_msg
                            continue
                    
                    prediction = extracted["response"]
                    # Normalize and validate prediction
                    if isinstance(prediction, str):
                        prediction = prediction.strip()
                    if prediction in [0, 1, "0", "1"]:
                        # Normalize to string "0" or "1"
                        pred_str = str(int(prediction))
                        return pred_str, msg_history
                    else:
                        self.log_fn(f"Invalid prediction value: {prediction}, retrying...")
                        self._stats["retry_attempts"] += 1
                        last_error = f"Invalid response value: {prediction}. Must be 0 or 1."
                else:
                    self._stats["failed_extractions"] += 1
                    self.log_fn(f"No valid JSON found in response, retrying...")
                    self._stats["retry_attempts"] += 1
                    last_error = "Could not parse JSON from your response. Please use the exact format shown."
                    
            except Exception as e:
                self._stats["failed_extractions"] += 1
                self.log_fn(f"Error on attempt {attempt + 1}: {e}")
                self._stats["retry_attempts"] += 1
                last_error = str(e)
                if attempt == self.max_retries - 1:
                    break
        
        # Fallback: return "0" if all retries failed
        self.log_fn("All retries failed, returning default prediction 0")
        return "0", msg_history
