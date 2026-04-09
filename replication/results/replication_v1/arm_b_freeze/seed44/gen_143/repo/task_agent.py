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
from agent.config import get_config

logger = logging.getLogger(__name__)

# Get configuration
_cfg = get_config()


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


def _extract_json_flexible(text: str) -> dict | None:
    """Extract JSON with multiple fallback strategies.
    
    Tries multiple approaches in order of reliability:
    1. Standard <json>...</json> blocks
    2. Markdown code blocks with json
    3. Balanced JSON objects with required keys
    4. Pattern-based extraction for malformed JSON
    5. Numeric response extraction as last resort
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
    
    # Strategy 3: Find balanced JSON objects with brace matching
    # This is more robust than regex for nested structures
    def find_json_objects(s: str) -> list[str]:
        """Find all balanced JSON objects in string using brace counting."""
        objects = []
        i = 0
        while i < len(s):
            if s[i] == '{':
                start = i
                brace_count = 1
                i += 1
                in_string = False
                escape_next = False
                
                while i < len(s) and brace_count > 0:
                    char = s[i]
                    if escape_next:
                        escape_next = False
                    elif char == '\\':
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
                    objects.append(s[start:i])
            else:
                i += 1
        return objects
    
    for candidate in find_json_objects(text):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and "response" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue
    
    # Strategy 4: Pattern-based extraction for common LLM output formats
    # Handle cases where JSON is embedded in explanatory text
    response_patterns = [
        # Standard key-value patterns
        r'"response"\s*:\s*(0|1)\s*[,}]',
        r'response["\']?\s*[:=]\s*(0|1)',
        # Look for the last number that could be a response
        r'[^\d](0|1)(?:\s*$|\s*[,}\]])',
    ]
    
    for pattern in response_patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            # Use the last match (usually the actual response, not examples)
            match = matches[-1]
            try:
                response_val = int(match.group(1))
                return {"response": response_val, "reasoning": "Extracted from text pattern"}
            except (ValueError, IndexError):
                continue
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = _cfg.agent.max_retries

    def _build_instruction(self, inputs: dict) -> str:
        """Build the grading instruction prompt from input fields."""
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        # Truncate very long inputs to prevent context overflow
        max_len = 8000
        problem = problem[:max_len] if len(problem) > max_len else problem
        solution = solution[:max_len] if len(solution) > max_len else solution
        guidelines = grading_guidelines[:max_len] if len(grading_guidelines) > max_len else grading_guidelines
        answer = student_answer[:max_len] if len(student_answer) > max_len else student_answer

        return f"""You are an expert {domain} grader evaluating student solutions.

Your task is to grade a student's answer by comparing it to the correct solution and following the grading guidelines.

PROBLEM:
{problem}

CORRECT SOLUTION:
{solution}

GRADING GUIDELINES:
{guidelines}

STUDENT'S ANSWER:
{answer}

Think step by step:
1. Analyze what the problem is asking for
2. Review the correct solution approach
3. Compare the student's answer to the correct solution
4. Check if the student followed the grading guidelines
5. Determine if the student's answer is correct (1) or incorrect (0)

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis here",
    "response": 1 or 0
}}
</json>

The "response" field must be either 1 (correct) or 0 (incorrect)."""

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_instruction(inputs)
        msg_history: list[dict] = []

        # Try with retries for robustness
        for attempt in range(self.max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=msg_history if msg_history else [],
                )
                
                # Extract prediction from JSON using flexible extraction
                last_message = msg_history[-1] if msg_history else {}
                response_text = last_message.get("text", "") if isinstance(last_message, dict) else ""
                extracted = _extract_json_flexible(response_text)
                
                if extracted and "response" in extracted:
                    prediction = extracted["response"]
                    # Validate prediction is 0 or 1 (handle both int and string forms)
                    if prediction in [0, 1, "0", "1", 0.0, 1.0]:
                        # Normalize to string "0" or "1"
                        pred_str = str(int(float(prediction)))
                        self.log_fn(f"Successfully extracted prediction: {pred_str}")
                        return pred_str, msg_history
                    else:
                        self.log_fn(f"Invalid prediction value: {prediction} (type: {type(prediction).__name__}), retrying...")
                else:
                    self.log_fn(f"No valid JSON found in response (attempt {attempt + 1}), retrying...")
                    
            except Exception as e:
                self.log_fn(f"Error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    break
        
        # Fallback: return "0" if all retries failed
        self.log_fn("All retries failed, returning default prediction 0")
        return "0", msg_history
