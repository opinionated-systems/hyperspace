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
    
    Tries multiple approaches in order:
    1. Standard <json>...</json> blocks
    2. Markdown code blocks with json
    3. Raw JSON objects with "response" key (balanced braces)
    4. JSON objects with "reasoning" key (full schema)
    5. Any valid JSON object at the end of text (last resort)
    6. Pattern matching for common LLM output formats
    7. Keyword-based extraction for correct/incorrect
    8. LLM-based extraction as final fallback
    """
    if not text or not isinstance(text, str):
        return None
    
    text = text.strip()
    if not text:
        return None
    
    # Strategy 1: Standard <json> tags
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
    
    # Strategy 3: Look for JSON-like structures with "response" key
    # Use a more robust pattern that handles nested braces
    json_pattern = r'\{[^{}]*"response"\s*:\s*(?:0|1|\d+)[^{}]*\}'
    for match in re.finditer(json_pattern, text, re.DOTALL):
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict) and "response" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue
    
    # Strategy 4: Look for JSON with "reasoning" key (full schema)
    full_json_pattern = r'\{\s*"reasoning"\s*:\s*"[^"]*"\s*,\s*"response"\s*:\s*(?:0|1|\d+)\s*\}'
    for match in re.finditer(full_json_pattern, text, re.DOTALL):
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
    
    # Strategy 5: Look for any JSON object at the end of text (last resort)
    # This handles cases where the model outputs JSON without any markers
    last_brace_idx = text.rfind('}')
    if last_brace_idx != -1:
        # Find the matching opening brace using stack-based approach
        brace_count = 0
        in_string = False
        escape_next = False
        for i in range(last_brace_idx, -1, -1):
            char = text[i]
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
                if char == '}':
                    brace_count += 1
                elif char == '{':
                    brace_count -= 1
                    if brace_count == 0:
                        try:
                            candidate = text[i:last_brace_idx + 1]
                            parsed = json.loads(candidate)
                            if isinstance(parsed, dict) and "response" in parsed:
                                return parsed
                        except json.JSONDecodeError:
                            continue
                        break
    
    # Strategy 6: Pattern matching for standalone digits (0 or 1)
    # This handles cases where the model just outputs the answer
    lines = text.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line in ['0', '1']:
            return {"response": int(line), "reasoning": "Extracted from standalone digit"}
        # Check for patterns like "Answer: 1" or "Result: 0"
        match = re.match(r'^(?:answer|result|prediction|grade|score)[\s:]*([01])$', line, re.IGNORECASE)
        if match:
            return {"response": int(match.group(1)), "reasoning": f"Extracted from pattern: {line}"}
    
    # Strategy 7: Look for "correct" or "incorrect" keywords
    text_lower = text.lower()
    has_positive = re.search(r'\b(correct|right|valid|true|yes)\b', text_lower)
    has_negative = re.search(r'\b(incorrect|wrong|invalid|false|no|error)\b', text_lower)
    
    if has_positive and not has_negative:
        return {"response": 1, "reasoning": "Extracted from positive keywords in text"}
    if has_negative and not has_positive:
        return {"response": 0, "reasoning": "Extracted from negative keywords in text"}
    
    return None


def _get_extraction_stats() -> dict:
    """Get statistics about JSON extraction attempts.
    
    Returns a dictionary with extraction method usage counts.
    This is useful for debugging and monitoring extraction performance.
    """
    return {
        "strategies": [
            "json_tags",
            "markdown_blocks", 
            "response_key_pattern",
            "full_schema_pattern",
            "balanced_braces",
            "standalone_digits",
            "keyword_analysis"
        ],
        "description": "JSON extraction uses 7 fallback strategies in order"
    }


def _validate_inputs(inputs: dict) -> tuple[bool, str]:
    """Validate that all required inputs are present and valid.
    
    Args:
        inputs: Dictionary containing the input fields to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(inputs, dict):
        return False, "Inputs must be a dictionary"
    
    required_fields = ["problem", "solution", "grading_guidelines", "student_answer"]
    
    for field in required_fields:
        if field not in inputs:
            return False, f"Missing required field: {field}"
        if not isinstance(inputs[field], str):
            return False, f"Field {field} must be a string"
        if len(inputs[field].strip()) == 0:
            return False, f"Field {field} is empty"
    
    return True, ""


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3
        self.fallback_prediction = "0"
        self._call_count = 0
        
    def _build_grading_prompt(
        self,
        domain: str,
        problem: str,
        solution: str,
        grading_guidelines: str,
        student_answer: str,
        attempt: int = 0
    ) -> str:
        """Build the grading prompt with optional retry guidance."""
        base_prompt = f"""You are an expert {domain} grader evaluating student solutions with high precision and attention to detail.

Your task is to grade a student's answer by comparing it to the correct solution and following the grading guidelines exactly.

PROBLEM:
{problem}

CORRECT SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

Think step by step and provide detailed reasoning:
1. Analyze what the problem is asking for - identify key requirements and expected outcomes
2. Review the correct solution approach - understand the logic and steps
3. Compare the student's answer to the correct solution - check for mathematical equivalence, logical correctness, and completeness
4. Check if the student followed the grading guidelines - pay special attention to any specific criteria mentioned
5. Determine if the student's answer is correct (1) or incorrect (0)
   - Award 1 if the answer is mathematically/logically correct, even if formatted differently
   - Award 0 if the answer is wrong, incomplete, or violates grading guidelines

IMPORTANT: You must respond ONLY in the following JSON format. Do not include any other text outside the JSON tags:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis here explaining your grading decision",
    "response": 1
}}
</json>

OR

<json>
{{
    "reasoning": "Your detailed step-by-step analysis here explaining your grading decision",
    "response": 0
}}
</json>

The "response" field MUST be either 1 (correct) or 0 (incorrect), with no quotes."""

        if attempt > 0:
            base_prompt += f"\n\n(Attempt {attempt + 1}/{self.max_retries}: Please ensure your response follows the exact JSON format specified above.)"
        
        return base_prompt

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        self._call_count += 1
        
        # Validate inputs first
        is_valid, error_msg = _validate_inputs(inputs)
        if not is_valid:
            self.log_fn(f"Input validation failed: {error_msg}")
            return self.fallback_prediction, []
        
        # Extract key fields for better prompting
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        # Try with retries for robustness
        msg_history = []
        for attempt in range(self.max_retries):
            try:
                instruction = self._build_grading_prompt(
                    domain=domain,
                    problem=problem,
                    solution=solution,
                    grading_guidelines=grading_guidelines,
                    student_answer=student_answer,
                    attempt=attempt
                )
                
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                
                # Extract prediction from JSON using flexible extraction
                response_text = msg_history[-1]["text"] if msg_history else ""
                extracted = _extract_json_flexible(response_text)
                
                if extracted and "response" in extracted:
                    prediction = extracted["response"]
                    # Validate prediction is 0 or 1
                    if prediction in [0, 1, "0", "1"]:
                        return str(prediction), msg_history
                    else:
                        self.log_fn(f"Invalid prediction value: {prediction}, retrying...")
                else:
                    self.log_fn(f"No valid JSON found in response (attempt {attempt + 1}), retrying...")
                    
            except Exception as e:
                self.log_fn(f"Error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    break
        
        # Fallback: return default prediction if all retries failed
        self.log_fn(f"All retries failed, returning default prediction {self.fallback_prediction}")
        return self.fallback_prediction, msg_history
