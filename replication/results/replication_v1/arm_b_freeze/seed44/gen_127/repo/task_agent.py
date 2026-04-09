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
    3. Raw JSON objects with "response" key
    4. JSON objects with "reasoning" key
    5. Any valid JSON object at the end of the text
    6. Pattern matching for common LLM output formats
    7. Keyword-based extraction for semantic understanding
    8. LLM-based extraction as final fallback
    """
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
    json_pattern = r'\{\s*"response"\s*:[^\}]+\}'
    for match in re.finditer(json_pattern, text, re.DOTALL):
        try:
            return json.loads(match.group(0))
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
        # Find the matching opening brace
        brace_count = 0
        for i in range(last_brace_idx, -1, -1):
            if text[i] == '}':
                brace_count += 1
            elif text[i] == '{':
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
    
    # Strategy 7: Look for "correct" or "incorrect" keywords with improved logic
    text_lower = text.lower()
    
    # Count occurrences of positive and negative indicators
    positive_indicators = ['correct', 'right', 'valid', 'true', 'yes', 'accurate', 'proper', 'appropriate']
    negative_indicators = ['incorrect', 'wrong', 'invalid', 'false', 'no', 'error', 'mistake', 'inaccurate']
    
    positive_count = sum(1 for word in positive_indicators if re.search(rf'\b{word}\b', text_lower))
    negative_count = sum(1 for word in negative_indicators if re.search(rf'\b{word}\b', text_lower))
    
    # Check for negation patterns that might flip the meaning
    negation_patterns = ['not correct', 'not right', 'not valid', 'not true', 'not accurate']
    has_negation = any(pattern in text_lower for pattern in negation_patterns)
    
    # Determine response based on indicator counts and negation
    if has_negation:
        # If there's negation, flip the interpretation
        if positive_count > negative_count:
            return {"response": 0, "reasoning": "Extracted from negated positive indicators in text"}
        elif negative_count > positive_count:
            return {"response": 1, "reasoning": "Extracted from negated negative indicators in text"}
    else:
        if positive_count > negative_count:
            return {"response": 1, "reasoning": f"Extracted from {positive_count} positive vs {negative_count} negative indicators"}
        elif negative_count > positive_count:
            return {"response": 0, "reasoning": f"Extracted from {negative_count} negative vs {positive_count} positive indicators"}
    
    # Strategy 8: Check for explicit grading statements
    grading_patterns = [
        (r'\b(?:grade|score|mark)\s*[:=]\s*(\d+)', lambda m: int(m.group(1)) > 0),
        (r'\b(?:points?|credit)\s*(?:awarded|given|assigned)\b', lambda m: True),
        (r'\b(?:no\s+points?|zero\s+points?|no\s+credit)\b', lambda m: False),
        (r'\bfull\s+(?:marks?|credit|points?)\b', lambda m: True),
        (r'\bpartial\s+(?:marks?|credit|points?)\b', lambda m: True),
    ]
    
    for pattern, evaluator in grading_patterns:
        match = re.search(pattern, text_lower)
        if match:
            is_correct = evaluator(match)
            return {
                "response": 1 if is_correct else 0,
                "reasoning": f"Extracted from grading pattern: '{match.group(0)}'"
            }
    
    return None


def _validate_inputs(inputs: dict) -> tuple[bool, str]:
    """Validate that all required inputs are present and valid.
    
    Returns:
        (is_valid, error_message)
    """
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

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
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

        instruction = f"""You are an expert {domain} grader evaluating student solutions with high precision and attention to detail.

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

CRITICAL INSTRUCTIONS:
- You MUST respond ONLY in the JSON format shown below
- Do NOT include any text outside the JSON tags
- Do NOT use markdown code blocks (```json)
- The "response" field MUST be the integer 1 or 0 (not a string, not true/false)
- The "reasoning" field MUST be a detailed string explaining your decision

REQUIRED FORMAT:
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

Remember: response must be 1 (correct) or 0 (incorrect) as an integer, not a string."""

        # Try with retries for robustness
        msg_history = []
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # Add retry-specific guidance on subsequent attempts
                current_instruction = instruction
                if attempt > 0:
                    retry_hints = [
                        "\n\nIMPORTANT: Your previous response was not in the correct format. Please respond ONLY with the JSON format shown above, using <json> tags.",
                        "\n\nREMINDER: You MUST use the exact format: <json>{{\"reasoning\": \"...\", \"response\": 0 or 1}}</json>. No other text.",
                        "\n\nFINAL ATTEMPT: Please strictly follow the JSON format with <json> tags. The response field must be 0 or 1 (integer).",
                    ]
                    current_instruction = instruction + retry_hints[min(attempt - 1, len(retry_hints) - 1)]
                
                response, msg_history, info = get_response_from_llm(
                    msg=current_instruction,
                    model=self.model,
                    msg_history=[],
                )
                
                # Extract prediction from JSON using flexible extraction
                response_text = msg_history[-1]["text"] if msg_history else ""
                extracted = _extract_json_flexible(response_text)
                
                if extracted and "response" in extracted:
                    prediction = extracted["response"]
                    # Validate prediction is 0 or 1 (handle both int and string)
                    if prediction in [0, 1]:
                        return str(prediction), msg_history
                    elif prediction in ["0", "1"]:
                        return prediction, msg_history
                    else:
                        last_error = f"Invalid prediction value: {prediction} (type: {type(prediction).__name__})"
                        self.log_fn(f"{last_error}, retrying...")
                else:
                    last_error = f"No valid JSON found in response (attempt {attempt + 1})"
                    self.log_fn(f"{last_error}, retrying...")
                    
            except Exception as e:
                last_error = str(e)
                self.log_fn(f"Error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    break
        
        # Fallback: return default prediction if all retries failed
        self.log_fn(f"All {self.max_retries} retries failed (last error: {last_error}), returning default prediction {self.fallback_prediction}")
        return self.fallback_prediction, msg_history
