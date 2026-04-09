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


def _extract_json_flexible(text: str) -> dict | None:
    """Extract JSON with multiple fallback strategies.
    
    Tries multiple approaches in order:
    1. Standard <json>...</json> blocks
    2. Markdown code blocks with json
    3. Raw JSON objects with "response" key
    4. JSON objects with "reasoning" key
    5. Any valid JSON object at the end of the text
    6. Bracket-balanced JSON extraction with nested structure support
    7. LLM-based extraction as final fallback
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
        # Find the matching opening brace using bracket counting
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
            if char == '"' and (i == 0 or text[i-1] != '\\'):
                in_string = not in_string
                continue
            if in_string:
                continue
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
    
    # Strategy 6: Find all potential JSON objects with proper brace balancing
    # This handles nested JSON structures more robustly
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
    
    # Strategy 7: Try to extract numeric response from text patterns
    # Look for patterns like "response": 1 or "response": 0 or "response": 0.5
    numeric_pattern = r'["\']?response["\']?\s*[:=]\s*(0(?:\.\d+)?|1(?:\.0+)?|[01])'
    match = re.search(numeric_pattern, text, re.IGNORECASE)
    if match:
        try:
            val = float(match.group(1))
            return {"response": val, "reasoning": "Extracted from text pattern"}
        except (ValueError, TypeError):
            pass
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning.
    
    Supports both binary (0/1) and partial credit (0.0-1.0) scoring modes.
    """

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "", partial_credit: bool = False) -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3
        self.partial_credit = partial_credit

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
        
        # Check if problem supports partial credit
        supports_partial = inputs.get("supports_partial_credit", self.partial_credit)

        if supports_partial:
            instruction = self._build_partial_credit_prompt(
                domain, problem, solution, grading_guidelines, student_answer
            )
        else:
            instruction = self._build_binary_prompt(
                domain, problem, solution, grading_guidelines, student_answer
            )

        # Try with retries for robustness
        for attempt in range(self.max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                
                # Extract prediction from JSON using flexible extraction
                extracted = _extract_json_flexible(msg_history[-1]["text"])
                
                if extracted and "response" in extracted:
                    prediction = extracted["response"]
                    # Validate prediction based on scoring mode
                    if supports_partial:
                        # Partial credit: 0.0 to 1.0
                        try:
                            pred_val = float(prediction)
                            if 0.0 <= pred_val <= 1.0:
                                return str(pred_val), msg_history
                        except (ValueError, TypeError):
                            pass
                        self.log_fn(f"Invalid partial credit value: {prediction}, retrying...")
                    else:
                        # Binary: 0 or 1
                        if prediction in [0, 1, "0", "1"]:
                            return str(prediction), msg_history
                        self.log_fn(f"Invalid binary prediction value: {prediction}, retrying...")
                else:
                    self.log_fn(f"No valid JSON found in response, retrying...")
                    
            except Exception as e:
                self.log_fn(f"Error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    break
        
        # Fallback: return default prediction if all retries failed
        default_pred = "0.0" if supports_partial else "0"
        self.log_fn(f"All retries failed, returning default prediction {default_pred}")
        return default_pred, msg_history if 'msg_history' in locals() else []

    def _build_binary_prompt(self, domain: str, problem: str, solution: str, 
                            grading_guidelines: str, student_answer: str) -> str:
        """Build prompt for binary (0/1) scoring mode."""
        return f"""You are an expert {domain} grader evaluating student solutions.

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

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis here",
    "response": 1 or 0
}}
</json>

The "response" field must be either 1 (correct) or 0 (incorrect)."""

    def _build_partial_credit_prompt(self, domain: str, problem: str, solution: str,
                                   grading_guidelines: str, student_answer: str) -> str:
        """Build prompt for partial credit (0.0-1.0) scoring mode."""
        return f"""You are an expert {domain} grader evaluating student solutions with partial credit.

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
4. Identify which parts the student got correct vs incorrect
5. Assess the student's reasoning quality and approach
6. Assign a partial credit score from 0.0 (completely wrong) to 1.0 (completely correct)

Scoring guidelines:
- 1.0: Fully correct solution with proper reasoning
- 0.7-0.9: Mostly correct with minor errors or gaps
- 0.4-0.6: Partially correct, significant progress but major gaps
- 0.1-0.3: Minimal correct elements, mostly wrong approach
- 0.0: Completely wrong or no valid attempt

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis here, including what was correct and what was wrong",
    "response": 0.0 to 1.0
}}
</json>

The "response" field must be a decimal number between 0.0 and 1.0 (inclusive)."""
