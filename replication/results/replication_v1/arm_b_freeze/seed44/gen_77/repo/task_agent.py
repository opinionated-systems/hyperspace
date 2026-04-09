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
    6. Pattern matching for standalone digits (0/1)
    7. JSON with single quotes instead of double quotes
    8. JSON with unquoted keys (JavaScript-style)
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
    
    # Strategy 7: Look for JSON with single quotes instead of double quotes
    # Some models output JSON with single quotes
    single_quote_pattern = r"\{\s*'reasoning'\s*:\s*'[^']*'\s*,\s*'response'\s*:\s*(?:0|1|\d+)\s*\}"
    for match in re.finditer(single_quote_pattern, text, re.DOTALL):
        try:
            # Replace single quotes with double quotes for valid JSON
            fixed_json = match.group(0).replace("'", '"')
            return json.loads(fixed_json)
        except json.JSONDecodeError:
            continue
    
    # Strategy 8: Look for JSON with unquoted keys (JavaScript-style)
    # Pattern: {reasoning: "...", response: 0|1}
    unquoted_key_pattern = r'\{\s*reasoning\s*:\s*"[^"]*"\s*,\s*response\s*:\s*(?:0|1|\d+)\s*\}'
    for match in re.finditer(unquoted_key_pattern, text, re.DOTALL):
        try:
            # Add quotes to unquoted keys
            fixed = match.group(0)
            fixed = re.sub(r'(\w+):', r'"\1":', fixed)
            return json.loads(fixed)
        except json.JSONDecodeError:
            continue
    
    # Strategy 9: Look for JSON with trailing commas (common LLM error)
    # Pattern: {"reasoning": "...", "response": 1,} or {"reasoning": "...", "response": 1, }
    trailing_comma_pattern = r'\{\s*"reasoning"\s*:\s*"[^"]*"\s*,\s*"response"\s*:\s*(?:0|1|\d+)\s*,\s*\}'
    for match in re.finditer(trailing_comma_pattern, text, re.DOTALL):
        try:
            # Remove trailing comma before closing brace
            fixed = re.sub(r',\s*\}', '}', match.group(0))
            return json.loads(fixed)
        except json.JSONDecodeError:
            continue
    
    # Strategy 10: Look for JSON with newlines inside string values
    # Some models output multi-line reasoning without proper escaping
    multiline_pattern = r'\{\s*"reasoning"\s*:\s*"[\s\S]*?"\s*,\s*"response"\s*:\s*(?:0|1|\d+)\s*\}'
    for match in re.finditer(multiline_pattern, text, re.DOTALL):
        try:
            candidate = match.group(0)
            # Try to fix unescaped newlines in the reasoning field
            # Find the reasoning value and escape newlines within it
            reasoning_match = re.search(r'"reasoning"\s*:\s*"([\s\S]*?)"\s*,\s*"response"', candidate, re.DOTALL)
            if reasoning_match:
                reasoning_val = reasoning_match.group(1)
                # Escape unescaped newlines and quotes
                fixed_reasoning = reasoning_val.replace('\\n', '\n').replace('\n', '\\n').replace('\\"', '"').replace('"', '\\"')
                fixed = candidate.replace(reasoning_val, fixed_reasoning)
                return json.loads(fixed)
        except json.JSONDecodeError:
            continue
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3
        self.extraction_stats = {
            "total_calls": 0,
            "successful_extractions": 0,
            "fallback_used": 0,
        }

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

EXAMPLES OF CORRECT JSON RESPONSES:

Example 1 - Correct answer:
<json>
{{
    "reasoning": "The student correctly identified that the derivative of x^2 is 2x. They applied the power rule accurately and arrived at the same result as the correct solution. The answer is mathematically equivalent even though the notation is slightly different.",
    "response": 1
}}
</json>

Example 2 - Incorrect answer:
<json>
{{
    "reasoning": "The student made an error in the calculation. They incorrectly applied the formula, resulting in a wrong answer. The correct solution shows the proper method, but the student's answer deviates from it in a way that produces an incorrect result.",
    "response": 0
}}
</json>

Example 3 - Partial credit (still incorrect):
<json>
{{
    "reasoning": "While the student showed some understanding of the concept, their answer is incomplete. They missed the second part of the solution which was required according to the grading guidelines. Partial understanding does not warrant a correct score.",
    "response": 0
}}
</json>

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

The "response" field MUST be either 1 (correct) or 0 (incorrect), with no quotes. Be conservative - only award 1 if the answer is fully correct."""

        # Try with retries for robustness
        msg_history = []
        for attempt in range(self.max_retries):
            try:
                # Add retry-specific guidance on subsequent attempts
                current_instruction = instruction
                if attempt > 0:
                    current_instruction = instruction + f"\n\n(Attempt {attempt + 1}/{self.max_retries}: Please ensure your response follows the exact JSON format specified above.)"
                
                response, msg_history, info = get_response_from_llm(
                    msg=current_instruction,
                    model=self.model,
                    msg_history=[],
                )
                
                # Extract prediction from JSON using flexible extraction
                response_text = msg_history[-1]["text"] if msg_history else ""
                self.extraction_stats["total_calls"] += 1
                extracted = _extract_json_flexible(response_text)
                
                if extracted and "response" in extracted:
                    self.extraction_stats["successful_extractions"] += 1
                    prediction = extracted["response"]
                    # Validate prediction is 0 or 1
                    if prediction in [0, 1, "0", "1"]:
                        # Log extraction method for debugging
                        reasoning = extracted.get("reasoning", "")
                        if reasoning and "Extracted from" in reasoning:
                            self.extraction_stats["fallback_used"] += 1
                            self.log_fn(f"Fallback extraction used: {reasoning[:100]}")
                        return str(prediction), msg_history
                    else:
                        self.log_fn(f"Invalid prediction value: {prediction}, retrying...")
                else:
                    self.log_fn(f"No valid JSON found in response (attempt {attempt + 1}), retrying...")
                    
            except Exception as e:
                self.log_fn(f"Error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    break
        
        # Fallback: return "0" if all retries failed
        success_rate = 0
        if self.extraction_stats["total_calls"] > 0:
            success_rate = self.extraction_stats["successful_extractions"] / self.extraction_stats["total_calls"] * 100
        self.log_fn(f"All retries failed, returning default prediction 0. Extraction success rate: {success_rate:.1f}%")
        return "0", msg_history

    def get_extraction_stats(self) -> dict:
        """Return extraction statistics for monitoring and debugging."""
        stats = dict(self.extraction_stats)
        if stats["total_calls"] > 0:
            stats["success_rate"] = stats["successful_extractions"] / stats["total_calls"]
            stats["fallback_rate"] = stats["fallback_used"] / stats["total_calls"]
        return stats
