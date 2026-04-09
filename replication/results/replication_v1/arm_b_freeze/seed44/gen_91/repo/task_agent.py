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
    
    Tries multiple approaches in order of reliability:
    1. Standard <json>...</json> blocks
    2. Markdown code blocks with json
    3. Raw JSON objects with "response" key
    4. JSON objects with "reasoning" key
    5. Any valid JSON object at the end of the text
    6. Pattern matching for common LLM output formats
    7. Heuristic extraction for malformed JSON
    8. LLM-based extraction as final fallback
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
    last_brace_idx = text.rfind('}')
    if last_brace_idx != -1:
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
    lines = text.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line in ['0', '1']:
            return {"response": int(line), "reasoning": "Extracted from standalone digit"}
        match = re.match(r'^(?:answer|result|prediction|grade|score)[\s:]*([01])$', line, re.IGNORECASE)
        if match:
            return {"response": int(match.group(1)), "reasoning": f"Extracted from pattern: {line}"}
    
    # Strategy 7: Heuristic extraction for malformed JSON with response field
    response_pattern = r'"response"\s*:\s*([01])'
    match = re.search(response_pattern, text)
    if match:
        response_val = int(match.group(1))
        reasoning_pattern = r'"reasoning"\s*:\s*"([^"]*)"'
        reasoning_match = re.search(reasoning_pattern, text)
        reasoning = reasoning_match.group(1) if reasoning_match else "Extracted via heuristic from malformed JSON"
        return {"response": response_val, "reasoning": reasoning}
    
    # Strategy 8: Extract from text containing "correct" or "incorrect"
    text_lower = text.lower()
    if "incorrect" in text_lower or "wrong" in text_lower or "error" in text_lower:
        return {"response": 0, "reasoning": "Extracted from negative indicators in text"}
    if "correct" in text_lower or "right" in text_lower or "valid" in text_lower:
        # Check for negation
        negation_pattern = r'\b(not|isn\'t|doesn\'t|no|never|hardly|barely)\b.*\bcorrect\b|\bcorrect\b.*\b(not|isn\'t|doesn\'t|no|never|hardly|barely)\b'
        if re.search(negation_pattern, text_lower):
            return {"response": 0, "reasoning": "Extracted from negated correctness indicator"}
        return {"response": 1, "reasoning": "Extracted from positive indicators in text"}
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3
        self.call_count = 0

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        self.call_count += 1
        call_id = self.call_count
        
        # Extract key fields for better prompting
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Truncate very long inputs to prevent token overflow
        max_len = 8000
        problem = problem[:max_len] if len(problem) > max_len else problem
        solution = solution[:max_len] if len(solution) > max_len else solution
        grading_guidelines = grading_guidelines[:max_len] if len(grading_guidelines) > max_len else grading_guidelines
        student_answer = student_answer[:max_len] if len(student_answer) > max_len else student_answer

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

The "response" field MUST be either 1 (correct) or 0 (incorrect), with no quotes.

Remember: Be strict but fair. Only award 1 if the answer is fully correct according to the grading guidelines."""

        # Try with retries for robustness
        msg_history = []
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # Add retry-specific guidance on subsequent attempts
                current_instruction = instruction
                if attempt > 0:
                    current_instruction = instruction + f"\n\n(Attempt {attempt + 1}/{self.max_retries}: Previous attempt failed due to: {last_error}. Please ensure your response follows the exact JSON format specified above with <json> tags.)"
                
                self.log_fn(f"Call {call_id}: Attempt {attempt + 1}/{self.max_retries}")
                
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
                    # Validate prediction is 0 or 1
                    if prediction in [0, 1, "0", "1"]:
                        self.log_fn(f"Call {call_id}: Success with prediction {prediction}")
                        return str(prediction), msg_history
                    else:
                        last_error = f"Invalid prediction value: {prediction}"
                        self.log_fn(f"Call {call_id}: {last_error}, retrying...")
                else:
                    last_error = "No valid JSON found in response"
                    self.log_fn(f"Call {call_id}: {last_error} (attempt {attempt + 1}), retrying...")
                    
            except Exception as e:
                last_error = str(e)
                self.log_fn(f"Call {call_id}: Error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    break
        
        # Fallback: return "0" if all retries failed
        self.log_fn(f"Call {call_id}: All retries failed, returning default prediction 0")
        return "0", msg_history if msg_history else []
