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
    6. Extract from malformed JSON with common LLM errors
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
    
    # Strategy 6: Extract from malformed JSON with common LLM errors
    # Handle cases like: {"reasoning": "...", "response": 1} followed by extra text
    # or JSON with unescaped quotes, trailing commas, etc.
    try:
        # Find the largest valid JSON substring
        start_idx = text.find('{')
        if start_idx != -1:
            for end_idx in range(len(text), start_idx, -1):
                try:
                    candidate = text[start_idx:end_idx]
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict) and "response" in parsed:
                        return parsed
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    
    # Strategy 7: Look for numeric response in text as last resort
    # Extract just the response value if we can find it
    response_patterns = [
        r'"response"\s*:\s*(0|1)\s*[,}]',
        r'response["\']?\s*[:=]\s*(0|1)',
        r'\b(response|answer|grade)\s*[:=]\s*(0|1)',
    ]
    for pattern in response_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
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
        self.max_retries = 3

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

        # Validate inputs to ensure we have meaningful content
        if not problem.strip():
            self.log_fn("Warning: Empty problem provided")
        if not solution.strip():
            self.log_fn("Warning: Empty solution provided")
        if not student_answer.strip():
            self.log_fn("Warning: Empty student answer provided")

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

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis here",
    "response": 1 or 0
}}
</json>

The "response" field must be either 1 (correct) or 0 (incorrect)."""

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
                    # Validate prediction is 0 or 1 (handle both int and string forms)
                    if prediction in [0, 1, "0", "1", 0.0, 1.0]:
                        # Normalize to string "0" or "1"
                        pred_str = str(int(float(prediction)))
                        return pred_str, msg_history
                    else:
                        self.log_fn(f"Invalid prediction value: {prediction} (type: {type(prediction).__name__}), retrying...")
                else:
                    self.log_fn(f"No valid JSON found in response, retrying...")
                    
            except Exception as e:
                self.log_fn(f"Error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    break
        
        # Fallback: return "0" if all retries failed
        self.log_fn("All retries failed, returning default prediction 0")
        return "0", msg_history if 'msg_history' in locals() else []
