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
    6. Relaxed pattern matching for malformed JSON
    """
    # Input validation
    if not text or not isinstance(text, str):
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
    
    # Strategy 6: Relaxed pattern matching for common malformed cases
    # Handle cases where quotes might be escaped incorrectly or single quotes used
    relaxed_pattern = r'[\{\s]*["\']reasoning["\']\s*:\s*["\']([^"\']*)["\']\s*,\s*["\']response["\']\s*:\s*(\d+)\s*[\}\s]*'
    match = re.search(relaxed_pattern, text, re.DOTALL)
    if match:
        return {
            "reasoning": match.group(1),
            "response": int(match.group(2))
        }
    
    # Strategy 7: Look for just a numeric response (0 or 1) as last resort
    # Extract the last occurrence of a standalone 0 or 1
    response_pattern = r'(?:^|\s|[\:\,\(\[])\s*(0|1)\s*(?:$|\s|[\,\;\.\)\]])'
    matches = list(re.finditer(response_pattern, text))
    if matches:
        # Use the last match as it's likely the final answer
        last_match = matches[-1]
        return {
            "reasoning": "Extracted from text pattern matching",
            "response": int(last_match.group(1))
        }
    
    return None


def _validate_inputs(inputs: dict) -> tuple[bool, str]:
    """Validate that all required fields are present in inputs.

    Returns:
        (is_valid, error_message)
    """
    if not isinstance(inputs, dict):
        return False, f"Expected dict, got {type(inputs).__name__}"

    required_fields = ["problem", "solution", "grading_guidelines", "student_answer"]
    missing = [f for f in required_fields if f not in inputs or inputs[f] is None]

    if missing:
        return False, f"Missing required fields: {', '.join(missing)}"

    # Check for empty strings
    empty = [f for f in required_fields if isinstance(inputs.get(f), str) and not inputs[f].strip()]
    if empty:
        return False, f"Empty required fields: {', '.join(empty)}"

    return True, ""


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
        # Validate inputs first
        is_valid, error_msg = _validate_inputs(inputs)
        if not is_valid:
            self.log_fn(f"Input validation failed: {error_msg}")
            return "0", []

        # Extract key fields for better prompting
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert {domain} grader evaluating student solutions with precision and rigor.

Your task is to grade a student's answer by comparing it to the correct solution and following the grading guidelines.

PROBLEM:
{problem}

CORRECT SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

Think step by step and provide detailed reasoning:

STEP 1 - Problem Analysis:
• What is the problem asking for?
• What are the key concepts and requirements?
• What would constitute a complete and correct answer?

STEP 2 - Solution Review:
• What is the correct approach to solve this problem?
• What are the critical steps in the solution?
• What is the final expected answer or result?

STEP 3 - Student Answer Evaluation:
• Does the student's answer address the problem directly?
• Are the student's reasoning and steps logically sound?
• Does the student arrive at the correct final answer?
• Are there any errors, omissions, or misconceptions?

STEP 4 - Grading Guidelines Check:
• Does the student meet all criteria specified in the grading guidelines?
• Are there any partial credit considerations?
• Is the answer format acceptable per the guidelines?

STEP 5 - Final Determination:
• Based on all the above analysis, is the answer CORRECT (1) or INCORRECT (0)?
• Be strict: only mark as correct if the answer fully satisfies all requirements

IMPORTANT: You must respond with valid JSON in the following format:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis covering all 5 steps above",
    "response": 1 or 0
}}
</json>

The "response" field must be either 1 (correct) or 0 (incorrect). Be conservative in your grading - when in doubt, mark incorrect."""

        msg_history = []
        
        # Try with retries for robustness
        for attempt in range(self.max_retries):
            try:
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
                        self.log_fn(f"Successfully extracted prediction: {prediction}")
                        return str(prediction), msg_history
                    else:
                        self.log_fn(f"Invalid prediction value: {prediction}, retrying...")
                else:
                    self.log_fn(f"No valid JSON found in response (attempt {attempt + 1}), retrying...")
                    # Log a snippet of the response for debugging
                    if response_text:
                        snippet = response_text[:200] + "..." if len(response_text) > 200 else response_text
                        self.log_fn(f"Response snippet: {snippet}")
                    
            except Exception as e:
                self.log_fn(f"Error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    break
        
        # Fallback: return "0" if all retries failed
        self.log_fn("All retries failed, returning default prediction 0")
        return "0", msg_history
