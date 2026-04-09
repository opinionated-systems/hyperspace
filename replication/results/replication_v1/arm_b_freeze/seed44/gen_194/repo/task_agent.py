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
import time
from collections import Counter
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
    6. Look for JSON with numeric response values (0 or 1) anywhere in text
    7. Look for standalone 0 or 1 at the end of the response
    8. Look for yes/no/true/false patterns in the text
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
    
    # Strategy 6: Look for simple "response": 0 or "response": 1 patterns
    # This catches cases where the JSON structure is malformed but the key info is there
    simple_response_pattern = r'"response"\s*:\s*(0|1)'
    match = re.search(simple_response_pattern, text)
    if match:
        return {"response": int(match.group(1))}
    
    # Strategy 7: Look for standalone 0 or 1 at the very end (last line)
    # This handles cases where the model just outputs the answer
    lines = text.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line == '0':
            return {"response": 0}
        if line == '1':
            return {"response": 1}
        # Skip empty lines and common suffixes
        if line and not line.lower() in ['correct', 'incorrect', 'true', 'false']:
            break
    
    # Strategy 8: Look for explicit correctness statements in the reasoning
    # This helps when the model explains its reasoning but doesn't format JSON properly
    text_lower = text.lower()
    
    # Look for patterns indicating correctness
    correct_patterns = [
        r'the student\'s answer is correct',
        r'the answer is correct',
        r'this is correct',
        r'student answered correctly',
        r'answer is right',
        r'correct solution',
        r'matches the correct solution',
        r'equivalent to the correct',
    ]
    
    # Look for patterns indicating incorrectness
    incorrect_patterns = [
        r'the student\'s answer is incorrect',
        r'the answer is incorrect',
        r'this is incorrect',
        r'student answered incorrectly',
        r'answer is wrong',
        r'incorrect solution',
        r'does not match',
        r'not equivalent to',
        r'is wrong',
    ]
    
    # Count matches (more specific patterns get higher weight)
    correct_score = 0
    incorrect_score = 0
    
    for pattern in correct_patterns:
        if re.search(pattern, text_lower):
            correct_score += 2
    
    for pattern in incorrect_patterns:
        if re.search(pattern, text_lower):
            incorrect_score += 2
    
    # Also check for simple keywords (lower weight)
    if re.search(r'\bcorrect\b', text_lower):
        correct_score += 1
    if re.search(r'\bincorrect\b|\bwrong\b', text_lower):
        incorrect_score += 1
    
    # Only return if we have a clear signal
    if correct_score > incorrect_score and correct_score >= 2:
        return {"response": 1}
    elif incorrect_score > correct_score and incorrect_score >= 2:
        return {"response": 0}
    
    return None


def _validate_prediction(prediction: Any) -> int | None:
    """Validate and normalize a prediction value.
    
    Returns:
        0 or 1 if valid, None otherwise
    """
    if prediction in [0, 1]:
        return prediction
    if prediction in ["0", "1"]:
        return int(prediction)
    if isinstance(prediction, str):
        pred_lower = prediction.lower().strip()
        if pred_lower in ["0", "false", "incorrect", "no", "wrong"]:
            return 0
        if pred_lower in ["1", "true", "correct", "yes", "right"]:
            return 1
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3
        self._call_count = 0
        self._success_count = 0

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        self._call_count += 1
        start_time = time.time()
        
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
        student_answer = student_answer[:max_len] if len(student_answer) > max_len else student_answer

        instruction = f"""You are an expert {domain} grader evaluating student solutions for the International Mathematical Olympiad (IMO).

Your task is to determine if the student's answer is CORRECT (1) or INCORRECT (0) by comparing it to the official correct solution.

=== PROBLEM ===
{problem}

=== CORRECT SOLUTION ===
{solution}

=== GRADING GUIDELINES ===
{grading_guidelines}

=== STUDENT'S ANSWER ===
{student_answer}

=== GRADING INSTRUCTIONS ===
Follow these steps carefully:
1. Understand what the problem is asking and what constitutes a correct answer
2. Review the official correct solution thoroughly
3. Read the student's answer completely
4. Compare the student's answer to the correct solution:
   - Does it arrive at the same final answer/result?
   - Is the reasoning/methodology sound and complete?
   - Does it satisfy all conditions in the problem?
5. Apply the grading guidelines to make your final determination

IMPORTANT: Be lenient in grading. If the student's answer is mathematically equivalent to the correct solution, even if formatted differently or using different notation, it should be marked as CORRECT (1).

=== OUTPUT FORMAT ===
You MUST respond with a valid JSON object in exactly this format:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis explaining why the answer is correct or incorrect",
    "response": 1
}}
</json>

OR

<json>
{{
    "reasoning": "Your detailed step-by-step analysis explaining why the answer is correct or incorrect",
    "response": 0
}}
</json>

The "response" field MUST be either:
- 1 (CORRECT) if the student's answer matches the correct solution
- 0 (INCORRECT) if the student's answer is wrong or incomplete

Do not include any text outside the JSON tags."""

        msg_history: list[dict] = []
        last_error = None
        all_responses = []  # Track all responses for potential majority voting
        
        # Try with retries for robustness
        for attempt in range(self.max_retries):
            try:
                # Add a reminder about output format on retries
                current_instruction = instruction
                if attempt > 0:
                    current_instruction += f"\n\n[ATTEMPT {attempt + 1}/{self.max_retries}] Please ensure your response follows the exact JSON format specified above."
                
                response, msg_history, info = get_response_from_llm(
                    msg=current_instruction,
                    model=self.model,
                    msg_history=[],
                )
                
                response_text = msg_history[-1]["text"] if msg_history else ""
                all_responses.append(response_text)
                
                # Extract prediction from JSON using flexible extraction
                extracted = _extract_json_flexible(response_text)
                
                if extracted and "response" in extracted:
                    prediction = extracted["response"]
                    validated = _validate_prediction(prediction)
                    
                    if validated is not None:
                        self._success_count += 1
                        elapsed = time.time() - start_time
                        self.log_fn(
                            f"Call {self._call_count}: Success in {elapsed:.2f}s "
                            f"(attempt {attempt + 1}/{self.max_retries}, "
                            f"success rate: {self._success_count}/{self._call_count})"
                        )
                        return str(validated), msg_history
                    else:
                        last_error = f"Invalid prediction value: {prediction}"
                        self.log_fn(f"Attempt {attempt + 1}: {last_error}, retrying...")
                else:
                    last_error = "No valid JSON found in response"
                    # Log a snippet of the response for debugging
                    snippet = response_text[:300].replace('\n', ' ')
                    self.log_fn(f"Attempt {attempt + 1}: {last_error} (response: {snippet}...), retrying...")
                    
            except Exception as e:
                last_error = str(e)
                self.log_fn(f"Error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    break
        
        # Fallback: Try to extract any signal from all responses before giving up
        # Use majority voting from all attempts if we have multiple responses
        if len(all_responses) > 1:
            predictions = []
            for resp_text in all_responses:
                extracted = _extract_json_flexible(resp_text)
                if extracted and "response" in extracted:
                    validated = _validate_prediction(extracted["response"])
                    if validated is not None:
                        predictions.append(validated)
            
            if predictions:
                # Majority vote
                vote_counts = Counter(predictions)
                majority = vote_counts.most_common(1)[0]
                if majority[1] >= len(predictions) / 2:  # At least half agree
                    elapsed = time.time() - start_time
                    self.log_fn(
                        f"Call {self._call_count}: Using majority vote ({majority[0]}) "
                        f"from {len(predictions)} valid predictions "
                        f"({elapsed:.2f}s)"
                    )
                    return str(majority[0]), msg_history
        
        # Final fallback: return "0" if all retries failed
        elapsed = time.time() - start_time
        self.log_fn(
            f"Call {self._call_count}: FAILED after {self.max_retries} attempts "
            f"({elapsed:.2f}s). Last error: {last_error}. "
            f"Returning default prediction 0."
        )
        return "0", msg_history
