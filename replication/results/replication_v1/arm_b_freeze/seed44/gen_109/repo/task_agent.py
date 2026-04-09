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


def _normalize_prediction(pred: int | str | bool) -> str | None:
    """Normalize various prediction formats to '0' or '1'.
    
    Args:
        pred: The prediction value to normalize
        
    Returns:
        '0', '1', or None if the prediction cannot be normalized
    """
    if pred in [0, "0", False, "false", "False"]:
        return "0"
    elif pred in [1, "1", True, "true", "True"]:
        return "1"
    return None


def _extract_json_flexible(text: str) -> dict | None:
    """Extract JSON with multiple fallback strategies.
    
    Tries multiple approaches in order of reliability:
    1. Standard <json>...</json> blocks
    2. Markdown code blocks with json
    3. Raw JSON objects with "response" key
    4. JSON objects with "reasoning" key
    5. Any valid JSON object at the end of the text
    6. Look for JSON with numeric response values (0 or 1) anywhere in text
    7. Look for explicit "correct" or "incorrect" statements
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
    
    # Strategy 7: Look for explicit correctness statements in the text
    # This helps when the model explains its decision in natural language
    text_lower = text.lower()
    # Look for strong indicators of correctness/incorrectness
    correct_indicators = [
        'the student\'s answer is correct',
        'the answer is correct',
        'this is correct',
        'student is correct',
        'answer is right',
        'correct solution',
        'fully correct',
        'mark as correct',
        'grade: correct',
        'verdict: correct',
        'conclusion: correct',
        'final answer: correct',
        'evaluation: correct',
        'the student has correctly',
        'the student solved',
        'correctly identifies',
        'correctly concludes',
        'correct approach',
        'valid solution',
        'correct reasoning',
    ]
    incorrect_indicators = [
        'the student\'s answer is incorrect',
        'the answer is incorrect',
        'this is incorrect',
        'student is incorrect',
        'answer is wrong',
        'incorrect solution',
        'mark as incorrect',
        'grade: incorrect',
        'verdict: incorrect',
        'conclusion: incorrect',
        'final answer: incorrect',
        'evaluation: incorrect',
        'the student has incorrectly',
        'the student failed',
        'incorrectly identifies',
        'incorrectly concludes',
        'incorrect approach',
        'invalid solution',
        'incorrect reasoning',
        'fundamental error',
        'logical error',
        'mathematical error',
    ]
    
    # Count occurrences (more weight to explicit statements)
    correct_score = sum(1 for ind in correct_indicators if ind in text_lower)
    incorrect_score = sum(1 for ind in incorrect_indicators if ind in text_lower)
    
    # Also check for standalone "correct" or "incorrect" near the end
    last_para = text_lower.split('\n\n')[-1] if '\n\n' in text_lower else text_lower[-500:]
    if 'incorrect' in last_para and 'not incorrect' not in last_para:
        incorrect_score += 2
    if 'correct' in last_para and 'not correct' not in last_para and 'incorrect' not in last_para:
        correct_score += 2
    
    if correct_score > incorrect_score:
        return {"response": 1, "reasoning": "Extracted from natural language analysis"}
    elif incorrect_score > correct_score:
        return {"response": 0, "reasoning": "Extracted from natural language analysis"}
    
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

        instruction = f"""You are an expert {domain} grader evaluating student solutions for the International Mathematical Olympiad (IMO).

Your task is to grade a student's answer by comparing it to the correct solution and following the grading guidelines. Be thorough, precise, and objective in your evaluation.

PROBLEM:
{problem}

CORRECT SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

Think step by step:
1. Analyze what the problem is asking for - identify key concepts, required steps, and expected outcomes
2. Review the correct solution approach - understand the logic, methodology, and key insights
3. Compare the student's answer to the correct solution systematically:
   - Does the student arrive at the correct final answer?
   - Is the reasoning process valid and logically sound?
   - Are there any mathematical errors or logical fallacies?
   - Does the student demonstrate understanding of core concepts?
   - Is the solution complete or are there gaps?
4. Apply the grading guidelines precisely - check for specific requirements mentioned
5. Consider edge cases: equivalent forms of correct answers, alternative valid approaches
6. Determine if the student's answer is correct (1) or incorrect (0)

GRADING CRITERIA:
- CORRECT (1): The student's answer demonstrates understanding of the problem, uses valid reasoning, and arrives at the correct conclusion. Minor presentation differences or alternative valid approaches are acceptable.
- INCORRECT (0): The answer contains fundamental errors, incorrect reasoning, mathematical mistakes, or arrives at the wrong conclusion. Missing key steps or misinterpreting the problem also results in INCORRECT.

IMPORTANT NOTES:
- Be strict but fair. IMO problems require rigorous solutions.
- A correct final answer with flawed reasoning should be marked INCORRECT.
- An incorrect final answer with correct partial reasoning should be marked INCORRECT (unless partial credit is explicitly allowed in guidelines).
- Equivalent mathematical expressions (e.g., 1/2 vs 0.5) are both correct.

Respond ONLY in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis here. Explain your thought process clearly, citing specific aspects of the student's solution.",
    "response": 1 or 0
}}
</json>

The "response" field MUST be either 1 (correct) or 0 (incorrect). Do not include any other text outside the JSON block."""

        # Try with retries for robustness
        last_error = None
        all_msg_histories = []
        
        for attempt in range(self.max_retries):
            try:
                # Add retry guidance after first attempt
                current_instruction = instruction
                if attempt > 0:
                    current_instruction = instruction + f"\n\n[Previous attempt failed: {last_error}. Please ensure your response is valid JSON with exactly 1 or 0 in the response field.]"
                
                response, msg_history, info = get_response_from_llm(
                    msg=current_instruction,
                    model=self.model,
                    msg_history=[],
                )
                all_msg_histories.extend(msg_history)
                
                # Extract prediction from JSON using flexible extraction
                last_text = msg_history[-1]["text"] if msg_history else ""
                extracted = _extract_json_flexible(last_text)
                
                if extracted and "response" in extracted:
                    prediction = extracted["response"]
                    # Normalize prediction using helper function
                    normalized = _normalize_prediction(prediction)
                    if normalized is not None:
                        return normalized, all_msg_histories
                    else:
                        self.log_fn(f"Invalid prediction value: {prediction}, retrying...")
                        last_error = f"Invalid prediction: {prediction}"
                else:
                    self.log_fn(f"No valid JSON found in response (attempt {attempt + 1}), retrying...")
                    last_error = "No valid JSON extracted"
                    
            except Exception as e:
                self.log_fn(f"Error on attempt {attempt + 1}: {e}")
                last_error = str(e)
                if attempt == self.max_retries - 1:
                    break
        
        # Fallback: try to extract any numeric response from the last message
        if all_msg_histories:
            last_text = all_msg_histories[-1].get("text", "")
            # Look for explicit 0 or 1 in the text
            if '"response": 1' in last_text or '"response":1' in last_text or '"response" : 1' in last_text:
                self.log_fn("Fallback extraction: found response=1")
                return "1", all_msg_histories
            elif '"response": 0' in last_text or '"response":0' in last_text or '"response" : 0' in last_text:
                self.log_fn("Fallback extraction: found response=0")
                return "0", all_msg_histories
            
            # Additional fallback: look for standalone 0 or 1 after "response"
            response_match = re.search(r'["\']?response["\']?\s*[:=]\s*([01])', last_text, re.IGNORECASE)
            if response_match:
                val = response_match.group(1)
                self.log_fn(f"Fallback extraction: found response={val}")
                return val, all_msg_histories
        
        # Final fallback: return "0" if all retries failed
        self.log_fn(f"All retries failed (last error: {last_error}), returning default prediction 0")
        return "0", all_msg_histories
