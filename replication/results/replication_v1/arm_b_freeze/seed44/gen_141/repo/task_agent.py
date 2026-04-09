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
    6. Look for JSON with numeric response values (0 or 1) anywhere in text
    7. Look for standalone 0 or 1 on their own lines (model's direct answer)
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
    
    # Strategy 7: Look for standalone 0 or 1 on their own lines
    # This handles cases where the model outputs a direct answer without JSON
    # Check for patterns like "Answer: 1" or "The answer is 0" or just "1" on its own line
    standalone_patterns = [
        r'(?:^|\n)\s*(?:answer|result|grade|score|prediction)\s*[:\-]?\s*(0|1)\s*(?:$|\n)',
        r'(?:^|\n)\s*(?:the answer is|the result is|final answer|therefore)\s*[:\-]?\s*(0|1)\s*(?:$|\n)',
        r'(?:^|\n)\s*(?:correct|incorrect)\s*[:\-]?\s*(0|1)\s*(?:$|\n)',
    ]
    for pattern in standalone_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return {"response": int(match.group(1))}
    
    # Strategy 8: Look for the last standalone 0 or 1 in the text
    # This is a last resort for when the model outputs just the number
    last_number_pattern = r'(?:^|\n)\s*(0|1)\s*(?:$|\n)'
    matches = list(re.finditer(last_number_pattern, text))
    if matches:
        # Use the last match as it's likely the final answer
        return {"response": int(matches[-1].group(1))}
    
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

Your task is to grade a student's answer by comparing it to the correct solution and following the grading guidelines. Be thorough and precise in your evaluation.

PROBLEM:
{problem}

CORRECT SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

Think step by step:
1. Analyze what the problem is asking for - identify key concepts and required steps
2. Review the correct solution approach - understand the logic and methodology
3. Compare the student's answer to the correct solution - check for:
   - Correctness of the final answer
   - Validity of the reasoning process
   - Mathematical rigor and completeness
   - Any partial credit considerations from the guidelines
4. Check if the student followed the grading guidelines precisely
5. Determine if the student's answer is correct (1) or incorrect (0)

IMPORTANT: The student answer is considered CORRECT (1) if it demonstrates understanding of the problem and arrives at the correct conclusion, even if the presentation differs from the official solution. The answer is INCORRECT (0) if it contains fundamental errors, incorrect reasoning, or arrives at the wrong conclusion.

You MUST respond using EXACTLY this JSON format (include the <json> tags):
<json>
{{
    "reasoning": "Your detailed step-by-step analysis here. Explain your thought process clearly.",
    "response": 1
}}
</json>

OR for incorrect answers:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis here. Explain your thought process clearly.",
    "response": 0
}}
</json>

CRITICAL: The "response" field MUST be either the integer 1 (correct) or the integer 0 (incorrect). Do not use strings, booleans, or any other format."""

        # Try with retries for robustness
        last_error = None
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
                    # Normalize prediction: convert strings to integers
                    if isinstance(prediction, str):
                        prediction = prediction.strip().lower()
                        if prediction in ["0", "false", "incorrect", "wrong", "no"]:
                            prediction = 0
                        elif prediction in ["1", "true", "correct", "right", "yes"]:
                            prediction = 1
                    # Validate prediction is 0 or 1
                    if prediction in [0, 1]:
                        return str(prediction), msg_history
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
        if 'msg_history' in locals() and msg_history:
            last_text = msg_history[-1].get("text", "")
            # Look for explicit 0 or 1 in the text
            if '"response": 1' in last_text or '"response":1' in last_text:
                self.log_fn("Fallback extraction: found response=1")
                return "1", msg_history
            elif '"response": 0' in last_text or '"response":0' in last_text:
                self.log_fn("Fallback extraction: found response=0")
                return "0", msg_history
        
        # Final fallback: return "0" if all retries failed
        self.log_fn(f"All retries failed (last error: {last_error}), returning default prediction 0")
        return "0", msg_history if 'msg_history' in locals() else []
