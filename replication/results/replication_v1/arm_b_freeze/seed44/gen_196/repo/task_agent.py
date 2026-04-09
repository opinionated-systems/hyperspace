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
    3. Raw JSON objects in text (with balanced braces)
    4. Look for JSON-like structures with "response" key
    5. Direct numeric response extraction as last resort
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
    
    # Strategy 3: Look for JSON objects with balanced braces
    # This handles cases where JSON is embedded without tags
    def find_json_objects(s: str) -> list[str]:
        """Find all potential JSON objects by tracking brace balance."""
        objects = []
        start = -1
        depth = 0
        in_string = False
        escape_next = False
        
        for i, char in enumerate(s):
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
                if char == '{':
                    if depth == 0:
                        start = i
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0 and start != -1:
                        objects.append(s[start:i+1])
                        start = -1
        return objects
    
    for json_str in find_json_objects(text):
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, dict) and "response" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue
    
    # Strategy 4: Look for simple JSON-like structures with "response" key
    json_pattern = r'\{\s*"response"\s*:[^\}]+\}'
    for match in re.finditer(json_pattern, text, re.DOTALL):
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
    
    # Strategy 5: Last resort - look for explicit "response" field with numeric value
    # This handles cases where the model outputs text like: "response": 1 or "response": 0
    response_pattern = r'"response"\s*:\s*([01])'
    match = re.search(response_pattern, text)
    if match:
        return {"response": int(match.group(1))}
    
    # Strategy 6: Look for standalone 0 or 1 at the end of the response
    # This is a very loose fallback for when JSON parsing completely fails
    standalone_pattern = r'(?:^|\s|[\n\r])([01])(?:\s*$|[\n\r])'
    matches = list(re.finditer(standalone_pattern, text))
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

        instruction = f"""You are an expert {domain} grader evaluating student solutions with high precision and attention to detail.

Your task is to grade a student's answer by comparing it to the correct solution and following the grading guidelines. Be thorough in your analysis.

PROBLEM:
{problem}

CORRECT SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

Think step by step:
1. Analyze what the problem is asking for - identify key requirements and constraints
2. Review the correct solution approach - understand the logic and methodology
3. Compare the student's answer to the correct solution - check for:
   - Correctness of the final answer
   - Validity of the reasoning process
   - Any missing steps or logical gaps
   - Partial credit considerations (if applicable)
4. Check if the student followed the grading guidelines - note any specific criteria mentioned
5. Determine if the student's answer is correct (1) or incorrect (0)
   - Award 1 if the answer is fully correct or meets all criteria
   - Award 0 if the answer is incorrect, incomplete, or violates key requirements

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis here. Explain your reasoning clearly, citing specific aspects of the student's answer that led to your conclusion.",
    "response": 1 or 0
}}
</json>

The "response" field must be either 1 (correct) or 0 (incorrect). Be decisive in your grading."""

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
