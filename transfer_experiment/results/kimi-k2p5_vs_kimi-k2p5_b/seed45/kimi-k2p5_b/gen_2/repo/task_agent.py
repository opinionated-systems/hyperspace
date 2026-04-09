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
    """Extract JSON using multiple strategies.
    
    Tries multiple methods in order of preference:
    1. Extract from <json>...</json> tags
    2. Extract from ```json...``` code blocks
    3. Find JSON objects directly in text
    4. Look for key-value patterns
    """
    # Strategy 1: <json> tags
    json_blocks = _extract_jsons(text)
    if json_blocks:
        return json_blocks[-1]
    
    # Strategy 2: ```json code blocks
    json_code_pattern = r'```json\s*(.*?)```'
    matches = re.findall(json_code_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue
    
    # Strategy 3: ``` code blocks (any language)
    code_pattern = r'```\s*(.*?)```'
    matches = re.findall(code_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue
    
    # Strategy 4: Find JSON objects directly (curly braces)
    # Look for outermost JSON objects
    brace_count = 0
    start_idx = -1
    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                try:
                    return json.loads(text[start_idx:i+1])
                except json.JSONDecodeError:
                    continue
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract key fields from inputs for better context
        problem = inputs.get('problem', '')
        solution = inputs.get('solution', '')
        grading_guidelines = inputs.get('grading_guidelines', '')
        student_answer = inputs.get('student_answer', '')
        points = inputs.get('points', '')
        reward = inputs.get('reward', '')
        
        instruction = f"""You are an expert mathematical grader evaluating student solutions to competition mathematics problems (IMO, Putnam, etc.).

Your task is to classify the student's answer into exactly one of three categories:
- "correct" - The student's answer is fully correct, complete, and rigorous. The solution solves the problem completely.
- "incorrect" - The student's answer is wrong, contains fundamental errors, or is incomplete with 0 points awarded. Even if the grading guidelines mention "Partial" as a rubric section, if the Points field shows 0, the answer is "incorrect".
- "partial" - The student's answer has some correct elements and received partial credit (Points > 0), but is incomplete or has significant gaps.

IMPORTANT: The "grading_guidelines" field may contain "(Partial)" or "(Almost)" as rubric markers indicating what the student achieved. However, your classification must be based on the actual Points awarded:
- If Points = 0: classify as "incorrect"
- If Points > 0 but less than full: classify as "partial"  
- If the solution is fully correct: classify as "correct"

Task input fields:
- Problem: {problem[:500]}{'...' if len(problem) > 500 else ''}

- Solution: {solution[:500]}{'...' if len(solution) > 500 else ''}

- Grading Guidelines: {grading_guidelines[:800]}{'...' if len(grading_guidelines) > 800 else ''}

- Student Answer: {student_answer[:800]}{'...' if len(student_answer) > 800 else ''}

- Points: {points}

- Reward: {reward}

Carefully analyze:
1. What the problem is asking
2. The official solution provided
3. The grading guidelines (which indicate what rubric items were achieved)
4. The Points field - this is the key indicator for your classification
5. The student's answer - check for correctness, completeness, and rigor

Respond in JSON format with the following schema:
<json>
{{
    "response": "correct" | "incorrect" | "partial"
}}
</json>

Important: Your response must be exactly one of the three words: "correct", "incorrect", or "partial" (all lowercase). Do not include any other text in the response field."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON using flexible extraction
        prediction = "incorrect"  # Default to incorrect if extraction fails
        try:
            last_message = msg_history[-1]["text"]
            self.log_fn(f"Raw response: {last_message[:500]}...")
            
            extracted = _extract_json_flexible(last_message)
            if extracted:
                if "response" in extracted:
                    raw_prediction = extracted["response"]
                elif len(extracted) == 1:
                    # If only one key, use its value
                    raw_prediction = list(extracted.values())[0]
                else:
                    # Try to find a value that looks like a grade
                    raw_prediction = None
                    for key, value in extracted.items():
                        if isinstance(value, str) and value.lower() in ["correct", "incorrect", "partial"]:
                            raw_prediction = value
                            break
                    if raw_prediction is None:
                        raw_prediction = str(extracted)
                
                # Normalize the prediction
                if isinstance(raw_prediction, str):
                    pred_lower = raw_prediction.lower().strip()
                    if pred_lower == "correct":
                        prediction = "correct"
                    elif pred_lower == "partial":
                        prediction = "partial"
                    else:
                        prediction = "incorrect"
                else:
                    prediction = "incorrect"
            
            self.log_fn(f"Extracted prediction: {prediction}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Fallback: try to find grade directly in text
            text_lower = msg_history[-1]["text"].lower()
            if "correct" in text_lower and "incorrect" not in text_lower and "partial" not in text_lower:
                prediction = "correct"
            elif "partial" in text_lower:
                prediction = "partial"
            else:
                prediction = "incorrect"

        return str(prediction), msg_history
