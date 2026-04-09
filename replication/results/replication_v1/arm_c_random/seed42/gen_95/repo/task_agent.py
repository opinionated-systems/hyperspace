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
    Also handles markdown code blocks and plain JSON.
    """
    results = []
    search_from = 0
    
    # First try to find <json>...</json> blocks
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
    
    # If no <json> blocks found, try markdown code blocks
    if not results:
        search_from = 0
        while True:
            start = text.find("```json", search_from)
            if start == -1:
                # Try without json specifier
                start = text.find("```", search_from)
                if start == -1:
                    break
                end = text.find("```", start + 3)
                if end == -1:
                    break
                inner = text[start + 3:end].strip()
            else:
                end = text.find("```", start + 7)
                if end == -1:
                    break
                inner = text[start + 7:end].strip()
            search_from = end + 3
            try:
                results.append(json.loads(inner))
            except json.JSONDecodeError:
                continue
    
    # If still no results, try to find JSON objects directly
    if not results:
        # Look for patterns like {"response": ...} or {"score": ...}
        json_pattern = r'\{[^{}]*"(?:response|score|evaluation|result)"[^{}]*\}'
        matches = re.findall(json_pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                results.append(json.loads(match))
            except json.JSONDecodeError:
                continue
    
    return results or None


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
        # Extract fields from inputs for better prompt formatting
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Build an enhanced structured prompt with explicit reasoning steps
        instruction = f"""You are an expert mathematics grader with deep expertise in mathematical problem solving and pedagogy. Your task is to carefully evaluate a student's answer to a mathematics problem.

## Problem Statement:
{problem}

## Correct Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Your Evaluation Task:
Think through this evaluation step-by-step, then provide your final assessment.

**Step 1 - Understanding Check**: 
Does the student correctly understand what the problem is asking? Identify the key concepts and requirements. Did they set up the problem correctly?

**Step 2 - Approach Analysis**: 
Is the student's approach valid? Is their reasoning logically sound? Did they use appropriate mathematical techniques? Note any logical gaps or errors.

**Step 3 - Execution Review**: 
Did the student execute their approach correctly? Are the calculations correct? Is the work shown clearly and organized?

**Step 4 - Conclusion Verification**: 
Did the student arrive at the correct final answer? Is the answer properly justified? Did they answer the specific question asked?

**Step 5 - Score Determination**: 
Based on the grading guidelines and your analysis above, what score does this answer deserve? Consider:
- What the student did correctly (award points for correct steps)
- Where they made errors (deduct points appropriately)
- Whether partial credit applies and how much
- The exact format required by the grading guidelines

## Response Format:
Provide your final evaluation as a concise numeric score or descriptive assessment that directly addresses the grading guidelines. Be specific and match the expected format exactly.

Respond ONLY in JSON format with the following schema:
<json>
{{
    "response": "Your evaluation score or assessment here (e.g., '7/7', 'Partial credit: 3/7', 'Correct', 'Incorrect', '0', '1', etc.)"
}}
</json>

Important: The "response" field must contain ONLY the evaluation result in the format specified by the grading guidelines, with no additional explanation or commentary."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with improved error handling
        prediction = "None"
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted and len(extracted) > 0:
                # Get the last JSON object (most recent response)
                last_json = extracted[-1]
                
                # Try common keys in order of preference
                for key in ["response", "score", "evaluation", "result", "grade", "answer"]:
                    if key in last_json:
                        prediction = last_json[key]
                        break
                else:
                    # If no recognized key, use the first string value found
                    for val in last_json.values():
                        if isinstance(val, str):
                            prediction = val
                            break
                    else:
                        # If no string values, convert first value to string
                        prediction = str(list(last_json.values())[0]) if last_json else "None"
                        
                # Clean up the prediction - remove extra whitespace and newlines
                if isinstance(prediction, str):
                    prediction = prediction.strip().replace('\n', ' ').replace('\r', '')
                    # Limit length to avoid overly long responses
                    if len(prediction) > 500:
                        prediction = prediction[:497] + "..."
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Try to extract any meaningful text from the response as fallback
            try:
                response_text = msg_history[-1].get("text", "")
                # Look for common patterns in grading responses
                text_lower = response_text.lower()
                if "incorrect" in text_lower or "wrong" in text_lower:
                    prediction = "Incorrect"
                elif "correct" in text_lower or "right" in text_lower:
                    prediction = "Correct"
                elif "partial" in text_lower:
                    prediction = "Partial credit"
                else:
                    # Try to extract any numeric score pattern
                    score_match = re.search(r'(\d+)/(\d+)', response_text)
                    if score_match:
                        prediction = score_match.group(0)
            except Exception:
                pass

        return str(prediction), msg_history
