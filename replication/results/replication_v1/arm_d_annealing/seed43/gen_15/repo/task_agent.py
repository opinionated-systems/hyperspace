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


def _extract_json_from_markdown(text: str) -> list[dict] | None:
    """Extract JSON from markdown code blocks (```json ... ```).
    
    Handles cases where the model uses markdown formatting instead of <json> tags.
    """
    results = []
    pattern = r'```(?:json)?\s*(.*?)```'
    import re
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            continue
    return results or None


def _extract_json_fuzzy(text: str) -> list[dict] | None:
    """Fuzzy JSON extraction for when standard extraction fails.
    
    Tries to find JSON objects even without proper <json> tags.
    Uses a stack-based approach to handle nested braces correctly.
    """
    results = []
    # Try to find JSON-like structures with curly braces
    brace_count = 0
    start_idx = None
    in_string = False
    escape_next = False
    
    for i, char in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if char == '\\':
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
            
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx is not None:
                try:
                    json_str = text[start_idx:i+1]
                    parsed = json.loads(json_str)
                    # Only accept objects with expected keys
                    if isinstance(parsed, dict) and ("response" in parsed or "reasoning" in parsed):
                        results.append(parsed)
                except json.JSONDecodeError:
                    pass
                start_idx = None
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3

    def _build_prompt(self, inputs: dict) -> str:
        """Build a structured prompt with chain-of-thought instructions."""
        domain = inputs.get("domain", "unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        return f"""You are an expert mathematical grader for {domain} problems.

Your task is to evaluate a student's answer to a mathematical problem.

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

INSTRUCTIONS:
1. First, analyze the student's answer step by step. Compare it to the official solution.
2. Check if the student has the correct final answer.
3. Verify if the student's reasoning is sound and follows logical steps.
4. Consider partial credit based on the grading guidelines.
5. Provide your final grade in the JSON format below.

IMPORTANT: You MUST respond with ONLY a JSON object wrapped in <json> tags. Do not include any other text outside the JSON tags.

Respond in this exact format:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis here",
    "response": "Your final grade/assessment here"
}}
</json>

The "response" field should contain only the final grade (e.g., "7", "5", "0", "Correct", "Incorrect", etc.).
The "reasoning" field should contain your detailed analysis of the student's answer."""

    def _extract_prediction(self, msg_history: list[dict]) -> tuple[str, str]:
        """Extract prediction and reasoning from message history.
        
        Returns:
            (prediction, reasoning) tuple
        """
        if not msg_history:
            return "None", ""
        
        last_text = msg_history[-1].get("text", "")
        
        # Try standard extraction first
        extracted = _extract_jsons(last_text)
        if not extracted:
            # Try markdown code blocks as second fallback
            extracted = _extract_json_from_markdown(last_text)
        if not extracted:
            # Try fuzzy extraction as final fallback
            extracted = _extract_json_fuzzy(last_text)
        
        if not extracted:
            return "None", ""
        
        last_json = extracted[-1]
        prediction = last_json.get("response", "None")
        reasoning = last_json.get("reasoning", "")
        
        # Clean up prediction
        if isinstance(prediction, (int, float)):
            prediction = str(prediction)
        elif isinstance(prediction, str):
            prediction = prediction.strip()
        else:
            prediction = str(prediction)
        
        return prediction, reasoning

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with retry logic.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        base_instruction = self._build_prompt(inputs)
        instruction = base_instruction
        
        prediction = "None"
        reasoning = ""
        msg_history = []
        all_msg_histories = []
        
        for attempt in range(self.max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                
                all_msg_histories.extend(msg_history)
                
                prediction, reasoning = self._extract_prediction(msg_history)
                
                # Validate that we got a meaningful prediction
                if prediction != "None" and prediction.strip():
                    self.log_fn(f"Successfully extracted prediction: {prediction}")
                    break
                else:
                    self.log_fn(f"Attempt {attempt + 1}: Failed to extract valid prediction, retrying...")
                    # Add a stronger hint for the next attempt
                    instruction = base_instruction + f"\n\nIMPORTANT: Your previous response did not contain a valid JSON with a 'response' field. Attempt {attempt + 1}/{self.max_retries}. Please respond ONLY with the JSON format specified above, wrapped in <json> tags."
                    
            except Exception as e:
                self.log_fn(f"Attempt {attempt + 1}: Error during LLM call: {e}")
                if attempt == self.max_retries - 1:
                    # Last attempt failed, return what we have
                    break
        
        # Log final result
        if prediction == "None" or not prediction.strip():
            self.log_fn("Warning: Could not extract valid prediction after all retries")
        
        # Return combined message history from all attempts
        return str(prediction), all_msg_histories if all_msg_histories else msg_history
