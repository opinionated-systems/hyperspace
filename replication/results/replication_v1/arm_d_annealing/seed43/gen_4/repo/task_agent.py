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
            # Try to fix common JSON issues
            try:
                # Remove trailing commas before closing braces/brackets
                fixed = re.sub(r',(\s*[}\]])', r'\1', inner)
                # Fix single quotes to double quotes (common LLM error)
                fixed = re.sub(r"(?<!\\)'", '"', fixed)
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                continue
    return results or None


def _extract_json_fuzzy(text: str) -> list[dict] | None:
    """Fuzzy JSON extraction for when standard extraction fails.
    
    Tries to find JSON objects even without proper <json> tags.
    """
    results = []
    # Try to find JSON-like structures with curly braces
    brace_count = 0
    start_idx = None
    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx is not None:
                try:
                    json_str = text[start_idx:i+1]
                    results.append(json.loads(json_str))
                except json.JSONDecodeError:
                    # Try common fixes
                    try:
                        fixed = re.sub(r',(\s*[}\]])', r'\1', json_str)
                        fixed = re.sub(r"(?<!\\)'", '"', fixed)
                        results.append(json.loads(fixed))
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

Your task is to evaluate a student's answer to a mathematical problem with precision and accuracy.

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

INSTRUCTIONS:
1. Carefully read the problem and understand what is being asked.
2. Study the official solution to understand the correct approach and final answer.
3. Analyze the student's answer step by step:
   - Does the student understand the problem?
   - Is the approach mathematically sound?
   - Are the calculations correct?
   - Does the student arrive at the correct final answer?
4. Check for partial credit:
   - Correct approach but calculation error
   - Correct final answer but incomplete reasoning
   - Partial progress toward solution
5. Be strict but fair in your evaluation.
6. Provide your final grade in the exact JSON format below.

IMPORTANT: Your response MUST be valid JSON wrapped in <json> tags.

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis here. Explain your thought process clearly.",
    "response": "Your final grade/assessment here"
}}
</json>

The "response" field should contain only the final grade (e.g., "7", "5", "0", "Correct", "Incorrect", "Partial", etc.).
The "reasoning" field should contain your detailed analysis of why you gave that grade."""

    def _extract_prediction(self, msg_history: list[dict]) -> tuple[str, str]:
        """Extract prediction and reasoning from message history.
        
        Returns:
            (prediction, reasoning) tuple
        """
        if not msg_history:
            return "None", ""
        
        # Search through all messages in reverse order to find the most recent valid JSON
        for msg in reversed(msg_history):
            text = msg.get("text", "")
            if not text:
                continue
            
            # Try standard extraction first
            extracted = _extract_jsons(text)
            if not extracted:
                # Try fuzzy extraction as fallback
                extracted = _extract_json_fuzzy(text)
            
            if extracted:
                last_json = extracted[-1]
                
                # Try multiple possible keys for the response
                prediction = None
                for key in ["response", "grade", "answer", "result", "prediction", "score"]:
                    if key in last_json:
                        prediction = last_json[key]
                        break
                
                if prediction is None:
                    prediction = "None"
                
                # Get reasoning
                reasoning = last_json.get("reasoning", "")
                if not reasoning:
                    # Try alternative keys for reasoning
                    for key in ["analysis", "explanation", "thought", "thoughts", "evaluation"]:
                        if key in last_json:
                            reasoning = last_json[key]
                            break
                
                # Clean up prediction
                if isinstance(prediction, (int, float)):
                    prediction = str(prediction)
                elif isinstance(prediction, str):
                    prediction = prediction.strip()
                else:
                    prediction = str(prediction)
                
                # Validate prediction is not empty
                if prediction and prediction != "None":
                    return prediction, str(reasoning)
        
        return "None", ""

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
        all_msg_history = []
        
        for attempt in range(self.max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                
                all_msg_history.extend(msg_history)
                
                prediction, reasoning = self._extract_prediction(msg_history)
                
                # Validate that we got a meaningful prediction
                if prediction != "None" and prediction.strip():
                    self.log_fn(f"Successfully extracted prediction: {prediction}")
                    break
                else:
                    self.log_fn(f"Attempt {attempt + 1}: Failed to extract valid prediction, retrying...")
                    # Build a more specific hint based on what went wrong
                    if not msg_history or not msg_history[-1].get("text", "").strip():
                        instruction = base_instruction + "\n\nIMPORTANT: Please provide a complete response with your analysis and grade."
                    else:
                        last_text = msg_history[-1].get("text", "")
                        if "<json>" not in last_text:
                            instruction = base_instruction + "\n\nIMPORTANT: Your response MUST be wrapped in <json>...</json> tags."
                        elif "</json>" not in last_text:
                            instruction = base_instruction + "\n\nIMPORTANT: You started with <json> but forgot to close with </json>."
                        else:
                            instruction = base_instruction + "\n\nIMPORTANT: Make sure your JSON is valid and includes both 'reasoning' and 'response' fields."
                    
            except Exception as e:
                self.log_fn(f"Attempt {attempt + 1}: Error during LLM call: {e}")
                if attempt == self.max_retries - 1:
                    # Last attempt failed, return what we have
                    break
        
        # Log final result
        if prediction == "None" or not prediction.strip():
            self.log_fn("Warning: Could not extract valid prediction after all retries")
        
        return str(prediction), all_msg_history
