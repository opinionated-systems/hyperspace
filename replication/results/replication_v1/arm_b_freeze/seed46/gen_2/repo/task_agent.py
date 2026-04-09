"""
Task agent: solves a given task with chain-of-thought reasoning and verification.

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


def _extract_json_with_regex(text: str) -> list[dict] | None:
    """Fallback JSON extraction using regex for curly braces."""
    results = []
    # Find JSON objects in code blocks or standalone
    patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
        r'\{[^{}]*"response"[^{}]*\}',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                # Try to find complete JSON object
                if '"response"' in match:
                    # Find the outermost braces
                    start = text.find(match)
                    if start != -1:
                        # Count braces to find matching close
                        brace_count = 0
                        end = start
                        for i, char in enumerate(text[start:]):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    end = start + i + 1
                                    break
                        try:
                            obj = json.loads(text[start:end])
                            if "response" in obj:
                                results.append(obj)
                        except json.JSONDecodeError:
                            continue
            except Exception:
                continue
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3

    def _build_grading_prompt(self, inputs: dict) -> str:
        """Build a structured prompt for IMO grading with chain-of-thought reasoning."""
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        prompt = f"""You are an expert {domain} grader evaluating student solutions.

## Problem Statement
{problem}

## Correct Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Your Task
Evaluate the student's answer step by step:

1. **Understand the Problem**: Identify what the problem is asking and the key concepts involved.

2. **Analyze the Correct Solution**: Understand the expected approach and correct answer.

3. **Review Grading Guidelines**: Note the specific criteria for scoring.

4. **Evaluate Student's Answer**:
   - Check if the approach is correct
   - Verify calculations and reasoning
   - Identify any errors or misconceptions
   - Assess completeness of the solution

5. **Determine Score**: Based on the grading guidelines, assign an appropriate score.

6. **Provide Reasoning**: Explain your evaluation clearly.

Respond in JSON format with the following schema:
<json>
{{
    "response": "Your final evaluation/score here",
    "reasoning": "Detailed explanation of your evaluation",
    "score": "Numerical score if applicable",
    "correctness": "correct/partially_correct/incorrect"
}}
</json>

The "response" field should contain your primary answer (score or evaluation)."""

        return prompt

    def _extract_prediction(self, text: str) -> tuple[str, dict]:
        """Extract prediction from response text with multiple fallback strategies."""
        # Try primary extraction method
        extracted = _extract_jsons(text)
        if not extracted:
            # Try fallback regex extraction
            extracted = _extract_json_with_regex(text)
        
        if not extracted:
            return "None", {}
        
        # Get the last valid JSON object
        data = extracted[-1]
        
        # Try to get response field
        prediction = data.get("response", "None")
        
        return str(prediction), data

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with retry logic.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_grading_prompt(inputs)
        
        prediction = "None"
        msg_history = []
        
        for attempt in range(self.max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=msg_history if attempt > 0 else [],
                )
                
                # Extract prediction from the last assistant message
                last_text = msg_history[-1]["text"] if msg_history else ""
                prediction, extracted_data = self._extract_prediction(last_text)
                
                if prediction != "None":
                    self.log_fn(f"Successfully extracted prediction on attempt {attempt + 1}")
                    break
                else:
                    self.log_fn(f"Failed to extract prediction on attempt {attempt + 1}, retrying...")
                    # Add a follow-up message to guide the model
                    if attempt < self.max_retries - 1:
                        instruction = "Please provide your evaluation in the required JSON format with a 'response' field."
                    
            except Exception as e:
                self.log_fn(f"Error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    break
        
        return str(prediction), msg_history
