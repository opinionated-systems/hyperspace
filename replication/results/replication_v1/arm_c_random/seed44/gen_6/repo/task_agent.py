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
    Now also handles nested JSON objects within the tags.
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
        
        # Try to parse the inner content
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # Try to find valid JSON objects within the content
            # This handles cases where the model adds extra text
            try:
                # Look for the first '{' and last '}' to extract the JSON object
                json_start = inner.find('{')
                json_end = inner.rfind('}')
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    potential_json = inner[json_start:json_end + 1]
                    results.append(json.loads(potential_json))
            except (json.JSONDecodeError, ValueError):
                continue
    return results or None


def _extract_json_with_retry(text: str, max_retries: int = 3) -> list[dict] | None:
    """Extract JSON with multiple fallback strategies.
    
    Tries multiple extraction methods:
    1. Standard <json>...</json> blocks
    2. JSON code blocks ```json...```
    3. Raw JSON objects in text (with improved nested brace handling)
    4. Direct key-value pattern matching for simple responses
    """
    results = []
    
    # Try standard extraction first
    results = _extract_jsons(text)
    if results:
        return results
    
    # Try JSON code blocks
    json_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(json_block_pattern, text, re.DOTALL)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            # Try to extract JSON from within the block
            json_start = match.find('{')
            json_end = match.rfind('}')
            if json_start != -1 and json_end != -1:
                try:
                    results.append(json.loads(match[json_start:json_end + 1].strip()))
                except json.JSONDecodeError:
                    continue
    
    if results:
        return results
    
    # Try to find raw JSON objects with balanced brace counting
    # This handles nested objects better than simple regex
    i = 0
    while i < len(text):
        if text[i] == '{':
            # Found potential start of JSON object
            brace_count = 1
            j = i + 1
            while j < len(text) and brace_count > 0:
                if text[j] == '{':
                    brace_count += 1
                elif text[j] == '}':
                    brace_count -= 1
                j += 1
            
            if brace_count == 0:
                # Found a balanced JSON object
                potential_json = text[i:j]
                try:
                    parsed = json.loads(potential_json)
                    # Only include if it has the expected fields
                    if isinstance(parsed, dict) and ('response' in parsed or 'reasoning' in parsed):
                        results.append(parsed)
                except json.JSONDecodeError:
                    pass
            i = j
        else:
            i += 1
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3

    def _build_grading_prompt(self, inputs: dict) -> str:
        """Build a comprehensive prompt for the grading task."""
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        return f"""You are an expert grader for {domain} problems. Your task is to evaluate a student's answer against the correct solution and grading guidelines.

## Problem
{problem}

## Correct Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Instructions
1. Carefully analyze the student's answer step by step
2. Compare it against the correct solution
3. Apply the grading guidelines strictly
4. Provide your reasoning before giving the final grade
5. Respond in JSON format with the following schema:

<json>
{{
    "reasoning": "Your detailed step-by-step analysis and reasoning",
    "response": "The final grade/score (a number or specific grade value)"
}}
</json>

Important: 
- The "response" field must contain only the final grade/score
- Use the "reasoning" field to show your work
- Be precise and follow the grading guidelines exactly"""

    def _try_extract_prediction(self, text: str) -> tuple[str, str | None]:
        """Try to extract prediction from response text.
        
        Returns:
            (prediction, reasoning)
        """
        try:
            extracted = _extract_json_with_retry(text)
            if extracted:
                # Use the first valid result that has a response field
                for result in extracted:
                    if isinstance(result, dict) and "response" in result:
                        prediction = result.get("response", "None")
                        reasoning = result.get("reasoning")
                        # Validate prediction is not empty or whitespace
                        if prediction and str(prediction).strip():
                            return str(prediction), reasoning
                # Fallback to last result if no valid response found
                last = extracted[-1]
                prediction = last.get("response", "None")
                reasoning = last.get("reasoning")
                return str(prediction), reasoning
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
        
        return "None", None

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with retry logic.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_grading_prompt(inputs)
        msg_history = []
        prediction = "None"
        last_error = None
        
        # Try with retries
        for attempt in range(self.max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=msg_history if attempt > 0 else [],
                )
                
                # Extract prediction
                text = msg_history[-1]["text"] if msg_history else ""
                pred, reasoning = self._try_extract_prediction(text)
                
                if pred != "None":
                    prediction = pred
                    if reasoning:
                        self.log_fn(f"Grading reasoning: {reasoning[:200]}...")
                    break
                
                # If extraction failed, log the issue for debugging
                last_error = f"Failed to extract valid prediction on attempt {attempt + 1}"
                self.log_fn(f"{last_error}. Response text length: {len(text)}")
                
                # If extraction failed, add a follow-up message asking for proper format
                if attempt < self.max_retries - 1:
                    instruction = (
                        "Your previous response did not contain a valid JSON object with 'response' and 'reasoning' fields. "
                        "Please respond ONLY in the required JSON format:\n\n"
                        "<json>\n"
                        '{\n  "reasoning": "Your detailed analysis here",\n  "response": "The final grade/score here"\n}'
                        "\n</json>"
                    )
                    
            except Exception as e:
                last_error = str(e)
                self.log_fn(f"Attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    break
        
        # Log final status
        if prediction == "None" and last_error:
            self.log_fn(f"All retry attempts exhausted. Last error: {last_error}")
        
        return str(prediction), msg_history
