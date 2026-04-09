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
    Also handles markdown code blocks with json tags.
    """
    results = []
    search_from = 0
    
    # First, try to find <json>...</json> tags
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
            # Try to clean up common JSON issues
            cleaned = _clean_json_string(inner)
            try:
                results.append(json.loads(cleaned))
            except json.JSONDecodeError:
                continue
    
    # Also look for markdown code blocks with json
    md_pattern = re.compile(r'```(?:json)?\s*\n?(.*?)\n?```', re.DOTALL)
    for match in md_pattern.finditer(text):
        inner = match.group(1).strip()
        try:
            parsed = json.loads(inner)
            if isinstance(parsed, dict) and parsed not in results:
                results.append(parsed)
        except json.JSONDecodeError:
            cleaned = _clean_json_string(inner)
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict) and parsed not in results:
                    results.append(parsed)
            except json.JSONDecodeError:
                continue
    
    return results or None


def _clean_json_string(text: str) -> str:
    """Clean up common JSON formatting issues."""
    # Remove trailing commas before closing braces/brackets
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    # Fix single quotes to double quotes (basic handling)
    text = re.sub(r"'([^']*?)':", r'"\1":', text)
    text = re.sub(r":\s*'([^']*?)'", r': "\1"', text)
    return text


def _extract_json_with_regex(text: str) -> list[dict] | None:
    """Fallback JSON extraction using regex for malformed responses."""
    results = []
    
    # Try to find JSON objects in code blocks
    json_pattern = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_pattern:
        try:
            results.append(json.loads(json_pattern.group(1)))
        except json.JSONDecodeError:
            cleaned = _clean_json_string(json_pattern.group(1))
            try:
                results.append(json.loads(cleaned))
            except json.JSONDecodeError:
                pass
    
    # Try to find any JSON-like structure with "response" key
    response_pattern = re.search(r'"response"\s*:\s*"([^"]*)"', text)
    if response_pattern and not results:
        results.append({"response": response_pattern.group(1)})
    
    # Try to find JSON-like structure with single quotes
    response_pattern_single = re.search(r"'response'\s*:\s*'([^']*)'", text)
    if response_pattern_single and not results:
        results.append({"response": response_pattern_single.group(1)})
    
    # Try to extract reasoning field if present
    reasoning_pattern = re.search(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
    if reasoning_pattern and results:
        results[-1]["reasoning"] = reasoning_pattern.group(1).replace('\\n', '\n').replace('\\"', '"')
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3

    def _build_grading_prompt(self, inputs: dict) -> str:
        """Build a structured prompt for the grading task with chain-of-thought."""
        domain = inputs.get("domain", "unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        return f"""You are an expert grader for {domain} problems. Your task is to evaluate a student's answer against the correct solution and grading guidelines.

## Problem Statement
{problem}

## Correct Solution
{solution}

## Grading Guidelines
{guidelines}

## Student's Answer
{student_answer}

## Instructions
1. First, analyze the student's answer step by step. Compare it against the correct solution.
2. Identify any errors, omissions, or alternative valid approaches.
3. Consider the grading guidelines carefully.
4. Provide your reasoning for the grade you will assign.
5. Finally, provide your grade/assessment in the JSON format below.

## Grading Rubric
When assigning grades, consider:
- **Correct**: The answer matches the solution or uses an equivalent valid approach with correct reasoning and final result.
- **Partial**: The answer shows some correct reasoning but has minor errors, incomplete steps, or partially correct results.
- **Incorrect**: The answer contains fundamental errors, wrong approach, or completely wrong results.

## Response Format (REQUIRED)
You MUST respond with valid JSON wrapped in <json>...</json> tags. Do not include any text outside the JSON tags.

<json>
{{
    "reasoning": "Your detailed step-by-step analysis and reasoning...",
    "response": "Your final grade/assessment (e.g., 'Correct', 'Partial', 'Incorrect', or a numeric score)"
}}
</json>

IMPORTANT: Ensure your JSON is valid and properly formatted. The 'response' field should contain only the grade, not the reasoning."""

    def _extract_prediction(self, text: str) -> tuple[str, str]:
        """Extract prediction and reasoning from response text.
        
        Returns:
            (prediction, reasoning) tuple
        """
        prediction = "None"
        reasoning = ""
        
        # Try primary extraction method
        extracted = _extract_jsons(text)
        if extracted is None:
            # Fallback to regex extraction
            extracted = _extract_json_with_regex(text)
        
        if extracted:
            # Find the most complete JSON object (one with both response and reasoning)
            best_json = None
            for json_obj in extracted:
                if "response" in json_obj:
                    if best_json is None:
                        best_json = json_obj
                    # Prefer JSON objects that also have reasoning
                    elif "reasoning" in json_obj and "reasoning" not in best_json:
                        best_json = json_obj
            
            # If no complete object found, use the last one with a response
            if best_json is None:
                for json_obj in reversed(extracted):
                    if "response" in json_obj:
                        best_json = json_obj
                        break
            
            if best_json:
                if "response" in best_json:
                    prediction = str(best_json["response"]).strip()
                if "reasoning" in best_json:
                    reasoning = str(best_json["reasoning"]).strip()
        
        return prediction, reasoning

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_grading_prompt(inputs)
        
        msg_history = []
        prediction = "None"
        reasoning = ""
        
        # Retry loop for robust extraction
        for attempt in range(self.max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=msg_history if attempt > 0 else [],
                )
                
                last_text = msg_history[-1]["text"] if msg_history else ""
                prediction, reasoning = self._extract_prediction(last_text)
                
                if prediction != "None":
                    self.log_fn(f"Successfully extracted prediction: {prediction}")
                    if reasoning:
                        self.log_fn(f"Reasoning: {reasoning[:200]}...")
                    break
                else:
                    self.log_fn(f"Attempt {attempt + 1}: Failed to extract prediction, retrying...")
                    # Add feedback for retry
                    if attempt < self.max_retries - 1:
                        instruction = f"""ERROR: Your previous response did not contain valid JSON with a 'response' field.

You MUST respond with ONLY valid JSON wrapped in <json>...</json> tags. Do not include any text before or after the JSON tags.

Correct format:
<json>
{{
    "reasoning": "Your detailed analysis here...",
    "response": "Correct"
}}
</json>

Now try again with the original task:

{self._build_grading_prompt(inputs)}"""
                    
            except Exception as e:
                self.log_fn(f"Error in attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    prediction = f"Error: {e}"
        
        return str(prediction), msg_history
