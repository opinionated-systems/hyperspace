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


def _extract_any_json(text: str) -> list[dict] | None:
    """Extract JSON objects from text using multiple strategies.
    
    Tries multiple patterns to find JSON objects in the text.
    """
    results = []
    
    # Strategy 1: Look for <json>...</json> blocks
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
    
    # Strategy 2: Look for ```json code blocks
    search_from = 0
    while True:
        start = text.find("```json", search_from)
        if start == -1:
            break
        end = text.find("```", start + 7)
        if end == -1:
            break
        inner = text[start + 7:end].strip()
        search_from = end + 3
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            continue
    
    # Strategy 3: Look for JSON objects with "response" key
    # Match patterns like {"response": "..."} or { "response" : "..." }
    pattern = r'\{\s*"response"\s*:\s*"([^"]*)"\s*\}'
    matches = re.findall(pattern, text)
    for match in matches:
        results.append({"response": match})
    
    # Strategy 4: Look for any JSON object in the text
    # Find content between curly braces
    pattern = r'\{[^{}]*\}'
    matches = re.findall(pattern, text)
    for match in matches:
        try:
            obj = json.loads(match)
            if "response" in obj:
                results.append(obj)
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
        
        # Parse grading guidelines to understand the scoring system
        points_info = ""
        if "points" in grading_guidelines.lower() or "mark" in grading_guidelines.lower():
            points_info = "\nNote: The grading guidelines specify a points-based scoring system. Extract the exact point value from the guidelines."
        
        # Determine expected output format from grading guidelines
        expected_format = "category"  # default
        if any(x in grading_guidelines.lower() for x in ["point", "mark", "score", "out of", "/"]):
            expected_format = "numeric"
        
        instruction = f"""You are an expert mathematics grader evaluating student solutions to competition mathematics problems.

## Problem:
{problem}

## Correct Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}{points_info}

## Student's Answer:
{student_answer}

## Your Task:
Carefully evaluate the student's answer against the grading guidelines. Follow these steps:

1. **Understand the Problem**: Identify what the problem is asking and what constitutes a correct solution.

2. **Analyze the Student's Answer**: 
   - Check if the student understands the problem correctly
   - Verify if the reasoning is mathematically sound
   - Check if the student reaches the correct conclusion
   - Identify any errors, gaps, or misconceptions

3. **Compare to Grading Guidelines**:
   - Look for specific criteria mentioned in the guidelines
   - Check if the student met partial credit conditions
   - Identify which parts of the solution are correct/incorrect

4. **Determine the Score**:
   - If the guidelines specify points, provide the exact point value
   - If the guidelines use categories (Correct/Partial/Incorrect/Almost), use those exact terms
   - Be precise and consistent with the guidelines

## Output Format:
You MUST respond with a JSON object containing your evaluation. Use EXACTLY this format:

<json>
{{
    "response": "Your evaluation here - either a number (points) or a category label"
}}
</json>

Important: 
- The "response" field should contain ONLY the evaluation (number or category)
- Do not include explanations in the response field
- Use the exact format shown above with <json> tags
- Expected format: {expected_format}"""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"Error calling LLM: {e}")
            return "Error", []

        # Extract prediction from JSON using multiple strategies
        prediction = "None"
        try:
            if not msg_history:
                return "None", msg_history
                
            response_text = msg_history[-1].get("text", "")
            if not response_text:
                return "None", msg_history
            
            # Try multiple extraction strategies
            extracted = _extract_any_json(response_text)
            
            if extracted:
                # Find the first JSON with a "response" key
                for obj in extracted:
                    if isinstance(obj, dict) and "response" in obj:
                        prediction = str(obj["response"]).strip()
                        break
            
            # If still None, try direct text extraction as fallback
            if prediction == "None":
                # Look for common patterns in the response
                lines = response_text.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    # Skip empty lines and common prefixes
                    if not line or line.startswith('#') or line.startswith('//') or line.startswith('/*'):
                        continue
                    # Look for numeric scores or category labels
                    if line in ['Correct', 'Partial', 'Incorrect', 'Almost', '0', '1', '2', '3', '4', '5', '6', '7']:
                        prediction = line
                        break
                    # Look for patterns like "Score: 5" or "Points: 3"
                    match = re.search(r'(?:score|points|grade|mark)s?[:\s]+(\d+)', line, re.IGNORECASE)
                    if match:
                        prediction = match.group(1)
                        break
                    # Look for standalone numbers that might be scores
                    match = re.search(r'^\s*(\d+(?:\.\d+)?)\s*$', line)
                    if match:
                        prediction = match.group(1)
                        break
                        
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
