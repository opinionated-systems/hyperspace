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
    Also handles nested JSON objects and common formatting issues.
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
        
        # Try to parse the JSON
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # Try to fix common JSON issues
            # 1. Remove trailing commas before closing braces/brackets
            fixed = re.sub(r',(\s*[}\]])', r'\1', inner)
            # 2. Fix unescaped newlines in strings (basic attempt)
            try:
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                # 3. Try to extract just the response field if it's a simple case
                try:
                    # Look for response field pattern
                    response_match = re.search(r'"response"\s*:\s*"([^"]*)"', fixed)
                    reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', fixed, re.DOTALL)
                    if response_match:
                        obj = {"response": response_match.group(1)}
                        if reasoning_match:
                            obj["reasoning"] = reasoning_match.group(1)
                        results.append(obj)
                except Exception:
                    continue
    return results or None


def _extract_json_with_regex(text: str) -> list[dict] | None:
    """Fallback JSON extraction using regex for malformed responses."""
    results = []
    
    # Try to find JSON objects in code blocks (handle nested braces)
    # Use a more robust pattern that balances braces
    code_block_pattern = re.search(r'```(?:json)?\s*\n?(\{[\s\S]*?\})\s*```', text, re.DOTALL)
    if code_block_pattern:
        try:
            json_str = code_block_pattern.group(1)
            # Fix trailing commas
            json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
            results.append(json.loads(json_str))
        except json.JSONDecodeError:
            pass
    
    # Try to find any JSON-like structure with "response" key
    response_pattern = re.search(r'"response"\s*:\s*"([^"]*)"', text)
    if response_pattern and not results:
        obj = {"response": response_pattern.group(1)}
        # Also try to extract reasoning
        reasoning_pattern = re.search(r'"reasoning"\s*:\s*"([^"]*)"', text, re.DOTALL)
        if reasoning_pattern:
            obj["reasoning"] = reasoning_pattern.group(1)
        results.append(obj)
    
    # Last resort: look for standalone response after certain keywords
    if not results:
        # Look for patterns like "Grade: Correct" or "Assessment: Partial"
        grade_pattern = re.search(r'(?:grade|assessment|evaluation|result)\s*[:\-]?\s*(correct|partial|incorrect|none)', text, re.IGNORECASE)
        if grade_pattern:
            results.append({"response": grade_pattern.group(1).capitalize()})
    
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
        
        # Check if this is a numeric/score-based grading task
        has_numeric_guidelines = any(char.isdigit() for char in guidelines) and ("point" in guidelines.lower() or "score" in guidelines.lower() or "mark" in guidelines.lower())
        
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

IMPORTANT: 
- Ensure your JSON is valid and properly formatted
- The 'response' field should contain only the grade, not the reasoning
- Do NOT use markdown inside the JSON values
- Do NOT include trailing commas in the JSON
- Valid response values: "Correct", "Partial", "Incorrect"{', or a numeric score' if has_numeric_guidelines else ''}"""

    def _extract_prediction(self, text: str) -> tuple[str, str]:
        """Extract prediction and reasoning from response text.
        
        Returns:
            (prediction, reasoning) tuple
        """
        prediction = "None"
        reasoning = ""
        
        # Try primary extraction method
        extracted = _extract_jsons(text)
        extraction_method = "primary"
        if extracted is None:
            # Fallback to regex extraction
            extracted = _extract_json_with_regex(text)
            extraction_method = "regex_fallback"
        
        if extracted:
            last_json = extracted[-1]
            if "response" in last_json:
                prediction = str(last_json["response"]).strip()
                # Normalize common grade formats
                prediction_lower = prediction.lower()
                if prediction_lower in ["correct", "right", "true", "yes"]:
                    prediction = "Correct"
                elif prediction_lower in ["partial", "partially correct", "partial credit"]:
                    prediction = "Partial"
                elif prediction_lower in ["incorrect", "wrong", "false", "no", "error"]:
                    prediction = "Incorrect"
            if "reasoning" in last_json:
                reasoning = str(last_json["reasoning"]).strip()
            
            self.log_fn(f"Extraction method: {extraction_method}, JSONs found: {len(extracted)}")
        else:
            self.log_fn(f"Warning: No JSON extracted from response (method: {extraction_method})")
        
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
                    # Add feedback for retry with more specific guidance
                    if attempt < self.max_retries - 1:
                        instruction = f"""ERROR: Your previous response did not contain valid JSON with a 'response' field.

You MUST respond with ONLY valid JSON wrapped in <json>...</json> tags. Do not include any text before or after the JSON tags.

Common mistakes to avoid:
1. Do NOT include markdown formatting inside the JSON
2. Do NOT use trailing commas (e.g., {{"a": 1,}} is invalid)
3. Do NOT include unescaped newlines in string values
4. Ensure all quotes are properly closed

Correct format:
<json>
{{
    "reasoning": "Your detailed analysis here...",
    "response": "Correct"
}}
</json>

Valid response values are: "Correct", "Partial", or "Incorrect"

Now try again with the original task:

{self._build_grading_prompt(inputs)}"""
                    
            except Exception as e:
                self.log_fn(f"Error in attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    prediction = f"Error: {e}"
        
        return str(prediction), msg_history
