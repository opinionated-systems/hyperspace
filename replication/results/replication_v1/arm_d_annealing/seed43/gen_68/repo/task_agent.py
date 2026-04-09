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
    Also handles nested JSON objects within the tags.
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
        
        # Extract content between tags
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        # Try to parse the inner content as JSON
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # Try to find valid JSON objects within the content
            # This handles cases where there might be extra text
            try:
                # Look for JSON object boundaries
                brace_start = inner.find('{')
                brace_end = inner.rfind('}')
                if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
                    json_candidate = inner[brace_start:brace_end + 1]
                    results.append(json.loads(json_candidate))
            except (json.JSONDecodeError, ValueError):
                # Try array format
                try:
                    bracket_start = inner.find('[')
                    bracket_end = inner.rfind(']')
                    if bracket_start != -1 and bracket_end != -1 and bracket_end > bracket_start:
                        json_candidate = inner[bracket_start:bracket_end + 1]
                        results.append(json.loads(json_candidate))
                except (json.JSONDecodeError, ValueError):
                    continue
    return results or None


def _extract_json_fuzzy(text: str) -> list[dict] | None:
    """Fuzzy JSON extraction as fallback for malformed responses.
    
    Tries to find JSON objects even without proper <json> tags.
    Uses multiple strategies to extract valid JSON from messy responses.
    """
    results = []
    
    # Strategy 1: Try to find JSON objects in code blocks
    code_block_pattern = r'```(?:json)?\s*(.*?)```'
    for match in re.finditer(code_block_pattern, text, re.DOTALL):
        try:
            results.append(json.loads(match.group(1).strip()))
        except json.JSONDecodeError:
            # Try to extract JSON from within the code block content
            content = match.group(1).strip()
            try:
                brace_start = content.find('{')
                brace_end = content.rfind('}')
                if brace_start != -1 and brace_end != -1:
                    results.append(json.loads(content[brace_start:brace_end + 1]))
            except (json.JSONDecodeError, ValueError):
                pass
            continue
    
    # Strategy 2: Try to find JSON objects with curly braces using brace counting
    if not results:
        # Find all potential JSON starting points
        for start_idx in [m.start() for m in re.finditer(r'\{', text)]:
            try:
                # Count braces to find complete JSON object
                brace_count = 0
                for i, char in enumerate(text[start_idx:]):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            candidate = text[start_idx:start_idx+i+1]
                            try:
                                parsed = json.loads(candidate)
                                # Only accept objects with expected keys
                                if any(key in parsed for key in ["response", "reasoning", "answer", "result", "grade"]):
                                    results.append(parsed)
                            except json.JSONDecodeError:
                                pass
                            break
            except Exception:
                continue
    
    # Strategy 3: Look for key-value patterns that might indicate JSON
    if not results:
        # Try to find patterns like "key": "value" or 'key': 'value'
        key_value_pattern = r'["\'](\w+)["\']\s*:\s*["\']([^"\']+)["\']'
        matches = list(re.finditer(key_value_pattern, text))
        if len(matches) >= 2:
            # Try to construct a JSON object from key-value pairs
            try:
                # Find the outermost braces
                start = text.find('{')
                end = text.rfind('}')
                if start != -1 and end != -1 and end > start:
                    results.append(json.loads(text[start:end+1]))
            except (json.JSONDecodeError, ValueError):
                pass
    
    # Strategy 4: Try to extract from markdown-style responses
    if not results:
        # Look for patterns like "reasoning: ... response: ..."
        reasoning_match = re.search(r'reasoning[:\s]+(.+?)(?=response[:\s]|$)', text, re.DOTALL | re.IGNORECASE)
        response_match = re.search(r'response[:\s]+(.+?)(?=\n\n|$)', text, re.DOTALL | re.IGNORECASE)
        if reasoning_match or response_match:
            constructed = {}
            if reasoning_match:
                constructed["reasoning"] = reasoning_match.group(1).strip()
            if response_match:
                constructed["response"] = response_match.group(1).strip()
            if constructed:
                results.append(constructed)
    
    # Strategy 5: Try to fix common JSON formatting issues
    if not results:
        # Try to find and fix single-quoted JSON
        single_quoted_pattern = r"\{[^{}]*\}"
        for match in re.finditer(single_quoted_pattern, text):
            try:
                candidate = match.group(0)
                # Replace single quotes with double quotes for JSON keys/values
                fixed = re.sub(r"'([^']*?)'\s*:", r'"\1":', candidate)
                fixed = re.sub(r":\s*'([^']*?)'", r': "\1"', fixed)
                results.append(json.loads(fixed))
            except (json.JSONDecodeError, ValueError):
                pass
    
    # Remove duplicates while preserving order
    seen = set()
    unique_results = []
    for r in results:
        r_str = json.dumps(r, sort_keys=True)
        if r_str not in seen:
            seen.add(r_str)
            unique_results.append(r)
    
    return unique_results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "", max_retries: int = 2) -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = max_retries

    def _build_grading_prompt(self, inputs: dict) -> str:
        """Build a structured prompt for grading."""
        domain = inputs.get("domain", "unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        return f"""You are an expert grader for {domain} problems. Your task is to evaluate a student's answer against the official solution and grading guidelines.

## Problem Statement
{problem}

## Official Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Instructions
1. First, analyze the student's answer step by step. Compare it to the official solution.
2. Identify what the student got correct and what they got wrong.
3. Consider the grading guidelines carefully.
4. Provide your reasoning for the grade you will assign.
5. Finally, provide your grade/assessment in the response field.

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis and reasoning for the grade",
    "response": "Your final grade/assessment (e.g., 'Correct', 'Partially Correct', 'Incorrect', or a numeric score)"
}}
</json>

Ensure your JSON is valid and properly formatted."""

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history with multiple fallback strategies."""
        try:
            last_message = msg_history[-1]["text"]
            
            # Try standard extraction first
            extracted = _extract_jsons(last_message)
            
            # If that fails, try fuzzy extraction
            if not extracted:
                extracted = _extract_json_fuzzy(last_message)
            
            if extracted:
                # Prefer response field, but fallback to other common fields
                last_extracted = extracted[-1]
                
                # Priority order for response fields
                priority_fields = ["response", "answer", "result", "grade", "evaluation", "assessment"]
                
                for field in priority_fields:
                    if field in last_extracted:
                        field_value = last_extracted[field]
                        # Handle different value types
                        if isinstance(field_value, str):
                            return field_value.strip()
                        elif isinstance(field_value, (int, float, bool)):
                            return str(field_value)
                        elif isinstance(field_value, (list, dict)):
                            return json.dumps(field_value)
                        else:
                            return str(field_value)
                
                # If no recognized field, use the whole JSON as string
                return json.dumps(last_extracted)
            else:
                self.log_fn("No JSON found in response, using raw text")
                # Fallback: use the raw response text (truncated)
                return last_message[:500] if len(last_message) > 500 else last_message
                
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            return "None"

    def _clean_prediction(self, prediction: str) -> str:
        """Clean up the prediction string."""
        if not isinstance(prediction, str):
            prediction = str(prediction)
        
        prediction = prediction.strip()
        
        # Remove common prefixes that might appear
        prefixes_to_remove = ["Response:", "Answer:", "Result:", "Grade:", "Evaluation:"]
        for prefix in prefixes_to_remove:
            if prediction.startswith(prefix):
                prediction = prediction[len(prefix):].strip()
        
        return prediction

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with retry logic.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_grading_prompt(inputs)
        
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                
                # Extract prediction
                prediction = self._extract_prediction(msg_history)
                prediction = self._clean_prediction(prediction)
                
                # Validate that we got a meaningful prediction
                if prediction and prediction != "None" and len(prediction) > 0:
                    self.log_fn(f"Successfully extracted prediction on attempt {attempt + 1}: {prediction}")
                    return prediction, msg_history
                
                # If prediction is empty/None, retry with a reminder
                if attempt < self.max_retries:
                    self.log_fn(f"Empty prediction on attempt {attempt + 1}, retrying...")
                    instruction += "\n\nIMPORTANT: Your previous response did not contain a valid grade. Please provide your grade in the 'response' field of the JSON."
                    
            except Exception as e:
                last_error = e
                self.log_fn(f"Error on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries:
                    continue
        
        # If all retries failed, return the last error or a default
        self.log_fn(f"All {self.max_retries + 1} attempts failed. Last error: {last_error}")
        return "Error: Failed to extract valid prediction", msg_history if 'msg_history' in locals() else []
