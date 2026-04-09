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
    Also handles nested JSON blocks by tracking brace depth with proper
    string literal handling (including escaped characters).
    """
    results = []
    search_from = 0
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        
        # Find the closing tag, but be smart about nested braces
        json_start = start + 6
        brace_start = text.find("{", json_start)
        if brace_start == -1:
            search_from = json_start
            continue
            
        # Track brace depth to find the matching closing brace
        depth = 1
        pos = brace_start + 1
        while pos < len(text) and depth > 0:
            if text[pos] == '{':
                depth += 1
            elif text[pos] == '}':
                depth -= 1
            elif text[pos] == '"':
                # Skip string content to avoid counting braces inside strings
                pos += 1
                while pos < len(text) and text[pos] != '"':
                    if text[pos] == '\\' and pos + 1 < len(text):
                        # Skip escaped character (including escaped quotes and backslashes)
                        pos += 2
                    else:
                        pos += 1
                if pos < len(text):
                    pos += 1
                continue
            elif text[pos] == "'":
                # Also handle single-quoted strings (for Python dict-like output)
                pos += 1
                while pos < len(text) and text[pos] != "'":
                    if text[pos] == '\\' and pos + 1 < len(text):
                        pos += 2
                    else:
                        pos += 1
                if pos < len(text):
                    pos += 1
                continue
            pos += 1
        
        # Now look for </json> after the closing brace
        if depth == 0:
            end_tag_start = text.find("</json>", pos - 1)
            if end_tag_start != -1:
                inner = text[brace_start:pos].strip()
                search_from = end_tag_start + 7
                try:
                    results.append(json.loads(inner))
                except json.JSONDecodeError:
                    # Try to clean up common issues
                    try:
                        # Remove trailing commas before closing braces/brackets
                        cleaned = re.sub(r',(\s*[}\]])', r'\1', inner)
                        # Fix single quotes to double quotes for JSON compatibility
                        cleaned = re.sub(r"'([^']*?)':", r'"\1":', cleaned)
                        cleaned = re.sub(r":\s*'([^']*?)'", r': "\1"', cleaned)
                        results.append(json.loads(cleaned))
                    except json.JSONDecodeError:
                        continue
            else:
                search_from = pos
        else:
            search_from = json_start + 1
            
    return results or None


def _extract_json_fallback(text: str) -> list[dict] | None:
    """Fallback JSON extraction using brace depth parsing for unwrapped JSON objects.

    This handles cases where the model outputs valid JSON without <json> tags.
    Searches for JSON objects with a "response" key.
    Uses robust brace depth tracking with proper string literal handling.
    """
    results = []
    
    def find_json_objects(s: str) -> list[str]:
        """Find potential JSON objects by tracking brace depth with string handling."""
        objects = []
        i = 0
        while i < len(s):
            if s[i] == '{':
                start = i
                depth = 1
                i += 1
                while i < len(s) and depth > 0:
                    if s[i] == '{':
                        depth += 1
                    elif s[i] == '}':
                        depth -= 1
                    elif s[i] == '"':
                        # Skip double-quoted strings
                        i += 1
                        while i < len(s) and s[i] != '"':
                            if s[i] == '\\' and i + 1 < len(s):
                                i += 2
                            else:
                                i += 1
                        if i < len(s):
                            i += 1
                        continue
                    elif s[i] == "'":
                        # Skip single-quoted strings
                        i += 1
                        while i < len(s) and s[i] != "'":
                            if s[i] == '\\' and i + 1 < len(s):
                                i += 2
                            else:
                                i += 1
                        if i < len(s):
                            i += 1
                        continue
                    i += 1
                if depth == 0:
                    objects.append(s[start:i])
            else:
                i += 1
        return objects
    
    def try_parse_with_fixes(obj_str: str) -> dict | None:
        """Try to parse JSON with various cleanup attempts."""
        # Direct parse attempt
        try:
            return json.loads(obj_str)
        except json.JSONDecodeError:
            pass
        
        # Try cleaning up common issues
        try:
            # Remove trailing commas
            cleaned = re.sub(r',(\s*[}\]])', r'\1', obj_str)
            # Fix single quotes to double quotes
            cleaned = re.sub(r"'([^']*?)':", r'"\1":', cleaned)
            cleaned = re.sub(r":\s*'([^']*?)'", r': "\1"', cleaned)
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        
        return None
    
    # Try to parse each potential JSON object
    for obj_str in find_json_objects(text):
        obj = try_parse_with_fixes(obj_str)
        if obj and isinstance(obj, dict) and "response" in obj:
            results.append(obj)
    
    # If no results, try regex pattern for simpler cases
    if not results:
        # Pattern to match JSON objects with response key (simpler cases)
        pattern = r'\{[^{}]*"response"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.finditer(pattern, text, re.DOTALL)
        for match in matches:
            obj = try_parse_with_fixes(match.group())
            if obj and "response" in obj:
                results.append(obj)

    # If still no results, try to parse the entire text as JSON
    if not results:
        obj = try_parse_with_fixes(text.strip())
        if obj and isinstance(obj, dict) and "response" in obj:
            results.append(obj)
    
    # Final fallback: try to extract any dict-like structure with response key
    if not results:
        # Look for patterns like "response": "..." or 'response': '...'
        response_pattern = r'["\']response["\']\s*:\s*["\']([^"\']+)["\']'
        match = re.search(response_pattern, text)
        if match:
            results.append({"response": match.group(1)})

    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with robust JSON extraction."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 2  # Number of retries for failed extractions

    def _build_grading_prompt(self, inputs: dict) -> str:
        """Build the grading prompt from input fields."""
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        return f"""You are an expert IMO (International Mathematical Olympiad) grader. Your task is to evaluate a student's solution to a mathematical problem.

Domain: {domain}

Problem:
{problem}

Official Solution:
{solution}

Grading Guidelines:
{grading_guidelines}

Student's Answer:
{student_answer}

Your task:
1. Carefully analyze the student's answer against the official solution
2. Apply the grading guidelines to determine the appropriate score
3. Provide your evaluation in the exact JSON format below

Respond in JSON format with the following schema:
<json>
{{
    "response": "Your detailed evaluation including the score"
}}
</json>

Important: Ensure your response is valid JSON with the "response" key containing your complete evaluation."""

    def _extract_prediction(self, raw_response: str) -> tuple[str, str]:
        """Extract prediction from LLM response using multiple methods.
        
        Returns:
            (prediction, extraction_method)
        """
        # Try primary extraction first
        try:
            extracted = _extract_jsons(raw_response)
            if extracted and len(extracted) > 0:
                last = extracted[-1]
                if "response" in last:
                    return last["response"], "primary"
                # If no "response" key, use the first available key
                if last:
                    first_key = list(last.keys())[0]
                    return last[first_key], "primary_alt_key"
        except Exception as e:
            self.log_fn(f"Primary extraction error: {e}")
        
        # Try fallback extraction
        try:
            extracted = _extract_json_fallback(raw_response)
            if extracted and len(extracted) > 0:
                last = extracted[-1]
                if "response" in last:
                    return last["response"], "fallback"
                if last:
                    first_key = list(last.keys())[0]
                    return last[first_key], "fallback_alt_key"
        except Exception as e:
            self.log_fn(f"Fallback extraction error: {e}")
        
        # Last resort: return raw response if it looks like text
        if raw_response and len(raw_response.strip()) > 0:
            return raw_response.strip(), "raw_text"
        
        return "None", "failed"

    def _format_prediction(self, prediction: any) -> str:
        """Convert prediction to string, handling various types."""
        if prediction is None:
            return "None"
        elif isinstance(prediction, (list, dict)):
            return json.dumps(prediction)
        else:
            return str(prediction)

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_grading_prompt(inputs)
        
        # Attempt LLM call with retries
        for attempt in range(self.max_retries + 1):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                break
            except Exception as e:
                self.log_fn(f"Error calling LLM (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries:
                    return f"Error: LLM call failed after {self.max_retries + 1} attempts", [{"role": "system", "text": f"Error: {e}"}]
                continue

        # Check if we got a valid response
        if not msg_history or len(msg_history) < 2:
            self.log_fn("Warning: Empty or incomplete message history from LLM")
            return "Error: No response from LLM", msg_history if msg_history else [{"role": "system", "text": "No response"}]

        raw_response = msg_history[-1].get("text", "")
        
        if not raw_response or not raw_response.strip():
            self.log_fn("Warning: Empty response text from LLM")
            return "Error: Empty response from LLM", msg_history
        
        # Extract prediction
        prediction, extraction_method = self._extract_prediction(raw_response)
        self.log_fn(f"Extraction method used: {extraction_method}")
        
        # Format and return
        formatted_prediction = self._format_prediction(prediction)
        return formatted_prediction, msg_history
