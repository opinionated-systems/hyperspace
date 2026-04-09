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


def _extract_json_fallback(text: str) -> list[dict] | None:
    """Fallback JSON extraction using regex for unwrapped JSON objects.

    This handles cases where the model outputs valid JSON without <json> tags.
    Searches for JSON objects with a "response" key.
    Improved to handle nested braces more robustly.
    """
    results = []
    
    # First, try to find JSON objects by parsing brace depth
    # This handles nested structures better than regex
    def find_json_objects(s: str) -> list[str]:
        """Find potential JSON objects by tracking brace depth."""
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
                    i += 1
                if depth == 0:
                    objects.append(s[start:i])
            else:
                i += 1
        return objects
    
    # Try to parse each potential JSON object
    for obj_str in find_json_objects(text):
        try:
            obj = json.loads(obj_str)
            if isinstance(obj, dict) and "response" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            continue
    
    # If no results, try regex pattern for simpler cases
    if not results:
        # Pattern to match JSON objects with response key (simpler cases)
        pattern = r'\{[^{}]*"response"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.finditer(pattern, text, re.DOTALL)
        for match in matches:
            try:
                obj = json.loads(match.group())
                if "response" in obj:
                    results.append(obj)
            except json.JSONDecodeError:
                continue

    # If still no results, try to parse the entire text as JSON
    if not results:
        try:
            obj = json.loads(text.strip())
            if isinstance(obj, dict) and "response" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            pass
    
    # Final fallback: try to extract any dict-like structure with response key
    if not results:
        # Look for patterns like "response": "..." or 'response': '...'
        response_pattern = r'["\']response["\']\s*:\s*["\']([^"\']+)["\']'
        match = re.search(response_pattern, text)
        if match:
            results.append({"response": match.group(1)})

    return results or None


def _clean_and_normalize_response(prediction: any) -> str:
    """Clean and normalize the prediction response.
    
    Handles various types and edge cases to ensure consistent output.
    """
    if prediction is None:
        return "None"
    elif isinstance(prediction, bool):
        return "true" if prediction else "false"
    elif isinstance(prediction, (list, dict)):
        return json.dumps(prediction)
    elif isinstance(prediction, (int, float)):
        return str(prediction)
    else:
        # String type - clean up common formatting issues
        pred_str = str(prediction).strip()
        # Remove surrounding quotes if present
        if (pred_str.startswith('"') and pred_str.endswith('"')) or \
           (pred_str.startswith("'") and pred_str.endswith("'")):
            pred_str = pred_str[1:-1]
        return pred_str


class TaskAgent:
    """Task agent that solves IMO grading problems with robust JSON extraction."""

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
        # Build a more structured prompt with clear instructions
        domain = inputs.get('domain', 'unknown')
        problem = inputs.get('problem', '')
        solution = inputs.get('solution', '')
        grading_guidelines = inputs.get('grading_guidelines', '')
        student_answer = inputs.get('student_answer', '')
        
        instruction = f"""You are an expert grading agent. Your task is to evaluate a student's answer to a problem.

Domain: {domain}

Problem:
{problem}

Correct Solution:
{solution}

Grading Guidelines:
{grading_guidelines}

Student's Answer:
{student_answer}

Your task: Evaluate the student's answer and provide your assessment in the response field.

IMPORTANT: You must respond using the exact JSON format below. Do not include any text outside the JSON tags.

<json>
{{
    "response": "Your evaluation/grade here"
}}
</json>"""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"Error calling LLM: {e}")
            return "Error: LLM call failed", [{"role": "system", "text": f"Error: {e}"}]

        # Check if we got a valid response
        if not msg_history or len(msg_history) < 2:
            self.log_fn("Warning: Empty or incomplete message history from LLM")
            return "Error: No response from LLM", msg_history if msg_history else [{"role": "system", "text": "No response"}]

        # Extract prediction from JSON using primary method
        prediction = None
        extraction_method = "none"
        raw_response = msg_history[-1].get("text", "")
        
        if not raw_response or not raw_response.strip():
            self.log_fn("Warning: Empty response text from LLM")
            return "Error: Empty response from LLM", msg_history
        
        # Try primary extraction first
        try:
            extracted = _extract_jsons(raw_response)
            if extracted and len(extracted) > 0:
                last_extracted = extracted[-1]
                if isinstance(last_extracted, dict) and "response" in last_extracted:
                    prediction = last_extracted["response"]
                    extraction_method = "primary"
                    self.log_fn(f"Primary extraction succeeded, response type: {type(prediction).__name__}")
        except Exception as e:
            self.log_fn(f"Primary extraction error: {e}")
        
        # Try fallback if primary failed
        if prediction is None:
            try:
                self.log_fn("Primary extraction failed, trying fallback...")
                extracted = _extract_json_fallback(raw_response)
                if extracted and len(extracted) > 0:
                    last_extracted = extracted[-1]
                    if isinstance(last_extracted, dict) and "response" in last_extracted:
                        prediction = last_extracted["response"]
                        extraction_method = "fallback"
                        self.log_fn(f"Fallback extraction succeeded, response type: {type(prediction).__name__}")
            except Exception as e:
                self.log_fn(f"Fallback extraction error: {e}")
        
        # Log extraction results
        if prediction is None:
            self.log_fn("Warning: No valid JSON with 'response' key found in LLM output")
            self.log_fn(f"Raw response preview: {raw_response[:200]}...")
        else:
            self.log_fn(f"Extraction method used: {extraction_method}")
        
        # Normalize and return the prediction
        normalized_prediction = _clean_and_normalize_response(prediction)
            
        return normalized_prediction, msg_history
