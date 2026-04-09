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
    """Fallback JSON extraction using multiple strategies for unwrapped JSON objects.

    This handles cases where the model outputs valid JSON without <json> tags.
    Searches for JSON objects with a "response" key.
    Uses a robust brace-depth parser with quote-aware parsing for better accuracy.
    """
    results = []
    
    # Strategy 1: Brace-depth parser with quote awareness
    # This handles nested structures and ignores braces inside strings
    def find_json_objects(s: str) -> list[str]:
        """Find potential JSON objects by tracking brace depth with quote awareness."""
        objects = []
        i = 0
        n = len(s)
        
        while i < n:
            # Look for start of object
            if s[i] == '{':
                start = i
                depth = 1
                i += 1
                in_string = False
                escape_next = False
                
                while i < n and depth > 0:
                    char = s[i]
                    
                    if escape_next:
                        escape_next = False
                        i += 1
                        continue
                    
                    if char == '\\' and in_string:
                        escape_next = True
                        i += 1
                        continue
                    
                    if char == '"' and not in_string:
                        in_string = True
                    elif char == '"' and in_string:
                        in_string = False
                    elif not in_string:
                        if char == '{':
                            depth += 1
                        elif char == '}':
                            depth -= 1
                    
                    i += 1
                    
                    if depth == 0:
                        objects.append(s[start:i])
                        break
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
    
    # Strategy 2: Try regex pattern for simpler flat JSON objects
    if not results:
        # Pattern to match simple JSON objects with response key (no nested objects)
        pattern = r'\{\s*"response"\s*:\s*"([^"]*)"\s*\}'
        matches = re.finditer(pattern, text, re.DOTALL)
        for match in matches:
            try:
                obj = json.loads(match.group())
                if "response" in obj:
                    results.append(obj)
            except json.JSONDecodeError:
                continue
    
    # Strategy 3: Try to parse the entire text as JSON
    if not results:
        try:
            obj = json.loads(text.strip())
            if isinstance(obj, dict) and "response" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            pass
    
    # Strategy 4: Extract response value using regex as last resort
    if not results:
        # Look for patterns like "response": "..." with escaped quotes handling
        response_pattern = r'["\']response["\']\s*:\s*["\']((?:[^"\']|\\["\'])+)["\']'
        match = re.search(response_pattern, text)
        if match:
            # Unescape any escaped quotes
            response_value = match.group(1).replace('\\"', '"').replace("\\'", "'")
            results.append({"response": response_value})

    return results or None


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
        # Extract key fields for a more focused prompt
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        instruction = f"""You are an expert IMO (International Mathematical Olympiad) grader. Your task is to evaluate a student's solution to a mathematical problem and classify it into one of four categories.

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
2. Apply the grading guidelines to determine the appropriate classification
3. You MUST classify the answer into EXACTLY ONE of these four labels:
   - "correct": The solution is complete and correct with no significant errors
   - "almost": The solution is nearly complete but has minor mistakes that are not negligible
   - "partial": The solution has some correct elements but is incomplete or has significant gaps
   - "incorrect": The solution is fundamentally wrong or makes no meaningful progress

4. Provide your evaluation in the exact JSON format below

Respond in JSON format with the following schema:
<json>
{{
    "response": "One of: correct, almost, partial, incorrect"
}}
</json>

IMPORTANT RULES:
- Your response value MUST be exactly one of these four words: "correct", "almost", "partial", or "incorrect"
- Do not include any other text, explanation, or formatting in the response field
- The response must be lowercase and match exactly one of the four allowed labels
- Base your classification strictly on the grading guidelines provided"""

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
        prediction = "None"
        extraction_method = "primary"
        raw_response = msg_history[-1].get("text", "")
        
        if not raw_response or not raw_response.strip():
            self.log_fn("Warning: Empty response text from LLM")
            return "Error: Empty response from LLM", msg_history
        
        try:
            extracted = _extract_jsons(raw_response)
            if extracted and len(extracted) > 0 and "response" in extracted[-1]:
                prediction = extracted[-1]["response"]
                self.log_fn(f"Primary extraction succeeded, response type: {type(prediction).__name__}")
            else:
                # Try fallback extraction
                self.log_fn("Primary extraction failed, trying fallback...")
                extracted = _extract_json_fallback(raw_response)
                if extracted and len(extracted) > 0 and "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                    extraction_method = "fallback"
                    self.log_fn(f"Fallback extraction succeeded, response type: {type(prediction).__name__}")
                else:
                    self.log_fn("Warning: No valid JSON with 'response' key found in LLM output")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Try fallback on exception
            try:
                extracted = _extract_json_fallback(raw_response)
                if extracted and len(extracted) > 0 and "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                    extraction_method = "fallback"
                    self.log_fn(f"Fallback extraction succeeded after exception, response type: {type(prediction).__name__}")
            except Exception as fallback_e:
                self.log_fn(f"Fallback extraction also failed: {fallback_e}")

        self.log_fn(f"Extraction method used: {extraction_method}")
        
        # Convert prediction to string, handling various types
        if prediction is None:
            prediction = "None"
        elif isinstance(prediction, (list, dict)):
            prediction = json.dumps(prediction)
        else:
            prediction = str(prediction)
        
        # Normalize prediction to one of the four valid labels
        prediction = self._normalize_prediction(prediction)
            
        return prediction, msg_history

    def _normalize_prediction(self, prediction: str) -> str:
        """Normalize prediction to one of the four valid labels.
        
        Valid labels: correct, almost, partial, incorrect
        """
        if not prediction:
            return "incorrect"
        
        # Convert to lowercase and strip whitespace
        normalized = prediction.lower().strip()
        
        # Remove quotes if present
        normalized = normalized.strip('"\'')
        
        # Direct match
        valid_labels = ["correct", "almost", "partial", "incorrect"]
        if normalized in valid_labels:
            return normalized
        
        # Check if any valid label appears in the text
        for label in valid_labels:
            if label in normalized:
                return label
        
        # Check for common variations
        if "completely" in normalized or "fully" in normalized or "perfect" in normalized:
            return "correct"
        if "almost" in normalized or "nearly" in normalized or "minor" in normalized:
            return "almost"
        if "partial" in normalized or "some" in normalized or "incomplete" in normalized:
            return "partial"
        if "incorrect" in normalized or "wrong" in normalized or "error" in normalized:
            return "incorrect"
        
        # Default to incorrect if no match found
        self.log_fn(f"Warning: Could not normalize prediction '{prediction}', defaulting to 'incorrect'")
        return "incorrect"
