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
        
        instruction = f"""You are an expert IMO (International Mathematical Olympiad) grader. Your task is to evaluate a student's solution to a mathematical problem and assign a categorical grade.

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
2. Apply the grading guidelines to determine the appropriate grade category
3. Select EXACTLY ONE of these four categorical labels based on the student's performance:
   - "correct": Full solution with correct answer and valid reasoning
   - "almost": Solution is nearly complete with only minor mistakes
   - "partial": Partial progress made but significant gaps remain
   - "incorrect": Wrong approach or no valid mathematical reasoning

CRITICAL INSTRUCTIONS:
- You MUST output a CATEGORICAL LABEL, not a number
- Do NOT output numerical scores like "0.5", "1", "10", or any number
- Do NOT output any text other than one of the four labels above
- The response field must contain ONLY the word: correct, almost, partial, or incorrect

Respond in JSON format with the following schema:
<json>
{{
    "response": "correct"
}}
</json>

OR

<json>
{{
    "response": "almost"
}}
</json>

OR

<json>
{{
    "response": "partial"
}}
</json>

OR

<json>
{{
    "response": "incorrect"
}}
</json>

REMEMBER: Output ONLY ONE of these exact words in the response field: "correct", "almost", "partial", or "incorrect". No numbers, no explanations, no other text."""

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
        
        # Validate and normalize the prediction to one of the expected labels
        valid_labels = ["correct", "almost", "partial", "incorrect"]
        prediction_lower = prediction.strip().lower()
        
        # First, check if the prediction is a number (common error from LLM)
        # Convert numerical scores to appropriate labels
        try:
            # Try to parse as a number
            num_val = float(prediction_lower)
            # Map numerical scores to categorical labels
            if num_val >= 0.9:  # High scores like 0.9, 1, 10
                prediction = "correct"
            elif num_val >= 0.7:  # Good scores like 0.7, 0.8
                prediction = "almost"
            elif num_val >= 0.3:  # Medium scores like 0.3, 0.4, 0.5, 0.6
                prediction = "partial"
            else:  # Low scores like 0, 0.1, 0.2
                prediction = "incorrect"
            self.log_fn(f"Converted numerical score {num_val} to label: {prediction}")
            return prediction, msg_history
        except ValueError:
            # Not a number, continue with text matching
            pass
        
        # Check if the prediction contains one of the valid labels
        matched_label = None
        for label in valid_labels:
            if label in prediction_lower:
                matched_label = label
                break
        
        if matched_label:
            prediction = matched_label
            self.log_fn(f"Normalized prediction to: {prediction}")
        else:
            self.log_fn(f"Warning: Prediction '{prediction}' does not match any valid label")
            # Default to "incorrect" if no valid label is found
            prediction = "incorrect"
            
        return prediction, msg_history
