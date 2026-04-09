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
    Also handles nested JSON objects within the content.
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
        
        # Try to parse the inner content as JSON
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # If direct parsing fails, try to extract JSON using brace matching
            # This handles cases where there might be nested structures
            try:
                json_obj = _extract_json_with_brace_matching(inner)
                if json_obj:
                    results.append(json_obj)
            except Exception:
                continue
    return results or None


def _extract_json_with_brace_matching(text: str) -> dict | None:
    """Extract a JSON object from text using brace counting.
    
    This handles nested JSON objects properly by tracking brace depth.
    Also handles strings that may contain braces.
    """
    start_idx = text.find('{')
    if start_idx == -1:
        return None
    
    brace_count = 1
    i = start_idx + 1
    in_string = False
    escape_next = False
    
    while i < len(text) and brace_count > 0:
        char = text[i]
        
        if escape_next:
            escape_next = False
        elif char == '\\' and in_string:
            escape_next = True
        elif char == '"' and not in_string:
            in_string = True
        elif char == '"' and in_string:
            in_string = False
        elif not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
        
        i += 1
    
    if brace_count == 0:
        candidate = text[start_idx:i]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # Try sanitizing before giving up
            try:
                sanitized = _sanitize_json_string(candidate)
                return json.loads(sanitized)
            except json.JSONDecodeError:
                pass
    return None


def _extract_json_with_regex(text: str) -> list[dict] | None:
    """Fallback JSON extraction using regex for malformed outputs.
    
    Handles nested JSON objects by using a stack-based brace matching approach
    instead of simple regex that fails on nested structures.
    """
    results = []
    # Try to find JSON objects in code blocks
    code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            continue
    
    # Try to find JSON objects with curly braces using stack-based matching
    # This handles nested objects properly
    if not results:
        i = 0
        while i < len(text):
            # Find the start of a potential JSON object
            if text[i] == '{':
                start = i
                brace_count = 1
                i += 1
                # Track braces to find the matching closing brace
                while i < len(text) and brace_count > 0:
                    if text[i] == '{':
                        brace_count += 1
                    elif text[i] == '}':
                        brace_count -= 1
                    i += 1
                # If we found a complete object with "response" key, try to parse it
                if brace_count == 0:
                    candidate = text[start:i]
                    if '"response"' in candidate:
                        try:
                            results.append(json.loads(candidate))
                        except json.JSONDecodeError:
                            continue
            else:
                i += 1
    
    return results or None


def _extract_response_value(text: str) -> str | None:
    """Last-resort extraction: try to find a response value directly.
    
    This handles cases where the JSON is malformed but we can still
    extract the value after "response": using regex patterns.
    """
    # Pattern to match "response": followed by a value (number, string, or boolean)
    # Handles: "response": 7, "response": "7", "response": true, etc.
    patterns = [
        # Number (integer or float)
        r'"response"\s*:\s*(-?\d+(?:\.\d+)?)',
        # String in double quotes
        r'"response"\s*:\s*"([^"]*)"',
        # String in single quotes
        r"'response'\s*:\s*'([^']*)'",
        # Boolean or null
        r'"response"\s*:\s*(true|false|null)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1)
            # Try to convert to appropriate type
            try:
                # Try integer first
                return str(int(value))
            except ValueError:
                try:
                    # Try float
                    return str(float(value))
                except ValueError:
                    # Return as string
                    return value
    
    return None


def _extract_reasoning_value(text: str) -> str | None:
    """Extract reasoning value from malformed JSON.
    
    This handles cases where the JSON is malformed but we can still
    extract the reasoning field using regex patterns.
    """
    # Pattern to match "reasoning": followed by a string value
    # Handles multi-line strings and escaped quotes
    pattern = r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        reasoning = match.group(1)
        # Unescape any escaped characters
        reasoning = reasoning.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"')
        return reasoning.strip()
    
    # Try single quotes
    pattern_single = r"'reasoning'\s*:\s*'((?:[^'\\]|\\.)*)'"
    match_single = re.search(pattern_single, text, re.DOTALL)
    if match_single:
        reasoning = match_single.group(1)
        reasoning = reasoning.replace('\\n', '\n').replace('\\t', '\t').replace("\\'", "'")
        return reasoning.strip()
    
    return None


def _sanitize_json_string(text: str) -> str:
    """Sanitize a JSON string by fixing common formatting issues.
    
    This helps recover from malformed JSON that LLMs sometimes produce.
    """
    # Remove trailing commas before closing braces/brackets
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    
    # Fix single quotes to double quotes for JSON compatibility
    # Only replace quotes that appear to be delimiting keys/values
    text = re.sub(r"(?<=[{,\s])'([^']+)'(?=\s*:)", r'"\1"', text)
    
    # Remove comments (both // and /* */ styles)
    text = re.sub(r'//[^\n]*', '', text)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    
    return text.strip()


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self._last_reasoning: str | None = None
    
    def get_last_reasoning(self) -> str | None:
        """Get the reasoning from the last grading decision.
        
        Returns:
            The reasoning string if available, None otherwise.
        """
        return self._last_reasoning

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract key fields for better prompt construction
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematical grader evaluating student solutions to competition mathematics problems.

Your task is to grade a student's answer to an IMO-level mathematics problem. You must carefully analyze all provided materials and provide a well-reasoned evaluation.

PROBLEM DOMAIN: {domain}

PROBLEM STATEMENT:
```
{problem}
```

OFFICIAL SOLUTION:
```
{solution}
```

GRADING GUIDELINES (RUBRIC):
```
{grading_guidelines}
```

STUDENT'S ANSWER:
```
{student_answer}
```

INSTRUCTIONS:
1. Read the problem carefully and understand what is being asked.
2. Study the official solution to understand the correct approach.
3. Review the grading guidelines to understand how points are awarded.
4. Analyze the student's answer step by step using chain-of-thought reasoning:
   - Did they understand the problem correctly?
   - Did they use the right approach?
   - Are their calculations correct?
   - Did they provide a complete proof/solution?
   - Where did they make errors, if any?
   - What partial credit should be awarded based on the rubric?
5. Assign a final score based on the grading guidelines.

IMPORTANT: Your response MUST be a valid JSON object wrapped in <json> tags. Do not include any text outside the JSON tags.

Respond in the following exact format:
<json>
{{
    "response": <your numerical score or evaluation>,
    "reasoning": "<your detailed chain-of-thought analysis explaining how you arrived at this score>"
}}
</json>

The "response" field should contain your final grading decision (typically a number or specific evaluation string as specified in the grading guidelines).
The "reasoning" field should contain your step-by-step analysis that led to this score.

Examples of valid responses:
- For a numeric score: <json>{{"response": 7, "reasoning": "The student correctly identified the key insight and provided a complete proof."}}</json>
- For a string evaluation: <json>{{"response": "Correct", "reasoning": "The solution matches the official solution exactly."}}</json>
- For partial credit: <json>{{"response": 3, "reasoning": "The student had the right approach but made an error in the final calculation, earning partial credit per the rubric."}}</json>

Ensure your JSON is properly formatted with no trailing commas or syntax errors."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with multiple fallback strategies
        prediction = "None"
        reasoning = None
        raw_text = msg_history[-1]["text"]
        
        # Log the raw response for debugging
        self.log_fn(f"Raw LLM response (first 1000 chars): {raw_text[:1000]}")
        
        extraction_attempts = [
            ("_extract_jsons", lambda: _extract_jsons(raw_text)),
            ("_extract_json_with_regex", lambda: _extract_json_with_regex(raw_text)),
        ]
        
        for attempt_name, attempt_fn in extraction_attempts:
            try:
                extracted = attempt_fn()
                if extracted and len(extracted) > 0:
                    # Check the last extracted JSON object for "response" key
                    last_obj = extracted[-1]
                    if isinstance(last_obj, dict) and "response" in last_obj:
                        prediction = last_obj["response"]
                        reasoning = last_obj.get("reasoning", None)
                        self.log_fn(f"Successfully extracted prediction using {attempt_name}: {prediction}")
                        if reasoning:
                            self.log_fn(f"Grading reasoning: {reasoning[:200]}...")
                        break
            except Exception as e:
                self.log_fn(f"Extraction attempt {attempt_name} failed: {e}")
                continue
        
        # Try sanitizing and re-parsing JSON before giving up
        if prediction == "None":
            try:
                sanitized = _sanitize_json_string(raw_text)
                # Try to find JSON objects in the sanitized text
                sanitized_extracted = _extract_jsons(sanitized)
                if sanitized_extracted and len(sanitized_extracted) > 0:
                    last_obj = sanitized_extracted[-1]
                    if isinstance(last_obj, dict) and "response" in last_obj:
                        prediction = last_obj["response"]
                        reasoning = last_obj.get("reasoning", None)
                        self.log_fn(f"Successfully extracted prediction using JSON sanitization: {prediction}")
            except Exception as e:
                self.log_fn(f"JSON sanitization attempt failed: {e}")
        
        # Last resort: try to extract response value directly from malformed JSON
        if prediction == "None":
            direct_value = _extract_response_value(raw_text)
            if direct_value is not None:
                prediction = direct_value
                self.log_fn(f"Extracted prediction using direct value extraction: {prediction}")
                # Also try to extract reasoning in this case
                if reasoning is None:
                    reasoning = _extract_reasoning_value(raw_text)
                    if reasoning:
                        self.log_fn(f"Extracted reasoning using direct extraction: {reasoning[:200]}...")
        
        if prediction == "None":
            self.log_fn(f"Failed to extract prediction from response. Raw text: {raw_text[:500]}")
        
        # Store reasoning in the message history for potential future use
        if reasoning and msg_history:
            msg_history[-1]["reasoning"] = reasoning
        
        # Store reasoning in instance variable for retrieval
        self._last_reasoning = reasoning

        return str(prediction), msg_history
