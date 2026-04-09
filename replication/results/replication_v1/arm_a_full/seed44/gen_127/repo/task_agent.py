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
    """
    start_idx = text.find('{')
    if start_idx == -1:
        return None
    
    brace_count = 1
    i = start_idx + 1
    while i < len(text) and brace_count > 0:
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
        i += 1
    
    if brace_count == 0:
        candidate = text[start_idx:i]
        try:
            return json.loads(candidate)
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
    Enhanced to handle more edge cases including nested quotes and escaped characters.
    """
    # Pattern to match "response": followed by a value (number, string, or boolean)
    # Handles: "response": 7, "response": "7", "response": true, etc.
    patterns = [
        # Number (integer or float) - with word boundary check
        (r'"response"\s*:\s*(-?\d+(?:\.\d+)?)(?:\s*[,}\]])', "number"),
        # String in double quotes - handle escaped quotes
        (r'"response"\s*:\s*"((?:[^"\\]|\\.)*)"', "string_double"),
        # String in single quotes
        (r"'response'\s*:\s*'([^']*)'", "string_single"),
        # Boolean or null
        (r'"response"\s*:\s*(true|false|null)(?:\s*[,}\]])', "boolean"),
        # Fallback: number without boundary check
        (r'"response"\s*:\s*(-?\d+(?:\.\d+)?)', "number_fallback"),
        # Fallback: boolean without boundary check
        (r'"response"\s*:\s*(true|false|null)', "boolean_fallback"),
    ]
    
    for pattern, pattern_type in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1)
            
            # Handle escaped characters in strings
            if pattern_type in ("string_double", "string_single"):
                # Unescape common escape sequences
                value = value.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t').replace('\\\\', '\\')
                return value
            
            # Try to convert to appropriate type for numbers/booleans
            if pattern_type in ("number", "number_fallback"):
                try:
                    # Try integer first
                    return str(int(value))
                except ValueError:
                    try:
                        # Try float
                        return str(float(value))
                    except ValueError:
                        return value
            
            if pattern_type in ("boolean", "boolean_fallback"):
                return value.lower()
            
            return value
    
    return None


def _extract_from_markdown_code_blocks(text: str) -> list[dict] | None:
    """Extract JSON from markdown code blocks with various formats.
    
    Handles cases where JSON is embedded in markdown code blocks
    with or without language specifiers.
    """
    results = []
    
    # Pattern for code blocks with json specifier
    json_block_pattern = r'```json\s*\n(.*?)\n?```'
    matches = re.findall(json_block_pattern, text, re.DOTALL | re.IGNORECASE)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            # Try to extract just the response field
            direct_value = _extract_response_value(match)
            if direct_value is not None:
                results.append({"response": direct_value})
    
    # Pattern for code blocks without specifier
    generic_block_pattern = r'```\s*\n(.*?)\n?```'
    matches = re.findall(generic_block_pattern, text, re.DOTALL)
    for match in matches:
        try:
            parsed = json.loads(match.strip())
            if isinstance(parsed, dict):
                results.append(parsed)
        except json.JSONDecodeError:
            pass
    
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
        # Extract key fields for better prompt construction
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematical grader evaluating student solutions to competition mathematics problems.

Your task is to grade a student's answer to an IMO-level mathematics problem. You must carefully analyze:
1. The problem statement
2. The official solution
3. The grading guidelines (rubric)
4. The student's submitted answer

Then provide your evaluation in the specified JSON format.

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
4. Analyze the student's answer step by step:
   - Did they understand the problem correctly?
   - Did they use the right approach?
   - Are their calculations correct?
   - Did they provide a complete proof/solution?
   - Where did they make errors, if any?
5. Assign a score based on the grading guidelines.

IMPORTANT: Your response MUST be a valid JSON object wrapped in <json> tags. Do not include any text outside the JSON tags.

Respond in the following exact format:
<json>
{{
    "response": <your numerical score or evaluation>
}}
</json>

The "response" field should contain your final grading decision (typically a number or specific evaluation string as specified in the grading guidelines).

Examples of valid responses:
- For a numeric score: <json>{{"response": 7}}</json>
- For a string evaluation: <json>{{"response": "Correct"}}</json>
- For a boolean: <json>{{"response": true}}</json>

Ensure your JSON is properly formatted with no trailing commas or syntax errors."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with multiple fallback strategies
        prediction = "None"
        raw_text = msg_history[-1]["text"]
        
        # Log the raw response for debugging
        self.log_fn(f"Raw LLM response (first 1000 chars): {raw_text[:1000]}")
        
        extraction_attempts = [
            ("_extract_jsons", lambda: _extract_jsons(raw_text)),
            ("_extract_from_markdown_code_blocks", lambda: _extract_from_markdown_code_blocks(raw_text)),
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
                        self.log_fn(f"Successfully extracted prediction using {attempt_name}: {prediction}")
                        break
            except Exception as e:
                self.log_fn(f"Extraction attempt {attempt_name} failed: {e}")
                continue
        
        # Last resort: try to extract response value directly from malformed JSON
        if prediction == "None":
            direct_value = _extract_response_value(raw_text)
            if direct_value is not None:
                prediction = direct_value
                self.log_fn(f"Extracted prediction using direct value extraction: {prediction}")
        
        # Validate numeric predictions are within reasonable bounds for IMO problems
        if isinstance(prediction, (int, float)) or (isinstance(prediction, str) and prediction.replace('.', '', 1).replace('-', '', 1).isdigit()):
            try:
                numeric_val = float(prediction) if isinstance(prediction, str) else prediction
                # IMO problems typically have scores 0-7, but allow some flexibility
                if numeric_val < 0 or numeric_val > 100:
                    self.log_fn(f"Warning: Unusual numeric prediction {prediction} - may need review")
            except (ValueError, TypeError):
                pass
        
        if prediction == "None":
            self.log_fn(f"Failed to extract prediction from response. Raw text: {raw_text[:500]}")

        # Add confidence score if prediction was successfully extracted
        confidence = self._calculate_confidence(raw_text, prediction)
        if confidence is not None:
            self.log_fn(f"Confidence score for prediction: {confidence}")
        
        return str(prediction), msg_history
    
    def _calculate_confidence(self, raw_text: str, prediction: str) -> float | None:
        """Calculate a confidence score for the prediction based on response quality.
        
        Returns a value between 0.0 and 1.0, where 1.0 indicates high confidence.
        Factors considered:
        - Presence of well-formed JSON
        - Clear reasoning in the response
        - Absence of uncertainty markers
        - Proper formatting and structure
        """
        if prediction == "None":
            return 0.0
        
        confidence = 0.7  # Start with a base confidence
        
        raw_lower = raw_text.lower()
        
        # Check for proper JSON format (well-structured response)
        if "<json>" in raw_text and "</json>" in raw_text:
            confidence += 0.15
            # Bonus if the JSON is properly formatted with newlines
            if '"response"' in raw_text:
                confidence += 0.05
        
        # Check for markdown code blocks as alternative format
        if "```json" in raw_lower or "```" in raw_text:
            confidence += 0.1
        
        # Check for reasoning indicators (shows thought process)
        reasoning_indicators = [
            "because", "therefore", "since", "as a result", 
            "consequently", "thus", "hence", "analysis", "reasoning"
        ]
        reasoning_count = sum(1 for indicator in reasoning_indicators if indicator in raw_lower)
        confidence += min(0.15, reasoning_count * 0.03)
        
        # Check for mathematical rigor indicators
        rigor_indicators = [
            "proof", "theorem", "lemma", "qed", "∎",
            "therefore", "implies", "sufficient", "necessary"
        ]
        rigor_count = sum(1 for indicator in rigor_indicators if indicator in raw_lower)
        confidence += min(0.1, rigor_count * 0.02)
        
        # Penalize uncertainty markers
        uncertainty_markers = [
            "unclear", "ambiguous", "not sure", "possibly", 
            "maybe", "perhaps", "might be", "uncertain", "unclear"
        ]
        uncertainty_count = sum(1 for marker in uncertainty_markers if marker in raw_lower)
        confidence -= min(0.25, uncertainty_count * 0.08)
        
        # Check for step-by-step analysis (good structure)
        step_markers = ["step", "first", "second", "third", "finally", "1.", "2.", "3."]
        step_count = sum(1 for marker in step_markers if marker in raw_lower)
        confidence += min(0.1, step_count * 0.02)
        
        # Penalize very short responses (may lack reasoning)
        if len(raw_text) < 100:
            confidence -= 0.15
        
        # Bonus for comprehensive responses
        if len(raw_text) > 500:
            confidence += 0.05
        
        # Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, confidence))
