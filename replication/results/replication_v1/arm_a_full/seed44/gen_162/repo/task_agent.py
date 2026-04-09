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


def _extract_json_from_code_blocks(text: str) -> list[dict] | None:
    """Extract JSON objects from markdown code blocks.
    
    Handles various code block formats like ```json, ```, etc.
    Also handles nested braces within code blocks.
    """
    results = []
    
    # Pattern for code blocks with language specifier
    code_block_patterns = [
        r'```json\s*\n?(.*?)\n?```',
        r'```\s*\n?(.*?)\n?```',
        r'`(.*?)`',
    ]
    
    for pattern in code_block_patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            match = match.strip()
            if not match:
                continue
            
            # Try direct parsing first
            try:
                results.append(json.loads(match))
                continue
            except json.JSONDecodeError:
                pass
            
            # Try brace matching for nested objects
            try:
                json_obj = _extract_json_with_brace_matching(match)
                if json_obj:
                    results.append(json_obj)
                    continue
            except Exception:
                pass
            
            # Try to find JSON object within the content
            start_idx = match.find('{')
            if start_idx != -1:
                try:
                    json_obj = _extract_json_with_brace_matching(match[start_idx:])
                    if json_obj:
                        results.append(json_obj)
                except Exception:
                    pass
    
    return results if results else None


def _extract_all_json_objects(text: str) -> list[dict] | None:
    """Extract all JSON objects from text using multiple strategies.
    
    Combines multiple extraction methods to maximize chances of finding valid JSON.
    """
    all_results = []
    
    # Try each extraction method
    extraction_methods = [
        ("xml_tags", lambda t: _extract_jsons(t)),
        ("code_blocks", lambda t: _extract_json_from_code_blocks(t)),
        ("brace_matching", lambda t: _extract_json_with_brace_matching(t)),
    ]
    
    for method_name, method_fn in extraction_methods:
        try:
            result = method_fn(text)
            if result:
                if isinstance(result, dict):
                    all_results.append((method_name, result))
                elif isinstance(result, list):
                    for item in result:
                        if isinstance(item, dict):
                            all_results.append((method_name, item))
        except Exception:
            continue
    
    # Also try to find any JSON-like objects with brace counting
    try:
        i = 0
        while i < len(text):
            if text[i] == '{':
                start = i
                brace_count = 1
                i += 1
                while i < len(text) and brace_count > 0:
                    if text[i] == '{':
                        brace_count += 1
                    elif text[i] == '}':
                        brace_count -= 1
                    i += 1
                if brace_count == 0:
                    candidate = text[start:i]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict) and obj not in [r[1] for r in all_results]:
                            all_results.append(("raw_braces", obj))
                    except json.JSONDecodeError:
                        pass
            else:
                i += 1
    except Exception:
        pass
    
    # Return unique results, preferring those with "response" key
    seen_ids = set()
    unique_results = []
    
    # Sort by whether they have "response" key (prefer those that do)
    sorted_results = sorted(all_results, key=lambda x: "response" not in x[1], reverse=False)
    
    for method_name, obj in sorted_results:
        obj_id = id(obj)
        if obj_id not in seen_ids:
            seen_ids.add(obj_id)
            unique_results.append(obj)
    
    return unique_results if unique_results else None


def _normalize_prediction(prediction: str) -> str:
    """Normalize prediction value for consistent output.
    
    Handles various formats and ensures consistent string representation.
    """
    if prediction is None:
        return "None"
    
    # Convert to string if not already
    pred_str = str(prediction).strip()
    
    # Handle boolean strings
    if pred_str.lower() in ('true', 'yes', 'correct'):
        return "Correct"
    if pred_str.lower() in ('false', 'no', 'incorrect'):
        return "Incorrect"
    
    # Try to extract just the numeric part if there's extra text
    numeric_match = re.match(r'^(-?\d+(?:\.\d+)?)', pred_str)
    if numeric_match:
        return numeric_match.group(1)
    
    return pred_str


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
    "response": <your numerical score or evaluation>,
    "reasoning": <detailed explanation of your grading decision>,
    "confidence": <confidence score from 0.0 to 1.0>,
    "error_analysis": <description of any errors found in the student's work, or "None" if correct>
}}
</json>

The "response" field should contain your final grading decision (typically a number or specific evaluation string as specified in the grading guidelines).

The "reasoning" field should contain a detailed explanation of why you assigned this score, referencing specific parts of the student's solution and the grading guidelines.

The "confidence" field should be a number between 0.0 and 1.0 indicating how confident you are in your grading decision (1.0 = very confident, 0.0 = not confident at all).

The "error_analysis" field should describe any errors or misconceptions in the student's work, or "None" if the solution is fully correct.

Examples of valid responses:
- For a numeric score: <json>{{"response": 7, "reasoning": "The student correctly identified the key insight and provided a complete proof.", "confidence": 0.95, "error_analysis": "None"}}</json>
- For a partial score: <json>{{"response": 3, "reasoning": "The student had the right approach but made a calculation error in step 3.", "confidence": 0.85, "error_analysis": "Incorrect calculation: 2+2 was computed as 5 instead of 4"}}</json>
- For a string evaluation: <json>{{"response": "Correct", "reasoning": "All steps are valid and the conclusion follows logically.", "confidence": 0.98, "error_analysis": "None"}}</json>

Ensure your JSON is properly formatted with no trailing commas or syntax errors.

FINAL CHECK: Before responding, verify that:
1. Your JSON is valid and parseable
2. All four fields (response, reasoning, confidence, error_analysis) are present
3. The "response" field contains only the final answer (number, string, or boolean)
4. The "confidence" field is a number between 0.0 and 1.0
5. There are no extra fields or comments in the JSON
6. The JSON is properly wrapped in <json>...</json> tags"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with multiple fallback strategies
        prediction = "None"
        reasoning = None
        confidence = None
        error_analysis = None
        raw_text = msg_history[-1]["text"]
        
        # Log the raw response for debugging
        self.log_fn(f"Raw LLM response (first 1000 chars): {raw_text[:1000]}")
        
        # Use comprehensive extraction method first
        extraction_attempts = [
            ("_extract_all_json_objects", lambda: _extract_all_json_objects(raw_text)),
            ("_extract_jsons", lambda: _extract_jsons(raw_text)),
            ("_extract_json_from_code_blocks", lambda: _extract_json_from_code_blocks(raw_text)),
            ("_extract_json_with_regex", lambda: _extract_json_with_regex(raw_text)),
        ]
        
        for attempt_name, attempt_fn in extraction_attempts:
            try:
                extracted = attempt_fn()
                if extracted and len(extracted) > 0:
                    # Check all extracted JSON objects for "response" key
                    # Prefer the one with "response" key, otherwise use the last one
                    target_obj = None
                    for obj in extracted:
                        if isinstance(obj, dict) and "response" in obj:
                            target_obj = obj
                            break
                    
                    # If no object with "response" found, use the last one
                    if target_obj is None:
                        target_obj = extracted[-1]
                    
                    if isinstance(target_obj, dict):
                        prediction = target_obj.get("response", "None")
                        # Extract additional fields for logging
                        reasoning = target_obj.get("reasoning")
                        confidence = target_obj.get("confidence")
                        error_analysis = target_obj.get("error_analysis")
                        self.log_fn(f"Successfully extracted prediction using {attempt_name}: {prediction}")
                        if reasoning:
                            self.log_fn(f"Reasoning: {reasoning[:200]}...")
                        if confidence is not None:
                            self.log_fn(f"Confidence: {confidence}")
                        if error_analysis and error_analysis != "None":
                            self.log_fn(f"Error analysis: {error_analysis[:200]}...")
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
        
        if prediction == "None":
            self.log_fn(f"Failed to extract prediction from response. Raw text: {raw_text[:500]}")

        # Normalize the prediction for consistent output
        normalized_prediction = _normalize_prediction(prediction)
        if normalized_prediction != str(prediction):
            self.log_fn(f"Normalized prediction from '{prediction}' to '{normalized_prediction}'")
        
        return normalized_prediction, msg_history
