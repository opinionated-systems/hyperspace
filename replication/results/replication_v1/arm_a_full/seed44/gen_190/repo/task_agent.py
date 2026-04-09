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
    """Task agent that solves IMO grading problems with enhanced accuracy."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with enhanced grading logic.

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

        # Pre-analyze the student answer for common issues
        answer_analysis = self._analyze_student_answer(student_answer, solution)

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

PRELIMINARY ANALYSIS:
{answer_analysis}

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
5. Assign a score based STRICTLY on the grading guidelines.

GRADING PRINCIPLES:
- Award partial credit for correct intermediate steps even if the final answer is wrong
- Deduct points for missing critical steps in the proof
- Consider alternative valid approaches that differ from the official solution
- Be precise: if the rubric specifies exact point values, use them exactly
- If the answer is completely blank or says "None", the score should be 0

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
        prediction = self._extract_prediction_with_fallbacks(response, msg_history)

        return str(prediction), msg_history

    def _analyze_student_answer(self, student_answer: str, official_solution: str) -> str:
        """Pre-analyze the student answer to identify common patterns and issues.
        
        This helps guide the LLM grader by providing preliminary observations.
        """
        analysis_parts = []
        
        # Check for empty or trivial answers
        if not student_answer or student_answer.strip() in ["", "None", "null", "N/A"]:
            analysis_parts.append("- The student answer appears to be empty or null")
            return "\n".join(analysis_parts)
        
        # Check answer length relative to solution
        student_lines = [line.strip() for line in student_answer.split("\n") if line.strip()]
        solution_lines = [line.strip() for line in official_solution.split("\n") if line.strip()]
        
        if len(student_lines) < 3:
            analysis_parts.append(f"- The student answer is very brief ({len(student_lines)} lines)")
        elif len(student_lines) > len(solution_lines) * 2:
            analysis_parts.append(f"- The student answer is quite lengthy ({len(student_lines)} lines vs {len(solution_lines)} in solution)")
        
        # Check for key mathematical symbols
        math_symbols = ["=", "+", "-", "*", "^", "\\frac", "\\sum", "\\int", "\\implies", "\\therefore"]
        has_math = any(sym in student_answer for sym in math_symbols)
        if not has_math:
            analysis_parts.append("- The answer contains minimal mathematical notation")
        
        # Check for common keywords indicating proof structure
        proof_keywords = ["proof", "theorem", "lemma", "claim", "therefore", "thus", "hence", "qed", "\\qed"]
        found_keywords = [kw for kw in proof_keywords if kw.lower() in student_answer.lower()]
        if found_keywords:
            analysis_parts.append(f"- Found proof-related keywords: {', '.join(found_keywords)}")
        
        # Check for final answer indicators
        final_answer_indicators = ["final answer", "answer:", "=", "\\boxed"]
        has_final_answer = any(ind in student_answer.lower() for ind in final_answer_indicators)
        if not has_final_answer:
            analysis_parts.append("- No clear final answer indicator found")
        
        if not analysis_parts:
            return "- The student answer appears to be a standard mathematical response"
        
        return "\n".join(analysis_parts)

    def _extract_prediction_with_fallbacks(self, response: str, msg_history: list[dict]) -> str:
        """Extract prediction from LLM response using multiple fallback strategies.
        
        This method implements a robust extraction pipeline that tries multiple
        strategies in order of reliability.
        """
        prediction = "None"
        raw_text = msg_history[-1]["text"] if msg_history else response
        
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
                        self.log_fn(f"Successfully extracted prediction using {attempt_name}: {prediction}")
                        return prediction
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
                        self.log_fn(f"Successfully extracted prediction using JSON sanitization: {prediction}")
                        return prediction
            except Exception as e:
                self.log_fn(f"JSON sanitization attempt failed: {e}")
        
        # Last resort: try to extract response value directly from malformed JSON
        if prediction == "None":
            direct_value = _extract_response_value(raw_text)
            if direct_value is not None:
                prediction = direct_value
                self.log_fn(f"Extracted prediction using direct value extraction: {prediction}")
                return prediction
        
        if prediction == "None":
            self.log_fn(f"Failed to extract prediction from response. Raw text: {raw_text[:500]}")

        return prediction
