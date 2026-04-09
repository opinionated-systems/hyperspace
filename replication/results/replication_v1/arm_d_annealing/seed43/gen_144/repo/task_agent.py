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
    """Extract JSON objects from <json>...</json> blocks with enhanced recovery.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Implements a multi-layer recovery strategy for malformed JSON:
    1. Direct JSON parsing
    2. Common fix application (trailing commas, quote fixes)
    3. Outermost object extraction for nested structures
    4. Key-value pair extraction as final fallback
    Also searches markdown code blocks as secondary source.
    """
    results = []
    search_from = 0
    
    # Primary: Extract from <json> tags
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        # Attempt multi-layer JSON recovery
        parsed = _parse_json_with_recovery(inner)
        if parsed is not None:
            results.append(parsed)
    
    # Secondary: Extract from markdown code blocks
    if not results:
        # Try multiple code block patterns
        patterns = [
            r'```(?:json)?\s*\n?(.*?)\n?```',  # Standard markdown
            r'`\s*(\{.*?\})\s*`',  # Inline code with JSON
            r'```\s*(\{.*?\})\s*```',  # Code block with JSON object
        ]
        for pattern in patterns:
            code_block_pattern = re.search(pattern, text, re.DOTALL)
            if code_block_pattern:
                json_content = code_block_pattern.group(1).strip()
                parsed = _parse_json_with_recovery(json_content)
                if parsed is not None:
                    results.append(parsed)
                    break
    
    # Tertiary: Look for JSON-like structures without tags
    if not results:
        # Find JSON objects that contain expected keys
        json_candidates = re.findall(r'\{[^{}]*"(?:response|reasoning|grade|score|answer)"[^{}]*\}', text, re.DOTALL)
        for candidate in json_candidates:
            parsed = _parse_json_with_recovery(candidate)
            if parsed is not None:
                results.append(parsed)
    
    return results or None


def _parse_json_with_recovery(text: str) -> dict | None:
    """Attempt to parse JSON with multiple recovery strategies.
    
    Returns the parsed dict or None if all strategies fail.
    """
    # Strategy 1: Direct parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Apply common fixes
    try:
        fixed = _fix_json_string(text)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Extract outermost JSON object
    try:
        obj = _extract_outermost_json(text)
        if obj:
            return obj
    except Exception:
        pass
    
    # Strategy 4: Extract key-value pairs manually
    try:
        obj = _extract_key_value_pairs(text)
        if obj and ("response" in obj or "reasoning" in obj):
            return obj
    except Exception:
        pass
    
    return None


def _extract_key_value_pairs(text: str) -> dict | None:
    """Extract key-value pairs from malformed JSON as a last resort.
    
    Looks for "key": "value" patterns to reconstruct a minimal valid JSON object.
    Handles string values, booleans, null, numeric values, and escaped quotes.
    Also handles single-quoted strings and unquoted values.
    """
    result = {}
    
    # Pattern 1: Standard double-quoted "key": "value" or "key": value
    pattern1 = r'"(\w+)"\s*:\s*(?:"([^"]*)"|(\d+|true|false|null))'
    matches = re.findall(pattern1, text)
    for match in matches:
        key = match[0]
        if match[1]:  # String value
            result[key] = match[1]
        elif match[2]:  # Non-string value
            val = match[2].lower()
            if val == "true":
                result[key] = True
            elif val == "false":
                result[key] = False
            elif val == "null":
                result[key] = None
            elif val.isdigit():
                result[key] = int(val)
            else:
                result[key] = val
    
    # Pattern 2: Single-quoted 'key': 'value' or 'key': value
    pattern2 = r"'(\w+)'\s*:\s*(?:'([^']*)'|(\d+|true|false|null))"
    matches2 = re.findall(pattern2, text)
    for match in matches2:
        key = match[0]
        if key not in result:  # Don't overwrite existing values
            if match[1]:  # String value
                result[key] = match[1]
            elif match[2]:  # Non-string value
                val = match[2].lower()
                if val == "true":
                    result[key] = True
                elif val == "false":
                    result[key] = False
                elif val == "null":
                    result[key] = None
                elif val.isdigit():
                    result[key] = int(val)
                else:
                    result[key] = val
    
    # Pattern 3: Look for response/grade in plain text if no JSON found
    if not result:
        # Look for patterns like "Grade: Correct" or "Response: Partial"
        text_patterns = [
            (r'(?:grade|response|score|answer)\s*[:=]\s*["\']?([^"\'\n,]+)["\']?', 'response'),
            (r'(?:reasoning|explanation|analysis)\s*[:=]\s*["\']?([^"\'\n]+)["\']?', 'reasoning'),
        ]
        for pattern, key in text_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result[key] = match.group(1).strip()
    
    return result if result else None


def _fix_json_string(text: str) -> str:
    """Apply common JSON fixes: remove trailing commas, normalize quotes, and handle escape sequences."""
    # Remove trailing commas before closing braces/brackets
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    # Fix single quotes to double quotes (basic) - but be careful with apostrophes in text
    # Only replace single quotes that appear to be JSON string delimiters
    text = re.sub(r"(?<=[\s,\[\{:])'([^']*?)'(?=[\s,\}\]:])", r'"\1"', text)
    # Remove comments (both // and /* */)
    text = re.sub(r'//[^\n]*', '', text)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    # Fix unescaped newlines in strings (common LLM issue)
    text = re.sub(r'(?<=")([^"]*)\n([^"]*)(?=")', lambda m: m.group(0).replace('\n', '\\n'), text)
    # Remove control characters
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
    return text.strip()


def _extract_outermost_json(text: str) -> dict | None:
    """Extract the outermost JSON object from text, handling nesting.
    
    Also handles string escaping to avoid counting braces inside strings.
    """
    brace_count = 0
    start_idx = -1
    in_string = False
    escape_next = False
    
    for i, char in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        
        if char == '\\' and in_string:
            escape_next = True
            continue
        
        if char == '"' and not in_string:
            in_string = True
        elif char == '"' and in_string:
            in_string = False
        elif not in_string:
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    try:
                        return json.loads(text[start_idx:i+1])
                    except json.JSONDecodeError:
                        # Try with fixes
                        try:
                            fixed = _fix_json_string(text[start_idx:i+1])
                            return json.loads(fixed)
                        except json.JSONDecodeError:
                            continue
    
    return None


def _extract_json_with_regex(text: str) -> list[dict] | None:
    """Fallback JSON extraction using regex for malformed responses.
    
    Uses multiple strategies to extract JSON from various formats.
    """
    results = []
    
    # Strategy 1: Try to find JSON objects in code blocks with various formats
    code_block_patterns = [
        r'```(?:json)?\s*(\{.*?\})\s*```',  # Standard markdown code blocks
        r'```\s*(\{[\s\S]*?\})\s*```',  # Code blocks with newlines
        r'`\s*(\{[^`]*\})\s*`',  # Inline code
    ]
    
    for pattern in code_block_patterns:
        json_pattern = re.search(pattern, text, re.DOTALL)
        if json_pattern:
            try:
                results.append(json.loads(json_pattern.group(1)))
            except json.JSONDecodeError:
                # Try with fixes
                try:
                    fixed = _fix_json_string(json_pattern.group(1))
                    results.append(json.loads(fixed))
                except json.JSONDecodeError:
                    # Try extracting key-value pairs as last resort
                    obj = _extract_key_value_pairs(json_pattern.group(1))
                    if obj:
                        results.append(obj)
    
    # Strategy 2: Try to find any JSON-like structure with expected keys
    if not results:
        # Look for patterns with response/reasoning/grade keys
        key_patterns = [
            r'"response"\s*:\s*"([^"]*)"',
            r'"grade"\s*:\s*"([^"]*)"',
            r'"score"\s*:\s*"([^"]*)"',
            r'"answer"\s*:\s*"([^"]*)"',
        ]
        for pattern in key_patterns:
            match = re.search(pattern, text)
            if match:
                key = pattern.split('"')[1]  # Extract the key name
                results.append({key: match.group(1)})
                break
    
    # Strategy 3: Try to find JSON without code blocks (look for { ... } patterns)
    if not results:
        # Find all potential JSON objects with expected keys
        potential_jsons = re.findall(
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*"(?:response|reasoning|grade|score|answer)"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', 
            text, re.DOTALL
        )
        for pj in potential_jsons:
            try:
                obj = json.loads(pj)
                if "response" in obj or "reasoning" in obj or "grade" in obj:
                    results.append(obj)
            except json.JSONDecodeError:
                try:
                    fixed = _fix_json_string(pj)
                    obj = json.loads(fixed)
                    if "response" in obj or "reasoning" in obj or "grade" in obj:
                        results.append(obj)
                except json.JSONDecodeError:
                    # Try extracting key-value pairs
                    obj = _extract_key_value_pairs(pj)
                    if obj and ("response" in obj or "reasoning" in obj or "grade" in obj):
                        results.append(obj)
                    continue
    
    # Strategy 4: Look for plain text patterns as last resort
    if not results:
        # Look for "The answer is X" or "Grade: X" patterns
        plain_text_patterns = [
            (r'(?:the answer is|grade is|score is|result is)[:\s]+([\w\s]+?)(?:\.|$)', 'response'),
            (r'(?:graded as|marked as|assessed as)[:\s]+([\w\s]+?)(?:\.|$)', 'response'),
        ]
        for pattern, key in plain_text_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                if value:
                    results.append({key: value})
                    break
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3
        self._extraction_stats = {"success": 0, "fallback": 0, "failed": 0}

    def _build_grading_prompt(self, inputs: dict) -> str:
        """Build a structured prompt for the grading task with chain-of-thought."""
        domain = inputs.get("domain", "unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
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

IMPORTANT: Ensure your JSON is valid and properly formatted. The 'response' field should contain only the grade, not the reasoning."""

    def _extract_prediction(self, text: str) -> tuple[str, str]:
        """Extract prediction and reasoning from response text with enhanced validation.
        
        Returns:
            (prediction, reasoning) tuple
        """
        prediction = "None"
        reasoning = ""
        
        # Try primary extraction method
        extracted = _extract_jsons(text)
        if extracted is not None:
            self._extraction_stats["success"] += 1
            self.log_fn("JSON extraction: primary method succeeded")
        else:
            # Fallback to regex extraction
            extracted = _extract_json_with_regex(text)
            if extracted is not None:
                self._extraction_stats["fallback"] += 1
                self.log_fn("JSON extraction: fallback method succeeded")
            else:
                self._extraction_stats["failed"] += 1
                self.log_fn("JSON extraction: all methods failed")
        
        if extracted:
            # Find the best JSON object - prefer one with both response and reasoning
            best_json = None
            for json_obj in extracted:
                if "response" in json_obj:
                    if best_json is None:
                        best_json = json_obj
                    # Prefer objects that also have reasoning
                    if "reasoning" in json_obj and "reasoning" not in best_json:
                        best_json = json_obj
            
            # Fall back to last json if no better candidate found
            if best_json is None:
                best_json = extracted[-1]
            
            if "response" in best_json:
                prediction = str(best_json["response"]).strip()
                # Validate prediction is not empty or just whitespace
                if not prediction:
                    prediction = "None"
                    self.log_fn("Warning: extracted response field was empty")
                # Clean up common formatting issues
                prediction = prediction.strip('"\'')
            if "reasoning" in best_json:
                reasoning = str(best_json["reasoning"]).strip()
        
        # Enhanced validation: normalize and validate grade format
        prediction = self._normalize_grade(prediction)
        
        return prediction, reasoning
    
    def _normalize_grade(self, prediction: str) -> str:
        """Normalize and validate grade format.
        
        Handles various grade formats and normalizes them to standard values.
        """
        if not prediction or prediction == "None":
            return "None"
        
        pred_lower = prediction.lower().strip()
        
        # Map common variations to standard values
        grade_mappings = {
            # Correct variations
            "correct": "Correct",
            "right": "Correct",
            "true": "Correct",
            "yes": "Correct",
            "pass": "Correct",
            "full": "Correct",
            "full marks": "Correct",
            # Partial variations
            "partial": "Partial",
            "partially correct": "Partial",
            "partial credit": "Partial",
            "half": "Partial",
            "some": "Partial",
            "incomplete": "Partial",
            # Incorrect variations
            "incorrect": "Incorrect",
            "wrong": "Incorrect",
            "false": "Incorrect",
            "no": "Incorrect",
            "fail": "Incorrect",
            "none": "None",
        }
        
        # Check for exact matches first
        if pred_lower in grade_mappings:
            return grade_mappings[pred_lower]
        
        # Check for numeric grades (0-7 for IMO-style scoring)
        if pred_lower.isdigit():
            num = int(pred_lower)
            if 0 <= num <= 7:
                return prediction  # Keep numeric grades as-is
        
        # Check for partial match in the text
        for key, value in grade_mappings.items():
            if key in pred_lower:
                self.log_fn(f"Grade normalized: '{prediction}' -> '{value}'")
                return value
        
        # If no match found, log warning but return original
        self.log_fn(f"Warning: unusual grade format detected: '{prediction}'")
        return prediction

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

Your response was:
---
{last_text[:500]}
---

You MUST respond with ONLY valid JSON wrapped in <json>...</json> tags. Do not include any text before or after the JSON tags.

Correct format:
<json>
{{
    "reasoning": "Your detailed analysis here...",
    "response": "Correct"
}}
</json>

IMPORTANT: 
- The JSON must be valid (no trailing commas, proper quotes)
- Both 'reasoning' and 'response' fields are required
- The 'response' field should contain only the grade

Now try again with the original task:

{self._build_grading_prompt(inputs)}"""
                    
            except Exception as e:
                self.log_fn(f"Error in attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    prediction = f"Error: {e}"
        
        # Log extraction statistics periodically
        total = sum(self._extraction_stats.values())
        if total > 0 and total % 10 == 0:
            self.log_fn(f"Extraction stats after {total} attempts: {self._extraction_stats}")
        
        return str(prediction), msg_history
