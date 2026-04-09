"""
Task agent: solves a given task with chain-of-thought reasoning.

Reimplemented from facebookresearch/HyperAgents task_agent.py.
Same interface, same JSON output format, same extraction logic.

This is the INITIAL task agent. The meta agent modifies this file
during self-improvement. The evaluation harness loads whatever
task_agent.py exists at the agent's repo path.

Enhancements:
- Better error handling and recovery
- Confidence scoring for predictions
- Structured output validation
- Retry logic for failed extractions
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from agent.llm_client import get_response_from_llm, EVAL_MODEL

# Compile regex patterns once for efficiency
_GRADE_PATTERN = re.compile(r'(?:grade|score|result|answer)[\s:]+([^\n]+)', re.IGNORECASE)
_JSON_BLOCK_PATTERN = re.compile(r'```(?:json)?\s*\n?(.*?)\n?```', re.DOTALL)

logger = logging.getLogger(__name__)

# Standard grade categories for normalization
GRADE_CATEGORIES = {
    "correct": ["correct", "right", "true", "yes", "pass", "success", "1", "100%", "accurate", "valid"],
    "incorrect": ["incorrect", "wrong", "false", "no", "fail", "failure", "0", "0%", "invalid", "error"],
    "partial": ["partial", "partially correct", "partially", "incomplete", "half", "mostly correct", "some"],
}


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks with robust parsing.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also attempts to extract JSON from markdown code blocks as fallback.
    Includes additional heuristics for malformed JSON and nested structures.
    
    Enhanced version with better handling of nested structures and edge cases.
    """
    results = []
    search_from = 0
    extraction_attempts = 0
    
    # First pass: extract from <json> tags
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            logger.warning("Found <json> tag without closing </json> tag")
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        extraction_attempts += 1
        
        # Try direct JSON parsing first
        try:
            parsed = json.loads(inner)
            if isinstance(parsed, dict):
                results.append(parsed)
                logger.debug(f"Successfully parsed JSON from <json> tag (attempt {extraction_attempts})")
            continue
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from within the text if it's wrapped in other content
        try:
            # Look for JSON-like content with braces (handle nested braces)
            brace_start = inner.find("{")
            brace_end = inner.rfind("}")
            if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
                parsed = json.loads(inner[brace_start:brace_end + 1])
                if isinstance(parsed, dict):
                    results.append(parsed)
                    logger.debug(f"Successfully parsed JSON from braces (attempt {extraction_attempts})")
                    continue
        except json.JSONDecodeError:
            pass
        
        # Try to fix common JSON formatting issues
        try:
            fixed = _fix_json_string(inner)
            parsed = json.loads(fixed)
            if isinstance(parsed, dict):
                results.append(parsed)
                logger.debug(f"Successfully parsed JSON after fixing (attempt {extraction_attempts})")
                continue
        except json.JSONDecodeError as e:
            logger.debug(f"Failed to parse JSON from <json> block: {e}")
            pass
    
    # Second pass: try to find JSON in markdown code blocks
    if not results:
        json_blocks = _JSON_BLOCK_PATTERN.findall(text)
        for block in json_blocks:
            block = block.strip()
            try:
                parsed = json.loads(block)
                if isinstance(parsed, dict):
                    results.append(parsed)
                    logger.debug("Successfully parsed JSON from markdown code block")
            except json.JSONDecodeError:
                # Try fixing common issues in markdown blocks too
                try:
                    fixed = _fix_json_string(block)
                    parsed = json.loads(fixed)
                    if isinstance(parsed, dict):
                        results.append(parsed)
                        logger.debug("Successfully parsed JSON from markdown block after fixing")
                except json.JSONDecodeError:
                    continue
    
    # Third pass: try to find any JSON-like structure in the text
    if not results:
        # Look for patterns like {"key": "value"} or {"key": value}
        json_like_pattern = re.compile(r'\{[^{}]*"[^"]+"[^{}]*\}')
        matches = json_like_pattern.findall(text)
        for match in matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, dict):
                    results.append(parsed)
                    logger.debug("Successfully parsed JSON from pattern match")
            except json.JSONDecodeError:
                continue
    
    # Fourth pass: try to find JSON with nested braces using stack-based matching
    if not results:
        try:
            # Find all potential JSON start positions
            for i, char in enumerate(text):
                if char == '{':
                    # Try to find matching closing brace using stack
                    stack = 1
                    for j in range(i + 1, len(text)):
                        if text[j] == '{':
                            stack += 1
                        elif text[j] == '}':
                            stack -= 1
                            if stack == 0:
                                # Found a complete JSON object
                                candidate = text[i:j+1]
                                try:
                                    parsed = json.loads(candidate)
                                    if isinstance(parsed, dict):
                                        results.append(parsed)
                                        logger.debug("Successfully parsed JSON from stack-based matching")
                                except json.JSONDecodeError:
                                    # Try fixing
                                    try:
                                        fixed = _fix_json_string(candidate)
                                        parsed = json.loads(fixed)
                                        if isinstance(parsed, dict):
                                            results.append(parsed)
                                            logger.debug("Successfully parsed JSON from stack-based matching after fixing")
                                    except json.JSONDecodeError:
                                        pass
                                break
        except Exception:
            pass
    
    # Fifth pass: try to extract from malformed responses with missing braces
    if not results:
        # Look for key-value pairs that might indicate a JSON structure
        # Pattern: "key": "value" or "key": value
        kv_pattern = re.compile(r'"(\w+)":\s*"([^"]*)"')
        kv_matches = kv_pattern.findall(text)
        if kv_matches:
            # Try to reconstruct a JSON object from key-value pairs
            reconstructed = {}
            for key, value in kv_matches:
                reconstructed[key] = value
            if reconstructed:
                # Check if we also have numeric values
                num_pattern = re.compile(r'"(\w+)":\s*(\d+(?:\.\d+)?)')
                num_matches = num_pattern.findall(text)
                for key, value in num_matches:
                    reconstructed[key] = float(value) if '.' in value else int(value)
                results.append(reconstructed)
                logger.debug("Successfully reconstructed JSON from key-value pairs")
    
    return results or None


def _normalize_grade(grade: str) -> tuple[str, float]:
    """Normalize a grade string to a standard category with confidence.
    
    Args:
        grade: Raw grade string from LLM response
        
    Returns:
        Tuple of (normalized_grade, confidence_score)
    """
    if not grade:
        return "Unknown", 0.0
    
    grade_lower = grade.lower().strip()
    
    # Check for exact matches first (highest confidence)
    for category, variations in GRADE_CATEGORIES.items():
        for var in variations:
            if grade_lower == var:
                return category.replace("partial", "Partially Correct").title(), 1.0
    
    # Check for partial matches (medium confidence)
    for category, variations in GRADE_CATEGORIES.items():
        for var in variations:
            if var in grade_lower:
                return category.replace("partial", "Partially Correct").title(), 0.8
    
    # Check for numeric scores (e.g., "7/10", "85%")
    numeric_match = re.search(r'(\d+(?:\.\d+)?)\s*/\s*(\d+)', grade)
    if numeric_match:
        numerator = float(numeric_match.group(1))
        denominator = float(numeric_match.group(2))
        if denominator > 0:
            ratio = numerator / denominator
            if ratio >= 0.9:
                return "Correct", 0.9
            elif ratio >= 0.5:
                return "Partially Correct", 0.9
            else:
                return "Incorrect", 0.9
    
    # Check for percentage
    percent_match = re.search(r'(\d+(?:\.\d+)?)\s*%', grade)
    if percent_match:
        percent = float(percent_match.group(1))
        if percent >= 90:
            return "Correct", 0.9
        elif percent >= 50:
            return "Partially Correct", 0.9
        else:
            return "Incorrect", 0.9
    
    # Return original with low confidence if no normalization applied
    return grade, 0.5


def _fix_json_string(text: str) -> str:
    """Apply common JSON fixes to a string.
    
    Fixes:
    - Remove trailing commas before closing braces/brackets
    - Convert single quotes to double quotes (common LLM mistake)
    - Remove comments
    - Handle unescaped newlines in strings
    """
    # Remove trailing commas before closing braces/brackets
    fixed = re.sub(r',(\s*[}\]])', r'\1', text)
    
    # Fix single quotes to double quotes (common LLM mistake)
    # Be careful not to replace escaped single quotes
    fixed = re.sub(r"(?<!\\)'", '"', fixed)
    
    # Remove C-style comments
    fixed = re.sub(r'/\*.*?\*/', '', fixed, flags=re.DOTALL)
    fixed = re.sub(r'//.*?$', '', fixed, flags=re.MULTILINE)
    
    # Replace unescaped newlines in strings with \n
    # This is a simplified approach - replace actual newlines between quotes
    def replace_newlines_in_string(match):
        content = match.group(1)
        # Replace newlines with escaped newlines
        content = content.replace('\n', '\\n').replace('\r', '')
        return '"' + content + '"'
    
    fixed = re.sub(r'"([^"]*)"', replace_newlines_in_string, fixed, flags=re.DOTALL)
    
    return fixed


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning.
    
    Enhanced with:
    - Better error handling and recovery
    - Confidence scoring for predictions
    - Structured output validation
    - Retry logic for failed extractions
    """

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "", max_retries: int = 2) -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = max_retries
        self._extraction_stats = {"success": 0, "fallback": 0, "failure": 0}

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with chain-of-thought reasoning.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract fields for better prompting
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert {domain} grader evaluating student solutions with meticulous attention to detail.

Your task is to grade a student's answer by comparing it against the official solution and following the grading guidelines precisely.

## Problem Statement:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Instructions:
1. **Analyze the student's answer step by step**: Break down their solution into logical steps and evaluate each one.
2. **Identify key concepts**: What mathematical/theoretical concepts should the student have used?
3. **Compare to official solution**: How does the student's approach differ? Are the differences acceptable alternatives or errors?
4. **Check for partial credit**: Even if the final answer is wrong, did they demonstrate understanding of key concepts?
5. **Consider common mistakes**: Are there typical errors that should receive partial credit per the guidelines?
6. **Provide detailed reasoning**: Explain your thought process clearly before giving the final grade.
7. **Respond in strict JSON format** wrapped in <json>...</json> tags with this exact schema:

<json>
{{
    "reasoning": "Your detailed step-by-step analysis of the student's answer, explaining what was correct, what was incorrect, and why...",
    "response": "The final grade/assessment (e.g., 'Correct', 'Partially Correct', 'Incorrect', or a numeric score like '7/10')"
}}
</json>

## Example Responses:

**Example 1 - Correct:**
<json>
{{
    "reasoning": "The student correctly applied the Pythagorean theorem, showed all work clearly, and arrived at the correct answer of 5. Their reasoning was sound and complete.",
    "response": "Correct"
}}
</json>

**Example 2 - Partially Correct:**
<json>
{{
    "reasoning": "The student correctly identified the need to use the quadratic formula and set up the equation properly. However, they made a sign error when calculating the discriminant (used b² - 4ac = 9 - 48 = -39 instead of 9 + 48 = 57). This led to an incorrect final answer, but the method was fundamentally correct.",
    "response": "Partially Correct"
}}
</json>

**Example 3 - Incorrect:**
<json>
{{
    "reasoning": "The student misunderstood the problem entirely, applying the wrong formula (used area formula instead of volume formula). The approach was fundamentally incorrect and the final answer is wrong.",
    "response": "Incorrect"
}}
</json>

## Critical Requirements:
- Your response MUST be valid JSON inside <json> tags
- The "reasoning" field must contain your complete analysis
- The "response" field must contain only the final grade (no extra text)
- Be fair and consistent with the grading guidelines
- Consider partial credit for correct methodology even with calculation errors

Think carefully and provide a fair, well-reasoned assessment."""

        # Try with retries for better robustness
        prediction = "None"
        reasoning = ""
        confidence = 0.0
        msg_history = []
        
        for attempt in range(self.max_retries + 1):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                
                # Extract prediction from JSON
                if msg_history and len(msg_history) > 0:
                    response_text = msg_history[-1].get("text", "")
                    extracted = _extract_jsons(response_text)
                    
                    if extracted:
                        last_json = extracted[-1]
                        self._extraction_stats["success"] += 1
                    
                        # Try multiple possible keys for the response (ordered by priority)
                        response_keys = ["response", "grade", "result", "answer", "assessment", "evaluation", "score", "verdict"]
                        raw_prediction = None
                        for key in response_keys:
                            if key in last_json:
                                value = last_json[key]
                                # Handle different value types
                                if isinstance(value, (str, int, float, bool)):
                                    raw_prediction = str(value)
                                elif isinstance(value, list) and len(value) > 0:
                                    raw_prediction = str(value[0])
                                elif isinstance(value, dict):
                                    raw_prediction = json.dumps(value)
                                else:
                                    raw_prediction = str(value)
                                break
                        
                        # Normalize the grade and get confidence
                        if raw_prediction:
                            prediction, confidence = _normalize_grade(raw_prediction)
                        
                        # Log reasoning if available
                        reasoning_keys = ["reasoning", "analysis", "explanation", "thought", "thinking", "rationale"]
                        for key in reasoning_keys:
                            if key in last_json:
                                reasoning_val = last_json[key]
                                if isinstance(reasoning_val, str):
                                    reasoning = reasoning_val
                                else:
                                    reasoning = str(reasoning_val)
                                self.log_fn(f"Reasoning ({key}): {reasoning[:200]}...")
                                break
                        
                        # Check for explicit confidence score if available
                        if "confidence" in last_json:
                            try:
                                explicit_confidence = float(last_json["confidence"])
                                # Blend explicit confidence with our calculated confidence
                                confidence = (confidence + explicit_confidence) / 2
                            except (ValueError, TypeError):
                                pass
                        
                        # Check for additional metadata
                        if "metadata" in last_json and isinstance(last_json["metadata"], dict):
                            self.log_fn(f"Metadata: {last_json['metadata']}")
                        
                        # Success! Break out of retry loop
                        break
                        
                    else:
                        # Fallback: try to extract any meaningful text from the response
                        self._extraction_stats["fallback"] += 1
                        grade_match = _GRADE_PATTERN.search(response_text)
                        if grade_match:
                            raw_prediction = grade_match.group(1).strip()
                            prediction, _ = _normalize_grade(raw_prediction)
                            confidence = 0.5  # Lower confidence for pattern-matched extraction
                            self.log_fn(f"Extracted grade via pattern matching: {prediction}")
                            break
                        else:
                            # Last resort: use the raw response (truncated)
                            if attempt < self.max_retries:
                                self.log_fn(f"No JSON found, retrying (attempt {attempt + 1}/{self.max_retries + 1})...")
                                continue
                            prediction = response_text[:500].strip()
                            confidence = 0.3
                            self.log_fn(f"Using raw response (no JSON found): {prediction[:100]}...")
                            break
                else:
                    if attempt < self.max_retries:
                        self.log_fn(f"Empty message history, retrying (attempt {attempt + 1}/{self.max_retries + 1})...")
                        continue
                    self.log_fn("Warning: Empty message history after all retries")
                    prediction = "Error: No response"
                    confidence = 0.0
                    break
                    
            except Exception as e:
                self.log_fn(f"Error on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries:
                    continue
                self._extraction_stats["failure"] += 1
                prediction = f"Error: {str(e)[:100]}"
                confidence = 0.0

        # Clean up the prediction
        prediction = prediction.strip()
        
        # Log final prediction with confidence and stats
        self.log_fn(f"Final prediction (confidence={confidence:.2f}): {str(prediction)[:100]}")
        self.log_fn(f"Extraction stats: {self._extraction_stats}")

        return str(prediction), msg_history

    def _validate_and_normalize_grade(self, grade: str) -> str:
        """Validate and normalize a grade string to standard format.
        
        Args:
            grade: Raw grade string from LLM response
            
        Returns:
            Normalized grade string
        """
        normalized, _ = _normalize_grade(grade)
        return normalized

    def extract_structured_response(
        self, 
        response_text: str, 
        required_keys: list[str] | None = None,
        default_values: dict | None = None
    ) -> dict:
        """Extract a structured response with validation.
        
        Args:
            response_text: Raw text from LLM response
            required_keys: List of keys that must be present
            default_values: Default values for missing keys
            
        Returns:
            Dictionary with extracted and validated values
        """
        if required_keys is None:
            required_keys = ["response", "reasoning"]
        
        if default_values is None:
            default_values = {
                "response": "Unknown",
                "reasoning": "No reasoning provided",
                "confidence": 0.0
            }
        
        result = dict(default_values)
        
        try:
            extracted = _extract_jsons(response_text)
            
            if extracted and len(extracted) > 0:
                last_json = extracted[-1]
                
                # Extract all available keys
                for key in last_json:
                    if key in default_values or key in required_keys:
                        value = last_json[key]
                        # Normalize the response field
                        if key == "response" and isinstance(value, str):
                            result[key] = self._validate_and_normalize_grade(value)
                        else:
                            result[key] = value
                
                # Validate required keys
                missing_keys = [k for k in required_keys if k not in result or result[k] is None]
                if missing_keys:
                    logger.warning(f"Missing required keys in response: {missing_keys}")
                    for key in missing_keys:
                        if key not in result:
                            result[key] = default_values.get(key, "")
            else:
                logger.warning("No JSON found in response, using defaults")
                
        except Exception as e:
            logger.error(f"Error extracting structured response: {e}")
        
        return result
