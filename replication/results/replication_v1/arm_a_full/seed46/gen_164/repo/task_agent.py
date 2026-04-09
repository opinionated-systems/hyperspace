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
    Also handles markdown code blocks (```json...```) as fallback.
    Includes additional heuristics for malformed JSON.
    """
    if not text:
        return None
        
    results = []
    search_from = 0
    
    # First try <json>...</json> tags
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        # Try to parse the inner content
        parsed = _try_parse_json(inner)
        if parsed is not None:
            results.append(parsed)
    
    # Fallback: try markdown code blocks
    if not results:
        search_from = 0
        while True:
            start = text.find("```json", search_from)
            if start == -1:
                # Also try without 'json' specifier
                start = text.find("```", search_from)
                if start == -1:
                    break
                end = text.find("```", start + 3)
            else:
                end = text.find("```", start + 7)
            if end == -1:
                break
            
            if text[start:start+7] == "```json":
                inner = text[start + 7:end].strip()
            else:
                inner = text[start + 3:end].strip()
            search_from = end + 3
            
            parsed = _try_parse_json(inner)
            if parsed is not None:
                results.append(parsed)
    
    # Last resort: try to find any JSON-like structure in the text
    if not results:
        parsed = _try_parse_json(text)
        if parsed is not None:
            results.append(parsed)
    
    return results or None


def _try_parse_json(text: str) -> dict | None:
    """Try to parse JSON from text with multiple fallback strategies.
    
    Returns the parsed dict or None if parsing fails.
    """
    if not text:
        return None
    
    # Strategy 1: Try direct parsing
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Find JSON object boundaries and parse
    try:
        json_start = text.find("{")
        json_end = text.rfind("}")
        if json_start != -1 and json_end != -1 and json_end > json_start:
            candidate = text[json_start:json_end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
    except Exception:
        pass
    
    # Strategy 3: Try fixing common JSON issues
    try:
        json_start = text.find("{")
        json_end = text.rfind("}")
        if json_start != -1 and json_end != -1 and json_end > json_start:
            candidate = text[json_start:json_end + 1]
            fixed = _fix_json(candidate)
            if fixed:
                try:
                    return json.loads(fixed)
                except json.JSONDecodeError:
                    pass
    except Exception:
        pass
    
    # Strategy 4: Handle nested braces by finding the outermost balanced braces
    try:
        start = text.find("{")
        if start != -1:
            # Count braces to find the matching closing brace
            count = 0
            end = -1
            for i, char in enumerate(text[start:]):
                if char == "{":
                    count += 1
                elif char == "}":
                    count -= 1
                    if count == 0:
                        end = start + i
                        break
            
            if end != -1:
                candidate = text[start:end + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    fixed = _fix_json(candidate)
                    if fixed:
                        try:
                            return json.loads(fixed)
                        except json.JSONDecodeError:
                            pass
    except Exception:
        pass
    
    return None


def _fix_json(text: str) -> str | None:
    """Attempt to fix common JSON formatting issues."""
    if not text:
        return None
    
    # Remove trailing commas before closing braces/brackets
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    
    # Fix single quotes to double quotes (carefully)
    # This is a simplified approach - replace outer single quotes
    if text.startswith("'") and text.endswith("'"):
        text = '"' + text[1:-1] + '"'
    
    # Fix unescaped newlines in strings (but not in already-escaped ones)
    text = re.sub(r'(?<!\\)\n', r'\\n', text)
    
    # Fix unescaped tabs in strings
    text = re.sub(r'(?<!\\)\t', r'\\t', text)
    
    # Fix unescaped carriage returns
    text = re.sub(r'(?<!\\)\r', r'\\r', text)
    
    # Fix common quote issues in values
    # Replace single quotes used as JSON delimiters with double quotes
    # But be careful not to change apostrophes within words
    text = re.sub(r"(?<=[{,:\s])'([^']+)'(?=\s*[,}])", r'"\1"', text)
    
    # Remove control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
    
    return text


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with structured reasoning.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract fields for structured prompting
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematical grader for International Mathematical Olympiad (IMO) problems.

Your task is to evaluate a student's answer by comparing it against the official solution and grading guidelines.

## Problem Domain
{domain}

## Problem Statement
{problem}

## Official Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Grade Definitions (STRICT)

You MUST assign exactly ONE of these four grades:

1. **"correct"** - The answer is:
   - Fully complete with all required steps
   - Logically sound with no errors
   - Matches the official solution's approach and conclusion
   - Would receive full marks (7/7)

2. **"incorrect"** - The answer is:
   - Fundamentally wrong in approach or conclusion
   - Contains critical logical or mathematical errors
   - Missing essential components that make the solution invalid
   - Would receive 0-1 marks out of 7

3. **"partial"** - The answer is:
   - Has some valid approach or correct intermediate steps
   - But incomplete, missing key steps, or has significant gaps
   - Shows understanding of the problem but fails to reach a complete solution
   - Would receive 2-4 marks out of 7

4. **"almost"** - The answer is:
   - Nearly complete and correct
   - Has only minor errors (computation, notation, or small logical gaps)
   - The core approach and reasoning are sound
   - Would receive 5-6 marks out of 7

## Evaluation Process

Follow this step-by-step:
1. Identify the key requirements from the official solution
2. Check if the student's approach matches the expected approach
3. Verify each step for correctness and completeness
4. Determine which grade category best fits based on the definitions above
5. Be STRICT - "almost" requires ONLY minor errors, "partial" has significant gaps

## Response Format

Respond in valid JSON format with the following schema:
<json>
{{
    "analysis": "Your detailed step-by-step analysis...",
    "response": "correct"
}}
</json>

IMPORTANT: The "response" field MUST be exactly one of: "correct", "incorrect", "partial", or "almost". No other values allowed."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = self._extract_prediction(msg_history)

        return str(prediction), msg_history

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history.
        
        Tries multiple extraction strategies and field names for robustness.
        Normalizes output to one of: correct, incorrect, partial, almost.
        """
        if not msg_history:
            return "incorrect"
        
        # Get the last assistant message
        last_msg = msg_history[-1]
        text = last_msg.get("text", "")
        
        if not text:
            return "incorrect"
        
        # Try to extract JSON blocks
        extracted = _extract_jsons(text)
        
        if not extracted:
            # Fallback: try to find any JSON-like structure in the text
            self.log_fn("No JSON blocks found, trying fallback extraction")
            try:
                # Look for patterns like "response": "..." or "grade": "..."
                response_match = re.search(r'["\']response["\']\s*:\s*["\']([^"\']+)["\']', text, re.IGNORECASE)
                if response_match:
                    normalized = self._normalize_label(response_match.group(1))
                    if normalized:
                        return normalized
                grade_match = re.search(r'["\']grade["\']\s*:\s*["\']([^"\']+)["\']', text, re.IGNORECASE)
                if grade_match:
                    normalized = self._normalize_label(grade_match.group(1))
                    if normalized:
                        return normalized
                score_match = re.search(r'["\']score["\']\s*:\s*["\']?([^"\'\s,}]+)', text, re.IGNORECASE)
                if score_match:
                    normalized = self._normalize_label(score_match.group(1))
                    if normalized:
                        return normalized
                # Try to find any of the target labels directly in text
                # Check in priority order to avoid misclassification
                for label in ["almost", "incorrect", "partial", "correct"]:
                    if re.search(rf'\b{label}\b', text, re.IGNORECASE):
                        return label
            except Exception as e:
                self.log_fn(f"Fallback extraction failed: {e}")
            return "incorrect"
        
        # Try to get response from extracted JSON
        last_extract = extracted[-1]
        
        # Log the analysis if available for debugging
        if "analysis" in last_extract:
            self.log_fn(f"Analysis: {str(last_extract['analysis'])[:200]}...")
        
        # Priority order for field names - response field has highest priority
        field_priority = ["response", "grade", "score", "result", "evaluation", "verdict", "label"]
        
        for field in field_priority:
            if field in last_extract:
                value = last_extract[field]
                normalized = self._normalize_label(str(value))
                if normalized:
                    return normalized
        
        # If no known field found, try to normalize any string value
        for key, value in last_extract.items():
            if isinstance(value, str) and value:
                normalized = self._normalize_label(value)
                if normalized:
                    return normalized
            # Also check if value is a list and extract first string element
            elif isinstance(value, list) and value:
                for item in value:
                    if isinstance(item, str):
                        normalized = self._normalize_label(item)
                        if normalized:
                            return normalized
        
        return "incorrect"
    
    def _normalize_label(self, value: str) -> str:
        """Normalize a value to one of the four expected labels.
        
        Returns one of: correct, incorrect, partial, almost, or empty string if no match.
        Uses a priority-based approach to handle ambiguous cases.
        """
        if not value:
            return ""
        
        value_lower = value.lower().strip().strip('"\'')
        
        # Direct matches (highest priority)
        if value_lower in ("correct", "incorrect", "partial", "almost"):
            return value_lower
        
        # Check for exact word matches first (before substring matching)
        words = re.findall(r'\b\w+\b', value_lower)
        
        # Priority 1: Check for "almost" patterns (must be checked before "correct")
        almost_patterns = [
            "almost", "nearly", "minor", "small error", "tiny error", "slight",
            "5/7", "6/7", "5 marks", "6 marks", "5-6", "mostly correct",
            "very close", "just missed", "minor mistake", "small mistake"
        ]
        for pattern in almost_patterns:
            if pattern in value_lower:
                return "almost"
        
        # Priority 2: Check for "incorrect" patterns (must be checked before "correct")
        incorrect_patterns = [
            "incorrect", "wrong", "false", "invalid", "error", "mistake",
            "0/7", "0 marks", "1/7", "1 mark", "0-1", "fundamentally wrong",
            "completely wrong", "totally wrong", "does not work", "flawed"
        ]
        for pattern in incorrect_patterns:
            if pattern in value_lower:
                return "incorrect"
        
        # Priority 3: Check for "partial" patterns
        partial_patterns = [
            "partial", "incomplete", "some", "half", "partly",
            "2/7", "3/7", "4/7", "2 marks", "3 marks", "4 marks",
            "2-4", "partially correct", "missing steps", "significant gaps"
        ]
        for pattern in partial_patterns:
            if pattern in value_lower:
                return "partial"
        
        # Priority 4: Check for "correct" patterns (lowest priority to avoid false positives)
        correct_patterns = [
            "correct", "right", "true", "valid", "proper", "accurate",
            "7/7", "full marks", "full credit", "100%", "complete solution",
            "fully correct", "entirely correct", "perfect"
        ]
        for pattern in correct_patterns:
            if pattern in value_lower:
                return "correct"
        
        # Numeric score mapping
        try:
            # Extract numeric score if present
            score_match = re.search(r'(\d+)\s*/\s*7', value_lower)
            if score_match:
                score = int(score_match.group(1))
                if score >= 7:
                    return "correct"
                elif score >= 5:
                    return "almost"
                elif score >= 2:
                    return "partial"
                else:
                    return "incorrect"
            
            # Check for standalone numbers
            num_match = re.search(r'\b([0-7])\b', value_lower)
            if num_match:
                score = int(num_match.group(1))
                if score >= 7:
                    return "correct"
                elif score >= 5:
                    return "almost"
                elif score >= 2:
                    return "partial"
                else:
                    return "incorrect"
        except (ValueError, IndexError):
            pass
        
        # Final fallback: substring matching with priority
        if "almost" in value_lower:
            return "almost"
        if "incorrect" in value_lower or "wrong" in value_lower:
            return "incorrect"
        if "partial" in value_lower:
            return "partial"
        if "correct" in value_lower:
            return "correct"
        
        return ""
