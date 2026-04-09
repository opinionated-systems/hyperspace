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
from typing import Any

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles nested JSON objects by tracking brace depth.
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
            # Try to extract JSON by finding balanced braces
            json_obj = _extract_balanced_json(inner)
            if json_obj:
                results.append(json_obj)
            else:
                # Try to fix common JSON issues
                fixed = _fix_common_json_issues(inner)
                if fixed:
                    try:
                        results.append(json.loads(fixed))
                    except json.JSONDecodeError:
                        pass
            continue
    return results or None


def _fix_common_json_issues(text: str) -> str | None:
    """Attempt to fix common JSON formatting issues.
    
    Returns fixed JSON string or None if cannot fix.
    """
    if not text or not text.strip():
        return None
    
    text = text.strip()
    
    # Fix 1: Remove trailing commas before closing braces/brackets
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    
    # Fix 2: Add quotes around unquoted keys (simple cases)
    # Match word characters followed by colon, not already in quotes
    text = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:)', r'\1"\2"\3', text)
    
    # Fix 3: Replace single quotes with double quotes for JSON keys/values
    # Simple heuristic: replace 'key': or 'value' patterns
    text = re.sub(r"'([a-zA-Z_][a-zA-Z0-9_]*)'\s*:", r'"\1":', text)
    
    # Fix 4: Ensure the text starts with { and ends with }
    if not text.startswith('{'):
        # Try to find the first {
        start_idx = text.find('{')
        if start_idx != -1:
            text = text[start_idx:]
    
    if not text.endswith('}'):
        # Try to find the last }
        end_idx = text.rfind('}')
        if end_idx != -1:
            text = text[:end_idx+1]
    
    return text if text.startswith('{') and text.endswith('}') else None


def _extract_balanced_json(text: str) -> dict | None:
    """Extract a JSON object by tracking brace depth.
    
    This handles cases where the JSON content contains nested braces
    that might confuse simple regex-based extraction.
    """
    start_idx = -1
    brace_depth = 0
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
            continue
        if char == '"' and in_string:
            in_string = False
            continue
        if not in_string:
            if char == '{':
                if brace_depth == 0:
                    start_idx = i
                brace_depth += 1
            elif char == '}':
                brace_depth -= 1
                if brace_depth == 0 and start_idx != -1:
                    try:
                        return json.loads(text[start_idx:i+1])
                    except json.JSONDecodeError:
                        continue
    return None


def _extract_json_with_regex(text: str) -> list[dict] | None:
    """Fallback JSON extraction using multiple strategies for malformed responses.
    
    Enhanced version with better handling of nested quotes, multiline strings,
    and various edge cases in LLM outputs.
    """
    results = []
    
    # Strategy 1: Try to find JSON objects in code blocks (improved pattern)
    # Handle both ```json and ``` code blocks with better nesting support
    code_block_patterns = [
        r'```json\s*([\s\S]*?)\s*```',
        r'```\s*([\s\S]*?)\s*```',
    ]
    for pattern in code_block_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            # Clean up the match - extract just the JSON object
            match = match.strip()
            # Try to find the first { and last } to handle extra text
            start = match.find('{')
            end = match.rfind('}')
            if start != -1 and end != -1 and end > start:
                match = match[start:end+1]
            
            try:
                parsed = json.loads(match)
                if isinstance(parsed, dict):
                    results.append(parsed)
                    continue
            except json.JSONDecodeError:
                pass
            
            # Try balanced brace extraction on the content
            balanced = _extract_balanced_json(match)
            if balanced:
                results.append(balanced)
                continue
            
            # Try fixing common issues
            fixed = _fix_common_json_issues(match)
            if fixed:
                try:
                    parsed = json.loads(fixed)
                    if isinstance(parsed, dict):
                        results.append(parsed)
                except json.JSONDecodeError:
                    pass
    
    # Strategy 2: Look for any JSON-like structure with balanced braces in the full text
    if not results:
        balanced = _extract_balanced_json(text)
        if balanced:
            results.append(balanced)
    
    # Strategy 3: Extract key-value pairs for response and reasoning (improved)
    # Handle both quoted and unquoted values, with better multiline support
    if not results:
        extracted = {}
        
        # Improved regex for quoted strings with escaped quotes
        # Matches: "key": "value" or "key": "value with \"escaped\" quotes"
        quoted_pattern = r'"(\w+)"\s*:\s*"((?:[^"\\]|\\.)*)"'
        for match in re.finditer(quoted_pattern, text, re.DOTALL):
            key = match.group(1)
            value = match.group(2).replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
            extracted[key] = value
        
        # Also try to find unquoted values: "key": value or "key": true/false/null
        unquoted_pattern = r'"(\w+)"\s*:\s*(true|false|null|\d+(?:\.\d+)?)\s*[,}\]]'
        for match in re.finditer(unquoted_pattern, text):
            key = match.group(1)
            value = match.group(2)
            # Convert to proper Python types
            if value == 'true':
                extracted[key] = True
            elif value == 'false':
                extracted[key] = False
            elif value == 'null':
                extracted[key] = None
            else:
                # Try to convert to number
                try:
                    if '.' in value:
                        extracted[key] = float(value)
                    else:
                        extracted[key] = int(value)
                except ValueError:
                    extracted[key] = value
        
        if extracted and ("response" in extracted or "reasoning" in extracted):
            results.append(extracted)
    
    # Strategy 4: Look for plain text responses without JSON structure (expanded)
    if not results:
        text_lower = text.lower()
        
        # Expanded patterns for grade extraction
        grade_patterns = [
            (r'\bgrade\s*[:=]\s*(correct|partial|incorrect)\b', 1),
            (r'\bthe\s+answer\s+is\s+(correct|partial|incorrect)\b', 1),
            (r'\bthis\s+is\s+(correct|partial|incorrect)\b', 1),
            (r'\bassessment\s*[:=]\s*(correct|partial|incorrect)\b', 1),
            (r'\bevaluation\s*[:=]\s*(correct|partial|incorrect)\b', 1),
            (r'\bresult\s*[:=]\s*(correct|partial|incorrect)\b', 1),
            (r'\bscore\s*[:=]\s*(correct|partial|incorrect)\b', 1),
            (r'\bfinal\s+(?:grade|score|assessment)\s*[:=]\s*(correct|partial|incorrect)\b', 1),
            # Also look for standalone grades at the end of sentences
            (r'(?:is|was|would\s+be)\s+(correct|partial|incorrect)[.\s]*$', 1),
        ]
        
        for pattern, group_idx in grade_patterns:
            match = re.search(pattern, text_lower)
            if match:
                grade = match.group(group_idx).capitalize()
                results.append({"response": grade})
                break
    
    # Strategy 5: Last resort - look for any occurrence of the grade words
    if not results:
        text_lower = text.lower()
        # Check for grade words with word boundaries to avoid partial matches
        if re.search(r'\bcorrect\b', text_lower) and not re.search(r'\bincorrect\b', text_lower):
            results.append({"response": "Correct"})
        elif re.search(r'\bpartial\w*\b', text_lower):
            results.append({"response": "Partial"})
        elif re.search(r'\bincorrect\b|\bwrong\b', text_lower):
            results.append({"response": "Incorrect"})
    
    return results or None


def _normalize_prediction(prediction: str) -> str:
    """Normalize prediction string for consistent comparison.
    
    Handles common variations like case differences, extra whitespace,
    punctuation differences, and partial matches.
    """
    if not prediction:
        return "None"
    
    # Strip whitespace and normalize case
    normalized = prediction.strip()
    
    # Handle common grade variations with exact matches
    grade_map = {
        "correct": "Correct",
        "partial": "Partial",
        "partially correct": "Partial",
        "partial credit": "Partial",
        "incorrect": "Incorrect",
        "wrong": "Incorrect",
        "right": "Correct",
        "true": "Correct",
        "false": "Incorrect",
        "full credit": "Correct",
        "no credit": "Incorrect",
        "half credit": "Partial",
        "mostly correct": "Partial",
        "mostly wrong": "Incorrect",
    }
    
    lower_pred = normalized.lower()
    if lower_pred in grade_map:
        return grade_map[lower_pred]
    
    # Handle partial matches and variations with punctuation
    # Remove punctuation for matching
    clean_pred = re.sub(r'[^\w\s]', '', lower_pred).strip()
    
    # Check for numeric grades (0-100 scale)
    numeric_match = re.search(r'\b(\d+(?:\.\d+)?)\s*(?:/\s*100)?\b', clean_pred)
    if numeric_match:
        try:
            score = float(numeric_match.group(1))
            if score >= 90:
                return "Correct"
            elif score >= 50:
                return "Partial"
            else:
                return "Incorrect"
        except ValueError:
            pass
    
    # Check for partial matches
    if "partial" in clean_pred or "partially" in clean_pred:
        return "Partial"
    if any(word in clean_pred for word in ["correct", "right", "true", "yes", "valid", "accurate"]):
        return "Correct"
    if any(word in clean_pred for word in ["incorrect", "wrong", "false", "no", "error", "invalid", "inaccurate"]):
        return "Incorrect"
    
    return normalized


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "", max_retries: int = 3) -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = max_retries
        self._extraction_stats = {"success": 0, "fallback": 0, "failure": 0}
        self._log_file = log_file
        self._grading_history: list[dict] = []
    
    def _log_grading_result(self, inputs: dict, prediction: str, reasoning: str, 
                            method: str, attempt_details: list[dict]) -> None:
        """Log a grading result to the log file if configured."""
        if not self._log_file:
            return
        
        import os
        from datetime import datetime, timezone
        
        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "domain": inputs.get("domain", "unknown"),
            "prediction": prediction,
            "reasoning": reasoning[:500] if reasoning else "",  # Truncate for size
            "extraction_method": method,
            "attempts": len(attempt_details),
            "attempt_details": attempt_details,
            "model": self.model,
        }
        
        self._grading_history.append(result)
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self._log_file) or ".", exist_ok=True)
            # Append as JSONL
            with open(self._log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, default=str) + "\n")
        except Exception as e:
            self.log_fn(f"Warning: Failed to write to log file {self._log_file}: {e}")
    
    def get_grading_history(self) -> list[dict]:
        """Return the grading history for analysis."""
        return list(self._grading_history)
    
    def export_grading_summary(self, output_path: str) -> bool:
        """Export a summary of all grading results to a JSON file.
        
        Returns True if successful, False otherwise.
        """
        if not self._grading_history:
            return False
        
        summary = {
            "total_graded": len(self._grading_history),
            "extraction_stats": dict(self._extraction_stats),
            "grade_distribution": {},
            "results": self._grading_history,
        }
        
        # Calculate grade distribution
        for result in self._grading_history:
            grade = result.get("prediction", "Unknown")
            summary["grade_distribution"][grade] = summary["grade_distribution"].get(grade, 0) + 1
        
        try:
            import os
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, default=str)
            return True
        except Exception as e:
            self.log_fn(f"Error exporting grading summary: {e}")
            return False
    
    def get_extraction_stats(self) -> dict:
        """Return the current extraction statistics."""
        return dict(self._extraction_stats)
    
    def reset_extraction_stats(self) -> None:
        """Reset extraction statistics to zero."""
        self._extraction_stats = {"success": 0, "fallback": 0, "failure": 0}

    def _build_grading_prompt(self, inputs: dict, is_retry: bool = False) -> str:
        """Build a structured prompt for the grading task with chain-of-thought."""
        domain = inputs.get("domain", "unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        base_prompt = f"""You are an expert grader for {domain} problems. Your task is to evaluate a student's answer against the correct solution and grading guidelines.

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

## Example of Good Reasoning Structure
Your reasoning should follow this pattern:
- Step 1: Identify the key concepts/methods needed for the problem
- Step 2: Check if the student's approach matches the correct solution
- Step 3: Identify any errors or gaps in the student's work
- Step 4: Evaluate partial credit based on grading guidelines
- Step 5: Summarize and provide final grade justification

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

        if is_retry:
            return f"""ERROR: Your previous response did not contain valid JSON with a 'response' field, or the JSON was malformed.

You MUST respond with ONLY valid JSON wrapped in <json>...</json> tags. Do not include any text before or after the JSON tags.

Correct format:
<json>
{{
    "reasoning": "Your detailed analysis here...",
    "response": "Correct"
}}
</json>

Now try again with the original task:

{base_prompt}"""
        
        return base_prompt

    def _extract_prediction(self, text: str) -> tuple[str, str, str]:
        """Extract prediction and reasoning from response text.
        
        Returns:
            (prediction, reasoning, method) tuple where method indicates extraction success
        """
        prediction = "None"
        reasoning = ""
        method = "failure"
        
        # Try primary extraction method
        extracted = _extract_jsons(text)
        if extracted:
            method = "success"
        else:
            # Fallback to regex extraction
            extracted = _extract_json_with_regex(text)
            if extracted:
                method = "fallback"
        
        if extracted:
            last_json = extracted[-1]
            if "response" in last_json:
                prediction = str(last_json["response"])
            if "reasoning" in last_json:
                reasoning = str(last_json["reasoning"])
        
        # Normalize the prediction
        prediction = _normalize_prediction(prediction)
        
        return prediction, reasoning, method

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_grading_prompt(inputs, is_retry=False)
        
        msg_history = []
        prediction = "None"
        reasoning = ""
        
        # Track all attempts for better debugging
        attempt_details = []
        
        # Retry loop for robust extraction
        for attempt in range(self.max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=msg_history if attempt > 0 else [],
                )
                
                last_text = msg_history[-1]["text"] if msg_history else ""
                prediction, reasoning, method = self._extract_prediction(last_text)
                
                # Update statistics
                self._extraction_stats[method] += 1
                
                # Log detailed attempt info
                attempt_info = {
                    "attempt": attempt + 1,
                    "method": method,
                    "prediction": prediction,
                    "response_length": len(last_text),
                    "has_reasoning": bool(reasoning),
                }
                attempt_details.append(attempt_info)
                
                if prediction != "None":
                    self.log_fn(f"[Attempt {attempt + 1}] Successfully extracted prediction: {prediction} (method: {method})")
                    if reasoning:
                        self.log_fn(f"[Attempt {attempt + 1}] Reasoning preview: {reasoning[:200]}...")
                    # Log extraction stats summary
                    self.log_fn(f"Extraction stats: {dict(self._extraction_stats)}")
                    break
                else:
                    self.log_fn(f"[Attempt {attempt + 1}] Failed to extract prediction (method: {method})")
                    # Log a preview of the problematic response for debugging
                    preview = last_text[:300].replace('\n', ' ')
                    self.log_fn(f"[Attempt {attempt + 1}] Response preview: {preview}...")
                    
                    # Add feedback for retry
                    if attempt < self.max_retries - 1:
                        instruction = self._build_grading_prompt(inputs, is_retry=True)
                    
            except Exception as e:
                error_msg = f"[Attempt {attempt + 1}] Error: {type(e).__name__}: {str(e)[:100]}"
                self.log_fn(error_msg)
                attempt_details.append({"attempt": attempt + 1, "error": str(e)})
                if attempt == self.max_retries - 1:
                    prediction = f"Error: {e}"
        
        # Log summary of all attempts if we failed
        if prediction == "None" or str(prediction).startswith("Error:"):
            self.log_fn(f"All {self.max_retries} attempts failed. Details: {attempt_details}")
        
        # Log the grading result if a log file is configured
        self._log_grading_result(inputs, str(prediction), reasoning, method, attempt_details)
        
        return str(prediction), msg_history
