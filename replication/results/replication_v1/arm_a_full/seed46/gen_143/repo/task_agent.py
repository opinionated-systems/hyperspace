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
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # Try to extract JSON from within the content if it's wrapped in other text
            try:
                # Find first { and last }
                json_start = inner.find("{")
                json_end = inner.rfind("}")
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    results.append(json.loads(inner[json_start:json_end + 1]))
            except json.JSONDecodeError:
                # Try to fix common JSON issues
                try:
                    fixed = _fix_json(inner[json_start:json_end + 1] if json_start != -1 and json_end != -1 else inner)
                    if fixed:
                        results.append(json.loads(fixed))
                except (json.JSONDecodeError, ValueError):
                    continue
    
    # Fallback: try markdown code blocks with json specifier
    if not results:
        search_from = 0
        while True:
            start = text.find("```json", search_from)
            if start == -1:
                break
            end = text.find("```", start + 7)
            if end == -1:
                break
            
            inner = text[start + 7:end].strip()
            search_from = end + 3
            try:
                results.append(json.loads(inner))
            except json.JSONDecodeError:
                # Try to fix common JSON issues
                try:
                    fixed = _fix_json(inner)
                    if fixed:
                        results.append(json.loads(fixed))
                except (json.JSONDecodeError, ValueError):
                    continue
    
    # Try markdown code blocks without json specifier
    if not results:
        search_from = 0
        while True:
            start = text.find("```", search_from)
            if start == -1:
                break
            end = text.find("```", start + 3)
            if end == -1:
                break
            
            inner = text[start + 3:end].strip()
            search_from = end + 3
            # Only try if it looks like JSON (starts with {)
            if inner.startswith("{"):
                try:
                    results.append(json.loads(inner))
                except json.JSONDecodeError:
                    # Try to fix common JSON issues
                    try:
                        fixed = _fix_json(inner)
                        if fixed:
                            results.append(json.loads(fixed))
                    except (json.JSONDecodeError, ValueError):
                        continue
    
    # Last resort: try to find any JSON-like structure in the text
    if not results:
        try:
            # Find outermost braces - look for the largest valid JSON object
            json_start = text.find("{")
            while json_start != -1:
                json_end = text.rfind("}")
                while json_end != -1 and json_end > json_start:
                    candidate = text[json_start:json_end + 1]
                    try:
                        results.append(json.loads(candidate))
                        break
                    except json.JSONDecodeError:
                        # Try to fix common JSON issues
                        try:
                            fixed = _fix_json(candidate)
                            if fixed:
                                results.append(json.loads(fixed))
                                break
                        except (json.JSONDecodeError, ValueError):
                            # Try next inner closing brace
                            json_end = text.rfind("}", json_start, json_end)
                            continue
                if results:
                    break
                # Try next opening brace
                json_start = text.find("{", json_start + 1)
        except (json.JSONDecodeError, ValueError):
            pass
    
    return results or None


def _fix_json(text: str) -> str | None:
    """Attempt to fix common JSON formatting issues."""
    if not text:
        return None
    
    text = text.strip()
    
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
    
    # Fix common quote issues in JSON values
    # Replace single quotes used as JSON delimiters with double quotes
    # This is a more aggressive fix for cases like {'key': 'value'}
    if "'" in text and '"' not in text:
        # If only single quotes are used, try to convert them
        text = text.replace("'", '"')
    
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

## Label Definitions (READ CAREFULLY)

You MUST choose EXACTLY ONE of these four labels:

1. **"correct"** - Student's answer is completely correct:
   - All reasoning steps are valid and logically sound
   - All calculations are correct
   - The final answer matches the official solution
   - The proof/argument is complete with no gaps

2. **"incorrect"** - Student's answer is fundamentally wrong:
   - The core approach or strategy is wrong
   - Major conceptual errors that invalidate the solution
   - The final answer is wrong AND the reasoning is flawed
   - No meaningful progress toward the solution

3. **"partial"** - Student made significant progress but has substantial gaps:
   - Correct initial approach but incomplete execution
   - Some correct steps but missing key components
   - Partial credit warranted (roughly 1-4 points out of 7)
   - Has the right idea but didn't finish or made significant errors
   - Example: Correct setup but wrong final answer, or missing crucial proof steps

4. **"almost"** - Student's answer is nearly correct with only minor issues:
   - The solution is essentially correct with only trivial errors
   - Minor computational slip that doesn't affect the main argument
   - Small notational issues or missing a minor case
   - Would receive 5-6 points out of 7 (high partial credit)
   - The core insight and main proof structure are correct

## Critical Distinctions
- "partial" = significant progress but substantial gaps (low-mid partial credit)
- "almost" = nearly perfect, minor issues only (high partial credit)
- When in doubt between "partial" and "incorrect", choose "partial" if they had the right idea
- When in doubt between "almost" and "correct", choose "almost" if there's ANY issue

## Instructions
1. First, carefully read and understand the official solution and grading guidelines.
2. Analyze the student's answer step by step:
   - Identify what mathematical concepts they used correctly
   - Identify any errors in their reasoning or calculations
   - Check if they reached the correct final answer
   - Note any missing steps or incomplete proofs
3. Compare their work against the grading guidelines to determine partial credit.
4. In your analysis, explicitly state which label criteria apply and why.
5. Provide your final evaluation in the "response" field using EXACTLY one of these labels: "correct", "incorrect", "partial", or "almost".

IMPORTANT: The response field must contain ONLY one of these four exact values: "correct", "incorrect", "partial", or "almost". Do not include any other text, numbers, or explanations in the response field.

Respond in valid JSON format with the following schema:
<json>
{{
    "analysis": "Your detailed step-by-step analysis of the student's answer...",
    "response": "correct"
}}
</json>"""

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
                # Use more flexible patterns to catch various quote styles
                response_patterns = [
                    r'["\']?response["\']?\s*:\s*["\']?([^"\'\s,}\]]+)["\']?',
                    r'["\']?grade["\']?\s*:\s*["\']?([^"\'\s,}\]]+)["\']?',
                    r'["\']?score["\']?\s*:\s*["\']?([^"\'\s,}\]]+)["\']?',
                    r'["\']?result["\']?\s*:\s*["\']?([^"\'\s,}\]]+)["\']?',
                    r'["\']?evaluation["\']?\s*:\s*["\']?([^"\'\s,}\]]+)["\']?',
                    r'["\']?verdict["\']?\s*:\s*["\']?([^"\'\s,}\]]+)["\']?',
                    r'["\']?label["\']?\s*:\s*["\']?([^"\'\s,}\]]+)["\']?',
                ]
                for pattern in response_patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        normalized = self._normalize_label(match.group(1))
                        if normalized:
                            return normalized
                
                # Try to find any of the target labels directly in text
                # Check in order of specificity to avoid false matches
                for label in ["almost", "partial", "incorrect", "correct"]:
                    # Use word boundary and case-insensitive matching
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
        
        # Priority order for field names - response is most specific to our use case
        field_priority = ["response", "grade", "score", "result", "evaluation", "verdict", "label", "answer"]
        
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
            elif isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], str):
                    normalized = self._normalize_label(value[0])
                    if normalized:
                        return normalized
        
        return "incorrect"
    
    def _normalize_label(self, value: str) -> str:
        """Normalize a value to one of the four expected labels.
        
        Returns one of: correct, incorrect, partial, almost, or empty string if no match.
        """
        if not value:
            return ""
        
        value_lower = value.lower().strip().strip('"\'')
        
        # Direct matches (exact)
        if value_lower in ("correct", "incorrect", "partial", "almost"):
            return value_lower
        
        # Common variations for "correct"
        correct_variations = (
            "true", "yes", "right", "valid", "7", "full", "full credit", "100%",
            "perfect", "complete", "fully correct", "all correct", "entirely correct",
            "completely correct", "totally correct", "absolutely correct"
        )
        if value_lower in correct_variations:
            return "correct"
        
        # Common variations for "incorrect"
        incorrect_variations = (
            "false", "no", "wrong", "invalid", "0", "none", "zero", "0/7",
            "not correct", "not right", "entirely wrong", "completely wrong",
            "totally wrong", "absolutely wrong", "failed", "fail", "error"
        )
        if value_lower in incorrect_variations:
            return "incorrect"
        
        # Common variations for "partial"
        partial_variations = (
            "partially correct", "partial credit", "some", "half", "3", "4",
            "partially", "partly correct", "some credit", "incomplete",
            "mostly wrong", "mostly incorrect", "partial success", "1", "2",
            "limited correct", "few correct", "minor progress"
        )
        if value_lower in partial_variations:
            return "partial"
        
        # Common variations for "almost"
        almost_variations = (
            "almost correct", "close", "minor error", "6", "nearly",
            "nearly correct", "mostly correct", "very close", "small error",
            "trivial error", "slight error", "tiny error", "minimal error",
            "near correct", "almost right", "nearly right", "5", "high partial"
        )
        if value_lower in almost_variations:
            return "almost"
        
        # Check for substrings - be careful about order to avoid false matches
        # Check for "incorrect" first (more specific than "correct")
        if "incorrect" in value_lower or "wrong" in value_lower:
            return "incorrect"
        # Check for "almost" before "correct" to catch "almost correct"
        if "almost" in value_lower:
            return "almost"
        # Check for "partial" 
        if "partial" in value_lower:
            return "partial"
        # Check for "correct" last (least specific)
        if "correct" in value_lower:
            return "correct"
        
        return ""
