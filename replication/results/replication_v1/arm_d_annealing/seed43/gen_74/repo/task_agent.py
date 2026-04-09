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
    Also handles nested JSON objects and common formatting issues.
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
        
        # Try to parse the JSON
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # Try common fixes: remove trailing commas, fix quotes
            fixed = _fix_json_string(inner)
            try:
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                # Try extracting just the outermost JSON object
                try:
                    obj = _extract_outermost_json(inner)
                    if obj:
                        results.append(obj)
                except Exception:
                    continue
    return results or None


def _fix_json_string(text: str) -> str:
    """Apply common JSON fixes."""
    # Remove trailing commas before closing braces/brackets
    import re
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    # Fix single quotes to double quotes (basic)
    text = text.replace("'", '"')
    return text


def _extract_outermost_json(text: str) -> dict | None:
    """Extract the outermost JSON object from text, handling nesting."""
    brace_count = 0
    start_idx = -1
    for i, char in enumerate(text):
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
                    continue
    return None


def _extract_json_with_regex(text: str) -> list[dict] | None:
    """Fallback JSON extraction using regex for malformed responses."""
    results = []
    
    # Try to find JSON objects in code blocks
    json_pattern = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_pattern:
        try:
            results.append(json.loads(json_pattern.group(1)))
        except json.JSONDecodeError:
            # Try with fixes
            try:
                fixed = _fix_json_string(json_pattern.group(1))
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                pass
    
    # Try to find any JSON-like structure with "response" key
    response_pattern = re.search(r'"response"\s*:\s*"([^"]*)"', text)
    if response_pattern and not results:
        results.append({"response": response_pattern.group(1)})
    
    # Try to find JSON without code blocks (look for { ... } patterns)
    if not results:
        # Find all potential JSON objects
        potential_jsons = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        for pj in potential_jsons:
            try:
                obj = json.loads(pj)
                if "response" in obj or "reasoning" in obj:
                    results.append(obj)
            except json.JSONDecodeError:
                try:
                    fixed = _fix_json_string(pj)
                    obj = json.loads(fixed)
                    if "response" in obj or "reasoning" in obj:
                        results.append(obj)
                except json.JSONDecodeError:
                    continue
    
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
        
        # Detect if this is a numeric scoring task
        is_numeric = self._detect_numeric_grading(guidelines)
        
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
    "response": "{self._get_response_example(guidelines, is_numeric)}"
}}
</json>

IMPORTANT: Ensure your JSON is valid and properly formatted. The 'response' field should contain only the grade, not the reasoning.
{self._get_response_guidance(is_numeric)}"""

    def _detect_numeric_grading(self, guidelines: str) -> bool:
        """Detect if the grading guidelines suggest numeric scoring."""
        if not guidelines:
            return False
        numeric_indicators = [
            "score", "points", "out of", "/", "0-", "1-", "2-", "3-", "4-", "5-",
            "0.", "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.",
            "maximum", "minimum", "total", "grade", "mark"
        ]
        guidelines_lower = guidelines.lower()
        return any(ind in guidelines_lower for ind in numeric_indicators)

    def _get_response_example(self, guidelines: str, is_numeric: bool) -> str:
        """Get an appropriate response example based on grading type."""
        if is_numeric:
            # Try to extract score range from guidelines
            import re
            range_match = re.search(r'(\d+)\s*[-/]\s*(\d+)', guidelines)
            if range_match:
                max_score = range_match.group(2)
                return f"Your score (0-{max_score})"
            return "Your numeric score (e.g., '7' or '3.5')"
        return "Your final grade/assessment (e.g., 'Correct', 'Partial', 'Incorrect')"

    def _get_response_guidance(self, is_numeric: bool) -> str:
        """Get additional guidance based on grading type."""
        if is_numeric:
            return "\nFor numeric scores: Provide only the number (e.g., '7' or '3.5'), not a description."
        return ""

    def _extract_prediction(self, text: str) -> tuple[str, str]:
        """Extract prediction and reasoning from response text.
        
        Returns:
            (prediction, reasoning) tuple
        """
        prediction = "None"
        reasoning = ""
        
        # Try primary extraction method
        extracted = _extract_jsons(text)
        if extracted is not None:
            self._extraction_stats["success"] += 1
            self.log_fn(f"JSON extraction: primary method succeeded ({len(extracted)} objects found)")
        else:
            # Fallback to regex extraction
            extracted = _extract_json_with_regex(text)
            if extracted is not None:
                self._extraction_stats["fallback"] += 1
                self.log_fn(f"JSON extraction: fallback method succeeded ({len(extracted)} objects found)")
            else:
                self._extraction_stats["failed"] += 1
                # Log a sample of the problematic text for debugging
                text_preview = text[:300].replace('\n', ' ')
                self.log_fn(f"JSON extraction: all methods failed. Text preview: {text_preview}...")
        
        if extracted:
            last_json = extracted[-1]
            if "response" in last_json:
                prediction = str(last_json["response"]).strip()
                prediction = self._normalize_prediction(prediction)
            if "reasoning" in last_json:
                reasoning = str(last_json["reasoning"]).strip()
        
        return prediction, reasoning

    def _normalize_prediction(self, prediction: str) -> str:
        """Normalize prediction to standard format.
        
        Handles both categorical grades and numeric scores.
        """
        if not prediction:
            return "None"
        
        # Check if it's a numeric score (including decimals)
        numeric_match = re.match(r'^\s*(\d+(?:\.\d+)?)\s*$', prediction)
        if numeric_match:
            # Return numeric score as-is (preserving decimal if present)
            return numeric_match.group(1)
        
        # Check for numeric with trailing text (e.g., "7 points", "3.5 out of 7")
        numeric_with_text = re.match(r'^\s*(\d+(?:\.\d+)?)\s*(?:points?|out of|/|marks?)', prediction, re.IGNORECASE)
        if numeric_with_text:
            return numeric_with_text.group(1)
        
        # Normalize categorical grades
        prediction_lower = prediction.lower().strip()
        
        # Exact matches first
        exact_mappings = {
            "correct": "Correct",
            "right": "Correct", 
            "true": "Correct",
            "yes": "Correct",
            "partial": "Partial",
            "partially correct": "Partial",
            "partial credit": "Partial",
            "incorrect": "Incorrect",
            "wrong": "Incorrect",
            "false": "Incorrect",
            "no": "Incorrect",
            "error": "Incorrect",
        }
        
        if prediction_lower in exact_mappings:
            return exact_mappings[prediction_lower]
        
        # Partial matches for phrases containing grade words
        if any(word in prediction_lower for word in ["correct", "right", "true", "yes"]):
            if any(word in prediction_lower for word in ["partial", "somewhat", "mostly"]):
                return "Partial"
            return "Correct"
        
        if any(word in prediction_lower for word in ["incorrect", "wrong", "false", "no", "error"]):
            return "Incorrect"
        
        if any(word in prediction_lower for word in ["partial", "partially"]):
            return "Partial"
        
        # Return original if no normalization applied
        return prediction

    def _analyze_extraction_error(self, text: str) -> str:
        """Analyze why JSON extraction failed and provide specific guidance."""
        issues = []
        
        # Check for missing JSON tags
        if "<json>" not in text:
            issues.append("- Missing opening <json> tag")
        if "</json>" not in text:
            issues.append("- Missing closing </json> tag")
        
        # Check for JSON tags but no valid JSON inside
        if "<json>" in text and "</json>" in text:
            start = text.find("<json>")
            end = text.find("</json>")
            if start < end:
                inner = text[start + 6:end].strip()
                if not inner:
                    issues.append("- JSON tags are empty")
                else:
                    # Check for common JSON errors
                    if inner.startswith("'") or inner.endswith("'"):
                        issues.append("- Using single quotes instead of double quotes for JSON")
                    if re.search(r',\s*[}\]]', inner):
                        issues.append("- Trailing comma before closing brace/bracket")
                    if "'" in inner and '"' not in inner:
                        issues.append("- JSON strings must use double quotes, not single quotes")
                    if "response" not in inner:
                        issues.append("- Missing 'response' field in JSON")
                    if "reasoning" not in inner:
                        issues.append("- Missing 'reasoning' field in JSON")
        
        # Check for text outside JSON tags
        if "<json>" in text:
            before_json = text[:text.find("<json>")].strip()
            if before_json:
                issues.append("- Text found before <json> tag (remove it)")
        if "</json>" in text:
            after_json_end = text.find("</json>") + 7
            after_json = text[after_json_end:].strip()
            if after_json:
                issues.append("- Text found after </json> tag (remove it)")
        
        if issues:
            return "Issues detected:\n" + "\n".join(issues)
        return "Unable to determine specific issue. Please ensure your response is valid JSON wrapped in <json> tags."

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
                        # Analyze what went wrong to give better feedback
                        error_analysis = self._analyze_extraction_error(last_text)
                        instruction = f"""ERROR: Your previous response did not contain valid JSON with a 'response' field.

Your response was:
---
{last_text[:500]}
---

{error_analysis}

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
- Wrap your ENTIRE response in <json>...</json> tags

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
