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
        
        # Truncate very long content to prevent context overflow
        max_len = 8000
        problem = problem[:max_len] + "... [truncated]" if len(problem) > max_len else problem
        solution = solution[:max_len] + "... [truncated]" if len(solution) > max_len else solution
        guidelines = guidelines[:max_len] + "... [truncated]" if len(guidelines) > max_len else guidelines
        student_answer = student_answer[:max_len] + "... [truncated]" if len(student_answer) > max_len else student_answer
        
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

## Response Format (REQUIRED - STRICT)
You MUST respond with ONLY valid JSON wrapped in <json>...</json> tags. Do not include any text outside the JSON tags.

<json>
{{
    "reasoning": "Your detailed step-by-step analysis and reasoning...",
    "response": "Your final grade/assessment (e.g., 'Correct', 'Partial', 'Incorrect', or a numeric score)",
    "confidence": "High|Medium|Low"
}}
</json>

IMPORTANT RULES:
- The JSON must be valid (no trailing commas, proper quotes, no comments)
- All three fields are required: 'reasoning', 'response', and 'confidence'
- The 'response' field should contain ONLY the grade, not the reasoning
- 'confidence' indicates how certain you are: High (very sure), Medium (somewhat sure), Low (uncertain)
- Do not include any text before or after the <json> tags
- Escape any quotes within your reasoning with backslash: \\"
"""

    def _normalize_grade(self, grade: str) -> str:
        """Normalize grade to standard format.
        
        Handles various grade formats:
        - Correct: 'Correct', 'correct', '1', 'true', 'True', 'yes', 'Yes'
        - Partial: 'Partial', 'partial', '0.5', 'partially correct'
        - Incorrect: 'Incorrect', 'incorrect', '0', 'false', 'False', 'no', 'No', 'wrong'
        
        Returns:
            Normalized grade: 'Correct', 'Partial', 'Incorrect', or original if unrecognized
        """
        grade_lower = grade.lower().strip()
        
        # Correct variations
        if grade_lower in ('correct', '1', 'true', 'yes', 'right', 'valid', 'pass', 'passed'):
            return 'Correct'
        
        # Partial variations
        if grade_lower in ('partial', '0.5', 'partially correct', 'partially', 'incomplete', 'half'):
            return 'Partial'
        
        # Incorrect variations
        if grade_lower in ('incorrect', '0', 'false', 'no', 'wrong', 'invalid', 'fail', 'failed', 'error'):
            return 'Incorrect'
        
        # Numeric scores: map to categories
        try:
            num = float(grade_lower)
            if num >= 0.8:
                return 'Correct'
            elif num >= 0.4:
                return 'Partial'
            else:
                return 'Incorrect'
        except ValueError:
            pass
        
        # Return original if no normalization applied
        return grade

    def _extract_prediction(self, text: str) -> tuple[str, str, str]:
        """Extract prediction, reasoning, and confidence from response text.
        
        Returns:
            (prediction, reasoning, confidence) tuple with normalized grade
        """
        prediction = "None"
        reasoning = ""
        confidence = "Unknown"
        
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
            last_json = extracted[-1]
            if "response" in last_json:
                raw_prediction = str(last_json["response"]).strip()
                prediction = self._normalize_grade(raw_prediction)
                if raw_prediction != prediction:
                    self.log_fn(f"Grade normalized: '{raw_prediction}' -> '{prediction}'")
            if "reasoning" in last_json:
                reasoning = str(last_json["reasoning"]).strip()
            if "confidence" in last_json:
                confidence = str(last_json["confidence"]).strip()
        
        return prediction, reasoning, confidence

    def forward(self, inputs: dict) -> tuple[str, list[dict], dict]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history, metadata) where metadata includes confidence and reasoning
        """
        instruction = self._build_grading_prompt(inputs)
        
        msg_history = []
        prediction = "None"
        reasoning = ""
        confidence = "Unknown"
        
        # Retry loop for robust extraction
        for attempt in range(self.max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=msg_history if attempt > 0 else [],
                )
                
                last_text = msg_history[-1]["text"] if msg_history else ""
                prediction, reasoning, confidence = self._extract_prediction(last_text)
                
                if prediction != "None":
                    self.log_fn(f"Successfully extracted prediction: {prediction} (confidence: {confidence})")
                    if reasoning:
                        self.log_fn(f"Reasoning: {reasoning[:200]}...")
                    break
                else:
                    self.log_fn(f"Attempt {attempt + 1}: Failed to extract prediction, retrying...")
                    # Add feedback for retry with more specific guidance
                    if attempt < self.max_retries - 1:
                        retry_prompt = self._build_grading_prompt(inputs)
                        instruction = (
                            "ERROR: Your previous response did not contain valid JSON with a 'response' field.\n\n"
                            f"Your response was:\n---\n{last_text[:500]}\n---\n\n"
                            "You MUST respond with ONLY valid JSON wrapped in <json>...</json> tags. "
                            "Do not include any text before or after the JSON tags.\n\n"
                            "Correct format:\n"
                            "<json>\n"
                            '{\n    "reasoning": "Your detailed analysis here...",\n'
                            '    "response": "Correct",\n'
                            '    "confidence": "High"\n'
                            "}\n"
                            "</json>\n\n"
                            "IMPORTANT:\n"
                            "- The JSON must be valid (no trailing commas, proper quotes)\n"
                            "- All three fields are required: 'reasoning', 'response', and 'confidence'\n"
                            "- The 'response' field should contain only the grade\n"
                            "- 'confidence' should be one of: High, Medium, Low\n\n"
                            "Now try again with the original task:\n\n" + retry_prompt
                        )
                    
            except Exception as e:
                self.log_fn(f"Error in attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    prediction = f"Error: {e}"
        
        # Log extraction statistics periodically
        total = sum(self._extraction_stats.values())
        if total > 0 and total % 10 == 0:
            self.log_fn(f"Extraction stats after {total} attempts: {self._extraction_stats}")
        
        metadata = {
            "reasoning": reasoning,
            "confidence": confidence,
            "extraction_stats": dict(self._extraction_stats),
        }
        
        return str(prediction), msg_history, metadata
