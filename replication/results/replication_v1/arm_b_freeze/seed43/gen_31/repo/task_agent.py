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
    Also handles markdown code blocks with json tag.
    """
    results = []
    search_from = 0
    
    # First try explicit <json> tags
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
            continue
    
    # Also try markdown code blocks ```json ... ```
    search_from = 0
    while True:
        start = text.find("```json", search_from)
        if start == -1:
            # Try without json specifier
            start = text.find("```", search_from)
            if start == -1:
                break
            end_marker = "```"
        else:
            end_marker = "```"
        
        # Find the closing ```
        inner_start = start + (7 if text[start:start+7] == "```json" else 3)
        end = text.find(end_marker, inner_start)
        if end == -1:
            break
        
        inner = text[inner_start:end].strip()
        search_from = end + 3
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            continue
    
    return results or None


def _extract_any_json(text: str) -> list[dict] | None:
    """Fallback JSON extraction that looks for any JSON objects in text."""
    results = []
    # Try to find JSON objects between curly braces
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
                    obj = json.loads(text[start_idx:i+1])
                    results.append(obj)
                except json.JSONDecodeError:
                    pass
                start_idx = -1
    return results or None


def _normalize_grade(grade: str) -> str:
    """Normalize grade to X/7 format.
    
    Handles various formats like:
    - "7/7" -> "7/7"
    - "5" -> "5/7"
    - "Correct" -> "7/7"
    - "Partial" -> "3/7"
    - "Incorrect" -> "0/7"
    - "0" -> "0/7"
    """
    if not grade or grade == "None":
        return "None"
    
    grade = grade.strip()
    
    # Already in X/7 format
    if re.match(r'^\d/7$', grade):
        return grade
    
    # X/Y format - convert to X/7 if denominator is 7
    match = re.match(r'^(\d)/(\d)$', grade)
    if match:
        numerator = match.group(1)
        denominator = match.group(2)
        if denominator == "7":
            return f"{numerator}/7"
        # Scale to /7 if different denominator
        try:
            val = int(numerator) / int(denominator) * 7
            return f"{int(round(val))}/7"
        except (ValueError, ZeroDivisionError):
            pass
    
    # Single digit - assume it's out of 7
    if re.match(r'^\d$', grade):
        return f"{grade}/7"
    
    # Text-based grades
    grade_lower = grade.lower()
    if grade_lower in ("correct", "full", "complete", "right", "true"):
        return "7/7"
    if grade_lower in ("incorrect", "wrong", "false", "none", "error"):
        return "0/7"
    if grade_lower in ("partial", "incomplete"):
        return "3/7"
    
    # Try to extract any digit as the grade
    digits = re.findall(r'\d', grade)
    if digits:
        score = int(digits[0])
        if score > 7:
            score = 7
        return f"{score}/7"
    
    return grade


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

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
        # Extract task components for better prompting
        domain = inputs.get("domain", "Unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematical grader for International Mathematical Olympiad (IMO) problems.

Your task is to evaluate a student's answer to a mathematics problem and provide a grade.

DOMAIN: {domain}

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

Think step by step following this structured analysis:

1. PROBLEM ANALYSIS
   - Identify the key mathematical concepts being tested
   - Determine what constitutes a complete and correct solution
   - Note the total marks available for this problem

2. SOLUTION REVIEW
   - Analyze the official solution's approach and key steps
   - Identify critical proof elements or calculations required
   - Note any alternative valid solution paths

3. STUDENT ANSWER EVALUATION
   - Check if the student understood the problem correctly
   - Verify each step of the student's reasoning
   - Identify any errors, gaps, or incorrect assumptions
   - Note any creative or alternative valid approaches
   - Check for mathematical rigor and logical completeness

4. GRADING RUBRIC APPLICATION
   - Full marks (7/7): Complete, correct solution with proper justification
   - High partial (5-6/7): Substantial progress, minor gaps or small errors
   - Medium partial (3-4/7): Significant progress but notable gaps
   - Low partial (1-2/7): Some understanding but major flaws or incomplete
   - No marks (0/7): Incorrect approach or no meaningful progress

5. FINAL DETERMINATION
   - Assign a specific numeric grade (e.g., "7/7", "5/7", "2/7", "0/7")
   - Provide clear justification for the grade
   - Be conservative: only award full marks for truly complete solutions

Respond ONLY in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis covering all 5 steps above",
    "grade": "The final grade as a numeric fraction (e.g., '7/7', '5/7', '3/7', '0/7')",
    "confidence": "High/Medium/Low - your confidence in this grading decision"
}}
</json>

IMPORTANT: The grade field MUST be in the format "X/7" where X is 0-7. Do not use text like "Correct" or "Partial" - use numeric scores."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with enhanced fallback mechanisms
        prediction = "None"
        try:
            last_message = msg_history[-1]["text"]
            
            # Try primary extraction method
            extracted = _extract_jsons(last_message)
            
            # Fallback to generic JSON extraction if primary fails
            if extracted is None:
                extracted = _extract_any_json(last_message)
            
            if extracted:
                # Prefer grade field (new schema), then response field (backward compat)
                last_json = extracted[-1]
                if "grade" in last_json:
                    prediction = last_json["grade"]
                elif "response" in last_json:
                    prediction = last_json["response"]
                elif "answer" in last_json:
                    prediction = last_json["answer"]
                elif "result" in last_json:
                    prediction = last_json["result"]
                elif "evaluation" in last_json:
                    prediction = last_json["evaluation"]
                else:
                    # If no known field, use the first string value found
                    for key, value in last_json.items():
                        if isinstance(value, str) and key != "reasoning" and key != "confidence":
                            prediction = value
                            break
                
                # Validate the prediction is not empty or just whitespace
                if prediction and prediction.strip():
                    prediction = prediction.strip()
                else:
                    prediction = "None"
            else:
                self.log_fn("No JSON found in response, attempting text extraction")
                # Last resort: look for grade-like patterns in text
                grade_patterns = [
                    # Match X/7 format specifically
                    r'(?:grade|score|mark)[\s]*[:=][\s]*["\']?(\d/7|\d/\d)["\']?',
                    r'(?:grade|score|mark)[\s]*[:=][\s]*["\']?([\w\s/\d]+)["\']?',
                    r'(?:final\s+)?grade[\s]*[:=][\s]*([\w\s/\d]+)',
                    r'(\d/7)',  # Direct match for X/7 format
                    r'(?:score|mark)[\s]*[:=][\s]*([\w\s/\d]+)',
                ]
                for pattern in grade_patterns:
                    match = re.search(pattern, last_message, re.IGNORECASE)
                    if match:
                        prediction = match.group(1).strip()
                        break
                
                # If still no match, look for numeric patterns near keywords
                if prediction == "None":
                    # Look for patterns like "grade of 5/7" or "score: 3"
                    fuzzy_patterns = [
                        r'grade\s+(?:of\s+|is\s+)?["\']?(\d/7|\d)["\']?',
                        r'(?:awarded|assigned|given)\s+["\']?(\d/7|\d)["\']?',
                        r'["\']?(\d/7)["\']?\s+(?:points|marks)',
                    ]
                    for pattern in fuzzy_patterns:
                        match = re.search(pattern, last_message, re.IGNORECASE)
                        if match:
                            prediction = match.group(1).strip()
                            break
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Normalize prediction to X/7 format if possible
        prediction = _normalize_grade(prediction)
        
        return str(prediction), msg_history
