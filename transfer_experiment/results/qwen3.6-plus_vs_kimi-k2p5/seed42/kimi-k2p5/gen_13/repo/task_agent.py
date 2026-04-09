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
    """Extract JSON objects from <json>...</json> blocks."""
    if not text or not isinstance(text, str):
        return None
    
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
        
        try:
            parsed = json.loads(inner)
            if isinstance(parsed, dict):
                results.append(parsed)
        except json.JSONDecodeError:
            # Try with single quotes replaced
            try:
                parsed = json.loads(inner.replace("'", '"'))
                if isinstance(parsed, dict):
                    results.append(parsed)
            except json.JSONDecodeError:
                pass
    
    return results if results else None


def _extract_response_robust(text: str) -> str | None:
    """Robust extraction using multiple pattern matching strategies."""
    if not text or not isinstance(text, str):
        return None
        
    text_lower = text.lower()
    valid_grades = ['correct', 'incorrect', 'partial', 'almost']
    
    # Priority 1: Look for <json> tags with response field
    try:
        json_results = _extract_jsons(text)
        if json_results:
            for item in json_results:
                if isinstance(item, dict) and "response" in item:
                    val = item["response"]
                    if isinstance(val, str) and val.lower() in valid_grades:
                        return 'partial' if val.lower() == 'almost' else val.lower()
    except Exception:
        pass
    
    # Priority 2: Look for JSON-like patterns with response field
    json_patterns = [
        r'"response"\s*:\s*"(correct|incorrect|partial|almost)"',
        r"'response'\s*:\s*'(correct|incorrect|partial|almost)'",
        r'response\s*:\s*"(correct|incorrect|partial|almost)"',
        r'response\s*:\s*(correct|incorrect|partial|almost)',
        r'{"response":\s*"(correct|incorrect|partial|almost)"}',
        r"{'response':\s*'(correct|incorrect|partial|almost)'}",
    ]
    
    for pattern in json_patterns:
        match = re.search(pattern, text_lower)
        if match:
            result = match.group(1)
            return 'partial' if result == 'almost' else result
    
    # Priority 3: Look for the grade as a standalone word at the end of lines
    lines = text_lower.split('\n')
    for line in reversed(lines):
        line_stripped = line.strip().rstrip('.,;:!?')
        for grade in valid_grades:
            if line_stripped == grade or re.search(rf'\b{grade}\b', line_stripped):
                if line_stripped.endswith(grade) or line_stripped == grade:
                    return 'partial' if grade == 'almost' else grade
    
    # Priority 4: Look for grade in quotes anywhere in text
    for grade in valid_grades:
        if re.search(rf'"{grade}"', text_lower) or re.search(rf"'{grade}'", text_lower) or re.search(rf'`{grade}`', text_lower):
            return 'partial' if grade == 'almost' else grade
    
    # Priority 5: Look for explicit decision patterns
    decision_patterns = [
        rf'\bgrade\s*[:=]\s*["\']?(correct|incorrect|partial|almost)["\']?',
        rf'\bverdict\s*[:=]\s*["\']?(correct|incorrect|partial|almost)["\']?',
        rf'\bfinal\s+(?:grade|verdict|answer)\s*[:=]\s*["\']?(correct|incorrect|partial|almost)["\']?',
        rf'\bthe\s+answer\s+is\s*[:=]?\s*["\']?(correct|incorrect|partial|almost)["\']?',
    ]
    
    for pattern in decision_patterns:
        match = re.search(pattern, text_lower)
        if match:
            result = match.group(1)
            return 'partial' if result == 'almost' else result
    
    # Priority 6: Look for grade in brackets or parentheses
    bracket_patterns = [
        rf'\((correct|incorrect|partial|almost)\)',
        rf'\[(correct|incorrect|partial|almost)\]',
        rf'\{{(correct|incorrect|partial|almost)\}}',
    ]
    for pattern in bracket_patterns:
        match = re.search(pattern, text_lower)
        if match:
            result = match.group(1)
            return 'partial' if result == 'almost' else result
    
    # Priority 7: Look for grade at the very end of the text
    text_stripped = text_lower.strip()
    for grade in valid_grades:
        if text_stripped.endswith(grade):
            return 'partial' if grade == 'almost' else grade
    
    # Priority 8: Simple word boundary search for grades
    for grade in valid_grades:
        if re.search(rf'\b{grade}\b', text_lower):
            return 'partial' if grade == 'almost' else grade
    
    return None


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
        # Extract fields from inputs
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Build the prompt
        prompt = f"""You are an expert mathematics grader for International Mathematical Olympiad (IMO) problems.

Your task is to carefully analyze a student's answer and determine if it is correct, incorrect, or partial.

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

ANALYSIS INSTRUCTIONS:
1. First, identify what the problem is asking for and what constitutes a complete solution.
2. Compare the student's approach to the official solution - are they using valid methods?
3. Check for any logical gaps, errors, or unjustified claims in the student's work.
4. Look specifically for these indicators in the grading guidelines:
   - "(Almost)" or "almost correct" → indicates the answer is very close to correct with minor issues
   - "(Correct)" → indicates full credit
   - "(Incorrect)" or "(0 points)" → indicates no credit
   - "(Partial)" or partial points → indicates partial credit

5. Determine the grade based on these criteria:
   - CORRECT: Complete, valid proof with all steps justified and correct conclusion
   - INCORRECT: Fundamental errors, wrong approach, or conclusion that doesn't follow
   - PARTIAL: Valid insights, correct lemmas, good approach but significant gaps remain
   - ALMOST: Nearly complete solution with only minor errors or omissions (use when guidelines indicate "almost")

IMPORTANT: Pay special attention to "(Almost)" markers in the grading guidelines - these indicate answers that deserve the "almost" label, which is distinct from "partial".

Provide your detailed analysis, then output your final grade in the following EXACT format:
<json>{{"response": "GRADE"}}</json>

Where GRADE is exactly one of: "correct", "incorrect", "partial", or "almost"

No other text after the JSON."""

        # Get response from the model
        response, msg_history, _ = get_response_from_llm(
            msg=prompt,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from response
        prediction = self._extract_prediction(response)
        
        return str(prediction), msg_history

    def _extract_prediction(self, response: str) -> str:
        """Extract prediction from response with robust fallback strategies."""
        if not response or not isinstance(response, str):
            self.log_fn("Empty or invalid response, defaulting to incorrect")
            return "incorrect"
        
        response_text = response
        self.log_fn(f"Raw response text: {response_text[:500]}...")
        
        # Try extraction strategies
        prediction = None
        
        # Strategy 1: Extract from <json> tags
        try:
            json_results = _extract_jsons(response_text)
            if json_results:
                for item in json_results:
                    if isinstance(item, dict) and "response" in item:
                        val = item["response"].lower().strip()
                        # Map "almost" to "partial" for consistency
                        if val == "almost":
                            prediction = "partial"
                            self.log_fn(f"Extracted prediction via <json> tags (mapped 'almost' to 'partial'): {prediction}")
                            break
                        elif val in ["correct", "incorrect", "partial"]:
                            prediction = val
                            self.log_fn(f"Extracted prediction via <json> tags: {prediction}")
                            break
        except Exception as e:
            self.log_fn(f"Error in <json> extraction: {e}")
        
        # Strategy 2: Robust extraction
        if prediction is None:
            try:
                result = _extract_response_robust(response_text)
                if result:
                    prediction = result
                    self.log_fn(f"Extracted prediction via robust extraction: {prediction}")
            except Exception as e:
                self.log_fn(f"Error in robust extraction: {e}")
        
        # Strategy 3: Direct text search for grades in quotes
        if prediction is None:
            text_lower = response_text.lower()
            # Check for "almost" first (mapped to partial)
            if '"almost"' in text_lower or "'almost'" in text_lower or '`almost`' in text_lower:
                prediction = "partial"
                self.log_fn(f"Extracted prediction via 'almost' detection: {prediction}")
            else:
                # Look for the grade in quotes or backticks
                for grade in ["correct", "incorrect", "partial"]:
                    if f'"{grade}"' in text_lower or f"'{grade}'" in text_lower or f"`{grade}`" in text_lower:
                        prediction = grade
                        self.log_fn(f"Extracted prediction via quotes: {prediction}")
                        break
        
        # Strategy 4: Simple word boundary search
        if prediction is None:
            text_lower = response_text.lower()
            # Check for "almost" first (mapped to partial)
            if 'almost' in text_lower and 'incorrect' not in text_lower:
                prediction = "partial"
                self.log_fn(f"Extracted prediction via simple 'almost' text search: {prediction}")
            # Check for grades in order of preference
            elif '"correct"' in text_lower or "'correct'" in text_lower or '`correct`' in text_lower:
                prediction = "correct"
                self.log_fn(f"Extracted prediction via simple text search: {prediction}")
            elif '"partial"' in text_lower or "'partial'" in text_lower or '`partial`' in text_lower:
                prediction = "partial"
                self.log_fn(f"Extracted prediction via simple text search: {prediction}")
            elif '"incorrect"' in text_lower or "'incorrect'" in text_lower or '`incorrect`' in text_lower:
                prediction = "incorrect"
                self.log_fn(f"Extracted prediction via simple text search: {prediction}")
            elif 'correct' in text_lower and 'incorrect' not in text_lower and 'partial' not in text_lower and 'almost' not in text_lower:
                prediction = "correct"
                self.log_fn(f"Extracted prediction via simple text search: {prediction}")
            elif 'partial' in text_lower and 'incorrect' not in text_lower:
                prediction = "partial"
                self.log_fn(f"Extracted prediction via simple text search: {prediction}")
            elif 'incorrect' in text_lower:
                prediction = "incorrect"
                self.log_fn(f"Extracted prediction via simple text search: {prediction}")
        
        # Validate the prediction
        if prediction not in ["correct", "incorrect", "partial"]:
            self.log_fn(f"Invalid or missing prediction: '{prediction}', defaulting to incorrect")
            prediction = "incorrect"
        
        return prediction
