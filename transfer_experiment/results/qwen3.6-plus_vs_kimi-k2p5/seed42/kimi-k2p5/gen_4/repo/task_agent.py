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
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            continue
    return results or None


def _extract_json_flexible(text: str) -> list[dict] | None:
    """Extract JSON objects using multiple strategies.
    
    Tries multiple patterns with improved robustness for edge cases.
    """
    results = []
    text_stripped = text.strip()
    
    # Strategy 1: <json>...</json> tags
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
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # Try to fix common JSON issues
            try:
                # Replace single quotes with double quotes
                fixed = inner.replace("'", '"')
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                continue
    
    # Strategy 2: ```json...``` code blocks
    pattern = r'```json\s*(.*?)\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            # Try to fix common JSON issues
            try:
                fixed = match.strip().replace("'", '"')
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                continue
    
    # Strategy 3: ```...``` code blocks (without json label)
    pattern = r'```\s*(\{[\s\S]*?\})\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            try:
                fixed = match.strip().replace("'", '"')
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                continue
    
    # Strategy 4: Raw JSON objects with "response" field (non-greedy)
    # Look for JSON-like structures: {"response": "..."}
    pattern = r'\{\s*"response"\s*:\s*"([^"]+)"\s*\}'
    matches = re.findall(pattern, text)
    for match in matches:
        results.append({"response": match})
    
    # Strategy 5: Single-quoted JSON-like structures
    pattern = r"\{\s*'response'\s*:\s*'([^']+)'\s*\}"
    matches = re.findall(pattern, text)
    for match in matches:
        results.append({"response": match})
    
    # Strategy 6: Case-insensitive response extraction with flexible quotes
    pattern = r'["\']?response["\']?\s*:\s*["\']?(correct|incorrect|partial)["\']?'
    matches = re.findall(pattern, text, re.IGNORECASE)
    for match in matches:
        results.append({"response": match.lower()})
    
    # Strategy 7: Look for JSON objects at the start or end of text
    for candidate in [text_stripped.split('\n')[0], text_stripped.split('\n')[-1], text_stripped]:
        try:
            if candidate.startswith('{') and candidate.endswith('}'):
                results.append(json.loads(candidate))
        except json.JSONDecodeError:
            try:
                if candidate.startswith('{') and candidate.endswith('}'):
                    fixed = candidate.replace("'", '"')
                    results.append(json.loads(fixed))
            except json.JSONDecodeError:
                continue
    
    return results or None


def _extract_response_direct(text: str) -> str | None:
    """Direct extraction of response value from text.
    
    Looks for explicit mentions of the grade in the text with improved patterns.
    """
    text_lower = text.lower()
    
    # Check for explicit grade mentions in quotes
    if '"correct"' in text_lower or "'correct'" in text_lower:
        return "correct"
    if '"incorrect"' in text_lower or "'incorrect'" in text_lower:
        return "incorrect"
    if '"partial"' in text_lower or "'partial'" in text_lower:
        return "partial"
    
    # Check for grade at end of text or in conclusion
    lines = text_lower.strip().split('\n')
    for line in reversed(lines):
        line = line.strip().rstrip('}').rstrip(',').strip()
        if line in ['correct', 'incorrect', 'partial']:
            return line
        # Check for patterns like "grade: correct" or "decision: partial"
        for grade in ['correct', 'incorrect', 'partial']:
            if grade in line and any(marker in line for marker in ['grade', 'decision', 'verdict', 'result', 'evaluation']):
                return grade
    
    # Check for grade at the beginning
    first_line = lines[0].strip().lstrip('{').strip() if lines else ""
    for grade in ['correct', 'incorrect', 'partial']:
        if first_line.startswith(grade) or first_line == grade:
            return grade
    
    return None


def _extract_response_llm_style(text: str) -> str | None:
    """Extract response using patterns common in LLM outputs.
    
    Handles various formatting styles that LLMs commonly use.
    """
    text_lower = text.lower().strip()
    
    # Pattern 1: Look for "The answer is X" or "Therefore, the answer is X"
    conclusion_patterns = [
        rf'the answer is\s+(correct|incorrect|partial)',
        rf'therefore,?\s+(?:the\s+)?answer is\s+(correct|incorrect|partial)',
        rf'final answer[:\s]+(correct|incorrect|partial)',
        rf'conclusion[:\s]+(correct|incorrect|partial)',
        rf'grading[:\s]+(correct|incorrect|partial)',
        rf'assessment[:\s]+(correct|incorrect|partial)',
    ]
    
    for pattern in conclusion_patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1)
    
    # Pattern 2: Look for standalone grade on its own line
    lines = text_lower.split('\n')
    for line in lines:
        line_clean = line.strip().rstrip('}').rstrip(',').rstrip('.').rstrip('"').rstrip("'").strip()
        if line_clean in ['correct', 'incorrect', 'partial']:
            return line_clean
    
    # Pattern 3: Look for grade in parentheses or brackets
    bracket_patterns = [
        rf'\((correct|incorrect|partial)\)',
        rf'\[(correct|incorrect|partial)\]',
        rf'\{{(correct|incorrect|partial)\}}',
    ]
    for pattern in bracket_patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1)
    
    return None


def _extract_response_robust(text: str) -> str | None:
    """Robust extraction using multiple pattern matching strategies.
    
    This function tries to find the grade in various formats with priority ordering.
    """
    text_lower = text.lower()
    stripped = text_lower.strip()
    
    # Priority 1: Look for JSON-like patterns with response field
    json_patterns = [
        r'"response"\s*:\s*"(correct|incorrect|partial)"',
        r"'response'\s*:\s*'(correct|incorrect|partial)'",
        r'response\s*:\s*(correct|incorrect|partial)',
    ]
    for pattern in json_patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1)
    
    # Priority 2: Look for the grade as a standalone word at the end of the text
    for grade in ['correct', 'incorrect', 'partial']:
        if stripped.rstrip('}').rstrip().endswith(grade):
            return grade
    
    # Priority 3: Look for grade in quotes (most reliable indicator)
    for grade in ['correct', 'incorrect', 'partial']:
        if f'"{grade}"' in text_lower or f"'{grade}'" in text_lower:
            return grade
    
    # Priority 4: Look for grade at start of text
    for grade in ['correct', 'incorrect', 'partial']:
        if stripped.lstrip('{').lstrip().startswith(grade):
            return grade
    
    # Priority 5: Look for explicit decision patterns
    decision_patterns = [
        rf'grade\s*[:=]\s*(correct|incorrect|partial)',
        rf'decision\s*[:=]\s*(correct|incorrect|partial)',
        rf'verdict\s*[:=]\s*(correct|incorrect|partial)',
        rf'answer\s*is\s*(correct|incorrect|partial)',
        rf'result\s*[:=]\s*(correct|incorrect|partial)',
    ]
    for pattern in decision_patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1)
    
    # Priority 6: Count word-boundary occurrences and pick the most frequent valid grade
    counts = {}
    for grade in ['correct', 'incorrect', 'partial']:
        counts[grade] = len(re.findall(rf'\b{grade}\b', text_lower))
    
    if counts and max(counts.values()) > 0:
        max_count = max(counts.values())
        # Return the grade with highest count
        for grade, count in counts.items():
            if count == max_count:
                return grade
    
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
        # Extract fields from inputs for clearer prompting
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        instruction = f"""You are an expert mathematics grader for International Mathematical Olympiad (IMO) problems.

Your task is to evaluate a student's answer to a mathematics problem and assign EXACTLY ONE of these three grades:
- "correct": The student's answer is completely correct, fully proves the statement, or completely solves the problem. All logical steps are valid and the conclusion is correct.
- "incorrect": The student's answer contains critical errors, logical fallacies, or fails to prove/solve the problem. The approach is fundamentally flawed or the conclusion is wrong.
- "partial": The student's answer demonstrates valid partial progress, contains correct lemmas, shows meaningful insights, or has the right approach but is incomplete or contains minor errors that don't invalidate the core reasoning.

GRADING CRITERIA:
1. Check if the student's answer matches the official solution's approach or achieves the same result through valid alternative reasoning.
2. "correct" requires a complete, valid proof or solution with no significant gaps.
3. "partial" is for answers with good ideas but missing pieces, or answers that are on the right track but incomplete.
4. "incorrect" is for answers with fundamental errors, wrong conclusions, or approaches that cannot work.

Problem:
{problem}

Official Solution:
{solution}

Grading Guidelines:
{grading_guidelines}

Student's Answer:
{student_answer}

Analyze the student's answer step by step:
1. Identify the key claims and logical structure
2. Compare with the official solution
3. Check for valid partial progress or correct sub-results
4. Determine the final grade based on the criteria above

You MUST respond with ONLY a JSON object in this exact format (no other text):
{{"response": "correct"}} or {{"response": "incorrect"}} or {{"response": "partial"}}

Do not include any explanation, markdown formatting, or additional text. Only output the JSON object."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from response using multiple strategies
        prediction = None
        response_text = ""
        
        # Get the response text from msg_history or use the direct response
        try:
            if msg_history and len(msg_history) > 0:
                if isinstance(msg_history[-1], dict) and "content" in msg_history[-1]:
                    response_text = msg_history[-1]["content"]
                elif isinstance(msg_history[-1], dict) and "text" in msg_history[-1]:
                    response_text = msg_history[-1]["text"]
                elif isinstance(msg_history[-1], str):
                    response_text = msg_history[-1]
            else:
                response_text = response if isinstance(response, str) else str(response)
        except Exception as e:
            self.log_fn(f"Error getting response text: {e}")
            response_text = str(response) if response else ""
        
        self.log_fn(f"Raw response text: {response_text[:500]}...")
        
        # Try multiple extraction strategies in order of reliability
        
        # Strategy 1: Robust extraction (highest priority - handles most formats)
        try:
            result = _extract_response_robust(response_text)
            if result:
                prediction = result
                self.log_fn(f"Extracted prediction via robust extraction: {prediction}")
        except Exception as e:
            self.log_fn(f"Error in robust extraction: {e}")
        
        # Strategy 2: Flexible JSON extraction
        if prediction is None:
            try:
                extracted = _extract_json_flexible(response_text)
                if extracted:
                    for item in extracted:
                        if isinstance(item, dict) and "response" in item:
                            val = item["response"]
                            if isinstance(val, str) and val.lower() in ["correct", "incorrect", "partial"]:
                                prediction = val.lower()
                                self.log_fn(f"Extracted prediction via JSON: {prediction}")
                                break
            except Exception as e:
                self.log_fn(f"Error in flexible JSON extraction: {e}")
        
        # Strategy 3: Direct response extraction
        if prediction is None:
            try:
                direct = _extract_response_direct(response_text)
                if direct:
                    prediction = direct
                    self.log_fn(f"Extracted prediction via direct extraction: {prediction}")
            except Exception as e:
                self.log_fn(f"Error in direct extraction: {e}")
        
        # Strategy 4: LLM-style extraction (handles common LLM output patterns)
        if prediction is None:
            try:
                llm_result = _extract_response_llm_style(response_text)
                if llm_result:
                    prediction = llm_result
                    self.log_fn(f"Extracted prediction via LLM-style extraction: {prediction}")
            except Exception as e:
                self.log_fn(f"Error in LLM-style extraction: {e}")
        
        # Strategy 5: Legacy extraction as fallback
        if prediction is None:
            try:
                extracted = _extract_jsons(response_text)
                if extracted and len(extracted) > 0:
                    last = extracted[-1]
                    if isinstance(last, dict) and "response" in last:
                        val = last["response"]
                        if val in ["correct", "incorrect", "partial"]:
                            prediction = val
                            self.log_fn(f"Extracted prediction via legacy JSON: {prediction}")
            except Exception as e:
                self.log_fn(f"Error in legacy extraction: {e}")
        
        # Validate the prediction
        if prediction not in ["correct", "incorrect", "partial"]:
            self.log_fn(f"Invalid or missing prediction: '{prediction}', defaulting to incorrect")
            prediction = "incorrect"
        
        return str(prediction), msg_history
