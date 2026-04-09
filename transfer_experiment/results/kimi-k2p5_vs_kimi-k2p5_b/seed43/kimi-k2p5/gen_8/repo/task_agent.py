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
    """Extract JSON objects from <json>...</json> blocks or raw JSON.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also tries to find raw JSON objects if tags are not present.
    """
    results = []
    search_from = 0
    
    # First try to find JSON in <json> tags
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
            # Try to find JSON object boundaries with nested brace handling
            try:
                json_start = inner.find("{")
                if json_start != -1:
                    # Count braces to find matching end
                    brace_count = 0
                    json_end = -1
                    for i, char in enumerate(inner[json_start:], start=json_start):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_end = i
                                break
                    if json_end != -1:
                        results.append(json.loads(inner[json_start:json_end+1]))
            except json.JSONDecodeError:
                continue
    
    # If no tagged JSON found, try to find raw JSON objects
    if not results:
        # Look for JSON objects with "response" field using a more robust pattern
        # Find all potential JSON starting points
        for match in re.finditer(r'\{[\s\n]*"', text):
            start_pos = match.start()
            # Try to parse from this position
            try:
                # Find the matching end brace
                brace_count = 0
                for i, char in enumerate(text[start_pos:]):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            try:
                                obj = json.loads(text[start_pos:start_pos+i+1])
                                if "response" in obj:
                                    results.append(obj)
                                    break  # Found a valid one, stop
                            except json.JSONDecodeError:
                                continue
            except Exception:
                continue
    
    return results or None


def _extract_response_direct(text: str) -> str | None:
    """Extract response by looking for keywords directly in text with improved patterns."""
    text_lower = text.lower()
    
    # Look for explicit classification statements with quotes - prioritize these
    # Higher priority = more reliable indicators
    patterns = [
        (r'"response"\s*:\s*"(correct|incorrect|partial)"', 1),  # Exact JSON format
        (r'"response"\s*:\s*\'(correct|incorrect|partial)\'', 1),  # Single quotes
        (r'<json>\s*\{[^}]*"response"\s*:\s*"(correct|incorrect|partial)"', 1),  # JSON block
        (r'response\s*[=:]\s*"(correct|incorrect|partial)"', 1),
        (r'response\s*[=:]\s*\'(correct|incorrect|partial)\'', 1),
        (r'classification[\s:]+"?(correct|incorrect|partial)"?', 2),
        (r'classify[\s:]+"?(correct|incorrect|partial)"?', 2),
        (r'response[\s:]+"?(correct|incorrect|partial)"?', 2),
        (r'grade[\s:]+"?(correct|incorrect|partial)"?', 2),
        (r'"?(correct|incorrect|partial)"?\s*is\s*the\s*correct\s*classification', 2),
        (r'the\s*answer\s*is\s*"?(correct|incorrect|partial)"?', 2),
        (r'should\s*be\s*classified\s*as\s*"?(correct|incorrect|partial)"?', 2),
        (r'classify\s*as\s*"?(correct|incorrect|partial)"?', 2),
        (r'\b(correct|incorrect|partial)\b[.!]?\s*$', 3),  # At end of text
        (r'\b(correct|incorrect|partial)\b\s*\n', 3),
    ]
    
    results = []
    for pattern, priority in patterns:
        match = re.search(pattern, text_lower)
        if match:
            results.append((match.group(1).lower(), priority, match.start()))
    
    # Sort by priority (lower is better), then by position (earlier is better)
    if results:
        results.sort(key=lambda x: (x[1], x[2]))
        return results[0][0]
    
    # Count occurrences of each label (excluding "not correct" etc)
    # Remove common negation patterns first
    cleaned_text = re.sub(r'\bnot\s+correct\b', '', text_lower)
    cleaned_text = re.sub(r'\bnot\s+partial\b', '', cleaned_text)
    cleaned_text = re.sub(r'\bnot\s+incorrect\b', '', cleaned_text)
    cleaned_text = re.sub(r'\bno\s+correct\b', '', cleaned_text)
    cleaned_text = re.sub(r'\bno\s+partial\b', '', cleaned_text)
    cleaned_text = re.sub(r'\bno\s+incorrect\b', '', cleaned_text)
    
    correct_count = len(re.findall(r'\bcorrect\b', cleaned_text))
    incorrect_count = len(re.findall(r'\bincorrect\b', text_lower))
    partial_count = len(re.findall(r'\bpartial\b', text_lower))
    almost_count = len(re.findall(r'\balmost\b', text_lower))
    wrong_count = len(re.findall(r'\bwrong\b', text_lower))
    
    # If only one appears, use that
    total = correct_count + incorrect_count + partial_count + almost_count + wrong_count
    if total > 0:
        if correct_count > 0 and incorrect_count == 0 and partial_count == 0 and almost_count == 0 and wrong_count == 0:
            return "correct"
        if (incorrect_count > 0 or wrong_count > 0) and correct_count == 0 and partial_count == 0 and almost_count == 0:
            return "incorrect"
        if partial_count > 0 and correct_count == 0 and incorrect_count == 0 and wrong_count == 0 and almost_count == 0:
            return "partial"
        if almost_count > 0 and correct_count == 0 and incorrect_count == 0 and partial_count == 0 and wrong_count == 0:
            return "partial"  # Treat "almost" as "partial"
    
    # Check for majority with strict threshold
    counts = [("correct", correct_count), ("incorrect", incorrect_count + wrong_count), 
              ("partial", partial_count + almost_count)]
    counts.sort(key=lambda x: x[1], reverse=True)
    if counts[0][1] > counts[1][1] + 1:  # Require clear majority
        return counts[0][0]
    
    return None


def _normalize_prediction(prediction: str) -> str:
    """Normalize prediction to one of the three valid labels."""
    pred_str = str(prediction).lower().strip()
    
    # Direct match
    if pred_str in ["correct", "incorrect", "partial"]:
        return pred_str
    
    # Handle "almost" as "partial"
    if pred_str == "almost":
        return "partial"
    
    # Check for exact word matches first
    words = pred_str.split()
    if "correct" in words and "incorrect" not in words:
        return "correct"
    if "incorrect" in words:
        return "incorrect"
    if "partial" in words or "almost" in words:
        return "partial"
    
    # Check for partial/almost first (more specific)
    if "partial" in pred_str or "almost" in pred_str:
        return "partial"
    
    # Check for incorrect/wrong/false (but not "not incorrect")
    if "incorrect" in pred_str or "wrong" in pred_str or "false" in pred_str:
        # Make sure it's not a negation like "not wrong"
        if "not wrong" not in pred_str and "not incorrect" not in pred_str:
            return "incorrect"
    
    # Check for correct (but not incorrect)
    if "correct" in pred_str and "incorrect" not in pred_str:
        return "correct"
    
    # Default fallback - be conservative
    return "incorrect"


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
        # Extract key fields from inputs for better prompting
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        instruction = f"""You are an expert mathematical olympiad grader. Evaluate the student's answer and classify it as exactly one of: "correct", "incorrect", or "partial".

## Problem Statement:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Classification Rules:

**"correct"** - Complete, correct solution with valid reasoning. Full marks (6-7 points).

**"incorrect"** - Fundamentally wrong with no valid mathematical work. No meaningful progress. 0 points.

**"partial"** - Meaningful progress with correct lemmas or intermediate results, but incomplete. 1-5 points.

## Decision Process:
1. Check grading guidelines first - they are the primary authority
2. If student has ANY correct lemma or valid intermediate result → "partial"
3. If complete and correct → "correct"
4. If no valid work at all → "incorrect"

IMPORTANT: When in doubt between "incorrect" and "partial", choose "incorrect" unless there is CLEAR evidence of meaningful progress.

## Your Response:
You MUST respond with ONLY a JSON object in this exact format:
<json>
{{
    "response": "correct"
}}
</json>

OR

<json>
{{
    "response": "incorrect"
}}
</json>

OR

<json>
{{
    "response": "partial"
}}
</json>

Replace "correct" with your actual classification. Do not include any other text before or after the JSON block."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "incorrect"  # Default fallback - be conservative
        try:
            response_text = msg_history[-1]["text"]
            
            # First try: extract from <json> tags
            extracted = _extract_jsons(response_text)
            
            if extracted and isinstance(extracted, list) and len(extracted) > 0:
                last_json = extracted[-1]
                if isinstance(last_json, dict) and "response" in last_json:
                    prediction = _normalize_prediction(last_json["response"])
                    self.log_fn(f"Extracted prediction from JSON: {prediction}")
                else:
                    self.log_fn(f"JSON found but no 'response' field: {last_json}")
                    # Try direct extraction as fallback
                    direct = _extract_response_direct(response_text)
                    if direct:
                        prediction = _normalize_prediction(direct)
                        self.log_fn(f"Extracted prediction via direct extraction: {prediction}")
            else:
                # Try direct extraction
                direct = _extract_response_direct(response_text)
                if direct:
                    prediction = _normalize_prediction(direct)
                    self.log_fn(f"Extracted prediction via direct extraction: {prediction}")
                else:
                    self.log_fn(f"No valid JSON found in response: {response_text[:500]}...")
                    # Last resort: look for the words directly in the text
                    text_lower = response_text.lower()
                    if '"correct"' in text_lower or "'correct'" in text_lower or '"response": "correct"' in text_lower:
                        prediction = "correct"
                    elif '"incorrect"' in text_lower or "'incorrect'" in text_lower or '"response": "incorrect"' in text_lower:
                        prediction = "incorrect"
                    elif '"partial"' in text_lower or "'partial'" in text_lower or '"response": "partial"' in text_lower:
                        prediction = "partial"
                    self.log_fn(f"Last resort extraction: {prediction}")
                
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            import traceback
            self.log_fn(traceback.format_exc())

        return str(prediction), msg_history
