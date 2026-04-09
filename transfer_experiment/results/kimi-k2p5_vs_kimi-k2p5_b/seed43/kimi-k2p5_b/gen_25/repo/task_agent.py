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


def _is_valid_label(text: str) -> bool:
    """Check if text is a valid classification label (not math/problem content)."""
    if not text:
        return False
    
    text_str = str(text).strip()
    text_lower = text_str.lower()
    
    # If it's exactly one of the four labels, it's valid
    if text_lower in ["correct", "incorrect", "partial", "almost"]:
        return True
    
    # If it contains one of the four labels as a standalone word, it's likely valid
    for label in ["almost", "partial", "incorrect", "correct"]:
        if re.search(rf'\b{label}\b', text_lower):
            return True
    
    # Check for mathematical expressions or problem content (invalid predictions)
    # Only reject if it looks like actual math content
    math_indicators = [
        r'\$\$.*?\$\$',  # LaTeX display math
        r'\\[a-zA-Z]+',   # LaTeX commands
        r'<step',         # Step tags
    ]
    
    math_count = 0
    for pattern in math_indicators:
        if re.search(pattern, text_str):
            math_count += 1
    
    # If strong math indicators found, it's likely math content
    if math_count >= 2:
        return False
    
    # Check for very long text (labels should be short)
    if len(text_str) > 500:
        return False
    
    return True


def _extract_json_from_text(text: str) -> dict | None:
    """Extract JSON object from text using multiple strategies."""
    text = text.strip()
    
    # Strategy 1: <json>...</json> tags
    json_tag_pattern = r'<json>\s*(.*?)\s*</json>'
    matches = re.findall(json_tag_pattern, text, re.DOTALL | re.IGNORECASE)
    for match in matches:
        try:
            data = json.loads(match.strip())
            if isinstance(data, dict) and "response" in data:
                return data
        except json.JSONDecodeError:
            fixed = _fix_json(match.strip())
            if fixed:
                try:
                    data = json.loads(fixed)
                    if isinstance(data, dict) and "response" in data:
                        return data
                except json.JSONDecodeError:
                    pass
    
    # Strategy 2: ```json...``` code blocks
    code_block_pattern = r'```json\s*(.*?)\s*```'
    matches = re.findall(code_block_pattern, text, re.DOTALL | re.IGNORECASE)
    for match in matches:
        try:
            data = json.loads(match.strip())
            if isinstance(data, dict) and "response" in data:
                return data
        except json.JSONDecodeError:
            fixed = _fix_json(match.strip())
            if fixed:
                try:
                    data = json.loads(fixed)
                    if isinstance(data, dict) and "response" in data:
                        return data
                except json.JSONDecodeError:
                    pass
    
    # Strategy 3: ```...``` code blocks with JSON content
    code_block_pattern2 = r'```\s*(\{[\s\S]*?\})\s*```'
    matches = re.findall(code_block_pattern2, text, re.DOTALL | re.IGNORECASE)
    for match in matches:
        try:
            data = json.loads(match.strip())
            if isinstance(data, dict) and "response" in data:
                return data
        except json.JSONDecodeError:
            fixed = _fix_json(match.strip())
            if fixed:
                try:
                    data = json.loads(fixed)
                    if isinstance(data, dict) and "response" in data:
                        return data
                except json.JSONDecodeError:
                    pass
    
    # Strategy 4: Raw JSON objects
    json_pattern = r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}'
    matches = re.findall(json_pattern, text, re.DOTALL)
    for match in matches:
        try:
            data = json.loads(match)
            if isinstance(data, dict) and "response" in data:
                return data
        except json.JSONDecodeError:
            fixed = _fix_json(match)
            if fixed:
                try:
                    data = json.loads(fixed)
                    if isinstance(data, dict) and "response" in data:
                        return data
                except json.JSONDecodeError:
                    pass
    
    return None


def _fix_json(text: str) -> str | None:
    """Try to fix common JSON formatting issues."""
    if not text or not text.strip():
        return None
    
    text = text.strip()
    
    # Fix 1: Remove trailing commas before closing braces
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*\]', ']', text)
    
    # Fix 2: Replace single quotes with double quotes (simple cases)
    if "'" in text and '"' not in text:
        text = text.replace("'", '"')
    
    # Fix 3: Ensure the text starts with { and ends with }
    if not text.startswith('{'):
        start = text.find('{')
        if start != -1:
            text = text[start:]
    
    if not text.endswith('}'):
        end = text.rfind('}')
        if end != -1:
            text = text[:end+1]
    
    return text if text.startswith('{') and text.endswith('}') else None


def _extract_response_direct(text: str) -> str | None:
    """Extract response by looking for keywords directly in text."""
    text_lower = text.lower()
    
    # Look for explicit classification statements with priority
    # Priority: almost > partial > incorrect > correct (most specific first)
    
    # Check for "almost" first (most specific)
    almost_patterns = [
        r'"response"\s*:\s*"almost"',
        r'"classification"\s*:\s*"almost"',
        r'"grade"\s*:\s*"almost"',
        r'"result"\s*:\s*"almost"',
        r'"evaluation"\s*:\s*"almost"',
        r'\bclassif(y|ied)\s+as\s+almost\b',
        r'\bthis\s+is\s+almost\b',
    ]
    for pattern in almost_patterns:
        if re.search(pattern, text_lower):
            return "almost"
    
    # Check for "partial"
    partial_patterns = [
        r'"response"\s*:\s*"partial"',
        r'"classification"\s*:\s*"partial"',
        r'"grade"\s*:\s*"partial"',
        r'"result"\s*:\s*"partial"',
        r'"evaluation"\s*:\s*"partial"',
        r'\bclassif(y|ied)\s+as\s+partial\b',
        r'\bthis\s+is\s+partial\b',
    ]
    for pattern in partial_patterns:
        if re.search(pattern, text_lower):
            return "partial"
    
    # Check for "incorrect"
    incorrect_patterns = [
        r'"response"\s*:\s*"incorrect"',
        r'"classification"\s*:\s*"incorrect"',
        r'"grade"\s*:\s*"incorrect"',
        r'"result"\s*:\s*"incorrect"',
        r'"evaluation"\s*:\s*"incorrect"',
        r'\bclassif(y|ied)\s+as\s+incorrect\b',
        r'\bthis\s+is\s+incorrect\b',
    ]
    for pattern in incorrect_patterns:
        if re.search(pattern, text_lower):
            return "incorrect"
    
    # Check for "correct"
    correct_patterns = [
        r'"response"\s*:\s*"correct"',
        r'"classification"\s*:\s*"correct"',
        r'"grade"\s*:\s*"correct"',
        r'"result"\s*:\s*"correct"',
        r'"evaluation"\s*:\s*"correct"',
        r'\bclassif(y|ied)\s+as\s+correct\b',
        r'\bthis\s+is\s+correct\b',
    ]
    for pattern in correct_patterns:
        if re.search(pattern, text_lower):
            return "correct"
    
    # Count standalone occurrences as fallback
    almost_count = len(re.findall(r'\balmost\b', text_lower))
    partial_count = len(re.findall(r'\bpartial\b', text_lower))
    incorrect_count = len(re.findall(r'\bincorrect\b', text_lower))
    correct_count = len(re.findall(r'\bcorrect\b', text_lower))
    
    # If only one appears, use that
    if almost_count > 0 and partial_count == 0 and incorrect_count == 0 and correct_count == 0:
        return "almost"
    if partial_count > 0 and almost_count == 0 and incorrect_count == 0 and correct_count == 0:
        return "partial"
    if incorrect_count > 0 and almost_count == 0 and partial_count == 0 and correct_count == 0:
        return "incorrect"
    if correct_count > 0 and almost_count == 0 and partial_count == 0 and incorrect_count == 0:
        return "correct"
    
    # Priority-based selection when multiple appear
    if almost_count > 0:
        return "almost"
    if partial_count > 0:
        return "partial"
    if incorrect_count > 0:
        return "incorrect"
    if correct_count > 0:
        return "correct"
    
    return None


def _normalize_prediction(prediction: str) -> str:
    """Normalize prediction to one of the four valid labels."""
    pred_str = str(prediction).lower().strip()
    
    # Direct match
    if pred_str in ["correct", "incorrect", "partial", "almost"]:
        return pred_str
    
    # Check if it's a valid label at all
    if not _is_valid_label(prediction):
        return "incorrect"
    
    # Priority: almost > partial > incorrect > correct
    if "almost" in pred_str:
        return "almost"
    if "partial" in pred_str:
        return "partial"
    if "incorrect" in pred_str:
        return "incorrect"
    if "correct" in pred_str:
        return "correct"
    
    return "incorrect"


def _check_grading_guidelines_for_almost(grading_guidelines: str) -> bool:
    """Check if grading guidelines strongly indicate 'almost' level.
    
    Only returns True if the guidelines EXPLICITLY indicate almost level
    through point values or clear statements, not just presence of markers.
    """
    if not grading_guidelines:
        return False
    
    guidelines_lower = grading_guidelines.lower()
    
    # Check for explicit point indicators of "almost" level (5-6 points)
    almost_point_indicators = [
        r'\b5[-\s]?6\s+points\b',
        r'\b6\s+points\b',
        r'\b5\s+points\b',
    ]
    
    for pattern in almost_point_indicators:
        if re.search(pattern, guidelines_lower):
            return True
    
    # Check for explicit (Almost) at the START of the guidelines (not just anywhere)
    # This indicates the PRIMARY classification is almost
    lines = guidelines_lower.split('\n')
    for line in lines[:3]:  # Check first 3 lines only
        if re.search(r'^\s*\(almost\)', line):
            return True
    
    return False


def _check_grading_guidelines_for_partial(grading_guidelines: str) -> bool:
    """Check if grading guidelines strongly indicate 'partial' level.
    
    Only returns True if the guidelines EXPLICITLY indicate partial level
    through point values (1-4 points), not just presence of (Partial) markers.
    """
    if not grading_guidelines:
        return False
    
    guidelines_lower = grading_guidelines.lower()
    
    # Check for explicit point indicators of "partial" level (1-4 points)
    partial_point_indicators = [
        r'\b1[-\s]?4\s+points\b',
        r'\b2[-\s]?3\s+points\b',
        r'\b3[-\s]?4\s+points\b',
        r'\b4\s+points\b',
        r'\b3\s+points\b',
        r'\b2\s+points\b',
        r'\b1\s+point\b',
    ]
    
    for pattern in partial_point_indicators:
        if re.search(pattern, guidelines_lower):
            return True
    
    return False


def _check_grading_guidelines_for_incorrect(grading_guidelines: str) -> bool:
    """Check if grading guidelines strongly indicate 'incorrect' level.
    
    Only returns True if the guidelines EXPLICITLY indicate incorrect level
    through point values (0 points) or clear statements of no progress.
    """
    if not grading_guidelines:
        return False
    
    guidelines_lower = grading_guidelines.lower()
    
    # Check for explicit 0 points indicator
    if re.search(r'\b0\s+points\b', guidelines_lower):
        return True
    
    # Check for explicit (Incorrect) at the START of the guidelines
    lines = guidelines_lower.split('\n')
    for line in lines[:3]:  # Check first 3 lines only
        if re.search(r'^\s*\(incorrect\)', line):
            return True
    
    # Check for strong indicators of no valid progress
    if re.search(r'\bno\s+valid\s+lemmas?\b', guidelines_lower) and \
       re.search(r'\bno\s+non-trivial\s+observations?\b', guidelines_lower):
        return True
    
    return False


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem."""
        # Extract key fields from inputs
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        instruction = f"""You are an expert mathematical olympiad grader. Your task is to classify the student's answer into EXACTLY ONE of four categories: "correct", "almost", "partial", or "incorrect".

## Problem Statement:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Classification Rules (READ CAREFULLY):

**"correct"** (7 points): Complete solution with valid reasoning throughout, correct final answer, no gaps. The solution is essentially perfect.

**"almost"** (5-6 points): Nearly complete solution. Main proof structure is sound. Only minor technical errors (NOT conceptual gaps). The solution is VERY CLOSE to complete - just needs small fixes. The student demonstrated strong understanding with only minor slips.

**"partial"** (1-4 points): Meaningful progress (valid lemmas, right approach) but SIGNIFICANT gaps remain. The solution is far from complete. Student made some real progress but major work is missing.

**"incorrect"** (0 points): No valid lemmas or non-trivial observations. Fundamentally flawed or trivial only. No meaningful mathematical progress.

## Decision Process (FOLLOW THIS ORDER):
1. Is the solution essentially complete with only minor issues? → "almost"
2. Is the solution completely correct with no gaps? → "correct"  
3. Did the student make meaningful progress with valid lemmas? → "partial"
4. Otherwise → "incorrect"

## CRITICAL DISTINCTIONS:

**"correct" vs "almost":**
- "correct" = Perfect or near-perfect (7 points)
- "almost" = Nearly complete, minor technical errors only (5-6 points)

**"almost" vs "partial":**
- "almost" = Solution is nearly there, main structure is correct, just needs small fixes
- "partial" = Solution has significant gaps, far from complete despite some valid progress

**"partial" vs "incorrect":**
- "partial" = Student proved at least one valid lemma or made meaningful progress
- "incorrect" = No valid mathematical progress at all

## OUTPUT FORMAT (STRICT JSON - MANDATORY):
You MUST output ONLY a JSON object. The response field MUST contain EXACTLY ONE of these four words: "correct", "almost", "partial", or "incorrect". NO OTHER TEXT is allowed in the response field.

<json>
{{
    "reasoning": "Your detailed analysis here. Be specific about what the student achieved and what is missing.",
    "response": "correct"
}}
</json>

Replace "correct" with "almost", "partial", or "incorrect" as appropriate."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from response
        prediction = "incorrect"  # Default fallback
        try:
            if msg_history and len(msg_history) > 0:
                response_text = msg_history[-1].get("text", "")
                
                if response_text:
                    # First try: extract from JSON
                    json_data = _extract_json_from_text(response_text)
                    
                    if json_data and isinstance(json_data, dict):
                        # Look for response field first
                        if "response" in json_data:
                            prediction = _normalize_prediction(json_data["response"])
                        else:
                            # Check other common fields
                            for field in ["classification", "grade", "result", "evaluation", "verdict", "outcome"]:
                                if field in json_data:
                                    prediction = _normalize_prediction(json_data[field])
                                    break
                    else:
                        # Try direct extraction from text
                        direct = _extract_response_direct(response_text)
                        if direct:
                            prediction = direct
                        else:
                            self.log_fn(f"No valid response found in: {response_text[:200]}...")
                else:
                    self.log_fn("Empty response text in msg_history")
            else:
                self.log_fn("Empty msg_history")
                
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            import traceback
            self.log_fn(traceback.format_exc())

        # Post-processing: Check grading guidelines for adjustments
        # Only adjust DOWNWARD (more negative), never UPWARD to avoid bias
        # NOTE: We only adjust if the guidelines EXPLICITLY contradict the prediction
        if prediction in ["correct", "almost", "partial", "incorrect"]:
            # Check for "incorrect" indicators - only if prediction is positive
            if prediction in ["correct", "almost", "partial"]:
                if _check_grading_guidelines_for_incorrect(grading_guidelines):
                    self.log_fn("Adjusting prediction to 'incorrect' based on grading guidelines")
                    prediction = "incorrect"
            
            # Check for "partial" indicators - only if prediction is "correct" or "almost"
            if prediction in ["correct", "almost"]:
                if _check_grading_guidelines_for_partial(grading_guidelines):
                    self.log_fn("Adjusting prediction to 'partial' based on grading guidelines")
                    prediction = "partial"
            
            # Check for "almost" indicators - only if prediction is "correct"
            if prediction == "correct":
                if _check_grading_guidelines_for_almost(grading_guidelines):
                    self.log_fn("Adjusting prediction to 'almost' based on grading guidelines")
                    prediction = "almost"

        # Final validation
        if prediction not in ["correct", "incorrect", "partial", "almost"]:
            prediction = "incorrect"

        return str(prediction), msg_history
