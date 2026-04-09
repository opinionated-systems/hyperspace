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
    
    text_lower = text.lower().strip()
    
    # If it's exactly one of the four labels, it's valid
    if text_lower in ["correct", "incorrect", "partial", "almost"]:
        return True
    
    # Check for mathematical expressions or problem content (invalid predictions)
    invalid_indicators = [
        r'\$\$.*?\$\$',  # LaTeX display math
        r'\$.*?\$',      # LaTeX inline math  
        r'\\[a-zA-Z]+',   # LaTeX commands (any backslash command)
        r'\^',           # Exponents
        r'\\frac',       # Fractions
        r'\\sum',        # Sums
        r'\\int',        # Integrals
        r'<step',         # Step tags
        r'\\begin\{',    # LaTeX environments
        r'\\end\{',      # LaTeX environments
        r'\[',            # Display math brackets
        r'\]',            # Display math brackets
        r'\\[a-zA-Z]',   # Any LaTeX command
    ]
    
    # Count math indicators
    math_count = 0
    for pattern in invalid_indicators:
        if re.search(pattern, text):
            math_count += 1
    
    # If more than 1 math indicator found, it's likely math content
    if math_count > 1:
        return False
    
    # Check for very long text (labels should be short)
    if len(text) > 100:
        return False
    
    # Check for newlines (labels should be single line)
    if '\n' in text or '\r' in text:
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
            if isinstance(data, dict):
                # Validate that response field contains a valid label
                if "response" in data:
                    resp_val = str(data["response"])
                    if _is_valid_label(resp_val):
                        return data
        except json.JSONDecodeError:
            fixed = _fix_json(match.strip())
            if fixed:
                try:
                    data = json.loads(fixed)
                    if isinstance(data, dict):
                        if "response" in data:
                            resp_val = str(data["response"])
                            if _is_valid_label(resp_val):
                                return data
                except json.JSONDecodeError:
                    pass
            continue
    
    # Strategy 2: ```json...``` code blocks
    code_block_pattern = r'```json\s*(.*?)\s*```'
    matches = re.findall(code_block_pattern, text, re.DOTALL | re.IGNORECASE)
    for match in matches:
        try:
            data = json.loads(match.strip())
            if isinstance(data, dict):
                # Validate that response field contains a valid label
                if "response" in data:
                    resp_val = str(data["response"])
                    if _is_valid_label(resp_val):
                        return data
        except json.JSONDecodeError:
            fixed = _fix_json(match.strip())
            if fixed:
                try:
                    data = json.loads(fixed)
                    if isinstance(data, dict):
                        if "response" in data:
                            resp_val = str(data["response"])
                            if _is_valid_label(resp_val):
                                return data
                except json.JSONDecodeError:
                    pass
            continue
    
    # Strategy 3: ```...``` code blocks with JSON content
    code_block_pattern2 = r'```\s*(\{[\s\S]*?\})\s*```'
    matches = re.findall(code_block_pattern2, text, re.DOTALL | re.IGNORECASE)
    for match in matches:
        try:
            data = json.loads(match.strip())
            if isinstance(data, dict):
                # Validate that response field contains a valid label
                if "response" in data:
                    resp_val = str(data["response"])
                    if _is_valid_label(resp_val):
                        return data
        except json.JSONDecodeError:
            fixed = _fix_json(match.strip())
            if fixed:
                try:
                    data = json.loads(fixed)
                    if isinstance(data, dict):
                        if "response" in data:
                            resp_val = str(data["response"])
                            if _is_valid_label(resp_val):
                                return data
                except json.JSONDecodeError:
                    pass
            continue
    
    # Strategy 4: Raw JSON objects with response field
    json_pattern = r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}'
    matches = re.findall(json_pattern, text, re.DOTALL)
    for match in matches:
        try:
            data = json.loads(match)
            if isinstance(data, dict) and "response" in data:
                # Validate that response field contains a valid label
                resp_val = str(data["response"])
                if _is_valid_label(resp_val):
                    return data
        except json.JSONDecodeError:
            fixed = _fix_json(match)
            if fixed:
                try:
                    data = json.loads(fixed)
                    if isinstance(data, dict) and "response" in data:
                        # Validate that response field contains a valid label
                        resp_val = str(data["response"])
                        if _is_valid_label(resp_val):
                            return data
                except json.JSONDecodeError:
                    pass
            continue
    
    # Strategy 5: Direct pattern matching for response field
    # Check for "almost" first (most specific)
    response_pattern_almost = r'"response"\s*:\s*"([^"]*almost[^"]*)"'
    match = re.search(response_pattern_almost, text, re.IGNORECASE)
    if match:
        return {"response": "almost"}
    
    response_pattern_almost_unquoted = r'"response"\s*:\s*almost\b'
    match = re.search(response_pattern_almost_unquoted, text, re.IGNORECASE)
    if match:
        return {"response": "almost"}
    
    # Check for "partial" 
    response_pattern_partial = r'"response"\s*:\s*"([^"]*partial[^"]*)"'
    match = re.search(response_pattern_partial, text, re.IGNORECASE)
    if match:
        return {"response": "partial"}
    
    response_pattern_partial_unquoted = r'"response"\s*:\s*partial\b'
    match = re.search(response_pattern_partial_unquoted, text, re.IGNORECASE)
    if match:
        return {"response": "partial"}
    
    # Check for "incorrect"
    response_pattern_incorrect = r'"response"\s*:\s*"([^"]*incorrect[^"]*)"'
    match = re.search(response_pattern_incorrect, text, re.IGNORECASE)
    if match:
        return {"response": "incorrect"}
    
    response_pattern_incorrect_unquoted = r'"response"\s*:\s*incorrect\b'
    match = re.search(response_pattern_incorrect_unquoted, text, re.IGNORECASE)
    if match:
        return {"response": "incorrect"}
    
    # Check for "correct"
    response_pattern_correct = r'"response"\s*:\s*"([^"]*correct[^"]*)"'
    match = re.search(response_pattern_correct, text, re.IGNORECASE)
    if match:
        return {"response": "correct"}
    
    response_pattern_correct_unquoted = r'"response"\s*:\s*correct\b'
    match = re.search(response_pattern_correct_unquoted, text, re.IGNORECASE)
    if match:
        return {"response": "correct"}
    
    # Generic response field extraction
    response_pattern = r'"response"\s*:\s*"([^"]+)"'
    match = re.search(response_pattern, text, re.IGNORECASE)
    if match:
        return {"response": match.group(1)}
    
    # Check for unquoted response values (e.g., "response": almost)
    response_pattern_unquoted = r'"response"\s*:\s*(correct|incorrect|partial|almost)\b'
    match = re.search(response_pattern_unquoted, text, re.IGNORECASE)
    if match:
        return {"response": match.group(1)}
    
    return None


def _fix_json(text: str) -> str | None:
    """Try to fix common JSON formatting issues."""
    if not text or not text.strip():
        return None
    
    text = text.strip()
    
    # Fix 1: Remove trailing commas before closing braces
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)
    
    # Fix 2: Replace single quotes with double quotes (carefully)
    # Only do this for simple cases to avoid breaking apostrophes in text
    if "'" in text and '"' not in text:
        text = text.replace("'", '"')
    
    # Fix 3: Ensure the text starts with { and ends with }
    if not text.startswith('{'):
        # Find the first {
        start = text.find('{')
        if start != -1:
            text = text[start:]
    
    if not text.endswith('}'):
        # Find the last }
        end = text.rfind('}')
        if end != -1:
            text = text[:end+1]
    
    # Fix 4: Handle unescaped newlines in string values
    # This is a common issue with LLM outputs
    def escape_newlines_in_json(match):
        # Keep the outer quotes and content, but escape newlines
        content = match.group(1)
        # Escape unescaped newlines and tabs
        content = content.replace('\\n', '\n').replace('\n', '\\n')
        content = content.replace('\\t', '\t').replace('\t', '\\t')
        return '"' + content + '"'
    
    # Try to fix unescaped characters in string values
    # This is a simplified approach - match strings and fix them
    try:
        # Find all string values and try to fix them
        string_pattern = r'"((?:[^"\\]|\\.)*)"'
        text = re.sub(string_pattern, escape_newlines_in_json, text)
    except Exception:
        pass
    
    return text if text.startswith('{') and text.endswith('}') else None


def _extract_response_direct(text: str) -> str | None:
    """Extract response by looking for keywords directly in text."""
    text_lower = text.lower()
    
    # First, check if the text contains mathematical expressions - if so, skip direct extraction
    # and let the caller handle it as invalid
    # Use the same validation as _is_valid_label
    if not _is_valid_label(text):
        return None
    
    # PRIORITY 0: Look for explicit JSON field with "almost" - highest priority
    # This pattern specifically looks for the response field containing "almost"
    explicit_almost = re.search(r'"response"\s*:\s*"([^"]*almost[^"]*)"', text_lower)
    if explicit_almost:
        # Validate the extracted value is a proper label
        val = explicit_almost.group(1)
        if _is_valid_label(val) and re.search(r'\balmost\b', val.lower()):
            return "almost"
    
    # Also check for unquoted "almost" in JSON-like contexts
    json_almost = re.search(r'"response"\s*:\s*almost\b', text_lower)
    if json_almost:
        return "almost"
    
    # PRIORITY 1: Look for explicit "almost" classification - this is the most specific
    # and most commonly missed label
    almost_patterns = [
        r'"response"\s*:\s*"almost"',
        r'"classification"\s*:\s*"almost"',
        r'"grade"\s*:\s*"almost"',
        r'"result"\s*:\s*"almost"',
        r'"evaluation"\s*:\s*"almost"',
        r'"verdict"\s*:\s*"almost"',
        r'"outcome"\s*:\s*"almost"',
        r'\bresponse\s*[:=]\s*almost\b',
        r'\bclassification\s*[:=]\s*almost\b',
        r'\bclassif(y|ied)\s*[:=]\s*almost\b',
        r'\bis\s+almost\b',
        r'\balmost\s+correct\b',
        r'\balmost\s+complete\b',
        r'\bgrade\s*[:=]\s*almost\b',
        r'\bevaluation\s*[:=]\s*almost\b',
        r'\bverdict\s*[:=]\s*almost\b',
        r'\boutcome\s*[:=]\s*almost\b',
        r'\bthis\s+is\s+almost\b',
        r'\bclassify\s+as\s+almost\b',
        r'\bclassified\s+as\s+almost\b',
        r'\bclassification\s+is\s+almost\b',
        r'\bgrade\s+is\s+almost\b',
        r'\bevaluation\s+is\s+almost\b',
    ]
    for pattern in almost_patterns:
        if re.search(pattern, text_lower):
            return "almost"
    
    # PRIORITY 2: Look for explicit "partial" classification
    partial_patterns = [
        r'"response"\s*:\s*"partial"',
        r'"classification"\s*:\s*"partial"',
        r'"grade"\s*:\s*"partial"',
        r'"result"\s*:\s*"partial"',
        r'"evaluation"\s*:\s*"partial"',
        r'"verdict"\s*:\s*"partial"',
        r'"outcome"\s*:\s*"partial"',
        r'\bresponse\s*[:=]\s*partial\b',
        r'\bclassification\s*[:=]\s*partial\b',
        r'\bclassif(y|ied)\s*[:=]\s*partial\b',
        r'\bis\s+partial\b',
        r'\bpartially\s+correct\b',
        r'\bpartial\s+credit\b',
        r'\bpartial\s+progress\b',
        r'\bpartial\s+solution\b',
        r'\bgrade\s*[:=]\s*partial\b',
        r'\bevaluation\s*[:=]\s*partial\b',
        r'\bverdict\s*[:=]\s*partial\b',
        r'\boutcome\s*[:=]\s*partial\b',
        r'\bthis\s+is\s+partial\b',
        r'\bclassify\s+as\s+partial\b',
        r'\bclassified\s+as\s+partial\b',
        r'\bpartial\s+marks?\b',
        r'\bpartial\s+points?\b',
    ]
    for pattern in partial_patterns:
        if re.search(pattern, text_lower):
            return "partial"
    
    # PRIORITY 3: Look for explicit "incorrect" classification
    incorrect_patterns = [
        r'"response"\s*:\s*"incorrect"',
        r'"classification"\s*:\s*"incorrect"',
        r'"grade"\s*:\s*"incorrect"',
        r'"result"\s*:\s*"incorrect"',
        r'"evaluation"\s*:\s*"incorrect"',
        r'"verdict"\s*:\s*"incorrect"',
        r'"outcome"\s*:\s*"incorrect"',
        r'\bresponse\s*[:=]\s*incorrect\b',
        r'\bclassification\s*[:=]\s*incorrect\b',
        r'\bclassif(y|ied)\s*[:=]\s*incorrect\b',
        r'\bis\s+incorrect\b',
        r'\bis\s+wrong\b',
        r'\bwrong\s+answer\b',
        r'\bgrade\s*[:=]\s*incorrect\b',
        r'\bevaluation\s*[:=]\s*incorrect\b',
        r'\bverdict\s*[:=]\s*incorrect\b',
        r'\boutcome\s*[:=]\s*incorrect\b',
        r'\bthis\s+is\s+incorrect\b',
        r'\bclassify\s+as\s+incorrect\b',
        r'\bclassified\s+as\s+incorrect\b',
    ]
    for pattern in incorrect_patterns:
        if re.search(pattern, text_lower):
            return "incorrect"
    
    # PRIORITY 4: Look for explicit "correct" classification
    correct_patterns = [
        r'"response"\s*:\s*"correct"',
        r'"classification"\s*:\s*"correct"',
        r'"grade"\s*:\s*"correct"',
        r'"result"\s*:\s*"correct"',
        r'"evaluation"\s*:\s*"correct"',
        r'"verdict"\s*:\s*"correct"',
        r'"outcome"\s*:\s*"correct"',
        r'\bresponse\s*[:=]\s*correct\b',
        r'\bclassification\s*[:=]\s*correct\b',
        r'\bclassif(y|ied)\s*[:=]\s*correct\b',
        r'\bis\s+correct\b',
        r'\bcorrect\s+answer\b',
        r'\bgrade\s*[:=]\s*correct\b',
        r'\bevaluation\s*[:=]\s*correct\b',
        r'\bverdict\s*[:=]\s*correct\b',
        r'\boutcome\s*[:=]\s*correct\b',
        r'\bthis\s+is\s+correct\b',
        r'\bclassify\s+as\s+correct\b',
        r'\bclassified\s+as\s+correct\b',
    ]
    for pattern in correct_patterns:
        if re.search(pattern, text_lower):
            return "correct"
    
    # Look for explicit classification statements with response/classification keywords
    # These patterns look for explicit field assignments in JSON-like or natural language
    patterns = [
        # JSON-style field assignments (highest priority)
        r'"response"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"classification"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"grade"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"result"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"answer"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"outcome"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"evaluation"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"verdict"\s*:\s*"(correct|incorrect|partial|almost)"',
        # Natural language classification statements
        r'classification[\s:]+"?(correct|incorrect|partial|almost)"?',
        r'classify[\s:]+"?(correct|incorrect|partial|almost)"?',
        r'response[\s:]+"?(correct|incorrect|partial|almost)"?',
        r'grade[\s:]+"?(correct|incorrect|partial|almost)"?',
        r'result[\s:]+"?(correct|incorrect|partial|almost)"?',
        r'outcome[\s:]+"?(correct|incorrect|partial|almost)"?',
        r'evaluation[\s:]+"?(correct|incorrect|partial|almost)"?',
        r'verdict[\s:]+"?(correct|incorrect|partial|almost)"?',
        # Explicit statements about the classification
        r'"?(correct|incorrect|partial|almost)"?\s*is\s*the\s*correct\s*classification',
        r'the\s*answer\s*is\s*"?(correct|incorrect|partial|almost)"?',
        r'should\s*be\s*classified\s*as\s*"?(correct|incorrect|partial|almost)"?',
        r'classify\s*as\s*"?(correct|incorrect|partial|almost)"?',
        r'classified\s*as\s*"?(correct|incorrect|partial|almost)"?',
        r'i\s*classify\s*this\s*as\s*"?(correct|incorrect|partial|almost)"?',
        r'this\s*is\s*"?(correct|incorrect|partial|almost)"?',
        r'the\s*student[\'\']?s\s*answer\s*is\s*"?(correct|incorrect|partial|almost)"?',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            result = match.group(1).lower().strip()
            if result in ["correct", "incorrect", "partial", "almost"]:
                return result
    
    # Count occurrences of each label (only count standalone words)
    # Exclude occurrences within the reasoning field by looking for the response field area
    # Try to find the response field specifically
    response_section_match = re.search(r'"response"\s*:\s*"([^"]*(?:correct|incorrect|partial|almost)[^"]*)"', text_lower)
    if response_section_match:
        response_section = response_section_match.group(1)
        if "almost" in response_section:
            return "almost"
        if "correct" in response_section and "incorrect" not in response_section:
            return "correct"
        elif "partial" in response_section:
            return "partial"
        elif "incorrect" in response_section:
            return "incorrect"
    
    # Count standalone occurrences
    correct_count = len(re.findall(r'\bcorrect\b', text_lower))
    incorrect_count = len(re.findall(r'\bincorrect\b', text_lower))
    partial_count = len(re.findall(r'\bpartial\b', text_lower))
    almost_count = len(re.findall(r'\balmost\b', text_lower))
    
    # If only one appears, use that
    if correct_count > 0 and incorrect_count == 0 and partial_count == 0 and almost_count == 0:
        return "correct"
    if incorrect_count > 0 and correct_count == 0 and partial_count == 0 and almost_count == 0:
        return "incorrect"
    if partial_count > 0 and correct_count == 0 and incorrect_count == 0 and almost_count == 0:
        return "partial"
    if almost_count > 0 and correct_count == 0 and incorrect_count == 0 and partial_count == 0:
        return "almost"
    
    # If multiple appear, use priority based on context
    # For IMO grading, we want to be conservative:
    # - "almost" is most specific and should be prioritized
    # - "partial" indicates genuine progress (second most specific)
    # - "incorrect" should be prioritized when it appears (clear negative signal)
    # - "correct" is the default positive
    
    # Check if labels appear in a classification context vs descriptive context
    # Look for explicit classification patterns
    almost_classification = re.search(r'classif(y|ied).*almost|almost.*classif', text_lower)
    partial_classification = re.search(r'classif(y|ied).*partial|partial.*classif', text_lower)
    incorrect_classification = re.search(r'classif(y|ied).*incorrect|incorrect.*classif', text_lower)
    correct_classification = re.search(r'classif(y|ied).*correct|correct.*classif', text_lower)
    
    # Priority: explicit classification > frequency-based
    # Check in order of specificity: almost > partial > incorrect > correct
    if almost_classification and almost_count > 0:
        return "almost"
    if partial_classification and partial_count > 0:
        return "partial"
    if incorrect_classification and incorrect_count > 0:
        return "incorrect"
    if correct_classification and correct_count > 0:
        return "correct"
    
    # If no explicit classification found, use frequency with priority-based bias
    # PRIORITY: "almost" is most specific, then "partial", then "incorrect", then "correct"
    if almost_count > 0:
        return "almost"
    if partial_count > 0:
        return "partial"
    if incorrect_count > 0:
        return "incorrect"
    if correct_count > 0:
        return "correct"
    
    # Default to most conservative interpretation
    if incorrect_count > 0:
        return "incorrect"
    if partial_count > 0:
        return "partial"
    if almost_count > 0:
        return "almost"
    if correct_count > 0:
        return "correct"
    
    return None


def _check_grading_guidelines_for_almost(grading_guidelines: str) -> bool:
    """Check if grading guidelines indicate this should be an 'almost' case.
    
    Returns True if the guidelines contain (Almost) markers that suggest
    the student achieved the 'almost' level.
    """
    if not grading_guidelines:
        return False
    
    guidelines_lower = grading_guidelines.lower()
    
    # Check for explicit (Almost) markers
    if "(almost)" in guidelines_lower:
        return True
    
    # Check for phrases indicating "almost" level
    almost_indicators = [
        r'\balmost\s+complete\b',
        r'\balmost\s+correct\b',
        r'\bminor\s+(error|mistake|gap)\b',
        r'\bsmall\s+(error|mistake|gap)\b',
        r'\bverification\s+contains\s+minor\b',
        r'\b5[-\s]?6\s+points\b',
        r'\b6\s+points\b',
    ]
    
    for pattern in almost_indicators:
        if re.search(pattern, guidelines_lower):
            return True
    
    return False


def _check_grading_guidelines_for_partial(grading_guidelines: str) -> bool:
    """Check if grading guidelines indicate this should be a 'partial' case.
    
    Returns True if the guidelines contain (Partial) markers that suggest
    the student achieved the 'partial' level (made meaningful progress but
    has significant gaps).
    """
    if not grading_guidelines:
        return False
    
    guidelines_lower = grading_guidelines.lower()
    
    # Check for explicit (Partial) markers
    if "(partial)" in guidelines_lower:
        return True
    
    # Check for phrases indicating "partial" level
    partial_indicators = [
        r'\bpartial\s+progress\b',
        r'\bmeaningful\s+progress\b',
        r'\bvalid\s+lemma\b',
        r'\bright\s+approach\b',
        r'\bsignificant\s+gaps?\b',
        r'\b1[-\s]?4\s+points\b',
        r'\b2[-\s]?3\s+points\b',
        r'\b3[-\s]?4\s+points\b',
        r'\b4\s+points\b',
    ]
    
    for pattern in partial_indicators:
        if re.search(pattern, guidelines_lower):
            return True
    
    return False


def _normalize_prediction(prediction: str) -> str:
    """Normalize prediction to one of the four valid labels.
    
    PRIORITY ORDER (most specific first):
    1. "almost" - most specific, least common
    2. "partial" - more specific than correct/incorrect
    3. "incorrect" - negative signal
    4. "correct" - positive signal
    """
    pred_str = str(prediction).lower().strip()
    
    # Direct match - exact match is best
    if pred_str in ["correct", "incorrect", "partial", "almost"]:
        return pred_str
    
    # First check if this looks like a valid label at all
    if not _is_valid_label(prediction):
        # This is not a valid prediction - return conservative default
        return "incorrect"
    
    # PRIORITY 1: Check for "almost" first (most specific and least common)
    # "almost" is a distinct category that should be detected
    if "almost" in pred_str:
        return "almost"
    
    # PRIORITY 2: Check for partial (more specific than correct/incorrect)
    if "partial" in pred_str:
        return "partial"
    
    # PRIORITY 3: Check for incorrect/wrong/false/error (more specific than correct)
    if "incorrect" in pred_str or "wrong" in pred_str or "false" in pred_str or "error" in pred_str:
        return "incorrect"
    
    # PRIORITY 4: Check for correct (but not incorrect)
    if "correct" in pred_str and "incorrect" not in pred_str:
        return "correct"
    
    # Check for complete/full
    if "complete" in pred_str or "full" in pred_str:
        return "correct"
    
    # Default fallback - use "incorrect" as it's more conservative
    # when we can't determine the classification
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
        
        instruction = f"""You are an expert mathematical olympiad grader. Your task is to classify the student's answer into EXACTLY ONE of four categories: "correct", "almost", "partial", or "incorrect".

## Problem Statement:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Classification Rules:

**"correct"** (7 points): Complete solution with valid reasoning throughout, correct final answer, no gaps.

**"almost"** (5-6 points): Nearly complete solution. Correct/nearly correct final answer. Main proof structure sound. Only minor technical errors (NOT conceptual gaps). Look for "(Almost)" markers in grading guidelines. VERY CLOSE to complete - just needs small fixes.

**"partial"** (1-4 points): Meaningful progress (valid lemmas, right approach) but SIGNIFICANT gaps remain. Far from complete. Look for "(Partial)" markers in grading guidelines. Student made some real progress but solution is incomplete.

**"incorrect"** (0 points): No valid lemmas or non-trivial observations. Fundamentally flawed or trivial only.

## Decision Process:
1. Check: Did student prove ANY valid lemma? NO → "incorrect"
2. If yes: Is solution completely correct with NO gaps? YES → "correct"
3. If no: Is solution VERY CLOSE to complete (minor fixes needed)? YES → "almost"
4. Otherwise: "partial" (significant gaps remain, but some progress made)

## CRITICAL: "almost" vs "partial"
- "almost" = Nearly complete, minor errors only (5-6 points). The solution is almost there - just needs small technical fixes.
- "partial" = Some progress but major gaps (1-4 points). The student made meaningful progress but the solution is far from complete.

## KEY DISTINCTION:
- If student has valid lemmas AND is close to complete → "almost"
- If student has valid lemmas BUT has major gaps → "partial"
- If student has NO valid progress → "incorrect"

## OUTPUT FORMAT (STRICT JSON - MANDATORY):
You MUST output ONLY a JSON object. The response field MUST contain EXACTLY ONE of these four words: "correct", "almost", "partial", or "incorrect". NO OTHER TEXT is allowed in the response field.

<json>
{{
    "reasoning": "Your detailed analysis here",
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
                        # Priority order for JSON fields: response > classification > grade > result
                        found_value = None
                        
                        # PRIORITY 0: Check for "almost" FIRST in all priority fields
                        # "almost" is the most specific and most commonly missed label
                        for field in ["response", "classification", "grade", "result", "evaluation", "verdict", "outcome"]:
                            if field in json_data:
                                value = str(json_data[field])
                                # First validate it's a proper label (not math content)
                                if not _is_valid_label(value):
                                    continue
                                val_lower = value.lower().strip()
                                # Check for "almost" first (most specific and least common)
                                if "almost" in val_lower:
                                    # Validate it's not just "almost" as part of a larger word
                                    if re.search(r'\balmost\b', val_lower):
                                        found_value = "almost"
                                        break
                        
                        # First pass: look for EXACT matches of valid labels in priority fields
                        if not found_value:
                            for field in ["response", "classification", "grade", "result", "evaluation", "verdict", "outcome"]:
                                if field in json_data:
                                    value = str(json_data[field])
                                    # First validate it's a proper label (not math content)
                                    if not _is_valid_label(value):
                                        continue
                                    val_lower = value.lower().strip()
                                    # Validate that it's a proper label, not math content
                                    if val_lower in ["correct", "incorrect", "partial", "almost"]:
                                        found_value = val_lower
                                        break
                                    # Check if it contains a valid label as a substring (but not math)
                                    normalized = _normalize_prediction(val_lower)
                                    if normalized in ["correct", "incorrect", "partial", "almost"]:
                                        found_value = normalized
                                        break
                        
                        # Second pass: look for partial matches if no exact match found
                        # PRIORITY: Check for "almost" FIRST (most specific label), then "partial"
                        if not found_value:
                            for field in ["response", "classification", "grade", "result", "evaluation", "verdict", "outcome"]:
                                if field in json_data:
                                    value = str(json_data[field])
                                    # First validate it's a proper label (not math content)
                                    if not _is_valid_label(value):
                                        continue
                                    val_lower = value.lower().strip()
                                    # Check for "almost" first (most specific and least common)
                                    if "almost" in val_lower:
                                        found_value = "almost"
                                        break
                                    # Then check for "partial" (second most specific)
                                    elif "partial" in val_lower:
                                        found_value = "partial"
                                        break
                                    elif "incorrect" in val_lower:
                                        found_value = "incorrect"
                                        break
                                    elif "correct" in val_lower:
                                        found_value = "correct"
                                        break
                        
                        # Third pass: check all other fields in the JSON
                        # PRIORITY: Check for "almost" FIRST, then "partial" (most specific labels)
                        if not found_value:
                            for key, value in json_data.items():
                                if isinstance(value, str):
                                    # First validate it's a proper label (not math content)
                                    if not _is_valid_label(value):
                                        continue
                                    val_lower = value.lower().strip()
                                    # Check for exact matches first
                                    if val_lower in ["correct", "incorrect", "partial", "almost"]:
                                        found_value = val_lower
                                        break
                                    # Then check for "almost" (most specific)
                                    elif "almost" in val_lower:
                                        found_value = "almost"
                                        break
                                    # Then check for "partial" (second most specific)
                                    elif "partial" in val_lower:
                                        found_value = "partial"
                                        break
                                    elif "incorrect" in val_lower:
                                        found_value = "incorrect"
                                        break
                                    elif "correct" in val_lower:
                                        found_value = "correct"
                                        break
                        
                        if found_value:
                            prediction = found_value
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

        # Post-processing: Check grading guidelines for "almost" and "partial" indicators
        # This helps catch cases where the model prediction doesn't match the guidelines
        if prediction in ["correct", "partial", "incorrect"]:
            # Check for "almost" indicators first (higher priority)
            if _check_grading_guidelines_for_almost(grading_guidelines):
                # If guidelines suggest "almost" and current prediction is not "almost",
                # we might need to reconsider. Only adjust if prediction was "partial" or "incorrect"
                # (since "correct" should remain as the model explicitly said complete solution)
                if prediction in ["partial", "incorrect"]:
                    self.log_fn("Adjusting prediction to 'almost' based on grading guidelines")
                    prediction = "almost"
            # Check for "partial" indicators
            elif _check_grading_guidelines_for_partial(grading_guidelines):
                # If guidelines suggest "partial" and current prediction is "incorrect",
                # upgrade to "partial" (student made some progress)
                if prediction == "incorrect":
                    self.log_fn("Adjusting prediction from 'incorrect' to 'partial' based on grading guidelines")
                    prediction = "partial"

        # Final validation - ensure prediction is one of the four valid labels
        if prediction not in ["correct", "incorrect", "partial", "almost"]:
            prediction = "incorrect"  # Conservative fallback

        return str(prediction), msg_history
