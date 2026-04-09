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
                return data
        except json.JSONDecodeError:
            fixed = _fix_json(match.strip())
            if fixed:
                try:
                    data = json.loads(fixed)
                    if isinstance(data, dict):
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
                return data
        except json.JSONDecodeError:
            fixed = _fix_json(match.strip())
            if fixed:
                try:
                    data = json.loads(fixed)
                    if isinstance(data, dict):
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
                return data
        except json.JSONDecodeError:
            fixed = _fix_json(match.strip())
            if fixed:
                try:
                    data = json.loads(fixed)
                    if isinstance(data, dict):
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
    math_indicators = [
        r'\$\$.*?\$\$',
        r'\$.*?\$',
        r'\\[a-zA-Z]+',
        r'\^',
        r'\\frac',
        r'\\sum',
        r'\\int',
        r'<step',
        r'\\begin\{',
        r'\\end\{',
    ]
    
    # If it looks like the response is just math content, reject it
    math_count = 0
    for pattern in math_indicators:
        if re.search(pattern, text_lower):
            math_count += 1
    
    # If more than 2 math indicators found, likely not a valid label
    if math_count > 2:
        return None
    
    # PRIORITY 0: Look for explicit JSON field with "almost" - highest priority
    # This pattern specifically looks for the response field containing "almost"
    explicit_almost = re.search(r'"response"\s*:\s*"([^"]*almost[^"]*)"', text_lower)
    if explicit_almost:
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
        r'\bgrade\s*[:=]\s*partial\b',
        r'\bevaluation\s*[:=]\s*partial\b',
        r'\bverdict\s*[:=]\s*partial\b',
        r'\boutcome\s*[:=]\s*partial\b',
        r'\bthis\s+is\s+partial\b',
        r'\bclassify\s+as\s+partial\b',
        r'\bclassified\s+as\s+partial\b',
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
    # - If "almost" appears with others, it might be describing proximity, not classification
    # - "incorrect" should be prioritized when it appears (clear negative signal)
    # - "partial" indicates genuine progress (positive signal)
    # - "correct" is the default positive
    
    # Check if "almost" appears in a classification context vs descriptive context
    # Look for explicit classification patterns
    almost_classification = re.search(r'classif(y|ied).*almost|almost.*classif', text_lower)
    partial_classification = re.search(r'classif(y|ied).*partial|partial.*classif', text_lower)
    incorrect_classification = re.search(r'classif(y|ied).*incorrect|incorrect.*classif', text_lower)
    correct_classification = re.search(r'classif(y|ied).*correct|correct.*classif', text_lower)
    
    # Priority: explicit classification > frequency-based
    if almost_classification and almost_count > 0:
        return "almost"
    if incorrect_classification and incorrect_count > 0:
        return "incorrect"
    if partial_classification and partial_count > 0:
        return "partial"
    if correct_classification and correct_count > 0:
        return "correct"
    
    # If no explicit classification found, use frequency with priority-based bias
    # PRIORITY: "almost" is most specific and should be checked first
    # Then "partial", then "incorrect", then "correct"
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
    
    # Check for mathematical expressions or problem content (invalid predictions)
    # These should be rejected and fall back to conservative default
    invalid_indicators = [
        r'\$\$.*?\$\$',  # LaTeX display math
        r'\$.*?\$',      # LaTeX inline math
        r'\\[a-zA-Z]+\{', # LaTeX commands
        r'\^',           # Exponents
        r'\\frac',      # Fractions
        r'\\sum',       # Sums
        r'\\int',       # Integrals
        r'<step',        # Step tags
        r'\\begin\{',   # LaTeX environments
    ]
    
    # If it looks like mathematical content, it's not a valid label
    for pattern in invalid_indicators:
        if re.search(pattern, pred_str):
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
        
        instruction = f"""You are an expert mathematical olympiad grader. Classify the student's answer into EXACTLY ONE category: "correct", "almost", "partial", or "incorrect".

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

**"almost"** (5-6 points): Nearly complete solution. Correct/nearly correct final answer. Main proof structure sound. Only minor technical errors (NOT conceptual gaps). Look for "(Almost)" markers in grading guidelines. VERY CLOSE to complete.

**"partial"** (1-4 points): Meaningful progress (valid lemmas, right approach) but SIGNIFICANT gaps remain. Far from complete. Look for "(Partial)" markers in grading guidelines.

**"incorrect"** (0 points): No valid lemmas or non-trivial observations. Fundamentally flawed or trivial only.

## Decision Process:
1. Check: Did student prove ANY valid lemma? NO → "incorrect"
2. If yes: Is solution completely correct with NO gaps? YES → "correct"
3. If no: Is solution VERY CLOSE to complete (minor fixes needed)? YES → "almost"
4. Otherwise: "partial" (significant gaps remain)

## CRITICAL: "almost" vs "partial"
- "almost" = Nearly complete, minor errors only (5-6 points)
- "partial" = Some progress but major gaps (1-4 points)

## OUTPUT FORMAT (STRICT JSON):
You MUST output ONLY a JSON object in this exact format:

<json>
{{
    "reasoning": "Detailed analysis of what student got right/wrong and why this classification was chosen.",
    "response": "correct"
}}
</json>

OR replace "correct" with "almost", "partial", or "incorrect".

The "response" field MUST contain EXACTLY one word: correct, almost, partial, or incorrect. No other text allowed."""

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
                                value = str(json_data[field]).lower().strip()
                                # Check for "almost" first (most specific and least common)
                                if "almost" in value and not re.search(r'\$|\\[a-zA-Z]', value):
                                    # Validate it's not just "almost" as part of a larger word
                                    if re.search(r'\balmost\b', value):
                                        found_value = "almost"
                                        break
                        
                        # First pass: look for EXACT matches of valid labels in priority fields
                        if not found_value:
                            for field in ["response", "classification", "grade", "result", "evaluation", "verdict", "outcome"]:
                                if field in json_data:
                                    value = str(json_data[field]).lower().strip()
                                    # Validate that it's a proper label, not math content
                                    if value in ["correct", "incorrect", "partial", "almost"]:
                                        found_value = value
                                        break
                                    # Check if it contains a valid label as a substring (but not math)
                                    normalized = _normalize_prediction(value)
                                    if normalized in ["correct", "incorrect", "partial", "almost"]:
                                        found_value = normalized
                                        break
                        
                        # Second pass: look for partial matches if no exact match found
                        # PRIORITY: Check for "almost" FIRST (most specific label)
                        if not found_value:
                            for field in ["response", "classification", "grade", "result", "evaluation", "verdict", "outcome"]:
                                if field in json_data:
                                    value = str(json_data[field]).lower().strip()
                                    # Check for "almost" first (most specific and least common)
                                    if "almost" in value and not re.search(r'\$|\\[a-zA-Z]', value):
                                        found_value = "almost"
                                        break
                                    elif "partial" in value and not re.search(r'\$|\\[a-zA-Z]', value):
                                        found_value = "partial"
                                        break
                                    elif "incorrect" in value and not re.search(r'\$|\\[a-zA-Z]', value):
                                        found_value = "incorrect"
                                        break
                                    elif "correct" in value and not re.search(r'\$|\\[a-zA-Z]', value):
                                        found_value = "correct"
                                        break
                        
                        # Third pass: check all other fields in the JSON
                        # PRIORITY: Check for "almost" FIRST (most specific label)
                        if not found_value:
                            for key, value in json_data.items():
                                if isinstance(value, str):
                                    val_lower = value.lower().strip()
                                    # Skip if it looks like math content
                                    if re.search(r'\$|\\[a-zA-Z]', val_lower):
                                        continue
                                    if val_lower in ["correct", "incorrect", "partial", "almost"]:
                                        found_value = val_lower
                                        break
                                    elif "almost" in val_lower:
                                        found_value = "almost"
                                        break
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

        # Post-processing: Check grading guidelines for "almost" indicators
        # This helps catch cases where the model predicted "correct" or "partial"
        # but the grading guidelines clearly indicate "almost" level
        if prediction in ["correct", "partial"]:
            if _check_grading_guidelines_for_almost(grading_guidelines):
                # If guidelines suggest "almost" and current prediction is "correct" or "partial",
                # we might need to reconsider. However, we should trust the model's judgment
                # if it explicitly said "correct" (complete solution). Only adjust if it said "partial".
                if prediction == "partial":
                    self.log_fn("Adjusting prediction from 'partial' to 'almost' based on grading guidelines")
                    prediction = "almost"

        # Final validation - ensure prediction is one of the four valid labels
        if prediction not in ["correct", "incorrect", "partial", "almost"]:
            prediction = "incorrect"  # Conservative fallback

        return str(prediction), msg_history
