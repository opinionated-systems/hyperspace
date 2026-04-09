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
    Enhanced with better error handling and recovery strategies.
    """
    if not text or not isinstance(text, str):
        return None
        
    results = []
    search_from = 0
    
    def try_parse_with_recovery(json_str: str) -> dict | None:
        """Try to parse JSON with multiple recovery strategies."""
        if not json_str:
            return None
        json_str = json_str.strip()
        
        # Direct parsing
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # Try with single quotes replaced
        try:
            return json.loads(json_str.replace("'", '"'))
        except json.JSONDecodeError:
            pass
        
        # Try fixing trailing commas
        try:
            cleaned = re.sub(r',\s*}', '}', json_str)
            cleaned = re.sub(r',\s*]', ']', cleaned)
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        
        return None
    
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        parsed = try_parse_with_recovery(inner)
        if parsed:
            results.append(parsed)
    
    return results or None


def _extract_json_flexible(text: str) -> list[dict] | None:
    """Extract JSON objects using multiple strategies.
    
    Tries multiple patterns with improved robustness for edge cases.
    Enhanced with better error recovery and more format support.
    """
    if not text or not isinstance(text, str):
        return None
        
    results = []
    text_stripped = text.strip()
    
    def try_parse_json(json_str: str) -> dict | None:
        """Try to parse JSON with multiple fallback strategies."""
        if not json_str:
            return None
        json_str = json_str.strip()
        
        # Try direct parsing first
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # Try fixing single quotes
        try:
            fixed = json_str.replace("'", '"')
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        
        # Try fixing trailing commas
        try:
            fixed = re.sub(r',\s*}', '}', json_str)
            fixed = re.sub(r',\s*]', ']', fixed)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        
        # Try fixing unquoted keys
        try:
            fixed = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        
        return None
    
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
        parsed = try_parse_json(inner)
        if parsed:
            results.append(parsed)
    
    # Strategy 2: ```json...``` code blocks
    pattern = r'```json\s*(.*?)\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        parsed = try_parse_json(match)
        if parsed:
            results.append(parsed)
    
    # Strategy 3: ```...``` code blocks (without json label)
    pattern = r'```\s*(\{[\s\S]*?\})\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        parsed = try_parse_json(match)
        if parsed:
            results.append(parsed)
    
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
    candidates = []
    lines = text_stripped.split('\n')
    if lines:
        candidates.extend([lines[0], lines[-1]])
    candidates.append(text_stripped)
    
    for candidate in candidates:
        candidate = candidate.strip()
        if candidate.startswith('{') and candidate.endswith('}'):
            parsed = try_parse_json(candidate)
            if parsed:
                results.append(parsed)
    
    # Strategy 8: Look for loose response:value patterns
    loose_pattern = r'\bresponse\s*[:=]\s*["\']?(correct|incorrect|partial)["\']?\b'
    matches = re.findall(loose_pattern, text, re.IGNORECASE)
    for match in matches:
        results.append({"response": match.lower()})
    
    return results or None


def _extract_response_direct(text: str) -> str | None:
    """Direct extraction of response value from text.
    
    Looks for explicit mentions of the grade in the text with improved patterns.
    Enhanced with better handling of edge cases and malformed responses.
    """
    if not text or not isinstance(text, str):
        return None
        
    text_lower = text.lower().strip()
    
    # Check for explicit grade mentions in various quote styles
    quote_patterns = [
        ('"correct"', 'correct'),
        ("'correct'", 'correct'),
        ('`correct`', 'correct'),
        ('"incorrect"', 'incorrect'),
        ("'incorrect'", 'incorrect'),
        ('`incorrect`', 'incorrect'),
        ('"partial"', 'partial'),
        ("'partial'", 'partial'),
        ('`partial`', 'partial'),
    ]
    
    for pattern, grade in quote_patterns:
        if pattern in text_lower:
            return grade
    
    # Check for grade at end of text or in conclusion (with more terminator handling)
    lines = text_lower.split('\n')
    for line in reversed(lines):
        line_clean = line.strip().rstrip('}').rstrip(',').rstrip('.').rstrip('"').rstrip("'").strip()
        if line_clean in ['correct', 'incorrect', 'partial']:
            return line_clean
        
        # Check for patterns like "grade: correct" or "decision: partial"
        for grade in ['correct', 'incorrect', 'partial']:
            if grade in line_clean:
                # Look for decision markers near the grade
                markers = ['grade', 'decision', 'verdict', 'result', 'evaluation', 'assessment']
                if any(marker in line_clean for marker in markers):
                    # Make sure the grade is the value, not part of a word
                    grade_pos = line_clean.find(grade)
                    if grade_pos >= 0:
                        # Check if it's a standalone word
                        before = line_clean[max(0, grade_pos-1):grade_pos]
                        after = line_clean[grade_pos+len(grade):min(len(line_clean), grade_pos+len(grade)+1)]
                        if (not before or not before.isalpha()) and (not after or not after.isalpha()):
                            return grade
    
    # Check for grade at the beginning with more patterns
    if lines:
        first_line = lines[0].strip()
        for grade in ['correct', 'incorrect', 'partial']:
            # Direct start
            if first_line.startswith(grade + ' ') or first_line == grade:
                return grade
            # After opening brace
            if first_line.lstrip('{').lstrip().startswith(grade + ' ') or first_line.lstrip('{').lstrip() == grade:
                return grade
            # After opening quote
            if first_line.lstrip('"').startswith(grade) or first_line.lstrip("'").startswith(grade):
                return grade
    
    # Check for grade surrounded by common delimiters
    for grade in ['correct', 'incorrect', 'partial']:
        delimiter_patterns = [
            rf'\({grade}\)',
            rf'\[{grade}\]',
            rf'\{{{grade}\}}',
            rf'<{grade}>',
        ]
        for pattern in delimiter_patterns:
            if re.search(pattern, text_lower):
                return grade
    
    return None


def _extract_response_llm_style(text: str) -> str | None:
    """Extract response using patterns common in LLM outputs.
    
    Handles various formatting styles that LLMs commonly use.
    Enhanced with more comprehensive pattern matching.
    """
    if not text or not isinstance(text, str):
        return None
        
    text_lower = text.lower().strip()
    
    # Pattern 1: Look for conclusion phrases with various formats
    conclusion_patterns = [
        rf'\bthe answer is\s*[:=]?\s*["\']?(correct|incorrect|partial)["\']?',
        rf'\btherefore,?\s*(?:the\s+)?answer is\s*[:=]?\s*["\']?(correct|incorrect|partial)["\']?',
        rf'\bfinal answer\s*[:=]+\s*["\']?(correct|incorrect|partial)["\']?',
        rf'\bconclusion\s*[:=]+\s*["\']?(correct|incorrect|partial)["\']?',
        rf'\bgrading\s*[:=]+\s*["\']?(correct|incorrect|partial)["\']?',
        rf'\bassessment\s*[:=]+\s*["\']?(correct|incorrect|partial)["\']?',
        rf'\bevaluation\s*[:=]+\s*["\']?(correct|incorrect|partial)["\']?',
        rf'\bverdict\s*[:=]+\s*["\']?(correct|incorrect|partial)["\']?',
        rf'\bthe student\'s answer is\s*["\']?(correct|incorrect|partial)["\']?',
        rf'\bthis is\s*["\']?(correct|incorrect|partial)["\']?',
    ]
    
    for pattern in conclusion_patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1)
    
    # Pattern 2: Look for standalone grade on its own line (with more cleaning)
    lines = text_lower.split('\n')
    for line in lines:
        line_clean = line.strip()
        # Remove common terminators
        for terminator in ['}', ',', '.', '"', "'", '`', ')', ']', '>']:
            line_clean = line_clean.rstrip(terminator).strip()
        for terminator in ['{', '"', "'", '`', '(', '[', '<']:
            line_clean = line_clean.lstrip(terminator).strip()
        if line_clean in ['correct', 'incorrect', 'partial']:
            return line_clean
    
    # Pattern 3: Look for grade in various delimiters
    bracket_patterns = [
        rf'\((correct|incorrect|partial)\)',
        rf'\[(correct|incorrect|partial)\]',
        rf'\{{(correct|incorrect|partial)\}}',
        rf'<(correct|incorrect|partial)>',
        rf'"(correct|incorrect|partial)"',
        rf"'(correct|incorrect|partial)'",
        rf'`(correct|incorrect|partial)`',
    ]
    for pattern in bracket_patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1)
    
    # Pattern 4: Look for grade after common markers
    marker_patterns = [
        rf'\*\*\s*(correct|incorrect|partial)\s*\*\*',  # Bold markdown
        rf'\*\s*(correct|incorrect|partial)\s*\*',        # Italic markdown
        rf'##?\s+(correct|incorrect|partial)\b',          # Headers
        rf'\bstatus\s*[:=]\s*["\']?(correct|incorrect|partial)["\']?',
        rf'\bresult\s*[:=]\s*["\']?(correct|incorrect|partial)["\']?',
    ]
    for pattern in marker_patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1)
    
    return None


def _extract_response_robust(text: str) -> str | None:
    """Robust extraction using multiple pattern matching strategies.
    
    This function tries to find the grade in various formats with priority ordering.
    Improved with better edge case handling and confidence scoring.
    """
    if not text or not isinstance(text, str):
        return None
        
    text_lower = text.lower()
    stripped = text_lower.strip()
    
    # Priority 1: Look for JSON-like patterns with response field (most reliable)
    json_patterns = [
        (r'"response"\s*:\s*"(correct|incorrect|partial)"', 10),  # Highest confidence
        (r"'response'\s*:\s*'(correct|incorrect|partial)'", 10),
        (r'response\s*:\s*"(correct|incorrect|partial)"', 9),
        (r'response\s*:\s*\'(correct|incorrect|partial)\'', 9),
        (r'response\s*:\s*(correct|incorrect|partial)', 8),
    ]
    
    best_match = None
    best_confidence = 0
    
    for pattern, confidence in json_patterns:
        match = re.search(pattern, text_lower)
        if match and confidence > best_confidence:
            best_match = match.group(1)
            best_confidence = confidence
    
    if best_match:
        return best_match
    
    # Priority 2: Look for the grade as a standalone word at the end of the text
    for grade in ['correct', 'incorrect', 'partial']:
        # Check for grade at end with various terminators
        end_patterns = [
            rf'{grade}\s*$',
            rf'{grade}\s*\.\s*$',
            rf'{grade}\s*}}\s*$',
            rf'{grade}\s*,\s*$',
        ]
        for pattern in end_patterns:
            if re.search(pattern, stripped):
                return grade
    
    # Priority 3: Look for grade in quotes (reliable indicator)
    for grade in ['correct', 'incorrect', 'partial']:
        quote_patterns = [
            rf'"{grade}"',
            rf"'{grade}'",
            rf'`{grade}`',
        ]
        for pattern in quote_patterns:
            if re.search(pattern, text_lower):
                return grade
    
    # Priority 4: Look for grade at start of text
    for grade in ['correct', 'incorrect', 'partial']:
        start_patterns = [
            rf'^{grade}\b',
            rf'^\s*{{\s*{grade}\b',
            rf'^\s*"{grade}"',
            rf'^\s*\'{grade}\'',
        ]
        for pattern in start_patterns:
            if re.search(pattern, stripped):
                return grade
    
    # Priority 5: Look for explicit decision patterns with higher confidence
    decision_patterns = [
        (rf'\bgrade\s*[:=]\s*["\']?(correct|incorrect|partial)["\']?', 7),
        (rf'\bdecision\s*[:=]\s*["\']?(correct|incorrect|partial)["\']?', 7),
        (rf'\bverdict\s*[:=]\s*["\']?(correct|incorrect|partial)["\']?', 7),
        (rf'\bfinal\s+(?:grade|decision|verdict)\s*[:=]\s*["\']?(correct|incorrect|partial)["\']?', 8),
        (rf'\banswer\s+is\s+["\']?(correct|incorrect|partial)["\']?', 6),
        (rf'\bresult\s*[:=]\s*["\']?(correct|incorrect|partial)["\']?', 6),
        (rf'\bevaluation\s*[:=]\s*["\']?(correct|incorrect|partial)["\']?', 6),
    ]
    
    for pattern, confidence in decision_patterns:
        match = re.search(pattern, text_lower)
        if match and confidence > best_confidence:
            best_match = match.group(1)
            best_confidence = confidence
    
    if best_match:
        return best_match
    
    # Priority 6: Count word-boundary occurrences with context weighting
    counts = {}
    context_weights = {
        'correct': 1.0,
        'incorrect': 1.0,
        'partial': 1.2,  # Slightly boost partial as it's less commonly mentioned
    }
    
    for grade in ['correct', 'incorrect', 'partial']:
        # Count word boundaries
        base_count = len(re.findall(rf'\b{grade}\b', text_lower))
        # Count as standalone lines (higher weight)
        line_count = len(re.findall(rf'^{grade}\s*$', text_lower, re.MULTILINE))
        counts[grade] = (base_count + line_count * 2) * context_weights[grade]
    
    if counts and max(counts.values()) > 0:
        max_count = max(counts.values())
        # Return the grade with highest weighted count
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
