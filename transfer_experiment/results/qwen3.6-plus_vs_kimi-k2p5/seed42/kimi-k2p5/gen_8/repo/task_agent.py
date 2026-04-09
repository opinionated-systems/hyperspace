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
        
        # Try fixing unquoted keys
        try:
            cleaned = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
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
    # Support correct, incorrect, partial, and almost (maps to partial)
    pattern = r'["\']?response["\']?\s*:\s*["\']?(correct|incorrect|partial|almost)["\']?'
    matches = re.findall(pattern, text, re.IGNORECASE)
    for match in matches:
        grade = match.lower()
        if grade == 'almost':
            grade = 'partial'
        results.append({"response": grade})
    
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
    loose_pattern = r'\bresponse\s*[:=]\s*["\']?(correct|incorrect|partial|almost)["\']?\b'
    matches = re.findall(loose_pattern, text, re.IGNORECASE)
    for match in matches:
        grade = match.lower()
        if grade == 'almost':
            grade = 'partial'
        results.append({"response": grade})
    
    return results or None


def _extract_response_direct(text: str) -> str | None:
    """Direct extraction of response value from text.
    
    Looks for explicit mentions of the grade in the text with improved patterns.
    Enhanced with better handling of edge cases and malformed responses.
    Supports: correct, incorrect, partial, almost (maps to partial)
    """
    if not text or not isinstance(text, str):
        return None
        
    text_lower = text.lower().strip()
    
    # All valid grades including 'almost' which maps to 'partial'
    valid_grades = ['correct', 'incorrect', 'partial', 'almost']
    
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
        ('"almost"', 'partial'),
        ("'almost'", 'partial'),
        ('`almost`', 'partial'),
    ]
    
    for pattern, grade in quote_patterns:
        if pattern in text_lower:
            return grade
    
    # Check for grade at end of text or in conclusion (with more terminator handling)
    lines = text_lower.split('\n')
    for line in reversed(lines):
        line_clean = line.strip().rstrip('}').rstrip(',').rstrip('.').rstrip('"').rstrip("'").strip()
        if line_clean in valid_grades:
            return 'partial' if line_clean == 'almost' else line_clean
        
        # Check for patterns like "grade: correct" or "decision: partial"
        for grade in valid_grades:
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
                            return 'partial' if grade == 'almost' else grade
    
    # Check for grade at the beginning with more patterns
    if lines:
        first_line = lines[0].strip()
        for grade in valid_grades:
            # Direct start
            if first_line.startswith(grade + ' ') or first_line == grade:
                return 'partial' if grade == 'almost' else grade
            # After opening brace
            if first_line.lstrip('{').lstrip().startswith(grade + ' ') or first_line.lstrip('{').lstrip() == grade:
                return 'partial' if grade == 'almost' else grade
            # After opening quote
            if first_line.lstrip('"').startswith(grade) or first_line.lstrip("'").startswith(grade):
                return 'partial' if grade == 'almost' else grade
    
    # Check for grade surrounded by common delimiters
    for grade in valid_grades:
        delimiter_patterns = [
            rf'\({grade}\)',
            rf'\[{grade}\]',
            rf'\{{{grade}\}}',
            rf'<{grade}>',
        ]
        for pattern in delimiter_patterns:
            if re.search(pattern, text_lower):
                return 'partial' if grade == 'almost' else grade
    
    return None


def _extract_response_llm_style(text: str) -> str | None:
    """Extract response using patterns common in LLM outputs.
    
    Handles various formatting styles that LLMs commonly use.
    Enhanced with more comprehensive pattern matching.
    Supports: correct, incorrect, partial, almost (maps to partial)
    """
    if not text or not isinstance(text, str):
        return None
        
    text_lower = text.lower().strip()
    
    # All valid grades including 'almost' which maps to 'partial'
    valid_grades_pattern = 'correct|incorrect|partial|almost'
    
    # Pattern 0: Check for "almost" context indicators first
    almost_indicators = [
        r'\(almost\)',
        r'almost correct',
        r'almost complete',
        r'partially correct',
        r'nearly correct',
        r'mostly correct',
        r'solution is almost',
        r'answer is almost',
    ]
    for pattern in almost_indicators:
        if re.search(pattern, text_lower):
            # Check if there's an explicit conflicting grade nearby
            has_explicit_correct = re.search(r'"correct"|\bfinal\s+(?:grade|answer|verdict)[\s\w]{0,30}\bcorrect\b', text_lower)
            has_explicit_incorrect = re.search(r'"incorrect"|\bfinal\s+(?:grade|answer|verdict)[\s\w]{0,30}\bincorrect\b', text_lower)
            if not has_explicit_correct and not has_explicit_incorrect:
                return 'partial'
    
    # Pattern 1: Look for conclusion phrases with various formats
    conclusion_patterns = [
        rf'\bthe answer is\s*[:=]?\s*["\']?({valid_grades_pattern})["\']?',
        rf'\btherefore,?\s*(?:the\s+)?answer is\s*[:=]?\s*["\']?({valid_grades_pattern})["\']?',
        rf'\bfinal answer\s*[:=]+\s*["\']?({valid_grades_pattern})["\']?',
        rf'\bconclusion\s*[:=]+\s*["\']?({valid_grades_pattern})["\']?',
        rf'\bgrading\s*[:=]+\s*["\']?({valid_grades_pattern})["\']?',
        rf'\bassessment\s*[:=]+\s*["\']?({valid_grades_pattern})["\']?',
        rf'\bevaluation\s*[:=]+\s*["\']?({valid_grades_pattern})["\']?',
        rf'\bverdict\s*[:=]+\s*["\']?({valid_grades_pattern})["\']?',
        rf'\bthe student\'s answer is\s*["\']?({valid_grades_pattern})["\']?',
        rf'\bthis is\s*["\']?({valid_grades_pattern})["\']?',
    ]
    
    for pattern in conclusion_patterns:
        match = re.search(pattern, text_lower)
        if match:
            result = match.group(1)
            return 'partial' if result == 'almost' else result
    
    # Pattern 2: Look for standalone grade on its own line (with more cleaning)
    valid_grades = ['correct', 'incorrect', 'partial', 'almost']
    lines = text_lower.split('\n')
    for line in lines:
        line_clean = line.strip()
        # Remove common terminators
        for terminator in ['}', ',', '.', '"', "'", '`', ')', ']', '>']:
            line_clean = line_clean.rstrip(terminator).strip()
        for terminator in ['{', '"', "'", '`', '(', '[', '<']:
            line_clean = line_clean.lstrip(terminator).strip()
        if line_clean in valid_grades:
            return 'partial' if line_clean == 'almost' else line_clean
    
    # Pattern 3: Look for grade in various delimiters
    bracket_patterns = [
        rf'\(({valid_grades_pattern})\)',
        rf'\[({valid_grades_pattern})\]',
        rf'\{{({valid_grades_pattern})\}}',
        rf'<({valid_grades_pattern})>',
        rf'"({valid_grades_pattern})"',
        rf"'({valid_grades_pattern})'",
        rf'`({valid_grades_pattern})`',
    ]
    for pattern in bracket_patterns:
        match = re.search(pattern, text_lower)
        if match:
            result = match.group(1)
            return 'partial' if result == 'almost' else result
    
    # Pattern 4: Look for grade after common markers
    marker_patterns = [
        rf'\*\*\s*({valid_grades_pattern})\s*\*\*',  # Bold markdown
        rf'\*\s*({valid_grades_pattern})\s*\*',        # Italic markdown
        rf'##?\s+({valid_grades_pattern})\b',          # Headers
        rf'\bstatus\s*[:=]\s*["\']?({valid_grades_pattern})["\']?',
        rf'\bresult\s*[:=]\s*["\']?({valid_grades_pattern})["\']?',
    ]
    for pattern in marker_patterns:
        match = re.search(pattern, text_lower)
        if match:
            result = match.group(1)
            return 'partial' if result == 'almost' else result
    
    return None


def _extract_response_robust(text: str) -> str | None:
    """Robust extraction using multiple pattern matching strategies.
    
    This function tries to find the grade in various formats with priority ordering.
    Improved with better edge case handling and confidence scoring.
    Supports: correct, incorrect, partial, almost
    """
    if not text or not isinstance(text, str):
        return None
        
    text_lower = text.lower()
    stripped = text_lower.strip()
    
    # All valid grades including 'almost' which maps to 'partial'
    valid_grades = ['correct', 'incorrect', 'partial', 'almost']
    
    # Priority 0: Look for explicit "almost" indicators in the text that suggest partial credit
    almost_context_patterns = [
        r'\(almost\)',
        r'almost correct',
        r'almost complete',
        r'partially correct',
        r'nearly correct',
        r'mostly correct',
    ]
    for pattern in almost_context_patterns:
        if re.search(pattern, text_lower):
            # If we find almost indicators, check if there's also a conflicting grade
            # Only return partial if no explicit correct/incorrect is found nearby
            has_explicit_correct = re.search(r'"correct"|\bcorrect\b[\s\w]{0,20}grade|\bgrade[\s\w]{0,20}\bcorrect\b', text_lower)
            has_explicit_incorrect = re.search(r'"incorrect"|\bincorrect\b[\s\w]{0,20}grade|\bgrade[\s\w]{0,20}\bincorrect\b', text_lower)
            if not has_explicit_correct and not has_explicit_incorrect:
                return 'partial'
    
    # Priority 1: Look for JSON-like patterns with response field (most reliable)
    json_patterns = [
        (r'"response"\s*:\s*"(correct|incorrect|partial|almost)"', 10),  # Highest confidence
        (r"'response'\s*:\s*'(correct|incorrect|partial|almost)'", 10),
        (r'response\s*:\s*"(correct|incorrect|partial|almost)"', 9),
        (r'response\s*:\s*\'(correct|incorrect|partial|almost)\'', 9),
        (r'response\s*:\s*(correct|incorrect|partial|almost)', 8),
    ]
    
    best_match = None
    best_confidence = 0
    
    for pattern, confidence in json_patterns:
        match = re.search(pattern, text_lower)
        if match and confidence > best_confidence:
            best_match = match.group(1)
            best_confidence = confidence
    
    if best_match:
        # Map 'almost' to 'partial' for consistency
        return 'partial' if best_match == 'almost' else best_match
    
    # Priority 2: Look for the grade as a standalone word at the end of the text
    for grade in valid_grades:
        # Check for grade at end with various terminators
        end_patterns = [
            rf'{grade}\s*$',
            rf'{grade}\s*\.\s*$',
            rf'{grade}\s*}}\s*$',
            rf'{grade}\s*,\s*$',
        ]
        for pattern in end_patterns:
            if re.search(pattern, stripped):
                return 'partial' if grade == 'almost' else grade
    
    # Priority 3: Look for grade in quotes (reliable indicator)
    for grade in valid_grades:
        quote_patterns = [
            rf'"{grade}"',
            rf"'{grade}'",
            rf'`{grade}`',
        ]
        for pattern in quote_patterns:
            if re.search(pattern, text_lower):
                return 'partial' if grade == 'almost' else grade
    
    # Priority 4: Look for grade at start of text
    for grade in valid_grades:
        start_patterns = [
            rf'^{grade}\b',
            rf'^\s*{{\s*{grade}\b',
            rf'^\s*"{grade}"',
            rf'^\s*\'{grade}\'',
        ]
        for pattern in start_patterns:
            if re.search(pattern, stripped):
                return 'partial' if grade == 'almost' else grade
    
    # Priority 5: Look for explicit decision patterns with higher confidence
    decision_patterns = [
        (rf'\bgrade\s*[:=]\s*["\']?(correct|incorrect|partial|almost)["\']?', 7),
        (rf'\bdecision\s*[:=]\s*["\']?(correct|incorrect|partial|almost)["\']?', 7),
        (rf'\bverdict\s*[:=]\s*["\']?(correct|incorrect|partial|almost)["\']?', 7),
        (rf'\bfinal\s+(?:grade|decision|verdict)\s*[:=]\s*["\']?(correct|incorrect|partial|almost)["\']?', 8),
        (rf'\banswer\s+is\s+["\']?(correct|incorrect|partial|almost)["\']?', 6),
        (rf'\bresult\s*[:=]\s*["\']?(correct|incorrect|partial|almost)["\']?', 6),
        (rf'\bevaluation\s*[:=]\s*["\']?(correct|incorrect|partial|almost)["\']?', 6),
    ]
    
    for pattern, confidence in decision_patterns:
        match = re.search(pattern, text_lower)
        if match and confidence > best_confidence:
            best_match = match.group(1)
            best_confidence = confidence
    
    if best_match:
        return 'partial' if best_match == 'almost' else best_match
    
    # Priority 6: Count word-boundary occurrences with context weighting
    counts = {}
    context_weights = {
        'correct': 1.0,
        'incorrect': 1.0,
        'partial': 1.3,  # Boost partial as it's less commonly mentioned
        'almost': 1.3,   # Also boost almost (maps to partial)
    }
    
    for grade in valid_grades:
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
        # Extract fields from inputs for clearer prompting
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Parse grading guidelines to extract key evaluation criteria
        parsed_guidelines = self._parse_grading_guidelines(grading_guidelines)
        
        instruction = f"""You are an expert mathematics grader for International Mathematical Olympiad (IMO) problems.

Your task is to evaluate a student's answer to a mathematics problem and assign EXACTLY ONE of these three grades:
- "correct": The student's answer is completely correct, fully proves the statement, or completely solves the problem. All logical steps are valid and the conclusion is correct.
- "incorrect": The student's answer contains critical errors, logical fallacies, or fails to prove/solve the problem. The approach is fundamentally flawed or the conclusion is wrong.
- "partial": The student's answer demonstrates valid partial progress, contains correct lemmas, shows meaningful insights, or has the right approach but is incomplete or contains minor errors. This includes "almost" correct answers that have good ideas but small mistakes.

IMPORTANT GRADING RULES:
1. If the grading guidelines contain "(Almost)" or "almost correct" or "almost complete", you MUST grade as "partial".
2. "correct" requires a complete, valid proof or solution with no significant gaps or errors.
3. "partial" is for answers with good ideas but missing pieces, incomplete solutions, OR answers marked as "almost" in guidelines.
4. "incorrect" is for answers with fundamental errors, wrong conclusions, or approaches that cannot work.
5. When in doubt between "correct" and "partial", choose "partial" if there are any gaps or unclear steps.

SPECIFIC EVALUATION POINTS:
{parsed_guidelines}

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
4. Look for indicators in the grading guidelines (especially "almost" markers)
5. Determine the final grade based on the criteria above

You MUST respond with ONLY a JSON object in this exact format (no other text):
<json>{{"response": "correct"}}</json> or <json>{{"response": "incorrect"}}</json> or <json>{{"response": "partial"}}</json>

Do not include any explanation, markdown formatting, or additional text. Only output the JSON object wrapped in <json> tags."""

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
        
        # Strategy 1: Extract from <json> tags (new format we now request)
        try:
            extracted = _extract_jsons(response_text)
            if extracted and len(extracted) > 0:
                for item in extracted:
                    if isinstance(item, dict) and "response" in item:
                        val = item["response"]
                        if isinstance(val, str) and val.lower() in ["correct", "incorrect", "partial"]:
                            prediction = val.lower()
                            self.log_fn(f"Extracted prediction via <json> tags: {prediction}")
                            break
        except Exception as e:
            self.log_fn(f"Error in <json> tag extraction: {e}")
        
        # Strategy 2: Robust extraction (handles most formats)
        if prediction is None:
            try:
                result = _extract_response_robust(response_text)
                if result:
                    prediction = result
                    self.log_fn(f"Extracted prediction via robust extraction: {prediction}")
            except Exception as e:
                self.log_fn(f"Error in robust extraction: {e}")
        
        # Strategy 3: Flexible JSON extraction
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
        
        # Strategy 4: Direct response extraction
        if prediction is None:
            try:
                direct = _extract_response_direct(response_text)
                if direct:
                    prediction = direct
                    self.log_fn(f"Extracted prediction via direct extraction: {prediction}")
            except Exception as e:
                self.log_fn(f"Error in direct extraction: {e}")
        
        # Strategy 5: LLM-style extraction (handles common LLM output patterns)
        if prediction is None:
            try:
                llm_result = _extract_response_llm_style(response_text)
                if llm_result:
                    prediction = llm_result
                    self.log_fn(f"Extracted prediction via LLM-style extraction: {prediction}")
            except Exception as e:
                self.log_fn(f"Error in LLM-style extraction: {e}")
        
        # Validate the prediction
        if prediction not in ["correct", "incorrect", "partial"]:
            self.log_fn(f"Invalid or missing prediction: '{prediction}', defaulting to incorrect")
            prediction = "incorrect"
        
        return str(prediction), msg_history

    def _parse_grading_guidelines(self, guidelines: str) -> str:
        """Parse grading guidelines to extract key evaluation criteria.
        
        This method analyzes the grading guidelines to identify specific
        evaluation points that should be checked when grading the answer.
        
        Args:
            guidelines: The grading guidelines text
            
        Returns:
            A formatted string of specific evaluation points
        """
        if not guidelines:
            return "No specific guidelines provided. Use general mathematical reasoning."
        
        points = []
        almost_indicators = []  # Special handling for "almost" markers
        correct_indicators = []  # Special handling for "correct" markers
        incorrect_indicators = []  # Special handling for "incorrect" markers
        lines = guidelines.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            line_lower = line.lower()
            
            # Check for "almost" indicators first (highest priority)
            if '(almost)' in line_lower or 'almost correct' in line_lower or 'almost complete' in line_lower:
                almost_indicators.append(f"[GRADE AS PARTIAL] {line}")
                continue
            
            # Check for explicit correct indicators
            if '(correct)' in line_lower or line_lower.startswith('correct:') or line_lower.startswith('correct '):
                correct_indicators.append(f"[GRADE AS CORRECT] {line}")
                continue
            
            # Check for explicit incorrect indicators
            if '(incorrect)' in line_lower or line_lower.startswith('incorrect:') or line_lower.startswith('incorrect '):
                incorrect_indicators.append(f"[GRADE AS INCORRECT] {line}")
                continue
            
            # Check for partial indicators
            if '(partial)' in line_lower or line_lower.startswith('partial:') or line_lower.startswith('partial '):
                almost_indicators.append(f"[GRADE AS PARTIAL] {line}")
                continue
            
            # Look for numbered points or bullet points
            if line[0].isdigit() and ('.' in line[:3] or ')' in line[:3]):
                points.append(line)
            elif line.startswith('-') or line.startswith('*') or line.startswith('•'):
                points.append(line)
            # Look for key phrases indicating evaluation criteria
            elif any(keyword in line_lower for keyword in 
                    ['must', 'should', 'required', 'check', 'verify', 'ensure', 
                     'point', 'criterion', 'score', 'credit']):
                points.append(line)
        
        # Combine results: explicit indicators first, then other points
        all_points = almost_indicators + correct_indicators + incorrect_indicators + points
        
        # If we found structured points, format them
        if all_points:
            return '\n'.join(f"- {p}" for p in all_points[:15])  # Limit to first 15 points
        
        # Otherwise, return the first few non-empty lines
        non_empty = [l.strip() for l in lines if l.strip()][:7]
        if non_empty:
            return '\n'.join(f"- {line}" for line in non_empty)
        
        return guidelines[:600]  # Fallback: return truncated guidelines
