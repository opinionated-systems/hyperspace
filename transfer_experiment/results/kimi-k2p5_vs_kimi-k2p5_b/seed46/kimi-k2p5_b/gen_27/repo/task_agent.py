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

# Valid grading labels for IMO evaluation - 3 class system
VALID_LABELS = ["correct", "partial", "incorrect"]


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks or raw JSON with enhanced robustness.
    
    Uses multiple strategies to find JSON content with improved handling of:
    - Nested braces and brackets
    - Trailing commas
    - Single quotes vs double quotes
    - Unicode characters
    - Malformed JSON structures
    - Markdown code blocks
    """
    if not text or not isinstance(text, str):
        return None
        
    results = []
    found_spans = []  # Track (start, end) of found JSON to avoid duplicates
    text = text.strip()
    
    def _is_duplicate(start: int, end: int) -> bool:
        """Check if this span overlaps with any already found."""
        for s, e in found_spans:
            if not (end <= s or start >= e):  # Overlapping
                return True
        return False
    
    def _add_result(parsed: dict, start: int, end: int) -> bool:
        """Add result if not duplicate and has expected keys."""
        if _is_duplicate(start, end):
            return False
        if isinstance(parsed, dict):
            if any(key in parsed for key in ['grade', 'response', 'label', 'evaluation', 'reasoning', 'result']):
                results.append(parsed)
                found_spans.append((start, end))
                return True
        return False
    
    # Strategy 1: Look for <json>...</json> tags (most reliable)
    search_from = 0
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            break
        inner_start = start + 6
        inner_end = end
        inner = text[inner_start:inner_end].strip()
        search_from = end + 7
        
        json_candidates = _generate_json_candidates(inner)
        
        for candidate in json_candidates:
            try:
                parsed = json.loads(candidate)
                if _add_result(parsed, start, end + 7):
                    break  # Success, move to next tag
            except json.JSONDecodeError:
                continue
    
    # Strategy 2: Look for JSON objects in markdown code blocks
    code_block_patterns = [
        (r'```json\s*\n?(.*?)\n?```', True),  # ```json ... ```
        (r'```\s*\n?(.*?)\n?```', False),     # ``` ... ```
    ]
    for pattern, is_json_block in code_block_patterns:
        for match in re.finditer(pattern, text, re.DOTALL):
            block = match.group(1).strip()
            if not block:
                continue
            
            # For non-json blocks, skip if it doesn't look like JSON
            if not is_json_block and not (block.startswith('{') or block.startswith('[')):
                continue
                
            json_candidates = _generate_json_candidates(block)
            
            for candidate in json_candidates:
                try:
                    parsed = json.loads(candidate)
                    if _add_result(parsed, match.start(), match.end()):
                        break
                except json.JSONDecodeError:
                    continue
    
    # Strategy 3: Look for raw JSON objects with balanced braces
    for start, end, json_str in _find_json_objects_with_spans(text):
        # Skip if this overlaps with already found spans
        if _is_duplicate(start, end):
            continue
            
        json_candidates = _generate_json_candidates(json_str)
        
        for candidate in json_candidates:
            try:
                parsed = json.loads(candidate)
                if _add_result(parsed, start, end):
                    break
            except json.JSONDecodeError:
                continue
    
    return results if results else None


def _generate_json_candidates(text: str) -> list[str]:
    """Generate multiple candidate versions of potentially malformed JSON."""
    candidates = [text]
    
    # Fix 1: Remove trailing commas before closing braces/brackets
    fixed = re.sub(r',(\s*[}\]])', r'\1', text)
    if fixed != text:
        candidates.append(fixed)
    
    # Fix 2: Replace single quotes with double quotes (carefully, not inside already quoted strings)
    # This is a simplified approach - replace ' that are not escaped with "
    fixed = re.sub(r"(?<!\\)'", '"', text)
    if fixed != text:
        candidates.append(fixed)
    
    # Fix 3: Remove comments (both // and /* */)
    fixed = re.sub(r'//.*?\n', '\n', text)
    fixed = re.sub(r'/\*.*?\*/', '', fixed, flags=re.DOTALL)
    if fixed != text:
        candidates.append(fixed)
    
    # Fix 4: Handle escaped quotes that might be double-escaped
    fixed = text.replace('\\"', '"').replace("\\'", "'")
    if fixed != text:
        candidates.append(fixed)
    
    # Fix 5: Remove control characters
    fixed = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
    if fixed != text:
        candidates.append(fixed)
    
    # Fix 6: Handle newlines in strings (replace with spaces)
    fixed = text.replace('\n', ' ').replace('\r', ' ')
    if fixed != text:
        candidates.append(fixed)
    
    return candidates


def _find_json_objects_with_spans(text: str) -> list[tuple[int, int, str]]:
    """Find potential JSON objects by tracking brace balance.
    
    Returns:
        List of (start_index, end_index, json_string) tuples
    """
    objects = []
    i = 0
    n = len(text)
    
    while i < n:
        if text[i] == '{':
            start = i
            brace_count = 1
            i += 1
            in_string = False
            escape_next = False
            
            while i < n and brace_count > 0:
                char = text[i]
                
                if escape_next:
                    escape_next = False
                    i += 1
                    continue
                
                if char == '\\':
                    escape_next = True
                    i += 1
                    continue
                
                if char == '"' and not in_string:
                    in_string = True
                elif char == '"' and in_string:
                    in_string = False
                elif not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                
                i += 1
                
            if brace_count == 0:
                end = i
                candidate = text[start:end]
                # Check if it looks like our expected JSON (has grading-related keys)
                lower_candidate = candidate.lower()
                if any(key in lower_candidate for key in ['grade', 'response', 'label', 'evaluation', 'reasoning', 'result']):
                    objects.append((start, end, candidate))
        else:
            i += 1
            
    return objects


def _extract_label_from_text(text: str) -> str | None:
    """Extract grading label from raw text using multiple strategies with priority ordering.
    
    Priority order (most specific to least specific):
    1. Explicit grade assignments ("grade is X", "grade: X")
    2. JSON-like structures without proper parsing
    3. Word boundary matches for each label
    """
    if not text or not isinstance(text, str):
        return None
        
    text_lower = text.lower()
    
    # Strategy 1: Look for explicit grade assignments with high confidence patterns
    # These patterns indicate a deliberate assignment of grade
    grade_patterns = [
        r'grade\s+is\s+["\']?(correct|partial|incorrect)["\']?',
        r'grade\s*[:=]\s*["\']?(correct|partial|incorrect)["\']?',
        r'assigned\s+grade\s*[:=]\s*["\']?(correct|partial|incorrect)["\']?',
        r'final\s+grade\s*[:=]\s*["\']?(correct|partial|incorrect)["\']?',
        r'evaluation\s*[:=]\s*["\']?(correct|partial|incorrect)["\']?',
        r'result\s*[:=]\s*["\']?(correct|partial|incorrect)["\']?',
        r'response\s*[:=]\s*["\']?(correct|partial|incorrect)["\']?',
        r'label\s*[:=]\s*["\']?(correct|partial|incorrect)["\']?',
        r'["\']?(correct|partial|incorrect)["\']?\s+is\s+appropriate',
        r'["\']?(correct|partial|incorrect)["\']?\s+is\s+the\s+(?:grade|evaluation)',
        r'["\']?(correct|partial|incorrect)["\']?\s+is\s+assigned',
        r'therefore\s*,?\s*(?:the\s+)?(?:grade|evaluation|answer)\s+is\s+["\']?(correct|partial|incorrect)["\']?',
    ]
    
    for pattern in grade_patterns:
        match = re.search(pattern, text_lower)
        if match:
            val = match.group(1).lower().strip()
            if val in VALID_LABELS:
                return val
    
    # Strategy 2: Look for label in quotes after specific keywords
    quote_patterns = [
        r'"(correct|partial|incorrect)"',
        r"'(correct|partial|incorrect)'",
        r'`(correct|partial|incorrect)`',
    ]
    for pattern in quote_patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1)
    
    # Strategy 3: Look for explicit label mentions with word boundaries
    # Check for "incorrect" first (most specific - contains "correct")
    if re.search(r'\bincorrect\b', text_lower):
        return "incorrect"
    
    # Check for "partial"
    if re.search(r'\bpartial\b', text_lower):
        return "partial"
    
    # Check for "correct" last (least specific) - but make sure it's not part of "incorrect"
    if re.search(r'\bcorrect\b', text_lower):
        # Double-check it's not "incorrect"
        if not re.search(r'\bincorrect\b', text_lower):
            return "correct"
    
    return None


def _normalize_grade(value: str) -> str | None:
    """Normalize a grade value to one of the valid labels with improved logic."""
    if value is None:
        return None
        
    if not isinstance(value, str):
        value = str(value)
    
    value = value.lower().strip()
    
    # Remove common prefixes/suffixes that might appear
    value = re.sub(r'^(is\s+|graded\s+as\s+|marked\s+as\s+|assigned\s+|final\s+|the\s+)', '', value)
    value = re.sub(r'\s*(grade|score|mark|evaluation|result)$', '', value)
    
    # Remove surrounding quotes and punctuation
    value = value.strip('"\'`,;:.! ')
    
    # Direct match
    if value in VALID_LABELS:
        return value
    
    # Check for exact matches with punctuation removed
    clean_value = re.sub(r'[^\w]', '', value)
    if clean_value in VALID_LABELS:
        return clean_value
    
    # Check for partial matches with priority (most specific first)
    # "incorrect" contains "correct" so check it first
    if "incorrect" in value:
        return "incorrect"
    if "partial" in value:
        return "partial"
    if "correct" in value:
        return "correct"
    
    # Handle common variations
    if value in ['right', 'true', 'valid', 'full', 'complete', '7', '7/7', 'full marks']:
        return "correct"
    if value in ['wrong', 'false', 'invalid', 'error', '0', '0/7', 'no credit', 'none']:
        return "incorrect"
    if value in ['partially', 'part', 'some', 'incomplete', 'half', 'partial credit']:
        return "partial"
    
    return None


def _validate_student_answer(student_answer: str) -> tuple[bool, str]:
    """Validate if the student answer contains meaningful content with expanded checks.
    
    Returns:
        (is_valid, reason): Tuple indicating if answer is valid and why it might not be
    """
    if not student_answer:
        return False, "empty"
    
    # Check if it's just whitespace or very short
    stripped = student_answer.strip()
    if not stripped:
        return False, "empty"
    
    if len(stripped) < 3:
        return False, "too_short"
    
    # Check for common "no answer" patterns (expanded list)
    no_answer_patterns = [
        r'^\s*no\s+answer\s*$',
        r'^\s*none\s*$',
        r'^\s*n/a\s*$',
        r'^\s*n\.a\.\s*$',
        r'^\s*not\s+applicable\s*$',
        r'^\s*skip\s*$',
        r'^\s*skipped\s*$',
        r'^\s*blank\s*$',
        r'^\s*empty\s*$',
        r'^\s*null\s*$',
        r'^\s*undefined\s*$',
        r'^\s*nil\s*$',
        r'^\s*idk\s*$',
        r'^\s*i\s+don\'t\s+know\s*$',
        r'^\s*i\s+do\s+not\s+know\s*$',
        r'^\s*no\s+solution\s*$',
        r'^\s*can\'t\s+solve\s*$',
        r'^\s*cannot\s+solve\s*$',
        r'^\s*unsure\s*$',
        r'^\s*unknown\s*$',
        r'^\s*unclear\s*$',
        r'^\s*pass\s*$',
        r'^\s*\?+\s*$',  # Just question marks
        r'^\s*\.+\s*$',  # Just dots
        r'^\s*-+\s*$',   # Just dashes
        r'^\s*_+\s*$',   # Just underscores
    ]
    
    for pattern in no_answer_patterns:
        if re.search(pattern, stripped, re.IGNORECASE):
            return False, "no_answer"
    
    # Check if answer is just whitespace and punctuation
    content_only = re.sub(r'[\s\W]', '', stripped)
    if len(content_only) < 2:
        return False, "insufficient_content"
    
    # Check for repeated single characters (like "aaaaaa" or "111111")
    if len(set(content_only.lower())) == 1:
        return False, "repetitive_content"
    
    return True, "valid"


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced accuracy."""

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
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Validate student answer first
        is_valid, validation_reason = _validate_student_answer(student_answer)
        if not is_valid:
            # Return incorrect for empty/invalid answers
            return "incorrect", [{"role": "system", "text": f"Invalid answer detected: {validation_reason}"}]
        
        instruction = f"""You are an expert mathematics grader for International Mathematical Olympiad (IMO) problems. Your task is to evaluate student answers and assign exactly one of three grades.

## GRADE DEFINITIONS (IMO SCORING SYSTEM):

**"correct"** (7 points): The answer is fully correct and complete. It contains a valid proof or solution with all necessary steps, logical reasoning, and reaches the correct conclusion. The student demonstrates full understanding of the problem. Only award this grade if the solution would receive full marks (7/7) in an actual IMO competition.

**"partial"** (1-6 points): The answer shows meaningful progress toward the solution with genuine mathematical insight, but is incomplete or has gaps. The student made non-trivial progress beyond just understanding the problem statement. Examples: found a key lemma but didn't complete the proof, solved a significant special case, or made substantial progress on a multi-part problem.

**"incorrect"** (0 points): The answer is wrong, fundamentally flawed, trivial, or shows no valid mathematical progress. The approach is completely wrong, the answer only restates the problem, or contains only trivial observations without real mathematical progress.

## GRADING GUIDELINES CONTEXT:
{grading_guidelines}

## PROBLEM:
{problem}

## OFFICIAL SOLUTION:
{solution}

## STUDENT'S ANSWER TO EVALUATE:
{student_answer}

## YOUR TASK:

Analyze the student's answer carefully following these steps:

1. **Understand the Problem**: Verify you understand what the problem is asking and what constitutes a correct solution.

2. **Review the Official Solution**: Note the key steps, techniques, and final answer required.

3. **Analyze the Student's Answer**: 
   - Check if the student addressed the actual problem asked
   - Identify any correct mathematical insights, lemmas, or techniques
   - Note any errors, gaps, or logical flaws
   - Assess the completeness of the solution
   - Check if the student reached a valid conclusion

4. **Compare to Guidelines**: Use the grading guidelines to determine the appropriate grade.

5. **Assign Grade**: Choose exactly one of: "correct", "partial", or "incorrect"

## GRADING CRITERIA SUMMARY:

- **Award "correct" ONLY if**: The solution is complete, rigorous, and would receive full marks (7/7)
- **Award "partial" if**: There's genuine mathematical progress (key insights, lemmas, special cases solved) but incomplete solution
- **Award "incorrect" if**: Wrong approach, no meaningful progress, trivial observations only, or fundamental flaws

## RESPONSE FORMAT (STRICT JSON REQUIRED):

You MUST respond with ONLY a JSON object wrapped in <json> tags. Do NOT include any other text, markdown formatting, or explanations outside the JSON tags.

<json>
{{
    "reasoning": "Detailed explanation of your evaluation including specific strengths and weaknesses found. Cite specific parts of the student's answer.",
    "grade": "correct"
}}
</json>

The "grade" field MUST be exactly one of: "correct", "partial", or "incorrect" (all lowercase, no quotes around the value, no extra punctuation).

Example valid responses:
<json>{{"reasoning": "The student provided a complete proof with all necessary steps.", "grade": "correct"}}</json>
<json>{{"reasoning": "The student found a key lemma but didn't complete the proof.", "grade": "partial"}}</json>
<json>{{"reasoning": "The approach is fundamentally flawed.", "grade": "incorrect"}}</json>"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction using multiple strategies with priority
        prediction = "None"
        try:
            raw_text = msg_history[-1]["text"] if msg_history else ""
            
            if not raw_text:
                self.log_fn("Empty response from LLM")
                return "incorrect", msg_history
            
            # Strategy 1: Try to extract from JSON tags (highest priority)
            extracted = _extract_jsons(raw_text)
            if extracted:
                for json_obj in extracted:
                    # Try multiple possible keys in order of preference
                    for key in ["grade", "label", "evaluation", "result", "response"]:
                        if key in json_obj:
                            val = json_obj[key]
                            if val is not None:
                                normalized = _normalize_grade(str(val))
                                if normalized:
                                    prediction = normalized
                                    break
                    if prediction != "None":
                        break
            
            # Strategy 2: If JSON extraction failed, try text extraction
            if prediction == "None":
                text_pred = _extract_label_from_text(raw_text)
                if text_pred:
                    prediction = text_pred
            
            # Strategy 3: Last resort - look for any valid label in the text
            if prediction == "None":
                text_lower = raw_text.lower()
                # Check in priority order (most specific first)
                # "incorrect" contains "correct" so check it first
                if re.search(r'\bincorrect\b', text_lower):
                    prediction = "incorrect"
                elif re.search(r'\bpartial\b', text_lower):
                    prediction = "partial"
                elif re.search(r'\bcorrect\b', text_lower):
                    # Make sure it's not "incorrect"
                    if not re.search(r'\bincorrect\b', text_lower):
                        prediction = "correct"
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Default to incorrect if we couldn't extract anything
        if prediction == "None":
            prediction = "incorrect"
            
        return str(prediction), msg_history
