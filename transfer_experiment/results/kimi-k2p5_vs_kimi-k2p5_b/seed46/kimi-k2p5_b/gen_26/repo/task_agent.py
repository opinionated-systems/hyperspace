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
    """
    results = []
    
    # Strategy 1: Look for <json>...</json> tags (most reliable)
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
        
        # Try multiple fixes for common JSON issues
        json_candidates = [inner]
        
        # Fix 1: Remove trailing commas before closing braces/brackets
        fixed1 = re.sub(r',(\s*[}\]])', r'\1', inner)
        if fixed1 != inner:
            json_candidates.append(fixed1)
        
        # Fix 2: Replace single quotes with double quotes (carefully)
        fixed2 = re.sub(r"(?<!\\)'", '"', inner)
        if fixed2 != inner:
            json_candidates.append(fixed2)
        
        # Fix 3: Remove comments (both // and /* */)
        fixed3 = re.sub(r'//.*?\n', '\n', inner)
        fixed3 = re.sub(r'/\*.*?\*/', '', fixed3, flags=re.DOTALL)
        if fixed3 != inner:
            json_candidates.append(fixed3)
        
        # Fix 4: Handle escaped characters better
        fixed4 = inner.replace('\\"', '"').replace("\\'", "'")
        if fixed4 != inner:
            json_candidates.append(fixed4)
        
        for candidate in json_candidates:
            try:
                results.append(json.loads(candidate))
                break  # Success, move to next tag
            except json.JSONDecodeError:
                continue
    
    # Strategy 2: Look for JSON objects in code blocks
    json_block_pattern = re.findall(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    for block in json_block_pattern:
        block = block.strip()
        if not block:
            continue
            
        json_candidates = [block]
        
        # Fix trailing commas
        fixed1 = re.sub(r',(\s*[}\]])', r'\1', block)
        if fixed1 != block:
            json_candidates.append(fixed1)
        
        # Fix single quotes
        fixed2 = re.sub(r"(?<!\\)'", '"', block)
        if fixed2 != block:
            json_candidates.append(fixed2)
        
        for candidate in json_candidates:
            try:
                results.append(json.loads(candidate))
                break
            except json.JSONDecodeError:
                continue
    
    # Strategy 3: Look for raw JSON objects with balanced braces
    # Use a more sophisticated approach to handle nested structures
    def find_json_objects(text: str) -> list[str]:
        """Find potential JSON objects by tracking brace balance."""
        objects = []
        i = 0
        while i < len(text):
            if text[i] == '{':
                start = i
                brace_count = 1
                i += 1
                while i < len(text) and brace_count > 0:
                    if text[i] == '{':
                        brace_count += 1
                    elif text[i] == '}':
                        brace_count -= 1
                    elif text[i] == '"':
                        # Skip string content
                        i += 1
                        while i < len(text) and text[i] != '"':
                            if text[i] == '\\' and i + 1 < len(text):
                                i += 2
                            else:
                                i += 1
                        continue
                    i += 1
                if brace_count == 0:
                    candidate = text[start:i]
                    # Check if it looks like our expected JSON
                    if any(key in candidate.lower() for key in ['grade', 'response', 'label', 'evaluation', 'reasoning']):
                        objects.append(candidate)
            i += 1
        return objects
    
    for json_str in find_json_objects(text):
        json_candidates = [json_str]
        
        # Fix trailing commas
        fixed1 = re.sub(r',(\s*[}\]])', r'\1', json_str)
        if fixed1 != json_str:
            json_candidates.append(fixed1)
        
        # Fix single quotes
        fixed2 = re.sub(r"(?<!\\)'", '"', json_str)
        if fixed2 != json_str:
            json_candidates.append(fixed2)
        
        for candidate in json_candidates:
            try:
                parsed = json.loads(candidate)
                # Only add if it has expected keys
                if any(key in parsed for key in ['grade', 'response', 'label', 'evaluation']):
                    results.append(parsed)
                    break
            except json.JSONDecodeError:
                continue
    
    return results or None


def _extract_label_from_text(text: str) -> str | None:
    """Extract grading label from raw text using multiple strategies with priority ordering.
    
    Priority order (most specific to least specific):
    1. Explicit grade assignments ("grade is X", "grade: X")
    2. JSON-like structures without proper parsing
    3. Word boundary matches for each label
    """
    text_lower = text.lower()
    
    # Strategy 1: Look for explicit grade assignments with high confidence patterns
    # These patterns indicate a deliberate assignment of grade
    grade_patterns = [
        r'grade\s+is\s+["\']?(correct|partial|incorrect)["\']?',
        r'grade\s*[:=]\s*["\']?(correct|partial|incorrect)["\']?',
        r'assigned\s+grade\s*[:=]\s*["\']?(correct|partial|incorrect)["\']?',
        r'final\s+grade\s*[:=]\s*["\']?(correct|partial|incorrect)["\']?',
        r'evaluation\s*[:=]\s*["\']?(correct|partial|incorrect)["\']?',
        r'response\s*[:=]\s*["\']?(correct|partial|incorrect)["\']?',
        r'["\']?(correct|partial|incorrect)["\']?\s+is\s+appropriate',
        r'["\']?(correct|partial|incorrect)["\']?\s+is\s+the\s+(?:grade|evaluation)',
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
    ]
    for pattern in quote_patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1)
    
    # Strategy 3: Look for explicit label mentions with word boundaries
    # Check for "incorrect" first (most specific - contains "correct")
    if re.search(r'\bin\s*correct\b|\bincorrect\b', text_lower):
        return "incorrect"
    
    # Check for "partial"
    if re.search(r'\bpartial\b', text_lower):
        return "partial"
    
    # Check for "correct" last (least specific)
    if re.search(r'\bcorrect\b', text_lower):
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
    value = re.sub(r'^(is\s+|graded\s+as\s+|marked\s+as\s+|assigned\s+|final\s+)', '', value)
    value = re.sub(r'\s*(grade|score|mark)$', '', value)
    
    # Direct match
    if value in VALID_LABELS:
        return value
    
    # Check for exact matches with punctuation
    clean_value = re.sub(r'[^\w]', '', value)
    if clean_value in VALID_LABELS:
        return clean_value
    
    # Check for partial matches with priority (most specific first)
    if "incorrect" in value:
        return "incorrect"
    if "partial" in value:
        return "partial"
    if "correct" in value:
        return "correct"
    
    return None


def _validate_student_answer(student_answer: str) -> tuple[bool, str]:
    """Validate if the student answer contains meaningful content with expanded checks."""
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
        r'^\s*not\s+applicable\s*$',
        r'^\s*skip\s*$',
        r'^\s*skipped\s*$',
        r'^\s*blank\s*$',
        r'^\s*empty\s*$',
        r'^\s*null\s*$',
        r'^\s*undefined\s*$',
        r'^\s*idk\s*$',
        r'^\s*i\s+don\'t\s+know\s*$',
        r'^\s*no\s+solution\s*$',
        r'^\s*can\'t\s+solve\s*$',
        r'^\s*unsure\s*$',
        r'^\s*unknown\s*$',
        r'^\s*\?+\s*$',  # Just question marks
        r'^\s*\.+\s*$',  # Just dots
    ]
    
    for pattern in no_answer_patterns:
        if re.search(pattern, stripped, re.IGNORECASE):
            return False, "no_answer"
    
    # Check if answer is just whitespace and punctuation
    content_only = re.sub(r'[\s\W]', '', stripped)
    if len(content_only) < 2:
        return False, "insufficient_content"
    
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

## GRADE DEFINITIONS (STRICT INTERPRETATION):

**"correct"**: The answer is fully correct and complete. It contains a valid proof or solution with all necessary steps. The student demonstrates full understanding of the problem and provides a rigorous solution. Only award this grade if the solution would receive full marks in an actual IMO competition.

**"partial"**: The answer has some correct elements and shows meaningful progress toward the solution, but is incomplete or has significant gaps. The student understood part of the problem and made non-trivial progress, but did not reach a complete solution. Partial credit applies when there's genuine mathematical insight but incomplete execution. The answer should show substantial progress beyond just understanding the problem statement.

**"incorrect"**: The answer is wrong, fundamentally flawed, or trivial. No valid mathematical progress toward the solution. The approach is completely wrong, or the answer is blank/empty. No meaningful understanding demonstrated. This includes answers that only restate the problem or provide trivial observations without real progress.

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
   - Identify any correct mathematical insights or techniques
   - Note any errors, gaps, or logical flaws
   - Assess the completeness of the solution
   - Check if the student reached a valid conclusion

4. **Compare to Guidelines**: Use the grading guidelines to determine partial vs incorrect distinctions. Pay special attention to what constitutes "partial" vs "incorrect" in the guidelines.

5. **Assign Grade**: Choose exactly one of: "correct", "partial", or "incorrect"

## IMPORTANT RULES:

- Be CONSERVATIVE with "correct" - only award for complete, rigorous solutions
- "partial" requires genuine mathematical progress, not just understanding the problem
- "incorrect" is for wrong answers, trivial observations, or no meaningful progress
- When in doubt between "partial" and "incorrect", prefer "incorrect" unless clear progress is shown

## RESPONSE FORMAT:

You MUST respond with ONLY a JSON object in <json> tags. Do NOT include any other text before or after the JSON.

<json>
{{
    "reasoning": "Detailed explanation of your evaluation including specific strengths and weaknesses found. Explain why you chose this grade.",
    "grade": "correct" | "partial" | "incorrect"
}}
</json>

The "grade" field MUST be exactly one of: "correct", "partial", or "incorrect" (all lowercase, no quotes around the value in the JSON)."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction using multiple strategies with priority
        prediction = "None"
        try:
            raw_text = msg_history[-1]["text"] if msg_history else ""
            
            # Strategy 1: Try to extract from JSON tags (highest priority)
            extracted = _extract_jsons(raw_text)
            if extracted:
                for json_obj in extracted:
                    # Try multiple possible keys in order of preference
                    for key in ["grade", "response", "label", "evaluation"]:
                        if key in json_obj:
                            val = json_obj[key]
                            normalized = _normalize_grade(val)
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
                if "incorrect" in text_lower:
                    prediction = "incorrect"
                elif "partial" in text_lower:
                    prediction = "partial"
                elif "correct" in text_lower:
                    prediction = "correct"
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
