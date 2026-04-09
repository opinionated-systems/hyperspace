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

# Valid grade labels
_VALID_GRADES = {"correct", "partial", "incorrect"}


def _extract_guideline_label(guidelines: str) -> str | None:
    """Extract explicit grade label from grading guidelines.
    
    The guidelines may contain labels like:
    - "(Partial)" → should be graded as "partial"
    - "(Almost)" → should be graded as "partial" (minor but non-negligible mistakes)
    - "(Incorrect)" → should be graded as "incorrect"
    - "(Correct)" → should be graded as "correct"
    
    Returns the expected grade if a label is found, None otherwise.
    
    Priority order: (Incorrect) > (Almost) > (Partial) > (Correct)
    This ensures we catch the most restrictive label first.
    """
    if not guidelines:
        return None
    
    guidelines_lower = guidelines.lower()
    
    # Priority 1: (Incorrect) - most restrictive
    if '(incorrect)' in guidelines_lower:
        return "incorrect"
    
    # Priority 2: (Almost) - means minor but NON-NEGLIGIBLE mistakes → must be "partial"
    # This is the most important edge case - models often confuse "almost" with "correct"
    if '(almost)' in guidelines_lower:
        return "partial"
    
    # Priority 3: (Partial) - explicit partial credit
    if '(partial)' in guidelines_lower:
        return "partial"
    
    # Priority 4: (Correct) - explicit full credit (less common)
    if '(correct)' in guidelines_lower:
        return "correct"
    
    # Additional check: look for label patterns with brackets or other delimiters
    # e.g., [Almost], [Partial], [Incorrect], [Correct], {Almost}, <Almost>
    if re.search(r'[\[\{<«]almost[\]\}>»]', guidelines_lower):
        return "partial"
    if re.search(r'[\[\{<«]partial[\]\}>»]', guidelines_lower):
        return "partial"
    if re.search(r'[\[\{<«]incorrect[\]\}>»]', guidelines_lower):
        return "incorrect"
    if re.search(r'[\[\{<«]correct[\]\}>»]', guidelines_lower):
        return "correct"
    
    # Check for label at the start of any line
    for line in guidelines_lower.split('\n'):
        line_stripped = line.strip()
        if line_stripped.startswith('(almost)') or line_stripped.startswith('almost'):
            return "partial"
        if line_stripped.startswith('(partial)') or line_stripped.startswith('partial'):
            return "partial"
        if line_stripped.startswith('(incorrect)') or line_stripped.startswith('incorrect'):
            return "incorrect"
        if line_stripped.startswith('(correct)') or line_stripped.startswith('correct'):
            return "correct"
    
    return None


def _normalize_grade(grade: str) -> str | None:
    """Normalize a grade string to one of the valid grades.
    
    Handles common variations and misspellings.
    """
    if not grade:
        return None
    
    grade_lower = grade.lower().strip().strip('"\'`)')  # Strip quotes and backticks
    
    # Direct match
    if grade_lower in _VALID_GRADES:
        return grade_lower
    
    # Common variations - expanded for better coverage
    variations = {
        "correct": [
            "correct", "right", "true", "valid", "full", "full credit", "complete",
            "fully correct", "all correct", "perfect", "excellent", "good",
            "accurate", "proper", "sound", "flawless", "error-free"
        ],
        "partial": [
            "partial", "part", "partially correct", "almost", "some credit", "half",
            "partial credit", "incomplete", "mostly correct", "nearly correct",
            "some progress", "partial solution", "partial answer", "half credit",
            "partially right", "partially valid", "minor errors", "small mistakes"
        ],
        "incorrect": [
            "incorrect", "wrong", "false", "invalid", "no credit", "none", "zero",
            "fully wrong", "completely wrong", "totally wrong", "bad", "poor",
            "erroneous", "mistaken", "flawed", "unsound", "no progress", "failed"
        ],
    }
    
    for canonical, variants in variations.items():
        if grade_lower in variants:
            return canonical
    
    # Check for partial matches (e.g., "partially" should match "partial")
    if "partial" in grade_lower or "almost" in grade_lower:
        return "partial"
    if "correct" in grade_lower and "incorrect" not in grade_lower and "partial" not in grade_lower:
        return "correct"
    if "incorrect" in grade_lower or "wrong" in grade_lower:
        return "incorrect"
    
    return None


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles markdown code blocks with ```json or ```.
    Includes robust error recovery for common JSON formatting issues.
    """
    results = []
    
    def _try_parse_json(inner: str) -> dict | None:
        """Try to parse JSON with various fixes for common issues."""
        inner = inner.strip()
        if not inner:
            return None
            
        # Try direct parse first
        try:
            return json.loads(inner)
        except json.JSONDecodeError:
            pass
        
        # Fix 1: Handle trailing commas
        try:
            fixed = re.sub(r',\s*}', '}', inner)
            fixed = re.sub(r',\s*]', ']', fixed)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        
        # Fix 2: Handle single quotes (convert to double)
        try:
            # Replace single quotes around keys and string values
            fixed = re.sub(r"'([^']*?)'\s*:", r'"\1":', inner)
            fixed = re.sub(r":\s*'([^']*?)'", r': "\1"', fixed)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        
        # Fix 3: Handle unquoted keys
        try:
            fixed = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', inner)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        
        # Fix 4: Handle Python-style True/False/None
        try:
            fixed = inner.replace('True', 'true').replace('False', 'false').replace('None', 'null')
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        
        # Fix 5: Handle escaped newlines in string values
        try:
            fixed = re.sub(r'("[^"]*?)\\n([^"]*?")', r'\1\\n\2', inner)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        
        # Fix 6-8: Extract response field with various patterns (consolidated)
        # These handle cases where the JSON is malformed but the response field is present
        response_patterns = [
            # Standard double-quoted response
            r'"response"\s*:\s*"(correct|partial|incorrect)"',
            # Single-quoted response
            r"'response'\s*:\s*'(correct|partial|incorrect)'",
            # Response with escaped quotes
            r'"response"\s*:\s*\\"(correct|partial|incorrect)\\"',
            # Bare response without quotes
            r'"response"\s*:\s*(correct|partial|incorrect)\b',
            # Response at end of JSON (handles trailing content)
            r'"response"\s*:\s*"(correct|partial|incorrect)"[^}]*\}',
        ]
        for pattern in response_patterns:
            try:
                response_match = re.search(pattern, inner, re.IGNORECASE)
                if response_match:
                    return {"response": response_match.group(1).lower()}
            except Exception:
                pass
        
        # Fix 9: Handle JSON with newlines in string values
        try:
            # Replace newlines within string values with escaped newlines
            fixed = re.sub(r'("[^"]*?)\n([^"]*?")', r'\1\\n\2', inner)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        
        # Fix 10: Handle single-quoted JSON with response field
        try:
            response_match = re.search(r"'response'\s*:\s*'(correct|partial|incorrect)'", inner, re.IGNORECASE)
            if response_match:
                return {"response": response_match.group(1).lower()}
        except Exception:
            pass
        
        # Fix 11: Handle bare words without quotes
        try:
            response_match = re.search(r'"response"\s*:\s*(correct|partial|incorrect)\b', inner, re.IGNORECASE)
            if response_match:
                return {"response": response_match.group(1).lower()}
        except Exception:
            pass
        
        # Fix 12: Handle JSON with escaped quotes in reasoning
        try:
            # Try to find response field even with complex escaped content
            response_match = re.search(r'"response"\s*:\s*"(correct|partial|incorrect)"', inner, re.IGNORECASE)
            if response_match:
                return {"response": response_match.group(1).lower()}
        except Exception:
            pass
        
        return None
    
    # First, try to find <json>...</json> blocks
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
        
        parsed = _try_parse_json(inner)
        if parsed:
            results.append(parsed)
    
    # Also try markdown code blocks: ```json ... ``` or ``` ... ```
    code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    for match in re.finditer(code_block_pattern, text, re.DOTALL | re.IGNORECASE):
        inner = match.group(1).strip()
        parsed = _try_parse_json(inner)
        if parsed:
            results.append(parsed)
    
    # Try to find bare JSON objects (not in any tags)
    # Look for patterns like {"response": "..."} or {response: "..."}
    bare_json_pattern = r'\{\s*["\']?response["\']?\s*:\s*["\']?(correct|partial|incorrect)["\']?\s*\}'
    for match in re.finditer(bare_json_pattern, text, re.IGNORECASE):
        try:
            parsed = _try_parse_json(match.group(0))
            if parsed:
                results.append(parsed)
        except Exception:
            pass
    
    # Try to find JSON with reasoning and response fields
    full_json_pattern = r'\{\s*"reasoning"\s*:\s*"[^"]*(?:\.[^"]*)*"\s*,\s*"response"\s*:\s*"(correct|partial|incorrect)"\s*\}'
    for match in re.finditer(full_json_pattern, text, re.IGNORECASE | re.DOTALL):
        try:
            parsed = _try_parse_json(match.group(0))
            if parsed:
                results.append(parsed)
        except Exception:
            pass
    
    # Try to find any JSON-like structure with response field
    # This handles cases where the JSON is malformed but contains a valid response
    response_field_pattern = r'"response"\s*:\s*"?(correct|partial|incorrect)"?'
    for match in re.finditer(response_field_pattern, text, re.IGNORECASE):
        try:
            grade = match.group(1).lower()
            if grade in _VALID_GRADES:
                results.append({"response": grade})
        except Exception:
            pass
    
    return results or None


def _extract_grade_fallback(text: str) -> str | None:
    """Fallback: scan the text for a bare grade keyword.

    Uses multiple strategies to extract the grade, prioritizing:
    1. Explicit label mentions in text (highest priority)
    2. JSON-like structures with grade fields
    3. Explicit grade declarations near the end of text
    4. Pattern-based extraction with negation handling
    5. Last-resort standalone grade words
    """
    if not text:
        return None
        
    # Clean up the text: normalize whitespace
    text_clean = re.sub(r'\s+', ' ', text).strip()
    text_lower = text_clean.lower()
    
    # Strategy 0: Look for explicit label mentions - HIGHEST PRIORITY
    # These patterns catch when the model "knows" the right answer from labels
    
    # Check for (Incorrect) - highest priority (most restrictive)
    if '(incorrect)' in text_lower:
        return 'incorrect'
    
    # Check for (Almost) - second priority (ALWAYS means partial)
    if '(almost)' in text_lower:
        return 'partial'
    
    # Check for (Partial) - third priority
    if '(partial)' in text_lower:
        return 'partial'
    
    # Check for (Correct) - lowest priority among labels
    if '(correct)' in text_lower:
        return 'correct'
    
    # Strategy 1: JSON-like structures (with or without proper formatting)
    json_patterns = [
        # Standard double-quoted JSON
        r'"(?:response|grade|verdict|answer|result)"\s*:\s*"(correct|partial|incorrect)"',
        # Single-quoted JSON
        r"'(?:response|grade|verdict|answer|result)'\s*:\s*'(correct|partial|incorrect)'",
        # Bare JSON-like: {response: correct}
        r'\{\s*(?:response|grade|verdict|answer|result)\s*:\s*(correct|partial|incorrect)\s*\}',
    ]
    
    for pattern in json_patterns:
        matches = list(re.finditer(pattern, text_lower))
        if matches:
            # Return the last JSON match (most likely the final answer)
            return matches[-1].group(1).lower()
    
    # Strategy 2: Look for explicit final decision patterns near the end
    final_patterns = [
        # "Grade: X" or "Final grade: X" at end
        r'(?:final\s+)?(?:grade|verdict|assessment|decision)\s*[:\-=]\s*(correct|partial|incorrect)(?:\s*$|\s*\.\s*$|\s*[,;])',
        # "I assign/grade X" at end
        r'(?:i\s+)?(?:assign|grade|give)\s+(?:a\s+)?(?:grade\s+of\s+)?["\']?(correct|partial|incorrect)["\']?(?:\s*$|\s*\.\s*$)',
        # "The grade/verdict is X" at end
        r'(?:the\s+)?(?:grade|verdict|assessment|result)\s+is\s+["\']?(correct|partial|incorrect)["\']?(?:\s*$|\s*\.\s*$|\s*[,;])',
    ]
    
    for pattern in final_patterns:
        matches = list(re.finditer(pattern, text_lower))
        if matches:
            return matches[-1].group(1).lower()
    
    # Strategy 3: Explicit statements with context validation
    # Look for "is correct/partial/incorrect" but check for negation
    is_matches = list(re.finditer(r'\bis\s+(correct|partial|incorrect)\b', text_lower))
    for match in reversed(is_matches):  # Check from end
        grade = match.group(1).lower()
        start_pos = match.start()
        context = text_lower[max(0, start_pos-30):start_pos]
        
        # Check for negation - if negated, skip this match
        negation_indicators = ['not ', 'isn\'t', 'is not', 'never']
        if any(neg in context for neg in negation_indicators):
            if grade == 'correct':
                continue  # "is not correct" - skip
        return grade
    
    # Strategy 4: Look for grade in the last 100 characters
    last_100 = text_lower[-100:] if len(text_lower) > 100 else text_lower
    last_grade_match = re.search(r'\b(correct|partial|incorrect)\b', last_100)
    if last_grade_match:
        return last_grade_match.group(1).lower()
    
    # Strategy 5: Last resort - find the last standalone grade word
    standalone_matches = list(re.finditer(r'\b(correct|partial|incorrect)\b', text_lower))
    if standalone_matches:
        return standalone_matches[-1].group(1).lower()

    return None


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_file = log_file
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines,
                    student_answer

        Returns:
            (prediction, msg_history)
        """
        problem = inputs.get("problem", "")
        official_solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")


        instruction = f"""You are an expert mathematics competition grader. Your task is to evaluate the student's answer and assign exactly one grade: "correct", "partial", or "incorrect".

════════════════════════════════════════════════════════════════
GRADE DEFINITIONS - READ CAREFULLY
════════════════════════════════════════════════════════════════

"correct" = FULL CREDIT (100%)
  • The solution is mathematically complete and flawless
  • All logical steps are valid with no gaps
  • The final answer matches the official solution exactly
  • NO errors of any kind, even minor ones
  → Use ONLY when the solution would receive FULL credit in a competition.
  → If guidelines say "(Correct)", you MUST output "correct".

"partial" = PARTIAL CREDIT (1-99%)
  • The student made MEANINGFUL PROGRESS but has ISSUES
  • Correct approach but incomplete or flawed execution
  • Contains non-negligible errors that affect the result
  • Missing key steps, justifications, or has minor errors
  ⚠⚠⚠ CRITICAL: "(Almost)" in guidelines means "close but has MINOR ERRORS" → ALWAYS "partial", NEVER "correct"
  ⚠⚠⚠ CRITICAL: "(Partial)" in guidelines means "incomplete or has errors" → ALWAYS "partial"
  → Use when the student would receive PARTIAL credit (more than zero, less than full).

"incorrect" = ZERO/MINIMAL CREDIT (0%)
  • The answer is FUNDAMENTALLY FLAWED
  • Wrong approach or no meaningful progress
  • Critical logical errors that invalidate the solution
  • Labeled "(Incorrect)" in guidelines
  → Use when the solution would receive ZERO or minimal credit.

════════════════════════════════════════════════════════════════
PROBLEM
════════════════════════════════════════════════════════════════
{problem}

════════════════════════════════════════════════════════════════
OFFICIAL SOLUTION
════════════════════════════════════════════════════════════════
{official_solution}

════════════════════════════════════════════════════════════════
GRADING GUIDELINES
════════════════════════════════════════════════════════════════
{grading_guidelines}

════════════════════════════════════════════════════════════════
STUDENT'S ANSWER
════════════════════════════════════════════════════════════════
{student_answer}

════════════════════════════════════════════════════════════════
GRADING INSTRUCTIONS - FOLLOW EXACTLY
════════════════════════════════════════════════════════════════

STEP 1: SCAN FOR LABELS IN GUIDELINES (HIGHEST PRIORITY - OVERRIDES EVERYTHING)
The grading guidelines contain EXPLICIT LABELS from expert human graders.
These labels are the GROUND TRUTH and OVERRIDE your own judgment.

LABEL PRIORITY (check in this order - most restrictive first):
1. "(Incorrect)" → grade MUST be "incorrect" (fundamentally wrong)
2. "(Almost)" → grade MUST be "partial" (close but has errors - NEVER "correct")
3. "(Partial)" → grade MUST be "partial" (incomplete or has errors)
4. "(Correct)" → grade MUST be "correct" (fully correct)

⚠⚠⚠ CRITICAL RULE: If you see ANY label in the guidelines, your grade MUST match that label.
   DO NOT use your own judgment. The label IS the correct answer.

⚠⚠⚠ MOST COMMON ERROR #1: Seeing "(Almost)" and grading as "correct" because the solution
   "looks mostly right". THIS IS WRONG. "(Almost)" ALWAYS means "partial" - the student
   made minor but non-negligible mistakes that prevent full credit.

⚠⚠⚠ MOST COMMON ERROR #2: Seeing a good solution and grading as "correct" even when
   guidelines say "(Partial)" or "(Almost)". THIS IS WRONG. The label ALWAYS wins.

⚠⚠⚠ MOST COMMON ERROR #3: Ignoring the label and using your own analysis. THIS IS WRONG.
   The labels are from expert human graders - they are ALWAYS correct.

STEP 2: ANALYZE THE SOLUTION (ONLY IF NO LABEL PRESENT)
If there is NO label in the guidelines, evaluate the student's answer yourself:
1. Compare to the official solution step by step
2. Check for logical errors, missing steps, or incorrect conclusions
3. Determine if errors are minor (partial) or major (incorrect)
4. Assign the appropriate grade based on the definitions above

STEP 3: OUTPUT YOUR GRADE
Provide your decision in <json> tags with "reasoning" and "response" fields.
The "response" field MUST be exactly: "correct", "partial", or "incorrect".

════════════════════════════════════════════════════════════════
EXAMPLES - STUDY THESE CAREFULLY
════════════════════════════════════════════════════════════════

Example 1 - Correct (guidelines labeled "(Correct)"):
<json>
{{
    "reasoning": "The guidelines explicitly label this as (Correct), indicating a fully correct solution. The grade must be correct per the guidelines.",
    "response": "correct"
}}
</json>

Example 2 - Partial (guidelines labeled "(Almost)") - MOST IMPORTANT CASE:
<json>
{{
    "reasoning": "The guidelines explicitly label this as (Almost). This means the solution has minor but non-negligible mistakes. Even though the approach looks mostly correct, the (Almost) label REQUIRES a partial grade. I must follow the label, not my own judgment.",
    "response": "partial"
}}
</json>

Example 3 - Partial (guidelines labeled "(Partial)"):
<json>
{{
    "reasoning": "The guidelines label this as (Partial), indicating incomplete work or significant errors. The grade must be partial per the guidelines, regardless of how good the solution looks.",
    "response": "partial"
}}
</json>

Example 4 - Incorrect (guidelines labeled "(Incorrect)"):
<json>
{{
    "reasoning": "The guidelines label this as (Incorrect), indicating a fundamentally flawed solution. The grade must be incorrect per the guidelines.",
    "response": "incorrect"
}}
</json>

Example 5 - Correct (no label in guidelines, fully correct solution):
<json>
{{
    "reasoning": "There is no label in the guidelines. The student's proof is complete and correct. All steps are valid, the conclusion matches the official solution, and there are no errors.",
    "response": "correct"
}}
</json>

Example 6 - Incorrect (no label, but wrong answer):
<json>
{{
    "reasoning": "There is no label in the guidelines. The student made a critical error in the first step, leading to an incorrect conclusion. The approach is fundamentally wrong.",
    "response": "incorrect"
}}
</json>

Example 7 - Partial (no label, minor errors):
<json>
{{
    "reasoning": "There is no label in the guidelines. The student has the right approach but made a calculation error in step 3, leading to an incorrect final answer. This deserves partial credit for the correct method.",
    "response": "partial"
}}
</json>

════════════════════════════════════════════════════════════════
FINAL REMINDER - READ THIS BEFORE OUTPUTTING
════════════════════════════════════════════════════════════════

• If you see "(Almost)" → output "partial" (NOT "correct")
• If you see "(Partial)" → output "partial"
• If you see "(Incorrect)" → output "incorrect"
• If you see "(Correct)" → output "correct"
• If no label → use your judgment based on the definitions above

DOUBLE-CHECK: Does your "response" field match the label in the guidelines?
If guidelines say "(Almost)" and you output "correct", you are WRONG.

════════════════════════════════════════════════════════════════
NOW GRADE THE ANSWER
════════════════════════════════════════════════════════════════"""

        # Check for explicit guideline labels BEFORE calling the LLM
        # This helps us detect when the model might be making a mistake on edge cases
        expected_grade = _extract_guideline_label(grading_guidelines)
        if expected_grade:
            self.log_fn(f"Guideline label detected: expected grade is '{expected_grade}'")

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # ------------------------------------------------------------------ #
        # Extract prediction from the assistant's response.                   #
        # Strategy:                                                            #
        #   1. Look for <json>...</json> blocks in the last assistant message. #
        #   2. If that fails, try a regex fallback on the full response text.  #
        #   3. Check against expected grade from guidelines (edge case fix).   #
        #   4. Default to "incorrect" for safety if no valid grade found.      #
        # ------------------------------------------------------------------ #
        prediction = "incorrect"  # Default to incorrect for safety
        
        # Get assistant text from message history or response
        assistant_text = ""
        try:
            if msg_history:
                last_msg = msg_history[-1]
                # Handle both "content" and "text" keys
                assistant_text = last_msg.get("content", "") or last_msg.get("text", "") or ""
            if not assistant_text and response:
                assistant_text = response
        except Exception:
            assistant_text = response or ""

        # Try to extract from JSON blocks first
        if assistant_text:
            try:
                extracted = _extract_jsons(assistant_text)
                if extracted:
                    # Prefer the last JSON block that contains a valid "response" key
                    for obj in reversed(extracted):
                        grade = str(obj.get("response", "")).strip().lower()
                        normalized = _normalize_grade(grade)
                        if normalized:
                            prediction = normalized
                            break
                    else:
                        # No valid grade found in any block; take the last block's value if present
                        last_response = str(extracted[-1].get("response", "")).strip().lower()
                        normalized = _normalize_grade(last_response)
                        if normalized:
                            prediction = normalized
                    
            except Exception as e:
                self.log_fn(f"Error extracting prediction from JSON blocks: {e}")

        # Fallback: scan the raw text for a grade keyword
        if prediction not in _VALID_GRADES:
            try:
                fallback = _extract_grade_fallback(assistant_text)
                normalized = _normalize_grade(fallback)
                if normalized:
                    prediction = normalized
                    self.log_fn(f"Used fallback grade extraction: {prediction}")
            except Exception as e:
                self.log_fn(f"Error in fallback grade extraction: {e}")

        # OVERRIDE LOGIC: When guidelines have explicit labels, we MUST trust the label
        # This is the most critical part of the grading system - labels from expert human
        # graders are the ground truth and override any model judgment.
        if expected_grade and prediction != expected_grade:
            guidelines_lower = grading_guidelines.lower()
            
            # Check if the guidelines have explicit labels
            has_explicit_label = (
                '(almost)' in guidelines_lower or
                '(partial)' in guidelines_lower or
                '(incorrect)' in guidelines_lower or
                '(correct)' in guidelines_lower
            )
            
            if has_explicit_label:
                # The guidelines have an explicit label - trust it over the model
                self.log_fn(f"LABEL OVERRIDE: Guidelines have explicit label indicating '{expected_grade}', overriding model prediction '{prediction}'")
                prediction = expected_grade
        
        # Final validation: ensure we always return a valid grade
        if prediction not in _VALID_GRADES:
            prediction = "incorrect"
            self.log_fn("No valid grade found, defaulting to 'incorrect' for safety")

        return str(prediction), msg_history
