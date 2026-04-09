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

VALID_GRADES = ["correct", "incorrect", "partial", "almost"]


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks or raw JSON."""
    results = []
    
    # Try <json>...</json> blocks
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

    # Try ```json code blocks
    search_from = 0
    while True:
        start = text.find("```json", search_from)
        if start == -1:
            break
        end = text.find("```", start + 7)
        if end == -1:
            break
        inner = text[start + 7:end].strip()
        search_from = end + 3
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            continue

    # Try raw JSON objects with balanced braces
    if not results:
        i = 0
        while i < len(text):
            if text[i] == '{':
                brace_count = 1
                j = i + 1
                while j < len(text) and brace_count > 0:
                    if text[j] == '{':
                        brace_count += 1
                    elif text[j] == '}':
                        brace_count -= 1
                    j += 1
                if brace_count == 0:
                    json_str = text[i:j].strip()
                    try:
                        results.append(json.loads(json_str))
                    except json.JSONDecodeError:
                        pass
                i = j
            else:
                i += 1

    return results or None


def _extract_grade_from_text(text: str) -> str | None:
    """Extract grade from various text patterns."""
    text_lower = text.lower()
    
    # Priority 1: JSON field patterns
    json_patterns = [
        r'"grade"\s*:\s*"([^"]*)"',
        r'"grade"\s*:\s*\'([^\']*)\'',
        r'"grade"\s*:\s*([a-zA-Z]+)',
        r'"response"\s*:\s*"([^"]*)"',
        r'"prediction"\s*:\s*"([^"]*)"',
        r'"result"\s*:\s*"([^"]*)"',
    ]
    
    for pattern in json_patterns:
        match = re.search(pattern, text_lower)
        if match:
            grade = match.group(1).strip().lower()
            if grade in VALID_GRADES:
                return grade
    
    # Priority 2: Explicit grade assignments
    text_patterns = [
        r'\bgrade\s*[:=]\s*"?([a-zA-Z]+)"?',
        r'\bfinal\s+grade\s*[:=]\s*"?([a-zA-Z]+)"?',
        r'\bassigned\s+grade\s*[:=]\s*"?([a-zA-Z]+)"?',
        r'\bthe\s+grade\s+is\s+"?([a-zA-Z]+)"?',
        r'\bi\s+assign\s+"?([a-zA-Z]+)"?',
        r'\bthis\s+is\s+"?([a-zA-Z]+)"?\b',
    ]
    
    for pattern in text_patterns:
        match = re.search(pattern, text_lower)
        if match:
            grade = match.group(1).strip().lower()
            if grade in VALID_GRADES:
                return grade
    
    # Priority 3: Quoted grades
    for grade in VALID_GRADES:
        if f'"{grade}"' in text_lower or f"'{grade}'" in text_lower:
            return grade
    
    # Priority 4: Last 100 words with word boundaries
    words = text_lower.split()
    last_part = ' '.join(words[-100:])
    
    if re.search(r'\balmost\b', last_part):
        return 'almost'
    elif re.search(r'\bpartial\b', last_part):
        return 'partial'
    elif re.search(r'\bincorrect\b', last_part):
        return 'incorrect'
    elif re.search(r'\bcorrect\b', last_part):
        return 'correct'
    
    return None


def _normalize_grade(grade: str) -> str:
    """Normalize grade to one of the expected lowercase values."""
    if not grade:
        return "none"
    
    grade_lower = grade.lower().strip().strip('"').strip("'")
    
    # Map to standard grades
    if grade_lower in VALID_GRADES:
        return grade_lower
    
    # Partial matches as fallback
    if "almost" in grade_lower:
        return "almost"
    elif "partial" in grade_lower:
        return "partial"
    elif "incorrect" in grade_lower:
        return "incorrect"
    elif "correct" in grade_lower:
        return "correct"
    
    return "none"


def _extract_grade(text: str) -> str:
    """Extract grade from LLM response text using multiple strategies."""
    if not text:
        return "none"
    
    # Strategy 1: Try to extract JSON objects (most reliable)
    extracted = _extract_jsons(text)
    if extracted:
        for json_obj in reversed(extracted):  # Check from last to first
            if isinstance(json_obj, dict):
                # Try common grade field names
                for field in ["grade", "response", "result", "prediction", "answer"]:
                    if field in json_obj:
                        val = str(json_obj[field]).lower().strip().strip('"').strip("'")
                        if val in VALID_GRADES:
                            return val
                # If no standard field found, try any value
                for val in json_obj.values():
                    val_str = str(val).lower().strip().strip('"').strip("'")
                    if val_str in VALID_GRADES:
                        return val_str
    
    # Strategy 2: Try text-based extraction
    extracted_text = _extract_grade_from_text(text)
    if extracted_text:
        return extracted_text
    
    # Strategy 3: Look for unquoted grades in the last part of the text
    # Use word boundaries to avoid partial matches (e.g., "correctly" shouldn't match "correct")
    text_lower = text.lower()
    last_part = ' '.join(text_lower.split()[-100:])
    
    # Check in order of specificity
    if re.search(r'\balmost\b', last_part):
        return 'almost'
    elif re.search(r'\bpartial\b', last_part):
        return 'partial'
    elif re.search(r'\bincorrect\b', last_part):
        return 'incorrect'
    elif re.search(r'\bcorrect\b', last_part):
        return 'correct'
    
    # Strategy 4: Look anywhere in text as last resort
    for grade in ["almost", "partial", "incorrect", "correct"]:
        if re.search(rf'\b{grade}\b', text_lower):
            return grade
    
    return "none"


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
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematical olympiad grader. Your task is to grade the student's solution based on the official solution and grading guidelines.

## Problem:
{problem}

## Official Solution:
{solution}

## Student's Answer:
{student_answer}

## Grading Guidelines (FOLLOW THESE EXACTLY):
{grading_guidelines}

## Grade Definitions:
- **correct**: Complete, correct solution with valid proof
- **incorrect**: Wrong, fundamental errors, no meaningful progress
- **partial**: Significant progress but incomplete (found invariant/lemma but didn't finish)
- **almost**: Nearly complete, only minor errors/gaps

## Grading Process (Follow This Order):

1. **FIRST - Parse the grading guidelines**: Identify what specific criteria are listed under each grade category (Partial, Almost, Correct, Incorrect).

2. **SECOND - Analyze the student's answer step by step**: 
   - Compare the student's work against the official solution
   - Identify which criteria from the guidelines the student has met
   - Check if student met criteria for "Correct" → Grade CORRECT
   - Else check if student met criteria for "Almost" → Grade ALMOST
   - Else check if student met criteria for "Partial" → Grade PARTIAL
   - Else → Grade INCORRECT

3. **THIRD - Verify your grade matches the guidelines**:
   - Re-read the guidelines to confirm your grade assignment aligns with the explicit criteria
   - Ensure you are not being too lenient or too strict

4. **Final decision**: Choose the grade that matches the grading guidelines criteria.

## Key Grading Principles

- **"correct"**: Only for complete, rigorous proofs. The solution must be fully correct with no gaps.
- **"almost"**: For solutions that are nearly complete - the main ideas are there, structure is sound, only minor gaps or errors exist. The student understood the core approach.
- **"partial"**: For solutions with significant progress but incomplete - found key insights, lemmas, or invariants but didn't complete the proof. The student made meaningful progress but didn't finish.
- **"incorrect"**: For solutions with no meaningful progress, fundamental errors, or wrong approach. The student did not demonstrate understanding of the key ideas.

## Examples of Correct Grading

**Example 1 - PARTIAL grading:**
Grading Guidelines say:
(Partial)
 1. Found a useful invariant.
 2. Proved a key lemma.
(Almost)
 1. Nearly complete solution with minor gaps.

Student's answer: Proves the invariant and the key lemma but doesn't complete the main proof.
→ CORRECT grade: "partial" (student met the Partial criteria)

**Example 2 - INCORRECT grading:**
Grading Guidelines say:
(Partial)
 1. Found a useful invariant.

Student's answer: Makes some calculations but no valid approach, no invariant found, fundamental misunderstanding.
→ CORRECT grade: "incorrect" (student did not meet any meaningful criteria)

**Example 3 - ALMOST grading:**
Grading Guidelines say:
(Almost)
 1. Complete proof with minor computational error.

Student's answer: Complete proof with a small arithmetic mistake that doesn't affect the main argument.
→ CORRECT grade: "almost" (student met Almost criteria)

**Example 4 - CORRECT grading:**
Grading Guidelines say:
(Correct)
 1. Complete and rigorous proof.

Student's answer: Provides a full, valid proof with all steps justified.
→ CORRECT grade: "correct" (student met Correct criteria)

**Example 5 - ALMOST vs PARTIAL distinction:**
Grading Guidelines say:
(Partial)
 1. Found the key lemma.
(Almost)
 1. Proved the key lemma and set up the main argument.

Student's answer: States and proves the key lemma, sets up the framework for the main proof, but has a gap in the final step.
→ CORRECT grade: "almost" (student went beyond Partial by proving the lemma and setting up the argument)

## Response Format (CRITICAL)

You MUST respond with ONLY a JSON object in this exact format:

```json
{{
    "grade": "correct"
}}
```

OR

```json
{{
    "grade": "incorrect"
}}
```

OR

```json
{{
    "grade": "partial"
}}
```

OR

```json
{{
    "grade": "almost"
}}
```

The grade field MUST be exactly one of: "correct", "incorrect", "partial", or "almost" (all lowercase, no extra spaces, no other text).

Grade:"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from the last message
        prediction = "none"
        try:
            last_message = msg_history[-1]["text"]
            prediction = _extract_grade(last_message)
            # Normalize the prediction
            prediction = _normalize_grade(prediction)
            
            # Log for debugging
            if prediction == "none":
                self.log_fn(f"Warning: Could not extract grade from response. Response preview: {last_message[:200]}...")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
