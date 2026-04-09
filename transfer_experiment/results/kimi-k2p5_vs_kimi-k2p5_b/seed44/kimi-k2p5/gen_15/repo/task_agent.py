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
    """Extract grade from various text patterns with improved accuracy."""
    if not text:
        return None
        
    text_lower = text.lower()
    
    # Priority 1: JSON field patterns (most reliable)
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
    
    # Priority 2: Explicit grade declarations (strong indicators)
    explicit_patterns = [
        r'\bfinal\s+grade\s*[:=]\s*"?([a-zA-Z]+)"?',
        r'\bthe\s+grade\s+is\s+"?([a-zA-Z]+)"?',
        r'\bi\s+assign\s+the\s+grade\s+"?([a-zA-Z]+)"?',
        r'\bassigned\s+grade\s*[:=]\s*"?([a-zA-Z]+)"?',
        r'\bgrade\s+assigned\s*[:=]\s*"?([a-zA-Z]+)"?',
    ]
    
    for pattern in explicit_patterns:
        match = re.search(pattern, text_lower)
        if match:
            grade = match.group(1).strip().lower()
            if grade in VALID_GRADES:
                return grade
    
    # Priority 3: Quoted grades (strong indicators)
    for grade in VALID_GRADES:
        # Look for the grade in quotes
        if f'"{grade}"' in text_lower or f"'{grade}'" in text_lower:
            return grade
    
    # Priority 4: Look for grade in code blocks or at end of text
    # Check the last 200 words for grade mentions (more context than before)
    words = text_lower.split()
    last_part = ' '.join(words[-200:])
    
    # Check for grades in order of specificity (most specific first)
    # "almost" and "partial" are more specific than "correct" and "incorrect"
    if re.search(r'\balmost\b', last_part):
        return 'almost'
    elif re.search(r'\bpartial\b', last_part):
        return 'partial'
    elif re.search(r'\bincorrect\b', last_part):
        return 'incorrect'
    elif re.search(r'\bcorrect\b', last_part):
        return 'correct'
    
    # Priority 5: Look anywhere in text as last resort
    for grade in ["almost", "partial", "incorrect", "correct"]:
        if re.search(rf'\b{grade}\b', text_lower):
            return grade
    
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
    """Extract grade from LLM response text using multiple strategies with improved robustness."""
    if not text:
        return "none"
    
    # Clean the text - remove common formatting issues
    cleaned_text = text.strip()
    
    # Strategy 1: Try to extract JSON objects (most reliable)
    extracted = _extract_jsons(cleaned_text)
    if extracted:
        # Check from last to first (most recent JSON is likely the answer)
        for json_obj in reversed(extracted):
            if isinstance(json_obj, dict):
                # Try common grade field names in order of likelihood
                for field in ["grade", "result", "prediction", "response", "answer", "evaluation"]:
                    if field in json_obj:
                        val = str(json_obj[field]).lower().strip().strip('"').strip("'")
                        if val in VALID_GRADES:
                            return val
                # If no standard field found, try any string value in the JSON
                for key, val in json_obj.items():
                    if isinstance(val, str):
                        val_str = val.lower().strip().strip('"').strip("'")
                        if val_str in VALID_GRADES:
                            return val_str
    
    # Strategy 2: Try text-based extraction with improved patterns
    extracted_text = _extract_grade_from_text(cleaned_text)
    if extracted_text:
        return extracted_text
    
    # Strategy 3: Look for grades in markdown code blocks
    # Sometimes the model puts the grade in a code block without proper JSON
    code_block_pattern = r'```(?:json)?\s*\{?\s*"?grade"?\s*[:=]\s*"?([a-z]+)"?\s*\}?\s*```'
    match = re.search(code_block_pattern, cleaned_text.lower())
    if match:
        grade = match.group(1).strip()
        if grade in VALID_GRADES:
            return grade
    
    # Strategy 4: Look for unquoted grades in the last part of the text
    # Use word boundaries to avoid partial matches (e.g., "correctly" shouldn't match "correct")
    text_lower = cleaned_text.lower()
    words = text_lower.split()
    last_part = ' '.join(words[-150:])  # Check last 150 words
    
    # Check in order of specificity (most specific grades first)
    if re.search(r'\balmost\b', last_part):
        return 'almost'
    elif re.search(r'\bpartial\b', last_part):
        return 'partial'
    elif re.search(r'\bincorrect\b', last_part):
        return 'incorrect'
    elif re.search(r'\bcorrect\b', last_part):
        return 'correct'
    
    # Strategy 5: Look anywhere in text as last resort
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

## CRITICAL GRADING INSTRUCTIONS

You MUST follow this EXACT decision process:

### Step 1: Parse the Grading Guidelines Carefully
Read the grading guidelines and identify the SPECIFIC criteria listed under each category:
- What does "Partial" require? (list the exact criteria)
- What does "Almost" require? (list the exact criteria)  
- What does "Correct" require? (list the exact criteria)
- What does "Incorrect" indicate? (no meaningful criteria met)

### Step 2: Map Student's Work to Criteria
Go through the student's answer and check which criteria they have EXPLICITLY met:
- Did they prove what the Partial criteria asks for?
- Did they prove what the Almost criteria asks for?
- Did they prove what the Correct criteria asks for?

### Step 3: Apply the Hierarchical Decision Rule
Check grades in this STRICT ORDER (highest to lowest):

1. **Check for CORRECT**: Does the student meet ALL criteria for "Correct" in the guidelines?
   - If YES → Grade "correct"
   - If NO → Continue to check Almost

2. **Check for ALMOST**: Does the student meet ALL criteria for "Almost" in the guidelines?
   - If YES → Grade "almost"
   - If NO → Continue to check Partial

3. **Check for PARTIAL**: Does the student meet ANY criteria for "Partial" in the guidelines?
   - If YES → Grade "partial"
   - If NO → Grade "incorrect"

4. **Default to INCORRECT**: If student meets NONE of the above criteria.

### Step 4: Verify Against Common Errors
Before finalizing, check you are NOT making these common mistakes:

**DON'T be too strict on Partial:**
- If the student found a key lemma/invariant mentioned in Partial criteria → "partial" NOT "incorrect"
- If the student made significant progress on the core approach → "partial" NOT "incorrect"

**DON'T be too lenient on Almost:**
- "Almost" requires meeting the SPECIFIC Almost criteria, not just "being close"
- Missing a key step mentioned in Almost criteria → "partial" NOT "almost"

**DON'T be too lenient on Correct:**
- "Correct" requires a COMPLETE proof with no gaps
- Missing any substantial step → "almost" or "partial" NOT "correct"

## Detailed Examples

**Example 1 - PARTIAL (not incorrect):**
Guidelines: (Partial) 1. Found a useful invariant. (Almost) 1. Nearly complete solution.
Student: Proves the invariant but main proof incomplete.
→ Grade: "partial" (met Partial criteria, did NOT meet Almost criteria)

**Example 2 - INCORRECT:**
Guidelines: (Partial) 1. Found a useful invariant.
Student: Random calculations, no valid approach, no invariant found.
→ Grade: "incorrect" (met NO criteria)

**Example 3 - ALMOST (not partial):**
Guidelines: (Partial) 1. Found the key lemma. (Almost) 1. Proved the key lemma AND set up main argument.
Student: States AND proves the key lemma, sets up framework, but has gap in final step.
→ Grade: "almost" (met Almost criteria, went beyond Partial)

**Example 4 - PARTIAL (not almost):**
Guidelines: (Partial) 1. Found the key lemma. (Almost) 1. Proved the key lemma AND set up main argument.
Student: States the key lemma but doesn't prove it.
→ Grade: "partial" (met Partial criteria, did NOT meet Almost criteria)

**Example 5 - ALMOST (not correct):**
Guidelines: (Correct) 1. Complete rigorous proof. (Almost) 1. Complete proof with minor computational error.
Student: Complete proof with small arithmetic mistake.
→ Grade: "almost" (met Almost criteria, did NOT meet Correct criteria)

**Example 6 - PARTIAL (not correct):**
Guidelines: (Partial) 1. Observed and verified X has no solution. 2. Transformed equation to form Y.
Student: Did both 1 and 2 from Partial, but proof is incomplete.
→ Grade: "partial" (met Partial criteria, proof not complete enough for Correct)

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

        max_retries = 2
        prediction = "none"
        msg_history = []
        
        for attempt in range(max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                
                # Extract prediction from the last message
                last_message = msg_history[-1]["text"]
                prediction = _extract_grade(last_message)
                prediction = _normalize_grade(prediction)
                
                if prediction != "none":
                    # Successfully extracted a valid grade
                    break
                else:
                    # Could not extract grade, add clarification to prompt for retry
                    if attempt < max_retries - 1:
                        self.log_fn(f"Warning: Could not extract grade on attempt {attempt + 1}. Retrying with clarification...")
                        instruction += "\n\nIMPORTANT: Your previous response did not contain a valid grade. Please respond with ONLY a JSON object containing the grade field."
                    else:
                        self.log_fn(f"Warning: Could not extract grade after {max_retries} attempts. Last response preview: {last_message[:200]}...")
                        
            except Exception as e:
                self.log_fn(f"Error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    prediction = "none"

        return str(prediction), msg_history
