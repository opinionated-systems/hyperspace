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
    
    # Priority 1: JSON field patterns (most reliable) - check from end of text first
    # Find all occurrences and take the last one (most likely the final answer)
    json_patterns = [
        (r'"grade"\s*:\s*"([^"]*)"', 1),
        (r'"grade"\s*:\s*\'([^\']*)\'', 1),
        (r'"grade"\s*:\s*([a-zA-Z]+)', 1),
        (r'"response"\s*:\s*"([^"]*)"', 1),
        (r'"prediction"\s*:\s*"([^"]*)"', 1),
        (r'"result"\s*:\s*"([^"]*)"', 1),
    ]
    
    last_match_pos = -1
    last_match_grade = None
    
    for pattern, group in json_patterns:
        for match in re.finditer(pattern, text_lower):
            if match.start() > last_match_pos:
                grade = match.group(group).strip().lower()
                if grade in VALID_GRADES:
                    last_match_pos = match.start()
                    last_match_grade = grade
    
    if last_match_grade:
        return last_match_grade
    
    # Priority 2: Explicit grade declarations (strong indicators)
    explicit_patterns = [
        r'\bfinal\s+grade\s*[:=]\s*"?([a-zA-Z]+)"?',
        r'\bthe\s+grade\s+is\s+"?([a-zA-Z]+)"?',
        r'\bi\s+assign\s+the\s+grade\s+"?([a-zA-Z]+)"?',
        r'\bassigned\s+grade\s*[:=]\s*"?([a-zA-Z]+)"?',
        r'\bgrade\s+assigned\s*[:=]\s*"?([a-zA-Z]+)"?',
        r'\bgrade\s*[:=]\s*"?(correct|incorrect|partial|almost)"?\b',
    ]
    
    last_match_pos = -1
    last_match_grade = None
    
    for pattern in explicit_patterns:
        for match in re.finditer(pattern, text_lower):
            if match.start() > last_match_pos:
                grade = match.group(1).strip().lower()
                if grade in VALID_GRADES:
                    last_match_pos = match.start()
                    last_match_grade = grade
    
    if last_match_grade:
        return last_match_grade
    
    # Priority 3: Quoted grades in the last 100 words (strong indicators)
    words = text_lower.split()
    last_100 = ' '.join(words[-100:]) if len(words) > 100 else text_lower
    
    for grade in VALID_GRADES:
        # Look for the grade in quotes in the last part
        if f'"{grade}"' in last_100 or f"'{grade}'" in last_100:
            return grade
    
    # Priority 4: Look for grade in the last 50 words (most recent context)
    last_50 = ' '.join(words[-50:]) if len(words) > 50 else text_lower
    
    # Check for grades in order of specificity (most specific first)
    # "almost" and "partial" are more specific than "correct" and "incorrect"
    if re.search(r'\balmost\b', last_50):
        return 'almost'
    elif re.search(r'\bpartial\b', last_50):
        return 'partial'
    elif re.search(r'\bincorrect\b', last_50):
        return 'incorrect'
    elif re.search(r'\bcorrect\b', last_50):
        return 'correct'
    
    # Priority 5: Look in last 200 words
    last_200 = ' '.join(words[-200:]) if len(words) > 200 else text_lower
    
    if re.search(r'\balmost\b', last_200):
        return 'almost'
    elif re.search(r'\bpartial\b', last_200):
        return 'partial'
    elif re.search(r'\bincorrect\b', last_200):
        return 'incorrect'
    elif re.search(r'\bcorrect\b', last_200):
        return 'correct'
    
    # Priority 6: Look anywhere in text as last resort
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
    text_lower = cleaned_text.lower()
    
    # Strategy 1: Try to extract JSON objects (most reliable)
    # Look for the LAST JSON object in the text (most likely to be the final answer)
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
    code_block_patterns = [
        r'```(?:json)?\s*\{?\s*"?grade"?\s*[:=]\s*"?([a-z]+)"?\s*\}?\s*```',
        r'```\s*\{?\s*"?grade"?\s*[:=]\s*"?(correct|incorrect|partial|almost)"?\s*\}?\s*```',
    ]
    for pattern in code_block_patterns:
        match = re.search(pattern, text_lower)
        if match:
            grade = match.group(1).strip()
            if grade in VALID_GRADES:
                return grade
    
    # Strategy 4: Look for grade declarations in the last 100 words
    words = text_lower.split()
    last_100 = ' '.join(words[-100:]) if len(words) > 100 else text_lower
    
    # Check in order of specificity (most specific grades first)
    if re.search(r'\balmost\b', last_100):
        return 'almost'
    elif re.search(r'\bpartial\b', last_100):
        return 'partial'
    elif re.search(r'\bincorrect\b', last_100):
        return 'incorrect'
    elif re.search(r'\bcorrect\b', last_100):
        return 'correct'
    
    # Strategy 5: Look in last 200 words
    last_200 = ' '.join(words[-200:]) if len(words) > 200 else text_lower
    
    if re.search(r'\balmost\b', last_200):
        return 'almost'
    elif re.search(r'\bpartial\b', last_200):
        return 'partial'
    elif re.search(r'\bincorrect\b', last_200):
        return 'incorrect'
    elif re.search(r'\bcorrect\b', last_200):
        return 'correct'
    
    # Strategy 6: Look anywhere in text as last resort
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

## Grade Definitions (STRICT INTERPRETATION):
- **correct**: ONLY for COMPLETE, FLAWLESS proofs. Every step rigorous, all cases covered, NO gaps, NO errors whatsoever. This is the HIGHEST standard.
- **incorrect**: ONLY for solutions with NO meaningful progress toward the problem. Random calculations, completely wrong approach, or failed to meet ANY Partial criteria.
- **partial**: For solutions with MEANINGFUL progress. Student must have met AT LEAST ONE specific Partial criterion (found key lemma/invariant, made meaningful transformation, proved intermediate result). This is BROADER than you think - any substantive progress qualifies.
- **almost**: For solutions that are NEARLY complete with ONLY truly minor issues. The main proof structure is complete, and any errors are truly negligible (tiny computational slip that doesn't affect logic).

## CRITICAL GRADING INSTRUCTIONS - READ CAREFULLY

### Understanding the Grade Hierarchy
The grades form a hierarchy: correct > almost > partial > incorrect
- "correct": FLAWLESS complete proof (rare - any error disqualifies)
- "almost": Nearly flawless, tiny issues only
- "partial": Meaningful progress made (be GENEROUS here)
- "incorrect": No meaningful progress

### Step 1: Parse the Grading Guidelines Carefully
Read the grading guidelines and identify the SPECIFIC criteria listed under each category:
- What does "Partial" require? (these are MINIMUM requirements - be GENEROUS)
- What does "Almost" require? (must meet ALL of these - be STRICT)
- What does "Correct" require? (must be FLAWLESS - be VERY STRICT)

### Step 2: Map Student's Work to Criteria
Go through the student's answer and check which criteria they have EXPLICITLY met:
- Did they attempt what Partial criteria asks for? (even partial success counts!)
- Did they complete what Almost criteria asks for?
- Did they achieve a flawless Correct proof?

### Step 3: Apply the Hierarchical Decision Rule
Check grades in this STRICT ORDER (highest to lowest):

1. **Check for CORRECT**: Is this a FLAWLESS, COMPLETE proof?
   - EVERY step justified?
   - ALL cases covered?
   - ZERO errors or gaps?
   - If YES to ALL → Grade "correct"
   - If ANY doubt → Continue to check Almost (be HONEST about flaws)

2. **Check for ALMOST**: Is the solution NEARLY complete with ONLY tiny issues?
   - Main proof structure complete?
   - Errors are truly MINOR (small arithmetic, tiny gaps that don't affect main logic)?
   - If YES to ALL → Grade "almost"
   - If missing KEY steps or has SUBSTANTIAL gaps → Continue to check Partial

3. **Check for PARTIAL**: Did the student make MEANINGFUL progress?
   - Did they find a key lemma/invariant mentioned in Partial? (even if not fully proved)
   - Did they make a meaningful transformation mentioned in Partial?
   - Did they prove ANY intermediate result from Partial?
   - Did they demonstrate understanding of the core approach?
   - If YES to ANY → Grade "partial" (be GENEROUS - this is the broadest category)
   - If NO real progress → Grade "incorrect"

4. **Default to INCORRECT**: If student meets NONE of the above criteria.

### Step 4: Verify Against Common Grading Errors
Before finalizing, check you are NOT making these common mistakes:

**CRITICAL: Be GENEROUS with "partial"**
- "Partial" is for ANY meaningful progress, not just "being close"
- If student found a key insight from Partial criteria → "partial" NOT "incorrect"
- If student attempted the core approach → "partial" NOT "incorrect"
- When in doubt between "partial" and "incorrect", choose "partial"

**CRITICAL: Be STRICT with "correct"**
- "Correct" means FLAWLESS - any error means "almost" or lower
- Missing ANY substantial step → NOT "correct"
- If there are errors (even small ones) → "almost" NOT "correct"
- When in doubt between "correct" and "almost", choose "almost"

**CRITICAL: Don't confuse "almost" with "partial"**
- "Almost" = Solution is MOSTLY COMPLETE with only tiny issues (be STRICT)
- "Partial" = Made significant progress but solution is INCOMPLETE (be GENEROUS)
- If missing KEY step mentioned in Almost → "partial" NOT "almost"

**CRITICAL: Don't confuse "partial" with "incorrect"**
- "Incorrect" = NO meaningful progress (be STRICT about this)
- "Partial" = ANY meaningful progress (be GENEROUS about this)
- If student demonstrated understanding of the problem → "partial" NOT "incorrect"

## Detailed Examples

**Example 1 - PARTIAL (not incorrect):**
Guidelines: (Partial) 1. Found a correct invariant. (Almost) 1. Nearly complete solution.
Student: Proves the invariant but main proof incomplete.
→ Grade: "partial" (met Partial criteria, did NOT meet Almost criteria)

**Example 2 - INCORRECT:**
Guidelines: (Partial) 1. Found a correct invariant.
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

**Example 7 - CORRECT vs ALMOST:**
Guidelines: (Correct) Complete proof. (Almost) Minor mistakes only.
Student: Complete proof but has a small error in one calculation that doesn't affect conclusion.
→ Grade: "almost" (has an error, so not "correct")

**Example 8 - ALMOST vs PARTIAL:**
Guidelines: (Almost) Solution almost complete, minor mistakes only. (Partial) Considered a prime p|xy+1.
Student: Considered prime p|xy+1, made good progress, but omitted a case and has non-negligible mistakes.
→ Grade: "partial" (didn't meet "minor mistakes only" criterion for Almost)

**Example 9 - PARTIAL vs INCORRECT (BE GENEROUS):**
Guidelines: (Partial) 1. Considered a prime p|xy+1.
Student: Mentions considering primes dividing xy+1, sets up some equations, but doesn't complete the proof.
→ Grade: "partial" (made meaningful progress on the core approach, met Partial criterion)

**Example 10 - CORRECT vs ALMOST (BE STRICT):**
Guidelines: (Correct) Complete rigorous proof.
Student: Complete proof with rigorous structure, but has one small logical gap in a lemma.
→ Grade: "almost" (has a gap, so not "correct")

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

        max_retries = 3
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
                        # Add more specific guidance based on what we saw
                        instruction += f"""\n\nCRITICAL ERROR: Your previous response did not contain a valid grade in the required JSON format.
Your response was: {last_message[:300]}...

You MUST respond with ONLY a JSON object in this exact format (no other text):
{{"grade": "correct"}}

or

{{"grade": "incorrect"}}

or

{{"grade": "partial"}}

or

{{"grade": "almost"}}

The grade field MUST be exactly one of: "correct", "incorrect", "partial", or "almost" (all lowercase).
Do not include any other text, explanation, or formatting. Just the JSON object."""
                    else:
                        self.log_fn(f"Warning: Could not extract grade after {max_retries} attempts. Last response preview: {last_message[:200]}...")
                        
            except Exception as e:
                self.log_fn(f"Error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    prediction = "none"

        return str(prediction), msg_history
