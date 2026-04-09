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
    """Extract grade from various formats in the text."""
    text_lower = text.lower()
    
    # Priority 1: Look for JSON grade field patterns with more flexible matching
    json_patterns = [
        r'"grade"\s*:\s*"([^"]*)"',
        r'"grade"\s*:\s*\'([^\']*)\'',
        r'"grade"\s*:\s*([a-zA-Z]+)',
        r'"response"\s*:\s*"([^"]*)"',
        r'"prediction"\s*:\s*"([^"]*)"',
        r'"result"\s*:\s*"([^"]*)"',
        r'"answer"\s*:\s*"([^"]*)"',
        r'"evaluation"\s*:\s*"([^"]*)"',
    ]
    
    for pattern in json_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            grade = match.group(1).strip().lower()
            if grade in ['correct', 'incorrect', 'partial', 'almost']:
                return grade
    
    # Priority 2: Look for explicit grade assignments in text
    text_patterns = [
        r'\bgrade\s*[:=]\s*"?([a-zA-Z]+)"?',
        r'\bfinal\s+grade\s*[:=]\s*"?([a-zA-Z]+)"?',
        r'\bassigned\s+grade\s*[:=]\s*"?([a-zA-Z]+)"?',
        r'\bthe\s+grade\s+is\s+"?([a-zA-Z]+)"?',
        r'\bi\s+assign\s+"?([a-zA-Z]+)"?',
        r'\bthis\s+is\s+"?([a-zA-Z]+)"?\b',
        r'\bgrade\s+is\s+"?([a-zA-Z]+)"?',
        r'\bevaluation\s*[:=]\s*"?([a-zA-Z]+)"?',
    ]
    
    for pattern in text_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            grade = match.group(1).strip().lower()
            if grade in ['correct', 'incorrect', 'partial', 'almost']:
                return grade
    
    # Priority 3: Look for grade keywords in specific phrases
    phrase_patterns = [
        (r'\bgrade\s+(?:should\s+be|is)\s+"?(almost|partial|incorrect|correct)"?\b', 1),
        (r'\b(?:the\s+)?student\s+(?:should\s+)?(?:receive|get|has)\s+(?:a\s+)?"?(almost|partial|incorrect|correct)"?\b', 1),
        (r'\bthis\s+(?:answer|solution)\s+is\s+"?(almost|partial|incorrect|correct)"?\b', 1),
    ]
    
    for pattern, group in phrase_patterns:
        match = re.search(pattern, text_lower)
        if match:
            grade = match.group(group).strip()
            if grade in ['correct', 'incorrect', 'partial', 'almost']:
                return grade
    
    # Priority 4: Look for standalone grade keywords in the last 100 words
    # This is where the conclusion typically appears
    words = text_lower.split()
    last_part = ' '.join(words[-100:])
    
    # Check in order of specificity (more specific first to avoid misclassification)
    if 'almost' in last_part:
        return 'almost'
    elif 'partial' in last_part:
        return 'partial'
    elif 'incorrect' in last_part:
        return 'incorrect'
    elif 'correct' in last_part:
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
    
    # Clean the text
    cleaned_text = text.strip()
    text_lower = cleaned_text.lower()
    
    # Strategy 1: Try to extract JSON objects (most reliable)
    extracted = _extract_jsons(cleaned_text)
    if extracted:
        # Check from last to first (most recent JSON is likely the answer)
        for json_obj in reversed(extracted):
            if isinstance(json_obj, dict):
                # Try common grade field names
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
    
    # Strategy 4: Look for quoted grades in the text (check in priority order)
    # Priority: almost > partial > incorrect > correct (more specific first)
    if '"almost"' in text_lower or "'almost'" in text_lower:
        return 'almost'
    elif '"partial"' in text_lower or "'partial'" in text_lower:
        return 'partial'
    elif '"incorrect"' in text_lower or "'incorrect'" in text_lower:
        return 'incorrect'
    elif '"correct"' in text_lower or "'correct'" in text_lower:
        return 'correct'
    
    # Strategy 5: Look for grade keywords in the last 100 words (expanded from 50)
    # The conclusion typically appears at the end
    words = text_lower.split()
    last_100 = ' '.join(words[-100:]) if len(words) > 100 else text_lower
    
    # Check in order of specificity (more specific grades first to avoid misclassification)
    if 'almost' in last_100:
        return 'almost'
    elif 'partial' in last_100:
        return 'partial'
    elif 'incorrect' in last_100:
        return 'incorrect'
    elif 'correct' in last_100:
        return 'correct'
    
    # Strategy 6: Look in the last 200 words as fallback
    if len(words) > 100:
        last_200 = ' '.join(words[-200:-100])
        if 'almost' in last_200:
            return 'almost'
        elif 'partial' in last_200:
            return 'partial'
        elif 'incorrect' in last_200:
            return 'incorrect'
        elif 'correct' in last_200:
            return 'correct'
    
    # Strategy 7: Look anywhere in text as last resort (specific first)
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

        instruction = f"""You are an expert mathematical olympiad grader. Your task is to evaluate a student's solution to a competition mathematics problem.

## Problem:
{problem}

## Official Solution:
{solution}

## Student's Answer:
{student_answer}

## Grading Guidelines:
{grading_guidelines}

## Grade Definitions (STUDY CAREFULLY)

**CORRECT**: The solution is COMPLETE and FULLY CORRECT.
- All required steps are present and logically sound
- The proof/solution is complete with no gaps
- Any calculations are correct (or errors are trivial and don't affect the result)
- The student has successfully solved the problem

**INCORRECT**: The solution shows NO MEANINGFUL PROGRESS or contains FUNDAMENTAL ERRORS.
- The approach is fundamentally wrong or misguided
- No valid mathematical progress toward the solution
- Contains critical logical flaws that invalidate the argument
- The student has NOT made significant progress
- Random calculations without valid approach = INCORRECT

**PARTIAL**: The student made SIGNIFICANT PROGRESS but the solution is INCOMPLETE.
- Found a useful invariant, lemma, or key insight
- Established a valid approach but didn't complete the proof
- Made meaningful progress but missing critical steps to finish
- Has the right idea but execution is incomplete
- The hard part (key insight) may be present, but the solution is not nearly complete

**ALMOST**: The solution is NEARLY COMPLETE with only MINOR ISSUES.
- The main proof structure is correct and complete
- Only minor computational errors, small omissions, or slight gaps
- The student essentially solved the problem but with small imperfections
- Much closer to correct than partial - the hard work is done
- If the solution would be correct with just a small fix, it's ALMOST

## Critical Distinctions (DECISION FRAMEWORK)

**PARTIAL vs INCORRECT**: 
- PARTIAL = meaningful progress (found invariant, proved lemma, valid approach started, key insight discovered)
- INCORRECT = no meaningful progress (wrong approach, no valid lemmas, fundamental misunderstanding, random calculations)
- KEY TEST: Did the student discover something useful? If yes → PARTIAL, if no → INCORRECT

**PARTIAL vs ALMOST**:
- PARTIAL = significant gaps remain, solution is incomplete, main result not reached
- ALMOST = solution is essentially complete, only minor issues, main result reached with small flaws
- KEY TEST: Is the solution nearly done? If yes → ALMOST, if no → PARTIAL

**ALMOST vs CORRECT**:
- ALMOST = minor errors or omissions exist (small arithmetic error, tiny gap in logic)
- CORRECT = completely correct with no meaningful errors
- KEY TEST: Would a small fix make it perfect? If yes → ALMOST, if already perfect → CORRECT

## Grading Process (FOLLOW STRICTLY)

1. **FIRST - Parse the grading guidelines carefully**: The guidelines explicitly list what constitutes each grade level. Look for:
   - (Partial) section: What specific achievements are listed?
   - (Almost) section: What specific criteria are listed?
   - (Correct) section: What complete solution criteria are listed?

2. **SECOND - Analyze the student's solution step by step**:
   - What key insights did they discover?
   - What lemmas or intermediate results did they prove?
   - Where does their solution stop or have errors?
   - Is the main proof structure complete?

3. **THIRD - Match against guidelines**:
   - Did they prove items listed under (Partial)? → Consider PARTIAL
   - Did they prove items listed under (Almost)? → Consider ALMOST  
   - Did they prove items listed under (Correct)? → Consider CORRECT
   - Did they make no valid progress? → Grade INCORRECT

4. **FOURTH - Apply the decision framework**:
   - Use the KEY TESTS above to distinguish between adjacent grades
   - Be especially careful with PARTIAL vs ALMOST (common confusion)

5. **FIFTH - Verify your grade**:
   - Re-read the guidelines to confirm your grade assignment aligns with the explicit criteria
   - Double-check: Does the student have a nearly complete solution (ALMOST) or just significant progress (PARTIAL)?

## Examples of Correct Grading

**Example 1 - PARTIAL grading:**
Grading Guidelines say:
(Partial)
 1. Found a useful invariant.
 2. Proved a key lemma.
(Almost)
 1. Nearly complete solution with minor gaps.

Student's answer: Proves the invariant and the key lemma but doesn't complete the main proof.
→ CORRECT grade: "partial" (student met the Partial criteria, solution is incomplete)

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
→ CORRECT grade: "almost" (student met Almost criteria, solution is nearly complete)

**Example 4 - ALMOST vs PARTIAL distinction:**
Student A: Has the right approach, proves some lemmas, but stops halfway through the main proof.
→ CORRECT grade: "partial" (significant progress but incomplete)

Student B: Completes the entire proof but has a small calculation error in the final step.
→ CORRECT grade: "almost" (nearly complete, just minor issue)

## Response Format (CRITICAL)

Respond ONLY with a JSON object in the following format. The grade field MUST be exactly one of: "correct", "incorrect", "partial", or "almost" (all lowercase, no extra spaces, no punctuation).

<json>
{{
    "grade": "correct" | "incorrect" | "partial" | "almost"
}}
</json>

IMPORTANT: Your entire response must be ONLY the JSON object above. Do not include any other text, explanation, or analysis outside the JSON tags.

Grade:"""

        max_retries = 3
        prediction = "none"
        msg_history = []
        current_instruction = instruction
        
        for attempt in range(max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=current_instruction,
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
                        current_instruction = instruction + f"""\n\nCRITICAL ERROR: Your previous response did not contain a valid grade in the required JSON format.
Your response was: {last_message[:300]}...

You MUST respond with ONLY a JSON object in this exact format (no other text):
<json>
{{"grade": "correct"}}
</json>

or

<json>
{{"grade": "incorrect"}}
</json>

or

<json>
{{"grade": "partial"}}
</json>

or

<json>
{{"grade": "almost"}}
</json>

The grade field MUST be exactly one of: "correct", "incorrect", "partial", or "almost" (all lowercase).
Do not include any other text, explanation, or formatting. Just the JSON object within <json> tags."""
                    else:
                        self.log_fn(f"Warning: Could not extract grade after {max_retries} attempts. Last response preview: {last_message[:200]}...")
                        
            except Exception as e:
                self.log_fn(f"Error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    prediction = "none"

        return str(prediction), msg_history
