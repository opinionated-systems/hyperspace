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
    """Extract JSON objects from <json>...</json> blocks or raw JSON.
    
    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also tries to find raw JSON objects if tags are not present.
    """
    if not text:
        return None
        
    results = []
    search_from = 0

    # First try to find <json>...</json> blocks
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

    # Also try to find ```json code blocks
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

    # If no results, try to find raw JSON objects (looking for balanced braces)
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
    """Extract grade from various formats in the text with improved patterns."""
    if not text:
        return None
        
    text_clean = text.strip()
    text_lower = text_clean.lower()
    
    # Priority 1: Look for JSON grade field patterns (most reliable)
    json_patterns = [
        r'"grade"\s*:\s*"([^"]*)"',
        r'"grade"\s*:\s*\'([^\']*)\'',
        r'"grade"\s*:\s*([a-zA-Z]+)',
        r'"response"\s*:\s*"([^"]*)"',
        r'"prediction"\s*:\s*"([^"]*)"',
        r'"result"\s*:\s*"([^"]*)"',
    ]
    
    for pattern in json_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            grade = match.group(1).strip().lower()
            if grade in VALID_GRADES:
                return grade
    
    # Priority 2: Look for explicit grade assignments in text
    text_patterns = [
        r'\bgrade\s*[:=]\s*["\']?([a-zA-Z]+)["\']?',
        r'\bfinal\s+grade\s*[:=]\s*["\']?([a-zA-Z]+)["\']?',
        r'\bassigned\s+grade\s*[:=]\s*["\']?([a-zA-Z]+)["\']?',
        r'\bthe\s+grade\s+is\s*["\']?([a-zA-Z]+)["\']?',
        r'\bi\s+assign\s*["\']?([a-zA-Z]+)["\']?',
        r'\bthis\s+is\s+([a-zA-Z]+)\b',
        r'\bverdict\s*[:=]\s*["\']?([a-zA-Z]+)["\']?',
        r'\bconclusion\s*[:=]\s*["\']?([a-zA-Z]+)["\']?',
    ]
    
    for pattern in text_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            grade = match.group(1).strip().lower()
            if grade in VALID_GRADES:
                return grade
    
    # Priority 3: Look for quoted grades (strong indicator)
    quoted_patterns = [
        (r'["\']almost["\']', 'almost'),
        (r'["\']partial["\']', 'partial'),
        (r'["\']incorrect["\']', 'incorrect'),
        (r'["\']correct["\']', 'correct'),
    ]
    for pattern, grade in quoted_patterns:
        if re.search(pattern, text_lower):
            return grade
    
    # Priority 4: Look for standalone grade keywords in the last 100 words
    # This is where the conclusion typically appears
    words = text_lower.split()
    last_part = ' '.join(words[-100:]) if len(words) > 100 else text_lower
    
    # Check in order of specificity (more specific first to avoid misclassification)
    # Use word boundaries to avoid matching "partial" inside "almost"
    if re.search(r'\balmost\b', last_part):
        return 'almost'
    elif re.search(r'\bpartial\b', last_part):
        return 'partial'
    elif re.search(r'\bincorrect\b', last_part):
        return 'incorrect'
    elif re.search(r'\bcorrect\b', last_part):
        return 'correct'
    
    return None


def _extract_grade(text: str) -> str:
    """Extract grade from LLM response text using multiple strategies."""
    if not text:
        return "none"
    
    # First try the improved text extraction
    result = _extract_grade_from_text(text)
    if result:
        return result
    
    # Try extracting from JSON blocks
    json_results = _extract_jsons(text)
    if json_results:
        for json_obj in reversed(json_results):  # Check from last to first
            if isinstance(json_obj, dict):
                # Try common grade field names
                for field in ["grade", "response", "result", "prediction"]:
                    if field in json_obj:
                        val = str(json_obj[field]).lower().strip().strip('"').strip("'")
                        if val in VALID_GRADES:
                            return val
                # If no standard field found, try any value
                for val in json_obj.values():
                    val_str = str(val).lower().strip().strip('"').strip("'")
                    if val_str in VALID_GRADES:
                        return val_str
    
    # Look for grade in the last 200 characters (where conclusion typically is)
    text_lower = text.lower()
    last_section = text_lower[-200:] if len(text_lower) > 200 else text_lower
    
    # Check in order of specificity in the last section first
    for grade in ["almost", "partial", "incorrect", "correct"]:
        if re.search(rf'\b{grade}\b', last_section):
            return grade
    
    # Last resort: look anywhere in text
    for grade in ["almost", "partial", "incorrect", "correct"]:
        if re.search(rf'\b{grade}\b', text_lower):
            return grade
    
    return "none"


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

        instruction = f"""You are an expert mathematical olympiad grader. Your task is to evaluate a student's solution and assign exactly one of four grades: "correct", "incorrect", "partial", or "almost".

## Problem Statement:
{problem}

## Official Solution:
{solution}

## Student's Answer:
{student_answer}

## Grading Guidelines (FOLLOW THESE EXACTLY):
{grading_guidelines}

## Grade Definitions (STRICT INTERPRETATION)

**CORRECT**: The solution is COMPLETE and FULLY CORRECT.
- All required steps are present and logically sound
- The proof/solution is complete with no gaps
- Any calculations are correct (or errors are trivial and don't affect the result)
- The student has successfully solved the problem
- Use this ONLY when the solution is essentially perfect

**INCORRECT**: The solution shows NO MEANINGFUL PROGRESS or contains FUNDAMENTAL ERRORS.
- The approach is fundamentally wrong or misguided
- No valid mathematical reasoning is presented
- The student completely misunderstood the problem
- Calculations are random or unrelated to the problem
- Use this for solutions that are essentially worthless
- BE STRICT: If there's no clear valid approach, grade INCORRECT
- KEY INDICATORS: Random formulas, no logical flow, fundamental misconceptions, no attempt at the actual problem

**PARTIAL**: The solution shows SIGNIFICANT PROGRESS but is INCOMPLETE.
- The student has made meaningful progress (30-75% complete)
- Key insights or lemmas may be proven
- BUT the MAIN RESULT is NOT fully proven
- This is the MOST COMMON grade for incomplete work
- CRITICAL: If the main theorem/result is not fully proven → MUST be "partial"
- Use this when there's good work but the solution doesn't reach the finish line
- DO NOT overuse this grade - ensure there IS meaningful progress first
- KEY INDICATORS: Valid approach started, some lemmas proven, clear understanding shown, but main claim not established

**ALMOST**: The solution is NEARLY COMPLETE with only MINOR issues.
- The main proof structure IS complete
- The main result IS proven
- Only small computational errors, minor gaps, or cosmetic issues remain
- The solution is 75-95% complete
- CRITICAL: The main result must be essentially proven
- Use this when the solution is very close to correct but has small flaws
- BE STRICT: Minor issues must truly be minor (don't affect main result)
- KEY INDICATORS: Main theorem stated and proven, solution structure complete, only small errors in calculations or minor logical gaps

## Decision Process (FOLLOW THIS ORDER)

1. **FIRST - Check for meaningful progress**:
   - Does the student show any valid mathematical reasoning?
   - Is there a coherent approach to the problem?
   - Look for: valid definitions, correct lemmas, logical structure, understanding of the problem
   - If NO meaningful progress → Grade: **incorrect**
   - BE STRICT: Random calculations or guesses = INCORRECT

2. **SECOND - Check if main result is FULLY proven**:
   - Is the main theorem/result completely proven?
   - Does the solution reach the final answer/conclusion?
   - If main result is NOT fully proven → Grade: **partial** (not "almost", not "correct")
   - This is the most important check - be strict here

3. **THIRD - Check completeness and errors**:
   - If main result IS proven, check for completeness:
     - 95-100% complete, no significant errors → Grade: **correct**
     - 75-95% complete, minor issues only → Grade: **almost**
     - Less than 75% complete → Grade: **partial**

4. **FOURTH - Verify against grading guidelines**:
   - Re-read the grading guidelines
   - Ensure your grade matches the explicit criteria in the guidelines
   - The guidelines are authoritative - follow them closely

## Critical Rules

1. **When in doubt, choose the LOWER grade** - it's better to be conservative
2. **If the main result is not proven, it CANNOT be "correct" or "almost"** - it MUST be "partial"
3. **"Partial" is for incomplete solutions with good progress** - this is very common
4. **"Almost" requires the main result to be essentially proven** - be strict about this
5. **"Incorrect" means NO meaningful progress** - be strict about valid reasoning
6. **Always verify your grade against the grading guidelines** before finalizing

## Examples of Correct Grading

**Example 1 - PARTIAL grading:**
Grading Guidelines say:
(Partial)
 1. Found a useful invariant.
 2. Proved a key lemma.
(Almost)
 1. Nearly complete solution with minor gaps.

Student's answer: Proves the invariant and the key lemma but doesn't complete the main proof.
→ CORRECT grade: "partial" (student met the Partial criteria, main result not proven)

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
→ CORRECT grade: "almost" (student met Almost criteria, main result proven)

**Example 4 - CORRECT vs PARTIAL distinction:**
Student proves several lemmas and makes progress but never proves the main theorem.
→ CORRECT grade: "partial" (main result not proven, regardless of how good the lemmas are)

**Example 5 - INCORRECT vs PARTIAL distinction:**
Student writes some formulas related to the problem but shows no understanding of how to solve it.
→ CORRECT grade: "incorrect" (no meaningful progress, just random formulas)

**Example 6 - ALMOST vs PARTIAL distinction:**
Student's answer: Presents a complete proof of the main theorem but makes a small arithmetic error in one calculation that doesn't affect the overall argument structure.
→ CORRECT grade: "almost" (main result proven, only minor computational error)

**Example 7 - ALMOST vs CORRECT distinction:**
Student's answer: Complete and correct proof with all steps logically sound, but has a minor notational inconsistency or a small gap that is easily filled.
→ CORRECT grade: "almost" (minor cosmetic issues prevent "correct")

**Example 8 - INCORRECT identification:**
Student's answer: Attempts to solve the problem but uses a theorem incorrectly, makes a fundamental logical error, or the approach is completely unrelated to the actual problem.
→ CORRECT grade: "incorrect" (fundamental errors, no valid reasoning)

**Example 9 - PARTIAL identification:**
Student's answer: Correctly identifies the key approach, proves one or two useful lemmas, but gets stuck and doesn't complete the proof of the main result.
→ CORRECT grade: "partial" (good progress, valid approach, but incomplete)

**Example 10 - CORRECT identification:**
Student's answer: Complete solution with all steps correct, logical flow is clear, all claims are properly justified, reaches the correct final answer.
→ CORRECT grade: "correct" (essentially perfect solution)

## Response Format (CRITICAL - MUST FOLLOW EXACTLY)

You MUST respond with ONLY a JSON object in this exact format. Do not include any other text before or after the JSON.

<json>
{{"grade": "correct"}}
</json>

OR

<json>
{{"grade": "incorrect"}}
</json>

OR

<json>
{{"grade": "partial"}}
</json>

OR

<json>
{{"grade": "almost"}}
</json>

The grade field MUST be exactly one of: "correct", "incorrect", "partial", or "almost" (all lowercase, no extra spaces or punctuation).

## Final Verification Checklist

Before outputting your grade, verify:
1. [ ] Did I check if there's meaningful progress? (If no → "incorrect")
2. [ ] Did I verify if the main result is proven? (If no → "partial")
3. [ ] If main result is proven, are there only minor issues? (If yes → "almost", if perfect → "correct")
4. [ ] Does my grade match the explicit criteria in the grading guidelines?
5. [ ] Am I being appropriately strict with the definitions?

Think through your reasoning step by step, then provide your final grade.

Your grade:"""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"Error getting LLM response: {e}")
            return "incorrect", []

        # Extract prediction
        prediction = "none"
        
        try:
            # Combine all possible sources of the response
            search_sources = []
            
            # First try to get the response from the msg_history
            if msg_history and len(msg_history) > 0:
                last_message = msg_history[-1].get("text", "")
                if last_message:
                    search_sources.append(("msg_history", last_message))
            
            # Also try the direct response
            if response:
                search_sources.append(("direct_response", response))
            
            # Try each source in order
            for source_name, source_text in search_sources:
                if not source_text:
                    continue
                    
                prediction = _extract_grade(source_text)
                
                if prediction != "none":
                    if source_name != "msg_history":
                        self.log_fn(f"Extracted grade from {source_name}: {prediction}")
                    break
            
            # If still none, log a warning with preview of what we got
            if prediction == "none" and search_sources:
                preview_text = search_sources[0][1][:500] if search_sources[0][1] else ""
                self.log_fn(f"Warning: Could not extract valid grade. Response preview: {preview_text}...")
                            
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            import traceback
            self.log_fn(f"Traceback: {traceback.format_exc()}")

        # Normalize prediction
        prediction = _normalize_grade(prediction)
        
        # Final validation
        if prediction not in VALID_GRADES:
            self.log_fn(f"Warning: Invalid prediction '{prediction}', defaulting to 'incorrect'")
            prediction = "incorrect"

        return str(prediction), msg_history
