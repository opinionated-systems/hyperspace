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
    # Try to find the LAST occurrence (most likely to be the final answer)
    json_patterns = [
        r'"grade"\s*:\s*"([^"]*)"',
        r'"grade"\s*:\s*\'([^\']*)\'',
        r'"grade"\s*:\s*([a-zA-Z]+)',
        r'"response"\s*:\s*"([^"]*)"',
        r'"prediction"\s*:\s*"([^"]*)"',
        r'"result"\s*:\s*"([^"]*)"',
    ]
    
    for pattern in json_patterns:
        # Find all matches and take the last one (most likely final answer)
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            grade = matches[-1].group(1).strip().lower()
            if grade in VALID_GRADES:
                return grade
    
    # Priority 2: Look for explicit grade assignments in text (last occurrence)
    text_patterns = [
        r'\bgrade\s*[:=]\s*["\']?([a-zA-Z]+)["\']?',
        r'\bfinal\s+grade\s*[:=]\s*["\']?([a-zA-Z]+)["\']?',
        r'\bassigned\s+grade\s*[:=]\s*["\']?([a-zA-Z]+)["\']?',
        r'\bthe\s+grade\s+is\s*["\']?([a-zA-Z]+)["\']?',
        r'\bi\s+assign\s*["\']?([a-zA-Z]+)["\']?',
        r'\bthis\s+is\s+([a-zA-Z]+)\b',
        r'\bverdict\s*[:=]\s*["\']?([a-zA-Z]+)["\']?',
        r'\bconclusion\s*[:=]\s*["\']?([a-zA-Z]+)["\']?',
        r'\bfinal\s+grade\s+selection\s*[:=]\s*["\']?([a-zA-Z]+)["\']?',
        r'\bgrade\s+should\s+be\s*["\']?([a-zA-Z]+)["\']?',
        r'\bgrade\s*:\s*["\']?([a-zA-Z]+)["\']?',
    ]
    
    for pattern in text_patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            grade = matches[-1].group(1).strip().lower()
            if grade in VALID_GRADES:
                return grade
    
    # Priority 3: Look for quoted grades (strong indicator) - last occurrence
    quoted_patterns = [
        (r'["\']almost["\']', 'almost'),
        (r'["\']partial["\']', 'partial'),
        (r'["\']incorrect["\']', 'incorrect'),
        (r'["\']correct["\']', 'correct'),
    ]
    for pattern, grade in quoted_patterns:
        matches = list(re.finditer(pattern, text_lower))
        if matches:
            return grade
    
    # Priority 4: Look for grade mentioned in final verification checklist
    # This is a strong indicator of the final decision
    checklist_pattern = r'final\s*check.*?(almost|partial|incorrect|correct)'
    checklist_match = re.search(checklist_pattern, text_lower)
    if checklist_match:
        grade = checklist_match.group(1).strip().lower()
        if grade in VALID_GRADES:
            return grade
    
    # Priority 5: Look for "grade: X" or "grade should be X" patterns
    grade_decision_pattern = r'(?:grade|verdict|conclusion)(?:\s+(?:is|should\s+be|selection))?[\s:]+["\']?(almost|partial|incorrect|correct)["\']?'
    decision_matches = list(re.finditer(grade_decision_pattern, text_lower))
    if decision_matches:
        grade = decision_matches[-1].group(1).strip().lower()
        if grade in VALID_GRADES:
            return grade
    
    # Priority 6: Look for standalone grade keywords in the last 300 words
    # This is where the conclusion typically appears
    words = text_lower.split()
    last_part = ' '.join(words[-300:]) if len(words) > 300 else text_lower
    
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
    
    # First try extracting from JSON blocks (most reliable)
    json_results = _extract_jsons(text)
    if json_results:
        # Check from last to first (most likely to contain final answer)
        for json_obj in reversed(json_results):
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
    
    # Then try the improved text extraction (looks for last occurrence)
    result = _extract_grade_from_text(text)
    if result:
        return result
    
    # Look for grade in the last 1000 characters (where conclusion typically is)
    text_lower = text.lower()
    last_section = text_lower[-1000:] if len(text_lower) > 1000 else text_lower
    
    # Check in order of specificity in the last section first
    # "almost" is most specific (contains "partial" as substring)
    for grade in ["almost", "partial", "incorrect", "correct"]:
        if re.search(rf'\b{grade}\b', last_section):
            return grade
    
    # Last resort: look anywhere in text, but prioritize later occurrences
    # by searching from the end
    for grade in ["almost", "partial", "incorrect", "correct"]:
        # Find all occurrences and take the last one
        matches = list(re.finditer(rf'\b{grade}\b', text_lower))
        if matches:
            # Check if this is in the last 25% of the text (likely conclusion)
            last_match = matches[-1]
            position_ratio = last_match.start() / len(text_lower)
            if position_ratio > 0.75:  # In the last quarter of text
                return grade
    
    # Final fallback: any occurrence
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

## Grading Guidelines (FOLLOW THESE EXACTLY - THESE ARE AUTHORITATIVE):
{grading_guidelines}

## Grade Definitions (STRICT INTERPRETATION)

**CORRECT**: The solution is COMPLETE and FULLY CORRECT.
- All required steps are present and logically sound
- The proof/solution is complete with no gaps
- Any calculations are correct (or errors are trivial and don't affect the result)
- The student has successfully solved the problem
- Use this ONLY when the solution is essentially perfect (95-100% complete)
- The main result is FULLY proven with no significant issues
- **WHEN IN DOUBT, DO NOT USE THIS GRADE** - if there are any non-trivial issues, use "almost" instead

**INCORRECT**: The solution shows NO MEANINGFUL PROGRESS or contains FUNDAMENTAL ERRORS.
- The approach is fundamentally wrong or misguided
- No valid mathematical reasoning is presented
- The student completely misunderstood the problem
- Calculations are random or unrelated to the problem
- The student did NOT meet ANY criteria from the "Partial" section in the grading guidelines
- Use this for solutions that are essentially worthless (0-30% complete)
- BE STRICT: If there's no clear valid approach, grade INCORRECT
- KEY INDICATORS: Random formulas, no logical flow, fundamental misconceptions, no attempt at the actual problem, failed to achieve any partial criteria
- **IMPORTANT**: If the student achieved ANY criterion from the "Partial" section, you CANNOT grade "incorrect"

**PARTIAL**: The solution shows SIGNIFICANT PROGRESS but is INCOMPLETE.
- The student has made meaningful progress (30-75% complete)
- Key insights or lemmas may be proven
- BUT the MAIN RESULT is NOT fully proven
- This is the MOST COMMON grade for incomplete work
- CRITICAL: If the main theorem/result is not fully proven → MUST be "partial"
- CRITICAL: The student MUST have achieved at least one criterion from the "Partial" section in the grading guidelines
- Use this when there's good work but the solution doesn't reach the finish line
- DO NOT overuse this grade - ensure there IS meaningful progress first
- KEY INDICATORS: Valid approach started, some lemmas proven, clear understanding shown, but main claim not established, achieved at least one partial criterion from guidelines
- **IMPORTANT**: This is DIFFERENT from "almost" - "partial" means the main result is NOT proven

**ALMOST**: The solution is NEARLY COMPLETE with only MINOR issues.
- The main proof structure IS complete
- The main result IS proven (this is REQUIRED)
- Only small computational errors, minor gaps, or cosmetic issues remain
- The solution is 75-95% complete
- CRITICAL: The main result must be essentially proven - if the main result is NOT proven, you CANNOT use "almost"
- Use this when the solution is very close to correct but has small flaws
- BE STRICT: Minor issues must truly be minor (don't affect main result)
- KEY INDICATORS: Main theorem stated and proven, solution structure complete, only small errors in calculations or minor logical gaps
- **CRITICAL DISTINCTION**: "Almost" = main result proven with minor issues; "Partial" = main result NOT proven but good progress made

## Decision Process (FOLLOW THIS ORDER EXACTLY)

1. **FIRST - Parse the Grading Guidelines CAREFULLY**:
   - Read the grading guidelines line by line
   - Identify the EXACT specific criteria listed under "(Partial)" in the grading guidelines
   - Identify the EXACT specific criteria listed under "(Almost)" in the grading guidelines
   - These criteria are the AUTHORITATIVE source for what constitutes each grade
   - The grading guidelines tell you EXACTLY what the student needed to achieve

2. **SECOND - Check if student met Partial criteria**:
   - Did the student achieve ANY of the criteria listed under "(Partial)" in the guidelines?
   - Look for EXPLICIT evidence in the student's answer that matches the criteria
   - If the student did NOT meet ANY Partial criteria → Grade: **incorrect**
   - BE STRICT: If the guidelines list specific achievements and the student achieved none, the grade is INCORRECT
   - **IMPORTANT**: If the student achieved ANY Partial criterion, you CANNOT grade "incorrect"

3. **THIRD - Determine if main result is proven (CRITICAL STEP)**:
   - This is the KEY distinction between "partial" vs "almost/correct"
   - Does the student's solution COMPLETELY prove the main theorem/result?
   - Does the solution reach the final conclusion/answer?
   - **If main result is NOT fully proven → Grade: "partial"** (STOP HERE - do not proceed)
   - **If main result IS proven → Continue to step 4**
   - BE STRICT: The main result must be FULLY proven, not just "mostly" proven or "essentially" proven

4. **FOURTH - If main result IS proven, check for issues**:
   - Are there ANY errors, gaps, or issues in the proof?
   - Are the errors/gaps MINOR (cosmetic, small calculation errors, easy to fix)?
   - Or are the errors/gaps SIGNIFICANT (affect the main argument, logical flaws)?
   - **If NO significant issues → Grade: "correct"**
   - **If only MINOR issues → Grade: "almost"**
   - **If SIGNIFICANT issues → The main result is NOT fully proven → Grade: "partial"** (go back to step 3)

5. **FIFTH - Verify against grading guidelines**:
   - Re-read the grading guidelines
   - Ensure your grade matches the explicit criteria in the guidelines
   - The guidelines are authoritative - follow them closely
   - If the guidelines say "Partial: 1. Found invariant" and the student didn't find it → NOT partial
   - **Double-check**: Does the "Almost" section in guidelines match your assessment?

## Critical Rules (VIOLATING THESE WILL CAUSE ERRORS)

1. **When in doubt, choose the LOWER grade** - it's better to be conservative
2. **If the main result is not proven, it CANNOT be "correct" or "almost"** - it MUST be "partial"
3. **"Partial" is for incomplete solutions with good progress** - this is very common
4. **"Almost" requires the main result to be essentially proven** - be strict about this
5. **"Incorrect" means NO meaningful progress** - be strict about valid reasoning
6. **Always verify your grade against the grading guidelines** before finalizing
7. **MOST IMPORTANT**: The grading guidelines list specific criteria for "Partial" and "Almost". If the student didn't meet ANY of the "Partial" criteria, the grade MUST be "incorrect" - even if the solution looks like it has some content.
8. **DO NOT BE LENIENT**: IMO grading is strict. A solution that "looks like it has some math" but doesn't meet the specific criteria is INCORRECT, not partial.
9. **CRITICAL - "Almost" vs "Partial" distinction**:
   - "Almost" = Main result IS proven + minor issues exist
   - "Partial" = Main result is NOT proven (regardless of how much progress was made)
   - If the main result is NOT proven, you CANNOT use "almost" or "correct"
10. **CRITICAL - "Partial" vs "Incorrect" distinction**:
    - If student achieved ANY criterion from "(Partial)" section → CANNOT be "incorrect"
    - "Incorrect" is ONLY for solutions with NO meaningful progress
    - If there's any valid mathematical work, it's at least "partial"

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

**Example 11 - INCORRECT when no Partial criteria met:**
Grading Guidelines say:
(Partial)
 1. Found a correct invariant mod 4.
 2. Proved a key lemma about the invariant.

Student's answer: Writes some equations and attempts to analyze the problem, but never actually finds the invariant mod 4 or proves the key lemma. The work looks like math but doesn't achieve the specific criteria listed.
→ CORRECT grade: "incorrect" (student did NOT meet any of the specific Partial criteria listed in the guidelines)

**Example 12 - INCORRECT vs PARTIAL based on guidelines:**
Grading Guidelines say:
(Partial)
 1. Constructed an external point that may lead to a solution.

Student's answer: Discusses the problem and mentions some geometric concepts, but never actually constructs the specific external point mentioned in the guidelines.
→ CORRECT grade: "incorrect" (failed to meet the specific Partial criterion)

**Example 13 - PARTIAL when main result not proven:**
Grading Guidelines say:
(Partial)
 1. Found the key insight about the problem structure.
 2. Set up the equations correctly.

Student's answer: Finds the key insight and sets up the equations correctly, but gets stuck halfway through the proof and never completes it.
→ CORRECT grade: "partial" (met partial criteria, main result not proven)

**Example 14 - ALMOST when main result proven with minor gap:**
Grading Guidelines say:
(Almost)
 1. Complete solution with minor logical gap that can be filled.

Student's answer: Presents a complete proof of the main result, but one step has a small logical gap that is easy to fill (e.g., "this follows from the lemma" without explicitly citing which lemma).
→ CORRECT grade: "almost" (main result proven, minor gap)

**Example 15 - ALMOST vs CORRECT distinction (IMPORTANT):**
Grading Guidelines say:
(Almost)
 1. Solution is almost complete, but made minor mistakes which are not negligible.

Student's answer: Provides a complete proof of the main theorem with all key steps, but contains a small computational error or minor logical oversight that doesn't invalidate the main argument.
→ CORRECT grade: "almost" (NOT "correct" - there are minor mistakes)

**Example 16 - ALMOST when main result proven but minor mistakes:**
Grading Guidelines say:
(Partial)
 1. (WLOG assumed n=k+1 and) proved that two different divisors of P share the position of a zero coefficient.
 2. Found a polynomial with only distinct real roots and has two consecutive zero coefficients.
(Almost)
 1. Solution is almost complete, but made minor mistakes which are not negligible.

Student's answer: Student provides a complete proof structure, proves the main result, but makes minor mistakes in the reasoning (e.g., small gaps in logic, minor computational errors).
→ CORRECT grade: "almost" (main result proven, minor mistakes - matches "Almost" criteria)

**Example 17 - PARTIAL when main result NOT proven despite good progress:**
Grading Guidelines say:
(Partial)
 1. Correctly guessed all the possible P.
 2. Proved that P is bijective.
 3. Proved that P(-P(x))=-x.
(Almost)
 1. Solution is almost complete, but made minor mistakes which are not negligible.

Student's answer: Student correctly guesses possible P, proves P is bijective, proves P(-P(x))=-x, but does NOT complete the full characterization of all solutions or prove the main theorem completely.
→ CORRECT grade: "partial" (achieved Partial criteria, but main result NOT fully proven)

**Example 18 - INCORRECT when no Partial criteria met:**
Grading Guidelines say:
(Partial)
 1. Guessed L=3 and then explained the strategy of the proof.
 2. Partitioned the given grid into squares of size of order sqrt(n) x sqrt(n).

Student's answer: Student makes some observations about the grid and discusses the problem, but never actually guesses L=3 or partitions the grid as required. The work shows some understanding but doesn't achieve the specific criteria listed.
→ CORRECT grade: "incorrect" (did NOT achieve any specific Partial criteria)

**Example 19 - ALMOST when solution is complete but has minor mistakes:**
Grading Guidelines say:
(Partial)
 1. (WLOG assumed n=k+1 and) proved that two different divisors of P share the position of a zero coefficient.
 2. Found a polynomial with only distinct real roots and has two consecutive zero coefficients.
(Almost)
 1. Solution is almost complete, but made minor mistakes which are not negligible.

Student's answer: Student provides a complete proof structure, proves the main result that P(x) has a non-real root, but makes minor mistakes in the reasoning (e.g., small gaps in logic, minor computational errors in the contradiction argument).
→ CORRECT grade: "almost" (main result proven, minor mistakes - matches "Almost" criteria exactly)

**Example 20 - CORRECT vs ALMOST distinction:**
Grading Guidelines say:
(Almost)
 1. Solution is almost complete, but made minor mistakes which are not negligible.

Student's answer: Complete proof with all steps logically sound, no errors, perfect solution.
→ CORRECT grade: "correct" (no mistakes, fully complete)

Student's answer: Complete proof of main result but with minor computational error or small logical gap.
→ CORRECT grade: "almost" (has minor mistakes - matches the "Almost" criterion)

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

## Step-by-Step Analysis (DO THIS EXPLICITLY)

Before giving your grade, write out your analysis:

1. **Parse the Grading Guidelines**: What specific criteria are listed under "(Partial)"? What are listed under "(Almost)"?

2. **Check Partial Criteria**: Go through each criterion under "(Partial)" in the guidelines. Did the student achieve it? Write YES or NO for each.

3. **Determine if ANY Partial criteria achieved**:
   - If NO criteria achieved → Grade: "incorrect" (STOP HERE)
   - If 1+ criteria achieved → Continue to step 4

4. **Verify Main Result (CRITICAL)**: Is the main theorem/result of the problem FULLY proven?
   - Look for: Does the solution reach the final conclusion? Is the main claim established?
   - **If NO → Grade: "partial" (STOP HERE - main result not proven)**
   - **If YES → Continue to step 5**

5. **Check for issues in the proof**:
   - Are there errors, gaps, or mistakes?
   - Are they MINOR (cosmetic, small calculation errors) or SIGNIFICANT (logical flaws)?
   - **If NO significant issues → Grade: "correct"**
   - **If MINOR issues only → Grade: "almost"**
   - **If SIGNIFICANT issues → Main result NOT fully proven → Grade: "partial"**

## Final Verification Checklist

Before outputting your grade, verify:
1. [ ] Did I identify the specific criteria under "(Partial)" in the grading guidelines?
2. [ ] Did the student achieve ANY of the Partial criteria? (If no → MUST be "incorrect")
3. [ ] **CRITICAL**: Did I verify if the main result is FULLY proven? (If no → MUST be "partial")
4. [ ] If main result IS proven, are there only minor issues? (If yes → "almost", if perfect → "correct")
5. [ ] Does my grade match the explicit criteria in the grading guidelines?
6. [ ] Am I being appropriately strict with the definitions?
7. [ ] Have I considered all four grades (correct, incorrect, partial, almost) before deciding?
8. [ ] **FINAL CHECK**: If I graded "almost" or "correct", is the main result truly proven? If not, change to "partial".

Think through your reasoning step by step following the Step-by-Step Analysis above, then provide your final grade.

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
