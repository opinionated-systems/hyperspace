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
                # Try to find the matching closing brace
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
    """Extract grade from various formats in the text with improved accuracy."""
    if not text:
        return None
        
    text_lower = text.lower()
    valid_grades = ['correct', 'incorrect', 'partial', 'almost']
    
    # Priority 1: Look for JSON grade field patterns with strict matching
    # Use word boundaries to ensure exact matches
    json_patterns = [
        r'"grade"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"grade"\s*:\s*\'(correct|incorrect|partial|almost)\'',
        r'"grade"\s*:\s*\b(correct|incorrect|partial|almost)\b',
        r'"response"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"prediction"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"result"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"answer"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"evaluation"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"assessment"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"output"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"label"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"classification"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"category"\s*:\s*"(correct|incorrect|partial|almost)"',
    ]
    
    for pattern in json_patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1)
    
    # Priority 2: Look for explicit grade assignments in text with word boundaries
    text_patterns = [
        r'\bgrade\s*[:=]\s*\b(correct|incorrect|partial|almost)\b',
        r'\bfinal\s+grade\s*[:=]\s*\b(correct|incorrect|partial|almost)\b',
        r'\bassigned\s+grade\s*[:=]\s*\b(correct|incorrect|partial|almost)\b',
        r'\bthe\s+grade\s+is\s+\b(correct|incorrect|partial|almost)\b',
        r'\bi\s+assign\s+\b(correct|incorrect|partial|almost)\b',
        r'\bthis\s+is\s+\b(correct|incorrect|partial|almost)\b',
        r'\bgrade\s+is\s*[:=]?\s*\b(correct|incorrect|partial|almost)\b',
        r'\bassigned\s*[:=]?\s*\b(correct|incorrect|partial|almost)\b',
        r'\btherefore[,]?\s+the\s+grade\s+is\s+\b(correct|incorrect|partial|almost)\b',
        r'\bthus[,]?\s+the\s+grade\s+is\s+\b(correct|incorrect|partial|almost)\b',
        r'\bconclusion[:]\s+\b(correct|incorrect|partial|almost)\b',
        r'\bdecision[:]\s+\b(correct|incorrect|partial|almost)\b',
        r'\bverdict[:]\s+\b(correct|incorrect|partial|almost)\b',
        r'\bfinal\s+answer[:]\s+\b(correct|incorrect|partial|almost)\b',
        r'\bthe\s+final\s+grade\s+is\s+\b(correct|incorrect|partial|almost)\b',
        r'\bmy\s+(?:grade|assessment|evaluation)\s+is\s+\b(correct|incorrect|partial|almost)\b',
        r'\bthe\s+(?:correct|appropriate|proper)\s+grade\s+is\s+\b(correct|incorrect|partial|almost)\b',
        r'\bgrade\s+(?:should|must)\s+be\s+\b(correct|incorrect|partial|almost)\b',
        r'\bthis\s+(?:solution|answer|work)\s+(?:is|should\s+be)\s+\b(correct|incorrect|partial|almost)\b',
    ]
    
    for pattern in text_patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1)
    
    # Priority 3: Look for grade keywords in quotes (strong indicator)
    quoted_patterns = [
        (r'["\']almost["\']', 'almost'),
        (r'["\']partial["\']', 'partial'),
        (r'["\']incorrect["\']', 'incorrect'),
        (r'["\']correct["\']', 'correct'),
    ]
    for pattern, grade in quoted_patterns:
        if re.search(pattern, text_lower):
            return grade
    
    # Priority 4: Look for grade in the last 300 words (conclusion area)
    # The conclusion typically appears at the end of the reasoning
    words = text_lower.split()
    last_part = ' '.join(words[-300:]) if len(words) > 300 else text_lower
    
    # Check in order of specificity (more specific first to avoid misclassification)
    # "almost" and "partial" are more specific than "incorrect" and "correct"
    # Use word boundaries to avoid partial matches
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
    if grade_lower in ["correct", "incorrect", "partial", "almost"]:
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
    
    valid_grades = ["correct", "incorrect", "partial", "almost"]
    cleaned_text = text.strip()
    text_lower = cleaned_text.lower()
    
    # Strategy 1: Try to extract JSON objects (most reliable)
    extracted = _extract_jsons(cleaned_text)
    if extracted:
        # Check from last to first (most recent JSON is likely the answer)
        for json_obj in reversed(extracted):
            if isinstance(json_obj, dict):
                # Try common grade field names (in order of likelihood)
                for field in ["grade", "response", "result", "prediction", "answer", 
                              "evaluation", "assessment", "output", "label", 
                              "classification", "category", "verdict", "decision"]:
                    if field in json_obj:
                        val = str(json_obj[field]).lower().strip().strip('"').strip("'")
                        if val in valid_grades:
                            return val
                # If no standard field found, try any value
                for val in json_obj.values():
                    val_str = str(val).lower().strip().strip('"').strip("'")
                    if val_str in valid_grades:
                        return val_str
    
    # Strategy 2: Try text-based extraction with strict patterns
    extracted_text = _extract_grade_from_text(cleaned_text)
    if extracted_text:
        return extracted_text
    
    # Strategy 3: Look for quoted grades in the text (check in priority order)
    # Priority: almost > partial > incorrect > correct (more specific first)
    quoted_patterns = [
        (r'["\']almost["\']', 'almost'),
        (r'["\']partial["\']', 'partial'),
        (r'["\']incorrect["\']', 'incorrect'),
        (r'["\']correct["\']', 'correct'),
    ]
    for pattern, grade in quoted_patterns:
        if re.search(pattern, text_lower):
            return grade
    
    # Strategy 4: Look for grade keywords in the last 300 words
    # The conclusion typically appears at the end
    words = text_lower.split()
    last_300 = ' '.join(words[-300:]) if len(words) > 300 else text_lower
    
    # Check in order of specificity (more specific grades first to avoid misclassification)
    # Use word boundaries for more accurate matching
    if re.search(r'\balmost\b', last_300):
        return 'almost'
    elif re.search(r'\bpartial\b', last_300):
        return 'partial'
    elif re.search(r'\bincorrect\b', last_300):
        return 'incorrect'
    elif re.search(r'\bcorrect\b', last_300):
        return 'correct'
    
    # Strategy 5: Look anywhere in text as last resort (specific first)
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

## Grading Guidelines (CRITICAL - FOLLOW EXACTLY):
{grading_guidelines}

## Grade Definitions - READ CAREFULLY AND APPLY STRICTLY

**CORRECT**: The solution is COMPLETE and FULLY CORRECT.
- All required steps are present and logically sound
- The proof/solution is complete with no gaps
- The student has successfully solved the problem
- Use this ONLY when the solution is essentially perfect (95-100% complete)
- NO significant errors allowed - only trivial typos that don't affect mathematical validity

**INCORRECT**: The solution shows NO MEANINGFUL PROGRESS or contains FUNDAMENTAL ERRORS.
- The approach is fundamentally wrong or misguided
- No valid mathematical progress toward the solution
- Random calculations without valid approach = INCORRECT
- Use this when the student failed to make any real progress
- The student misunderstood the problem or used completely wrong methods

**PARTIAL**: The student made SIGNIFICANT PROGRESS but the solution is INCOMPLETE.
- Found a useful invariant, lemma, or key insight
- Established a valid approach but didn't complete the proof
- Has the right idea but execution is incomplete
- The solution stops well before completion
- KEY DISTINCTION: The student found important pieces but is FAR from a complete solution (typically 30-70% complete)
- Use this when the student found key insights but the solution is clearly incomplete
- The main result/proof is NOT established

**ALMOST**: The solution is NEARLY COMPLETE with only MINOR ISSUES.
- The main proof structure is correct and complete
- Only minor computational errors, small omissions, or slight gaps
- The student essentially solved the problem but with small imperfections
- KEY DISTINCTION: The solution is VERY CLOSE to complete (80-95% complete) - just needs small fixes
- Use this when the student has essentially solved the problem
- The main result IS established, just with minor issues

## CRITICAL DISTINCTION: PARTIAL vs ALMOST (MOST IMPORTANT - READ TWICE!)

This is the most common grading error. Be extremely careful here.

**THE KEY QUESTION: Is the main proof structure complete?**

**Choose PARTIAL when:**
- The student found key insights, lemmas, or invariants
- The solution stops early or is missing major proof sections
- The student is on the right track but far from done
- The solution has significant gaps or missing major steps
- Example: Found the key lemma but didn't prove the main result
- Example: Has the right approach but only completed 30-70% of the proof
- Example: Made good progress but stopped before the final conclusion
- Example: Has valid intermediate results but the main theorem is unproven
- **RULE OF THUMB**: If you can't say "the student essentially solved it", use PARTIAL

**Choose ALMOST when:**
- The student has a nearly complete proof structure (80-95% complete)
- Only minor details, small errors, or slight gaps remain
- The core solution is essentially correct
- The proof is complete except for small computational errors
- **RULE OF THUMB**: The solution would be CORRECT if the minor issues were fixed
- **CRITICAL**: The main proof must be structurally complete - not missing major sections
- Example: Complete proof with one small calculation error
- Example: Full solution with a minor gap that doesn't affect the main argument
- Example: All main ideas correct, just missing a small verification

**DECISION RULE (USE THIS - BE CONSERVATIVE):**

**STEP 1: Check for INCORRECT first**
- Does the solution show NO meaningful progress?
- Is the approach fundamentally wrong?
- Are there random calculations without valid approach?
→ If YES to any: Grade "incorrect"

**STEP 2: Check for CORRECT**
- Is the solution 95-100% complete?
- Are ALL steps logically sound with NO significant errors?
- Is the proof essentially perfect?
→ If YES to all: Grade "correct"

**STEP 3: Distinguish PARTIAL vs ALMOST (MOST CRITICAL)**
Ask yourself: "Did the student essentially solve the problem?"

- If the main result is NOT established → "partial"
- If major proof sections are missing → "partial"
- If the solution stops before the conclusion → "partial"
- If only intermediate results exist → "partial"

- If the main result IS established with minor issues → "almost"
- If the proof is structurally complete but has small errors → "almost"
- If fixing small issues would make it correct → "almost"

**COMPLETION PERCENTAGE GUIDE:**
- Less than 75% complete → "partial" (be conservative)
- 75-90% complete with minor issues → "almost"
- 90-100% complete with only tiny gaps → "almost" or "correct"
- 95-100% complete perfect → "correct"

**WHEN IN DOUBT:**
- Between "partial" and "almost" → Choose "partial" (safer)
- Between "almost" and "correct" → Choose "almost" (unless perfect)

## Grading Process (FOLLOW THESE STEPS EXACTLY)

### Step 1: Parse the Grading Guidelines Structure
The guidelines use markers like "(Partial)", "(Almost)", "(Correct)" to list specific criteria.
- Look for these exact markers in the guidelines
- Each marker indicates what achievements correspond to that grade
- Count how many criteria the student satisfies under each category
- Note: The guidelines are the PRIMARY source for grading decisions

### Step 2: Analyze What the Student Actually Did
Read the student's answer carefully and identify:
- What specific results did they prove?
- What lemmas or key insights did they discover?
- Where exactly does their solution stop or have gaps?
- Did they complete the full proof or only part of it?
- What percentage of the solution is complete? (estimate: 30%, 50%, 70%, 90%, 100%)
- Does the solution have the main proof structure in place?
- **CRITICAL**: Is the main theorem/result actually proven?

### Step 3: Match Student Achievements to Guidelines
Compare the student's work against the criteria in the guidelines:

**If the student met criteria under (Partial) but NOT (Almost) or (Correct):**
→ Grade: "partial"

**If the student met criteria under (Almost) but NOT (Correct):**
→ Grade: "almost"

**If the student met criteria under (Correct):**
→ Grade: "correct"

**If the student made NO meaningful progress toward the solution:**
→ Grade: "incorrect"

### Step 4: Apply Conservative Completion Check
After matching to guidelines, verify with completion percentage:
- Less than 75% complete → "partial" (unless guidelines say otherwise)
- 75-90% complete with minor issues → "almost"
- 90-100% complete with only tiny gaps → "almost" or "correct"
- 95-100% complete perfect → "correct"
- No meaningful progress → "incorrect"

### Step 5: Final Verification (CRITICAL - DO NOT SKIP)
Before deciding, explicitly answer these questions in your reasoning:
1. "What percentage of the proof is complete?" (give a specific number: 30%, 60%, 90%, 100%)
2. "Is the main theorem/result actually proven?" → If NO, cannot be "correct" or "almost"
3. "Did the student complete the full proof structure?" → If NO, cannot be "almost" or "correct"
4. "Did the student find key insights but stop early?" → If yes, "partial"
5. "Did the student make no real progress?" → If yes, "incorrect"
6. "Is the solution nearly complete with only minor issues?" → If yes AND main result proven, "almost"
7. "Are there major gaps or missing sections?" → If yes, "partial" or "incorrect"
8. "Does the solution have the main proof structure in place?" → If NO, cannot be "almost"

**FINAL CHECK - BE STRICT AND CONSERVATIVE:**
- If you chose "almost": Verify the solution is at least 75% complete AND main result is proven AND structure is complete
- If you chose "partial": Verify the solution is less than 75% complete OR missing major sections OR main result unproven
- If you chose "correct": Verify the solution is 95-100% complete with no significant errors
- If you chose "incorrect": Verify the student made no meaningful progress

**COMMON MISTAKES TO AVOID:**
- Do NOT grade "almost" if the main result is not established
- Do NOT grade "almost" if the solution is missing major proof sections
- Do NOT grade "partial" if the solution is essentially complete with just minor errors
- "Almost" requires: (1) main result proven, (2) 75%+ complete, (3) structure in place
- "Partial" is for solutions that are incomplete (30-75% complete) or missing main result
- When in doubt between "partial" and "almost", choose "partial"

## Examples of Correct Grading

**Example 1 - PARTIAL grading:**
Grading Guidelines say:
(Partial)
 1. Found a useful invariant.
 2. Proved a key lemma.
(Almost)
 1. Nearly complete solution with minor gaps.

Student's answer: Proves the invariant and the key lemma but doesn't complete the main proof. The main theorem remains unproven. The solution is about 50% complete.
→ CORRECT grade: "partial" (student met the Partial criteria but not Almost; main result NOT established)

**Example 2 - INCORRECT grading:**
Grading Guidelines say:
(Partial)
 1. Found a useful invariant.

Student's answer: Makes some calculations but no valid approach, no invariant found, fundamental misunderstanding of the problem.
→ CORRECT grade: "incorrect" (student did not meet any meaningful criteria)

**Example 3 - ALMOST grading:**
Grading Guidelines say:
(Almost)
 1. Complete proof with minor computational error.

Student's answer: Complete proof with a small arithmetic mistake that doesn't affect the main argument. The main result IS established. About 95% complete.
→ CORRECT grade: "almost" (student met Almost criteria; main result proven with minor issue)

**Example 4 - CORRECT grading:**
Grading Guidelines say:
(Correct)
 1. Complete and correct proof.

Student's answer: Full correct proof with all steps valid. 100% complete. No errors.
→ CORRECT grade: "correct" (student met Correct criteria)

**Example 5 - PARTIAL vs ALMOST distinction (CRITICAL - READ CAREFULLY):**

Case A: Student found the key lemma and set up the right approach but stopped before completing the main proof. The main theorem is NOT proven. The solution is about 50% complete and missing major sections.
→ CORRECT grade: "partial" (significant progress but main result NOT established; less than 75% complete)

Case B: Student wrote a complete proof with one small algebraic error in the final step. The main result IS established. The solution is about 95% complete with the main structure in place.
→ CORRECT grade: "almost" (nearly complete, minor issue; main result proven)

**Example 6 - When to use ALMOST vs PARTIAL:**

ALMOST examples (main result MUST be proven):
- Complete proof with a small calculation error (90% complete, main result proven)
- Full solution missing only a trivial verification step (85% complete, main result proven)
- All main ideas correct, just one small gap that doesn't affect the main argument (80% complete, main result proven)

PARTIAL examples (main result NOT fully proven):
- Found key lemma but didn't complete the main proof (60% complete, main result unproven)
- Has the right approach but stopped halfway (50% complete, main result unproven)
- Made good progress but missing the final conclusion (70% complete, main result unproven)

**Example 7 - Edge case: Multiple small errors:**
Student's answer: Complete proof structure with several small calculation errors throughout, but the main argument is sound and the main result IS established.
→ CORRECT grade: "almost" (main structure is complete, main result proven, errors are minor)

**Example 8 - Edge case: Right idea, wrong execution:**
Student's answer: Found the correct approach and key lemma, but made a fundamental error in applying it that invalidates the proof. The main result is NOT established.
→ CORRECT grade: "partial" (good insight but execution failed; major gap in proof; main result unproven)

**Example 9 - CRITICAL: Main result not proven:**
Student's answer: Excellent work on finding the key lemma, setting up the right framework, and making significant progress. However, the main theorem is never actually proven - the solution stops before establishing the final result.
→ CORRECT grade: "partial" (cannot be "almost" because main result is NOT proven, regardless of how good the intermediate work is)

## Response Format

Respond ONLY with a JSON object in the following format. The grade field MUST be exactly one of: "correct", "incorrect", "partial", or "almost" (all lowercase, no extra spaces).

<json>
{{
    "grade": "correct" | "incorrect" | "partial" | "almost"
}}
</json>

Grade:"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction using the comprehensive extraction function
        prediction = "none"
        
        try:
            if msg_history and len(msg_history) > 0:
                last_message = msg_history[-1].get("text", "")
                if last_message:
                    prediction = _extract_grade(last_message)
                    
                    # Log for debugging if extraction fails
                    if prediction == "none":
                        self.log_fn(f"Warning: Could not extract valid grade from response. Response preview: {last_message[:500]}...")
                        
                        # Try one more time with a simplified extraction - look for any valid grade word
                        # Priority: almost > partial > incorrect > correct (more specific first)
                        text_lower = last_message.lower()
                        for grade in ["almost", "partial", "incorrect", "correct"]:
                            if re.search(rf'\b{grade}\b', text_lower):
                                prediction = grade
                                self.log_fn(f"Fallback extraction found grade: {grade}")
                                break
                else:
                    self.log_fn("Warning: Last message has no text content")
            else:
                self.log_fn("Warning: No message history available")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            import traceback
            self.log_fn(f"Traceback: {traceback.format_exc()}")

        # Normalize prediction to expected values
        prediction = _normalize_grade(prediction)
        
        # Final validation - ensure we have a valid grade
        if prediction not in VALID_GRADES:
            self.log_fn(f"Warning: Invalid prediction '{prediction}', defaulting to 'incorrect'")
            prediction = "incorrect"

        return str(prediction), msg_history
