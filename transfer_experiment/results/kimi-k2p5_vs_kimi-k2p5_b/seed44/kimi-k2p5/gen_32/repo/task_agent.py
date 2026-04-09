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
    
    # Try to find <json>...</json> blocks
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
    
    # Try to find ```json code blocks
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
    
    # Try to find raw JSON objects with balanced braces
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


def _extract_grade(text: str) -> str:
    """Extract grade from LLM response text using multiple strategies."""
    if not text:
        return "none"
    
    valid_grades = ["correct", "incorrect", "partial", "almost"]
    text_clean = text.strip()
    text_lower = text_clean.lower()
    
    # Strategy 1: Extract from <json>...</json> blocks (highest priority)
    json_block_pattern = r'<json>\s*(\{.*?\})\s*</json>'
    matches = re.findall(json_block_pattern, text_clean, re.DOTALL)
    for match in matches:
        try:
            json_obj = json.loads(match)
            if isinstance(json_obj, dict) and "grade" in json_obj:
                val = str(json_obj["grade"]).lower().strip().strip('"').strip("'")
                if val in valid_grades:
                    return val
        except json.JSONDecodeError:
            continue
    
    # Strategy 2: Extract from ```json code blocks
    code_block_pattern = r'```json\s*(\{.*?\})\s*```'
    matches = re.findall(code_block_pattern, text_clean, re.DOTALL)
    for match in matches:
        try:
            json_obj = json.loads(match)
            if isinstance(json_obj, dict) and "grade" in json_obj:
                val = str(json_obj["grade"]).lower().strip().strip('"').strip("'")
                if val in valid_grades:
                    return val
        except json.JSONDecodeError:
            continue
    
    # Strategy 3: Look for raw JSON with "grade" field
    grade_field_pattern = r'["\']?grade["\']?\s*:\s*["\']?(correct|incorrect|partial|almost)["\']?'
    match = re.search(grade_field_pattern, text_lower)
    if match:
        return match.group(1)
    
    # Strategy 4: Look for explicit grade declarations with strong patterns
    strong_patterns = [
        r'["\']grade["\']\s*:\s*["\'](correct|incorrect|partial|almost)["\']',
        r'"grade":\s*"(correct|incorrect|partial|almost)"',
        r"'grade':\s*'(correct|incorrect|partial|almost)'",
        r'grade:\s*(correct|incorrect|partial|almost)(?:\s|$|,)',
        r'\bfinal\s+grade\s*[:=]\s*["\']?(correct|incorrect|partial|almost)["\']?\b',
        r'\bthe\s+grade\s+is\s+["\']?(correct|incorrect|partial|almost)["\']?\b',
        r'\bi\s+assign\s+["\']?(correct|incorrect|partial|almost)["\']?\b',
        r'\bconclusion[:]\s*["\']?(correct|incorrect|partial|almost)["\']?\b',
        r'\bverdict[:]\s*["\']?(correct|incorrect|partial|almost)["\']?\b',
        r'\bthis\s+solution\s+is\s+["\']?(correct|incorrect|partial|almost)["\']?\b',
    ]
    for pattern in strong_patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1)
    
    # Strategy 5: Look for grade in the last 100 words (conclusion area)
    words = text_lower.split()
    last_part = ' '.join(words[-100:]) if len(words) > 100 else text_lower
    
    # Check for grades with word boundaries in conclusion
    for grade in ["almost", "partial", "incorrect", "correct"]:
        if re.search(rf'\b{grade}\b', last_part):
            return grade
    
    # Strategy 6: Look anywhere in text as last resort
    for grade in ["almost", "partial", "incorrect", "correct"]:
        if re.search(rf'\b{grade}\b', text_lower):
            return grade
    
    return "none"


def _normalize_grade(grade: str) -> str:
    """Normalize grade to one of the expected lowercase values."""
    if not grade:
        return "none"
    
    grade_lower = grade.lower().strip().strip('"').strip("'")
    
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
- Use this for solutions that are essentially worthless

**PARTIAL**: The solution shows SIGNIFICANT PROGRESS but is INCOMPLETE.
- The student has made meaningful progress (30-75% complete)
- Key insights or lemmas may be proven
- BUT the MAIN RESULT is NOT fully proven
- This is the MOST COMMON grade for incomplete work
- CRITICAL: If the main theorem/result is not fully proven → MUST be "partial"
- Use this when there's good work but the solution doesn't reach the finish line

**ALMOST**: The solution is NEARLY COMPLETE with only MINOR issues.
- The main proof structure IS complete
- The main result IS proven
- Only small computational errors, minor gaps, or cosmetic issues remain
- The solution is 75-95% complete
- CRITICAL: The main result must be essentially proven
- Use this when the solution is very close to correct but has small flaws

## Decision Process (FOLLOW THIS ORDER)

1. **FIRST - Check for meaningful progress**:
   - Does the student show any valid mathematical reasoning?
   - If NO meaningful progress → Grade: **incorrect**

2. **SECOND - Check if main result is FULLY proven**:
   - Is the main theorem/result completely proven?
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
5. **Always verify your grade against the grading guidelines** before finalizing

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

Your grade:"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction
        prediction = "none"
        
        try:
            # First try to get the response from the msg_history
            if msg_history and len(msg_history) > 0:
                last_message = msg_history[-1].get("text", "")
                if last_message:
                    prediction = _extract_grade(last_message)
                    
                    if prediction == "none":
                        self.log_fn(f"Warning: Could not extract valid grade from msg_history. Response preview: {last_message[:500]}...")
                else:
                    self.log_fn("Warning: Last message has no text content")
            else:
                self.log_fn("Warning: No message history available")
            
            # If still none, try the direct response
            if prediction == "none" and response:
                prediction = _extract_grade(response)
                if prediction != "none":
                    self.log_fn(f"Extracted grade from direct response: {prediction}")
            
            # Final fallback: look for any valid grade word in the entire response
            if prediction == "none":
                search_text = ""
                if msg_history and len(msg_history) > 0:
                    search_text = msg_history[-1].get("text", "")
                if not search_text and response:
                    search_text = response
                
                if search_text:
                    text_lower = search_text.lower()
                    # Look for grades in order of specificity
                    for grade in ["almost", "partial", "incorrect", "correct"]:
                        if re.search(rf'\b{grade}\b', text_lower):
                            prediction = grade
                            self.log_fn(f"Fallback extraction found grade: {grade}")
                            break
                            
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
