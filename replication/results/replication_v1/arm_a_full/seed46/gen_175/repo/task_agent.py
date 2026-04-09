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


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles markdown code blocks (```json...```) as fallback.
    Includes robust nested brace counting for complex JSON structures.
    """
    results = []
    search_from = 0
    
    # First try <json>...</json> tags
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
            # Try to extract JSON from within the content if it's wrapped in other text
            try:
                # Find first { and last } with proper brace counting
                json_str = _extract_json_with_brace_counting(inner)
                if json_str:
                    results.append(json.loads(json_str))
            except (json.JSONDecodeError, ValueError):
                continue
    
    # Fallback: try markdown code blocks
    if not results:
        search_from = 0
        while True:
            start = text.find("```json", search_from)
            if start == -1:
                # Also try without 'json' specifier
                start = text.find("```", search_from)
                if start == -1:
                    break
                end = text.find("```", start + 3)
            else:
                end = text.find("```", start + 7)
            if end == -1:
                break
            
            if text[start:start+7] == "```json":
                inner = text[start + 7:end].strip()
            else:
                inner = text[start + 3:end].strip()
            search_from = end + 3
            try:
                results.append(json.loads(inner))
            except json.JSONDecodeError:
                # Try with brace counting for nested structures
                try:
                    json_str = _extract_json_with_brace_counting(inner)
                    if json_str:
                        results.append(json.loads(json_str))
                except (json.JSONDecodeError, ValueError):
                    continue
    
    # Final fallback: try to find JSON objects directly in text
    if not results:
        try:
            json_str = _extract_json_with_brace_counting(text)
            if json_str:
                results.append(json.loads(json_str))
        except (json.JSONDecodeError, ValueError):
            pass
    
    return results or None


def _extract_json_with_brace_counting(text: str) -> str | None:
    """Extract a JSON object from text using proper brace counting.
    
    This handles nested braces correctly, ensuring we capture complete
    JSON objects even when they contain nested structures.
    
    Args:
        text: The text to search for JSON objects
        
    Returns:
        The extracted JSON string, or None if no valid JSON found
    """
    # Find the first opening brace
    start = text.find("{")
    if start == -1:
        return None
    
    # Count braces to find the matching closing brace
    brace_count = 0
    in_string = False
    escape_next = False
    
    for i, char in enumerate(text[start:]):
        if escape_next:
            escape_next = False
            continue
            
        if char == "\\" and in_string:
            escape_next = True
            continue
            
        if char == '"' and not in_string:
            in_string = True
        elif char == '"' and in_string:
            in_string = False
        elif not in_string:
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    # Found the matching closing brace
                    return text[start:start + i + 1]
    
    # If we get here, braces weren't balanced
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with structured reasoning.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract fields for structured prompting
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematical grader for International Mathematical Olympiad (IMO) problems.

Your task is to evaluate a student's answer by comparing it against the official solution and grading guidelines.

## Problem Domain
{domain}

## Problem Statement
{problem}

## Official Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Instructions
Follow this structured evaluation process:

1. **Understanding Check**: Briefly summarize what the problem is asking and what the correct approach should be.

2. **Step-by-Step Analysis**: Go through the student's answer carefully:
   - Identify each key step or claim they make
   - Check if each step is mathematically valid
   - Note any errors, gaps, or incorrect assumptions
   - Compare their approach to the official solution
   - Check if they used the correct mathematical notation and terminology

3. **Partial Credit Assessment**: Based on the grading guidelines, determine:
   - Which parts of the solution they completed correctly
   - What partial credit they deserve for incomplete or partially correct work
   - Whether they demonstrated understanding of key concepts
   - Whether they made significant progress toward the solution

4. **Final Grade Decision**: Assign a grade that reflects:
   - Full correctness (Correct/7) if completely solved with valid reasoning
   - Partial credit (Partial/1-6) for incomplete or partially correct solutions
   - Incorrect (Incorrect/0) for fundamentally wrong or empty answers

## Grade Category Definitions (CRITICAL - READ CAREFULLY)

### Decision Flowchart (Use This!)
1. Is the solution completely correct with valid reasoning? → **Correct**
2. Is the approach fundamentally wrong or is there no meaningful work? → **Incorrect**
3. Did the student demonstrate understanding of the correct approach but fail to complete it? → **Partial**
4. Is the solution nearly correct with only minor flaws? → **Almost**

### Detailed Category Definitions with Examples

**Correct (7 points)**: The solution is completely correct with valid reasoning and clear presentation.
- All steps are mathematically sound
- The final answer matches the official solution
- Reasoning is clear and logically valid
- Example: Complete proof with all steps justified, correct final answer

**Almost (5-6 points)**: The solution is NEARLY correct with MINOR flaws that DON'T invalidate the core approach.
- Small computational errors in an otherwise correct approach (e.g., arithmetic mistake at the end)
- Missing trivial justifications that don't affect the core logic (e.g., "clearly" or "obviously" steps)
- Slightly incomplete final answers where the method is clearly correct
- Minor notational issues that don't obscure the mathematical reasoning
- The student clearly understood the problem and had the right approach
- KEY DISTINCTION: If you removed the minor flaw, the solution would be Correct
- Example: Correct proof with a small calculation error in the final step; correct method but missing one trivial justification

**Partial (1-4 points)**: The solution shows SIGNIFICANT PROGRESS but has SUBSTANTIAL gaps or errors.
- Correct initial approach but incomplete execution (stopped halfway)
- Some correct steps mixed with significant errors
- Demonstrated understanding of key concepts but failed to complete the solution
- Made meaningful progress but the solution is NOT close to complete
- KEY DISTINCTION: The approach was right, but there's significant work missing or major errors that prevent it from being "Almost"
- Example: Started with the right lemma but couldn't complete the proof; correct first half but wrong second half; significant gaps in reasoning

**Incorrect (0 points)**: The solution shows NO MEANINGFUL PROGRESS or is FUNDAMENTALLY WRONG.
- Completely wrong approach or misunderstanding of the problem
- No relevant work or empty answer
- Major conceptual errors throughout
- Failed to demonstrate any understanding of the problem
- Example: Wrong theorem applied; completely irrelevant work; blank answer

## Critical Distinctions (READ CAREFULLY)

### Almost vs Partial - The Key Difference:
- **Almost**: The solution is essentially complete and correct. The flaws are MINOR and don't affect the core validity. If you fixed the small issues, you'd have a Correct solution.
- **Partial**: The solution is INCOMPLETE or has SIGNIFICANT issues. There's real progress, but major work is missing or there are substantial errors.

**Ask yourself**: "If I fixed the errors, would this be a complete correct solution?"
- If YES and errors are minor → **Almost**
- If NO, there's still significant work missing → **Partial**

### Examples by Category:

**Correct**: "Let f(x) = x² + 3x + 2. The discriminant is 9 - 8 = 1, so roots are (-3 ± 1)/2 = -1, -2. Therefore f(x) = (x+1)(x+2)." (Complete, correct, well-reasoned)

**Almost**: "Let f(x) = x² + 3x + 2. The discriminant is 9 - 8 = 1, so roots are (-3 ± 1)/2 = -2, -1.5. Therefore f(x) = (x+2)(x+1.5)." (Correct method, small arithmetic error in final answer)

**Partial**: "Let f(x) = x² + 3x + 2. We can try to find roots using the quadratic formula. The discriminant is b² - 4ac..." (Started correctly but didn't finish, or made significant errors in execution)

**Incorrect**: "Let f(x) = x² + 3x + 2. Using the Pythagorean theorem, we find a² + b² = c²..." (Wrong approach entirely)

## Common Misclassification Traps (AVOID THESE!)

### Trap 1: Labeling "Partial" when it should be "Incorrect"
- **The trap**: Seeing ANY correct statement and calling it Partial
- **The reality**: If the work doesn't demonstrate understanding of the PROBLEM'S CORE CONCEPT, it's Incorrect
- **Example**: Student writes "This is a quadratic equation" but then applies the wrong formula entirely → INCORRECT (0 points), not Partial
- **Rule**: For Partial, the student must demonstrate understanding of the CORRECT APPROACH, not just write relevant-sounding words

### Trap 2: Labeling "Partial" when it should be "Almost"
- **The trap**: Seeing incomplete work and defaulting to Partial
- **The reality**: If the solution is 90% complete with just a small error, it's Almost, not Partial
- **Example**: Student solves 9 out of 10 steps correctly, makes a sign error in step 10 → ALMOST (5-6 points), not Partial
- **Rule**: Almost = "essentially correct", Partial = "significantly incomplete"

### Trap 3: Being too generous with "Partial" credit
- **The trap**: Giving Partial (1-4 points) for work that shows no real understanding
- **The reality**: Partial requires DEMONSTRATED understanding of key concepts with meaningful progress
- **Example**: Student writes random formulas that happen to include some correct symbols → INCORRECT, not Partial
- **Rule**: Partial requires EVIDENCE of understanding, not just presence of some correct elements

### Trap 4: Missing "Incorrect" classifications
- **The trap**: Reluctance to give 0 points, defaulting to Partial (1-2 points)
- **The reality**: If the approach is fundamentally wrong, the answer deserves 0 points
- **Example**: Student uses integration by parts on a problem requiring induction → INCORRECT
- **Rule**: Wrong approach = Incorrect, regardless of how much work is shown

## Decision Checklist (Use for Every Grading Decision)

Before finalizing your grade, verify:

1. **For "Correct"**: Is EVERY step valid? Is the reasoning complete? → If yes, grade Correct
2. **For "Incorrect"**: Is the approach fundamentally wrong? Is there NO meaningful progress on the correct solution path? → If yes, grade Incorrect (don't be afraid of 0 points!)
3. **For "Almost"**: Would fixing minor flaws yield a complete solution? Is the core approach correct? → If yes, grade Almost
4. **For "Partial"**: Did the student demonstrate understanding of the correct approach? Is there significant progress despite gaps? → If yes, grade Partial

## Red Flags for "Incorrect" (Don't Miss These!)
- Applying completely wrong theorems or formulas
- No logical connection between steps
- Answer is blank or says "I don't know"
- Work is irrelevant to the problem asked
- Fundamental misunderstanding of the problem type
- Random calculations with no coherent strategy

## Red Flags for "Almost" vs "Partial" Confusion
- If you find yourself thinking "they were on the right track" → check if it's Almost
- If the error is "just a small mistake" → it's Almost, not Partial
- If the solution is "mostly there" → it's Almost
- If significant portions are missing or wrong → it's Partial

## Grading Rubric Reference
- 7 points: Complete, correct solution with clear reasoning → **Correct**
- 5-6 points: Minor gaps or unclear reasoning, but essentially correct approach → **Almost**
- 3-4 points: Significant progress, correct approach but incomplete or with substantial errors → **Partial**
- 1-2 points: Some relevant work or correct initial steps but major gaps → **Partial**
- 0 points: No meaningful progress or completely wrong approach → **Incorrect**

## Response Format
You MUST respond in valid JSON format with the following schema:
<json>
{{
    "understanding": "Brief summary of the problem and correct approach",
    "analysis": "Detailed step-by-step analysis of the student's answer, including what they got right and wrong",
    "partial_credit_reasoning": "Explanation of partial credit based on grading guidelines and rubric. Explicitly state which grade category applies and why.",
    "grade_category_check": "Self-verification: State which category you're choosing and explicitly justify why it fits that category and not the others. Reference the decision flowchart above.",
    "response": "Your final grade: must be exactly one of: 'Correct', 'Almost', 'Partial', or 'Incorrect'"
}}
</json>

Important: 
- Ensure your JSON is valid and properly formatted
- The "response" field MUST contain exactly one of: 'Correct', 'Almost', 'Partial', or 'Incorrect' (case-sensitive)
- Use the "grade_category_check" field to verify your decision - this helps prevent misclassification
- Be careful to distinguish 'Almost' (minor flaws) from 'Partial' (significant gaps)
- When in doubt between 'Almost' and 'Partial', ask: "If I fixed the errors, would this be complete?" If yes → Almost, if no → Partial
- DO NOT be afraid to assign 'Incorrect' when the answer is fundamentally wrong - 0 points is the correct grade for wrong approaches
- The "response" field must contain ONLY the grade word, no extra text, no punctuation, no explanation"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction using the dedicated method
        prediction = self._extract_prediction(msg_history)
        
        return str(prediction), msg_history

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history.
        
        Tries multiple extraction strategies and field names for robustness.
        Includes grade normalization for consistent output format.
        Logs detailed analysis for debugging purposes.
        """
        if not msg_history:
            self.log_fn("Warning: Empty message history")
            return "None"
        
        # Get the last assistant message
        last_assistant_msg = None
        for msg in reversed(msg_history):
            if msg.get("role") == "assistant" or "text" in msg:
                last_assistant_msg = msg.get("text", msg.get("content", ""))
                break
        
        if not last_assistant_msg:
            self.log_fn("Warning: No assistant message found in history")
            return "None"
        
        # Try to extract JSON blocks
        extracted = _extract_jsons(last_assistant_msg)
        
        if not extracted:
            # Fallback: try to find any JSON-like structure in the text
            self.log_fn("No JSON blocks found, trying fallback extraction")
            try:
                # Look for patterns like "response": "..." or "grade": "..."
                response_match = re.search(r'["\']response["\']\s*:\s*["\']([^"\']+)["\']', last_assistant_msg)
                if response_match:
                    grade = _normalize_grade(response_match.group(1))
                    self.log_fn(f"Extracted grade from text pattern: {grade}")
                    return grade
                grade_match = re.search(r'["\']grade["\']\s*:\s*["\']([^"\']+)["\']', last_assistant_msg)
                if grade_match:
                    grade = _normalize_grade(grade_match.group(1))
                    self.log_fn(f"Extracted grade from text pattern: {grade}")
                    return grade
                score_match = re.search(r'["\']score["\']\s*:\s*["\']?([^"\'\s,}]+)', last_assistant_msg)
                if score_match:
                    grade = _normalize_grade(score_match.group(1))
                    self.log_fn(f"Extracted grade from text pattern: {grade}")
                    return grade
                # Try to find grade/score in plain text
                text_grade_match = re.search(r'(?:grade|score|result)\s*[:=]\s*["\']?([^"\'\n,}]+)', last_assistant_msg, re.IGNORECASE)
                if text_grade_match:
                    grade = _normalize_grade(text_grade_match.group(1).strip())
                    self.log_fn(f"Extracted grade from text: {grade}")
                    return grade
                # Try to find standalone grade words
                for word in ["Correct", "Almost", "Partial", "Incorrect"]:
                    if re.search(rf'\b{word}\b', last_assistant_msg):
                        self.log_fn(f"Extracted grade from standalone word: {word}")
                        return word
            except Exception as e:
                self.log_fn(f"Fallback extraction failed: {e}")
            return "None"
        
        # Try to get response from extracted JSON
        last_extract = extracted[-1]
        
        # Log detailed analysis for debugging
        if "understanding" in last_extract:
            self.log_fn(f"Understanding: {str(last_extract['understanding'])[:200]}...")
        if "analysis" in last_extract:
            self.log_fn(f"Analysis: {str(last_extract['analysis'])[:200]}...")
        if "partial_credit_reasoning" in last_extract:
            self.log_fn(f"Partial Credit: {str(last_extract['partial_credit_reasoning'])[:200]}...")
        if "grade_category_check" in last_extract:
            self.log_fn(f"Category Check: {str(last_extract['grade_category_check'])[:200]}...")
        
        # Priority order for field names - response is most important
        field_priority = ["response", "grade", "score", "result", "final_grade", "evaluation", "verdict", "category", "classification"]
        
        for field in field_priority:
            if field in last_extract:
                value = last_extract[field]
                grade = _normalize_grade(value)
                self.log_fn(f"Extracted grade from JSON field '{field}': {grade}")
                return grade
        
        # If no known field found, return the first string value that looks like a grade
        for key, value in last_extract.items():
            if isinstance(value, str) and value:
                normalized = _normalize_grade(value)
                if normalized in ["Correct", "Almost", "Partial", "Incorrect", "0", "1", "2", "3", "4", "5", "6", "7"]:
                    self.log_fn(f"Extracted grade from first valid string value '{key}': {normalized}")
                    return normalized
        
        # Last resort: try any string value
        for key, value in last_extract.items():
            if isinstance(value, str) and value:
                grade = _normalize_grade(value)
                self.log_fn(f"Extracted grade from first string value '{key}': {grade}")
                return grade
        
        self.log_fn("Warning: Could not extract grade from JSON")
        return "None"


def _normalize_grade(grade: str | int | float) -> str:
    """Normalize grade to standard format.
    
    Converts various grade formats to a consistent representation.
    Handles both string grades (Correct/Partial/Incorrect/Almost) and numeric scores (0-7).
    
    Args:
        grade: The grade value to normalize (can be string, int, or float)
        
    Returns:
        Normalized grade string: 'Correct', 'Almost', 'Partial', 'Incorrect', or '0'-'7'
    """
    if grade is None:
        return "None"
    
    # Handle non-string types (int, float, etc.)
    if isinstance(grade, (int, float)):
        # IMO scores are integers 0-7
        if 0 <= grade <= 7:
            return str(int(grade))
        return str(grade)
    
    if not isinstance(grade, str):
        return str(grade)
    
    grade = grade.strip()
    if not grade:
        return "None"
    
    # First check for exact match (case-sensitive) - this is most reliable
    valid_grades = {"Correct", "Almost", "Partial", "Incorrect", "0", "1", "2", "3", "4", "5", "6", "7"}
    if grade in valid_grades:
        return grade
    
    grade_lower = grade.lower()
    
    # Map common variations to standard grades
    # IMPORTANT: Order matters - check for more specific patterns first
    grade_map = {
        # Correct variations
        "correct": "Correct",
        "right": "Correct",
        "true": "Correct",
        "yes": "Correct",
        "full": "Correct",
        "full credit": "Correct",
        "complete": "Correct",
        "solved": "Correct",
        "valid": "Correct",
        "acceptable": "Correct",
        "perfect": "Correct",
        "fully correct": "Correct",
        "all correct": "Correct",
        "entirely correct": "Correct",
        "totally correct": "Correct",
        # Almost variations - check these BEFORE partial/correct
        "almost": "Almost",
        "almost correct": "Almost",
        "nearly correct": "Almost",
        "minor errors": "Almost",
        "small errors": "Almost",
        "mostly correct": "Almost",
        "nearly": "Almost",
        "essentially correct": "Almost",
        "minor flaw": "Almost",
        "minor mistake": "Almost",
        "small mistake": "Almost",
        "trivial error": "Almost",
        "slight error": "Almost",
        "close": "Almost",
        "very close": "Almost",
        "nearly there": "Almost",
        "almost there": "Almost",
        "almost solved": "Almost",
        "mostly right": "Almost",
        "mostly solved": "Almost",
        # Partial variations
        "partial": "Partial",
        "partially correct": "Partial",
        "partial credit": "Partial",
        "partially": "Partial",
        "some credit": "Partial",
        "incomplete": "Partial",
        "unfinished": "Partial",
        "significant progress": "Partial",
        "some progress": "Partial",
        "halfway": "Partial",
        "half done": "Partial",
        "partial solution": "Partial",
        "in progress": "Partial",
        "started correctly": "Partial",
        "on the right track": "Partial",
        "good start": "Partial",
        "some understanding": "Partial",
        "partial understanding": "Partial",
        # Incorrect variations
        "incorrect": "Incorrect",
        "wrong": "Incorrect",
        "false": "Incorrect",
        "no": "Incorrect",
        "none": "Incorrect",
        "zero": "Incorrect",
        "invalid": "Incorrect",
        "unacceptable": "Incorrect",
        "fail": "Incorrect",
        "failed": "Incorrect",
        "not correct": "Incorrect",
        "not right": "Incorrect",
        "not valid": "Incorrect",
        "completely wrong": "Incorrect",
        "totally wrong": "Incorrect",
        "entirely wrong": "Incorrect",
        "fundamentally wrong": "Incorrect",
        "wrong approach": "Incorrect",
        "incorrect approach": "Incorrect",
        "doesn't understand": "Incorrect",
        "no understanding": "Incorrect",
        "blank": "Incorrect",
        "empty": "Incorrect",
        "no answer": "Incorrect",
        "irrelevant": "Incorrect",
        "nonsense": "Incorrect",
        "garbage": "Incorrect",
        # Numeric scores
        "0": "0",
        "1": "1",
        "2": "2",
        "3": "3",
        "4": "4",
        "5": "5",
        "6": "6",
        "7": "7",
    }
    
    # Check for exact match first (case-insensitive)
    if grade_lower in grade_map:
        return grade_map[grade_lower]
    
    # Try to extract numeric score from patterns like "Score: 5", "Grade: 3", "5/7", "(6)"
    # Pattern for standalone digit 0-7 (word boundary to avoid matching parts of larger numbers)
    numeric_match = re.search(r'(?:^|\s|\()[0-7](?:$|\s|\)|\.|,|;)', grade)
    if numeric_match:
        # Extract just the digit
        digit_match = re.search(r'[0-7]', numeric_match.group(0))
        if digit_match:
            return digit_match.group(0)
    
    # Pattern for "X points" or "X out of 7"
    points_match = re.search(r'(\d+)\s*(?:points?|pts?|/\s*7|out\s+of\s*7)', grade_lower)
    if points_match:
        score = int(points_match.group(1))
        if 0 <= score <= 7:
            return str(score)
    
    # Check for grade in parentheses or brackets
    bracket_match = re.search(r'[\(\[]([0-7]|correct|partial|incorrect|almost)[\)\]]', grade_lower)
    if bracket_match:
        inner = bracket_match.group(1)
        if inner in grade_map:
            return grade_map[inner]
        if inner.isdigit() and 0 <= int(inner) <= 7:
            return inner
    
    # Check for grade in quotes
    quote_match = re.search(r'["\'](correct|partial|incorrect|almost|none)["\']', grade_lower)
    if quote_match:
        inner = quote_match.group(1)
        if inner in grade_map:
            return grade_map[inner]
    
    # Check if grade contains keywords (order matters - check specific categories first)
    # Check for "almost" category first (nearly correct with minor flaws)
    if any(word in grade_lower for word in ["almost", "nearly correct", "minor error", "small error", "trivial", "essentially correct", "nearly", "minor mistake", "small mistake", "trivial error", "slight error", "close", "very close", "nearly there", "almost there", "almost solved", "mostly right", "mostly solved"]):
        return "Almost"
    # Check for "partial" category (significant progress but substantial gaps)
    if "partial" in grade_lower or "partially" in grade_lower:
        return "Partial"
    if any(word in grade_lower for word in ["incomplete", "unfinished", "missing", "significant progress", "some progress", "halfway", "half done", "partial solution", "in progress", "started correctly", "on the right track", "good start", "some understanding", "partial understanding"]):
        return "Partial"
    # Check for "incorrect" category (fundamentally wrong) - check BEFORE correct to avoid misclassification
    if any(word in grade_lower for word in ["incorrect", "wrong", "false", "none", "zero", "error", "invalid", "unacceptable", "fail", "failed", "not correct", "not right", "not valid", "completely wrong", "totally wrong", "entirely wrong", "fundamentally wrong", "wrong approach", "incorrect approach", "doesn't understand", "no understanding", "blank", "empty", "no answer", "irrelevant", "nonsense", "garbage"]):
        return "Incorrect"
    # Check for "correct" category (fully correct) - check LAST to avoid catching "not correct", "incorrect", etc.
    if any(word in grade_lower for word in ["correct", "right", "true", "full", "complete", "solved", "valid", "acceptable", "perfect", "fully correct", "all correct", "entirely correct", "totally correct"]):
        return "Correct"
    
    # If it's a single digit 0-7, return it
    if grade.isdigit() and len(grade) == 1 and 0 <= int(grade) <= 7:
        return grade
    
    # Default: return the original grade
    return grade
