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
    
    # Strategy 4: Look for grade in code blocks or specific formatting
    code_block_patterns = [
        (r'```\s*\n?\s*"grade"\s*:\s*"(almost)"', 'almost'),
        (r'```\s*\n?\s*"grade"\s*:\s*"(partial)"', 'partial'),
        (r'```\s*\n?\s*"grade"\s*:\s*"(incorrect)"', 'incorrect'),
        (r'```\s*\n?\s*"grade"\s*:\s*"(correct)"', 'correct'),
    ]
    for pattern, grade in code_block_patterns:
        if re.search(pattern, text_lower):
            return grade
    
    # Strategy 5: Look for grade keywords in the last 300 words
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
    
    # Strategy 6: Look anywhere in text as last resort (specific first)
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

        instruction = f"""You are an expert mathematical olympiad grader. Evaluate the student's solution and assign exactly one grade: "correct", "incorrect", "partial", or "almost".

## Problem:
{problem}

## Official Solution:
{solution}

## Student's Answer:
{student_answer}

## Grading Guidelines (FOLLOW THESE EXACTLY):
{grading_guidelines}

## Grade Definitions

**CORRECT**: Perfect or near-perfect solution (95-100% complete)
- All steps valid, complete proof, no significant errors
- Use ONLY when essentially perfect

**INCORRECT**: No meaningful progress or fundamentally wrong approach
- Random calculations, misunderstood problem, no valid mathematical progress
- Use when student failed to make real progress

**PARTIAL**: Significant progress but INCOMPLETE (30-75% complete)
- Found key insights, lemmas, or valid approach
- Main result NOT proven OR major sections missing
- Solution stops before completion
- **KEY**: Cannot be "almost" if main result is unproven

**ALMOST**: Nearly complete with MINOR issues only (75-95% complete)
- Main proof structure IS complete
- Main result IS established
- Only small computational errors, minor gaps, or slight issues
- **KEY**: Must have (1) main result proven, (2) 75%+ complete, (3) structure in place

## CRITICAL Decision Framework

**STEP 1: Is there meaningful progress?**
- NO real progress, wrong approach, random work → "incorrect"

**STEP 2: Is the solution 95-100% perfect?**
- YES → "correct"

**STEP 3: Is the main result proven AND structure complete?**
- NO → "partial" (regardless of how good the intermediate work is)
- YES but with minor issues (75-95% complete) → "almost"

**THE MOST IMPORTANT QUESTION:**
"Is the main theorem/result actually proven by the student?"
- If NO → Must be "partial" or "incorrect" (never "almost" or "correct")
- If YES but with minor issues → "almost"
- If YES and perfect → "correct"

**Completion Guide:**
- < 75% complete OR main result unproven → "partial"
- 75-95% complete, main result proven, minor issues → "almost"
- 95-100% complete, perfect → "correct"

**When in doubt:**
- Between "partial" and "almost" → Choose "partial" (safer)
- Between "almost" and "correct" → Choose "almost" (unless perfect)

## Grading Process

1. **Analyze Guidelines**: Look for "(Partial)", "(Almost)", "(Correct)" markers
2. **Check Student Work**: What did they actually prove? Is the main result established?
3. **Estimate Completion**: What percentage is complete? (be honest and conservative)
4. **Apply Decision Framework**: Use the steps above
5. **Final Verification**: 
   - Did student prove the main result? (YES/NO)
   - What percentage complete? (e.g., 60%, 85%, 95%)
   - Are there major gaps? (YES/NO)

## Common Mistakes to Avoid

- NEVER grade "almost" if main result is not proven
- NEVER grade "almost" if solution is < 75% complete
- "Partial" is for incomplete solutions OR when main result is unproven
- "Almost" requires ALL three: main result proven + 75%+ complete + minor issues only

## Examples

**Example 1 - PARTIAL:**
Student found key lemma and made good progress (60% complete) but main theorem unproven.
→ Grade: "partial"

**Example 2 - ALMOST:**
Student wrote complete proof (90% complete) with one small calculation error. Main result proven.
→ Grade: "almost"

**Example 3 - INCORRECT:**
Student made random calculations with no valid approach. No meaningful progress.
→ Grade: "incorrect"

**Example 4 - CORRECT:**
Student provided complete, correct proof. 100% complete, no errors.
→ Grade: "correct"

**Example 5 - CRITICAL (Main result not proven):**
Student found key lemma, set up framework, made significant progress. But main theorem never proven.
→ Grade: "partial" (NOT "almost" - main result must be proven for "almost")

## Response Format

Respond with ONLY this JSON format. The grade MUST be exactly one of: "correct", "incorrect", "partial", or "almost".

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
