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
    """Extract JSON objects from <json>...</json> blocks with improved robustness."""
    results = []
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
        
        # Try multiple parsing strategies
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # Strategy 1: Try to find JSON object boundaries
            try:
                json_start = inner.find("{")
                json_end = inner.rfind("}")
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    results.append(json.loads(inner[json_start:json_end+1]))
            except json.JSONDecodeError:
                # Strategy 2: Try to fix common JSON issues
                try:
                    # Remove trailing commas before closing braces
                    fixed = re.sub(r',\s*}', '}', inner)
                    fixed = re.sub(r',\s*]', ']', fixed)
                    # Fix single quotes to double quotes for keys
                    fixed = re.sub(r"'([^']*)'\s*:", r'"\1":', fixed)
                    # Fix single quotes to double quotes for string values
                    fixed = re.sub(r":\s*'([^']*)'", r': "\1"', fixed)
                    # Try to extract again
                    json_start = fixed.find("{")
                    json_end = fixed.rfind("}")
                    if json_start != -1 and json_end != -1 and json_end > json_start:
                        results.append(json.loads(fixed[json_start:json_end+1]))
                except json.JSONDecodeError:
                    continue
    return results or None


def _extract_label_from_text(text: str) -> str | None:
    """Extract grading label from raw text with improved pattern matching."""
    text_lower = text.lower()
    
    # Priority order: check for explicit JSON-like patterns first
    # These are more reliable than standalone words
    patterns = [
        # JSON format patterns - double quotes (highest priority)
        r'"response"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"label"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"grade"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"evaluation"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"result"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"category"\s*:\s*"(correct|incorrect|partial|almost)"',
        # JSON format patterns - single quotes
        r"'response'\s*:\s*'(correct|incorrect|partial|almost)'",
        r"'label'\s*:\s*'(correct|incorrect|partial|almost)'",
        r"'grade'\s*:\s*'(correct|incorrect|partial|almost)'",
        r"'evaluation'\s*:\s*'(correct|incorrect|partial|almost)'",
        r"'result'\s*:\s*'(correct|incorrect|partial|almost)'",
        r"'category'\s*:\s*'(correct|incorrect|partial|almost)'",
        # Assignment patterns
        r'response\s*=\s*"(correct|incorrect|partial|almost)"',
        r'label\s*=\s*"(correct|incorrect|partial|almost)"',
        r'grade\s*=\s*"(correct|incorrect|partial|almost)"',
        r'evaluation\s*=\s*"(correct|incorrect|partial|almost)"',
        r'result\s*=\s*"(correct|incorrect|partial|almost)"',
        r'category\s*=\s*"(correct|incorrect|partial|almost)"',
        # Markdown/code block patterns
        r'response\s*:\s*(correct|incorrect|partial|almost)',
        r'label\s*:\s*(correct|incorrect|partial|almost)',
        r'grade\s*:\s*(correct|incorrect|partial|almost)',
        r'evaluation\s*:\s*(correct|incorrect|partial|almost)',
        r'result\s*:\s*(correct|incorrect|partial|almost)',
        r'category\s*:\s*(correct|incorrect|partial|almost)',
        # Conclusion patterns
        r'therefore[\s,]+(?:the\s+)?(?:answer|grade|label|category|result)\s+(?:is|should\s+be)\s+["\']?(correct|incorrect|partial|almost)["\']?',
        r'conclusion[\s:]+["\']?(correct|incorrect|partial|almost)["\']?',
        r'final[\s:]+["\']?(correct|incorrect|partial|almost)["\']?',
        r'classification[\s:]+["\']?(correct|incorrect|partial|almost)["\']?',
        r'assigned[\s:]+["\']?(correct|incorrect|partial|almost)["\']?',
        r'categorize[d]?[\s:]+["\']?(correct|incorrect|partial|almost)["\']?',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1)
    
    # Check for standalone labels with word boundaries
    # Use a scoring system to handle multiple matches
    # But be careful - "incorrect" might appear in reasoning about what's wrong
    # Prioritize labels that appear in conclusion/final sections
    
    # Look for the label in the last 200 characters (conclusion section)
    conclusion_text = text_lower[-200:] if len(text_lower) > 200 else text_lower
    
    scores = {
        "almost": 0,
        "incorrect": 0,
        "partial": 0,
        "correct": 0
    }
    
    # Count occurrences with word boundaries in conclusion (weighted higher)
    for label in scores.keys():
        conclusion_matches = re.findall(rf'\b{label}\b', conclusion_text)
        full_matches = re.findall(rf'\b{label}\b', text_lower)
        # Weight conclusion matches 3x more than body matches
        scores[label] = len(conclusion_matches) * 3 + (len(full_matches) - len(conclusion_matches))
    
    # Return the label with highest score that appears at least once
    max_score = 0
    best_label = None
    for label, score in scores.items():
        if score > max_score:
            max_score = score
            best_label = label
    
    return best_label


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer, points

        Returns:
            (prediction, msg_history)
        """
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Build an improved prompt with clearer distinctions between categories
        instruction = f"""You are an expert mathematics grader for International Mathematical Olympiad (IMO) style problems.

Your task is to evaluate a student's answer and classify it into EXACTLY ONE of four categories based on the IMO 7-point scoring system.

## LABEL DEFINITIONS (IMO 7-Point Scoring System):

**"correct" = 7 points** - Complete, rigorous solution.
- All key steps are present and correctly justified.
- No significant logical gaps or errors.
- Would receive 7/7 marks.

**"almost" = 6 points** - Nearly complete with only minor issues.
- The main proof structure is complete and correct.
- Only minor technical issues remain (e.g., small calculation error, minor gap, missing edge case).
- The student would lose EXACTLY 1 mark.
- KEY: The solution is 85-99% complete. The core argument is sound.
- CRITICAL: The solution must have ALL major proof components present. Only minor polish is missing.

**"partial" = 1 point** - Meaningful progress but substantial gaps remain.
- At least one correct non-trivial insight or step toward the solution.
- Major components of the proof are missing or incorrect.
- The student would lose 5-6 marks (receive only 1-2/7).
- KEY: The solution is 10-60% complete. There are SUBSTANTIAL gaps.
- CRITICAL: This is NOT "almost". "Partial" means the solution is FAR from complete.

**"incorrect" = 0 points** - No meaningful progress.
- Little to no valid mathematical progress demonstrated.
- Wrong method, misunderstanding of the problem, or trivial/incorrect observations only.
- Would receive 0/7 marks.

## CRITICAL DISTINCTIONS - READ CAREFULLY:

**DISTINCTION 1: "almost" (6/7) vs "partial" (1/7)**
This is the MOST COMMON source of grading errors. Be extremely careful:

- "almost" = Would lose EXACTLY 1 mark. The solution is NEARLY COMPLETE (85-99%).
  * The main proof structure is intact.
  * ALL key proof components are present.
  * Only minor technical issues remain.
  * Examples: small calculation error, minor gap in one step, missing one edge case, typo in formula.
  * The solution would be essentially correct with minor fixes.
  
- "partial" = Would lose 5-6 marks. The solution has SUBSTANTIAL GAPS (10-60%).
  * Major proof components are MISSING.
  * The solution is far from complete.
  * Examples: missing entire proof sections, major logical gaps, only initial insights present, key theorems not applied.
  * The solution requires significant additional work to be complete.

**DECISION RULE:** If the solution is missing ANY key proof component (not just minor polish), it is "partial", not "almost". When in doubt, choose "partial".

**DISTINCTION 2: "partial" (1/7) vs "incorrect" (0/7)**
- "partial" requires at least ONE correct NON-TRIVIAL insight or step.
  * Must be a genuine mathematical insight, not just restating the problem.
  * Must make actual progress toward the solution.
  * Examples: correct setup of equations, valid initial approach, correct identification of key lemma.
  
- "incorrect" means NO meaningful mathematical progress.
  * Completely wrong approach, or trivial observations only.
  * Restating the problem without progress does NOT count as partial.
  * Examples: stating definitions without application, random calculations, misunderstanding the problem statement.

**DECISION RULE:** If the student only restates the problem or makes trivial observations → "incorrect". If they have at least one correct non-trivial insight → "partial". When in doubt, choose "incorrect".

## DECISION FRAMEWORK (Apply in Order):

**Step 1: Check for "correct" (7 points)**
- Is the solution complete with all key steps justified?
- Are there no significant gaps or errors?
- Would it receive 7/7 marks?
- If YES → "correct"

**Step 2: Check for "almost" (6 points)**
- Is the main proof structure essentially complete and correct?
- Are ALL key proof components present?
- Are there only minor gaps/errors that would lose exactly 1 mark?
- Is the solution 85-99% complete with only minor polish needed?
- Would it receive 6/7 marks?
- If YES → "almost"
- IMPORTANT: If ANY major component is missing, do NOT choose "almost".

**Step 3: Check for "partial" (1 point)**
- Did the student make at least ONE correct NON-TRIVIAL step?
- Is there meaningful progress toward the solution?
- Are there substantial gaps (would lose 5-6 marks)?
- Is the solution 10-60% complete?
- If YES → "partial"
- IMPORTANT: If the solution looks like math but has no real insight, choose "incorrect" instead.

**Step 4: Default to "incorrect" (0 points)**
- If none of the above apply, the solution has no meaningful progress.
- This includes: wrong approach, trivial observations only, restating problem without progress.

## COMMON ERROR PATTERNS TO AVOID:

1. **Over-calling "partial"**: Do NOT call a solution "partial" just because it looks like math. It needs at least one correct NON-TRIVIAL insight. If in doubt, choose "incorrect".

2. **Over-calling "almost"**: Do NOT call a solution "almost" if it's missing major proof components. "Almost" requires ALL main proof components to be present. If in doubt, choose "partial".

3. **Under-calling "correct"**: If the solution is complete with only typos or trivial issues, it is "correct", not "almost".

## CONSERVATIVE GRADING GUIDELINES:

When uncertain between categories, ALWAYS choose the LOWER score:
- Doubt between "partial" and "almost"? → Choose "partial" (1 point, not 6).
- Doubt between "incorrect" and "partial"? → Choose "incorrect" (0 points, not 1).
- Doubt between "almost" and "correct"? → Choose "almost" (6 points, not 7).

**KEY PRINCIPLE**: It's better to under-grade than over-grade. Be conservative and rigorous.

## GRADING GUIDELINES CONTEXT:
The grading guidelines indicate what achievements to look for and how points are awarded:
{grading_guidelines}

Use these guidelines to determine which achievements the student has completed.

## PROBLEM:
{problem}

## OFFICIAL SOLUTION:
{solution}

## STUDENT'S ANSWER TO EVALUATE:
{student_answer}

## YOUR TASK:

Analyze the student's answer step by step:
1. Identify what achievements (if any) the student completed.
2. Identify specific gaps, errors, or missing components.
3. Estimate the completeness percentage (0%, 10-60%, 85-99%, or 100%).
4. Determine how many marks would be lost (0, 1, 5-6, or 7).
5. Select the appropriate label based on the decision framework above.

**FINAL CHECK**: Before selecting your answer, verify:
- If choosing "almost": Are ALL major proof components present? Is it truly 85-99% complete?
- If choosing "partial": Is there at least ONE genuine non-trivial insight? Is it NOT just restating the problem?
- If choosing "incorrect": Is there truly NO meaningful mathematical progress?

Respond with ONLY a JSON object in <json> tags:

<json>
{{
    "reasoning": "Analysis: (1) What achievements did the student complete? (2) What specific gaps/errors exist? (3) Estimated completeness percentage. (4) How many marks would be lost (0/1/5-6/7)? (5) Which label applies and why. Be explicit about distinguishing 'almost' vs 'partial' and 'partial' vs 'incorrect'.",
    "response": "correct" | "almost" | "partial" | "incorrect"
}}
</json>

The "response" field MUST be exactly one of: "correct", "almost", "partial", or "incorrect" (all lowercase)."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction
        prediction = "None"
        raw_text = msg_history[-1]["text"] if msg_history else ""
        
        try:
            # Strategy 1: Extract from <json> tags (most reliable)
            extracted = _extract_jsons(raw_text)
            if extracted:
                for json_obj in extracted:
                    if isinstance(json_obj, dict):
                        # Try the expected keys first
                        for key in ["response", "label", "grade", "evaluation", "result", "category"]:
                            val = json_obj.get(key, "")
                            if isinstance(val, str):
                                val_lower = val.strip().lower()
                                if val_lower in ["correct", "incorrect", "partial", "almost"]:
                                    prediction = val_lower.capitalize()
                                    break
                        # Also check all values in the JSON object
                        if prediction == "None":
                            for key, val in json_obj.items():
                                if isinstance(val, str):
                                    val_lower = val.strip().lower()
                                    if val_lower in ["correct", "incorrect", "partial", "almost"]:
                                        prediction = val_lower.capitalize()
                                        break
                        if prediction != "None":
                            break
            
            # Strategy 2: Direct text extraction as fallback
            if prediction == "None":
                text_pred = _extract_label_from_text(raw_text)
                if text_pred:
                    prediction = text_pred.capitalize()
            
            # Strategy 3: Look for explicit grade mentions in reasoning
            if prediction == "None":
                grade_patterns = [
                    r'grade[\s:]+["\']?(correct|almost|partial|incorrect)["\']?',
                    r'classification[\s:]+["\']?(correct|almost|partial|incorrect)["\']?',
                    r'assigned[\s:]+["\']?(correct|almost|partial|incorrect)["\']?',
                    r'categorize[d]?[\s:]+["\']?(correct|almost|partial|incorrect)["\']?',
                    r'category[\s:]+["\']?(correct|almost|partial|incorrect)["\']?',
                    r'label[\s:]+["\']?(correct|almost|partial|incorrect)["\']?',
                    r'evaluation[\s:]+["\']?(correct|almost|partial|incorrect)["\']?',
                    r'result[\s:]+["\']?(correct|almost|partial|incorrect)["\']?',
                ]
                for pattern in grade_patterns:
                    match = re.search(pattern, raw_text.lower())
                    if match:
                        prediction = match.group(1).capitalize()
                        break
            
            # Strategy 4: Look for the label in the reasoning section
            if prediction == "None":
                # Check for explicit mentions like "therefore the answer is partial"
                conclusion_patterns = [
                    r'therefore[\s,]+(?:the\s+)?(?:answer|grade|label|category|result|evaluation)\s+(?:is|should\s+be)\s+["\']?(correct|almost|partial|incorrect)["\']?',
                    r'conclusion[\s:]+["\']?(correct|almost|partial|incorrect)["\']?',
                    r'final[\s:]+["\']?(correct|almost|partial|incorrect)["\']?',
                    r'final\s+(?:answer|grade|label|category|result|evaluation)[\s:]+["\']?(correct|almost|partial|incorrect)["\']?',
                    r'(?:answer|grade|label|category|result|evaluation)\s+(?:is|should\s+be)\s+["\']?(correct|almost|partial|incorrect)["\']?',
                ]
                for pattern in conclusion_patterns:
                    match = re.search(pattern, raw_text.lower())
                    if match:
                        prediction = match.group(1).capitalize()
                        break
            
            # Strategy 5: Look for the label in markdown code blocks
            if prediction == "None":
                # Look for patterns like ```json or ``` with the label inside
                code_block_patterns = [
                    r'```(?:json)?\s*\{[^}]*"(?:response|label|grade|result)"\s*:\s*"(correct|almost|partial|incorrect)"[^}]*\}',
                    r'```(?:json)?\s*\{[^}]*\'(?:response|label|grade|result)\'\s*:\s*\'(correct|almost|partial|incorrect)\'[^}]*\}',
                ]
                for pattern in code_block_patterns:
                    match = re.search(pattern, raw_text.lower(), re.DOTALL)
                    if match:
                        prediction = match.group(1).capitalize()
                        break
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Final fallback - use "Incorrect" as default
        if prediction == "None" or prediction.lower() not in ["correct", "almost", "partial", "incorrect"]:
            prediction = "Incorrect"
        
        self.log_fn(f"Final prediction: {prediction}")
        return str(prediction), msg_history
