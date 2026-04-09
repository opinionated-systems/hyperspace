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
        # JSON format patterns - double quotes
        r'"response"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"label"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"grade"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"evaluation"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"result"\s*:\s*"(correct|incorrect|partial|almost)"',
        # JSON format patterns - single quotes
        r"'response'\s*:\s*'(correct|incorrect|partial|almost)'",
        r"'label'\s*:\s*'(correct|incorrect|partial|almost)'",
        r"'grade'\s*:\s*'(correct|incorrect|partial|almost)'",
        r"'evaluation'\s*:\s*'(correct|incorrect|partial|almost)'",
        r"'result'\s*:\s*'(correct|incorrect|partial|almost)'",
        # Assignment patterns
        r'response\s*=\s*"(correct|incorrect|partial|almost)"',
        r'label\s*=\s*"(correct|incorrect|partial|almost)"',
        r'grade\s*=\s*"(correct|incorrect|partial|almost)"',
        r'evaluation\s*=\s*"(correct|incorrect|partial|almost)"',
        r'result\s*=\s*"(correct|incorrect|partial|almost)"',
        # Markdown/code block patterns
        r'response\s*:\s*(correct|incorrect|partial|almost)',
        r'label\s*:\s*(correct|incorrect|partial|almost)',
        r'grade\s*:\s*(correct|incorrect|partial|almost)',
        r'evaluation\s*:\s*(correct|incorrect|partial|almost)',
        r'result\s*:\s*(correct|incorrect|partial|almost)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1)
    
    # Check for standalone labels with word boundaries
    # Use a scoring system to handle multiple matches
    scores = {
        "almost": 0,
        "incorrect": 0,
        "partial": 0,
        "correct": 0
    }
    
    # Count occurrences with word boundaries
    for label in scores.keys():
        matches = re.findall(rf'\b{label}\b', text_lower)
        scores[label] = len(matches)
    
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

## LABEL DEFINITIONS (IMO Scoring Convention):

**"correct" = 7 points** - Complete, rigorous solution with no gaps or errors.
- Full marks: the student solved the problem completely.
- All key steps are present and correctly justified.
- No significant logical gaps or errors.

**"almost" = 6 points** - Nearly complete solution with only minor issues.
- The student essentially solved the problem but lost exactly 1 point for minor gaps/errors.
- The main proof structure is complete and correct.
- Only small technical issues remain (e.g., minor calculation error, small gap in reasoning).
- Would receive 6/7 marks.

**"partial" = 1 point** - Meaningful progress but incomplete with substantial gaps.
- The student made genuine progress but is far from a complete solution.
- At least one correct non-trivial insight or step toward the solution.
- Major components of the proof are missing or incorrect.
- Would receive 1/7 marks (or at most 2/7).

**"incorrect" = 0 points** - No meaningful progress or fundamentally wrong approach.
- Little to no valid mathematical progress demonstrated.
- Wrong method, misunderstanding of the problem, or trivial/incorrect observations only.
- Would receive 0/7 marks.

## DECISION FRAMEWORK:

**Step 1: Check for "correct" (7 points)**
- Is the solution complete with all key steps justified?
- Are there no significant gaps or errors?
- If YES → "correct"

**Step 2: Check for "almost" (6 points)**
- Is the main proof structure essentially complete?
- Are there only minor gaps/errors that would lose exactly 1 mark?
- Would the solution receive 6/7 marks?
- If YES → "almost"

**Step 3: Check for "partial" (1 point)**
- Did the student make at least one correct non-trivial step?
- Is there meaningful progress toward the solution?
- Are there substantial gaps (would lose 5-6 marks)?
- If YES → "partial"

**Step 4: Default to "incorrect" (0 points)**
- If none of the above apply, the solution has no meaningful progress.

## CRITICAL DISTINCTIONS:

**"almost" vs "partial":**
- "almost" = 85-95% complete, minor issues only (6/7 marks)
- "partial" = 30-70% complete, substantial gaps (1/7 marks)
- When in doubt, if the solution is not nearly complete, choose "partial" over "almost"

**"partial" vs "incorrect":**
- "partial" requires at least one correct non-trivial insight
- "incorrect" means no valid mathematical progress

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

Analyze the student's answer and determine which label applies based on the IMO 7-point scoring system.

Respond with ONLY a JSON object in <json> tags:

<json>
{{
    "reasoning": "Analysis: (1) What achievements did the student complete? (2) What gaps/errors exist? (3) Estimated completeness (0-100%). (4) Which point category (0/1/6/7) applies and why.",
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
            # Strategy 1: Extract from <json> tags
            extracted = _extract_jsons(raw_text)
            if extracted:
                for json_obj in extracted:
                    if isinstance(json_obj, dict):
                        for key in ["response", "label", "grade", "evaluation", "result"]:
                            val = json_obj.get(key, "")
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
                    r'grade[\s:]+"?(correct|almost|partial|incorrect)"?',
                    r'classification[\s:]+"?(correct|almost|partial|incorrect)"?',
                    r'assigned[\s:]+"?(correct|almost|partial|incorrect)"?',
                    r'categorize[\s:]+"?(correct|almost|partial|incorrect)"?',
                ]
                for pattern in grade_patterns:
                    match = re.search(pattern, raw_text.lower())
                    if match:
                        prediction = match.group(1).capitalize()
                        break
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Final fallback
        if prediction == "None" or prediction.lower() not in ["correct", "almost", "partial", "incorrect"]:
            prediction = "Incorrect"
        
        self.log_fn(f"Final prediction: {prediction}")
        return str(prediction), msg_history
