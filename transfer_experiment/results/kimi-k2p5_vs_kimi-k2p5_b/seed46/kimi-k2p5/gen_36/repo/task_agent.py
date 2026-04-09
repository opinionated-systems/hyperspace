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

Your task is to evaluate a student's answer and classify it into EXACTLY ONE of four categories.

## LABEL DEFINITIONS (IMO Scoring Convention):

1. **"correct" (7 points)** - Complete, rigorous solution. All steps justified, no gaps, valid proof.
   - The student fully solved the problem with a complete, correct proof.
   - Would receive full marks (7/7).

2. **"almost" (6 points)** - Nearly complete solution with correct main strategy. Only minor errors or small gaps.
   - The student ESSENTIALLY SOLVED the problem - the core solution is correct and complete.
   - Would receive 6/7 marks (lost exactly 1 point for minor issues).
   - NOT for solutions with major gaps or missing key components.

3. **"partial" (1 point)** - Shows meaningful understanding but INCOMPLETE with substantial gaps.
   - The student made genuine progress but is far from a complete solution.
   - Would receive 1/7 marks (or at most 2/7).
   - This is NOT for "nearly complete" solutions - use "almost" for those.

4. **"incorrect" (0 points)** - Wrong approach, fundamental errors, or no meaningful progress.
   - The solution shows little to no valid mathematical progress.
   - Would receive 0/7 marks.

## CRITICAL DECISION RULES:

**"almost" vs "partial"** (MOST IMPORTANT DISTINCTION):
- "almost" = solution is NEARLY COMPLETE (85-95% done), only minor technical issues
  * The proof structure is essentially complete
  * Only small gaps or minor errors remain
  * Would lose exactly 1 mark out of 7
  
- "partial" = SIGNIFICANT PORTIONS are missing or wrong (30-70% complete)
  * Major components of the proof are missing
  * The solution is incomplete in substantial ways
  * Would lose 5-6 marks out of 7

**"partial" vs "incorrect"**:
- "partial" = student made MEANINGFUL PROGRESS with correct insights
  * At least one correct non-trivial step toward the solution
  * Shows understanding of some aspect of the problem
  
- "incorrect" = NO meaningful progress, fundamentally wrong approach
  * Wrong method or misunderstanding of the problem
  * No valid mathematical insight demonstrated

**"almost" vs "correct"**:
- "correct" = fully rigorous, would get full marks (7/7)
- "almost" = small but noticeable gaps that would lose exactly 1 mark (6/7)

## GRADING GUIDELINES CONTEXT:
The grading guidelines below indicate what achievements to look for. Use them to assess the student's progress:
{grading_guidelines}

## PROBLEM:
{problem}

## OFFICIAL SOLUTION:
{solution}

## STUDENT'S ANSWER TO EVALUATE:
{student_answer}

## YOUR TASK:

Step 1: Analyze what the student got right and what they got wrong.
Step 2: Estimate the percentage of completeness (0-100%).
Step 3: Check if the student made ANY meaningful progress toward the solution.
Step 4: Determine if the solution is substantially complete (85%+) or has significant gaps.
Step 5: Apply the decision rules above to select the appropriate grade.

IMPORTANT: 
- Be conservative in assigning "almost" - only use it when the solution is truly nearly complete with just minor issues.
- "almost" means 6/7 marks (essentially solved). "partial" means 1/7 marks (meaningful but incomplete progress).
- Most incomplete solutions should be "partial", not "almost".

Respond with ONLY a JSON object in <json> tags:

<json>
{{
    "reasoning": "Detailed analysis: (1) What did the student get right? (2) What are the gaps/errors? (3) Estimated completeness percentage. (4) Why this specific grade was chosen based on the definitions and decision rules.",
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
