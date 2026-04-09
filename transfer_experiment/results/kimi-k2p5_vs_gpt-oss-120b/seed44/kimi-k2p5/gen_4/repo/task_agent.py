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
    """Extract JSON objects from <json>...</json> blocks (case-insensitive).

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    """
    results = []
    text_lower = text.lower()
    search_from = 0
    while True:
        start = text_lower.find("<json>", search_from)
        if start == -1:
            break
        end = text_lower.find("</json>", start)
        if end == -1:
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            continue
    return results or None


def _analyze_grading_guidelines(grading_guidelines: str) -> dict:
    """Analyze the grading guidelines to understand the expected evaluation criteria.
    
    The grading guidelines have a specific structure:
    - (Partial) section: lists achievements that earn partial credit
    - (Almost) section: describes what was almost achieved but had issues
    
    Key patterns for "almost" grade:
    - "Verification contains minor mistakes only" in Almost section
    - "Applied ... but not completed" in Almost section
    - "Solution is almost complete" in Almost section
    - "minor mistakes which are not negligible" in Almost section
    - "Omitted" in Almost section
    
    The "almost" grade is between correct and partial - the solution is nearly
    correct but has minor issues that prevent it from being fully correct.
    """
    guidelines_lower = grading_guidelines.lower()
    
    # Check for Almost section and its content
    has_almost_section = "(almost)" in guidelines_lower
    
    # Extract the Almost section content (text after "(almost)" until end or next section)
    almost_section = ""
    if has_almost_section:
        almost_start = guidelines_lower.find("(almost)")
        # Look for next section marker or end
        next_section = guidelines_lower.find("(", almost_start + 1)
        if next_section > almost_start:
            almost_section = guidelines_lower[almost_start:next_section]
        else:
            almost_section = guidelines_lower[almost_start:]
    
    # Key phrases in Almost section that indicate "almost" grade
    almost_indicators = [
        "verification contains minor mistakes only",
        "applied infinite descent",
        "but not completed",
        "solution is almost complete",
        "minor mistakes which are not negligible",
        "omitted",
        "almost correct",
        "almost proved",
        "almost verified",
    ]
    
    has_almost_indicators = any(indicator in almost_section for indicator in almost_indicators)
    
    # Check for phrases that indicate the solution is essentially correct (minor issues only)
    minor_mistakes_only = "verification contains minor mistakes only" in almost_section
    
    # Check for phrases indicating partial (more significant gaps)
    failed_to = "failed to" in guidelines_lower or "did not" in guidelines_lower
    
    # Count items in Partial section (achievements)
    partial_items = guidelines_lower.count("(partial)")
    
    # Check for explicit correct indicators
    correct_indicators = [
        "fully correct",
        "completely correct",
        "correct solution",
        "no errors",
        "no mistakes",
    ]
    has_correct_indicators = any(ind in guidelines_lower for ind in correct_indicators)
    
    return {
        "has_almost_section": has_almost_section,
        "has_almost_indicators": has_almost_indicators,
        "minor_mistakes_only": minor_mistakes_only,
        "failed_to": failed_to,
        "partial_items": partial_items,
        "has_correct_indicators": has_correct_indicators,
        "almost_section": almost_section[:200] if almost_section else "",
    }


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
        # Extract key fields for better prompting
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Analyze grading guidelines structure
        guideline_analysis = _analyze_grading_guidelines(grading_guidelines)
        
        instruction = f"""You are an expert mathematical grader. Your task is to evaluate a student's answer to a mathematics problem.

DOMAIN: {domain}

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER TO EVALUATE:
{student_answer}

---

GRADING INSTRUCTIONS:

The grading guidelines have a specific structure with three key sections:

1. (Partial) section: Lists achievements that earn partial credit. These are positive accomplishments by the student.

2. (Almost) section: Describes what was "almost" achieved but had issues. This is a distinct grade between correct and partial.

3. The final grade is determined by analyzing both sections together.

GRADE DEFINITIONS (in order of quality):

- "correct": The solution is fully correct with NO mistakes. The (Almost) section may say "Verification contains minor mistakes only" but this is a special case - if there are ANY mistakes, the grade is NOT correct.

- "almost": The solution is nearly complete and correct, but has minor issues:
   * "Verification contains minor mistakes only" → ALMOST (not correct because there ARE mistakes)
   * "Applied [strategy] but not completed" → ALMOST
   * "Solution is almost complete" → ALMOST
   * "minor mistakes which are not negligible" → ALMOST
   * "Omitted [some case/verification]" → ALMOST
   The "almost" grade indicates the student understood the core solution but couldn't quite finish or had small gaps.

- "partial": The solution has correct elements but significant gaps remain:
   * Missing key insights from the (Partial) section
   * "failed to prove" or "did not verify" major claims
   * Only achieved some items in the (Partial) section

- "incorrect": The solution is wrong or fundamentally flawed, missing most key insights.

CRITICAL RULES:
1. If the (Almost) section exists and describes ANY issues, the grade is "almost" (not "correct")
2. "correct" requires ZERO mistakes - any mistake means "almost" or lower
3. "partial" is for solutions with correct elements but significant gaps
4. "incorrect" is for solutions that are fundamentally wrong

Based on this analysis, determine if the student's answer is:
- "correct" - Fully correct solution with NO mistakes
- "almost" - Nearly correct, minor issues only
- "partial" - Has correct elements but significant gaps
- "incorrect" - Wrong or significantly flawed

You must respond with ONLY a JSON object in the following format (no other text):
<json>
{{
    "response": "correct" | "almost" | "partial" | "incorrect"
}}
</json>"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with multiple fallback strategies
        prediction = "None"
        try:
            # Try to extract from JSON tags
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted and "response" in extracted[-1]:
                prediction = extracted[-1]["response"]
            else:
                # Fallback: look for direct answer in text
                text = msg_history[-1]["text"].lower()
                # Check for grade keywords with priority (most specific first)
                if "almost" in text:
                    prediction = "almost"
                elif "partial" in text:
                    prediction = "partial"
                elif "incorrect" in text:
                    prediction = "incorrect"
                elif "correct" in text:
                    prediction = "correct"
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Final fallback: simple keyword matching
            text = msg_history[-1]["text"].lower()
            # Check for grade keywords with priority (most specific first)
            if "almost" in text:
                prediction = "almost"
            elif "partial" in text:
                prediction = "partial"
            elif "incorrect" in text:
                prediction = "incorrect"
            elif "correct" in text:
                prediction = "correct"

        # Validate prediction is one of the allowed values
        valid_predictions = {"correct", "almost", "partial", "incorrect"}
        if prediction not in valid_predictions:
            self.log_fn(f"Invalid prediction '{prediction}', defaulting to 'incorrect'")
            prediction = "incorrect"

        return str(prediction), msg_history
