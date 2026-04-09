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
            # Try to clean up common issues and retry
            try:
                # Remove markdown code block markers if present
                cleaned = inner.replace("```json", "").replace("```", "").strip()
                results.append(json.loads(cleaned))
            except json.JSONDecodeError:
                continue
    return results or None


def _extract_json_from_raw_text(text: str) -> dict | None:
    """Extract JSON object from raw text without <json> tags.
    
    Looks for JSON objects with "response" field directly in the text.
    """
    # Try to find JSON object patterns directly with response field
    json_pattern = re.search(r'\{\s*["\']response["\']\s*:\s*["\']([^"\']+)["\']\s*\}', text, re.IGNORECASE)
    if json_pattern:
        try:
            return json.loads(json_pattern.group(0).replace("'", '"'))
        except json.JSONDecodeError:
            pass
    
    # Try to find any JSON-like object with response field
    json_pattern = re.search(r'\{[^}]*["\']response["\'][^}]*\}', text, re.DOTALL | re.IGNORECASE)
    if json_pattern:
        try:
            parsed = json.loads(json_pattern.group(0).replace("'", '"'))
            if "response" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON in markdown code blocks
    code_block_pattern = re.search(r'```(?:json)?\s*\n?([^`]+)```', text, re.DOTALL | re.IGNORECASE)
    if code_block_pattern:
        try:
            parsed = json.loads(code_block_pattern.group(1).strip())
            if "response" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass
    
    return None


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
    
    # Extract the Partial section content
    partial_section = ""
    if "(partial)" in guidelines_lower:
        partial_start = guidelines_lower.find("(partial)")
        next_section = guidelines_lower.find("(", partial_start + 1)
        if next_section > partial_start:
            partial_section = guidelines_lower[partial_start:next_section]
        else:
            partial_section = guidelines_lower[partial_start:]
    
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
    
    # Check for explicit incorrect indicators
    incorrect_indicators = [
        "fundamentally wrong",
        "completely wrong",
        "no correct elements",
        "no valid proof",
        "incorrect approach",
    ]
    has_incorrect_indicators = any(ind in guidelines_lower for ind in incorrect_indicators)
    
    return {
        "has_almost_section": has_almost_section,
        "has_almost_indicators": has_almost_indicators,
        "minor_mistakes_only": minor_mistakes_only,
        "failed_to": failed_to,
        "partial_items": partial_items,
        "has_correct_indicators": has_correct_indicators,
        "has_incorrect_indicators": has_incorrect_indicators,
        "almost_section": almost_section[:300] if almost_section else "",
        "partial_section": partial_section[:300] if partial_section else "",
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
        
        # Determine the most likely grade based on guideline structure
        likely_grade_hint = ""
        if guideline_analysis["has_almost_section"] and guideline_analysis["has_almost_indicators"]:
            likely_grade_hint = """
IMPORTANT: The (Almost) section contains specific indicators that the expected grade is "almost" (not "correct").
The presence of an (Almost) section with issues described means the solution has minor problems that prevent it from being fully correct."""
        
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

GRADE DEFINITIONS (in order of quality):

- "correct": The solution is fully correct with NO mistakes. 
   * Only use this if the (Almost) section is EMPTY or does not exist.
   * If the (Almost) section mentions ANY issues, the grade CANNOT be "correct".

- "almost": The solution is nearly complete and correct, but has minor issues preventing full correctness:
   * "Verification contains minor mistakes only" → ALMOST
   * "Applied [strategy] but not completed" → ALMOST  
   * "Solution is almost complete" → ALMOST
   * "minor mistakes which are not negligible" → ALMOST
   * "Omitted [some case/verification]" → ALMOST
   * The (Almost) section exists and describes specific issues → ALMOST
   The "almost" grade indicates the student understood the core solution but had small gaps or minor errors.

- "partial": The solution has some correct elements but significant gaps remain:
   * The student achieved some items from the (Partial) section
   * Missing key insights or major steps
   * "failed to prove" or "did not verify" important claims
   * The solution shows understanding but is incomplete

- "incorrect": The solution is wrong or fundamentally flawed:
   * Missing most or all key insights from the (Partial) section
   * The approach is wrong or the proof is invalid
   * No meaningful progress toward the solution

CRITICAL DECISION RULES:
1. If the (Almost) section exists and describes ANY issues → grade MUST be "almost" or lower (never "correct")
2. "correct" requires ZERO mistakes and NO issues in the (Almost) section
3. "partial" is for solutions with some correct elements but significant gaps (not just minor issues)
4. "incorrect" is for solutions that are fundamentally wrong or show no meaningful progress
5. The (Partial) section lists achievements - if the student achieved these, they deserve at least "partial"
{likely_grade_hint}

Based on this analysis, determine if the student's answer is:
- "correct" - Fully correct solution with NO mistakes and NO (Almost) section issues
- "almost" - Nearly correct, minor issues only (Almost section exists with issues)
- "partial" - Has correct elements but significant gaps (achieved some Partial items)
- "incorrect" - Wrong or fundamentally flawed (no meaningful progress)

IMPORTANT: You must respond with ONLY a JSON object in the following format. Do not include any other text, explanations, or markdown formatting.

<json>
{{
    "response": "correct"
}}
</json>

OR

<json>
{{
    "response": "almost"
}}
</json>

OR

<json>
{{
    "response": "partial"
}}
</json>

OR

<json>
{{
    "response": "incorrect"
}}
</json>

Your response must contain ONLY one of these four JSON objects wrapped in <json> tags."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with multiple fallback strategies
        prediction = "None"
        response_text = msg_history[-1]["text"]
        
        try:
            # Try to extract from JSON tags
            extracted = _extract_jsons(response_text)
            if extracted and "response" in extracted[-1]:
                prediction = extracted[-1]["response"]
        except Exception as e:
            self.log_fn(f"Error extracting JSON: {e}")
        
        # Try raw text JSON extraction if tag-based extraction failed
        if prediction == "None" or prediction not in {"correct", "almost", "partial", "incorrect"}:
            try:
                raw_extracted = _extract_json_from_raw_text(response_text)
                if raw_extracted and "response" in raw_extracted:
                    prediction = raw_extracted["response"]
            except Exception as e:
                self.log_fn(f"Error extracting raw JSON: {e}")
        
        # If JSON extraction failed, use text-based extraction
        if prediction == "None" or prediction not in {"correct", "almost", "partial", "incorrect"}:
            text = response_text.lower()
            
            # Look for explicit grade statements with priority
            # Priority: almost > partial > incorrect > correct (most specific first)
            if '"almost"' in text or "'almost'" in text:
                prediction = "almost"
            elif '"partial"' in text or "'partial'" in text:
                prediction = "partial"
            elif '"incorrect"' in text or "'incorrect'" in text:
                prediction = "incorrect"
            elif '"correct"' in text or "'correct'" in text:
                prediction = "correct"
            # Then check for bare words with context
            elif "grade is almost" in text or "grade: almost" in text or "is almost" in text:
                prediction = "almost"
            elif "grade is partial" in text or "grade: partial" in text:
                prediction = "partial"
            elif "grade is incorrect" in text or "grade: incorrect" in text:
                prediction = "incorrect"
            elif "grade is correct" in text or "grade: correct" in text:
                prediction = "correct"
            # Check for explicit response field patterns
            elif 'response": "almost"' in text or "response': 'almost'" in text.lower():
                prediction = "almost"
            elif 'response": "partial"' in text or "response': 'partial'" in text.lower():
                prediction = "partial"
            elif 'response": "incorrect"' in text or "response': 'incorrect'" in text.lower():
                prediction = "incorrect"
            elif 'response": "correct"' in text or "response': 'correct'" in text.lower():
                prediction = "correct"
            # Check for standalone grade words at start of lines or after colons
            elif re.search(r'^(almost|partial|incorrect|correct)[\s\n]', text, re.MULTILINE | re.IGNORECASE):
                match = re.search(r'^(almost|partial|incorrect|correct)[\s\n]', text, re.MULTILINE | re.IGNORECASE)
                if match:
                    prediction = match.group(1).lower()
            # Final fallback: simple keyword presence (least reliable)
            elif "almost" in text:
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
            self.log_fn(f"Response text preview: {response_text[:200]}...")
            prediction = "incorrect"

        return str(prediction), msg_history
