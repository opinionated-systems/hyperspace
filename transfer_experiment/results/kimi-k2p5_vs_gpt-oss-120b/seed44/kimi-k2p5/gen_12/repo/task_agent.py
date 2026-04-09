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
        
        # Try multiple parsing strategies
        parsed = None
        
        # Strategy 1: Direct parse
        try:
            parsed = json.loads(inner)
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Remove markdown code block markers
        if parsed is None:
            try:
                cleaned = inner.replace("```json", "").replace("```", "").strip()
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                pass
        
        # Strategy 3: Handle single quotes instead of double quotes
        if parsed is None:
            try:
                cleaned = inner.replace("'", '"').strip()
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                pass
        
        # Strategy 4: Extract just the response field using regex
        if parsed is None:
            try:
                match = re.search(r'["\']response["\']\s*:\s*["\']([^"\']+)["\']', inner)
                if match:
                    val = match.group(1)
                    if val in {"correct", "almost", "partial", "incorrect"}:
                        parsed = {"response": val}
            except Exception:
                pass
        
        if parsed is not None and isinstance(parsed, dict) and "response" in parsed:
            results.append(parsed)
            
    return results or None


def _extract_json_from_raw_text(text: str) -> dict | None:
    """Extract JSON object from raw text without <json> tags.
    
    Looks for JSON objects with "response" field directly in the text.
    """
    # Strategy 1: Try to find JSON object patterns directly with response field
    json_pattern = re.search(r'\{\s*["\']response["\']\s*:\s*["\']([^"\']+)["\']\s*\}', text, re.IGNORECASE)
    if json_pattern:
        try:
            return json.loads(json_pattern.group(0).replace("'", '"'))
        except json.JSONDecodeError:
            pass
    
    # Strategy 2: Try to find any JSON-like object with response field
    json_pattern = re.search(r'\{[^}]*["\']response["\'][^}]*\}', text, re.DOTALL | re.IGNORECASE)
    if json_pattern:
        try:
            parsed = json.loads(json_pattern.group(0).replace("'", '"'))
            if "response" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass
    
    # Strategy 3: Try to find JSON in markdown code blocks
    code_block_pattern = re.search(r'```(?:json)?\s*\n?([^`]+)```', text, re.DOTALL | re.IGNORECASE)
    if code_block_pattern:
        try:
            parsed = json.loads(code_block_pattern.group(1).strip())
            if "response" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass
    
    # Strategy 4: Look for response field with various quote styles
    response_match = re.search(r'response["\']?\s*[:=]\s*["\']?(correct|almost|partial|incorrect)["\']?', text, re.IGNORECASE)
    if response_match:
        val = response_match.group(1).lower()
        if val in {"correct", "almost", "partial", "incorrect"}:
            return {"response": val}
    
    return None


def _analyze_grading_guidelines(grading_guidelines: str) -> dict:
    """Analyze the grading guidelines to determine the expected grade.
    
    The grading guidelines have a specific structure with sections:
    - (Partial) section: Lists achievements that earn partial credit
    - (Almost) section: Describes what was "almost" achieved but had issues
    
    KEY RULE: If the (Almost) section exists and has content describing issues,
    the expected grade is "almost" (not "correct").
    
    The "almost" grade indicates the solution is nearly correct but has minor
    issues preventing full correctness. The "partial" grade indicates more 
    significant gaps in the solution.
    """
    guidelines_lower = grading_guidelines.lower()
    
    # Check for Almost section and extract its content
    has_almost_section = "(almost)" in guidelines_lower
    almost_section_text = ""
    almost_has_content = False
    almost_has_minor_issues = False
    almost_has_major_gaps = False
    
    if has_almost_section:
        almost_start = guidelines_lower.find("(almost)")
        next_section = guidelines_lower.find("(", almost_start + 8)  # Skip past "(almost)"
        if next_section > almost_start:
            almost_section_text = guidelines_lower[almost_start:next_section]
        else:
            almost_section_text = guidelines_lower[almost_start:]
        # Check if Almost section has meaningful content (not just the header)
        # Look for numbered items or substantive text after the header
        content_after_header = almost_section_text.replace("(almost)", "").strip()
        almost_has_content = len(content_after_header) > 5
        
        # Analyze the nature of issues in the Almost section
        almost_lower = almost_section_text.lower()
        minor_indicators = [
            "minor mistakes", "verification contains", "almost complete",
            "not completed", "omitted", "small gap", "minor",
            "verification", "mistakes only"
        ]
        major_indicators = [
            "failed to prove", "did not verify", "fundamental", "wrong approach",
            "no meaningful progress", "incomplete proof", "significant",
            "major gaps"
        ]
        almost_has_minor_issues = any(ind in almost_lower for ind in minor_indicators)
        almost_has_major_gaps = any(ind in almost_lower for ind in major_indicators)
    
    # Check for Partial section
    has_partial_section = "(partial)" in guidelines_lower
    
    # Determine expected grade based on guideline structure
    # Priority: almost > partial > incorrect
    # KEY CHANGE: If (Almost) section exists with ANY content, default to "almost"
    # Only downgrade to "partial" if there are explicit major gap indicators
    expected_grade = None
    if has_almost_section and almost_has_content:
        # Default to "almost" when (Almost) section exists with content
        # This is the most common case - minor issues in an otherwise good solution
        if almost_has_major_gaps and not almost_has_minor_issues:
            expected_grade = "partial"  # Only major gaps suggest partial
        else:
            expected_grade = "almost"  # Default: (Almost) section with content = "almost" grade
    elif has_partial_section:
        expected_grade = "partial"
    
    return {
        "has_almost_section": has_almost_section,
        "almost_has_content": almost_has_content,
        "almost_has_minor_issues": almost_has_minor_issues,
        "almost_has_major_gaps": almost_has_major_gaps,
        "almost_section_preview": almost_section_text[:200] if almost_section_text else "",
        "has_partial_section": has_partial_section,
        "expected_grade_hint": expected_grade,
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
        
        # Check if student answer is incomplete (ends abruptly)
        student_answer_stripped = student_answer.strip()
        is_incomplete = False
        if student_answer_stripped:
            # Check for abrupt endings (no conclusion, ends mid-sentence, etc.)
            last_chars = student_answer_stripped[-50:].lower()
            incomplete_indicators = [
                "...", "case", "if", "then", "thus", "so", "we", "let", "since",
                "therefore", "hence", "it follows", "this implies"
            ]
            # Check if it doesn't end with a proper conclusion marker
            proper_endings = ["."]
            has_proper_ending = any(student_answer_stripped.endswith(end) for end in proper_endings)
            
            # Check for mid-sentence endings
            if not has_proper_ending or any(last_chars.endswith(ind) for ind in ["case", "if", "then", "we", "let", "since"]):
                is_incomplete = True
        
        # Build grade guidance based on guideline structure
        grade_guidance = ""
        if guideline_analysis["expected_grade_hint"] == "almost":
            grade_guidance = """
CRITICAL GUIDANCE: The grading guidelines contain an (Almost) section with content describing issues.
This is the PRIMARY SIGNAL for the "almost" grade. The solution is nearly correct but has minor issues.
YOU MUST GRADE THIS AS "almost" - this is the strongest indicator in the guidelines.
DO NOT grade as "correct" or "partial" when the (Almost) section exists with content."""
        elif guideline_analysis["expected_grade_hint"] == "partial":
            if guideline_analysis["has_almost_section"]:
                grade_guidance = """
CRITICAL GUIDANCE: The grading guidelines contain an (Almost) section with SIGNIFICANT issues or gaps.
This suggests the expected grade is "partial" - the solution has correct elements but significant gaps remain.
Note: The (Almost) section exists but describes major gaps rather than minor issues."""
            else:
                grade_guidance = """
CRITICAL GUIDANCE: The grading guidelines contain a (Partial) section but NO (Almost) section.
This suggests the expected grade is "partial" - the solution has some correct elements but significant gaps.
Since there's no (Almost) section, "almost" is NOT the correct grade."""
        
        # Add guidance about incomplete answers
        if is_incomplete:
            grade_guidance += """
ADDITIONAL GUIDANCE: The student's answer appears to be INCOMPLETE (ends abruptly or mid-sentence).
An incomplete answer CANNOT be graded as "correct" or "almost". Consider "partial" if some progress was made, or "incorrect" if no meaningful progress."""
        
        instruction = f"""You are an expert mathematical grader. Your task is to evaluate a student's answer to a mathematics problem and output ONLY a grade label.

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

- "correct": The solution is fully correct with NO mistakes, NO gaps, and NO unproven claims.
   * The proof must be complete - all claims must be justified with actual proofs, not just stated.
   * Citing external theorems without proof is NOT a complete solution unless those theorems are trivial.
   * Only use this if the (Almost) section is EMPTY or does not exist.
   * If the (Almost) section mentions ANY issues, the grade CANNOT be "correct".

- "almost": The solution is NEARLY complete and correct, but has MINOR issues preventing full correctness:
   * KEY INDICATOR: The (Almost) section exists with content describing specific issues
   * "Verification contains minor mistakes only" → ALMOST
   * "Applied [strategy] but not completed" → ALMOST  
   * "Solution is almost complete" → ALMOST
   * "minor mistakes which are not negligible" → ALMOST
   * "Omitted [some case/verification]" → ALMOST
   * "made minor mistakes which are not negligible" → ALMOST
   The "almost" grade is SPECIFICALLY for when the (Almost) section exists with content - this is the PRIMARY signal.

- "partial": The solution has some correct elements but SIGNIFICANT gaps remain:
   * The student achieved items from the (Partial) section
   * Missing key insights or major steps (not just minor issues)
   * "failed to prove" or "did not verify" important claims
   * The solution shows understanding but is incomplete
   * The answer ends abruptly or is clearly incomplete
   * Use this when there's NO (Almost) section but the solution has gaps

- "incorrect": The solution is wrong or fundamentally flawed:
   * Missing most or all key insights from the (Partial) section
   * The approach is wrong or the proof is invalid
   * No meaningful progress toward the solution
   * Relies on unproven theorems or claims without verification
   * The answer is incomplete and doesn't achieve the (Partial) items

CRITICAL DECISION RULES:
1. CHECK FOR (ALMOST) SECTION FIRST: If the (Almost) section exists and has content describing issues → grade MUST be "almost" (NOT "correct", NOT "partial")
2. "correct" requires: NO (Almost) section with content, ZERO mistakes, NO gaps, NO unproven claims
3. "partial" is for solutions with some correct elements but NO (Almost) section - significant gaps but not "almost" level
4. "incorrect" is for solutions that are fundamentally wrong or show no meaningful progress
5. The (Partial) section lists achievements - if the student achieved these, they deserve at least "partial"
6. A complete-looking proof that relies on unproven theorems or makes unverified claims is NOT "correct"
7. DECISION FLOW: Check (Almost) section first → If exists with content, grade "almost" → If no (Almost), check if solution has significant gaps → "partial" or "incorrect"
{grade_guidance}

Based on this analysis, determine if the student's answer is:
- "correct" - Fully correct, complete proof with NO mistakes, NO gaps, NO unproven claims, and NO (Almost) section with content
- "almost" - The (Almost) section exists with content describing issues. This is the PRIMARY SIGNAL. The solution is nearly correct with minor issues.
- "partial" - Has correct elements but NO (Almost) section. Significant gaps remain (achieved some Partial items).
- "incorrect" - Wrong, fundamentally flawed, relies on unproven claims, or no meaningful progress

DECISION CHECKLIST:
1. Does the (Almost) section exist with content? → Grade "almost"
2. Is there NO (Almost) section but the solution has some correct elements? → Grade "partial"  
3. Is the solution fundamentally wrong with no meaningful progress? → Grade "incorrect"
4. Is the solution fully correct with NO issues in (Almost) section? → Grade "correct"

---

OUTPUT FORMAT - EXTREMELY IMPORTANT:

You MUST respond with EXACTLY ONE of these four options, and NOTHING ELSE:

<json>{{"response": "correct"}}</json>

OR

<json>{{"response": "almost"}}</json>

OR

<json>{{"response": "partial"}}</json>

OR

<json>{{"response": "incorrect"}}</json>

RULES:
1. Output ONLY the JSON object wrapped in <json> tags
2. Do NOT include any explanations, reasoning, or additional text
3. Do NOT use markdown code blocks (```)
4. Do NOT add any other words before or after the JSON
5. Your entire response must be exactly one of the four options above"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with multiple fallback strategies
        prediction = "None"
        response_text = msg_history[-1]["text"]
        
        # First, try to extract from <json> tags
        try:
            extracted = _extract_jsons(response_text)
            if extracted:
                for item in extracted:
                    if isinstance(item, dict) and "response" in item:
                        val = item["response"]
                        if val in {"correct", "almost", "partial", "incorrect"}:
                            prediction = val
                            break
        except Exception as e:
            self.log_fn(f"Error extracting JSON from tags: {e}")
        
        # Try raw text JSON extraction if tag-based extraction failed
        if prediction == "None" or prediction not in {"correct", "almost", "partial", "incorrect"}:
            try:
                raw_extracted = _extract_json_from_raw_text(response_text)
                if raw_extracted and "response" in raw_extracted:
                    val = raw_extracted["response"]
                    if val in {"correct", "almost", "partial", "incorrect"}:
                        prediction = val
            except Exception as e:
                self.log_fn(f"Error extracting raw JSON: {e}")
        
        # If JSON extraction failed, use text-based extraction with strict priority
        if prediction == "None" or prediction not in {"correct", "almost", "partial", "incorrect"}:
            text = response_text.lower()
            
            # Priority order: almost > partial > incorrect > correct
            # This order helps distinguish between overlapping keywords
            
            # Check for response field patterns (most reliable)
            if 'response": "almost"' in text or "response': 'almost'" in text:
                prediction = "almost"
            elif 'response": "partial"' in text or "response': 'partial'" in text:
                prediction = "partial"
            elif 'response": "incorrect"' in text or "response': 'incorrect'" in text:
                prediction = "incorrect"
            elif 'response": "correct"' in text or "response': 'correct'" in text:
                prediction = "correct"
            # Check for quoted grade values
            elif '"almost"' in text or "'almost'" in text:
                prediction = "almost"
            elif '"partial"' in text or "'partial'" in text:
                prediction = "partial"
            elif '"incorrect"' in text or "'incorrect'" in text:
                prediction = "incorrect"
            elif '"correct"' in text or "'correct'" in text:
                prediction = "correct"
            # Check for grade statements with context
            elif "grade is almost" in text or "grade: almost" in text:
                prediction = "almost"
            elif "grade is partial" in text or "grade: partial" in text:
                prediction = "partial"
            elif "grade is incorrect" in text or "grade: incorrect" in text:
                prediction = "incorrect"
            elif "grade is correct" in text or "grade: correct" in text:
                prediction = "correct"
            # Check for standalone grade words at start of lines
            elif re.search(r'^(almost|partial|incorrect|correct)[\s\n:]', text, re.MULTILINE | re.IGNORECASE):
                match = re.search(r'^(almost|partial|incorrect|correct)[\s\n:]', text, re.MULTILINE | re.IGNORECASE)
                if match:
                    prediction = match.group(1).lower()
            # Final fallback: keyword presence (least reliable, maintain priority)
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
