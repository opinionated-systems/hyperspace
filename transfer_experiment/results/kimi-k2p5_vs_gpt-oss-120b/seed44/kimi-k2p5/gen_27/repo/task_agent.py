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
    """Extract JSON objects from <json>...</json> blocks (case-insensitive)."""
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
    """Extract JSON object from raw text without <json> tags."""
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


def _extract_label_from_reasoning(text: str) -> str | None:
    """Extract the grade label from reasoning text by looking for explicit conclusions."""
    text_lower = text.lower()
    
    # Look for explicit grade statements - be comprehensive
    conclusion_patterns = [
        # Direct grade statements
        r'grade\s*(?:is|should\s*be|:)\s*["\']?(correct|almost|partial|incorrect)["\']?',
        r'(?:the\s*)?grade\s*(?:is|:)\s*["\']?(correct|almost|partial|incorrect)["\']?',
        # Conclusion statements
        r'therefore[,:]?\s*(?:the\s*)?(?:grade\s*)?(?:is\s*)?["\']?(correct|almost|partial|incorrect)["\']?',
        r'thus[,:]?\s*(?:the\s*)?(?:grade\s*)?(?:is\s*)?["\']?(correct|almost|partial|incorrect)["\']?',
        r'conclusion[,:]?\s*(?:the\s*)?(?:grade\s*)?(?:is\s*)?["\']?(correct|almost|partial|incorrect)["\']?',
        r'in\s*conclusion[,:]?\s*(?:the\s*)?(?:grade\s*)?(?:is\s*)?["\']?(correct|almost|partial|incorrect)["\']?',
        # Assignment statements
        r'assign\s*(?:the\s*)?grade\s*["\']?(correct|almost|partial|incorrect)["\']?',
        r'(?:this|the)\s*(?:solution|answer|proof|attempt)\s*(?:is|should\s*be\s*graded\s*as)\s*["\']?(correct|almost|partial|incorrect)["\']?',
        r'(?:i\s*)?(?:would\s*)?(?:grade|rate|score|mark)\s*(?:this|it|the\s*solution)\s*(?:as|at)\s*["\']?(correct|almost|partial|incorrect)["\']?',
        # Final assessment
        r'final\s*(?:grade|assessment|evaluation)[,:]?\s*["\']?(correct|almost|partial|incorrect)["\']?',
        r'assessment[,:]?\s*["\']?(correct|almost|partial|incorrect)["\']?',
        # This is X
        r'this\s+is\s+["\']?(correct|almost|partial|incorrect)["\']?',
        r'it\s+is\s+["\']?(correct|almost|partial|incorrect)["\']?',
    ]
    
    for pattern in conclusion_patterns:
        match = re.search(pattern, text_lower)
        if match:
            val = match.group(1).lower()
            if val in {"correct", "almost", "partial", "incorrect"}:
                return val
    
    # Look for grade mentioned in context of being assigned
    grade_context_patterns = [
        r'(?:deserves|warrants|merits|earns)\s+(?:a|an)?\s*["\']?(correct|almost|partial|incorrect)["\']?',
        r'["\']?(correct|almost|partial|incorrect)["\']?\s+(?:is\s+)?(?:appropriate|suitable|deserved|warranted)',
    ]
    
    for pattern in grade_context_patterns:
        match = re.search(pattern, text_lower)
        if match:
            val = match.group(1).lower()
            if val in {"correct", "almost", "partial", "incorrect"}:
                return val
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem."""
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Analyze guidelines for specific criteria
        guidelines_lower = grading_guidelines.lower()
        has_almost_section = "(almost)" in guidelines_lower
        has_partial_section = "(partial)" in guidelines_lower or "partial credit" in guidelines_lower
        
        # Extract key criteria from guidelines to emphasize in the prompt
        criteria_text = ""
        if has_almost_section:
            # Try to extract the (Almost) section
            almost_match = re.search(r'\(almost\)[^\(]*', grading_guidelines, re.IGNORECASE)
            if almost_match:
                criteria_text += f"\n\n(Almost) criteria from guidelines: {almost_match.group(0)}"
        if has_partial_section:
            # Try to extract the (Partial) section
            partial_match = re.search(r'\(partial\)[^\(]*', grading_guidelines, re.IGNORECASE)
            if partial_match:
                criteria_text += f"\n\n(Partial) criteria from guidelines: {partial_match.group(0)}"
        
        # Build the instruction with clearer structure and stronger differentiation
        instruction = f"""You are an expert mathematical grader for competition mathematics (IMO, USAMO, Putnam, etc.). Your task is to evaluate the student's solution and assign exactly one of four possible grades.

PROBLEM DOMAIN: {domain}

PROBLEM STATEMENT:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}{criteria_text}

STUDENT'S ANSWER:
{student_answer}

=== GRADING RUBRIC ===

You must assign exactly ONE of these four grades. Be decisive - avoid defaulting to safe choices.

1. "correct" - Complete and correct proof (7/7 marks)
   - All steps mathematically sound and justified
   - No gaps, no missing cases, no logical errors
   - Proper handling of all edge cases
   - Clear, valid reasoning throughout
   - Use ONLY when fully confident the solution is 100% correct

2. "almost" - Nearly perfect with minor issues (5-6/7 marks)
   - Core proof structure is complete and correct
   - Issues are truly minor: small calculation errors, typos, minor notation issues
   - The main argument is valid and would work with trivial fixes
   - NOT for solutions with significant gaps or major errors
   - Compare against official solution - if the approach matches but has tiny flaws, use this
   {"- Guidelines specify (Almost) criteria - check carefully" if has_almost_section else ""}

3. "partial" - Meaningful progress with significant gaps (1-4/7 marks)
   - Has valid mathematical reasoning and correct elements
   - Shows genuine understanding of the problem
   - BUT missing key steps, has major errors, or incomplete proof structure
   - Good initial setup but fails to complete the argument
   - Correct approach but critical execution flaws
   - "Structured partial progress" - organized attempt with some correct parts
   {"- Guidelines specify (Partial) criteria - check carefully" if has_partial_section else ""}

4. "incorrect" - Little to no meaningful progress (0/7 marks)
   - Approach is fundamentally wrong or completely misguided
   - No significant correct mathematical progress
   - Answer is irrelevant, nonsense, or shows zero understanding
   - Major conceptual errors throughout
   - Only use when there's truly nothing redeeming about the attempt

=== DECISION FRAMEWORK ===
Follow this STRICT decision tree. Do NOT skip steps.

STEP 1 - Check for "incorrect":
- Does the solution contain NO valid, relevant mathematics? → "incorrect"
- Is the approach fundamentally wrong with no redeeming value? → "incorrect"
- Is it just nonsense, irrelevant, or shows zero understanding? → "incorrect"
- If ANY of the above are true, you MUST output "incorrect". Stop here.

STEP 2 - Check for "partial" vs "almost/correct":
- Does the solution have significant gaps in the proof structure? → "partial"
- Are there major errors that prevent the proof from working? → "partial"
- Is the solution incomplete despite having some correct elements? → "partial"
- Does it show understanding but fail to complete the argument? → "partial"
- If ANY of the above are true and STEP 1 was false, output "partial". Stop here.

STEP 3 - Check for "almost" vs "correct":
- Are there only minor issues (typos, small calculation errors, notation issues)? → "almost"
- Would the proof work with trivial fixes? → "almost"
- Is the core structure complete but with tiny flaws? → "almost"
- If ANY of the above are true and STEPS 1-2 were false, output "almost". Stop here.

STEP 4 - Only if STEPS 1-3 all fail:
- Is the proof 100% complete with all steps justified? → "correct"
- Are there absolutely no gaps or errors? → "correct"

Key questions:
1. Does the solution contain any valid, relevant mathematics? If NO → "incorrect"
2. Is the proof complete with all critical steps present? If NO → "partial"
3. Are the errors trivial (typos, minor calc) or major (missing key steps)?
   - Trivial errors → "almost"
   - Major gaps/errors → "partial"
4. Is there a reasonable attempt with some correct parts? If YES → at least "partial"

=== CRITICAL INSTRUCTIONS ===
- You MUST be STRICT and CONSERVATIVE in grading
- "correct" should be rare - only for truly perfect solutions
- "almost" requires the proof to be ESSENTIALLY COMPLETE with only trivial issues
- "partial" is for solutions with SIGNIFICANT GAPS or MAJOR ERRORS - be willing to use this
- "incorrect" is for solutions with NO redeeming mathematical value
- When in doubt, choose the LOWER grade (be conservative)
- A solution with missing critical steps is "partial", NOT "almost"
- A solution with major logical errors is "partial" or "incorrect", NOT "almost"

=== COMMON MISTAKES TO AVOID ===
- Do NOT give "partial" just because the student wrote something mathematical
- "partial" requires MEANINGFUL progress toward the solution, not just random math
- If the approach is completely wrong, it's "incorrect" even if calculations are correct
- If the student misunderstands the problem statement, it's "incorrect"
- If there's no logical connection to the actual problem, it's "incorrect"

=== OUTPUT FORMAT ===
Respond with EXACTLY ONE JSON object in this format:
<json>{{"response": "correct"}}</json>
<json>{{"response": "almost"}}</json>
<json>{{"response": "partial"}}</json>
<json>{{"response": "incorrect"}}</json>

No other text before or after the JSON."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction
        prediction = "None"
        response_text = msg_history[-1]["text"]
        
        # Try JSON tag extraction
        try:
            extracted = _extract_jsons(response_text)
            if extracted:
                for item in extracted:
                    if isinstance(item, dict) and "response" in item:
                        val = item["response"]
                        if val in {"correct", "almost", "partial", "incorrect"}:
                            prediction = val
                            break
        except Exception:
            pass
        
        # Try raw text extraction
        if prediction not in {"correct", "almost", "partial", "incorrect"}:
            try:
                raw_extracted = _extract_json_from_raw_text(response_text)
                if raw_extracted and "response" in raw_extracted:
                    val = raw_extracted["response"]
                    if val in {"correct", "almost", "partial", "incorrect"}:
                        prediction = val
            except Exception:
                pass
        
        # Try to extract from reasoning text
        if prediction not in {"correct", "almost", "partial", "incorrect"}:
            try:
                reasoning_label = _extract_label_from_reasoning(response_text)
                if reasoning_label:
                    prediction = reasoning_label
            except Exception:
                pass
        
        # Text-based fallback with priority order
        if prediction not in {"correct", "almost", "partial", "incorrect"}:
            text = response_text.lower()
            
            # Priority 1: Look for explicit JSON response field patterns
            json_patterns = [
                (r'["\']response["\']?\s*:\s*["\']almost["\']', "almost"),
                (r'["\']response["\']?\s*:\s*["\']partial["\']', "partial"),
                (r'["\']response["\']?\s*:\s*["\']incorrect["\']', "incorrect"),
                (r'["\']response["\']?\s*:\s*["\']correct["\']', "correct"),
            ]
            for pattern, label in json_patterns:
                if re.search(pattern, text):
                    prediction = label
                    break
            
            # Priority 2: Look for standalone quoted values (be careful with context)
            if prediction not in {"correct", "almost", "partial", "incorrect"}:
                # Check for grade in quotes with word boundaries
                quoted_patterns = [
                    (r'["\']almost["\']', "almost"),
                    (r'["\']partial["\']', "partial"),
                    (r'["\']incorrect["\']', "incorrect"),
                    (r'["\']correct["\']', "correct"),
                ]
                for pattern, label in quoted_patterns:
                    if re.search(pattern, text):
                        prediction = label
                        break
            
            # Priority 3: Look for grade as standalone word with context
            if prediction not in {"correct", "almost", "partial", "incorrect"}:
                # Use word boundaries to avoid partial matches
                word_patterns = [
                    (r'\balmost\b', "almost"),
                    (r'\bpartial\b', "partial"),
                    (r'\bincorrect\b', "incorrect"),
                    (r'\bcorrect\b', "correct"),
                ]
                for pattern, label in word_patterns:
                    if re.search(pattern, text):
                        prediction = label
                        break

        # Default to incorrect if invalid - be conservative
        if prediction not in {"correct", "almost", "partial", "incorrect"}:
            prediction = "incorrect"
        
        # Final verification: analyze the model's reasoning to validate the grade
        response_lower = response_text.lower()
        
        # Check for indicators that the solution has NO redeeming value (should be incorrect)
        no_value_indicators = [
            "no valid mathematics", "no meaningful progress", "no understanding",
            "completely wrong", "nonsense", "irrelevant", "nothing correct",
            "zero understanding", "fundamentally flawed", "fundamentally wrong",
            "no correct steps", "no valid steps", "does not address", "misunderstands",
            "wrong approach", "incorrect approach", "invalid approach"
        ]
        
        has_no_value = any(indicator in response_lower for indicator in no_value_indicators)
        
        # If model predicted partial/almost/correct but reasoning shows no value, downgrade to incorrect
        if has_no_value and prediction in {"partial", "almost", "correct"}:
            prediction = "incorrect"
        
        # Check for indicators of major issues that should prevent "correct" or "almost"
        major_issue_indicators = [
            "gap", "missing", "incomplete", "not proven", "not complete",
            "major error", "significant error", "critical flaw", "does not work",
            "fails to", "unable to", "does not prove", "missing proof",
            "logical error", "flawed reasoning", "incorrect logic"
        ]
        
        has_major_issues = any(indicator in response_lower for indicator in major_issue_indicators)
        
        # If predicted "correct" or "almost" but major issues exist, downgrade
        if has_major_issues and prediction in {"correct", "almost"}:
            prediction = "partial"
        
        # Check for indicators that solution is actually correct (to prevent over-downgrading)
        correct_indicators = [
            "complete proof", "correct proof", "valid proof", "sound reasoning",
            "all steps correct", "correctly proves", "successfully proves"
        ]
        has_correct_indicators = any(indicator in response_lower for indicator in correct_indicators)
        
        # If model said correct and we have evidence, trust it
        if prediction == "correct" and has_correct_indicators:
            pass  # Keep as correct
        elif prediction == "correct" and has_major_issues:
            prediction = "partial"  # Downgrade if issues found
        
        # Additional check: if predicted "partial" but reasoning shows it's really incorrect
        # Look for strong language about wrong approach combined with partial prediction
        if prediction == "partial":
            strong_wrong_indicators = [
                "approach is wrong", "approach is incorrect", "wrong strategy",
                "does not understand", "misunderstood", "irrelevant to the problem"
            ]
            has_strong_wrong = any(indicator in response_lower for indicator in strong_wrong_indicators)
            if has_strong_wrong and has_no_value:
                prediction = "incorrect"

        return str(prediction), msg_history
