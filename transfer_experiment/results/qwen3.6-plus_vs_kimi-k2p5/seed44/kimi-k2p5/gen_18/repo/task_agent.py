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

__version__ = "2.6.0"

# Valid grade labels for normalization
_VALID_GRADES = {"correct", "almost", "partial", "incorrect"}

# Grade check order - check "incorrect" before "correct" to avoid substring match
_GRADE_CHECK_ORDER = ["incorrect", "almost", "partial", "correct"]

# Synonym mappings for common LLM output variations
_GRADE_SYNONYMS = {
    "mostly correct": "almost",
    "nearly correct": "almost",
    "almost correct": "almost",
    "partially correct": "partial",
    "partly correct": "partial",
    "somewhat correct": "partial",
    "some understanding": "partial",
    "partial credit": "partial",
    "completely correct": "correct",
    "fully correct": "correct",
    "totally correct": "correct",
    "right": "correct",
    "true": "correct",
    "valid": "correct",
    "full marks": "correct",
    "completely wrong": "incorrect",
    "totally wrong": "incorrect",
    "entirely wrong": "incorrect",
    "wrong": "incorrect",
    "false": "incorrect",
    "not correct": "incorrect",
}


def _normalize_prediction(raw: str) -> str:
    """Normalize a raw prediction string to one of the valid grade labels.
    
    Uses word boundary matching to avoid false substring matches
    (e.g., "incorrect" should NOT match "correct").
    
    Args:
        raw: Raw prediction string
        
    Returns:
        Capitalized grade label or "None" if not recognized
    """
    if not raw or not isinstance(raw, str):
        return "None"
    
    cleaned = raw.strip().strip('"').strip("'").lower()
    
    # Strip common prefixes like "grade:", "prediction:", etc.
    cleaned = re.sub(r'^(grade|prediction|result|answer|verdict|evaluation|response|label)\s*[:=]\s*', '', cleaned).strip()
    
    # Remove trailing punctuation
    cleaned = re.sub(r'[.,;:!?]+$', '', cleaned).strip()
    
    # Direct match first (exact match after cleaning)
    if cleaned in _VALID_GRADES:
        return cleaned.capitalize()
    
    # Check for multi-word synonym phrases first (longer phrases first to avoid partial matches)
    sorted_synonyms = sorted(_GRADE_SYNONYMS.items(), key=lambda x: -len(x[0]))
    for phrase, grade in sorted_synonyms:
        if phrase in cleaned:
            return grade.capitalize()
    
    # Use word-boundary regex with ordered check to avoid false substring matches
    # Check "incorrect" before "correct" to avoid "incorrect" matching "correct"
    for grade in _GRADE_CHECK_ORDER:
        if re.search(r'\b' + re.escape(grade) + r'\b', cleaned):
            return grade.capitalize()
    
    # Additional check: look for grade words at the start or end of string
    for grade in _GRADE_CHECK_ORDER:
        if cleaned.startswith(grade + ' ') or cleaned.endswith(' ' + grade) or cleaned == grade:
            return grade.capitalize()
    
    return "None"


def _extract_grade_from_reasoning(text: str) -> str | None:
    """Extract grade from reasoning text when JSON extraction fails.
    
    Looks for explicit grade statements in the reasoning/analysis section.
    Uses word boundaries to avoid false matches.
    """
    if not text or not isinstance(text, str):
        return None
        
    text_lower = text.lower()
    
    # Look for explicit grade statements in reasoning with word boundaries
    # More specific patterns first to avoid false matches
    # Check "incorrect" before "correct" in patterns to avoid substring issues
    patterns = [
        r'grade\s*[:=]\s*["\']?(incorrect|almost|partial|correct)["\']?',
        r'classification\s*[:=]\s*["\']?(incorrect|almost|partial|correct)["\']?',
        r'evaluation\s*[:=]\s*["\']?(incorrect|almost|partial|correct)["\']?',
        r'verdict\s*[:=]\s*["\']?(incorrect|almost|partial|correct)["\']?',
        r'prediction\s*[:=]\s*["\']?(incorrect|almost|partial|correct)["\']?',
        r'result\s*[:=]\s*["\']?(incorrect|almost|partial|correct)["\']?',
        r'\bthis answer is\s+["\']?(incorrect|almost|partial|correct)["\']?',
        r'\bthe answer is\s+["\']?(incorrect|almost|partial|correct)["\']?',
        r'\banswer is\s+["\']?(incorrect|almost|partial|correct)["\']?',
        r'should be graded as\s+["\']?(incorrect|almost|partial|correct)["\']?',
        r'qualifies as\s+["\']?(incorrect|almost|partial|correct)["\']?',
        r'falls under\s+["\']?(incorrect|almost|partial|correct)["\']?',
        r'belongs to\s+["\']?(incorrect|almost|partial|correct)["\']?',
        r'\bfinal grade\s+["\']?(incorrect|almost|partial|correct)["\']?',
        r'\bi would grade\s+["\']?(incorrect|almost|partial|correct)["\']?',
        r'\bthe grade is\s+["\']?(incorrect|almost|partial|correct)["\']?',
        r'\bassigned grade\s+["\']?(incorrect|almost|partial|correct)["\']?',
        r'\bgrade:\s*["\']?(incorrect|almost|partial|correct)["\']?',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            grade = match.group(1).lower()
            if grade in _VALID_GRADES:
                return grade.capitalize()
    
    # Last resort: look for standalone grade words with word boundaries
    # Check in order: incorrect, almost, partial, correct (to avoid "incorrect" matching "correct")
    for grade in _GRADE_CHECK_ORDER:
        # Use word boundary regex
        pattern = r'\b' + re.escape(grade) + r'\b'
        if re.search(pattern, text_lower):
            return grade.capitalize()
    
    return None


def _repair_json(json_str: str) -> str:
    """Repair common JSON formatting issues in LLM outputs.
    
    Args:
        json_str: Raw JSON string that may have formatting issues
        
    Returns:
        Repaired JSON string
    """
    cleaned = json_str.strip()
    
    # Remove trailing commas before } or ]
    cleaned = re.sub(r',\s*}', '}', cleaned)
    cleaned = re.sub(r',\s*]', ']', cleaned)
    
    # Fix single quotes to double quotes (but be careful with apostrophes in text)
    # Only replace quotes that appear to be JSON delimiters
    cleaned = re.sub(r"(?<=[{\[,\s])'([^']+)'(?=\s*[:}\],])", r'"\1"', cleaned)
    
    # Fix common JSON issues: unquoted keys
    cleaned = re.sub(r'(\{|,\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', cleaned)
    
    # Fix double curly braces from f-string escaping
    cleaned = cleaned.replace('{{', '{').replace('}}', '}')
    
    # Remove control characters
    cleaned = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', cleaned)
    
    # Fix escaped newlines that might break parsing
    cleaned = cleaned.replace('\\n', ' ').replace('\\t', ' ')
    
    return cleaned


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    """
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
        
        # Skip empty blocks
        if not inner:
            continue
            
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # Try to clean up common issues
            try:
                cleaned = _repair_json(inner)
                results.append(json.loads(cleaned))
            except json.JSONDecodeError:
                # Try extracting just the first valid JSON object
                try:
                    # Find the first { and matching }
                    brace_start = inner.find('{')
                    if brace_start != -1:
                        brace_count = 1
                        brace_end = brace_start + 1
                        while brace_end < len(inner) and brace_count > 0:
                            if inner[brace_end] == '{':
                                brace_count += 1
                            elif inner[brace_end] == '}':
                                brace_count -= 1
                            brace_end += 1
                        if brace_count == 0:
                            json_str = inner[brace_start:brace_end]
                            cleaned = _repair_json(json_str)
                            results.append(json.loads(cleaned))
                except (json.JSONDecodeError, ValueError):
                    continue
    return results or None


def _extract_json_from_markdown(text: str) -> list[dict] | None:
    """Extract JSON objects from markdown code blocks as a fallback."""
    results = []
    pattern = r'```(?:json)?\s*(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        match = match.strip()
        if not match:
            continue
        try:
            results.append(json.loads(match))
        except json.JSONDecodeError:
            try:
                cleaned = _repair_json(match)
                results.append(json.loads(cleaned))
            except json.JSONDecodeError:
                continue
    return results or None


def _extract_json_braces(text: str) -> list[dict] | None:
    """Extract JSON objects by finding matching braces as a last resort."""
    results = []
    i = 0
    while i < len(text):
        if text[i] == '{':
            count = 1
            j = i + 1
            while j < len(text) and count > 0:
                if text[j] == '{':
                    count += 1
                elif text[j] == '}':
                    count -= 1
                j += 1
            if count == 0:
                json_str = text[i:j]
                try:
                    results.append(json.loads(json_str))
                except json.JSONDecodeError:
                    try:
                        cleaned = _repair_json(json_str)
                        results.append(json.loads(cleaned))
                    except json.JSONDecodeError:
                        pass
            i = j
        else:
            i += 1
    return results or None


def _validate_grade_consistency(prediction: str, reasoning: str) -> tuple[str, bool]:
    """Validate that the grade is consistent with the reasoning text.
    
    Args:
        prediction: The predicted grade
        reasoning: The reasoning text from the LLM
        
    Returns:
        Tuple of (validated_grade, is_consistent)
    """
    if prediction == "None" or not reasoning:
        return prediction, True
    
    reasoning_lower = reasoning.lower()
    pred_lower = prediction.lower()
    
    # Check for contradictory language in reasoning
    contradictions = {
        "correct": ["wrong", "incorrect", "not right", "not correct", "error", "mistake", "incomplete", "missing"],
        "almost": ["completely wrong", "totally incorrect", "no understanding", "fundamentally flawed", "missing critical", "incomplete proof"],
        "partial": ["complete solution", "fully correct", "perfect", "entirely correct", "fully valid"],
        "incorrect": ["correct", "right answer", "perfect solution", "valid proof", "complete proof"]
    }
    
    # Check if reasoning contradicts the prediction
    contradiction_words = contradictions.get(pred_lower, [])
    for word in contradiction_words:
        if word in reasoning_lower:
            # Look for negation that might negate the contradiction
            negation_pattern = r'\b(not|no|never|hardly|barely|isn\'t|aren\'t|wasn\'t|weren\'t)\s+\w*\s*' + re.escape(word)
            if not re.search(negation_pattern, reasoning_lower):
                logger.warning(f"Potential inconsistency: grade '{prediction}' but reasoning contains '{word}'")
                return prediction, False
    
    return prediction, True


def _post_process_validation(prediction: str, student_answer: str, official_solution: str) -> str:
    """Post-process validation to catch common grading errors.
    
    Args:
        prediction: The predicted grade
        student_answer: The student's answer text
        official_solution: The official solution text
        
    Returns:
        Potentially adjusted grade
    """
    if prediction == "None":
        return prediction
    
    pred_lower = prediction.lower()
    answer_lower = student_answer.lower() if student_answer else ""
    
    # Check for incomplete proofs that might be over-graded
    incomplete_markers = [
        "we will prove", "we need to prove", "it remains to show",
        "we claim", "note that", "observe that", "consider"
    ]
    
    # Count how many incomplete markers exist without completion
    incomplete_count = sum(1 for marker in incomplete_markers if marker in answer_lower)
    
    # Check if answer seems to stop abruptly
    has_abrupt_ending = False
    if answer_lower:
        # Check if ends with "..." or incomplete sentence
        if answer_lower.rstrip().endswith(("...", ".", "we have", "thus", "therefore")):
            # Check length ratio - if much shorter than solution, might be incomplete
            if len(student_answer) < len(official_solution) * 0.5:
                has_abrupt_ending = True
    
    # Over-grading checks
    if pred_lower in ["correct", "almost"]:
        # If marked as correct/almost but seems incomplete
        if incomplete_count >= 3 and has_abrupt_ending:
            logger.warning(f"Potential over-grading: {prediction} but answer appears incomplete")
            # Downgrade: Correct -> Partial, Almost -> Partial
            if pred_lower == "correct":
                return "Partial"
            elif pred_lower == "almost":
                return "Partial"
    
    # Check for "Partial" that might be under-graded (should be "Incorrect")
    if pred_lower == "partial":
        # Look for signs of fundamentally wrong approach
        wrong_approach_markers = [
            "contradiction" in answer_lower and "assume" in answer_lower,
        ]
        # If answer is very short and doesn't demonstrate key insights
        if len(student_answer) < 200 and incomplete_count < 2:
            # Might be incorrect if no real progress shown
            pass  # Keep as Partial for now
    
    return prediction


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
        # Build a structured prompt with clearly labeled fields
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematical grader for International Mathematical Olympiad (IMO) problems.

Your task is to evaluate a student's answer and assign exactly one of four grades: "Correct", "Almost", "Partial", or "Incorrect".

## Problem Statement:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Grading Rubric - READ CAREFULLY:

1. "Correct" - The answer is fully correct, complete, and matches the official solution. All key steps and reasoning are present and valid. The solution would receive full marks.
   - MUST have: complete logical structure, all key steps, correct conclusion
   - NO significant gaps, missing steps, or logical errors allowed

2. "Almost" - The answer demonstrates COMPLETE understanding with only MINOR errors:
   - Small arithmetic/calculation errors in an otherwise correct derivation
   - Typos in notation or variable names
   - Missing trivial final simplification steps
   - The student clearly knows how to solve the problem but made careless mistakes
   - Key insight: If you fixed these minor errors, the solution would be "Correct"
   - The core approach and reasoning are sound throughout
   - IMPORTANT: The solution must be ESSENTIALLY COMPLETE - missing critical steps disqualifies "Almost"

3. "Partial" - The answer shows SOME understanding but has SIGNIFICANT issues:
   - Missing critical proof steps or logical gaps
   - Incomplete solution (stopped halfway or missed key cases)
   - Significant errors in reasoning that affect the conclusion
   - Correct approach but major calculation errors that invalidate the result
   - Key insight: The student is on the right track but needs substantial work

4. "Incorrect" - The answer is fundamentally wrong:
   - Wrong approach or method entirely
   - Fundamental misunderstanding of the problem
   - No meaningful progress toward the solution
   - Claims that are not substantiated or justified

## CRITICAL DISTINCTIONS - PAY EXTREMELY CLOSE ATTENTION:

**"Correct" vs "Almost" - BE STRICT:**
- "Correct" means PERFECT - no errors, no gaps, no missing steps
- "Almost" means the solution is complete but has tiny, fixable flaws
- If there are ANY missing critical steps or logical gaps, it is NOT "Correct"
- When in doubt, grade "Almost" or lower - do not give "Correct" generously

**"Almost" vs "Partial" - THIS IS THE MOST COMMON ERROR:**
- "Almost" = The solution is essentially complete with only tiny, fixable flaws
  * The student demonstrates full mastery of the solution method
  * Errors are superficial (typos, arithmetic slips, minor notation issues)
  * The proof structure is complete from start to finish
  * If you fixed the errors, you'd have a "Correct" answer
  * Example: Correct proof with one small calculation error
  * COUNTER-EXAMPLE: Missing a key lemma or proof step → NOT "Almost", this is "Partial"

- "Partial" = Significant portions are missing or wrong
  * The student shows some understanding but has major gaps
  * Missing critical steps, incomplete reasoning, or major errors
  * Would need substantial work to become correct
  * Example: Correct approach but stopped halfway, or major logical flaw
  * Example: Makes some valid observations but doesn't complete the proof

**"Partial" vs "Incorrect" - BE CAREFUL:**
- "Partial" = Genuine understanding of some key concepts (on the right track)
  * Student demonstrates at least one key insight or valid approach
  * Has made meaningful progress toward the solution
  * Even if incomplete, the direction is correct

- "Incorrect" = Little to no understanding (wrong approach or no progress)
  * Wrong method or approach entirely
  * No valid key insights demonstrated
  * Claims without justification or proof
  * Fundamentally misunderstands the problem

## Decision Process - FOLLOW THIS STRICTLY:
1. Is the answer PERFECT? (complete, all steps valid, no gaps) → "Correct"
2. If not perfect, is it essentially complete with only minor errors? → "Almost"
3. If not "Almost", does the student show genuine understanding with valid key insights? → "Partial"
4. If none of the above → "Incorrect"

## RED FLAGS - These indicate LOWER grades:
- Missing proof steps that are in the official solution
- Claims without justification
- "We will prove..." but then not completing the proof
- Major logical gaps in reasoning
- Wrong approach to the problem

## GREEN FLAGS - These indicate HIGHER grades:
- Complete logical flow from assumptions to conclusion
- All key steps from official solution present
- Valid mathematical reasoning throughout
- Correct final answer with proper justification

## IMPORTANT REMINDERS:
- "Correct" is for PERFECT answers only - be strict!
- "Almost" is for NEARLY CORRECT answers with minor flaws - the student clearly knows the solution
- "Partial" is for answers with SIGNIFICANT gaps or errors - the student is on the right track but missing key pieces
- "Incorrect" is for answers with little to no valid understanding
- When in doubt between two grades, choose the LOWER one (be conservative)
- Pay special attention to the grading guidelines provided above - they contain specific criteria for this problem

## Output Format - CRITICAL - FOLLOW EXACTLY:
You MUST output ONLY a JSON object wrapped in <json> tags. Do not include any other text, explanations, or markdown formatting outside the JSON tags.

Your response MUST follow this EXACT format (replace X with your grade):
<json>
{{"grade": "X"}}
</json>

Valid values for "grade" are exactly: "Correct", "Almost", "Partial", or "Incorrect" (case-sensitive, with first letter capitalized).

Example 1 - For a correct answer:
<json>
{{"grade": "Correct"}}
</json>

Example 2 - For an almost correct answer:
<json>
{{"grade": "Almost"}}
</json>

Example 3 - For a partial answer:
<json>
{{"grade": "Partial"}}
</json>

Example 4 - For an incorrect answer:
<json>
{{"grade": "Incorrect"}}
</json>

## FINAL INSTRUCTION - READ CAREFULLY:
After analyzing the student's answer against the official solution and grading guidelines, you MUST output ONLY the JSON block shown above. No other text is allowed.

Think step by step about the grade, but your final output must be ONLY the JSON object with the grade field."""

        # First attempt with full prompt
        prediction, msg_history = self._get_prediction_with_prompt(
            instruction, student_answer, solution
        )
        
        # If first attempt failed, try with a simpler prompt
        if prediction == "None":
            self.log_fn("First attempt failed, trying with simpler prompt...")
            simple_instruction = f"""Grade this math answer as one of: Correct, Almost, Partial, or Incorrect.

Problem: {problem[:500]}...

Official Solution: {solution[:500]}...

Student Answer: {student_answer[:1000]}...

Output ONLY this exact format:
<json>
{{"grade": "X"}}
</json>

Where X is exactly one of: Correct, Almost, Partial, Incorrect"""
            
            prediction2, msg_history2 = self._get_prediction_with_prompt(
                simple_instruction, student_answer, solution
            )
            if prediction2 != "None":
                prediction = prediction2
                msg_history = msg_history2

        return str(prediction), msg_history
    
    def _get_prediction_with_prompt(self, instruction: str, student_answer: str, solution: str) -> tuple[str, list[dict]]:
        """Get prediction using a specific prompt."""
        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON using multiple methods
        prediction = "None"
        try:
            if msg_history and len(msg_history) > 0:
                text = msg_history[-1].get("text", "")
                
                if not text:
                    self.log_fn("Empty response from LLM")
                    return "None", msg_history

                self.log_fn(f"Raw LLM response (first 500 chars): {text[:500]}...")

                # Try multiple extraction methods in order of reliability
                extracted = None
                extraction_method = None
                
                # Method 1: Extract from <json> tags
                extracted = _extract_jsons(text)
                if extracted:
                    extraction_method = "<json> tags"
                
                # Method 2: Extract from markdown code blocks
                if not extracted:
                    extracted = _extract_json_from_markdown(text)
                    if extracted:
                        extraction_method = "markdown code blocks"
                    
                # Method 3: Extract by brace matching
                if not extracted:
                    extracted = _extract_json_braces(text)
                    if extracted:
                        extraction_method = "brace matching"

                if extracted:
                    last_extract = extracted[-1]
                    self.log_fn(f"Extracted JSON using {extraction_method}: {last_extract}")

                    # Try multiple field names in order of preference
                    # Prioritize "grade" since that's what the prompt asks for
                    pred_value = None
                    field_priority = ["grade", "prediction", "response", "evaluation", "result", "answer", "verdict"]
                    for field in field_priority:
                        if field in last_extract:
                            pred_value = last_extract[field]
                            self.log_fn(f"Found grade in field '{field}': {pred_value}")
                            break

                    if pred_value is not None:
                        prediction = _normalize_prediction(str(pred_value))
                        self.log_fn(f"Normalized prediction: {prediction}")
                    else:
                        # Try to find any value that looks like a valid grade
                        for key, value in last_extract.items():
                            if isinstance(value, str):
                                normalized = _normalize_prediction(value)
                                if normalized != "None":
                                    prediction = normalized
                                    self.log_fn(f"Found valid grade in field '{key}': {prediction}")
                                    break
                        if prediction == "None":
                            self.log_fn(f"No valid grade found in JSON fields: {list(last_extract.keys())}")
                else:
                    # No JSON found, try to extract grade from reasoning text first
                    self.log_fn("No JSON found, trying to extract from reasoning text")
                    grade_from_reasoning = _extract_grade_from_reasoning(text)
                    if grade_from_reasoning:
                        prediction = grade_from_reasoning
                        self.log_fn(f"Extracted grade from reasoning: {prediction}")
                    else:
                        # Fall back to normalizing raw text
                        prediction = _normalize_prediction(text)
                        self.log_fn(f"No JSON found, normalized from text: {prediction}")
                    
                    # Last resort: look for capitalized grade words in the last line
                    if prediction == "None":
                        lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
                        if lines:
                            last_line = lines[-1]
                            for grade in ["Correct", "Almost", "Partial", "Incorrect"]:
                                if grade in last_line:
                                    prediction = grade
                                    self.log_fn(f"Found grade in last line: {prediction}")
                                    break
                
                # Apply post-processing validation to catch over-grading/under-grading
                original_prediction = prediction
                prediction = _post_process_validation(prediction, student_answer, solution)
                if prediction != original_prediction:
                    self.log_fn(f"Post-processing adjusted grade: {original_prediction} -> {prediction}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            import traceback
            self.log_fn(f"Traceback: {traceback.format_exc()}")

        return prediction, msg_history
