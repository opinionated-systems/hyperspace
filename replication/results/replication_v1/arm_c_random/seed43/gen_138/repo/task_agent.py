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
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # Try to find valid JSON within the content
            try:
                # Look for JSON object pattern
                json_start = inner.find('{')
                json_end = inner.rfind('}')
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    results.append(json.loads(inner[json_start:json_end+1]))
            except json.JSONDecodeError:
                continue
    return results or None


def _extract_json_from_markdown(text: str) -> dict | None:
    """Extract JSON from markdown code blocks (```json ... ```)."""
    pattern = r'```json\s*(.*?)\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            # Try to find valid JSON within the content
            try:
                json_start = match.find('{')
                json_end = match.rfind('}')
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    return json.loads(match[json_start:json_end+1].strip())
            except json.JSONDecodeError:
                continue
    return None


def _validate_grade(prediction: str, grading_guidelines: str) -> tuple[str, bool]:
    """Validate that the extracted grade is reasonable.
    
    Enhanced validation with strict support for IMO 0-7 point scale.
    Only accepts single digit grades 0-7 for consistency.
    
    Args:
        prediction: The extracted grade prediction
        grading_guidelines: Optional grading guidelines for context
        
    Returns:
        (validated_grade, is_valid): Tuple of the validated grade and whether it's valid
    """
    if not prediction or prediction == "None":
        return "None", False
    
    pred_clean = prediction.strip()
    pred_lower = pred_clean.lower()
    
    # Fast path: strict validation for single digit 0-7
    if pred_clean in ["0", "1", "2", "3", "4", "5", "6", "7"]:
        return pred_clean, True
    
    # Check for fractional grades like "3/7" or "5 / 7" - extract just the numerator
    fractional_match = re.search(r'\b([0-7])\s*/\s*7\b', pred_clean)
    if fractional_match:
        return fractional_match.group(1), True
    
    # Check for numeric grades embedded in text (0-7 for IMO problems)
    # Use word boundary to avoid matching numbers within other numbers
    numeric_match = re.search(r'(?:^|\s)([0-7])(?:\s|$|[^0-9])', pred_clean)
    if numeric_match:
        return numeric_match.group(1), True
    
    # Check for full credit patterns -> 7
    full_patterns = [
        r'\bfull\s*(?:credit|points?|score|marks?)?\b',
        r'\bcomplete\s*(?:solution|answer|credit|work)?\b',
        r'\ball\s*(?:points?|credit|marks?)?\b',
        r'\bperfect\s*(?:score|solution|answer)?\b',
        r'\bcorrect\s*(?:solution|answer|work)?\b',
        r'\bentirely\s*correct\b',
        r'\bfully\s*correct\b',
    ]
    for pattern in full_patterns:
        if re.search(pattern, pred_lower):
            return "7", True
    
    # Check for zero/no credit patterns -> 0
    zero_patterns = [
        r'\bno\s*(?:credit|points?|score|marks?|progress)?\b',
        r'\bzero\s*(?:credit|points?|score|marks?)?\b',
        r'\b0\s*(?:points?|credit|score|marks?)?\b',
        r'\bincorrect\s*(?:solution|answer|work)?\b',
        r'\bwrong\s*(?:solution|answer|work)?\b',
        r'\bnone\b',
        r'\bblank\b',
        r'\bno\s*answer\b',
        r'\bno\s*solution\b',
        r'\bempty\b',
    ]
    for pattern in zero_patterns:
        if re.search(pattern, pred_lower):
            return "0", True
    
    # Check for partial credit patterns (1-6)
    partial_patterns = [
        (r'\bpartial\s*(?:credit|points?|score|marks?)?\b', None),
        (r'\bsome\s*(?:credit|points?|score|progress|marks?)?\b', None),
        (r'\bincomplete\s*(?:solution|answer|work)?\b', None),
        (r'\bpartially\s*correct\b', None),
        (r'\bmostly\s*correct\b', "6"),
        (r'\bsubstantial\s*(?:progress|work|credit)?\b', "5"),
        (r'\bgood\s*progress\b', "4"),
        (r'\bsome\s*progress\b', "3"),
        (r'\blimited\s*(?:progress|work)?\b', "2"),
        (r'\bminimal\s*(?:progress|work|credit)?\b', "1"),
    ]
    for pattern, default_grade in partial_patterns:
        if re.search(pattern, pred_lower):
            # Try to extract a numeric grade from the text
            numeric_in_text = re.search(r'\b([1-6])\b', pred_clean)
            if numeric_in_text:
                return numeric_in_text.group(1), True
            # If no specific number found but pattern matched with default
            if default_grade:
                return default_grade, True
    
    # If no clear grade found, mark as invalid
    return pred_clean, False


def _calculate_confidence(reasoning: str, grade: str, is_valid: bool) -> dict:
    """Calculate confidence metrics for the grading decision.
    
    Analyzes the reasoning text to determine how confident the model is
    in its grading decision. This helps identify uncertain grades that
    may need human review.
    
    Args:
        reasoning: The model's reasoning text
        grade: The assigned grade (0-7)
        is_valid: Whether the grade passed validation
        
    Returns:
        Dictionary with confidence metrics:
        - score: Overall confidence score (0.0-1.0)
        - level: 'high', 'medium', or 'low'
        - indicators: List of detected confidence indicators
        - needs_review: Whether the grade should be flagged for review
    """
    if not reasoning or not is_valid:
        return {
            "score": 0.0,
            "level": "low",
            "indicators": ["missing_reasoning" if not reasoning else "invalid_grade"],
            "needs_review": True
        }
    
    reasoning_lower = reasoning.lower()
    indicators = []
    
    # High confidence indicators
    high_confidence_patterns = [
        (r'\bclearly\s+(?:correct|wrong|incorrect)\b', 'clearly_definitive'),
        (r'\bdefinitely\s+(?:correct|wrong|incorrect)\b', 'definitely_definitive'),
        (r'\bunambiguous\b', 'unambiguous'),
        (r'\bstraightforward\s+(?:solution|approach)\b', 'straightforward'),
        (r'\bmatches\s+(?:the\s+)?official\s+solution\b', 'matches_official'),
        (r'\bcomplete\s+(?:and\s+)?correct\b', 'complete_correct'),
        (r'\bno\s+(?:errors?|mistakes?)\b', 'no_errors'),
    ]
    
    # Low confidence indicators
    low_confidence_patterns = [
        (r'\bunclear\b', 'unclear'),
        (r'\bambiguous\b', 'ambiguous'),
        (r'\buncertain\b', 'uncertain'),
        (r'\bnot\s+sure\b', 'not_sure'),
        (r'\bdifficult\s+to\s+(?:say|determine|judge)\b', 'difficult_to_judge'),
        (r'\bmight\s+be\b', 'might_be'),
        (r'\bcould\s+be\b', 'could_be'),
        (r'\bpossibly\b', 'possibly'),
        (r'\bperhaps\b', 'perhaps'),
        (r'\bpartially\s+(?:correct|right)\b', 'partially_correct'),
        (r'\bincomplete\s+(?:solution|answer)\b', 'incomplete'),
        (r'\balternative\s+(?:approach|method|solution)\b', 'alternative_approach'),
        (r'\bwithout\s+(?:the\s+)?full\s+(?:solution|work)\b', 'missing_work'),
        (r'\bcannot\s+(?:fully|completely)\s+(?:verify|assess|determine)\b', 'cannot_verify'),
    ]
    
    # Count indicators
    high_count = 0
    for pattern, indicator in high_confidence_patterns:
        if re.search(pattern, reasoning_lower):
            indicators.append(f"+{indicator}")
            high_count += 1
    
    low_count = 0
    for pattern, indicator in low_confidence_patterns:
        if re.search(pattern, reasoning_lower):
            indicators.append(f"-{indicator}")
            low_count += 1
    
    # Calculate base confidence score
    base_score = 0.5
    base_score += high_count * 0.15
    base_score -= low_count * 0.15
    
    # Adjust based on grade extremity (0 and 7 are usually more confident)
    if grade in ["0", "7"]:
        base_score += 0.1
    elif grade in ["3", "4"]:
        # Middle grades often indicate uncertainty
        base_score -= 0.05
    
    # Clamp to [0, 1]
    confidence_score = max(0.0, min(1.0, base_score))
    
    # Determine level
    if confidence_score >= 0.7:
        level = "high"
    elif confidence_score >= 0.4:
        level = "medium"
    else:
        level = "low"
    
    # Flag for review if low confidence or too many uncertainty indicators
    needs_review = level == "low" or low_count >= 3
    
    return {
        "score": round(confidence_score, 2),
        "level": level,
        "indicators": indicators,
        "needs_review": needs_review
    }


def _log_structured(log_fn, event: str, data: dict) -> None:
    """Log structured data as JSON for better observability.
    
    Args:
        log_fn: Logging function (e.g., logger.info)
        event: Event name/type
        data: Dictionary of data to log
    """
    entry = {"event": event, **data}
    log_fn(json.dumps(entry, default=str))


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced reasoning and validation.
    
    This agent uses an LLM to evaluate student answers against official solutions
    for IMO (International Mathematical Olympiad) problems. It supports:
    - Chain-of-thought reasoning
    - Multiple JSON extraction strategies
    - Grade validation (0-7 scale)
    - Fallback extraction from raw text
    - Confidence scoring for quality assurance
    
    Attributes:
        model: The LLM model to use for grading
        log_fn: Logging function for observability
    """

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self._last_confidence: dict | None = None

    def _build_prompt(self, inputs: dict) -> str:
        """Build a structured prompt for IMO grading with chain-of-thought reasoning."""
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        # Build grading guidelines section with better formatting
        guidelines_section = ""
        if grading_guidelines:
            guidelines_section = f"""
## Grading Guidelines
{grading_guidelines}

## IMO Grading Scale Reference
- **7 points**: Complete, correct solution with proper reasoning
- **6 points**: Correct solution with minor gaps or non-essential errors
- **5 points**: Substantial progress with significant gaps or errors
- **4 points**: Good progress but incomplete or contains errors
- **3 points**: Some meaningful progress toward solution
- **2 points**: Limited progress, some relevant ideas
- **1 point**: Minimal progress, few relevant ideas
- **0 points**: No progress, completely wrong, or blank"""
        else:
            guidelines_section = """
## IMO Grading Scale (0-7)
- **7 points**: Complete, correct solution with proper reasoning
- **6 points**: Correct solution with minor gaps or non-essential errors
- **5 points**: Substantial progress with significant gaps or errors
- **4 points**: Good progress but incomplete or contains errors
- **3 points**: Some meaningful progress toward solution
- **2 points**: Limited progress, some relevant ideas
- **1 point**: Minimal progress, few relevant ideas
- **0 points**: No progress, completely wrong, or blank"""

        return f"""You are an expert IMO (International Mathematical Olympiad) grader with deep expertise in mathematical problem evaluation.

Your task is to evaluate a student's solution to a mathematical problem with precision and consistency.

## Problem Domain
{domain}

## Problem Statement
{problem}

## Official Solution
{solution}
{guidelines_section}

## Student's Answer
{student_answer}

## Evaluation Instructions

Follow this systematic evaluation process:

1. **Initial Assessment**: Read the student's answer completely. Is it blank, partially complete, or a full solution?

2. **Solution Comparison**: Compare the student's approach to the official solution:
   - Does the student use the same method as the official solution?
   - If different, is the alternative approach mathematically valid?
   - Are there gaps in the student's reasoning?

3. **Error Analysis**: Identify any errors:
   - Computational errors (arithmetic, algebra)
   - Logical errors (invalid deductions, circular reasoning)
   - Missing steps (unproven claims, skipped justifications)

4. **Progress Assessment**: Evaluate what progress the student HAS made:
   - Did they understand the problem correctly?
   - Did they make any meaningful progress toward the solution?
   - What percentage of the solution is complete?

5. **Grade Assignment**: Based on the IMO 0-7 scale, assign the appropriate grade:
   - Consider both what's correct AND what's missing
   - Be consistent with the grading guidelines
   - When in doubt between two grades, explain your reasoning

## Output Format - CRITICAL

You MUST respond with a valid JSON object wrapped in <json> tags. The JSON must have exactly these two fields:

<json>
{{
    "reasoning": "Your detailed step-by-step analysis here. Include: (1) initial assessment, (2) comparison to official solution, (3) error analysis, (4) progress assessment, and (5) justification for the specific grade assigned.",
    "response": "X"
}}
</json>

## CRITICAL RULES for the "response" field:
- MUST contain ONLY a single digit: 0, 1, 2, 3, 4, 5, 6, or 7
- NO other text, explanations, or formatting allowed
- NO quotes around the number in the JSON value
- Example CORRECT: "response": "5"
- Example INCORRECT: "response": "5 points", "response": "grade: 5"

Valid grades:
- 7 = full credit (complete correct solution)
- 6 = mostly complete with minor issues
- 5 = substantial progress with gaps
- 4 = good progress but significant issues
- 3 = some meaningful progress
- 2 = limited progress
- 1 = minimal progress
- 0 = no credit (blank, completely wrong, or no progress)"""

    def _extract_prediction(self, msg_history: list[dict]) -> tuple[str, str]:
        """Extract prediction and reasoning from message history.
        
        Uses multiple extraction strategies in order of reliability:
        1. <json> tags (primary format)
        2. Markdown code blocks
        3. Loose JSON pattern matching
        4. Text pattern matching (last resort)
        
        Returns:
            (prediction, reasoning)
        """
        prediction = "None"
        reasoning = ""
        
        try:
            # Handle different message formats
            last_msg = ""
            if msg_history:
                last_entry = msg_history[-1]
                if isinstance(last_entry, dict):
                    # Try common keys for message content
                    last_msg = last_entry.get("text") or last_entry.get("content", "")
                    if not last_msg and "message" in last_entry:
                        msg_obj = last_entry["message"]
                        if isinstance(msg_obj, dict):
                            last_msg = msg_obj.get("content", "")
            
            if not last_msg:
                return prediction, reasoning
            
            # Strategy 1: Try <json> tags first (primary format)
            extracted = _extract_jsons(last_msg)
            if extracted:
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = str(last_json["response"]).strip()
                if "reasoning" in last_json:
                    reasoning = str(last_json["reasoning"])
                # Validate extracted grade immediately
                if prediction in ["0", "1", "2", "3", "4", "5", "6", "7"]:
                    return prediction, reasoning
            
            # Strategy 2: Try markdown code blocks
            md_json = _extract_json_from_markdown(last_msg)
            if md_json:
                if "response" in md_json:
                    prediction = str(md_json["response"]).strip()
                if "reasoning" in md_json:
                    reasoning = str(md_json["reasoning"])
                if prediction in ["0", "1", "2", "3", "4", "5", "6", "7"]:
                    return prediction, reasoning
            
            # Strategy 3: Fallback - try to find any JSON-like structure with response field
            json_match = re.search(r'\{[^}]*"response"[^}]*\}', last_msg)
            if json_match:
                try:
                    fallback = json.loads(json_match.group())
                    pred = str(fallback.get("response", "None")).strip()
                    if pred in ["0", "1", "2", "3", "4", "5", "6", "7"]:
                        prediction = pred
                        if "reasoning" in fallback:
                            reasoning = str(fallback["reasoning"])
                        return prediction, reasoning
                except json.JSONDecodeError:
                    pass
            
            # Strategy 4: Last resort - look for grade patterns in text
            if prediction == "None":
                # Look for "response": "X" pattern (even outside JSON)
                response_match = re.search(r'"response"\s*:\s*"([0-7])"', last_msg)
                if response_match:
                    prediction = response_match.group(1)
                    return prediction, reasoning
                
                # Look for "Grade: X" or "Final grade: X" patterns
                grade_match = re.search(r'(?:grade|score|mark|final grade|final score)\s*:?\s*([0-7])\b', last_msg, re.IGNORECASE)
                if grade_match:
                    prediction = grade_match.group(1)
                    return prediction, reasoning
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
        
        return prediction, reasoning

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with enhanced reasoning and validation.

        This method orchestrates the grading process by:
        1. Building a structured prompt from the input data
        2. Calling the LLM to generate a grading decision
        3. Extracting and validating the grade from the response
        4. Calculating confidence metrics for quality assurance
        5. Applying fallback extraction if needed

        Args:
            inputs: dict containing:
                - domain: Subject area (e.g., "Mathematics")
                - problem: The problem statement
                - solution: Official/reference solution
                - grading_guidelines: Rubric for grading
                - student_answer: The student's submitted answer

        Returns:
            tuple of (prediction, msg_history) where:
                - prediction: The validated grade (0-7) or "None" if extraction failed
                - msg_history: Full conversation history with the LLM

        Raises:
            No exceptions are raised; all errors are caught and logged.
            Returns "None" on any failure.
        """
        # Validate required inputs
        required_fields = ["problem", "student_answer"]
        missing = [f for f in required_fields if not inputs.get(f)]
        if missing:
            self.log_fn(f"Missing required inputs: {missing}")
            return "None", []

        instruction = self._build_prompt(inputs)
        grading_guidelines = inputs.get("grading_guidelines", "")

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"LLM call failed: {e}")
            _log_structured(self.log_fn, "llm_error", {
                "error_type": type(e).__name__,
                "error_message": str(e),
            })
            return "None", []

        # Extract prediction with enhanced extraction
        prediction, reasoning = self._extract_prediction(msg_history)
        
        # Validate the grade
        validated_grade, is_valid = _validate_grade(prediction, grading_guidelines)
        
        # Log the reasoning and validation result
        if reasoning:
            self.log_fn(f"Reasoning: {reasoning[:200]}...")
        self.log_fn(f"Extracted grade: {prediction}, Validated: {validated_grade}, Is valid: {is_valid}")
        
        # Structured logging for better observability
        _log_structured(self.log_fn, "grade_extraction", {
            "prediction": prediction,
            "validated_grade": validated_grade,
            "is_valid": is_valid,
            "has_reasoning": bool(reasoning),
        })
        
        # If grade is invalid, try to extract from the full response text
        if not is_valid and response:
            # Try to find any numeric grade in the response
            numeric_match = re.search(r'\b([0-7])\b', response)
            if numeric_match:
                validated_grade = numeric_match.group(1)
                is_valid = True
                self.log_fn(f"Fallback extraction found grade: {validated_grade}")
            else:
                # Try to find grade patterns in the full response
                grade_patterns = [
                    r'(?:grade|score|mark|final grade|final score)\s*:?\s*([0-7])',
                    r'\bfull\s*(?:credit|points?|score)?\b',
                    r'\bcorrect\s*(?:solution|answer)?\b',
                    r'\bno\s*(?:credit|points?|score|marks?)?\b',
                    r'\bzero\s*(?:credit|points?|score|marks?)?\b',
                    r'\bincorrect\s*(?:solution|answer)?\b',
                ]
                for pattern in grade_patterns:
                    match = re.search(pattern, response, re.IGNORECASE)
                    if match:
                        if 'full' in pattern or 'correct' in pattern:
                            validated_grade = "7"
                        elif 'no' in pattern or 'zero' in pattern or 'incorrect' in pattern:
                            validated_grade = "0"
                        else:
                            validated_grade = match.group(1)
                        is_valid = True
                        self.log_fn(f"Pattern-based fallback found grade: {validated_grade}")
                        break

        # Calculate confidence score based on reasoning
        confidence = _calculate_confidence(reasoning, str(validated_grade), is_valid)
        self._last_confidence = confidence
        
        # Log confidence metrics
        self.log_fn(f"Confidence: {confidence['score']} ({confidence['level']}), needs_review: {confidence['needs_review']}")
        if confidence['indicators']:
            self.log_fn(f"Confidence indicators: {confidence['indicators'][:5]}")  # Log first 5
        
        # Structured logging for final result with confidence
        _log_structured(self.log_fn, "grade_final", {
            "final_grade": str(validated_grade),
            "is_valid": is_valid,
            "fallback_used": not is_valid and prediction == "None",
            "confidence_score": confidence['score'],
            "confidence_level": confidence['level'],
            "needs_review": confidence['needs_review'],
        })

        return str(validated_grade), msg_history
    
    def get_last_confidence(self) -> dict | None:
        """Get the confidence metrics from the last grading decision.
        
        Returns:
            Dictionary with confidence metrics or None if no grading done yet.
        """
        return self._last_confidence
