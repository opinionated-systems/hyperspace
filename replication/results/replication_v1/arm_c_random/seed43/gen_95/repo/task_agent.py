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
    Includes enhanced recovery for malformed JSON.
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
        
        # Try direct JSON parsing first
        try:
            results.append(json.loads(inner))
            continue
        except json.JSONDecodeError:
            pass
        
        # Recovery strategy 1: Find JSON object boundaries
        try:
            json_start = inner.find('{')
            json_end = inner.rfind('}')
            if json_start != -1 and json_end != -1 and json_end > json_start:
                results.append(json.loads(inner[json_start:json_end+1]))
                continue
        except json.JSONDecodeError:
            pass
        
        # Recovery strategy 2: Handle common JSON formatting issues
        try:
            # Remove trailing commas before closing braces
            cleaned = re.sub(r',\s*}', '}', inner)
            cleaned = re.sub(r',\s*]', ']', cleaned)
            # Fix single quotes to double quotes
            cleaned = cleaned.replace("'", '"')
            # Try to find and parse the JSON object
            json_start = cleaned.find('{')
            json_end = cleaned.rfind('}')
            if json_start != -1 and json_end != -1 and json_end > json_start:
                results.append(json.loads(cleaned[json_start:json_end+1]))
                continue
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Recovery strategy 3: Extract key-value pairs manually for simple cases
        try:
            # Look for "response" field specifically
            response_match = re.search(r'"response"\s*:\s*"?([^",}]+)"?', inner)
            reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', inner, re.DOTALL)
            if response_match:
                obj = {"response": response_match.group(1).strip().strip('"')}
                if reasoning_match:
                    obj["reasoning"] = reasoning_match.group(1)
                results.append(obj)
        except Exception:
            pass
    
    return results or None


def _extract_json_from_markdown(text: str) -> dict | None:
    """Extract JSON from markdown code blocks (```json ... ```).
    
    Includes enhanced recovery for malformed JSON in markdown blocks.
    """
    # Try both ```json and ``` code blocks
    patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(\{.*?\})\s*```',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            match = match.strip()
            
            # Try direct parsing
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                pass
            
            # Recovery strategy 1: Find JSON boundaries
            try:
                json_start = match.find('{')
                json_end = match.rfind('}')
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    return json.loads(match[json_start:json_end+1])
            except json.JSONDecodeError:
                pass
            
            # Recovery strategy 2: Clean common issues
            try:
                cleaned = re.sub(r',\s*}', '}', match)
                cleaned = re.sub(r',\s*]', ']', cleaned)
                cleaned = cleaned.replace("'", '"')
                json_start = cleaned.find('{')
                json_end = cleaned.rfind('}')
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    return json.loads(cleaned[json_start:json_end+1])
            except (json.JSONDecodeError, ValueError):
                pass
            
            # Recovery strategy 3: Extract key fields
            try:
                response_match = re.search(r'"response"\s*:\s*"?([^",}]+)"?', match)
                reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', match, re.DOTALL)
                if response_match:
                    obj = {"response": response_match.group(1).strip().strip('"')}
                    if reasoning_match:
                        obj["reasoning"] = reasoning_match.group(1)
                    return obj
            except Exception:
                pass
    
    return None


def _validate_grade(prediction: str, grading_guidelines: str) -> tuple[str, bool]:
    """Validate that the extracted grade is reasonable.
    
    Enhanced validation with strict support for IMO 0-7 point scale.
    Uses multi-stage validation with pattern matching and context analysis.
    
    Returns:
        (validated_grade, is_valid)
    """
    if not prediction or prediction == "None":
        return "None", False
    
    pred_clean = prediction.strip()
    pred_lower = pred_clean.lower()
    
    # Stage 1: Direct digit validation (most reliable)
    if pred_clean in ["0", "1", "2", "3", "4", "5", "6", "7"]:
        return pred_clean, True
    
    # Stage 2: Extract from common grade formats
    # Fractional grades like "3/7" or "5 / 7"
    fractional_match = re.search(r'\b([0-7])\s*/\s*7\b', pred_clean)
    if fractional_match:
        return fractional_match.group(1), True
    
    # "X out of 7" format
    outof_match = re.search(r'\b([0-7])\s+out\s+of\s+7\b', pred_lower)
    if outof_match:
        return outof_match.group(1), True
    
    # Stage 3: Look for explicit grade mentions with context
    # Pattern: "grade is X", "score of X", "X points"
    explicit_patterns = [
        r'(?:grade|score|mark)\s*(?:is|:|=)\s*([0-7])\b',
        r'(?:grade|score|mark)\s+of\s+([0-7])\b',
        r'\b([0-7])\s+points?\b',
        r'\b([0-7])\s*(?:/|out\s+of)\s*7\b',
    ]
    for pattern in explicit_patterns:
        match = re.search(pattern, pred_lower)
        if match:
            return match.group(1), True
    
    # Stage 4: Semantic pattern matching for edge cases
    # Full credit patterns -> 7
    full_credit_indicators = [
        r'\bfull\s*(?:credit|points?|score|marks?)?\b',
        r'\bcomplete\s*(?:solution|answer|credit)?\b',
        r'\ball\s*(?:points?|credit|marks?)?\b',
        r'\bperfect\s*(?:score|solution|answer)?\b',
        r'\bcorrect\s*(?:solution|answer|approach)?\b',
        r'\bentirely\s+correct\b',
        r'\bfully\s+correct\b',
        r'\bsolved\s+(?:completely|fully|correctly)\b',
    ]
    for pattern in full_credit_indicators:
        if re.search(pattern, pred_lower):
            return "7", True
    
    # Zero credit patterns -> 0
    zero_credit_indicators = [
        r'\bno\s*(?:credit|points?|score|marks?)?\b',
        r'\bzero\s*(?:credit|points?|score|marks?)?\b',
        r'\b0\s*(?:points?|credit|score|marks?)?\b',
        r'\bincorrect\s*(?:solution|answer|approach)?\b',
        r'\bwrong\s*(?:solution|answer|approach)?\b',
        r'\bno\s+(?:solution|answer|progress|work)\b',
        r'\bblank\s*(?:answer|response)?\b',
        r'\bentirely\s+incorrect\b',
        r'\bcompletely\s+wrong\b',
    ]
    for pattern in zero_credit_indicators:
        if re.search(pattern, pred_lower):
            return "0", True
    
    # Stage 5: Partial credit with numeric extraction
    partial_indicators = [
        r'\bpartial\s*(?:credit|points?|score)?\b',
        r'\bsome\s*(?:credit|points?|score|progress)?\b',
        r'\bincomplete\s*(?:solution|answer)?\b',
        r'\bpartially\s*correct\b',
        r'\bminor\s+(?:progress|insight|step)\b',
        r'\blimited\s+(?:progress|success|correctness)\b',
    ]
    for pattern in partial_indicators:
        if re.search(pattern, pred_lower):
            # Try to extract a specific numeric grade from the text
            numeric_in_text = re.search(r'\b([1-6])\b', pred_clean)
            if numeric_in_text:
                return numeric_in_text.group(1), True
            # Default to 3 for partial credit without specific number
            return "3", True
    
    # Stage 6: Last resort - look for any digit 0-7 in the text
    # This is a fallback and should be used cautiously
    any_digit = re.search(r'\b([0-7])\b', pred_clean)
    if any_digit:
        return any_digit.group(1), True
    
    # If no clear grade found, mark as invalid
    return pred_clean, False


def _calculate_confidence(reasoning: str, grade: str, is_valid: bool) -> dict:
    """Calculate confidence metrics for the grading decision.
    
    Analyzes the reasoning text to determine how confident the model is
    in its grading decision. Uses weighted scoring with contextual analysis.
    
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
    
    # High confidence indicators with weights
    high_confidence_patterns = [
        (r'\bclearly\s+(?:correct|wrong|incorrect|right)\b', 'clearly_definitive', 0.20),
        (r'\bdefinitely\s+(?:correct|wrong|incorrect|right)\b', 'definitely_definitive', 0.20),
        (r'\bunambiguous\w*\b', 'unambiguous', 0.18),
        (r'\bstraightforward\s+(?:solution|approach|proof)\b', 'straightforward', 0.15),
        (r'\bmatches\s+(?:the\s+)?(?:official|reference)\s+(?:solution|answer)\b', 'matches_official', 0.18),
        (r'\bcomplete\s+(?:and\s+)?correct\b', 'complete_correct', 0.20),
        (r'\bno\s+(?:errors?|mistakes?|issues?|problems?)\b', 'no_errors', 0.15),
        (r'\bperfect\s+(?:solution|answer|proof)\b', 'perfect_solution', 0.20),
        (r'\bentirely\s+(?:correct|wrong|incorrect)\b', 'entirely_definitive', 0.18),
        (r'\bfully\s+(?:correct|justified|proven)\b', 'fully_correct', 0.17),
        (r'\bcorrectly\s+(?:solved|proved|derived|calculated)\b', 'correctly_solved', 0.15),
        (r'\bsound\s+(?:logic|reasoning|argument)\b', 'sound_logic', 0.12),
        (r'\bvalid\s+(?:proof|solution|approach)\b', 'valid_proof', 0.12),
    ]
    
    # Low confidence indicators with weights
    low_confidence_patterns = [
        (r'\bunclear\w*\b', 'unclear', 0.18),
        (r'\bambiguous\w*\b', 'ambiguous', 0.18),
        (r'\buncertain\w*\b', 'uncertain', 0.20),
        (r'\bnot\s+sure\b', 'not_sure', 0.20),
        (r'\bdifficult\s+to\s+(?:say|determine|judge|assess)\b', 'difficult_to_judge', 0.18),
        (r'\bmight\s+be\b', 'might_be', 0.15),
        (r'\bcould\s+be\b', 'could_be', 0.15),
        (r'\bpossibly\b', 'possibly', 0.12),
        (r'\bperhaps\b', 'perhaps', 0.12),
        (r'\bpartially\s+(?:correct|right|valid)\b', 'partially_correct', 0.14),
        (r'\bincomplete\s+(?:solution|answer|proof)\b', 'incomplete', 0.16),
        (r'\balternative\s+(?:approach|method|solution)\b', 'alternative_approach', 0.10),
        (r'\bwithout\s+(?:the\s+)?full\s+(?:solution|work|proof)\b', 'missing_work', 0.14),
        (r'\bcannot\s+(?:fully|completely)\s+(?:verify|assess|determine|judge)\b', 'cannot_verify', 0.18),
        (r'\bhard\s+to\s+(?:tell|say|determine)\b', 'hard_to_tell', 0.15),
        (r'\bvague\b', 'vague', 0.16),
        (r'\bquestionable\b', 'questionable', 0.14),
        (r'\bdubious\b', 'dubious', 0.16),
        (r'\bneeds?\s+(?:more|further)\s+(?:review|checking|verification)\b', 'needs_review', 0.15),
    ]
    
    # Calculate weighted scores
    high_score = 0.0
    for pattern, indicator, weight in high_confidence_patterns:
        if re.search(pattern, reasoning_lower):
            indicators.append(f"+{indicator}")
            high_score += weight
    
    low_score = 0.0
    for pattern, indicator, weight in low_confidence_patterns:
        if re.search(pattern, reasoning_lower):
            indicators.append(f"-{indicator}")
            low_score += weight
    
    # Calculate base confidence score starting from neutral
    base_score = 0.5 + high_score - low_score
    
    # Adjust based on grade extremity with refined logic
    # Extreme grades (0, 7) are often more confident, but middle grades can be confident too
    if grade in ["0", "7"]:
        # Check if reasoning supports the extreme grade
        if grade == "7" and any(i.startswith("+") for i in indicators):
            base_score += 0.08
        elif grade == "0" and any(i.startswith("+") for i in indicators):
            base_score += 0.08
    elif grade in ["1", "2"]:
        # Low partial credit - check for clear justification
        if "+partially_correct" in indicators or "+incomplete" in indicators:
            base_score += 0.05
    elif grade in ["5", "6"]:
        # High partial credit - should have strong justification
        if not any(i.startswith("+") for i in indicators):
            base_score -= 0.05
    
    # Adjust based on reasoning length (very short reasoning is suspicious)
    reasoning_length = len(reasoning.split())
    if reasoning_length < 20:
        base_score -= 0.1  # Too short
        indicators.append("-short_reasoning")
    elif reasoning_length > 100:
        base_score += 0.03  # Detailed reasoning is good
        indicators.append("+detailed_reasoning")
    
    # Clamp to [0, 1]
    confidence_score = max(0.0, min(1.0, base_score))
    
    # Determine level with adjusted thresholds
    if confidence_score >= 0.75:
        level = "high"
    elif confidence_score >= 0.45:
        level = "medium"
    else:
        level = "low"
    
    # Flag for review based on multiple factors
    needs_review = (
        level == "low" or 
        low_score >= 0.35 or  # Multiple uncertainty indicators
        (grade in ["3", "4"] and confidence_score < 0.6) or  # Middle grades need higher confidence
        (not is_valid) or
        reasoning_length < 15  # Very short reasoning
    )
    
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

        return f"""You are an expert IMO (International Mathematical Olympiad) grader with years of experience evaluating mathematical proofs and solutions.

Your task is to evaluate a student's solution to a mathematical problem with precision and consistency.

## Problem Domain
{domain}

## Problem Statement
{problem}

## Official Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Evaluation Framework

When grading, consider these key aspects:

1. **Correctness**: Does the solution arrive at the correct answer? Are all mathematical steps valid?
2. **Completeness**: Are all necessary steps present? Is the reasoning fully developed?
3. **Clarity**: Is the solution well-organized and easy to follow?
4. **Rigor**: Are claims properly justified? Is the logic sound?
5. **Creativity**: Does the student show original thinking or alternative valid approaches?

## IMO Grading Scale (0-7)

- **7 points**: Complete, correct solution with clear reasoning
- **6 points**: Correct solution with minor gaps or presentation issues
- **5 points**: Correct approach with significant progress, minor errors
- **4 points**: Substantial progress toward solution, some correct key ideas
- **3 points**: Meaningful progress, at least one correct key idea
- **2 points**: Some progress, correct approach started but not developed
- **1 point**: Minor progress, some relevant work but mostly incorrect
- **0 points**: No meaningful progress, completely wrong, or blank

## Instructions

1. Analyze the student's answer step by step, comparing against the official solution
2. Identify specific strengths: correct calculations, valid lemmas, good approaches
3. Identify specific weaknesses: errors, missing steps, logical gaps, unclear reasoning
4. Consider if alternative approaches are mathematically valid even if different from official solution
5. Determine the grade based on the IMO scale above
6. Provide detailed reasoning explaining your evaluation

## Output Format

You MUST respond with a valid JSON object wrapped in <json> tags. The JSON must have exactly these two fields:

<json>
{{
    "reasoning": "Your detailed analysis here. Explain: (1) What the student did correctly, (2) What errors or gaps exist, (3) How this compares to the official solution, (4) Why you assigned this specific grade based on the IMO scale...",
    "response": "X"
}}
</json>

CRITICAL REQUIREMENTS:
- The "response" field MUST contain ONLY a single digit from 0 to 7 (e.g., "7", "5", "0")
- Do NOT include any other text, explanations, or formatting in the response field
- The "reasoning" field should contain your full analysis with specific details
- Valid grades are ONLY: 0, 1, 2, 3, 4, 5, 6, 7
- Be precise and consistent with the IMO grading standards"""

    def _extract_prediction(self, msg_history: list[dict]) -> tuple[str, str]:
        """Extract prediction and reasoning from message history.
        
        Uses a multi-stage extraction strategy with increasing fallback options.
        
        Returns:
            (prediction, reasoning)
        """
        prediction = "None"
        reasoning = ""
        
        try:
            # Stage 1: Extract message content from various formats
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
            
            # Stage 2: Try <json> tags (primary format)
            extracted = _extract_jsons(last_msg)
            if extracted:
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = str(last_json["response"]).strip()
                if "reasoning" in last_json:
                    reasoning = str(last_json["reasoning"])
                if prediction != "None":
                    return prediction, reasoning
            
            # Stage 3: Try markdown code blocks
            md_json = _extract_json_from_markdown(last_msg)
            if md_json:
                if "response" in md_json:
                    prediction = str(md_json["response"]).strip()
                if "reasoning" in md_json:
                    reasoning = str(md_json["reasoning"])
                if prediction != "None":
                    return prediction, reasoning
            
            # Stage 4: Try to find any JSON-like structure with response field
            # Look for patterns like {"response": "X", "reasoning": "..."}
            json_patterns = [
                r'\{[^}]*"response"\s*:\s*"([^"]*?)"[^}]*"reasoning"\s*:\s*"([^"]*?)"[^}]*\}',
                r'\{[^}]*"reasoning"\s*:\s*"([^"]*?)"[^}]*"response"\s*:\s*"([^"]*?)"[^}]*\}',
                r'\{[^}]*"response"\s*:\s*([0-7])[^}]*\}',
            ]
            for pattern in json_patterns:
                match = re.search(pattern, last_msg, re.DOTALL)
                if match:
                    try:
                        if len(match.groups()) >= 2:
                            if "reasoning" in pattern[:50]:
                                reasoning = match.group(1)
                                prediction = str(match.group(2)).strip()
                            else:
                                prediction = str(match.group(1)).strip()
                                reasoning = match.group(2)
                        else:
                            prediction = str(match.group(1)).strip()
                        if prediction != "None":
                            return prediction, reasoning
                    except (IndexError, ValueError):
                        continue
            
            # Stage 5: Look for explicit grade mentions in text
            grade_patterns = [
                r'(?:grade|score|mark|final grade|final score)\s*[:=]\s*([0-7])\b',
                r'(?:grade|score|mark)\s+is\s+([0-7])\b',
                r'(?:grade|score|mark)\s+of\s+([0-7])\b',
                r'\b([0-7])\s*points?\b',
                r'\bgrade\s*[:=]\s*"([0-7])"',
                r'"response"\s*:\s*"([0-7])"',
            ]
            for pattern in grade_patterns:
                match = re.search(pattern, last_msg, re.IGNORECASE)
                if match:
                    prediction = match.group(1)
                    return prediction, reasoning
            
            # Stage 6: Extract any reasoning text found
            reasoning_patterns = [
                r'"reasoning"\s*:\s*"([^"]{20,500})"',
                r'reasoning[:\s]+(.{20,500}?)(?:\n\n|\Z|grade|score)',
            ]
            for pattern in reasoning_patterns:
                match = re.search(pattern, last_msg, re.IGNORECASE | re.DOTALL)
                if match:
                    reasoning = match.group(1).strip()
                    break
                    
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
            # Stage 1: Try to find explicit grade mentions with context
            explicit_patterns = [
                r'(?:grade|score|mark|final grade|final score)\s*[:=]\s*([0-7])\b',
                r'(?:grade|score|mark)\s+is\s+([0-7])\b',
                r'(?:grade|score|mark)\s+of\s+([0-7])\b',
                r'\b([0-7])\s*points?\b',
                r'\b([0-7])\s*/\s*7\b',
            ]
            for pattern in explicit_patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    validated_grade = match.group(1)
                    is_valid = True
                    self.log_fn(f"Fallback extraction found grade: {validated_grade}")
                    break
            
            # Stage 2: Try semantic patterns if no explicit grade found
            if not is_valid:
                semantic_patterns = [
                    (r'\bfull\s*(?:credit|points?|score|marks?)?\b', "7"),
                    (r'\bcomplete\s*(?:solution|answer|credit)?\b', "7"),
                    (r'\ball\s*(?:points?|credit|marks?)?\b', "7"),
                    (r'\bperfect\s*(?:score|solution|answer)?\b', "7"),
                    (r'\bentirely\s+correct\b', "7"),
                    (r'\bno\s*(?:credit|points?|score|marks?)?\b', "0"),
                    (r'\bzero\s*(?:credit|points?|score|marks?)?\b', "0"),
                    (r'\bno\s+(?:solution|answer|progress|work)\b', "0"),
                    (r'\bincorrect\s*(?:solution|answer|approach)?\b', "0"),
                    (r'\bwrong\s*(?:solution|answer|approach)?\b', "0"),
                    (r'\bentirely\s+incorrect\b', "0"),
                ]
                for pattern, default_grade in semantic_patterns:
                    if re.search(pattern, response, re.IGNORECASE):
                        validated_grade = default_grade
                        is_valid = True
                        self.log_fn(f"Semantic fallback found grade: {validated_grade}")
                        break
            
            # Stage 3: Last resort - any digit 0-7
            if not is_valid:
                numeric_match = re.search(r'\b([0-7])\b', response)
                if numeric_match:
                    validated_grade = numeric_match.group(1)
                    is_valid = True
                    self.log_fn(f"Numeric fallback found grade: {validated_grade}")

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
