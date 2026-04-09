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
    Enhanced to handle nested JSON and malformed content.
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
        
        # Try to parse the JSON
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError as e:
            # Try to fix common JSON issues
            # 1. Handle trailing commas
            fixed = re.sub(r',\s*}', '}', inner)
            fixed = re.sub(r',\s*]', ']', fixed)
            # 2. Handle single quotes (convert to double)
            fixed = fixed.replace("'", '"')
            try:
                results.append(json.loads(fixed))
                logger.debug(f"Fixed malformed JSON: {e}")
            except json.JSONDecodeError:
                logger.debug(f"Failed to parse JSON: {e}, content: {inner[:100]!r}")
                continue
    return results or None


def _extract_json_from_markdown(text: str) -> list[dict] | None:
    """Extract JSON objects from markdown code blocks.

    Fallback for when <json> tags are not used but markdown code blocks are.
    Enhanced to handle various code block formats and malformed JSON.
    """
    results = []
    # Match ```json ... ``` or just ``` ... ``` blocks
    # Also handle cases where the language tag has extra whitespace
    pattern = r'```(?:json)?\s*\n?(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        content = match.strip()
        if not content:
            continue
        
        # Try to parse the JSON
        try:
            results.append(json.loads(content))
        except json.JSONDecodeError as e:
            # Try to fix common JSON issues
            # 1. Handle trailing commas
            fixed = re.sub(r',\s*}', '}', content)
            fixed = re.sub(r',\s*]', ']', fixed)
            # 2. Handle single quotes (convert to double)
            fixed = fixed.replace("'", '"')
            # 3. Handle unquoted keys (simple cases)
            fixed = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', fixed)
            try:
                results.append(json.loads(fixed))
                logger.debug(f"Fixed malformed markdown JSON: {e}")
            except json.JSONDecodeError:
                logger.debug(f"Failed to parse markdown JSON: {e}, content: {content[:100]!r}")
                continue
    return results or None


def _normalize_grade(grade: str) -> str:
    """Normalize grade to a standard format.

    Handles various grade formats and normalizes them.
    Enhanced to handle numeric grades and more variations.
    """
    if not isinstance(grade, str):
        grade = str(grade)
    
    grade = grade.strip().lower()
    
    # Handle empty or whitespace-only grades
    if not grade:
        return 'None'
    
    # First, check for numeric grades (0, 1, 2, etc.)
    # These are common in IMO-style grading
    try:
        numeric_grade = float(grade)
        if numeric_grade == 0:
            return 'Incorrect'
        elif numeric_grade >= 1:
            return 'Correct'
        elif 0 < numeric_grade < 1:
            return 'Partial'
    except ValueError:
        pass
    
    # Map common variations to standard formats
    correct_variations = [
        'correct', 'right', 'true', 'yes', 'full', 'full credit', 
        'full marks', 'complete', 'valid', 'accepted', 'pass', 'perfect',
        'satisfactory', 'adequate', 'sufficient'
    ]
    incorrect_variations = [
        'incorrect', 'wrong', 'false', 'no', 'none', 'zero',
        'invalid', 'rejected', 'fail', 'error', 'mistake', 'unsatisfactory',
        'inadequate', 'insufficient', 'unacceptable'
    ]
    partial_variations = [
        'partial', 'partial credit', 'half', 'incomplete',
        'partially correct', 'partially right', 'some credit', 'partially',
        'mostly correct', 'mostly wrong', 'partial success'
    ]
    
    # Check for exact matches first, then substring matches
    if grade in correct_variations:
        return 'Correct'
    if grade in incorrect_variations:
        return 'Incorrect'
    if grade in partial_variations:
        return 'Partial'
    
    # Substring matching for more flexibility
    if any(v in grade for v in correct_variations):
        return 'Correct'
    elif any(v in grade for v in incorrect_variations):
        return 'Incorrect'
    elif any(v in grade for v in partial_variations):
        return 'Partial'
    
    # Handle special cases like "0/1", "1/1", "0/2", "2/2", etc.
    if '/' in grade:
        parts = grade.split('/')
        if len(parts) == 2:
            try:
                numerator = float(parts[0].strip())
                denominator = float(parts[1].strip())
                if denominator > 0:
                    ratio = numerator / denominator
                    if ratio == 0:
                        return 'Incorrect'
                    elif ratio >= 0.5:
                        return 'Correct'
                    else:
                        return 'Partial'
            except ValueError:
                pass
    
    # Handle percentage grades
    if '%' in grade:
        try:
            pct = float(grade.replace('%', '').strip())
            if pct == 0:
                return 'Incorrect'
            elif pct >= 50:
                return 'Correct'
            else:
                return 'Partial'
        except ValueError:
            pass
    
    # Return original if no normalization applied
    return grade.strip()


def _calculate_confidence_score(grade: str, reasoning: str) -> float:
    """Calculate a confidence score for the grading decision.
    
    Analyzes the reasoning text to determine how confident the grader
    was in their assessment. Returns a score between 0.0 and 1.0.
    
    Args:
        grade: The assigned grade (Correct, Incorrect, Partial, or None)
        reasoning: The detailed reasoning text from the grader
        
    Returns:
        A confidence score between 0.0 (low confidence) and 1.0 (high confidence)
    """
    if not reasoning or not isinstance(reasoning, str):
        return 0.5  # Default medium confidence for missing reasoning
    
    reasoning_lower = reasoning.lower()
    confidence_indicators = {
        'high': [
            'clearly', 'definitely', 'certainly', 'absolutely', 'unambiguously',
            'without doubt', 'conclusive', 'definitive', 'obvious', 'evident',
            'undoubtedly', 'certain', 'sure', 'confident'
        ],
        'low': [
            'unclear', 'ambiguous', 'uncertain', 'possibly', 'maybe', 'might',
            'could be', 'unclear whether', 'difficult to determine', 'not sure',
            'uncertain', 'doubtful', 'questionable', 'inconclusive'
        ]
    }
    
    # Count confidence indicators
    high_count = sum(1 for indicator in confidence_indicators['high'] 
                     if indicator in reasoning_lower)
    low_count = sum(1 for indicator in confidence_indicators['low'] 
                    if indicator in reasoning_lower)
    
    # Base confidence from indicators
    if high_count > low_count:
        base_confidence = 0.7 + min(0.3, (high_count - low_count) * 0.1)
    elif low_count > high_count:
        base_confidence = 0.3 - min(0.3, (low_count - high_count) * 0.1)
    else:
        base_confidence = 0.5
    
    # Adjust based on grade type
    grade_adjustments = {
        'Correct': 0.1,    # Slightly boost confidence for clear correct answers
        'Incorrect': 0.05, # Small boost for clear incorrect answers
        'Partial': -0.1,   # Reduce confidence for partial (often ambiguous)
        'None': -0.2       # Significantly reduce for unknown grades
    }
    
    adjustment = grade_adjustments.get(grade, 0.0)
    final_confidence = max(0.0, min(1.0, base_confidence + adjustment))
    
    return round(final_confidence, 2)


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _build_grading_prompt(self, inputs: dict) -> str:
        """Build a structured grading prompt with clear evaluation criteria."""
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        return f"""You are an expert mathematical grader specializing in {domain} problems.

Your task is to evaluate a student's answer to a mathematics problem and assign a grade.

## Problem Statement:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Evaluation Framework:

Follow this structured evaluation process:

### Step 1: Problem Understanding
- What is the problem asking for?
- What are the key constraints and conditions?
- What is the expected answer format?

### Step 2: Solution Analysis
- What is the correct approach according to the official solution?
- What are the critical steps that must be present?
- What constitutes a complete vs. incomplete solution?

### Step 3: Student Answer Evaluation
- Did the student understand the problem correctly?
- What approach did the student take?
- Are the student's steps logically valid?
- Did the student show sufficient work and reasoning?
- Is the final answer mathematically correct?

### Step 4: Grade Assignment
Based on the grading guidelines, assign the appropriate grade considering:
- Correctness of the final answer
- Validity of the reasoning process
- Completeness of the solution
- Adherence to the expected solution method

## Response Format:

You MUST respond in JSON format wrapped in <json> tags:

<json>
{{
    "reasoning": "Your detailed step-by-step analysis following the Evaluation Framework above",
    "response": "The final grade you assign (e.g., '0', '1', '2', 'Correct', 'Incorrect', 'Partial')",
    "confidence": "Your confidence level in this grade: High, Medium, or Low"
}}
</json>

Important: 
- The "response" field must contain ONLY the grade value, nothing else.
- The "confidence" field should reflect your certainty: use "High" when the grade is clear and unambiguous, "Medium" when there is some ambiguity but you can make a reasonable judgment, and "Low" when the answer is difficult to evaluate."""

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history with multiple fallback strategies."""
        if not msg_history:
            return "None"
        
        last_message = msg_history[-1].get("text", "")
        
        # Strategy 1: Extract from <json> tags
        extracted = _extract_jsons(last_message)
        if extracted:
            return self._get_grade_from_json(extracted[-1])
        
        # Strategy 2: Extract from markdown code blocks
        extracted = _extract_json_from_markdown(last_message)
        if extracted:
            return self._get_grade_from_json(extracted[-1])
        
        # Strategy 3: Look for grade patterns in plain text
        return self._extract_grade_from_text(last_message)

    def _get_grade_from_json(self, json_obj: dict) -> str:
        """Extract grade from JSON object with field priority."""
        # Priority order for grade fields
        priority_fields = ["response", "grade", "answer", "result", "score", "evaluation"]
        
        for field in priority_fields:
            if field in json_obj:
                value = json_obj[field]
                if isinstance(value, str):
                    return _normalize_grade(value)
                elif isinstance(value, (int, float)):
                    return str(value)
        
        # If no recognized field, use the first string value found
        for key, value in json_obj.items():
            if isinstance(value, str):
                return _normalize_grade(value)
            elif isinstance(value, (int, float)):
                return str(value)
        
        return "None"

    def _get_confidence_from_json(self, json_obj: dict) -> str:
        """Extract confidence level from JSON object.
        
        Returns High, Medium, Low, or None if not found.
        """
        confidence_fields = ["confidence", "certainty", "confidence_level", "certainty_level"]
        
        for field in confidence_fields:
            if field in json_obj:
                value = json_obj[field]
                if isinstance(value, str):
                    value_lower = value.lower().strip()
                    if value_lower in ['high', 'medium', 'low']:
                        return value_lower.capitalize()
                    # Handle numeric confidence (0-1 or 0-100)
                    try:
                        num_val = float(value_lower)
                        if num_val >= 0.7 or num_val >= 70:
                            return 'High'
                        elif num_val >= 0.4 or num_val >= 40:
                            return 'Medium'
                        else:
                            return 'Low'
                    except ValueError:
                        # Check for descriptive terms
                        if any(term in value_lower for term in ['high', 'strong', 'very', 'certain', 'sure', 'confident']):
                            return 'High'
                        elif any(term in value_lower for term in ['medium', 'moderate', 'fair', 'reasonable']):
                            return 'Medium'
                        elif any(term in value_lower for term in ['low', 'weak', 'uncertain', 'unsure', 'doubtful']):
                            return 'Low'
        
        return None

    def _extract_grade_from_text(self, text: str) -> str:
        """Extract grade from plain text using pattern matching.
        
        Enhanced with additional patterns for robust extraction from
        various response formats including IMO-style numeric grades.
        Includes comprehensive logging for debugging extraction failures.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Look for explicit grade statements with flexible patterns
        patterns = [
            # Standard grade assignments
            (r'grade[\s]*[:=][\s]*["\']?([^"\'\n]+)["\']?', 'standard_grade'),
            (r'response[\s]*[:=][\s]*["\']?([^"\'\n]+)["\']?', 'response_field'),
            (r'final grade[\s]*[:=][\s]*["\']?([^"\'\n]+)["\']?', 'final_grade'),
            (r'assign[\s]+["\']?([^"\'\n]+)["\']?', 'assign_statement'),
            # IMO-style grade statements
            (r'(?:the\s+)?(?:student\s+(?:receives?|gets?|earns?)|I\s+(?:assign|give|award))\s+["\']?(\d+(?:\.\d+)?)["\']?\s*(?:points?|marks?)?', 'imo_style'),
            (r'(?:score|grade|mark)[\s]*[:=\s]+["\']?(\d+(?:\.\d+)?)["\']?', 'score_assignment'),
            # Grade at end of sentence
            (r'(?:grade|score|mark|result)[\s]+(?:is|of)[\s]+["\']?([^"\'\n.]+)["\']?', 'grade_is_statement'),
            # Standalone grades in common formats
            (r'\b([0-2](?:\.0|\.5)?)\b', 'standalone_numeric'),
            (r'\b(Correct|Incorrect|Partial|Right|Wrong)\b', 'standalone_word'),
        ]
        
        for pattern, pattern_name in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                normalized = _normalize_grade(extracted)
                logger.debug(f"Pattern '{pattern_name}' matched: '{extracted}' -> '{normalized}'")
                return normalized
        
        # Fallback: look for numeric grades (0, 1, 2) as standalone tokens
        # This handles cases where the model just outputs a number
        numeric_matches = re.findall(r'\b([0-2])\b', text)
        if numeric_matches:
            # Return the last numeric match (often the final grade)
            result = _normalize_grade(numeric_matches[-1])
            logger.debug(f"Fallback numeric match: {numeric_matches} -> '{result}'")
            return result
        
        # Additional fallback: look for grade-like words anywhere in the text
        # This helps catch cases where the model outputs something like "The answer is correct"
        text_lower = text.lower()
        if 'correct' in text_lower and 'incorrect' not in text_lower:
            logger.debug("Fallback keyword match: 'correct'")
            return 'Correct'
        if 'incorrect' in text_lower or 'wrong' in text_lower:
            logger.debug("Fallback keyword match: 'incorrect/wrong'")
            return 'Incorrect'
        if 'partial' in text_lower:
            logger.debug("Fallback keyword match: 'partial'")
            return 'Partial'
        
        # Final fallback: check for common grade indicators in sentences
        # Look for phrases like "the answer is X" or "I would say X"
        final_patterns = [
            (r'(?:the\s+)?(?:answer|solution|grade|evaluation)\s+(?:is|would\s+be)\s+["\']?([^"\'\n.]{1,20})["\']?', 'answer_is'),
            (r'(?:therefore|thus|hence|so)\s*,?\s*(?:the\s+)?(?:grade|score|result)\s+(?:is|would\s+be)\s+["\']?([^"\'\n.]{1,20})["\']?', 'therefore_grade'),
        ]
        for pattern, pattern_name in final_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                normalized = _normalize_grade(extracted)
                logger.debug(f"Final fallback pattern '{pattern_name}' matched: '{extracted}' -> '{normalized}'")
                return normalized
        
        logger.warning(f"No grade pattern matched in text (first 200 chars): {text[:200]!r}")
        return "None"

    def forward(self, inputs: dict) -> tuple[str, list[dict], dict]:
        """Run the task agent on a single problem with enhanced error handling.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history, metadata) where metadata includes confidence info
        """
        # Validate required inputs
        required_fields = ["problem", "solution", "grading_guidelines", "student_answer"]
        missing_fields = [f for f in required_fields if not inputs.get(f)]
        if missing_fields:
            self.log_fn(f"Warning: Missing input fields: {missing_fields}")
        
        instruction = self._build_grading_prompt(inputs)

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"LLM call failed: {e}")
            return "None", [], {"confidence": None, "confidence_score": 0.0}

        # Extract prediction using multiple strategies
        prediction = self._extract_prediction(msg_history)
        
        # Extract confidence from JSON response if available
        confidence = None
        confidence_score = 0.0
        reasoning = ""
        
        if msg_history:
            last_message = msg_history[-1].get("text", "")
            
            # Try to extract JSON and get confidence
            extracted = _extract_jsons(last_message)
            if not extracted:
                extracted = _extract_json_from_markdown(last_message)
            
            if extracted:
                json_obj = extracted[-1]
                confidence = self._get_confidence_from_json(json_obj)
                reasoning = json_obj.get("reasoning", "")
        
        # Calculate confidence score using both explicit confidence and reasoning analysis
        if confidence:
            # Map explicit confidence to score
            confidence_map = {'High': 0.85, 'Medium': 0.6, 'Low': 0.35}
            explicit_score = confidence_map.get(confidence, 0.5)
            # Blend with calculated score from reasoning
            calculated_score = _calculate_confidence_score(prediction, reasoning)
            confidence_score = round((explicit_score + calculated_score) / 2, 2)
        else:
            # Use calculated score only
            confidence_score = _calculate_confidence_score(prediction, reasoning)
        
        if prediction == "None":
            self.log_fn(f"Failed to extract prediction from response: {response[:200] if response else 'empty'}")
            # Log the full response for debugging
            if response:
                self.log_fn(f"Full response length: {len(response)} chars")
        else:
            self.log_fn(f"Successfully extracted prediction: {prediction} (confidence: {confidence or 'calculated'}, score: {confidence_score})")

        metadata = {
            "confidence": confidence,
            "confidence_score": confidence_score,
            "reasoning": reasoning[:500] if reasoning else ""  # Truncate for brevity
        }

        return str(prediction), msg_history, metadata
