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
        except json.JSONDecodeError as e:
            logger.debug(f"JSON decode error in <json> block: {e}")
            continue
    return results or None


def _extract_json_from_markdown(text: str) -> list[dict] | None:
    """Extract JSON objects from markdown code blocks.

    Fallback for when <json> tags are not used but markdown code blocks are.
    """
    results = []
    # Match ```json ... ``` or just ``` ... ``` blocks
    pattern = r'```(?:json)?\s*(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
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
        'full marks', 'complete', 'valid', 'accepted', 'pass'
    ]
    incorrect_variations = [
        'incorrect', 'wrong', 'false', 'no', 'none', 'zero',
        'invalid', 'rejected', 'fail', 'error', 'mistake'
    ]
    partial_variations = [
        'partial', 'partial credit', 'half', 'incomplete',
        'partially correct', 'partially right', 'some credit'
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
    
    # Return original if no normalization applied
    return grade.strip()


def _extract_confidence(text: str) -> float | None:
    """Extract confidence score from text.
    
    Looks for confidence indicators in the response, such as:
    - Explicit confidence scores (e.g., "confidence: 0.95")
    - Certainty phrases (e.g., "very confident", "uncertain")
    - Returns a value between 0.0 and 1.0
    """
    # Look for explicit confidence scores
    confidence_patterns = [
        r'confidence\s*[:=]\s*([0-9.]+)',
        r'confidence score\s*[:=]\s*([0-9.]+)',
        r'certainty\s*[:=]\s*([0-9.]+)',
    ]
    
    for pattern in confidence_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                score = float(match.group(1))
                # Normalize to 0-1 range if needed
                if score > 1.0 and score <= 100:
                    score = score / 100.0
                return max(0.0, min(1.0, score))
            except ValueError:
                continue
    
    # Look for certainty phrases
    high_confidence_phrases = [
        'very confident', 'highly confident', 'certain', 'definitely',
        'clearly', 'obviously', 'without doubt'
    ]
    medium_confidence_phrases = [
        'somewhat confident', 'fairly confident', 'reasonably sure',
        'likely', 'probably', 'appears to be'
    ]
    low_confidence_phrases = [
        'uncertain', 'not sure', 'unclear', 'ambiguous', 'difficult to determine',
        'might be', 'could be', 'possibly'
    ]
    
    text_lower = text.lower()
    
    if any(phrase in text_lower for phrase in high_confidence_phrases):
        return 0.9
    elif any(phrase in text_lower for phrase in medium_confidence_phrases):
        return 0.6
    elif any(phrase in text_lower for phrase in low_confidence_phrases):
        return 0.3
    
    return None


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

Your task is to evaluate a student's answer to a mathematics problem and assign a grade with a confidence score.

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

Also provide a confidence score (0.0 to 1.0) indicating how certain you are in your evaluation.

## Response Format:

You MUST respond in JSON format wrapped in <json> tags:

<json>
{{
    "reasoning": "Your detailed step-by-step analysis following the Evaluation Framework above",
    "response": "The final grade you assign (e.g., '0', '1', '2', 'Correct', 'Incorrect', 'Partial')",
    "confidence": 0.95
}}
</json>

Important: The "response" field must contain ONLY the grade value, nothing else."""

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

    def _get_confidence_from_json(self, json_obj: dict) -> float | None:
        """Extract confidence score from JSON object."""
        # Look for confidence field
        if "confidence" in json_obj:
            value = json_obj["confidence"]
            if isinstance(value, (int, float)):
                return max(0.0, min(1.0, float(value)))
        return None

    def _extract_grade_from_text(self, text: str) -> str:
        """Extract grade from plain text using pattern matching.
        
        Enhanced with additional patterns for robust extraction from
        various response formats including IMO-style numeric grades.
        """
        # Look for explicit grade statements with flexible patterns
        patterns = [
            # Standard grade assignments
            r'grade[\s]*[:=][\s]*["\']?([^"\'\n]+)["\']?',
            r'response[\s]*[:=][\s]*["\']?([^"\'\n]+)["\']?',
            r'final grade[\s]*[:=][\s]*["\']?([^"\'\n]+)["\']?',
            r'assign[\s]+["\']?([^"\'\n]+)["\']?',
            # IMO-style grade statements
            r'(?:the\s+)?(?:student\s+(?:receives?|gets?|earns?)|I\s+(?:assign|give|award))\s+["\']?(\d+(?:\.\d+)?)["\']?\s*(?:points?|marks?)?',
            r'(?:score|grade|mark)[\s]*[:=\s]+["\']?(\d+(?:\.\d+)?)["\']?',
            # Grade at end of sentence
            r'(?:grade|score|mark|result)[\s]+(?:is|of)[\s]+["\']?([^"\'\n.]+)["\']?',
            # Standalone grades in common formats
            r'\b([0-2](?:\.0|\.5)?)\b',
            r'\b(Correct|Incorrect|Partial|Right|Wrong)\b',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return _normalize_grade(match.group(1).strip())
        
        # Fallback: look for numeric grades (0, 1, 2) as standalone tokens
        # This handles cases where the model just outputs a number
        numeric_matches = re.findall(r'\b([0-2])\b', text)
        if numeric_matches:
            # Return the last numeric match (often the final grade)
            return _normalize_grade(numeric_matches[-1])
        
        return "None"

    def _extract_prediction_with_confidence(self, msg_history: list[dict]) -> tuple[str, float | None]:
        """Extract both prediction and confidence from message history.
        
        Returns:
            (prediction, confidence) where confidence is None if not found
        """
        if not msg_history:
            return "None", None
        
        last_message = msg_history[-1].get("text", "")
        
        # Strategy 1: Extract from <json> tags
        extracted = _extract_jsons(last_message)
        if extracted:
            json_obj = extracted[-1]
            grade = self._get_grade_from_json(json_obj)
            confidence = self._get_confidence_from_json(json_obj)
            if confidence is None:
                confidence = _extract_confidence(last_message)
            return grade, confidence
        
        # Strategy 2: Extract from markdown code blocks
        extracted = _extract_json_from_markdown(last_message)
        if extracted:
            json_obj = extracted[-1]
            grade = self._get_grade_from_json(json_obj)
            confidence = self._get_confidence_from_json(json_obj)
            if confidence is None:
                confidence = _extract_confidence(last_message)
            return grade, confidence
        
        # Strategy 3: Look for grade patterns in plain text
        grade = self._extract_grade_from_text(last_message)
        confidence = _extract_confidence(last_message)
        return grade, confidence

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with enhanced error handling.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_grading_prompt(inputs)

        # Retry mechanism for robustness
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
            except Exception as e:
                self.log_fn(f"LLM call failed (attempt {attempt + 1}): {e}")
                if attempt == max_retries:
                    return "None", []
                continue

            # Extract prediction using multiple strategies
            prediction = self._extract_prediction(msg_history)
            
            if prediction != "None":
                return str(prediction), msg_history
            
            if attempt == max_retries:
                self.log_fn(f"Failed to extract prediction after {max_retries + 1} attempts")
                return "None", msg_history
            
            self.log_fn(f"Retrying extraction (attempt {attempt + 1})")

        return "None", []

    def forward_with_confidence(self, inputs: dict) -> tuple[str, float | None, list[dict]]:
        """Run the task agent and return prediction with confidence score.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, confidence, msg_history) where confidence is 0.0-1.0 or None
        """
        instruction = self._build_grading_prompt(inputs)

        # Retry mechanism for robustness
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
            except Exception as e:
                self.log_fn(f"LLM call failed (attempt {attempt + 1}): {e}")
                if attempt == max_retries:
                    return "None", None, []
                continue

            # Extract prediction and confidence using multiple strategies
            prediction, confidence = self._extract_prediction_with_confidence(msg_history)
            
            if prediction != "None":
                return str(prediction), confidence, msg_history
            
            if attempt == max_retries:
                self.log_fn(f"Failed to extract prediction after {max_retries + 1} attempts")
                return "None", None, msg_history
            
            self.log_fn(f"Retrying extraction (attempt {attempt + 1})")

        return "None", None, []

    def forward_batch(self, inputs_list: list[dict]) -> list[tuple[str, list[dict]]]:
        """Process multiple grading tasks in batch.

        Args:
            inputs_list: List of input dicts, each with domain, problem, 
                        solution, grading_guidelines, student_answer

        Returns:
            List of (prediction, msg_history) tuples
        """
        results = []
        for inputs in inputs_list:
            prediction, msg_history = self.forward(inputs)
            results.append((prediction, msg_history))
        return results

    def forward_batch_with_confidence(self, inputs_list: list[dict]) -> list[tuple[str, float | None, list[dict]]]:
        """Process multiple grading tasks in batch with confidence scores.

        Args:
            inputs_list: List of input dicts, each with domain, problem,
                        solution, grading_guidelines, student_answer

        Returns:
            List of (prediction, confidence, msg_history) tuples
        """
        results = []
        for inputs in inputs_list:
            prediction, confidence, msg_history = self.forward_with_confidence(inputs)
            results.append((prediction, confidence, msg_history))
        return results

    def forward_with_feedback(self, inputs: dict) -> tuple[str, float | None, dict, list[dict]]:
        """Run the task agent and return prediction with confidence and detailed feedback.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, confidence, feedback_dict, msg_history) where:
            - prediction: the grade assigned
            - confidence: confidence score 0.0-1.0 or None
            - feedback_dict: dict with keys 'strengths', 'weaknesses', 'suggestions', 'reasoning'
            - msg_history: the conversation history
        """
        instruction = self._build_grading_prompt_with_feedback(inputs)

        # Retry mechanism for robustness
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
            except Exception as e:
                self.log_fn(f"LLM call failed (attempt {attempt + 1}): {e}")
                if attempt == max_retries:
                    return "None", None, {}, []
                continue

            # Extract prediction, confidence, and feedback
            prediction, confidence, feedback = self._extract_prediction_with_feedback(msg_history)
            
            if prediction != "None":
                return str(prediction), confidence, feedback, msg_history
            
            if attempt == max_retries:
                self.log_fn(f"Failed to extract prediction after {max_retries + 1} attempts")
                return "None", None, {}, msg_history
            
            self.log_fn(f"Retrying extraction (attempt {attempt + 1})")

        return "None", None, {}, []

    def _build_grading_prompt_with_feedback(self, inputs: dict) -> str:
        """Build a structured grading prompt that requests detailed feedback."""
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        return f"""You are an expert mathematical grader specializing in {domain} problems.

Your task is to evaluate a student's answer to a mathematics problem and assign a grade with a confidence score and detailed feedback.

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

Also provide a confidence score (0.0 to 1.0) indicating how certain you are in your evaluation.

## Response Format:

You MUST respond in JSON format wrapped in <json> tags:

<json>
{{
    "reasoning": "Your detailed step-by-step analysis following the Evaluation Framework above",
    "response": "The final grade you assign (e.g., '0', '1', '2', 'Correct', 'Incorrect', 'Partial')",
    "confidence": 0.95,
    "feedback": {{
        "strengths": ["List of what the student did well"],
        "weaknesses": ["List of areas where the student could improve"],
        "suggestions": ["Specific actionable suggestions for improvement"]
    }}
}}
</json>

Important: The "response" field must contain ONLY the grade value, nothing else."""

    def _extract_prediction_with_feedback(self, msg_history: list[dict]) -> tuple[str, float | None, dict]:
        """Extract prediction, confidence, and feedback from message history."""
        if not msg_history:
            return "None", None, {}
        
        last_message = msg_history[-1].get("text", "")
        
        # Strategy 1: Extract from <json> tags
        extracted = _extract_jsons(last_message)
        if extracted:
            json_obj = extracted[-1]
            grade = self._get_grade_from_json(json_obj)
            confidence = self._get_confidence_from_json(json_obj)
            feedback = self._get_feedback_from_json(json_obj)
            if confidence is None:
                confidence = _extract_confidence(last_message)
            return grade, confidence, feedback
        
        # Strategy 2: Extract from markdown code blocks
        extracted = _extract_json_from_markdown(last_message)
        if extracted:
            json_obj = extracted[-1]
            grade = self._get_grade_from_json(json_obj)
            confidence = self._get_confidence_from_json(json_obj)
            feedback = self._get_feedback_from_json(json_obj)
            if confidence is None:
                confidence = _extract_confidence(last_message)
            return grade, confidence, feedback
        
        # Strategy 3: Look for grade patterns in plain text
        grade = self._extract_grade_from_text(last_message)
        confidence = _extract_confidence(last_message)
        return grade, confidence, {}

    def _get_confidence_from_json(self, json_obj: dict) -> float | None:
        """Extract confidence score from JSON object."""
        if "confidence" in json_obj:
            value = json_obj["confidence"]
            if isinstance(value, (int, float)):
                return max(0.0, min(1.0, float(value)))
        return None

    def _get_feedback_from_json(self, json_obj: dict) -> dict:
        """Extract feedback dict from JSON object."""
        feedback = json_obj.get("feedback", {})
        if isinstance(feedback, dict):
            return {
                "strengths": feedback.get("strengths", []),
                "weaknesses": feedback.get("weaknesses", []),
                "suggestions": feedback.get("suggestions", []),
                "reasoning": json_obj.get("reasoning", "")
            }
        return {
            "strengths": [],
            "weaknesses": [],
            "suggestions": [],
            "reasoning": json_obj.get("reasoning", "")
        }
