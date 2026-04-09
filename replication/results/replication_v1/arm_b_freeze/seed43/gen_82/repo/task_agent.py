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
    Also handles markdown code blocks with json tag.
    Includes advanced cleanup for common LLM output issues.
    """
    results = []
    search_from = 0
    
    # First try to find explicit <json> tags
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
            # Try to clean up common issues
            try:
                cleaned = _clean_json_string(inner)
                results.append(json.loads(cleaned))
            except json.JSONDecodeError:
                continue
    
    # Also try markdown code blocks
    if not results:
        search_from = 0
        while True:
            start = text.find("```json", search_from)
            if start == -1:
                break
            end = text.find("```", start + 7)
            if end == -1:
                break
            inner = text[start + 7:end].strip()
            search_from = end + 3
            try:
                results.append(json.loads(inner))
            except json.JSONDecodeError:
                try:
                    cleaned = _clean_json_string(inner)
                    results.append(json.loads(cleaned))
                except json.JSONDecodeError:
                    continue
    
    return results or None


def _clean_json_string(text: str) -> str:
    """Clean up common JSON formatting issues from LLM outputs.
    
    Handles:
    - Trailing commas before closing braces/brackets
    - Single quotes instead of double quotes
    - Unescaped newlines in strings
    - Comments (// and /* */)
    """
    # Remove trailing commas before closing braces/brackets
    cleaned = re.sub(r',\s*}', '}', text)
    cleaned = re.sub(r',\s*]', ']', cleaned)
    
    # Remove single-line comments
    cleaned = re.sub(r'//[^\n]*', '', cleaned)
    
    # Remove multi-line comments
    cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
    
    # Replace single quotes with double quotes (carefully)
    # This is a simplified approach - handles common cases
    cleaned = re.sub(r"(?<!\\)'", '"', cleaned)
    
    # Normalize whitespace
    cleaned = cleaned.strip()
    
    return cleaned


def _extract_any_json(text: str) -> list[dict] | None:
    """Fallback JSON extraction that looks for any JSON objects in text."""
    results = []
    # Try to find JSON objects between curly braces
    brace_count = 0
    start_idx = -1
    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                try:
                    obj = json.loads(text[start_idx:i+1])
                    results.append(obj)
                except json.JSONDecodeError:
                    pass
                start_idx = -1
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.confidence_threshold = 0.7  # Minimum confidence for auto-acceptance
        self.enable_confidence_scoring = True  # Feature flag for confidence scoring

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract task components for better prompting
        domain = inputs.get("domain", "Unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        # Check for empty student answer early
        if not student_answer or not student_answer.strip():
            return "Incorrect", [{"role": "system", "text": "Empty student answer detected, returning Incorrect"}]

        instruction = f"""You are an expert mathematical grader for International Mathematical Olympiad (IMO) problems.

Your task is to evaluate a student's answer to a mathematics problem and provide a grade with detailed reasoning.

DOMAIN: {domain}

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

Think step by step and provide a thorough analysis following this structure:

1. PROBLEM ANALYSIS: Identify the key mathematical concepts, theorems, and techniques required to solve this problem.

2. SOLUTION REVIEW: Analyze the official solution's approach, key steps, and expected answer format. Note what constitutes a complete and correct solution.

3. STUDENT WORK ANALYSIS: 
   - Identify what approach the student took
   - Note any correct steps, valid insights, or correct intermediate results
   - Identify errors, gaps, misconceptions, or missing steps
   - Check if the final answer matches the expected form

4. GRADING CRITERIA CHECK:
   - Systematically verify if the student met each criterion in the grading guidelines
   - Award partial credit for incomplete but valid reasoning
   - Note any specific point deductions for errors or omissions

5. FINAL DETERMINATION: Assign a clear grade based on:
   - Completeness of the solution
   - Mathematical correctness
   - Adherence to grading guidelines
   - Quality of reasoning shown

Respond ONLY in JSON format with the following exact schema:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis covering all 5 steps above. Be specific about what the student did right and wrong.",
    "response": "The final grade/prediction. Use exactly one of: 'Correct', 'Incorrect', 'Partial', or a numeric score if specified in guidelines."
}}
</json>

Important guidelines:
- Be objective, consistent, and thorough in your analysis
- Award partial credit when the student shows valid reasoning even if the final answer is incorrect
- If the student answer is empty, completely irrelevant, or shows no understanding, grade as 'Incorrect'
- If the student shows significant correct work but has minor errors, consider 'Partial'
- 'Correct' should only be used when the solution is essentially complete and correct
- 'Partial' is for solutions with significant correct work but notable gaps or errors
- 'Incorrect' is for solutions that are wrong, empty, or show fundamental misunderstanding
- Only output the JSON block, no additional text before or after
- Ensure your JSON is valid with no trailing commas
- Do not include markdown code blocks around the JSON"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with fallback mechanisms
        prediction = self._extract_prediction(msg_history)
        
        # Calculate confidence in the prediction
        last_message = msg_history[-1]["text"] if msg_history else ""
        confidence = self._calculate_confidence(last_message, prediction)
        
        # If confidence is below threshold, request a second opinion
        if self.enable_confidence_scoring and confidence < self.confidence_threshold and prediction != "None":
            self.log_fn(f"Low confidence ({confidence:.2f}) for grade '{prediction}'. Requesting second opinion...")
            prediction, confidence = self._request_second_opinion(inputs, prediction, confidence)
            self.log_fn(f"After second opinion: grade='{prediction}', confidence={confidence:.2f}")
        
        # Log confidence for monitoring
        if self.enable_confidence_scoring:
            self.log_fn(f"Final grade: '{prediction}' (confidence: {confidence:.2f})")

        return str(prediction), msg_history

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history with multiple fallback strategies.
        
        Args:
            msg_history: List of message dicts with 'role' and 'text' keys
            
        Returns:
            Extracted prediction string or "None" if extraction fails
        """
        if not msg_history:
            return "None"
        
        try:
            last_message = msg_history[-1]["text"]
            
            # Check if the response indicates an LLM error
            if last_message.startswith("Error:") or last_message.startswith("Error "):
                self.log_fn(f"LLM returned error response: {last_message[:100]}...")
                # Try to extract any grade information from the error message
                error_grade = self._extract_from_text(last_message)
                if error_grade != "None":
                    return error_grade
                return "None"
            
            # Try primary extraction method
            extracted = _extract_jsons(last_message)
            
            # Fallback to generic JSON extraction if primary fails
            if extracted is None:
                extracted = _extract_any_json(last_message)
            
            if not extracted:
                # Try to extract grade from plain text as last resort
                return self._extract_from_text(last_message)
            
            # Prefer response field, but accept other common field names
            last_json = extracted[-1]
            
            # Priority order for grade fields - expanded with more variations
            grade_fields = ["response", "grade", "answer", "result", "evaluation", 
                           "prediction", "score", "verdict", "assessment", "grading",
                           "final_grade", "final_answer", "final_result", "final_verdict",
                           "determination", "conclusion", "status", "outcome"]
            
            for field in grade_fields:
                if field in last_json:
                    field_value = last_json[field]
                    # Handle both string and numeric values
                    if isinstance(field_value, (str, int, float, bool)):
                        normalized = self._normalize_grade(str(field_value))
                        if normalized != "None":
                            return normalized
            
            # If no known field, use the first string or numeric value found
            for key, value in last_json.items():
                if isinstance(value, (str, int, float)) and key != "reasoning":
                    normalized = self._normalize_grade(str(value))
                    if normalized in ["Correct", "Incorrect", "Partial"]:
                        return normalized
            
            # Last resort: check if there's a value that looks like a grade
            for key, value in last_json.items():
                if isinstance(value, str):
                    normalized = self._normalize_grade(value)
                    if normalized in ["Correct", "Incorrect", "Partial"]:
                        return normalized
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
        
        return "None"
    
    def _normalize_grade(self, grade: str) -> str:
        """Normalize a grade string to standard format.
        
        Args:
            grade: Raw grade string from LLM
            
        Returns:
            Normalized grade: 'Correct', 'Incorrect', 'Partial', or 'None' if unrecognizable
        """
        if not grade or not isinstance(grade, str):
            return "None"
            
        grade_lower = grade.lower().strip()
        
        # Handle empty or whitespace-only strings
        if not grade_lower:
            return "None"
        
        # Map common variations to standard grades
        correct_variants = ["correct", "true", "right", "yes", "pass", "1", "100", 
                           "full", "full credit", "complete", "accurate", "valid"]
        incorrect_variants = ["incorrect", "false", "wrong", "no", "fail", "0", "0/100",
                           "none", "zero", "invalid", "error", "mistake"]
        partial_variants = ["partial", "partially correct", "partial credit", "incomplete", 
                           "0.5", "50", "half", "some credit", "partially"]
        
        if grade_lower in correct_variants:
            return "Correct"
        elif grade_lower in incorrect_variants:
            return "Incorrect"
        elif grade_lower in partial_variants:
            return "Partial"
        
        # Check for numeric scores that might indicate partial credit
        # Handle both 0-1 scale and 0-100 scale
        try:
            # Remove common suffixes/prefixes
            cleaned = grade_lower.replace("/100", "").replace("%", "").replace("points", "").strip()
            numeric = float(cleaned)
            
            # Determine scale based on magnitude
            if numeric > 10:  # Assume 0-100 scale
                if numeric >= 80:
                    return "Correct"
                elif numeric <= 20:
                    return "Incorrect"
                else:
                    return "Partial"
            else:  # Assume 0-1 scale
                if numeric >= 0.8:
                    return "Correct"
                elif numeric <= 0.2:
                    return "Incorrect"
                else:
                    return "Partial"
        except (ValueError, TypeError):
            pass
        
        # Check for grade embedded in other text
        if "correct" in grade_lower and "incorrect" not in grade_lower and "partial" not in grade_lower:
            return "Correct"
        elif "incorrect" in grade_lower or ("wrong" in grade_lower and "not wrong" not in grade_lower):
            return "Incorrect"
        elif "partial" in grade_lower:
            return "Partial"
        
        # Return "None" for unrecognizable grades instead of passing through
        return "None"
    
    def _extract_from_text(self, text: str) -> str:
        """Extract grade from plain text when JSON parsing fails.
        
        Args:
            text: Raw text to search for grade indicators
            
        Returns:
            Extracted grade or "None"
        """
        text_lower = text.lower()
        
        # Look for explicit grade statements with more comprehensive patterns
        grade_patterns = [
            (r'grade\s*[:=]\s*["\']?(correct|incorrect|partial|true|false|right|wrong)["\']?', 1),
            (r'final\s*(?:grade|determination|verdict)\s*[:=]\s*["\']?(correct|incorrect|partial|true|false|right|wrong)["\']?', 1),
            (r'response\s*[:=]\s*["\']?(correct|incorrect|partial|true|false|right|wrong)["\']?', 1),
            (r'(?:the\s+)?(?:answer|grade|result|verdict)\s+is\s+["\']?(correct|incorrect|partial|true|false|right|wrong)["\']?', 1),
            (r'["\']?(correct|incorrect|partial|true|false|right|wrong)["\']?\s*(?:grade|score|verdict|result)', 1),
            (r'(?:graded?\s+as|marked\s+as)\s+["\']?(correct|incorrect|partial|true|false|right|wrong)["\']?', 1),
            (r'(?:student|answer)\s+(?:is|was)\s+["\']?(correct|incorrect|partial|true|false|right|wrong)["\']?', 1),
            (r'(?:i\s+)?(?:would\s+)?(?:grade|rate|score|mark)\s+(?:this\s+)?(?:as\s+)?["\']?(correct|incorrect|partial|true|false|right|wrong)["\']?', 1),
            (r'(?:this\s+)?(?:answer|solution|response)\s+(?:is\s+)?(?:clearly\s+)?["\']?(correct|incorrect|partial|true|false|right|wrong)["\']?', 1),
            (r'(?:therefore|thus|hence),?\s*(?:the\s+)?(?:grade|verdict|result)\s+(?:is\s+)?["\']?(correct|incorrect|partial|true|false|right|wrong)["\']?', 1),
        ]
        
        for pattern, group in grade_patterns:
            match = re.search(pattern, text_lower)
            if match:
                grade = match.group(group).lower()
                normalized = self._normalize_grade(grade)
                if normalized in ["Correct", "Incorrect", "Partial"]:
                    return normalized
        
        # Count mentions as fallback with weighted scoring
        correct_count = len(re.findall(r'\bcorrect\b|\bright\b|\btrue\b|\bpass\b|\bvalid\b|\baccurate\b', text_lower))
        incorrect_count = len(re.findall(r'\bincorrect\b|\bwrong\b|\bfalse\b|\bfail\b|\binvalid\b|\berror\b', text_lower))
        partial_count = len(re.findall(r'\bpartial\b|\bincomplete\b|\bpartially\b|\bpartial\s+credit\b', text_lower))
        
        # Apply weights based on context - handle negations
        negation_patterns = [
            (r'\bnot\s+correct\b', 'correct', -3),
            (r'\bnot\s+right\b', 'correct', -3),
            (r'\bnot\s+true\b', 'correct', -3),
            (r'\bnot\s+incorrect\b', 'incorrect', -3),
            (r'\bnot\s+wrong\b', 'incorrect', -3),
            (r'\bnot\s+partial\b', 'partial', -2),
            (r'\bno\s+partial\b', 'partial', -2),
        ]
        
        for pattern, grade_type, penalty in negation_patterns:
            if re.search(pattern, text_lower):
                if grade_type == 'correct':
                    correct_count += penalty
                elif grade_type == 'incorrect':
                    incorrect_count += penalty
                elif grade_type == 'partial':
                    partial_count += penalty
        
        # Boost scores for explicit statements
        if re.search(r'\b(?:completely|entirely|totally)\s+(?:correct|right)\b', text_lower):
            correct_count += 2
        if re.search(r'\b(?:completely|entirely|totally)\s+(?:incorrect|wrong)\b', text_lower):
            incorrect_count += 2
        
        # Determine final grade based on weighted counts
        if incorrect_count > correct_count and incorrect_count > partial_count:
            return "Incorrect"
        elif partial_count > correct_count and partial_count > incorrect_count:
            return "Partial"
        elif correct_count > 0 and correct_count >= incorrect_count:
            return "Correct"
        
        return "None"
    
    def _calculate_confidence(self, text: str, extracted_grade: str) -> float:
        """Calculate confidence score for a grade extraction.
        
        Uses multiple signals to determine how confident we are in the grade:
        - Explicit grade statements (high confidence)
        - Multiple consistent indicators (medium-high confidence)
        - Single indicator or ambiguous text (low confidence)
        
        Args:
            text: The text from which the grade was extracted
            extracted_grade: The grade that was extracted (Correct, Incorrect, Partial, or None)
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not self.enable_confidence_scoring:
            return 1.0  # Return max confidence if feature is disabled
        
        if extracted_grade == "None":
            return 0.0  # No confidence in unrecognizable grades
        
        text_lower = text.lower()
        confidence = 0.5  # Base confidence
        
        # Boost confidence for explicit grade statements
        explicit_patterns = [
            r'grade\s*[:=]\s*["\']?(correct|incorrect|partial)["\']?',
            r'final\s*(?:grade|verdict)\s*[:=]\s*["\']?(correct|incorrect|partial)["\']?',
            r'(?:the\s+)?(?:answer|grade|result|verdict)\s+is\s+["\']?(correct|incorrect|partial)["\']?',
        ]
        
        for pattern in explicit_patterns:
            if re.search(pattern, text_lower):
                confidence += 0.3  # Strong signal
                break
        
        # Boost for multiple consistent indicators
        grade_keywords = {
            "Correct": [r'\bcorrect\b', r'\bright\b', r'\btrue\b', r'\bvalid\b', r'\baccurate\b'],
            "Incorrect": [r'\bincorrect\b', r'\bwrong\b', r'\bfalse\b', r'\binvalid\b', r'\berror\b'],
            "Partial": [r'\bpartial\b', r'\bincomplete\b', r'\bpartially\b'],
        }
        
        if extracted_grade in grade_keywords:
            matches = sum(1 for pattern in grade_keywords[extracted_grade] if re.search(pattern, text_lower))
            if matches >= 3:
                confidence += 0.15
            elif matches >= 2:
                confidence += 0.1
        
        # Penalize for contradictory indicators
        contradictory = False
        if extracted_grade == "Correct" and re.search(r'\bincorrect\b|\bwrong\b', text_lower):
            contradictory = True
        elif extracted_grade == "Incorrect" and re.search(r'\bcorrect\b|\bright\b(?!\s+away)', text_lower):
            contradictory = True
        
        if contradictory:
            confidence -= 0.2
        
        # Penalize for negations
        negation_patterns = [r'\bnot\s+', r'\bno\s+', r'\bnever\s+', r'\bhardly\s+']
        for pattern in negation_patterns:
            if re.search(pattern + extracted_grade.lower(), text_lower):
                confidence -= 0.15
                break
        
        # Boost for reasoning/explanation presence (indicates thoughtful analysis)
        if len(text) > 200 and ('because' in text_lower or 'since' in text_lower or 'therefore' in text_lower):
            confidence += 0.05
        
        # Clamp to [0, 1] range
        return max(0.0, min(1.0, confidence))
    
    def _request_second_opinion(self, inputs: dict, first_grade: str, first_confidence: float) -> tuple[str, float]:
        """Request a second LLM call when confidence in first grade is low.
        
        This implements a simple self-consistency check by re-querying the model
        with a slightly modified prompt to see if it reaches the same conclusion.
        
        Args:
            inputs: The original task inputs
            first_grade: The grade from the first LLM call
            first_confidence: Confidence score from first extraction
            
        Returns:
            Tuple of (final_grade, final_confidence)
        """
        # Build a focused prompt for second opinion
        domain = inputs.get("domain", "Unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        second_prompt = f"""You are an expert grader in {domain}. Please carefully re-evaluate this student answer.

Problem: {problem[:500]}{"..." if len(problem) > 500 else ""}

Correct Solution: {solution[:500]}{"..." if len(solution) > 500 else ""}

Grading Guidelines: {grading_guidelines[:300]}{"..." if len(grading_guidelines) > 300 else ""}

Student Answer: {student_answer[:800]}{"..." if len(student_answer) > 800 else ""}

Another grader initially assessed this as: {first_grade} (confidence: {first_confidence:.2f})

Please provide your independent assessment. First explain your reasoning, then provide your final grade in this exact format:
<json>{{"grade": "Correct" or "Incorrect" or "Partial", "confidence": "high/medium/low"}}</json>"""

        try:
            response_text, _, _ = get_response_from_llm(
                msg=second_prompt,
                model=self.model,
                temperature=0.2,  # Slightly higher temperature for variation
                msg_history=[],
            )
            
            # Extract grade from second opinion
            second_grade = self._extract_prediction({"response": response_text})
            second_confidence = self._calculate_confidence(response_text, second_grade)
            
            # Combine the two opinions
            if first_grade == second_grade:
                # Agreement - boost confidence
                combined_confidence = min(1.0, max(first_confidence, second_confidence) + 0.15)
                return first_grade, combined_confidence
            else:
                # Disagreement - use the one with higher confidence, but mark as uncertain
                if second_confidence > first_confidence:
                    return second_grade, second_confidence * 0.8  # Penalty for disagreement
                else:
                    return first_grade, first_confidence * 0.8  # Penalty for disagreement
                    
        except Exception as e:
            self.log_fn(f"Error in second opinion request: {e}")
            # Fall back to first grade if second opinion fails
            return first_grade, first_confidence * 0.7  # Reduced confidence due to failure
