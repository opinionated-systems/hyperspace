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
    """Extract JSON objects from <json>...</json> blocks."""
    if not text or not isinstance(text, str):
        return None
    
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
            parsed = json.loads(inner)
            if isinstance(parsed, dict):
                results.append(parsed)
        except json.JSONDecodeError:
            # Try with single quotes replaced
            try:
                parsed = json.loads(inner.replace("'", '"'))
                if isinstance(parsed, dict):
                    results.append(parsed)
            except json.JSONDecodeError:
                pass
    
    return results if results else None


def _extract_json_flexible(text: str) -> list[dict] | None:
    """Extract JSON objects using multiple strategies."""
    if not text or not isinstance(text, str):
        return None
        
    results = []
    
    def try_parse_json(json_str: str) -> dict | None:
        """Try to parse JSON with fallback strategies."""
        if not json_str:
            return None
        json_str = json_str.strip()
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # Try fixing single quotes
        try:
            return json.loads(json_str.replace("'", '"'))
        except json.JSONDecodeError:
            pass
        
        # Try fixing common JSON issues
        try:
            # Fix unquoted keys
            fixed = re.sub(r'(\w+):', r'"\1":', json_str)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        
        return None
    
    # Strategy 1: <json>...</json> tags
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
        parsed = try_parse_json(inner)
        if parsed and isinstance(parsed, dict):
            results.append(parsed)
    
    # Strategy 2: ```json...``` code blocks
    pattern = r'```json\s*(.*?)\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        parsed = try_parse_json(match)
        if parsed and isinstance(parsed, dict):
            results.append(parsed)
    
    # Strategy 3: ```...``` code blocks (without json label)
    pattern = r'```\s*(.*?)\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        parsed = try_parse_json(match)
        if parsed and isinstance(parsed, dict):
            results.append(parsed)
    
    # Strategy 4: Raw JSON objects with "response" field
    pattern = r'\{\s*"response"\s*:\s*"([^"]+)"\s*\}'
    matches = re.findall(pattern, text)
    for match in matches:
        results.append({"response": match})
    
    # Strategy 5: Raw JSON objects with 'response' field (single quotes)
    pattern = r"\{\s*'response'\s*:\s*'([^']+)'\s*\}"
    matches = re.findall(pattern, text)
    for match in matches:
        results.append({"response": match})
    
    # Strategy 6: Case-insensitive response extraction
    pattern = r'["\']?response["\']?\s*:\s*["\']?(correct|incorrect|partial|almost)["\']?'
    matches = re.findall(pattern, text, re.IGNORECASE)
    for match in matches:
        grade = match.lower()
        if grade == 'almost':
            grade = 'partial'
        results.append({"response": grade})
    
    # Strategy 7: Look for any JSON-like object containing response
    pattern = r'\{[^}]*"response"[^}]*\}'
    matches = re.findall(pattern, text)
    for match in matches:
        parsed = try_parse_json(match)
        if parsed and isinstance(parsed, dict) and "response" in parsed:
            results.append(parsed)
    
    return results if results else None


def _extract_response_direct(text: str) -> str | None:
    """Direct extraction of response value from text."""
    if not text or not isinstance(text, str):
        return None
        
    text_lower = text.lower().strip()
    valid_grades = ['correct', 'incorrect', 'partial', 'almost']
    
    # Check for quoted grades (various quote styles)
    quote_patterns = [
        ('"correct"', 'correct'), ("'correct'", 'correct'), ('`correct`', 'correct'),
        ('"incorrect"', 'incorrect'), ("'incorrect'", 'incorrect'), ('`incorrect`', 'incorrect'),
        ('"partial"', 'partial'), ("'partial'", 'partial'), ('`partial`', 'partial'),
        ('"almost"', 'partial'), ("'almost'", 'partial'), ('`almost`', 'partial'),
    ]
    
    for pattern, grade in quote_patterns:
        if pattern in text_lower:
            return grade
    
    # Check for grade at end of text (last non-empty line)
    lines = [l.strip() for l in text_lower.split('\n') if l.strip()]
    for line in reversed(lines):
        # Remove common trailing punctuation and braces
        line_clean = line.rstrip('}').rstrip(',').rstrip('.').rstrip(';').rstrip(':').rstrip('!').rstrip('?')
        line_clean = line_clean.rstrip('"').rstrip("'").rstrip('`').rstrip(')').rstrip(']').rstrip('}').strip()
        if line_clean in valid_grades:
            return 'partial' if line_clean == 'almost' else line_clean
        # Check if line ends with a grade
        for grade in valid_grades:
            if line_clean.endswith(' ' + grade) or line_clean.endswith(':' + grade) or line_clean == grade:
                return 'partial' if grade == 'almost' else grade
    
    # Check for grade at the beginning
    if lines:
        first_line = lines[0].strip()
        for grade in valid_grades:
            if first_line.startswith(grade + ' ') or first_line == grade:
                return 'partial' if grade == 'almost' else grade
    
    # Check for grade in parentheses or brackets
    for grade in valid_grades:
        if f'({grade})' in text_lower or f'[{grade}]' in text_lower or f'{{{grade}}}' in text_lower:
            return 'partial' if grade == 'almost' else grade
    
    return None


def _extract_response_llm_style(text: str) -> str | None:
    """Extract response using patterns common in LLM outputs."""
    if not text or not isinstance(text, str):
        return None
        
    text_lower = text.lower().strip()
    valid_grades_pattern = 'correct|incorrect|partial|almost'
    
    # Look for conclusion phrases (expanded list)
    conclusion_patterns = [
        rf'\bthe answer is\s*[:=]?\s*["\']?({valid_grades_pattern})["\']?',
        rf'\bfinal answer\s*[:=]+\s*["\']?({valid_grades_pattern})["\']?',
        rf'\bconclusion\s*[:=]+\s*["\']?({valid_grades_pattern})["\']?',
        rf'\bverdict\s*[:=]+\s*["\']?({valid_grades_pattern})["\']?',
        rf'\bgrade\s*[:=]+\s*["\']?({valid_grades_pattern})["\']?',
        rf'\bthe grade is\s*[:=]?\s*["\']?({valid_grades_pattern})["\']?',
        rf'\bi grade this\s*[:=]?\s*["\']?({valid_grades_pattern})["\']?',
        rf'\bthis is\s*[:=]?\s*["\']?({valid_grades_pattern})["\']?',
        rf'\bresult\s*[:=]+\s*["\']?({valid_grades_pattern})["\']?',
        rf'\bevaluation\s*[:=]+\s*["\']?({valid_grades_pattern})["\']?',
    ]
    
    for pattern in conclusion_patterns:
        match = re.search(pattern, text_lower)
        if match:
            result = match.group(1)
            return 'partial' if result == 'almost' else result
    
    # Look for grade in various delimiters
    bracket_patterns = [
        rf'\(({valid_grades_pattern})\)',
        rf'\[({valid_grades_pattern})\]',
        rf'\{{({valid_grades_pattern})\}}',
        rf'"({valid_grades_pattern})"',
        rf"'({valid_grades_pattern})'",
        rf'`({valid_grades_pattern})`',
    ]
    for pattern in bracket_patterns:
        match = re.search(pattern, text_lower)
        if match:
            result = match.group(1)
            return 'partial' if result == 'almost' else result
    
    # Look for standalone grades at the end of sentences or lines
    end_patterns = [
        rf'\b({valid_grades_pattern})[.!?]*\s*$',  # At end of text
        rf'\b({valid_grades_pattern})[.!?]*\s*\n',  # At end of line
    ]
    for pattern in end_patterns:
        match = re.search(pattern, text_lower)
        if match:
            result = match.group(1)
            return 'partial' if result == 'almost' else result
    
    return None


def _extract_response_robust(text: str) -> str | None:
    """Robust extraction using multiple pattern matching strategies."""
    if not text or not isinstance(text, str):
        return None
        
    text_lower = text.lower()
    valid_grades = ['correct', 'incorrect', 'partial', 'almost']
    
    # Priority 1: Look for <json> tags with response field
    try:
        json_results = _extract_jsons(text)
        if json_results:
            for item in json_results:
                if isinstance(item, dict) and "response" in item:
                    val = item["response"]
                    if isinstance(val, str) and val.lower() in valid_grades:
                        return 'partial' if val.lower() == 'almost' else val.lower()
    except Exception:
        pass
    
    # Priority 2: Look for JSON-like patterns with response field
    json_patterns = [
        r'"response"\s*:\s*"(correct|incorrect|partial|almost)"',
        r"'response'\s*:\s*'(correct|incorrect|partial|almost)'",
        r'response\s*:\s*"(correct|incorrect|partial|almost)"',
        r'response\s*:\s*(correct|incorrect|partial|almost)',
        r'{"response":\s*"(correct|incorrect|partial|almost)"}',
        r"{'response':\s*'(correct|incorrect|partial|almost)'}",
    ]
    
    for pattern in json_patterns:
        match = re.search(pattern, text_lower)
        if match:
            result = match.group(1)
            return 'partial' if result == 'almost' else result
    
    # Priority 3: Look for the grade as a standalone word at the end of lines
    lines = text_lower.split('\n')
    for line in reversed(lines):
        line_stripped = line.strip().rstrip('.,;:!?')
        for grade in valid_grades:
            if line_stripped == grade or re.search(rf'\b{grade}\b', line_stripped):
                if line_stripped.endswith(grade) or line_stripped == grade:
                    return 'partial' if grade == 'almost' else grade
    
    # Priority 4: Look for grade in quotes anywhere in text
    for grade in valid_grades:
        if re.search(rf'"{grade}"', text_lower) or re.search(rf"'{grade}'", text_lower) or re.search(rf'`{grade}`', text_lower):
            return 'partial' if grade == 'almost' else grade
    
    # Priority 5: Look for explicit decision patterns
    decision_patterns = [
        rf'\bgrade\s*[:=]\s*["\']?(correct|incorrect|partial|almost)["\']?',
        rf'\bverdict\s*[:=]\s*["\']?(correct|incorrect|partial|almost)["\']?',
        rf'\bfinal\s+(?:grade|verdict|answer)\s*[:=]\s*["\']?(correct|incorrect|partial|almost)["\']?',
        rf'\bthe\s+answer\s+is\s*[:=]?\s*["\']?(correct|incorrect|partial|almost)["\']?',
    ]
    
    for pattern in decision_patterns:
        match = re.search(pattern, text_lower)
        if match:
            result = match.group(1)
            return 'partial' if result == 'almost' else result
    
    # Priority 6: Look for grade in brackets or parentheses
    bracket_patterns = [
        rf'\((correct|incorrect|partial|almost)\)',
        rf'\[(correct|incorrect|partial|almost)\]',
        rf'\{{(correct|incorrect|partial|almost)\}}',
    ]
    for pattern in bracket_patterns:
        match = re.search(pattern, text_lower)
        if match:
            result = match.group(1)
            return 'partial' if result == 'almost' else result
    
    # Priority 7: Look for grade at the very end of the text
    text_stripped = text_lower.strip()
    for grade in valid_grades:
        if text_stripped.endswith(grade):
            return 'partial' if grade == 'almost' else grade
    
    # Priority 8: Simple word boundary search for grades
    for grade in valid_grades:
        if re.search(rf'\b{grade}\b', text_lower):
            return 'partial' if grade == 'almost' else grade
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with improved accuracy."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with chain-of-thought reasoning.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract fields from inputs for clearer prompting
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Parse grading guidelines to extract key evaluation criteria
        parsed_guidelines = self._parse_grading_guidelines(grading_guidelines)
        
        # First, do a reasoning step to analyze the answer
        reasoning_prompt = f"""You are an expert mathematics grader for International Mathematical Olympiad (IMO) problems.

Your task is to carefully analyze a student's answer and determine if it is correct, incorrect, or partial.

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

ANALYSIS INSTRUCTIONS:
1. First, identify what the problem is asking for and what constitutes a complete solution.
2. Compare the student's approach to the official solution - are they using valid methods?
3. Check for any logical gaps, errors, or unjustified claims in the student's work.
4. Look specifically for these indicators in the grading guidelines:
   - "(Almost)" or "almost correct" → indicates partial credit
   - "(Correct)" → indicates full credit
   - "(Incorrect)" or "(0 points)" → indicates no credit
   - "(Partial)" or partial points → indicates partial credit

5. Determine the grade based on these criteria:
   - CORRECT: Complete, valid proof with all steps justified and correct conclusion
   - INCORRECT: Fundamental errors, wrong approach, or conclusion that doesn't follow
   - PARTIAL: Valid insights, correct lemmas, good approach but incomplete or with minor errors

Provide your detailed analysis below, then state your final grade decision clearly."""

        # Get reasoning from the model
        reasoning_response, reasoning_history, _ = get_response_from_llm(
            msg=reasoning_prompt,
            model=self.model,
            msg_history=[],
        )
        
        # Now extract the final grade with a structured prompt
        extraction_prompt = f"""Based on the analysis above, you must now output ONLY the final grade.

Grade the student's answer as EXACTLY ONE of: "correct", "incorrect", or "partial".

DECISION RULES:
- Use "correct" ONLY if the answer is a complete, rigorous proof with no gaps
- Use "incorrect" if there are fundamental flaws, wrong conclusions, or the approach cannot work
- Use "partial" if there are good ideas, correct intermediate results, or the answer is "almost" correct per the guidelines

CRITICAL: Your response must be ONLY a JSON object in this EXACT format:
<json>{{"response": "GRADE"}}</json>

Where GRADE is exactly one of: "correct", "incorrect", or "partial"

No other text, no markdown, no explanation - ONLY the JSON object."""

        # Get the final structured response
        response, msg_history, info = get_response_from_llm(
            msg=extraction_prompt,
            model=self.model,
            msg_history=reasoning_history,
        )

        # Extract prediction from response using robust extraction
        prediction = self._extract_prediction(response, msg_history)
        
        return str(prediction), msg_history

    def _extract_prediction(self, response: str, msg_history: list[dict]) -> str:
        """Extract prediction from response with multiple fallback strategies."""
        response_text = ""
        
        # Get the response text from msg_history
        try:
            if msg_history and len(msg_history) > 0:
                last_msg = msg_history[-1]
                if isinstance(last_msg, dict):
                    response_text = last_msg.get("content", "") or last_msg.get("text", "") or str(last_msg)
                else:
                    response_text = str(last_msg)
            else:
                response_text = str(response) if isinstance(response, str) else ""
        except Exception as e:
            self.log_fn(f"Error getting response text: {e}")
            response_text = str(response) if response else ""
        
        # Ensure response_text is a string
        if not isinstance(response_text, str):
            response_text = str(response_text) if response_text else ""
        
        self.log_fn(f"Raw response text: {response_text[:500]}...")
        
        # Try multiple extraction strategies
        prediction = None
        
        # Strategy 1: Extract from <json> tags
        try:
            json_results = _extract_jsons(response_text)
            if json_results:
                for item in json_results:
                    if isinstance(item, dict) and "response" in item:
                        val = item["response"].lower().strip()
                        if val in ["correct", "incorrect", "partial"]:
                            prediction = val
                            self.log_fn(f"Extracted prediction via <json> tags: {prediction}")
                            break
        except Exception as e:
            self.log_fn(f"Error in <json> extraction: {e}")
        
        # Strategy 2: Look for explicit grade mentions
        if prediction is None:
            text_lower = response_text.lower()
            # Look for the grade in quotes or backticks
            for grade in ["correct", "incorrect", "partial"]:
                if f'"{grade}"' in text_lower or f"'{grade}'" in text_lower or f"`{grade}`" in text_lower:
                    # Verify it's not part of a larger word by checking context
                    prediction = grade
                    self.log_fn(f"Extracted prediction via quotes: {prediction}")
                    break
        
        # Strategy 3: Look for the grade as a standalone word at the end
        if prediction is None:
            text_lower = response_text.lower().strip()
            for grade in ["correct", "incorrect", "partial"]:
                if text_lower.endswith(grade) or f"grade: {grade}" in text_lower or f"is {grade}" in text_lower:
                    prediction = grade
                    self.log_fn(f"Extracted prediction via standalone word: {prediction}")
                    break
        
        # Strategy 4: Default to incorrect if no valid prediction found
        if prediction is None:
            self.log_fn(f"Could not extract valid prediction, defaulting to incorrect")
            prediction = "incorrect"
        
        return prediction

    def _parse_grading_guidelines(self, guidelines: str) -> str:
        
        # Try multiple extraction strategies in order of reliability
        
        # Strategy 1: Extract from <json> tags (new format we now request)
        try:
            extracted = _extract_jsons(response_text)
            if extracted and len(extracted) > 0:
                for item in extracted:
                    if isinstance(item, dict) and "response" in item:
                        val = item["response"]
                        if isinstance(val, str) and val.lower() in ["correct", "incorrect", "partial"]:
                            prediction = val.lower()
                            self.log_fn(f"Extracted prediction via <json> tags: {prediction}")
                            break
        except Exception as e:
            self.log_fn(f"Error in <json> tag extraction: {e}")
        
        # Strategy 2: Robust extraction (handles most formats)
        if prediction is None:
            try:
                result = _extract_response_robust(response_text)
                if result:
                    prediction = result
                    self.log_fn(f"Extracted prediction via robust extraction: {prediction}")
            except Exception as e:
                self.log_fn(f"Error in robust extraction: {e}")
        
        # Strategy 3: Flexible JSON extraction
        if prediction is None:
            try:
                extracted = _extract_json_flexible(response_text)
                if extracted:
                    for item in extracted:
                        if isinstance(item, dict) and "response" in item:
                            val = item["response"]
                            if isinstance(val, str) and val.lower() in ["correct", "incorrect", "partial"]:
                                prediction = val.lower()
                                self.log_fn(f"Extracted prediction via JSON: {prediction}")
                                break
            except Exception as e:
                self.log_fn(f"Error in flexible JSON extraction: {e}")
        
        # Strategy 4: Direct response extraction
        if prediction is None:
            try:
                direct = _extract_response_direct(response_text)
                if direct:
                    prediction = direct
                    self.log_fn(f"Extracted prediction via direct extraction: {prediction}")
            except Exception as e:
                self.log_fn(f"Error in direct extraction: {e}")
        
        # Strategy 5: LLM-style extraction (handles common LLM output patterns)
        if prediction is None:
            try:
                llm_result = _extract_response_llm_style(response_text)
                if llm_result:
                    prediction = llm_result
                    self.log_fn(f"Extracted prediction via LLM-style extraction: {prediction}")
            except Exception as e:
                self.log_fn(f"Error in LLM-style extraction: {e}")
        
        # Strategy 6: Last resort - simple text search for grades
        if prediction is None:
            try:
                text_lower = response_text.lower()
                # Check for grades in order of preference
                if '"correct"' in text_lower or "'correct'" in text_lower or '`correct`' in text_lower:
                    prediction = "correct"
                    self.log_fn(f"Extracted prediction via simple text search: {prediction}")
                elif '"partial"' in text_lower or "'partial'" in text_lower or '`partial`' in text_lower:
                    prediction = "partial"
                    self.log_fn(f"Extracted prediction via simple text search: {prediction}")
                elif '"incorrect"' in text_lower or "'incorrect'" in text_lower or '`incorrect`' in text_lower:
                    prediction = "incorrect"
                    self.log_fn(f"Extracted prediction via simple text search: {prediction}")
                elif 'correct' in text_lower and 'incorrect' not in text_lower and 'partial' not in text_lower:
                    prediction = "correct"
                    self.log_fn(f"Extracted prediction via simple text search: {prediction}")
                elif 'partial' in text_lower and 'incorrect' not in text_lower:
                    prediction = "partial"
                    self.log_fn(f"Extracted prediction via simple text search: {prediction}")
                elif 'incorrect' in text_lower:
                    prediction = "incorrect"
                    self.log_fn(f"Extracted prediction via simple text search: {prediction}")
            except Exception as e:
                self.log_fn(f"Error in simple text search: {e}")
        
        # Validate the prediction
        if prediction not in ["correct", "incorrect", "partial"]:
            self.log_fn(f"Invalid or missing prediction: '{prediction}', defaulting to incorrect")
            prediction = "incorrect"
        
        return str(prediction), msg_history

    def _parse_grading_guidelines(self, guidelines: str) -> str:
        """Parse grading guidelines to extract key evaluation criteria.
        
        This method analyzes the grading guidelines to identify specific
        evaluation points that should be checked when grading the answer.
        
        Args:
            guidelines: The grading guidelines text
            
        Returns:
            A formatted string of specific evaluation points
        """
        if not guidelines:
            return "No specific guidelines provided. Use general mathematical reasoning."
        
        points = []
        almost_indicators = []  # Special handling for "almost" markers
        correct_indicators = []  # Special handling for "correct" markers
        incorrect_indicators = []  # Special handling for "incorrect" markers
        lines = guidelines.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            line_lower = line.lower()
            
            # Check for "almost" indicators first (highest priority)
            if '(almost)' in line_lower or 'almost correct' in line_lower or 'almost complete' in line_lower:
                almost_indicators.append(f"[GRADE AS PARTIAL] {line}")
                continue
            
            # Check for explicit correct indicators
            if '(correct)' in line_lower or line_lower.startswith('correct:') or line_lower.startswith('correct '):
                correct_indicators.append(f"[GRADE AS CORRECT] {line}")
                continue
            
            # Check for explicit incorrect indicators
            if '(incorrect)' in line_lower or line_lower.startswith('incorrect:') or line_lower.startswith('incorrect '):
                incorrect_indicators.append(f"[GRADE AS INCORRECT] {line}")
                continue
            
            # Check for partial indicators
            if '(partial)' in line_lower or line_lower.startswith('partial:') or line_lower.startswith('partial '):
                almost_indicators.append(f"[GRADE AS PARTIAL] {line}")
                continue
            
            # Look for numbered points or bullet points
            if line[0].isdigit() and ('.' in line[:3] or ')' in line[:3]):
                points.append(line)
            elif line.startswith('-') or line.startswith('*') or line.startswith('•'):
                points.append(line)
            # Look for key phrases indicating evaluation criteria
            elif any(keyword in line_lower for keyword in 
                    ['must', 'should', 'required', 'check', 'verify', 'ensure', 
                     'point', 'criterion', 'score', 'credit']):
                points.append(line)
        
        # Combine results: explicit indicators first, then other points
        all_points = almost_indicators + correct_indicators + incorrect_indicators + points
        
        # If we found structured points, format them
        if all_points:
            return '\n'.join(f"- {p}" for p in all_points[:15])  # Limit to first 15 points
        
        # Otherwise, return the first few non-empty lines
        non_empty = [l.strip() for l in lines if l.strip()][:7]
        if non_empty:
            return '\n'.join(f"- {line}" for line in non_empty)
        
        return guidelines[:600]  # Fallback: return truncated guidelines
