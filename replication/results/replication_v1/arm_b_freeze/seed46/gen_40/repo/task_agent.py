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


def _try_parse_json(text: str) -> dict | None:
    """Try to parse JSON with various cleanup strategies.
    
    Returns the parsed dict or None if all strategies fail.
    """
    strategies = [
        # Strategy 1: Direct parse
        lambda t: t,
        # Strategy 2: Remove trailing commas
        lambda t: re.sub(r',(\s*[}\]])', r'\1', t),
        # Strategy 3: Fix single quotes
        lambda t: t.replace("'", '"'),
        # Strategy 4: Remove comments
        lambda t: re.sub(r'/\*.*?\*/', '', re.sub(r'//.*?\n', '\n', t), flags=re.DOTALL),
    ]
    
    for strategy in strategies:
        try:
            return json.loads(strategy(text))
        except json.JSONDecodeError:
            continue
    return None


def _extract_first_json_object(text: str) -> dict | None:
    """Extract the first valid JSON object from text using brace counting."""
    brace_start = text.find('{')
    if brace_start == -1:
        return None
    
    brace_count = 0
    for i, char in enumerate(text[brace_start:]):
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                candidate = text[brace_start:brace_start+i+1]
                parsed = _try_parse_json(candidate)
                if parsed is not None:
                    return parsed
                break
    return None


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles nested JSON objects and common formatting issues.
    """
    results = []
    search_from = 0
    
    while True:
        # Find opening tag (try both cases)
        start = text.find("<json>", search_from)
        if start == -1:
            start = text.find("<JSON>", search_from)
        if start == -1:
            break
            
        # Find closing tag (try both cases)
        tag_end = start + 6
        end = text.find("</json>", tag_end)
        if end == -1:
            end = text.find("</JSON>", tag_end)
        if end == -1:
            break
            
        inner = text[tag_end:end].strip()
        search_from = end + 7
        
        # Try direct parsing first
        parsed = _try_parse_json(inner)
        if parsed is not None:
            results.append(parsed)
            continue
            
        # Fall back to extracting first JSON object via brace counting
        parsed = _extract_first_json_object(inner)
        if parsed is not None:
            results.append(parsed)
    
    return results or None


def _extract_json_from_markdown(text: str) -> list[dict] | None:
    """Extract JSON objects from markdown code blocks.

    Fallback for when <json> tags are not used but markdown code blocks are.
    Enhanced to handle nested braces and common formatting issues.
    """
    results = []
    # Match ```json ... ``` or just ``` ... ``` blocks (non-greedy but with proper handling)
    # Use a more robust pattern that handles nested content better
    pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        match = match.strip()
        if not match:
            continue
        try:
            results.append(json.loads(match))
        except json.JSONDecodeError:
            # Try common fixes
            try:
                # Remove trailing commas before closing braces/brackets
                fixed = re.sub(r',(\s*[}\]])', r'\1', match)
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                # Try to extract the first valid JSON object using brace counting
                try:
                    brace_start = match.find('{')
                    if brace_start != -1:
                        brace_count = 0
                        for i, char in enumerate(match[brace_start:]):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    results.append(json.loads(match[brace_start:brace_start+i+1]))
                                    break
                except json.JSONDecodeError:
                    # Try fixing single quotes
                    try:
                        fixed = match.replace("'", '"')
                        results.append(json.loads(fixed))
                    except json.JSONDecodeError:
                        # Try extracting array if object fails
                        try:
                            bracket_start = match.find('[')
                            if bracket_start != -1:
                                bracket_count = 0
                                for i, char in enumerate(match[bracket_start:]):
                                    if char == '[':
                                        bracket_count += 1
                                    elif char == ']':
                                        bracket_count -= 1
                                        if bracket_count == 0:
                                            results.append(json.loads(match[bracket_start:bracket_start+i+1]))
                                            break
                        except json.JSONDecodeError:
                            # Try removing comments
                            try:
                                cleaned = re.sub(r'//.*?\n', '\n', match)
                                cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
                                results.append(json.loads(cleaned))
                            except json.JSONDecodeError:
                                continue
    return results or None


def _normalize_grade(grade: str) -> str:
    """Normalize grade to a standard format.

    Handles various grade formats and normalizes them.
    Enhanced to handle numeric grades (IMO-style 0-7 scale) and more variations.
    """
    if not isinstance(grade, str):
        grade = str(grade)
    
    original = grade.strip()
    # Remove surrounding quotes and common punctuation
    original = original.strip('"\'.,;:')
    grade = original.lower()
    
    # First, check for numeric grades (0, 1, 2, etc.)
    # IMO-style grading: 0-7 scale where 7 is full marks
    try:
        numeric_grade = float(grade)
        # IMO scale: 0 = no progress, 1-6 = partial progress, 7 = full solution
        # For binary classification: 0 = Incorrect, 1-7 = Correct
        # For ternary: 0 = Incorrect, 1-6 = Partial, 7 = Correct
        if numeric_grade == 0:
            return 'Incorrect'
        elif numeric_grade >= 6:  # 6 or 7 on IMO scale = essentially correct
            return 'Correct'
        else:  # 1-5 = partial credit
            return 'Partial'
    except ValueError:
        pass
    
    # Map common variations to standard formats
    correct_variations = [
        'correct', 'right', 'true', 'yes', 'full', 'full credit', 
        'full marks', 'complete', 'valid', 'accepted', 'pass', 'solved',
        'success', 'accurate', 'perfect', 'excellent', 'good', '7', '6',
        'fully correct', 'entirely correct', 'completely correct',
        'correct answer', 'correct solution', 'is correct'
    ]
    incorrect_variations = [
        'incorrect', 'wrong', 'false', 'no', 'none', 'zero', '0',
        'invalid', 'rejected', 'fail', 'error', 'mistake', 'unsolved',
        'failure', 'bad', 'unacceptable', 'unsatisfactory',
        'completely wrong', 'entirely wrong', 'totally wrong',
        'is incorrect', 'not correct', 'not right'
    ]
    partial_variations = [
        'partial', 'partial credit', 'half', 'incomplete',
        'partially correct', 'partially right', 'some credit',
        'partially solved', 'almost', 'nearly', 'minor errors',
        '1', '2', '3', '4', '5', 'partially', 'somewhat',
        'mostly correct', 'mostly right', 'incomplete solution',
        'partial solution', 'partial answer'
    ]
    
    # Check for exact matches first, then substring matches
    if grade in correct_variations:
        return 'Correct'
    if grade in incorrect_variations:
        return 'Incorrect'
    if grade in partial_variations:
        return 'Partial'
    
    # Substring matching for more flexibility (but be careful not to over-match)
    # Check for strong indicators at word boundaries
    import re
    
    # Check for "correct" as a standalone word (but not "incorrect")
    if re.search(r'\bcorrect\b', grade) and not re.search(r'\b(in|not|partially|mostly)\s+correct\b', grade):
        return 'Correct'
    
    # Check for "incorrect" or "wrong"
    if re.search(r'\b(incorrect|wrong|false)\b', grade):
        return 'Incorrect'
    
    # Check for partial indicators
    if re.search(r'\b(partial|incomplete|partially|mostly|almost|nearly|some)\b', grade):
        return 'Partial'
    
    # Return original if no normalization applied (preserve case)
    return original


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

Follow this structured evaluation process carefully:

### Step 1: Problem Understanding
- What is the problem asking for? Identify the core question.
- What are the key constraints, conditions, and given information?
- What is the expected answer format (numeric, proof, construction, etc.)?

### Step 2: Solution Analysis
- What is the correct approach according to the official solution?
- What are the critical steps, theorems, or techniques that must be present?
- What constitutes a complete solution vs. an incomplete one?
- Are there multiple valid approaches? If so, does the student's approach align with any valid method?

### Step 3: Student Answer Evaluation
- Did the student understand the problem correctly?
- What approach did the student take? Is it mathematically sound?
- Are the student's steps logically valid and well-reasoned?
- Did the student show sufficient work and reasoning, or just state answers?
- Is the final answer mathematically correct?
- Did the student make any computational errors, conceptual errors, or logical gaps?

### Step 4: Grade Assignment
Based on the grading guidelines, assign the appropriate grade considering:
- Correctness of the final answer (most important)
- Validity and clarity of the reasoning process
- Completeness of the solution (all required steps present)
- Adherence to mathematical rigor and notation

## Grade Definitions (STRICT - USE ONLY THESE):

- **Correct**: The student's answer is completely correct with valid reasoning, correct final answer, and complete solution. Minor notation issues that don't affect correctness are acceptable.
- **Incorrect**: The student's answer is completely wrong, has no valid reasoning, the final answer is incorrect, or shows no meaningful progress toward the solution.
- **Partial**: The student's answer has some valid elements (correct approach, partial progress, or correct answer with flawed reasoning) but is incomplete, has significant errors, or lacks proper justification.

For IMO-style problems with numeric grades (0-7 scale):
- **0**: No progress or completely wrong → Grade as "Incorrect"
- **1-5**: Some progress but incomplete, has errors, or incorrect final answer → Grade as "Partial"
- **6-7**: Correct or nearly correct solution with valid reasoning → Grade as "Correct"

## Response Format (CRITICAL - FOLLOW EXACTLY):

You MUST respond in JSON format wrapped in <json> tags. The response field must contain ONLY one of: "Correct", "Incorrect", or "Partial".

<json>
{{
    "reasoning": "Your detailed step-by-step analysis following the Evaluation Framework above. Be specific about what the student did right/wrong.",
    "response": "Correct"
}}
</json>

OR

<json>
{{
    "reasoning": "Your detailed step-by-step analysis following the Evaluation Framework above. Be specific about what the student did right/wrong.",
    "response": "Incorrect"
}}
</json>

OR

<json>
{{
    "reasoning": "Your detailed step-by-step analysis following the Evaluation Framework above. Be specific about what the student did right/wrong.",
    "response": "Partial"
}}
</json>

CRITICAL RULES - FOLLOW EXACTLY:
1. The "response" field must contain ONLY one of these exact values: "Correct", "Incorrect", or "Partial" (case-sensitive)
2. Do NOT include explanations, quotes, or additional text in the response field
3. Do NOT use numeric grades (0, 1, 2, etc.) in the response field - use ONLY the text labels above
4. The reasoning field should contain your full analysis with specific details
5. Always wrap your entire response in <json>...</json> tags
6. Ensure the JSON is valid - no trailing commas, proper quotes, etc."""

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history with multiple fallback strategies."""
        if not msg_history:
            return "None"
        
        last_message = msg_history[-1].get("text", "")
        if not last_message:
            return "None"
        
        # Strategy 1: Extract from <json> tags (most reliable)
        extracted = _extract_jsons(last_message)
        if extracted:
            result = self._get_grade_from_json(extracted[-1])
            if result != "None":
                return result
        
        # Strategy 2: Extract from markdown code blocks
        extracted = _extract_json_from_markdown(last_message)
        if extracted:
            result = self._get_grade_from_json(extracted[-1])
            if result != "None":
                return result
        
        # Strategy 3: Look for grade patterns in plain text
        result = self._extract_grade_from_text(last_message)
        if result != "None":
            return result
        
        # Strategy 4: Look for explicit grade statements with more patterns
        explicit_patterns = [
            r'["\']?(?:the\s+)?(?:final\s+)?grade\s*(?:is|was|should\s+be)\s*["\']?\s*(Correct|Incorrect|Partial)\s*["\']?',
            r'["\']?(?:the\s+)?(?:final\s+)?response\s*(?:is|was|should\s+be)\s*["\']?\s*(Correct|Incorrect|Partial)\s*["\']?',
            r'["\']?(?:the\s+)?(?:student\s+)?(?:answer|solution)\s*(?:is|was)\s*["\']?\s*(Correct|Incorrect|Partial)\s*["\']?',
            r'\b(Correct|Incorrect|Partial)\b\s*(?:grade|assessment|evaluation)',
            r'(?:assessment|evaluation|verdict)\s*:\s*["\']?\s*(Correct|Incorrect|Partial)\s*["\']?',
        ]
        for pattern in explicit_patterns:
            match = re.search(pattern, last_message, re.IGNORECASE)
            if match:
                grade = match.group(1) if match.lastindex else match.group(0)
                normalized = _normalize_grade(grade)
                if normalized in ['Correct', 'Incorrect', 'Partial']:
                    return normalized
        
        # Strategy 5: Last resort - check if the entire message is just a grade
        stripped = last_message.strip()
        if stripped in ['0', '1', '2', '3', '4', '5', '6', '7']:
            return _normalize_grade(stripped)
        if stripped.lower() in ['correct', 'incorrect', 'partial']:
            return _normalize_grade(stripped)
        
        # Strategy 6: Check for grade at the very end of the message
        last_lines = stripped.split('\n')[-3:]  # Last 3 lines
        for line in reversed(last_lines):
            line_clean = line.strip().strip('"\'.,;:')
            if line_clean.lower() in ['correct', 'incorrect', 'partial']:
                return _normalize_grade(line_clean)
            # Check for "Grade: X" or "Response: X" at end
            if ':' in line_clean:
                parts = line_clean.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip().lower()
                    value = parts[1].strip().strip('"\'.,;:')
                    if key in ['grade', 'response', 'result', 'answer']:
                        normalized = _normalize_grade(value)
                        if normalized in ['Correct', 'Incorrect', 'Partial']:
                            return normalized
        
        return "None"

    def _get_grade_from_json(self, json_obj: dict) -> str:
        """Extract grade from JSON object with field priority."""
        if not isinstance(json_obj, dict):
            return "None"
        
        # Priority order for grade fields (most specific to least specific)
        priority_fields = [
            "response",   # The task agent's expected output field (highest priority)
            "grade",      # Most specific - explicitly a grade
            "result",     # Generic result field
            "answer",     # Answer field
            "evaluation", # Evaluation result
            "score",      # Numeric score
            "verdict",    # Verdict/assessment
            "assessment", # Assessment field
        ]
        
        for field in priority_fields:
            if field in json_obj:
                value = json_obj[field]
                if isinstance(value, str):
                    normalized = _normalize_grade(value)
                    if normalized in ['Correct', 'Incorrect', 'Partial']:
                        return normalized
                elif isinstance(value, (int, float)):
                    # Convert numeric to string and normalize
                    normalized = _normalize_grade(str(value))
                    if normalized in ['Correct', 'Incorrect', 'Partial']:
                        return normalized
                elif isinstance(value, bool):
                    # Handle boolean values
                    return 'Correct' if value else 'Incorrect'
        
        # If no recognized field, use the first string or numeric value found
        for key, value in json_obj.items():
            if isinstance(value, str):
                normalized = _normalize_grade(value)
                if normalized in ['Correct', 'Incorrect', 'Partial']:
                    return normalized
            elif isinstance(value, (int, float)):
                normalized = _normalize_grade(str(value))
                if normalized in ['Correct', 'Incorrect', 'Partial']:
                    return normalized
            elif isinstance(value, bool):
                return 'Correct' if value else 'Incorrect'
        
        return "None"

    def _extract_grade_from_text(self, text: str) -> str:
        """Extract grade from plain text using pattern matching."""
        # Look for explicit grade statements with improved patterns
        patterns = [
            r'grade[\s]*[:=][\s]*["\']?([^"\'\n]+)["\']?',
            r'response[\s]*[:=][\s]*["\']?([^"\'\n]+)["\']?',
            r'final grade[\s]*[:=][\s]*["\']?([^"\'\n]+)["\']?',
            r'assign[\s]+["\']?([^"\'\n]+)["\']?',
            r'["\']?grade["\']?\s*is\s*["\']?([^"\'\n]+)["\']?',
            r'["\']?score["\']?\s*[:=]\s*["\']?([^"\'\n]+)["\']?',
            r'["\']?result["\']?\s*[:=]\s*["\']?([^"\'\n]+)["\']?',
            r'["\']?evaluation["\']?\s*[:=]\s*["\']?([^"\'\n]+)["\']?',
            r'["\']?answer["\']?\s*[:=]\s*["\']?([^"\'\n]+)["\']?',
            r'["\']?verdict["\']?\s*[:=]\s*["\']?([^"\'\n]+)["\']?',
            r'["\']?assessment["\']?\s*[:=]\s*["\']?([^"\'\n]+)["\']?',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result = _normalize_grade(match.group(1).strip())
                if result in ['Correct', 'Incorrect', 'Partial']:
                    return result
        
        # Look for standalone grades at the end of the text (common pattern)
        lines = text.strip().split('\n')
        for line in reversed(lines[-15:]):  # Check last 15 lines for more coverage
            line = line.strip()
            # Skip empty lines and common non-grade lines
            if not line or line.startswith('#') or line.startswith('//') or line.startswith('*') or line.startswith('{'):
                continue
            # Look for common grade patterns
            grade_patterns = [
                r'^\s*([0-7])\s*$',  # Single digit 0-7 (IMO scale)
                r'^\s*["\']?([0-7])["\']?\s*$',  # Digit in quotes
                r'^\s*["\']?(Correct|Incorrect|Partial)["\']?\s*$',  # Standard grades
                r'^\s*["\']?(correct|incorrect|partial)["\']?\s*$',  # Lowercase
                r'^\s*Grade:\s*["\']?([^"\'\n]+)["\']?\s*$',  # "Grade: X" format
                r'^\s*The grade is\s*["\']?([^"\'\n]+)["\']?\s*$',  # "The grade is X"
                r'^\s*Response:\s*["\']?([^"\'\n]+)["\']?\s*$',  # "Response: X" format
                r'^\s*Result:\s*["\']?([^"\'\n]+)["\']?\s*$',  # "Result: X" format
                r'^\s*Answer:\s*["\']?([^"\'\n]+)["\']?\s*$',  # "Answer: X" format
                r'^\s*Evaluation:\s*["\']?([^"\'\n]+)["\']?\s*$',  # "Evaluation: X" format
            ]
            for gp in grade_patterns:
                match = re.search(gp, line, re.IGNORECASE)
                if match:
                    result = _normalize_grade(match.group(1).strip())
                    if result in ['Correct', 'Incorrect', 'Partial']:
                        return result
        
        return "None"

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with enhanced error handling.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Validate required inputs
        required_fields = ["problem", "solution", "student_answer"]
        missing_fields = [f for f in required_fields if not inputs.get(f)]
        if missing_fields:
            self.log_fn(f"Missing required fields: {missing_fields}")
            return "None", []
        
        instruction = self._build_grading_prompt(inputs)

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"LLM call failed: {e}")
            return "None", []

        # Extract prediction using multiple strategies
        prediction = self._extract_prediction(msg_history)
        
        if prediction == "None":
            # Log more details for debugging
            last_msg = msg_history[-1].get("text", "") if msg_history else ""
            self.log_fn(f"Failed to extract prediction. Last message preview: {last_msg[:1000] if last_msg else 'empty'}")
            # Try to extract from the raw response as well
            if response and response != last_msg:
                prediction = self._extract_prediction([{"text": response}])
                if prediction != "None":
                    self.log_fn(f"Successfully extracted from raw response: {prediction}")
            
            # Final fallback: try to find any grade-like text in the response
            if prediction == "None" and response:
                # Look for the grade words directly in the response
                response_lower = response.lower()
                if 'incorrect' in response_lower and 'correct' not in response_lower.replace('incorrect', ''):
                    prediction = "Incorrect"
                    self.log_fn(f"Fallback extraction: Incorrect")
                elif 'partial' in response_lower:
                    prediction = "Partial"
                    self.log_fn(f"Fallback extraction: Partial")
                elif 'correct' in response_lower and 'incorrect' not in response_lower:
                    prediction = "Correct"
                    self.log_fn(f"Fallback extraction: Correct")

        return str(prediction), msg_history
