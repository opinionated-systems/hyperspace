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
    Enhanced to handle nested JSON structures and edge cases.
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
        
        # Try direct parsing first
        try:
            results.append(json.loads(inner))
            continue
        except json.JSONDecodeError:
            pass
        
        # Try to find valid JSON within the content using brace counting
        try:
            json_start = inner.find('{')
            if json_start == -1:
                continue
                
            # Use brace counting to find the matching closing brace
            brace_count = 0
            json_end = -1
            for i, char in enumerate(inner[json_start:], start=json_start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i
                        break
            
            if json_end != -1:
                potential_json = inner[json_start:json_end+1]
                results.append(json.loads(potential_json))
                continue
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Last resort: try simple find/rfind approach
        try:
            json_start = inner.find('{')
            json_end = inner.rfind('}')
            if json_start != -1 and json_end != -1 and json_end > json_start:
                results.append(json.loads(inner[json_start:json_end+1]))
        except json.JSONDecodeError:
            continue
            
    return results or None


def _extract_json_from_markdown(text: str) -> dict | None:
    """Extract JSON from markdown code blocks (```json ... ```).
    
    Enhanced to handle nested JSON structures and edge cases.
    """
    # Try json-specific blocks first
    pattern = r'```json\s*(.*?)\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    # Also try generic code blocks
    generic_pattern = r'```\s*(.*?)\s*```'
    generic_matches = re.findall(generic_pattern, text, re.DOTALL)
    all_matches = matches + [m for m in generic_matches if m not in matches]
    
    for match in all_matches:
        match = match.strip()
        
        # Try direct parsing first
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            pass
        
        # Try brace counting for nested structures
        try:
            json_start = match.find('{')
            if json_start == -1:
                continue
                
            brace_count = 0
            json_end = -1
            for i, char in enumerate(match[json_start:], start=json_start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i
                        break
            
            if json_end != -1:
                potential_json = match[json_start:json_end+1]
                return json.loads(potential_json)
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Last resort: simple find/rfind
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
    
    Returns:
        (validated_grade, is_valid)
    """
    if not prediction or prediction == "None":
        return "None", False
    
    pred_clean = prediction.strip()
    pred_lower = pred_clean.lower()
    
    # Strict validation: only accept single digit 0-7
    # This ensures consistent output format
    if pred_clean in ["0", "1", "2", "3", "4", "5", "6", "7"]:
        return pred_clean, True
    
    # Check for fractional grades like "3/7" or "5 / 7" - extract just the numerator
    fractional_match = re.search(r'\b([0-7])\s*/\s*7\b', pred_clean)
    if fractional_match:
        return fractional_match.group(1), True
    
    # Check for numeric grades embedded in text (0-7 for IMO problems)
    numeric_match = re.search(r'\b([0-7])\b', pred_clean)
    if numeric_match:
        return numeric_match.group(1), True
    
    # Check for full credit patterns -> 7
    full_patterns = [
        r'\bfull\s*(?:credit|points?|score)?\b',
        r'\bcomplete\s*(?:solution|answer|credit)?\b',
        r'\ball\s*(?:points?|credit|marks?)?\b',
        r'\bperfect\s*(?:score|solution)?\b',
        r'\bcorrect\s*(?:solution|answer)?\b',
    ]
    for pattern in full_patterns:
        if re.search(pattern, pred_lower):
            return "7", True
    
    # Check for zero/no credit patterns -> 0
    zero_patterns = [
        r'\bno\s*(?:credit|points?|score|marks?)?\b',
        r'\bzero\s*(?:credit|points?|score|marks?)?\b',
        r'\b0\s*(?:points?|credit|score|marks?)?\b',
        r'\bincorrect\s*(?:solution|answer)?\b',
        r'\bwrong\s*(?:solution|answer)?\b',
        r'\bnone\b',
    ]
    for pattern in zero_patterns:
        if re.search(pattern, pred_lower):
            return "0", True
    
    # If no clear grade found, mark as invalid
    return pred_clean, False


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced reasoning and validation."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _build_prompt(self, inputs: dict) -> str:
        """Build a structured prompt for IMO grading with chain-of-thought reasoning.
        
        This method constructs a detailed prompt that guides the LLM to evaluate
        student solutions according to IMO grading standards (0-7 point scale).
        
        Args:
            inputs: Dictionary containing problem data with keys:
                - domain: Problem domain (e.g., "Mathematics")
                - problem: The problem statement
                - solution: The official solution
                - grading_guidelines: Specific grading criteria
                - student_answer: The student's submitted answer
                
        Returns:
            A formatted prompt string ready for LLM consumption
        """
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        return f"""You are an expert IMO (International Mathematical Olympiad) grader with extensive experience in mathematical problem evaluation.

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

## IMO Grading Philosophy

IMO grading follows specific principles:
- **Complete solutions get 7 points**: A correct, complete solution with proper justification
- **Partial credit is awarded for progress**: Even incomplete solutions can earn points for meaningful progress
- **Creativity is valued**: Alternative valid approaches are equally acceptable
- **Rigorous justification matters**: Correct answers without proper proof receive reduced credit

## Detailed Grade Rubric

### 7 Points (Complete Solution)
- Complete, correct solution with rigorous justification
- All logical steps are sound and well-explained
- May have minor presentation issues but no mathematical errors
- Alternative valid approaches are equally acceptable

### 6 Points (Near-Complete)
- Essentially correct solution with minor flaw
- Small computational error that doesn't affect main result
- Missing trivial case or edge condition
- Slight gap in reasoning that is easily fixable
- All major ideas are correct

### 5 Points (Substantial Progress)
- Significant progress toward complete solution
- Solved major part of problem correctly
- Correct approach with some significant gaps
- Good understanding demonstrated despite incomplete execution
- Most key ideas present but not fully developed

### 3-4 Points (Partial Progress)
- Some correct elements and understanding shown
- Good initial approach or setup
- Partial results that demonstrate knowledge
- Correct calculations for part of problem
- Shows engagement with problem but limited success

### 1-2 Points (Minimal Progress)
- Some relevant ideas or concepts mentioned
- Basic understanding of problem statement
- Attempted approach even if unsuccessful
- Shows some mathematical thinking
- Little concrete progress but not completely blank

### 0 Points (No Credit)
- Blank or completely irrelevant answer
- No understanding of problem demonstrated
- Completely wrong approach with no redeeming features
- Answer unrelated to problem asked

## Evaluation Instructions

1. **Read Carefully**: Read the student's answer completely before forming conclusions

2. **Compare to Official Solution**:
   - Check if student reached correct final answer
   - Verify logical structure matches valid proof techniques
   - Identify any errors, gaps, or alternative valid approaches

3. **Assess Mathematical Rigor**:
   - Are claims properly justified?
   - Is the logical flow sound?
   - Are there unstated assumptions?

4. **Consider Alternative Approaches**:
   - Different valid methods are equally acceptable
   - Judge by mathematical correctness, not similarity to official solution

5. **Award Partial Credit Generously**:
   - Meaningful progress deserves recognition
   - Correct ideas with execution errors still earn points
   - Look for what the student did right, not just what they did wrong

## Critical Grading Considerations

- **Correct answer without proof**: Maximum 1-2 points (must show reasoning)
- **Correct approach with calculation error**: 4-6 points depending on severity
- **Complete solution with minor gap**: 6 points (not 7)
- **Partial solution with good ideas**: 3-5 points
- **Only setup/no progress**: 0-1 points
- **Wrong approach but some correct observations**: 1-2 points

## Output Format (STRICT REQUIREMENT)

You MUST respond with a valid JSON object wrapped in <json> tags. The JSON must have exactly these two fields:

<json>
{{
    "reasoning": "Your detailed analysis here. Explain the student's approach, identify errors or gaps, compare to official solution, and justify your grade...",
    "response": "X"
}}
</json>

CRITICAL RULES:
- The "response" field MUST contain ONLY a single digit: 0, 1, 2, 3, 4, 5, 6, or 7
- Do NOT include quotes around the number, explanations, or any other text in the response field
- The "reasoning" field should contain your complete analysis (can be multiple sentences)
- Ensure the JSON is valid and properly formatted
- Double-check that your grade matches your reasoning (e.g., don't give 7 if you mention major errors)

EXAMPLES OF VALID RESPONSES:
<json>
{{
    "reasoning": "The student provided a complete and correct solution with proper justification. All steps are logically sound and the proof is rigorous.",
    "response": "7"
}}
</json>

<json>
{{
    "reasoning": "The student had the right approach and made good progress, but there was a computational error in the final step that led to an incorrect answer. The methodology was sound.",
    "response": "5"
}}
</json>

<json>
{{
    "reasoning": "The student showed some understanding of the problem and attempted a valid approach, but made significant errors and did not reach a solution. Some correct initial steps were present.",
    "response": "2"
}}
</json>"""

    def _extract_prediction(self, msg_history: list[dict]) -> tuple[str, str]:
        """Extract prediction and reasoning from message history with enhanced extraction.
        
        Uses multiple extraction strategies with priority ordering for robustness.
        
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
            
            # Strategy 1: Try <json> tags first (most reliable)
            extracted = _extract_jsons(last_msg)
            if extracted:
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = str(last_json["response"]).strip()
                if "reasoning" in last_json:
                    reasoning = str(last_json["reasoning"])
                if prediction != "None":
                    return prediction, reasoning
            
            # Strategy 2: Try markdown code blocks
            md_json = _extract_json_from_markdown(last_msg)
            if md_json:
                if "response" in md_json:
                    prediction = str(md_json["response"]).strip()
                if "reasoning" in md_json:
                    reasoning = str(md_json["reasoning"])
                if prediction != "None":
                    return prediction, reasoning
            
            # Strategy 3: Try to find JSON with brace counting for nested structures
            start = last_msg.find('{')
            end = last_msg.rfind('}')
            if start != -1 and end != -1 and end > start:
                try:
                    # Use brace counting to find the matching closing brace
                    brace_count = 0
                    json_end = -1
                    for i, char in enumerate(last_msg[start:], start=start):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_end = i
                                break
                    
                    if json_end != -1:
                        potential_json = last_msg[start:json_end+1]
                        fallback = json.loads(potential_json)
                        if "response" in fallback:
                            prediction = str(fallback["response"]).strip()
                            if "reasoning" in fallback:
                                reasoning = str(fallback["reasoning"])
                            if prediction != "None":
                                return prediction, reasoning
                except (json.JSONDecodeError, ValueError):
                    pass
            
            # Strategy 4: Simple outermost braces approach
            if start != -1 and end != -1 and end > start:
                try:
                    potential_json = last_msg[start:end+1]
                    fallback = json.loads(potential_json)
                    if "response" in fallback:
                        prediction = str(fallback["response"]).strip()
                        if "reasoning" in fallback:
                            reasoning = str(fallback["reasoning"])
                        if prediction != "None":
                            return prediction, reasoning
                except json.JSONDecodeError:
                    pass
            
            # Strategy 5: Look for JSON-like patterns with response field
            json_pattern = r'\{[^{}]*"response"[^{}]*\}'
            json_matches = re.findall(json_pattern, last_msg)
            for json_match in json_matches:
                try:
                    fallback = json.loads(json_match)
                    pred = str(fallback.get("response", "None")).strip()
                    if pred and pred != "None":
                        prediction = pred
                        if "reasoning" in fallback:
                            reasoning = str(fallback["reasoning"])
                        return prediction, reasoning
                except json.JSONDecodeError:
                    continue
            
            # Strategy 6: Look for explicit grade statements in text
            if prediction == "None":
                grade_patterns = [
                    r'["\']response["\']\s*:\s*["\']?([0-7])["\']?',
                    r'["\']response["\']\s*:\s*["\']?([0-7])["\']?\s*[,}]',
                    r'response["\']?\s*:\s*["\']?([0-7])["\']?',
                    r'(?:grade|score|mark|final grade|final score)\s*:?\s*([0-7])',
                    r'(?:grade|score|mark)\s+is\s+([0-7])',
                    r'(?:grade|score|mark)\s+of\s+([0-7])',
                    r'\bgives?\s+(?:a\s+)?([0-7])\s*(?:points?)?',
                    r'\baward\s+(?:a\s+)?([0-7])',
                    r'\bassign\s+(?:a\s+)?([0-7])',
                    r'\bgrade\s*[=:]\s*([0-7])\b',
                    r'\bscore\s*[=:]\s*([0-7])\b',
                ]
                for pattern in grade_patterns:
                    grade_match = re.search(pattern, last_msg, re.IGNORECASE)
                    if grade_match:
                        prediction = grade_match.group(1)
                        break
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
        
        return prediction, reasoning

    def _build_retry_prompt(self, inputs: dict, previous_response: str, error_reason: str) -> str:
        """Build a retry prompt when the initial response was invalid.
        
        Args:
            inputs: Original problem inputs
            previous_response: The previous LLM response that failed validation
            error_reason: Why the previous response was invalid
            
        Returns:
            A formatted retry prompt
        """
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        return f"""You are an expert IMO (International Mathematical Olympiad) grader. Your previous response was invalid.

## Previous Response (INVALID)
{previous_response}

## Error
{error_reason}

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

## Instructions for Evaluation (RETRY)

Your previous response could not be parsed correctly. This is a RETRY - please provide a valid JSON response following these strict rules:

### CRITICAL FORMAT RULES:
1. The "response" field MUST contain ONLY a single digit: 0, 1, 2, 3, 4, 5, 6, or 7
2. Do NOT include quotes around the number in the response field
3. Do NOT add explanations like "grade is" or "I give" in the response field
4. The "reasoning" field should contain your complete analysis (this is where explanations go)
5. Ensure the JSON is valid and properly formatted

### CORRECT FORMAT EXAMPLE:
<json>
{{
    "reasoning": "The student made good progress with the initial approach but had a computational error in the final step. The methodology was sound overall.",
    "response": "5"
}}
</json>

### INCORRECT FORMAT EXAMPLES (DO NOT DO THESE):
- "response": "The grade is 5"  ❌ (explanation in response field)
- "response": "5 points"  ❌ (extra text)
- "response": "5/7"  ❌ (not a single digit)
- "response": 5  ❌ (number without quotes - must be string)

## Output Format (STRICT REQUIREMENT)

You MUST respond with a valid JSON object wrapped in <json> tags:

<json>
{{
    "reasoning": "Your detailed analysis here...",
    "response": "X"
}}
</json>

Where X is exactly one digit: 0, 1, 2, 3, 4, 5, 6, or 7"""

    def _verify_grade_consistency(self, grade: str, reasoning: str) -> tuple[str, bool]:
        """Verify that the grade is consistent with the reasoning.
        
        Uses a nuanced approach that considers context and severity of indicators.
        
        Args:
            grade: The extracted grade
            reasoning: The reasoning text
            
        Returns:
            (verified_grade, is_consistent)
        """
        if not reasoning or grade == "None":
            return grade, True
            
        reasoning_lower = reasoning.lower()
        
        # Define severity levels for different indicators
        critical_errors = [
            "completely wrong", "totally incorrect", "fundamentally flawed",
            "does not solve", "fails to prove", "incorrect proof", "wrong answer"
        ]
        
        major_errors = [
            "major error", "critical flaw", "significant mistake", "serious gap",
            "key step is wrong", "main argument fails", "conclusion is incorrect"
        ]
        
        minor_issues = [
            "minor flaw", "small error", "trivial mistake", "insignificant gap",
            "could be clearer", "slightly incomplete", "minor omission"
        ]
        
        positive_indicators = [
            "correct approach", "right idea", "valid method", "good strategy",
            "sound reasoning", "proper technique", "appropriate method"
        ]
        
        strong_positive = [
            "complete solution", "fully correct", "perfect proof", "excellent work",
            "all steps correct", "thoroughly justified", "rigorous proof"
        ]
        
        partial_progress = [
            "partial progress", "some correct steps", "partial result",
            "incomplete but valid", "good start but", "correct up to"
        ]
        
        # Count occurrences of each category
        critical_count = sum(1 for ind in critical_errors if ind in reasoning_lower)
        major_count = sum(1 for ind in major_errors if ind in reasoning_lower)
        minor_count = sum(1 for ind in minor_issues if ind in reasoning_lower)
        positive_count = sum(1 for ind in positive_indicators if ind in reasoning_lower)
        strong_positive_count = sum(1 for ind in strong_positive if ind in reasoning_lower)
        partial_count = sum(1 for ind in partial_progress if ind in reasoning_lower)
        
        # Grade 7: Should have strong positive indicators and no critical/major errors
        if grade == "7":
            if critical_count > 0:
                self.log_fn(f"Grade 7 but reasoning contains critical errors - inconsistency detected")
                return grade, False
            if major_count > 1:  # Allow one major issue if overall positive
                self.log_fn(f"Grade 7 but reasoning contains multiple major errors - inconsistency detected")
                return grade, False
            if strong_positive_count == 0 and positive_count < 2:
                # Grade 7 should have strong justification
                self.log_fn(f"Grade 7 but reasoning lacks strong positive indicators - potential inconsistency")
                return grade, False
                    
        # Grade 6: Should have positive indicators with at most minor issues
        if grade == "6":
            if critical_count > 0:
                self.log_fn(f"Grade 6 but reasoning contains critical errors - inconsistency detected")
                return grade, False
            if major_count > 0 and strong_positive_count == 0:
                self.log_fn(f"Grade 6 with major errors but no strong positives - potential inconsistency")
                return grade, False
                    
        # Grade 5: Should have substantial progress with some issues
        if grade == "5":
            if critical_count > 1:
                self.log_fn(f"Grade 5 but reasoning contains multiple critical errors - potential inconsistency")
                return grade, False
            if positive_count == 0 and partial_count == 0:
                self.log_fn(f"Grade 5 but reasoning lacks positive indicators - potential inconsistency")
                return grade, False
                    
        # Grade 0: Should have no meaningful positive indicators
        if grade == "0":
            if strong_positive_count > 0:
                self.log_fn(f"Grade 0 but reasoning contains strong positive indicators - inconsistency detected")
                return grade, False
            if positive_count > 1 and critical_count == 0 and major_count == 0:
                self.log_fn(f"Grade 0 but reasoning has multiple positives without critical errors - potential inconsistency")
                return grade, False
            if partial_count > 0 and critical_count == 0:
                # Has partial progress but grade 0 - might be inconsistent
                self.log_fn(f"Grade 0 but reasoning mentions partial progress - potential inconsistency")
                return grade, False
                    
        # Grades 1-4: Should have partial progress indicators
        if grade in ["1", "2", "3", "4"]:
            if strong_positive_count > 0 and critical_count == 0:
                self.log_fn(f"Grade {grade} but reasoning contains strong positives - potential inconsistency")
                return grade, False
            if positive_count == 0 and partial_count == 0:
                self.log_fn(f"Grade {grade} but reasoning lacks any positive indicators - potential inconsistency")
                return grade, False
                    
        return grade, True

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with enhanced reasoning, validation, and retry.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_prompt(inputs)
        grading_guidelines = inputs.get("grading_guidelines", "")
        max_retries = 2
        all_msg_histories = []  # Track all attempts for debugging
        
        for attempt in range(max_retries + 1):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                all_msg_histories.extend(msg_history)
            except Exception as e:
                self.log_fn(f"LLM call failed (attempt {attempt + 1}): {e}")
                if attempt == max_retries:
                    return "None", all_msg_histories
                continue

            # Extract prediction with enhanced extraction
            prediction, reasoning = self._extract_prediction(msg_history)
            
            # Validate the grade
            validated_grade, is_valid = _validate_grade(prediction, grading_guidelines)
            
            # Log the reasoning and validation result
            if reasoning:
                self.log_fn(f"Reasoning: {reasoning[:200]}...")
            self.log_fn(f"Attempt {attempt + 1}: Extracted grade: {prediction}, Validated: {validated_grade}, Is valid: {is_valid}")
            
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
                        (r'(?:grade|score|mark|final grade|final score)\s*:?\s*([0-7])', 'numeric'),
                        (r'\bfull\s*(?:credit|points?|score)?\b', 'full'),
                        (r'\bcorrect\s*(?:solution|answer)?\b', 'correct'),
                        (r'\bno\s*(?:credit|points?|score|marks?)?\b', 'zero'),
                        (r'\bzero\s*(?:credit|points?|score|marks?)?\b', 'zero'),
                        (r'\bincorrect\s*(?:solution|answer)?\b', 'zero'),
                        (r'\bpartial\s*(?:credit|points?|score)?\b', 'partial'),
                        (r'\bsome\s*(?:credit|points?|score)?\b', 'partial'),
                    ]
                    for pattern, pattern_type in grade_patterns:
                        match = re.search(pattern, response, re.IGNORECASE)
                        if match:
                            if pattern_type == 'full' or pattern_type == 'correct':
                                validated_grade = "7"
                            elif pattern_type == 'zero':
                                validated_grade = "0"
                            elif pattern_type == 'partial':
                                # For partial credit, try to find a specific number or default to 3
                                numeric_in_match = re.search(r'\b([0-7])\b', match.group(0) if match.groups() else "")
                                if numeric_in_match:
                                    validated_grade = numeric_in_match.group(1)
                                else:
                                    validated_grade = "3"  # Default partial credit
                            else:
                                validated_grade = match.group(1)
                            is_valid = True
                            self.log_fn(f"Pattern-based fallback found grade: {validated_grade}")
                            break
            
            # Verify grade-reasoning consistency
            if is_valid and reasoning:
                validated_grade, is_consistent = self._verify_grade_consistency(validated_grade, reasoning)
                if not is_consistent and attempt < max_retries:
                    self.log_fn(f"Grade-reasoning inconsistency detected, retrying...")
                    error_reason = f"The grade '{validated_grade}' may not match the reasoning provided. The reasoning suggests a different grade might be appropriate."
                    instruction = self._build_retry_prompt(inputs, response, error_reason)
                    continue
            
            # If we have a valid grade, return it
            if is_valid:
                return str(validated_grade), all_msg_histories
            
            # If invalid and we have retries left, build retry prompt
            if attempt < max_retries:
                error_reason = f"Could not extract a valid grade from the response. The response field contained: '{prediction}'. Please provide a valid single digit (0-7) in the response field."
                instruction = self._build_retry_prompt(inputs, response, error_reason)
                self.log_fn(f"Invalid grade, retrying with corrected prompt...")
        
        # All retries exhausted, return best effort
        self.log_fn(f"All retries exhausted, returning best effort: {validated_grade}")
        return str(validated_grade) if validated_grade != "None" else "0", all_msg_histories
