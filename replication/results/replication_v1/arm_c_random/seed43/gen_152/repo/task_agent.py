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
    """Extract JSON objects from <json>...</json> blocks with enhanced robustness.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles markdown code blocks, raw JSON objects, and common LLM output errors.
    Includes automatic JSON repair for common syntax issues.
    """
    results = []
    search_from = 0
    
    def _try_parse_json(json_str: str) -> dict | None:
        """Try to parse JSON with progressive error recovery."""
        json_str = json_str.strip()
        if not json_str:
            return None
            
        # Attempt 1: Direct parsing
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # Attempt 2: Remove trailing commas before closing braces/brackets
        try:
            cleaned = re.sub(r',(\s*[}\]])', r'\1', json_str)
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        
        # Attempt 3: Fix unescaped newlines in strings
        try:
            # Replace newlines within string values with escaped versions
            cleaned = re.sub(r'(?<=")([^"]*?)\n([^"]*?)(?=")', r'\1\\n\2', json_str)
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        
        # Attempt 4: Fix single quotes (common LLM error)
        try:
            cleaned = json_str.replace("'", '"')
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        
        # Attempt 5: Extract just the fields we need using regex
        try:
            reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', json_str, re.DOTALL)
            response_match = re.search(r'"response"\s*:\s*"?([^"},\s]*)"?', json_str)
            if response_match:
                result = {"response": response_match.group(1).strip()}
                if reasoning_match:
                    result["reasoning"] = reasoning_match.group(1).strip()
                return result
        except Exception:
            pass
        
        return None
    
    # First, try to find <json>...</json> blocks
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        parsed = _try_parse_json(inner)
        if parsed:
            results.append(parsed)
    
    # If no <json> blocks found, try markdown code blocks
    if not results:
        md_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        for match in re.finditer(md_pattern, text, re.DOTALL):
            parsed = _try_parse_json(match.group(1).strip())
            if parsed:
                results.append(parsed)
    
    # Last resort: try to find any JSON-like structure with "response" field
    if not results:
        # Look for objects with "response" key - use a more flexible pattern
        json_pattern = r'\{[^{}]*"response"[^{}]*(?:\}[^{}]*\})?'
        for match in re.finditer(json_pattern, text):
            parsed = _try_parse_json(match.group())
            if parsed:
                results.append(parsed)
    
    # Final fallback: look for any JSON object with required fields
    if not results:
        # Try to find any complete JSON object
        brace_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        for match in re.finditer(brace_pattern, text, re.DOTALL):
            parsed = _try_parse_json(match.group())
            if parsed and ("response" in parsed or "reasoning" in parsed):
                results.append(parsed)
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced reasoning and validation."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3

    def _build_prompt(self, inputs: dict) -> str:
        """Build a structured prompt for IMO grading with chain-of-thought reasoning."""
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        return f"""You are an expert IMO (International Mathematical Olympiad) grader with extensive experience in mathematical problem evaluation.

Your task is to evaluate a student's solution to a mathematical problem with precision, consistency, and fairness.

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

Follow this systematic approach:

1. **Approach Analysis** (20% of reasoning)
   - Identify the student's overall strategy
   - Note key mathematical techniques employed
   - Assess whether the approach is appropriate for the problem

2. **Step-by-Step Verification** (40% of reasoning)
   - Verify each claim against the official solution
   - Check calculations for accuracy
   - Identify any gaps or incorrect steps

3. **Error Classification** (if applicable)
   - Conceptual: Misunderstanding of the problem statement or core concepts
   - Computational: Arithmetic or algebraic mistakes
   - Logical: Flawed reasoning or invalid deductions
   - Completeness: Missing cases, steps, or edge conditions

4. **Partial Credit Assessment** (30% of reasoning)
   - Identify all correct portions of the solution
   - Map correct work to the grading rubric
   - Justify the assigned grade with specific evidence

5. **Final Grade Determination** (10% of reasoning)
   Use the standard IMO 0-7 scale:
   - 7: Complete, correct solution with rigorous reasoning
   - 6: Minor flaw or omission that doesn't affect core correctness
   - 5-3: Significant progress with identifiable gaps
   - 2-1: Some meaningful progress or insight
   - 0: No significant progress, irrelevant, or completely incorrect

## Response Format (REQUIRED)

You MUST respond with a valid JSON object enclosed in <json> tags:

<json>
{{
    "reasoning": "Your detailed analysis following the framework above. Include: (1) approach summary, (2) verification results, (3) error classification if any, (4) partial credit justification, (5) final grade rationale.",
    "response": "X"
}}
</json>

CRITICAL: The "response" field MUST contain ONLY a single digit from 0 to 7 (e.g., "7", "5", "2", "0"). No other text, explanations, or formatting allowed in this field.

The "reasoning" field should contain your complete analysis and justification for the grade."""

    def _validate_grade(self, grade: str) -> str | None:
        """Validate and normalize the grade to IMO 0-7 scale.
        
        Returns normalized grade string or None if invalid.
        Uses multi-layer validation with fuzzy matching for robustness.
        """
        if not grade:
            return None
        
        # Clean up the grade string
        grade = str(grade).strip()
        grade_lower = grade.lower()
        
        # Layer 1: Direct digit extraction with boundary validation
        import re
        # Look for standalone digits 0-7 (word boundaries to avoid extracting from larger numbers)
        standalone_digits = re.findall(r'\b([0-7])\b', grade)
        if standalone_digits:
            return standalone_digits[-1]  # Use last valid digit found
        
        # Layer 2: Extract any number and clamp to valid range
        numbers = re.findall(r'\d+', grade)
        if numbers:
            num = int(numbers[0])
            if 0 <= num <= 7:
                return str(num)
            elif num > 7:
                return '7'  # Cap at maximum
            # num < 0 would be caught by standalone digit check above
        
        # Layer 3: Semantic grade classification with confidence scoring
        grade_keywords = {
            '7': ['full credit', 'complete solution', 'perfect', 'correct solution', 'full marks', 'full score'],
            '6': ['minor flaw', 'small error', 'almost complete', 'tiny mistake', 'nearly perfect'],
            '5': ['significant progress', 'mostly correct', 'good attempt', 'substantial work'],
            '4': ['moderate progress', 'partial solution', 'some correct steps', 'fair attempt'],
            '3': ['partial credit', 'incomplete', 'some progress', 'few correct steps'],
            '2': ['limited progress', 'minimal work', 'small contribution', 'minor insight'],
            '1': ['minimal progress', 'little work', 'tiny insight', 'barely started'],
            '0': ['no credit', 'no progress', 'irrelevant', 'blank', 'wrong approach', 'incorrect', 'no solution', 'empty']
        }
        
        best_match = None
        best_score = 0
        
        for score, keywords in grade_keywords.items():
            for keyword in keywords:
                if keyword in grade_lower:
                    # Longer matches get higher confidence
                    confidence = len(keyword)
                    if confidence > best_score:
                        best_score = confidence
                        best_match = score
        
        if best_match:
            return best_match
        
        # Layer 4: Pattern-based heuristics for edge cases
        # Check for fractions that might indicate partial credit
        fraction_match = re.search(r'(\d+)\s*/\s*(\d+)', grade)
        if fraction_match:
            numerator = int(fraction_match.group(1))
            denominator = int(fraction_match.group(2))
            if denominator > 0:
                ratio = numerator / denominator
                # Map ratio to IMO scale (0-7)
                mapped = round(ratio * 7)
                return str(max(0, min(7, mapped)))
        
        # Check for percentage
        percent_match = re.search(r'(\d+)\s*%', grade_lower)
        if percent_match:
            percent = int(percent_match.group(1))
            mapped = round(percent / 100 * 7)
            return str(max(0, min(7, mapped)))
        
        return None

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with enhanced reasoning and validation.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_prompt(inputs)
        
        all_msg_history = []
        prediction = "None"
        last_error = ""
        
        # Retry loop for better grade extraction
        for attempt in range(self.max_retries):
            # Build retry message with context from previous failures
            if attempt == 0:
                msg = instruction
            else:
                msg = (
                    f"Previous response had an issue: {last_error}\n\n"
                    f"Please respond with a valid JSON object containing:\n"
                    f'{{"reasoning": "your analysis", "response": "0-7"}}\n\n'
                    f"The 'response' field MUST contain ONLY a single integer from 0 to 7.\n\n"
                    f"{instruction}"
                )
            
            response, msg_history, info = get_response_from_llm(
                msg=msg,
                model=self.model,
                msg_history=[] if attempt == 0 else msg_history,
            )
            
            all_msg_history.extend(msg_history)
            
            # Extract prediction from JSON with better error handling
            reasoning = ""
            try:
                # Try to extract from the last assistant message
                last_msg = msg_history[-1]["text"] if msg_history else ""
                extracted = _extract_jsons(last_msg)
                
                if extracted:
                    last_json = extracted[-1]
                    if "response" in last_json:
                        raw_prediction = last_json["response"]
                        validated = self._validate_grade(raw_prediction)
                        if validated:
                            prediction = validated
                            if "reasoning" in last_json:
                                reasoning = last_json["reasoning"]
                            # Log the reasoning for debugging
                            if reasoning:
                                self.log_fn(f"Reasoning: {reasoning[:200]}...")
                            break  # Valid grade found, exit retry loop
                        else:
                            last_error = f"Invalid grade format: '{raw_prediction}'"
                            self.log_fn(f"Invalid grade format on attempt {attempt + 1}: {raw_prediction}")
                    else:
                        last_error = "No 'response' field found in JSON"
                        self.log_fn(f"No 'response' field found on attempt {attempt + 1}")
                else:
                    # Fallback: try to find any JSON-like structure
                    json_match = re.search(r'\{[^}]*"response"[^}]*\}', last_msg)
                    if json_match:
                        try:
                            fallback = json.loads(json_match.group())
                            raw_prediction = fallback.get("response", "None")
                            validated = self._validate_grade(raw_prediction)
                            if validated:
                                prediction = validated
                                break
                        except json.JSONDecodeError:
                            pass
                    
                    # Last resort: try to find any digit in the response
                    digits = re.findall(r'\b[0-7]\b', last_msg)
                    if digits:
                        prediction = digits[-1]  # Use last digit found
                        self.log_fn(f"Extracted grade via digit search on attempt {attempt + 1}: {prediction}")
                        break
                    
                    last_error = "Could not extract valid JSON or grade from response"
                    self.log_fn(f"No valid JSON found on attempt {attempt + 1}")
                            
            except Exception as e:
                last_error = str(e)
                self.log_fn(f"Error extracting prediction on attempt {attempt + 1}: {e}")
        
        # Log final result with summary
        attempts_used = min(attempt + 1, self.max_retries)
        if prediction == "None":
            self.log_fn(f"WARNING: Failed to extract valid grade after {attempts_used} attempt(s). Defaulting to '0'.")
            prediction = "0"  # Default to 0 instead of None for failed extractions
        else:
            self.log_fn(f"Final grade: {prediction} (after {attempts_used} attempt(s))")

        return str(prediction), all_msg_history
