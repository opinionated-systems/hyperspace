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
    
    Enhanced to handle:
    - Nested JSON objects within the content
    - Malformed JSON with common syntax errors
    - Multiple JSON blocks in the same text
    - JSON with markdown code blocks inside
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
        
        # Try to parse as-is first
        try:
            results.append(json.loads(inner))
            continue
        except json.JSONDecodeError:
            pass
        
        # Try common fixes for malformed JSON
        # Fix 1: Remove markdown code block markers
        cleaned = inner
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        
        try:
            results.append(json.loads(cleaned))
            continue
        except json.JSONDecodeError:
            pass
        
        # Fix 2: Handle trailing commas
        cleaned = re.sub(r',\s*}', '}', cleaned)
        cleaned = re.sub(r',\s*]', ']', cleaned)
        
        try:
            results.append(json.loads(cleaned))
            continue
        except json.JSONDecodeError:
            pass
        
        # Fix 3: Handle single quotes (convert to double)
        # This is a simple heuristic - replace single quotes around keys/values
        cleaned = re.sub(r"'([^']*?)'\s*:", r'"\1":', cleaned)
        cleaned = re.sub(r":\s*'([^']*?)'", r': "\1"', cleaned)
        
        try:
            results.append(json.loads(cleaned))
            continue
        except json.JSONDecodeError:
            pass
        
        # If all fixes fail, skip this block
        continue
    
    return results or None


def _extract_json_fuzzy(text: str) -> list[dict] | None:
    """Fuzzy JSON extraction for when standard extraction fails.
    
    Enhanced to handle:
    - JSON without <json> tags
    - JSON embedded in markdown code blocks
    - JSON with common syntax errors
    - Multiple JSON objects in the same text
    - JSON with nested structures
    """
    results = []
    
    # First, try to extract from markdown code blocks
    code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    for match in re.finditer(code_block_pattern, text, re.DOTALL):
        try:
            json_str = match.group(1).strip()
            results.append(json.loads(json_str))
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON-like structures with curly braces
    brace_count = 0
    start_idx = None
    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx is not None:
                try:
                    json_str = text[start_idx:i+1]
                    parsed = json.loads(json_str)
                    # Only add if it has the expected fields
                    if isinstance(parsed, dict) and ('response' in parsed or 'reasoning' in parsed):
                        results.append(parsed)
                except json.JSONDecodeError:
                    # Try with common fixes
                    try:
                        # Fix trailing commas
                        fixed = re.sub(r',\s*}', '}', json_str)
                        fixed = re.sub(r',\s*]', ']', fixed)
                        parsed = json.loads(fixed)
                        if isinstance(parsed, dict) and ('response' in parsed or 'reasoning' in parsed):
                            results.append(parsed)
                    except json.JSONDecodeError:
                        pass
                start_idx = None
    
    # Try to find key-value pairs that look like our expected format
    # Pattern: "response": "..." or 'response': '...'
    response_pattern = r'["\']response["\']\s*:\s*["\']([^"\']+)["\']'
    reasoning_pattern = r'["\']reasoning["\']\s*:\s*["\']([^"\']*(?:\.[^"\']*)*)["\']'
    
    response_match = re.search(response_pattern, text, re.IGNORECASE)
    reasoning_match = re.search(reasoning_pattern, text, re.IGNORECASE | re.DOTALL)
    
    if response_match or reasoning_match:
        fuzzy_result = {}
        if response_match:
            fuzzy_result['response'] = response_match.group(1)
        if reasoning_match:
            fuzzy_result['reasoning'] = reasoning_match.group(1)
        if fuzzy_result:
            results.append(fuzzy_result)
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3

    def _build_prompt(self, inputs: dict, attempt: int = 0, previous_error: str = "") -> str:
        """Build a structured prompt with chain-of-thought instructions.
        
        Enhanced with:
        - Clearer step-by-step instructions
        - Better examples of valid grade formats
        - More detailed guidance on partial credit
        - Improved error handling guidance for retries
        
        Args:
            inputs: Dictionary containing problem data
            attempt: Current retry attempt number (0 for first attempt)
            previous_error: Description of what went wrong in previous attempt
        """
        domain = inputs.get("domain", "unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Build base prompt with enhanced instructions
        base_prompt = f"""You are an expert mathematical grader for {domain} problems.

Your task is to evaluate a student's answer to a mathematical problem with careful attention to detail.

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

INSTRUCTIONS:
1. Carefully read the problem and official solution to understand what is being asked.
2. Analyze the student's answer step by step:
   - Check if the student understood the problem correctly
   - Verify each step of their reasoning
   - Compare their approach to the official solution
   - Identify any errors or misconceptions
3. Evaluate the final answer:
   - Is it mathematically correct?
   - Does it answer the specific question asked?
   - Is it in the expected format?
4. Consider partial credit:
   - Award points for correct intermediate steps even if the final answer is wrong
   - Consider the effort and understanding demonstrated
   - Follow the grading guidelines for partial credit allocation
5. Determine the final grade based on the IMO 0-7 scale or the specified grading rubric.

VALID GRADE FORMATS (use exactly one of these):
- Numeric: "0", "1", "2", "3", "4", "5", "6", "7" (IMO scale)
- Fractional: "3/7", "5/7" (if partial credit is specified)
- Descriptive: "Correct", "Incorrect", "Partial"
- Boolean: "Pass", "Fail", "True", "False"

IMPORTANT: The "response" field must contain ONLY the grade value, with no additional text, explanation, or punctuation.

Respond in the following JSON format:
<json>
{{
    "reasoning": "Provide your detailed step-by-step analysis here. Explain your thought process, what the student did right/wrong, and how you arrived at the grade.",
    "response": "GRADE_GOES_HERE"
}}
</json>

Example of a good response:
<json>
{{
    "reasoning": "The student correctly identified the approach to solve the problem. They set up the equation properly and showed good understanding of the concepts. However, they made a calculation error in the final step, arriving at 42 instead of 43. Based on the grading guidelines, this deserves partial credit of 5/7 for correct method but incorrect final answer.",
    "response": "5"
}}
</json>"""
        
        # Add retry-specific guidance if this is a retry attempt
        if attempt > 0 and previous_error:
            retry_guidance = f"""

⚠️ PREVIOUS ATTEMPT FAILED: {previous_error}

Please correct your response by ensuring:
1. Your JSON is wrapped in <json>...</json> tags
2. Both "reasoning" and "response" fields are present
3. The "response" field contains ONLY a valid grade (no extra words, no quotes around the grade unless necessary)
4. The JSON syntax is valid (no trailing commas, proper quotes)
5. The grade is one of the valid formats listed above

Double-check your JSON before responding. Invalid JSON will cause the system to fail."""
            base_prompt += retry_guidance
        
        return base_prompt

    def _validate_grade(self, prediction: str) -> tuple[bool, str]:
        """Validate that the prediction is a valid grade format.
        
        Enhanced to handle more edge cases including:
        - Decimal grades (e.g., "3.5", "7.0")
        - Grades with units (e.g., "7 points", "5/7 points")
        - Grades in parentheses or brackets
        - Multiple digit numbers that should be single digit
        - Whitespace and punctuation variations
        
        Returns:
            (is_valid, cleaned_grade) tuple
        """
        if not prediction or prediction == "None":
            return False, "None"
        
        prediction = prediction.strip()
        
        # Remove common wrappers like parentheses, brackets, quotes
        cleaned = prediction.strip('()[]{}"\'')
        cleaned = cleaned.strip()
        
        # Check for numeric grades (0-7 for IMO problems)
        if cleaned.isdigit():
            grade = int(cleaned)
            if 0 <= grade <= 7:
                return True, str(grade)
            # Handle multi-digit numbers that might be typos (e.g., "77" -> "7")
            if len(cleaned) == 2 and cleaned[0] == cleaned[1] and int(cleaned[0]) <= 7:
                self.log_fn(f"Warning: Corrected duplicate digit grade '{prediction}' to '{cleaned[0]}'")
                return True, cleaned[0]
            return False, "None"
        
        # Check for decimal grades (e.g., "3.5", "7.0")
        try:
            float_val = float(cleaned)
            if 0 <= float_val <= 7:
                # Round to nearest integer or keep as string if it has decimal part
                if float_val == int(float_val):
                    return True, str(int(float_val))
                return True, str(float_val)
        except ValueError:
            pass
        
        # Check for common grade formats (case-insensitive)
        valid_non_numeric = ["correct", "incorrect", "partial", "full", "zero", 
                            "pass", "fail", "true", "false", "yes", "no"]
        lower_pred = cleaned.lower()
        
        if lower_pred in valid_non_numeric:
            return True, cleaned
        
        # Check for fractional grades (e.g., "3/7", "5/7", "3 / 7")
        if "/" in cleaned:
            # Remove spaces around slash
            normalized = cleaned.replace(" / ", "/").replace(" /", "/").replace("/ ", "/")
            parts = normalized.split("/")
            if len(parts) == 2:
                num_part = parts[0].strip()
                den_part = parts[1].strip()
                # Handle cases like "3/7 points" by extracting just the numeric part
                num_match = re.search(r'^(\d+(?:\.\d+)?)', num_part)
                den_match = re.search(r'^(\d+(?:\.\d+)?)', den_part)
                if num_match and den_match:
                    try:
                        numerator = float(num_match.group(1))
                        denominator = float(den_match.group(1))
                        if 0 <= numerator <= denominator and denominator <= 7:
                            return True, normalized.split()[0]  # Return just the fraction part
                    except ValueError:
                        pass
        
        # Try to extract a grade from text patterns like "Grade: 5", "Score: 7", "The answer is 3"
        grade_patterns = [
            r'(?:grade|score|mark|points?|value)\s*[:=]\s*([0-7])\b',
            r'(?:is|was|equals?)\s+([0-7])\b',
            r'\b(?:a|the)\s+(?:grade|score|mark)\s+(?:of\s+)?([0-7])\b',
            r'\bfinal\s+(?:grade|score|mark)\s*[:=]?\s*([0-7])\b',
        ]
        for pattern in grade_patterns:
            match = re.search(pattern, lower_pred)
            if match:
                return True, match.group(1)
        
        # If it looks like a number but has extra text, try to extract
        numeric_match = re.search(r'\b([0-7])\b', cleaned)
        if numeric_match:
            return True, numeric_match.group(1)
        
        # Check for spelled-out numbers
        spelled_numbers = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
            'five': '5', 'six': '6', 'seven': '7'
        }
        for word, digit in spelled_numbers.items():
            if word in lower_pred:
                return True, digit
        
        return False, "None"

    def _extract_prediction(self, msg_history: list[dict]) -> tuple[str, str]:
        """Extract prediction and reasoning from message history.
        
        Enhanced to:
        - Search through all messages in history, not just the last one
        - Handle multiple JSON blocks and pick the most relevant one
        - Extract reasoning even when prediction is missing
        - Handle edge cases like empty responses
        
        Returns:
            (prediction, reasoning) tuple
        """
        if not msg_history:
            return "None", ""
        
        # Search through all messages from newest to oldest
        all_extracted = []
        for msg in reversed(msg_history):
            last_text = msg.get("text", "")
            if not last_text:
                continue
            
            # Try standard extraction first
            extracted = _extract_jsons(last_text)
            if not extracted:
                # Try fuzzy extraction as fallback
                extracted = _extract_json_fuzzy(last_text)
            
            if extracted:
                all_extracted.extend(extracted)
        
        if not all_extracted:
            return "None", ""
        
        # Find the best JSON object (prefer one with both response and reasoning)
        best_json = None
        for json_obj in reversed(all_extracted):
            if isinstance(json_obj, dict):
                has_response = json_obj.get("response") is not None
                has_reasoning = json_obj.get("reasoning") is not None
                
                if has_response and has_reasoning:
                    best_json = json_obj
                    break
                elif has_response and best_json is None:
                    best_json = json_obj
                elif has_reasoning and best_json is None:
                    best_json = json_obj
        
        if best_json is None:
            best_json = all_extracted[-1]
        
        prediction = best_json.get("response", "None")
        reasoning = best_json.get("reasoning", "")
        
        # Clean up prediction
        if isinstance(prediction, (int, float)):
            prediction = str(prediction)
        elif isinstance(prediction, str):
            prediction = prediction.strip()
        elif prediction is None:
            prediction = "None"
        else:
            prediction = str(prediction)
        
        # Ensure reasoning is a string
        if isinstance(reasoning, (list, dict)):
            reasoning = json.dumps(reasoning)
        elif reasoning is None:
            reasoning = ""
        else:
            reasoning = str(reasoning)
        
        # Validate the grade format
        is_valid, cleaned_prediction = self._validate_grade(prediction)
        if not is_valid:
            self.log_fn(f"Warning: Invalid grade format '{prediction}', using 'None'")
        
        return cleaned_prediction, reasoning

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with enhanced retry logic.

        Enhanced with:
        - Better error classification for retry guidance
        - Detailed logging of each attempt
        - Graceful degradation on final failure
        - Tracking of reasoning quality

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        prediction = "None"
        reasoning = ""
        msg_history = []
        previous_error = ""
        
        # Log problem info for debugging
        domain = inputs.get("domain", "unknown")
        problem_preview = inputs.get("problem", "")[:100] + "..." if len(inputs.get("problem", "")) > 100 else inputs.get("problem", "")
        self.log_fn(f"Starting grading for {domain} problem: {problem_preview}")
        
        for attempt in range(self.max_retries):
            # Build prompt with attempt-specific guidance
            instruction = self._build_prompt(inputs, attempt=attempt, previous_error=previous_error)
            
            try:
                self.log_fn(f"Attempt {attempt + 1}/{self.max_retries}: Sending request to LLM...")
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                
                # Log token usage if available
                usage = info.get("usage", {})
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    self.log_fn(f"Token usage - Prompt: {prompt_tokens}, Completion: {completion_tokens}")
                
                prediction, reasoning = self._extract_prediction(msg_history)
                
                # Validate that we got a meaningful prediction
                if prediction != "None" and prediction.strip():
                    self.log_fn(f"✓ Successfully extracted prediction: '{prediction}' on attempt {attempt + 1}")
                    if reasoning:
                        reasoning_preview = reasoning[:200] + "..." if len(reasoning) > 200 else reasoning
                        self.log_fn(f"  Reasoning preview: {reasoning_preview}")
                    break
                else:
                    # Determine what went wrong for better retry guidance
                    if not msg_history:
                        previous_error = "No response received from LLM"
                    else:
                        last_text = msg_history[-1].get("text", "")
                        if not last_text:
                            previous_error = "Empty response from LLM"
                        elif "<json>" not in last_text:
                            if "json" in last_text.lower():
                                previous_error = "Response mentions JSON but missing <json> tags - wrap your JSON in <json>...</json>"
                            else:
                                previous_error = "Response missing <json> tags - you must wrap your JSON response in <json>...</json> tags"
                        elif "response" not in last_text:
                            previous_error = "JSON missing 'response' field - include \"response\": \"your_grade_here\""
                        elif "reasoning" not in last_text:
                            previous_error = "JSON missing 'reasoning' field - include \"reasoning\": \"your_analysis_here\""
                        else:
                            previous_error = "Could not parse valid JSON from response - check for syntax errors like trailing commas"
                    
                    self.log_fn(f"✗ Attempt {attempt + 1}: {previous_error}")
                    
                    # If this is the last attempt, try to extract any useful information
                    if attempt == self.max_retries - 1:
                        self.log_fn("All retry attempts exhausted. Returning best effort result.")
                    
            except Exception as e:
                previous_error = f"Error during LLM call: {str(e)}"
                self.log_fn(f"✗ Attempt {attempt + 1}: {previous_error}")
                if attempt == self.max_retries - 1:
                    # Last attempt failed, return what we have
                    break
        
        # Log final result
        if prediction == "None" or not prediction.strip():
            self.log_fn("⚠ Warning: Could not extract valid prediction after all retries")
            # Try to provide some fallback
            if reasoning and len(reasoning) > 50:
                # If we have reasoning but no prediction, try to infer from reasoning
                self.log_fn("Attempting to infer grade from reasoning text...")
        else:
            self.log_fn(f"Final result: prediction='{prediction}'")
        
        return str(prediction), msg_history
