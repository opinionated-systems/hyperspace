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
from typing import Any

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles nested JSON objects by tracking brace depth.
    Includes detailed error logging for debugging extraction failures.
    """
    results = []
    search_from = 0
    extraction_attempts = 0
    success_count = 0
    
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            logger.warning(f"Found opening <json> at position {start} but no closing </json>")
            break
        
        inner = text[start + 6:end].strip()
        search_from = end + 7
        extraction_attempts += 1
        
        # Try to parse the inner content as JSON
        try:
            parsed = json.loads(inner)
            results.append(parsed)
            success_count += 1
            logger.debug(f"Successfully parsed JSON from <json> block #{extraction_attempts}")
        except json.JSONDecodeError as e:
            # Try to extract JSON by finding balanced braces
            json_obj = _extract_balanced_json(inner)
            if json_obj:
                results.append(json_obj)
                success_count += 1
                logger.debug(f"Extracted JSON using balanced brace method from block #{extraction_attempts}")
            else:
                logger.warning(f"Failed to parse JSON from <json> block #{extraction_attempts}: {e}")
                logger.debug(f"Problematic content (first 200 chars): {inner[:200]!r}")
            continue
    
    if extraction_attempts > 0:
        logger.info(f"JSON extraction: {success_count}/{extraction_attempts} blocks successfully parsed")
    
    return results or None


def _extract_balanced_json(text: str) -> dict | None:
    """Extract a JSON object by tracking brace depth.
    
    This handles cases where the JSON content contains nested braces
    that might confuse simple regex-based extraction.
    Also handles edge cases like empty objects and trailing content.
    """
    start_idx = -1
    brace_depth = 0
    in_string = False
    escape_next = False
    candidate_count = 0
    
    for i, char in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if char == '\\' and in_string:
            escape_next = True
            continue
        if char == '"' and not in_string:
            in_string = True
            continue
        if char == '"' and in_string:
            in_string = False
            continue
        if not in_string:
            if char == '{':
                if brace_depth == 0:
                    start_idx = i
                brace_depth += 1
            elif char == '}':
                brace_depth -= 1
                if brace_depth == 0 and start_idx != -1:
                    candidate_count += 1
                    candidate = text[start_idx:i+1]
                    try:
                        parsed = json.loads(candidate)
                        # Validate that we got a dict (not a list or primitive)
                        if isinstance(parsed, dict):
                            logger.debug(f"Balanced JSON extraction: found valid object at candidate #{candidate_count}")
                            return parsed
                    except json.JSONDecodeError:
                        # Try next candidate if available
                        continue
    
    if candidate_count > 0:
        logger.debug(f"Balanced JSON extraction: {candidate_count} candidates found but none valid")
    
    return None


def _extract_json_with_regex(text: str) -> list[dict] | None:
    """Fallback JSON extraction using multiple strategies for malformed responses.
    
    Enhanced with additional strategies for handling edge cases like:
    - Truncated JSON
    - Missing closing braces
    - Unescaped quotes in strings
    - Mixed quote styles
    - JSON embedded in explanatory text
    """
    results = []
    
    # Strategy 1: Try to find JSON objects in code blocks
    code_block_pattern = r'```(?:json)?\s*(\{[\s\S]*?\})\s*```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    for match in matches:
        try:
            # Try to parse the matched content
            parsed = json.loads(match)
            if isinstance(parsed, dict):
                results.append(parsed)
        except json.JSONDecodeError:
            # Try balanced brace extraction
            balanced = _extract_balanced_json(match)
            if balanced:
                results.append(balanced)
    
    # Strategy 2: Look for any JSON-like structure with balanced braces
    if not results:
        balanced = _extract_balanced_json(text)
        if balanced:
            results.append(balanced)
    
    # Strategy 3: Extract key-value pairs for response and reasoning
    if not results:
        # Try to find response field with various quote styles
        response_patterns = [
            r'"response"\s*:\s*"([^"]*(?:\\.[^"]*)*)"',
            r'"response"\s*:\s*\'([^\']*)\'',
            r'"response"\s*:\s*([A-Za-z]+)',  # Unquoted value
        ]
        reasoning_patterns = [
            r'"reasoning"\s*:\s*"([^"]*(?:\\.[^"]*)*)"',
            r'"reasoning"\s*:\s*\'([^\']*)\'',
        ]
        
        extracted = {}
        for pattern in response_patterns:
            response_match = re.search(pattern, text, re.DOTALL)
            if response_match:
                extracted["response"] = response_match.group(1).replace('\\"', '"').replace('\\n', '\n')
                break
        
        for pattern in reasoning_patterns:
            reasoning_match = re.search(pattern, text, re.DOTALL)
            if reasoning_match:
                extracted["reasoning"] = reasoning_match.group(1).replace('\\"', '"').replace('\\n', '\n')
                break
        
        if extracted:
            results.append(extracted)
    
    # Strategy 4: Look for single-quoted JSON or unquoted keys
    if not results:
        # Try to find patterns like response: "value" or 'response': 'value'
        single_quote_pattern = r"'response'\s*:\s*'([^']*)'"
        single_match = re.search(single_quote_pattern, text, re.DOTALL)
        if single_match:
            results.append({"response": single_match.group(1)})
        
        # Try unquoted key pattern
        unquoted_pattern = r'response\s*:\s*"([^"]*)"'
        unquoted_match = re.search(unquoted_pattern, text, re.DOTALL)
        if unquoted_match:
            results.append({"response": unquoted_match.group(1)})
    
    # Strategy 5: Look for grade/assessment at end of response
    if not results:
        # Common patterns at end of text
        end_patterns = [
            r'(?:grade|assessment|score|verdict)\s*[:=]\s*["\']?([^"\'\n]+)["\']?',
            r'(?:the answer is|i would say|final grade)[:\s]+["\']?([^"\'\n]+)["\']?',
            r'(?:conclusion|summary|verdict)[:\s]+["\']?([^"\'\n]+)["\']?',
            r'(?:therefore|thus|hence)[,\s]+["\']?([^"\'\n]+)["\']?',
        ]
        for pattern in end_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                results.append({"response": match.group(1).strip()})
                break
    
    # Strategy 6: Try to fix common JSON errors and re-parse
    if not results:
        # Try to fix trailing commas
        fixed_text = re.sub(r',(\s*[}\]])', r'\1', text)
        # Try to fix unquoted keys
        fixed_text = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', fixed_text)
        balanced = _extract_balanced_json(fixed_text)
        if balanced:
            results.append(balanced)
    
    # Strategy 7: Extract from truncated JSON (missing closing braces)
    if not results:
        # Look for partial JSON with response field
        partial_match = re.search(r'"response"\s*:\s*"([^"]*)"[^}]*$', text, re.DOTALL)
        if partial_match:
            results.append({"response": partial_match.group(1)})
    
    return results or None


def _normalize_prediction(prediction: str) -> str:
    """Normalize prediction string for consistent comparison.
    
    Handles common variations like case differences, extra whitespace,
    and punctuation differences. Also handles numeric scores (0-100)
    by mapping them to grade categories.
    
    Enhanced to handle more edge cases including:
    - Fractional scores (e.g., "3/4", "7/10")
    - Letter grades (A, B, C, D, F)
    - Percentage signs
    - Multiple word phrases
    - Common typos and variations
    """
    if not prediction:
        return "None"
    
    # Strip whitespace and normalize case
    normalized = prediction.strip()
    
    # Try to parse as numeric score (0-100)
    try:
        # Remove any non-numeric characters except decimal point
        numeric_str = ''.join(c for c in normalized if c.isdigit() or c == '.')
        if numeric_str:
            score = float(numeric_str)
            # Map numeric score to grade category
            if score >= 80:
                return "Correct"
            elif score >= 40:
                return "Partial"
            else:
                return "Incorrect"
    except (ValueError, TypeError):
        pass
    
    # Try to parse fractional scores (e.g., "3/4", "7/10")
    try:
        if '/' in normalized:
            parts = normalized.split('/')
            if len(parts) == 2:
                numerator = float(parts[0].strip())
                denominator = float(parts[1].strip())
                if denominator > 0:
                    ratio = numerator / denominator
                    if ratio >= 0.8:
                        return "Correct"
                    elif ratio >= 0.4:
                        return "Partial"
                    else:
                        return "Incorrect"
    except (ValueError, TypeError, ZeroDivisionError):
        pass
    
    # Handle letter grades
    letter_grades = {
        'a': "Correct",
        'a+': "Correct",
        'a-': "Correct",
        'b': "Partial",
        'b+': "Partial",
        'b-': "Partial",
        'c': "Partial",
        'c+': "Partial",
        'c-': "Incorrect",
        'd': "Incorrect",
        'd+': "Incorrect",
        'd-': "Incorrect",
        'f': "Incorrect",
    }
    
    lower_pred = normalized.lower().rstrip('.').strip()
    if lower_pred in letter_grades:
        return letter_grades[lower_pred]
    
    # Handle common grade variations
    grade_map = {
        "correct": "Correct",
        "partial": "Partial",
        "partially correct": "Partial",
        "partial credit": "Partial",
        "mostly correct": "Partial",
        "mostly right": "Partial",
        "somewhat correct": "Partial",
        "somewhat right": "Partial",
        "nearly correct": "Partial",
        "close": "Partial",
        "almost": "Partial",
        "incorrect": "Incorrect",
        "wrong": "Incorrect",
        "right": "Correct",
        "true": "Correct",
        "false": "Incorrect",
        "full": "Correct",
        "full credit": "Correct",
        "complete credit": "Correct",
        "no credit": "Incorrect",
        "zero": "Incorrect",
        "0": "Incorrect",
        "none": "Incorrect",
        "half": "Partial",
        "half credit": "Partial",
        "50%": "Partial",
        "satisfactory": "Correct",
        "unsatisfactory": "Incorrect",
        "complete": "Correct",
        "incomplete": "Partial",
        "done": "Correct",
        "not done": "Incorrect",
        "valid": "Correct",
        "invalid": "Incorrect",
        "acceptable": "Correct",
        "unacceptable": "Incorrect",
        "good": "Correct",
        "bad": "Incorrect",
        "poor": "Incorrect",
        "excellent": "Correct",
        "perfect": "Correct",
        "imperfect": "Partial",
        "flawed": "Partial",
        "error-free": "Correct",
        "has errors": "Partial",
        "major errors": "Incorrect",
        "minor errors": "Partial",
    }
    
    if lower_pred in grade_map:
        return grade_map[lower_pred]
    
    # Check for multi-word phrases containing key terms
    if any(phrase in lower_pred for phrase in ["mostly correct", "mostly right", "nearly correct", "almost correct", "close to correct"]):
        return "Partial"
    if any(phrase in lower_pred for phrase in ["mostly wrong", "mostly incorrect", "nearly wrong", "almost wrong"]):
        return "Incorrect"
    if any(phrase in lower_pred for phrase in ["completely correct", "entirely correct", "totally correct", "100% correct", "fully correct"]):
        return "Correct"
    if any(phrase in lower_pred for phrase in ["completely wrong", "entirely wrong", "totally wrong", "100% wrong", "fully wrong", "completely incorrect"]):
        return "Incorrect"
    
    # Check for partial match (e.g., "mostly correct" -> "Partial")
    if "partial" in lower_pred or "part" in lower_pred or "some" in lower_pred:
        return "Partial"
    if "correct" in lower_pred or "right" in lower_pred or "valid" in lower_pred or "true" in lower_pred:
        return "Correct"
    if "incorrect" in lower_pred or "wrong" in lower_pred or "error" in lower_pred or "invalid" in lower_pred or "false" in lower_pred:
        return "Incorrect"
    
    # Check for percentage-based indicators
    if '%' in normalized:
        try:
            pct_str = ''.join(c for c in normalized if c.isdigit() or c == '.')
            if pct_str:
                pct = float(pct_str)
                if pct >= 80:
                    return "Correct"
                elif pct >= 40:
                    return "Partial"
                else:
                    return "Incorrect"
        except (ValueError, TypeError):
            pass
    
    return normalized


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3
        self._extraction_stats = {"success": 0, "fallback": 0, "failure": 0}

    def _build_grading_prompt(self, inputs: dict, is_retry: bool = False) -> str:
        """Build a structured prompt for the grading task with chain-of-thought."""
        domain = inputs.get("domain", "unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        base_prompt = f"""You are an expert grader for {domain} problems. Your task is to evaluate a student's answer against the correct solution and grading guidelines.

## Problem Statement
{problem}

## Correct Solution
{solution}

## Grading Guidelines
{guidelines}

## Student's Answer
{student_answer}

## Instructions
1. First, analyze the student's answer step by step. Compare it against the correct solution.
2. Identify any errors, omissions, or alternative valid approaches.
3. Consider the grading guidelines carefully.
4. Provide your reasoning for the grade you will assign.
5. Finally, provide your grade/assessment in the JSON format below.

## Grading Rubric
When assigning grades, use EXACTLY one of these three categories:
- **Correct**: The answer matches the solution or uses an equivalent valid approach with correct reasoning and final result. Award this for fully correct answers.
  - Examples: Complete and accurate solution, correct final answer with valid reasoning, mathematically sound approach
  - Note: Alternative valid approaches that arrive at the correct answer should also be marked as Correct

- **Partial**: The answer shows some correct reasoning but has minor errors, incomplete steps, or partially correct results. Award this for answers that are on the right track but not fully correct.
  - Examples: Correct approach but arithmetic error, incomplete solution with some correct steps, correct setup but wrong final answer, partial understanding demonstrated
  - Note: Award Partial when the student demonstrates understanding of key concepts but fails to complete the solution correctly

- **Incorrect**: The answer contains fundamental errors, wrong approach, or completely wrong results. Award this for answers that are substantially wrong.
  - Examples: Wrong approach entirely, no valid mathematical reasoning, completely incorrect answer with no redeeming qualities, answer to a different problem
  - Note: Only use Incorrect when the answer shows no meaningful understanding of the problem

## Special Considerations for Mathematical Problems
- **Equivalent forms**: Answers in equivalent forms (e.g., 1/2 vs 0.5, √2 vs 2^(1/2)) should be considered correct
- **Alternative methods**: Different valid solution methods should be considered correct if they arrive at the right answer
- **Partial credit**: Award Partial for answers that show understanding of the problem structure but have execution errors
- **Notation**: Minor notation issues should not automatically make an answer Incorrect if the mathematical reasoning is sound
- **Incomplete work**: If the student shows correct work but doesn't state the final answer clearly, consider Partial

## Response Format (REQUIRED - STRICT)
You MUST respond with valid JSON wrapped in <json>...</json> tags. Do not include any text outside the JSON tags.

<json>
{{
    "reasoning": "Your detailed step-by-step analysis and reasoning...",
    "response": "Correct"
}}
</json>

IMPORTANT RULES:
1. The 'response' field MUST contain ONLY one of: "Correct", "Partial", or "Incorrect" (exactly as written, with capital first letter)
2. Do NOT use any other values like "True", "False", numbers, or variations like "correct" (lowercase)
3. The 'reasoning' field should contain your detailed analysis
4. Ensure your JSON is valid - check for proper quotes, commas, and braces
5. Do not include any text before <json> or after </json>
6. Be lenient with equivalent mathematical expressions and alternative valid approaches"""

        if is_retry:
            return f"""ERROR: Your previous response did not contain valid JSON with a 'response' field, or the JSON was malformed, or the response value was not one of the allowed values.

You MUST respond with ONLY valid JSON wrapped in <json>...</json> tags. Do not include any text before or after the JSON tags.

The 'response' field MUST be exactly one of: "Correct", "Partial", or "Incorrect"

Correct format example:
<json>
{{
    "reasoning": "The student correctly applied the quadratic formula and arrived at the right answer. All steps are valid.",
    "response": "Correct"
}}
</json>

Another example for partial credit:
<json>
{{
    "reasoning": "The student started with the right approach but made an arithmetic error in the final step.",
    "response": "Partial"
}}
</json>

Example of incorrect:
<json>
{{
    "reasoning": "The student used an entirely wrong formula and the answer is not close to the correct solution.",
    "response": "Incorrect"
}}
</json>

Now try again with the original task:

{base_prompt}"""
        
        return base_prompt

    def _extract_prediction(self, text: str) -> tuple[str, str, str]:
        """Extract prediction and reasoning from response text.
        
        Returns:
            (prediction, reasoning, method) tuple where method indicates extraction success
        """
        prediction = "None"
        reasoning = ""
        method = "failure"
        
        # Try primary extraction method
        extracted = _extract_jsons(text)
        if extracted:
            method = "success"
        else:
            # Fallback to regex extraction
            extracted = _extract_json_with_regex(text)
            if extracted:
                method = "fallback"
        
        if extracted:
            last_json = extracted[-1]
            if "response" in last_json:
                prediction = str(last_json["response"])
            if "reasoning" in last_json:
                reasoning = str(last_json["reasoning"])
        
        # Normalize the prediction
        prediction = _normalize_prediction(prediction)
        
        return prediction, reasoning, method

    def get_extraction_stats(self) -> dict:
        """Return statistics about extraction methods used."""
        total = sum(self._extraction_stats.values())
        if total == 0:
            return {"total": 0, "success_rate": 0.0}
        return {
            "total": total,
            "success": self._extraction_stats["success"],
            "fallback": self._extraction_stats["fallback"],
            "failure": self._extraction_stats["failure"],
            "success_rate": (self._extraction_stats["success"] + self._extraction_stats["fallback"]) / total,
        }

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_grading_prompt(inputs, is_retry=False)
        
        msg_history = []
        prediction = "None"
        reasoning = ""
        
        # Retry loop for robust extraction
        for attempt in range(self.max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=msg_history if attempt > 0 else [],
                )
                
                last_text = msg_history[-1]["text"] if msg_history else ""
                prediction, reasoning, method = self._extract_prediction(last_text)
                
                # Update statistics
                self._extraction_stats[method] += 1
                
                if prediction != "None":
                    self.log_fn(f"Successfully extracted prediction: {prediction} (method: {method})")
                    if reasoning:
                        self.log_fn(f"Reasoning: {reasoning[:200]}...")
                    break
                else:
                    self.log_fn(f"Attempt {attempt + 1}: Failed to extract prediction, retrying...")
                    # Add feedback for retry
                    if attempt < self.max_retries - 1:
                        instruction = self._build_grading_prompt(inputs, is_retry=True)
                    
            except Exception as e:
                self.log_fn(f"Error in attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    prediction = f"Error: {e}"
        
        # Final validation: ensure prediction is one of the allowed values
        allowed = {"Correct", "Partial", "Incorrect"}
        if prediction not in allowed and not prediction.startswith("Error:"):
            self.log_fn(f"Warning: Invalid prediction '{prediction}', defaulting to 'Incorrect'")
            prediction = "Incorrect"
        
        return str(prediction), msg_history
