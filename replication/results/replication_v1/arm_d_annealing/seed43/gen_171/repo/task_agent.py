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
    Includes detailed logging for debugging extraction failures.
    """
    results = []
    search_from = 0
    blocks_found = 0
    
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            logger.debug(f"Unclosed <json> tag found at position {start}")
            break
        
        blocks_found += 1
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        # Skip empty blocks
        if not inner:
            logger.debug(f"Empty JSON block #{blocks_found} found, skipping")
            continue
            
        try:
            parsed = json.loads(inner)
            if isinstance(parsed, dict):
                results.append(parsed)
                logger.debug(f"Successfully parsed JSON block #{blocks_found}")
            else:
                logger.debug(f"JSON block #{blocks_found} is not a dict, skipping")
        except json.JSONDecodeError as e:
            # Log the error with context for debugging
            preview = inner[:100].replace('\n', ' ')
            logger.debug(f"JSON decode error in block #{blocks_found}: {e}. Content preview: {preview}...")
            continue
    
    if blocks_found > 0 and not results:
        logger.warning(f"Found {blocks_found} JSON blocks but none parsed successfully")
    
    return results or None


def _extract_json_with_regex(text: str) -> list[dict] | None:
    """Fallback JSON extraction using regex for malformed responses.
    
    Handles nested braces, code blocks, and various formatting edge cases.
    Improved with better brace balancing and more robust pattern matching.
    """
    results = []
    
    def extract_balanced_json(content: str) -> list[dict]:
        """Extract all balanced JSON objects from content using stack-based parsing."""
        objects = []
        brace_count = 0
        start_idx = -1
        in_string = False
        escape_next = False
        
        for i, char in enumerate(content):
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"' and not in_string:
                in_string = True
                continue
            if char == '"' and in_string:
                in_string = False
                continue
            if in_string:
                continue
            
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    try:
                        json_obj = json.loads(content[start_idx:i+1])
                        if isinstance(json_obj, dict):
                            objects.append(json_obj)
                    except json.JSONDecodeError:
                        pass
                    start_idx = -1
        return objects
    
    # Try to find JSON objects in code blocks first
    code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    for match in re.finditer(code_block_pattern, text, re.DOTALL):
        content = match.group(1).strip()
        objects = extract_balanced_json(content)
        results.extend(objects)
    
    # If no results from code blocks, try to find any balanced JSON object in text
    if not results:
        results = extract_balanced_json(text)
    
    # Final fallback: try to find key-value patterns for response and reasoning
    if not results:
        result = {}
        # Look for response pattern with more flexible matching
        response_patterns = [
            r'["\']response["\']\s*:\s*["\']([^"\']+)["\']',
            r'["\']response["\']\s*:\s*(\d+)',
            r'response\s*:\s*([\w\s-]+)(?:\n|$)',
        ]
        for pattern in response_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result["response"] = match.group(1).strip()
                break
        
        # Look for reasoning pattern
        reasoning_patterns = [
            r'["\']reasoning["\']\s*:\s*["\']([^"\']*(?:\.[^"\']*)*)["\']',
            r'["\']reasoning["\']\s*:\s*"([^"]*)"',
            r'reasoning\s*:\s*(.+?)(?:\n\s*["\']|$)',
        ]
        for pattern in reasoning_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                result["reasoning"] = match.group(1).strip()
                break
        
        if result:
            results.append(result)
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning.
    
    This agent uses a structured prompting approach with JSON output format
    to evaluate student answers against correct solutions. It includes robust
    extraction logic with multiple fallback strategies for parsing LLM responses.
    
    Attributes:
        model: The LLM model to use for grading
        log_fn: Logging function for agent activity
        max_retries: Maximum number of retry attempts for failed extractions
    """

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3
        self._extraction_stats = {"success": 0, "fallback": 0, "failure": 0}

    def _build_grading_prompt(self, inputs: dict) -> str:
        """Build a structured prompt for the grading task with chain-of-thought."""
        domain = inputs.get("domain", "unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        return f"""You are an expert grader for {domain} problems. Your task is to evaluate a student's answer against the correct solution and grading guidelines.

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
When assigning grades, use ONLY one of these three categories:
- **Correct**: The answer matches the solution exactly OR uses an equivalent valid approach with correct reasoning and final result. Award this when the student demonstrates full understanding.
- **Partial**: The answer shows some correct reasoning but has minor errors, incomplete steps, or partially correct results. Award this when the student demonstrates partial understanding but doesn't fully solve the problem.
- **Incorrect**: The answer contains fundamental errors, wrong approach, or completely wrong results. Award this when the student demonstrates little to no understanding of the problem.

## Response Format (REQUIRED)
You MUST respond with valid JSON wrapped in <json>...</json> tags. Do not include any text outside the JSON tags.

<json>
{{
    "reasoning": "Your detailed step-by-step analysis and reasoning...",
    "response": "Correct" or "Partial" or "Incorrect"
}}
</json>

IMPORTANT: 
- Ensure your JSON is valid and properly formatted.
- The 'response' field MUST contain ONLY one of: "Correct", "Partial", or "Incorrect" (case-sensitive).
- Do not include any other text, explanations, or formatting in the 'response' field.
- Be decisive - choose the single best grade that matches the student's performance."""

    def _normalize_grade(self, grade: str) -> str:
        """Normalize grade to standard format.
        
        Converts various grade formats to a consistent set of standard grades:
        - 'Correct', 'correct', 'right', 'true', 'yes', 'valid', 'accurate' -> 'Correct'
        - 'Partial', 'partial', 'partially correct', 'partial credit' -> 'Partial'
        - 'Incorrect', 'incorrect', 'wrong', 'false', 'no', 'invalid', 'error' -> 'Incorrect'
        - Numeric scores are preserved as-is
        
        Args:
            grade: Raw grade string from LLM response
            
        Returns:
            Normalized grade string
        """
        if not grade or grade == "None":
            return "None"
        
        grade_lower = grade.lower().strip()
        
        # Check for numeric grades (preserve these)
        try:
            # Try to parse as a number
            float(grade_lower)
            return grade.strip()  # Return original numeric grade
        except ValueError:
            pass
        
        # Check for percentage format (preserve these)
        if grade_lower.endswith('%'):
            try:
                float(grade_lower[:-1])
                return grade.strip()
            except ValueError:
                pass
        
        # Normalize to standard categories
        # Check for partial credit first (more specific patterns)
        partial_variants = ['partially correct', 'partial credit', 'partial', 'half', 'incomplete', 'some']
        for variant in partial_variants:
            if variant in grade_lower:
                return "Partial"
        
        # Check for incorrect (before correct to avoid matching "incorrect" as "correct")
        incorrect_variants = ['incorrect', 'wrong', 'false', 'no', 'invalid', 'error', 'bad', 'unacceptable']
        for variant in incorrect_variants:
            if variant in grade_lower:
                return "Incorrect"
        
        # Check for correct
        correct_variants = ['correct', 'right', 'true', 'yes', 'valid', 'accurate', 'acceptable', 'good']
        for variant in correct_variants:
            if variant in grade_lower:
                return "Correct"
        
        # If no match found, return original with warning
        self.log_fn(f"Warning: Unrecognized grade format '{grade}', returning as-is")
        return grade.strip()

    def _validate_grade(self, grade: str) -> bool:
        """Validate that a grade is in a recognized format.
        
        Args:
            grade: Grade string to validate
            
        Returns:
            True if grade is valid, False otherwise
        """
        if not grade or grade == "None":
            return False
        
        # Standard categories
        standard_grades = ['correct', 'partial', 'incorrect']
        
        grade_lower = grade.lower().strip()
        
        # Check standard categories
        if grade_lower in standard_grades:
            return True
        
        # Check for numeric grades (0-100 or 0.0-1.0)
        try:
            val = float(grade_lower)
            if 0 <= val <= 100 or 0 <= val <= 1:
                return True
        except ValueError:
            pass
        
        # Check for percentage format
        if grade_lower.endswith('%'):
            try:
                val = float(grade_lower[:-1])
                if 0 <= val <= 100:
                    return True
            except ValueError:
                pass
        
        return False

    def _extract_prediction(self, text: str) -> tuple[str, str]:
        """Extract prediction and reasoning from response text.
        
        Uses a two-tier extraction strategy:
        1. Primary: JSON tag extraction (_extract_jsons)
        2. Fallback: Regex-based extraction (_extract_json_with_regex)
        
        Tracks extraction statistics for monitoring extraction performance.
        Includes grade normalization and validation for consistent output.
        
        Args:
            text: Raw LLM response text
            
        Returns:
            (prediction, reasoning) tuple where prediction defaults to "None"
            if extraction fails
        """
        prediction = "None"
        reasoning = ""
        
        # Try primary extraction method
        extracted = _extract_jsons(text)
        if extracted is None:
            # Fallback to regex extraction
            extracted = _extract_json_with_regex(text)
            if extracted:
                self._extraction_stats["fallback"] += 1
                self.log_fn(f"Used fallback extraction for this response")
            else:
                self._extraction_stats["failure"] += 1
        else:
            self._extraction_stats["success"] += 1
        
        if extracted:
            last_json = extracted[-1]
            if "response" in last_json:
                raw_prediction = str(last_json["response"])
                # Normalize and validate the grade
                normalized = self._normalize_grade(raw_prediction)
                if self._validate_grade(normalized):
                    prediction = normalized
                else:
                    # If validation fails, try to extract from the raw text
                    self.log_fn(f"Grade validation failed for '{raw_prediction}', attempting recovery")
                    prediction = self._normalize_grade(raw_prediction)
            if "reasoning" in last_json:
                reasoning = str(last_json["reasoning"])
        
        return prediction, reasoning

    def get_extraction_stats(self) -> dict[str, int]:
        """Return extraction statistics for monitoring.
        
        Returns:
            Dictionary with success, fallback, and failure counts
        """
        return self._extraction_stats.copy()

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single grading problem.

        Executes the grading workflow with retry logic for robust extraction.
        Will make up to max_retries attempts if extraction fails.

        Args:
            inputs: dict containing:
                - domain: Problem domain (e.g., "mathematics")
                - problem: The problem statement
                - solution: The correct solution
                - grading_guidelines: Guidelines for grading
                - student_answer: The student's answer to evaluate

        Returns:
            (prediction, msg_history) tuple where prediction is the grade
            and msg_history is the conversation history
        """
        instruction = self._build_grading_prompt(inputs)
        
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
                prediction, reasoning = self._extract_prediction(last_text)
                
                if prediction != "None":
                    self.log_fn(f"Successfully extracted prediction: {prediction}")
                    if reasoning:
                        self.log_fn(f"Reasoning: {reasoning[:200]}...")
                    break
                else:
                    self.log_fn(f"Attempt {attempt + 1}: Failed to extract valid prediction, retrying...")
                    # Add feedback for retry
                    if attempt < self.max_retries - 1:
                        instruction = f"""ERROR: Your previous response did not contain valid JSON with a 'response' field containing one of: "Correct", "Partial", or "Incorrect".

You MUST respond with ONLY valid JSON wrapped in <json>...</json> tags. Do not include any text before or after the JSON tags.

Correct format:
<json>
{{
    "reasoning": "Your detailed analysis here...",
    "response": "Correct"
}}
</json>

The 'response' field MUST contain exactly one of these three values (case-sensitive):
- "Correct" - if the student's answer is fully correct
- "Partial" - if the student's answer is partially correct  
- "Incorrect" - if the student's answer is wrong

Now try again with the original task:

{self._build_grading_prompt(inputs)}"""
                    
            except Exception as e:
                self.log_fn(f"Error in attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    prediction = f"Error: {e}"
        
        return str(prediction), msg_history
