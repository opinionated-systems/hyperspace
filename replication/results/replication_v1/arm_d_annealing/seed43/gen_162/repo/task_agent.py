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
    Improved with better brace balancing, Unicode support, and more robust pattern matching.
    """
    results = []
    
    def extract_balanced_json(content: str) -> list[dict]:
        """Extract all balanced JSON objects from content using stack-based parsing.
        
        Handles nested objects, escaped characters, and Unicode content properly.
        """
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
                        json_str = content[start_idx:i+1]
                        # Pre-process: normalize Unicode and fix common issues
                        json_str = json_str.replace('\xa0', ' ')  # Non-breaking space
                        json_str = json_str.replace('\u2018', "'").replace('\u2019', "'")  # Smart quotes
                        json_str = json_str.replace('\u201c', '"').replace('\u201d', '"')  # Smart double quotes
                        json_obj = json.loads(json_str)
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
            r'["\']response["\']\s*:\s*true',
            r'["\']response["\']\s*:\s*false',
            r'response\s*:\s*([\w\s-]+?)(?:\n|$|,)',
        ]
        for pattern in response_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result["response"] = match.group(1).strip() if match.groups() else match.group(0).split(':')[1].strip()
                break
        
        # Look for reasoning pattern with improved handling for multi-line content
        reasoning_patterns = [
            r'["\']reasoning["\']\s*:\s*["\']([^"\']*(?:\.[^"\']*)*)["\']',
            r'["\']reasoning["\']\s*:\s*"([^"]*)"',
            r'["\']reasoning["\']\s*:\s*"([\s\S]*?)"\s*(?:,|\})',
            r'reasoning\s*:\s*(.+?)(?:\n\s*["\']|response\s*:|$)',
        ]
        for pattern in reasoning_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                reasoning_text = match.group(1).strip()
                # Clean up the reasoning text
                reasoning_text = reasoning_text.replace('\\n', '\n').replace('\\t', '\t')
                result["reasoning"] = reasoning_text
                break
        
        if result:
            results.append(result)
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3

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
When assigning grades, consider:
- **Correct**: The answer matches the solution or uses an equivalent valid approach with correct reasoning and final result.
- **Partial**: The answer shows some correct reasoning but has minor errors, incomplete steps, or partially correct results.
- **Incorrect**: The answer contains fundamental errors, wrong approach, or completely wrong results.

## Few-Shot Examples

### Example 1: Correct Answer
Problem: Find the sum of 2+3.
Solution: 2+3 = 5
Student Answer: The sum is 5.

<json>
{{
    "reasoning": "The student's answer correctly identifies the sum as 5, which matches the solution. The reasoning is straightforward and correct.",
    "response": "Correct"
}}
</json>

### Example 2: Partial Answer
Problem: Solve x^2 - 4 = 0.
Solution: x = 2 or x = -2
Student Answer: x = 2

<json>
{{
    "reasoning": "The student found one correct solution (x=2) but missed the second solution (x=-2). This shows partial understanding of the problem.",
    "response": "Partial"
}}
</json>

### Example 3: Incorrect Answer
Problem: Prove that the sum of angles in a triangle is 180 degrees.
Solution: [Standard geometric proof]
Student Answer: The sum is 360 degrees because a square has 360 degrees.

<json>
{{
    "reasoning": "The student's answer is fundamentally incorrect. They confused triangle angle sum with quadrilateral angle sum, demonstrating a misunderstanding of basic geometry concepts.",
    "response": "Incorrect"
}}
</json>

## Response Format (REQUIRED)
You MUST respond with valid JSON wrapped in <json>...</json> tags. Do not include any text outside the JSON tags.

<json>
{{
    "reasoning": "Your detailed step-by-step analysis and reasoning...",
    "response": "Your final grade/assessment (e.g., 'Correct', 'Partial', 'Incorrect', or a numeric score)"
}}
</json>

IMPORTANT: Ensure your JSON is valid and properly formatted. The 'response' field should contain only the grade, not the reasoning."""

    def _extract_prediction(self, text: str) -> tuple[str, str]:
        """Extract prediction and reasoning from response text.
        
        Uses multiple extraction strategies in order of reliability:
        1. Primary: Extract JSON from <json> tags
        2. Fallback: Regex-based JSON extraction for malformed responses
        3. Last resort: Direct pattern matching for response/reasoning fields
        
        Returns:
            (prediction, reasoning) tuple
        """
        prediction = "None"
        reasoning = ""
        
        # Strategy 1: Try primary extraction method (JSON tags)
        extracted = _extract_jsons(text)
        if extracted is None:
            # Strategy 2: Fallback to regex extraction for malformed responses
            extracted = _extract_json_with_regex(text)
        
        if extracted:
            # Use the last valid JSON object (most likely the final answer)
            last_json = extracted[-1]
            if "response" in last_json:
                prediction = str(last_json["response"]).strip()
            if "reasoning" in last_json:
                reasoning = str(last_json["reasoning"]).strip()
        
        # Strategy 3: If still no prediction, try direct pattern extraction
        if prediction == "None":
            # Look for common grade patterns in the text
            grade_patterns = [
                (r'\b(Correct)\b', "Correct"),
                (r'\b(Partial)\b', "Partial"),
                (r'\b(Partially Correct)\b', "Partial"),
                (r'\b(Incorrect)\b', "Incorrect"),
                (r'\bgrade[\s]*:[\s]*([\w\s]+?)(?:\n|$)', None),
            ]
            for pattern, default_val in grade_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    prediction = default_val if default_val else match.group(1).strip()
                    break
        
        return prediction, reasoning

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
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
                    self.log_fn(f"Attempt {attempt + 1}: Failed to extract prediction, retrying...")
                    # Add feedback for retry
                    if attempt < self.max_retries - 1:
                        instruction = f"""ERROR: Your previous response did not contain valid JSON with a 'response' field.

You MUST respond with ONLY valid JSON wrapped in <json>...</json> tags. Do not include any text before or after the JSON tags.

Correct format:
<json>
{{
    "reasoning": "Your detailed analysis here...",
    "response": "Correct"
}}
</json>

Now try again with the original task:

{self._build_grading_prompt(inputs)}"""
                    
            except Exception as e:
                self.log_fn(f"Error in attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    prediction = f"Error: {e}"
        
        return str(prediction), msg_history
