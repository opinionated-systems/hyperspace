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

## Detailed Grading Rubric
When assigning grades, use ONLY these three categories:

### CORRECT
Assign when the student answer:
- Matches the solution exactly in both method and final answer
- Uses an equivalent valid approach with correct reasoning and final result
- Has minor notation differences that don't affect mathematical validity
- Contains alternative correct methods (e.g., different but valid proof techniques)
- Has trivial arithmetic errors that don't change the fundamental correctness

### PARTIAL
Assign when the student answer:
- Shows meaningful correct reasoning and significant progress toward the solution
- Demonstrates understanding of key concepts but has gaps in the solution
- Contains computational errors that affect the final answer
- Has incomplete final results but correct approach
- Shows partial understanding with some correct steps mixed with errors
- Has the right idea but fails to execute it fully

### INCORRECT
Assign when the student answer:
- Contains fundamental conceptual errors about the problem type
- Uses a completely wrong approach (e.g., using algebra for a geometry problem requiring proof)
- Shows no meaningful understanding of the problem
- Contains random guesses or completely off-track reasoning
- Has critical errors that invalidate the entire solution
- Is blank or contains no relevant mathematical content

## Examples of Good Grading

Example 1 - Correct (exact match):
<json>
{{
    "reasoning": "The student correctly identified the key theorem (Pythagorean theorem) and applied it step-by-step. Their calculation a² + b² = c² with a=3, b=4 yields c=5, which matches the solution. The notation differs slightly but is mathematically equivalent.",
    "response": "Correct"
}}
</json>

Example 2 - Correct (alternative valid method):
<json>
{{
    "reasoning": "While the solution uses induction, the student correctly applied a direct combinatorial argument. Both approaches are mathematically valid and lead to the same correct result. The student's method is actually more elegant than the provided solution.",
    "response": "Correct"
}}
</json>

Example 3 - Partial (correct approach, computational error):
<json>
{{
    "reasoning": "The student correctly set up the equation 2x + 5 = 15 and understood the need to isolate x. However, they made an arithmetic error in the final step, calculating x = 4 instead of x = 5. The approach and most steps are correct, but the final answer is wrong due to a computational error.",
    "response": "Partial"
}}
</json>

Example 4 - Partial (incomplete solution):
<json>
{{
    "reasoning": "The student correctly identified that this is a modular arithmetic problem and set up the congruence relations properly. They showed good understanding of the concepts and made significant progress. However, they stopped before completing the final calculation to determine the remainder, leaving the solution incomplete.",
    "response": "Partial"
}}
</json>

Example 5 - Incorrect (fundamental conceptual error):
<json>
{{
    "reasoning": "The student attempted to solve this calculus problem using algebraic manipulation instead of integration. They completely missed the fundamental concept that this problem requires finding the area under a curve. The approach is fundamentally wrong for this type of problem.",
    "response": "Incorrect"
}}
</json>

Example 6 - Incorrect (completely wrong approach):
<json>
{{
    "reasoning": "The student used the quadratic formula to solve a number theory problem about prime factorization. This shows a complete misunderstanding of the problem domain. The approach is entirely inappropriate and demonstrates no understanding of the required mathematical concepts.",
    "response": "Incorrect"
}}
</json>

## Response Format (REQUIRED)
You MUST respond with valid JSON wrapped in <json>...</json> tags. Do not include any text outside the JSON tags.

<json>
{{
    "reasoning": "Your detailed step-by-step analysis and reasoning...",
    "response": "Correct"  // Must be exactly one of: "Correct", "Partial", or "Incorrect"
}}
</json>

IMPORTANT: 
- Ensure your JSON is valid and properly formatted
- The 'response' field must contain ONLY one of these three exact values: "Correct", "Partial", or "Incorrect"
- Do not add extra text, explanations, or formatting outside the JSON tags
- Be decisive - choose the grade that best fits the student's work
- Your reasoning should explicitly justify why the grade matches the rubric criteria above"""

    def _extract_prediction(self, text: str) -> tuple[str, str]:
        """Extract prediction and reasoning from response text.
        
        Uses a two-tier extraction strategy:
        1. Primary: JSON tag extraction (_extract_jsons)
        2. Fallback: Regex-based extraction (_extract_json_with_regex)
        
        Tracks extraction statistics for monitoring extraction performance.
        
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
            # Find the best JSON object - prefer one with both response and reasoning
            best_json = None
            for json_obj in extracted:
                if "response" in json_obj:
                    if best_json is None:
                        best_json = json_obj
                    # Prefer objects with both fields
                    elif "reasoning" in json_obj and "reasoning" not in best_json:
                        best_json = json_obj
            
            # Fall back to last JSON if no better match found
            if best_json is None:
                best_json = extracted[-1]
            
            if "response" in best_json:
                prediction = str(best_json["response"]).strip()
                # Normalize common variations to standard grades
                prediction_lower = prediction.lower()
                if prediction_lower in ["correct", "right", "true", "yes", "1", "100%", "accurate", "valid"]:
                    prediction = "Correct"
                elif prediction_lower in ["partial", "partially correct", "partial credit", "half", "0.5", "50%", "somewhat correct", "incomplete"]:
                    prediction = "Partial"
                elif prediction_lower in ["incorrect", "wrong", "false", "no", "0", "0%", "invalid", "error", "mistake"]:
                    prediction = "Incorrect"
                # Handle numeric grades that might be mapped
                elif prediction_lower in ["2", "2/2", "full", "full credit"]:
                    prediction = "Correct"
                elif prediction_lower in ["1", "1/2"]:
                    prediction = "Partial"
                elif prediction_lower in ["0", "0/2", "none"]:
                    prediction = "Incorrect"
            if "reasoning" in best_json:
                reasoning = str(best_json["reasoning"]).strip()
        
        # Validate prediction is one of the allowed values
        if prediction not in ["Correct", "Partial", "Incorrect"]:
            # Try to extract from raw text as last resort
            text_lower = text.lower()
            if '"correct"' in text_lower or "'correct'" in text_lower or 'response": "correct"' in text_lower:
                prediction = "Correct"
            elif '"partial"' in text_lower or "'partial'" in text_lower or 'response": "partial"' in text_lower:
                prediction = "Partial"
            elif '"incorrect"' in text_lower or "'incorrect'" in text_lower or 'response": "incorrect"' in text_lower:
                prediction = "Incorrect"
            else:
                # If we can't determine, mark as None to trigger retry
                prediction = "None"
        
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
        last_raw_response = ""
        
        # Retry loop for robust extraction
        for attempt in range(self.max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=msg_history if attempt > 0 else [],
                )
                
                last_text = msg_history[-1]["text"] if msg_history else ""
                last_raw_response = last_text
                prediction, reasoning = self._extract_prediction(last_text)
                
                if prediction != "None":
                    self.log_fn(f"Successfully extracted prediction: {prediction}")
                    if reasoning:
                        self.log_fn(f"Reasoning: {reasoning[:200]}...")
                    break
                else:
                    self.log_fn(f"Attempt {attempt + 1}: Failed to extract valid prediction, retrying...")
                    # Add feedback for retry with more specific guidance
                    if attempt < self.max_retries - 1:
                        instruction = f"""ERROR: Your previous response could not be parsed correctly.

Your response was:
---
{last_raw_response[:500]}
---

The issue: The response field must contain EXACTLY one of: "Correct", "Partial", or "Incorrect" (with these exact capitalizations).

You MUST respond with ONLY valid JSON wrapped in <json>...</json> tags. Do not include any text before or after the JSON tags.

Correct format:
<json>
{{
    "reasoning": "Your detailed analysis here...",
    "response": "Correct"
}}
</json>

Common mistakes to avoid:
- Do not include markdown formatting (like ```json) inside the <json> tags
- Do not add extra text before or after the JSON
- The response field must be exactly "Correct", "Partial", or "Incorrect" (case-sensitive)
- Do not add quotes around the entire JSON or use escape characters unnecessarily

Now try again with the original task:

{self._build_grading_prompt(inputs)}"""
                    
            except Exception as e:
                self.log_fn(f"Error in attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    prediction = f"Error: {e}"
        
        return str(prediction), msg_history
