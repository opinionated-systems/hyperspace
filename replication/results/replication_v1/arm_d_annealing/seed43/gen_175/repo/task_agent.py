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
    Also handles markdown code blocks with json tag.
    """
    results = []
    search_from = 0
    blocks_found = 0
    
    # First, try to find <json>...</json> blocks
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
            # Try to clean up common JSON issues
            cleaned = _clean_json_string(inner)
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict):
                    results.append(parsed)
                    logger.debug(f"Successfully parsed JSON block #{blocks_found} after cleaning")
                    continue
            except json.JSONDecodeError:
                pass
            
            # Log the error with context for debugging
            preview = inner[:100].replace('\n', ' ')
            logger.debug(f"JSON decode error in block #{blocks_found}: {e}. Content preview: {preview}...")
            continue
    
    # If no <json> blocks found, try markdown code blocks
    if not results:
        import re
        markdown_pattern = r'```(?:json)?\s*\n?\s*(\{[\s\S]*?\})\s*\n?```'
        for match in re.finditer(markdown_pattern, text):
            blocks_found += 1
            inner = match.group(1).strip()
            if not inner:
                continue
            try:
                parsed = json.loads(inner)
                if isinstance(parsed, dict):
                    results.append(parsed)
                    logger.debug(f"Successfully parsed markdown JSON block #{blocks_found}")
            except json.JSONDecodeError:
                # Try cleaning
                cleaned = _clean_json_string(inner)
                try:
                    parsed = json.loads(cleaned)
                    if isinstance(parsed, dict):
                        results.append(parsed)
                        logger.debug(f"Successfully parsed markdown JSON block #{blocks_found} after cleaning")
                except json.JSONDecodeError:
                    pass
    
    if blocks_found > 0 and not results:
        logger.warning(f"Found {blocks_found} JSON blocks but none parsed successfully")
    
    return results or None


def _clean_json_string(text: str) -> str:
    """Clean up common JSON formatting issues."""
    # Remove trailing commas before closing braces/brackets
    import re
    text = re.sub(r',\s*([}\]])', r'\1', text)
    # Fix single quotes to double quotes (carefully)
    text = re.sub(r"(?<!\\)'", '"', text)
    # Remove comments
    text = re.sub(r'//.*?\n', '\n', text)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    return text.strip()


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
                        # Try cleaning before giving up
                        cleaned = _clean_json_string(content[start_idx:i+1])
                        try:
                            json_obj = json.loads(cleaned)
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
            r'["\']response["\']\s*:\s*([\d.]+)',
            r'response\s*:\s*([\w\s-]+)(?:\n|$)',
            r'response["\']?\s*[=:]\s*["\']?([^"\'\n,}]+)',
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
            r'["\']reasoning["\']\s*:\s*\'([^\']*)\'',
            r'reasoning\s*:\s*(.+?)(?:\n\s*["\']|$)',
            r'reasoning["\']?\s*:\s*"([^"]+)"',
            r'reasoning["\']?\s*[=:]\s*["\']?([^"\'\n,}]+)',
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
        
        # Determine expected response format from guidelines
        expected_format = self._detect_expected_format(guidelines)
        
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

## Response Format (REQUIRED)
You MUST respond with valid JSON wrapped in <json>...</json> tags. Do not include any text outside the JSON tags.

<json>
{{
    "reasoning": "Your detailed step-by-step analysis and reasoning...",
    "response": "Your final grade/assessment{expected_format}"
}}
</json>

IMPORTANT: Ensure your JSON is valid and properly formatted. The 'response' field should contain only the grade, not the reasoning."""

    def _detect_expected_format(self, guidelines: str) -> str:
        """Detect the expected response format from grading guidelines."""
        guidelines_lower = guidelines.lower()
        
        # Check for numeric score patterns
        if any(pattern in guidelines_lower for pattern in ["score", "points", "out of", "/", "0-", "1-", "2-", "3-", "4-", "5-", "6-", "7-", "8-", "9-", "10-"]):
            # Look for specific score ranges
            import re
            # Try to find patterns like "X out of Y", "X/Y", "score of X", etc.
            score_patterns = [
                r'(\d+)\s*out of\s*(\d+)',
                r'(\d+)\s*/\s*(\d+)',
                r'score\s*(?:of|:)?\s*(\d+)',
                r'(\d+)\s*points',
                r'0\s*[-–]\s*(\d+)',
            ]
            for pattern in score_patterns:
                match = re.search(pattern, guidelines_lower)
                if match:
                    max_score = match.group(2) if match.group(2) else match.group(1)
                    return f" (e.g., a numeric score from 0-{max_score})"
            return " (e.g., a numeric score)"
        
        # Check for letter grade patterns
        if any(pattern in guidelines_lower for pattern in ["grade", "a+", "a-", "b+", "b-", "c+", "c-", "d+", "d-", "f"]):
            return " (e.g., 'A', 'B', 'C', 'D', 'F' or 'A+', 'A-', etc.)"
        
        # Check for percentage patterns
        if any(pattern in guidelines_lower for pattern in ["percent", "%"]):
            return " (e.g., a percentage like '85%' or 'Pass/Fail')"
        
        # Default to categorical
        return " (e.g., 'Correct', 'Partial', 'Incorrect', or a numeric score)"

    def _extract_prediction(self, text: str) -> tuple[str, str]:
        """Extract prediction and reasoning from response text.
        
        Returns:
            (prediction, reasoning) tuple
        """
        prediction = "None"
        reasoning = ""
        
        # Try primary extraction method
        extracted = _extract_jsons(text)
        if extracted is None:
            # Fallback to regex extraction
            extracted = _extract_json_with_regex(text)
        
        if extracted:
            last_json = extracted[-1]
            if "response" in last_json:
                prediction = str(last_json["response"])
            if "reasoning" in last_json:
                reasoning = str(last_json["reasoning"])
        
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
                    # Add feedback for retry with more specific guidance
                    if attempt < self.max_retries - 1:
                        # Check what went wrong and provide targeted feedback
                        if "<json>" not in last_text:
                            error_msg = "ERROR: Your response did not include <json>...</json> tags."
                        elif "</json>" not in last_text:
                            error_msg = "ERROR: Your response had an opening <json> tag but no closing </json> tag."
                        elif '"response"' not in last_text:
                            error_msg = "ERROR: Your JSON is missing the required 'response' field."
                        elif '"reasoning"' not in last_text:
                            error_msg = "ERROR: Your JSON is missing the 'reasoning' field (optional but recommended)."
                        else:
                            error_msg = "ERROR: Your JSON could not be parsed. Check for syntax errors like trailing commas or unclosed quotes."
                        
                        instruction = f"""{error_msg}

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
