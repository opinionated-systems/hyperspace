"""
Task agent: solves a given task with chain-of-thought reasoning.

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

    Uses a robust brace-matching approach to handle nested JSON structures
    and avoids the lazy regex bug that truncates content with nested braces.
    Also handles markdown code blocks within the content.
    
    Args:
        text: The text containing <json>...</json> blocks.
        
    Returns:
        A list of parsed JSON dicts, or None if no valid JSON found.
    """
    results = []
    search_from = 0
    extraction_attempts = 0
    
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end_tag_start = text.find("</json>", start)
        if end_tag_start == -1:
            logger.warning("Found opening <json> tag but no closing </json> tag")
            break
            
        # Extract content between tags
        inner_start = start + 6
        inner = text[inner_start:end_tag_start].strip()
        search_from = end_tag_start + 7
        extraction_attempts += 1
        
        # Try direct JSON parse first
        try:
            parsed = json.loads(inner)
            results.append(parsed)
            logger.debug(f"Successfully parsed JSON directly (attempt {extraction_attempts})")
            continue
        except json.JSONDecodeError as e:
            logger.debug(f"Direct JSON parse failed (attempt {extraction_attempts}): {e}")
            pass
        
        # Try to extract JSON from markdown code blocks
        code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        code_blocks = re.findall(code_block_pattern, inner, re.DOTALL)
        for block_idx, block in enumerate(code_blocks):
            try:
                parsed = json.loads(block.strip())
                results.append(parsed)
                logger.debug(f"Successfully parsed JSON from code block {block_idx + 1} (attempt {extraction_attempts})")
                break  # Successfully parsed, move to next <json> block
            except json.JSONDecodeError as e:
                logger.debug(f"Code block {block_idx + 1} parse failed (attempt {extraction_attempts}): {e}")
                continue
        else:
            # No valid JSON in code blocks, try brace matching
            json_obj = _extract_json_by_brace_matching(inner)
            if json_obj:
                results.append(json_obj)
                logger.debug(f"Successfully parsed JSON via brace matching (attempt {extraction_attempts})")
    
    if results:
        logger.info(f"Extracted {len(results)} JSON object(s) from {extraction_attempts} <json> block(s)")
    else:
        logger.warning(f"No valid JSON found in {extraction_attempts} <json> block(s)")
    
    return results or None


def _extract_json_by_brace_matching(text: str) -> dict | None:
    """Extract a JSON object using brace counting for proper nesting.
    
    This handles nested JSON structures correctly by tracking brace depth.
    
    Args:
        text: The text to search for JSON objects.
        
    Returns:
        A parsed JSON dict if found, None otherwise.
    """
    # Find all potential JSON object starts
    for match in re.finditer(r'\{\s*"', text):
        start = match.start()
        brace_count = 0
        in_string = False
        escape_next = False
        
        for i in range(start, len(text)):
            char = text[i]
            
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\':
                escape_next = True
                continue
            
            if char == '"' and not in_string:
                in_string = True
            elif char == '"' and in_string:
                in_string = False
            elif not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        # Found complete JSON object
                        try:
                            return json.loads(text[start:i+1])
                        except json.JSONDecodeError:
                            break  # Try next candidate
        
    return None


def _extract_json_fallback(text: str) -> dict | None:
    """Fallback JSON extraction for non-tagged JSON or markdown code blocks.

    Tries to find JSON in markdown code blocks or raw JSON objects.
    Uses a robust brace-matching approach to handle nested structures.
    
    Args:
        text: The text to search for JSON objects.
        
    Returns:
        A parsed JSON dict if found, None otherwise.
        
    Priority:
        1. Markdown code blocks with JSON
        2. Raw JSON objects with expected keys (response, grade, score, etc.)
        3. Any valid JSON object
    """
    # First try markdown code blocks
    patterns = [
        r'```json\s*([\s\S]*?)\s*```',  # Markdown JSON blocks
        r'```\s*([\s\S]*?)\s*```',  # Generic code blocks
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue
    
    # Try to find JSON objects by matching braces with proper nesting
    json_candidates = []
    
    # Find all potential JSON object starts and extract using brace matching
    for match in re.finditer(r'\{\s*"', text):
        candidate = _extract_json_by_brace_matching(text[match.start():])
        if candidate and candidate not in json_candidates:
            json_candidates.append(candidate)
    
    # Try to parse each candidate, preferring ones with expected keys
    expected_keys = ["response", "grade", "score", "answer", "reasoning", "result"]
    for candidate in json_candidates:
        if any(key in candidate for key in expected_keys):
            return candidate
    
    # If no prioritized candidate found, return the first valid one
    if json_candidates:
        return json_candidates[0]
    
    return None


def _validate_and_normalize_prediction(prediction: str, grading_guidelines: str) -> str:
    """Validate and normalize the prediction based on grading guidelines.
    
    Ensures the prediction matches expected format from grading guidelines.
    Handles various edge cases and normalizes common variations.
    
    Args:
        prediction: The raw prediction string from the LLM
        grading_guidelines: The grading guidelines to validate against
        
    Returns:
        A normalized prediction string
    """
    original_prediction = prediction
    
    if not prediction or prediction.strip() == "":
        logger.warning("Empty prediction received, returning 'None'")
        return "None"
    
    prediction = prediction.strip()
    
    # Remove common prefixes that LLMs sometimes add
    prefixes_to_remove = [
        "the answer is", "answer:", "score:", "grade:", 
        "final answer:", "prediction:", "result:", "output:",
        "i think", "my answer is", "the grade is", "the score is",
        "therefore,", "thus,", "so,", "hence,", "consequently,",
        "in conclusion,", "to summarize,", "in summary,"
    ]
    pred_lower = prediction.lower()
    for prefix in prefixes_to_remove:
        if pred_lower.startswith(prefix):
            prediction = prediction[len(prefix):].strip()
            pred_lower = prediction.lower()
            logger.debug(f"Removed prefix '{prefix}', new prediction: '{prediction}'")
            break
    
    # Remove surrounding quotes if present
    if (prediction.startswith('"') and prediction.endswith('"')) or \
       (prediction.startswith("'") and prediction.endswith("'")):
        prediction = prediction[1:-1].strip()
        pred_lower = prediction.lower()
        logger.debug(f"Removed surrounding quotes, new prediction: '{prediction}'")
    
    # Remove trailing punctuation that might confuse matching
    prediction = prediction.rstrip('.;,!?')
    pred_lower = prediction.lower()
    
    # Extract expected score patterns from grading guidelines
    # Common IMO patterns: "7" (full score), "0" (no score), "1-6" (partial)
    if re.search(r'\b[0-7]\b', grading_guidelines):
        # IMO-style 0-7 scoring - look for single digit 0-7
        match = re.search(r'\b([0-7])\b', prediction)
        if match:
            normalized = match.group(1)
            if original_prediction.strip() != normalized:
                logger.info(f"Normalized IMO score from '{original_prediction}' to '{normalized}'")
            return normalized
        # Also check for spelled-out numbers
        number_words = {"zero": "0", "one": "1", "two": "2", "three": "3", 
                       "four": "4", "five": "5", "six": "6", "seven": "7",
                       "0 points": "0", "1 point": "1", "2 points": "2", 
                       "3 points": "3", "4 points": "4", "5 points": "5",
                       "6 points": "6", "7 points": "7", "full marks": "7",
                       "no marks": "0", "partial credit": "3"}
        for word, digit in number_words.items():
            if re.search(rf'\b{re.escape(word)}\b', pred_lower):
                logger.info(f"Normalized number word '{word}' to '{digit}'")
                return digit
    
    # Check for "Correct"/"Incorrect" format
    if "correct" in grading_guidelines.lower() or "incorrect" in grading_guidelines.lower():
        pred_lower = prediction.lower()
        # Check for explicit correct/incorrect mentions
        if "incorrect" in pred_lower or "wrong" in pred_lower or "false" in pred_lower or "error" in pred_lower:
            if original_prediction.strip().lower() != "incorrect":
                logger.info(f"Normalized to 'Incorrect' from '{original_prediction}'")
            return "Incorrect"
        elif "correct" in pred_lower or "right" in pred_lower or "true" in pred_lower or "valid" in pred_lower:
            if original_prediction.strip().lower() != "correct":
                logger.info(f"Normalized to 'Correct' from '{original_prediction}'")
            return "Correct"
    
    # Check for Yes/No format
    if re.search(r'\b(yes|no)\b', grading_guidelines, re.IGNORECASE):
        if re.search(r'\byes\b', pred_lower):
            return "Yes"
        elif re.search(r'\bno\b', pred_lower):
            return "No"
    
    # Check for Pass/Fail format
    if re.search(r'\b(pass|fail)\b', grading_guidelines, re.IGNORECASE):
        if re.search(r'\bpass\b', pred_lower):
            return "Pass"
        elif re.search(r'\bfail\b', pred_lower):
            return "Fail"
    
    # Check for numeric ranges in guidelines (e.g., "0-100", "1-10")
    range_match = re.search(r'\b(\d+)\s*[-–—]\s*(\d+)\b', grading_guidelines)
    if range_match:
        min_val, max_val = int(range_match.group(1)), int(range_match.group(2))
        # Try to find a number in the prediction within the valid range
        num_match = re.search(r'\b(\d+)\b', prediction)
        if num_match:
            val = int(num_match.group(1))
            if min_val <= val <= max_val:
                return str(val)
    
    # Handle decimal/float predictions (preserve precision)
    decimal_match = re.search(r'\b(\d+\.\d+)\b', prediction)
    if decimal_match:
        return decimal_match.group(1)
    
    # Handle percentage predictions
    if "%" in prediction or "percent" in pred_lower:
        pct_match = re.search(r'(\d+(?:\.\d+)?)\s*%', prediction)
        if pct_match:
            return pct_match.group(1) + "%"
    
    # Handle JSON-like strings that weren't properly parsed
    if prediction.startswith('{') and prediction.endswith('}'):
        try:
            parsed = json.loads(prediction)
            for key in ["response", "grade", "score", "answer", "result", "value"]:
                if key in parsed:
                    return str(parsed[key]).strip()
        except json.JSONDecodeError:
            pass
    
    # Clean up common formatting issues
    prediction = prediction.strip('"\'[]{}()')  # Remove surrounding brackets/quotes
    prediction = prediction.replace("\\", "")  # Remove escape characters
    
    if original_prediction.strip() != prediction:
        logger.info(f"Normalized prediction from '{original_prediction}' to '{prediction}'")
    
    return prediction


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract fields for structured prompting
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert grader for the International Mathematical Olympiad (IMO). Your task is to evaluate a student's answer to a mathematical problem with precision and consistency.

## Domain
{domain}

## Problem Statement
{problem}

## Official Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Your Task

Please evaluate the student's answer following this structured approach:

### Step 1: Problem Analysis
- Summarize what the problem is asking
- Identify key mathematical concepts and required techniques
- Note any critical assumptions or constraints

### Step 2: Solution Mapping
- Break down the official solution into key milestones
- Identify which milestones are essential vs. optional
- Note common alternative valid approaches

### Step 3: Student Answer Evaluation
- Map the student's work to solution milestones
- Identify correct steps with clear justification
- Flag any errors, gaps, or logical flaws
- Check for partial credit opportunities per guidelines

### Step 4: Grade Determination
- Apply grading guidelines systematically
- Consider: correctness, completeness, and clarity
- Assign the most appropriate grade/score

Respond ONLY in JSON format wrapped in <json> tags with the following exact schema:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis following the 4 steps above...",
    "response": "The final grade/score as specified in the grading guidelines"
}}
</json>

CRITICAL INSTRUCTIONS:
1. The "response" field must contain ONLY the grade/score value (e.g., "7", "2", "0", "Correct", "Incorrect", etc.) exactly as specified in the grading guidelines.
2. Do NOT add explanations, reasoning, or extra text in the "response" field.
3. Do NOT use markdown formatting, code blocks, or any other formatting inside the JSON.
4. The JSON must be valid and properly escaped.
5. Wrap your entire JSON response in <json>...</json> tags."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        last_text = msg_history[-1]["text"] if msg_history else ""
        extraction_method = "none"
        
        try:
            # First try: extract from <json> tags
            extracted = _extract_jsons(last_text)
            if extracted:
                extraction_method = "json_tags"
                # Try to get response field, fall back to other common fields
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = str(last_json["response"])
                elif "grade" in last_json:
                    prediction = str(last_json["grade"])
                elif "score" in last_json:
                    prediction = str(last_json["score"])
                elif "answer" in last_json:
                    prediction = str(last_json["answer"])
                else:
                    # If no recognized field, use the whole JSON as string
                    prediction = json.dumps(last_json)
            else:
                # Second try: fallback extraction for non-tagged JSON
                fallback = _extract_json_fallback(last_text)
                if fallback:
                    extraction_method = "fallback"
                    if "response" in fallback:
                        prediction = str(fallback["response"])
                    elif "grade" in fallback:
                        prediction = str(fallback["grade"])
                    elif "score" in fallback:
                        prediction = str(fallback["score"])
                    elif "answer" in fallback:
                        prediction = str(fallback["answer"])
                    else:
                        prediction = json.dumps(fallback)
                    self.log_fn(f"Used fallback JSON extraction: {prediction}")
                else:
                    # Third try: direct text extraction for simple responses
                    extraction_method = "direct"
                    # Look for the last line that might be the answer
                    lines = [line.strip() for line in last_text.split('\n') if line.strip()]
                    if lines:
                        # Check if the last non-empty line looks like a simple answer
                        last_line = lines[-1]
                        if len(last_line) < 50 and not last_line.startswith('{') and not last_line.startswith('<'):
                            prediction = last_line
                            self.log_fn(f"Used direct text extraction: {prediction}")
            
            # Validate and normalize the prediction
            original_prediction = prediction
            prediction = _validate_and_normalize_prediction(prediction, grading_guidelines)
            
            if original_prediction != prediction:
                self.log_fn(f"Normalized prediction from '{original_prediction}' to '{prediction}'")
            
            self.log_fn(f"Extraction method: {extraction_method}, final prediction: {prediction}")
            
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Last resort: try to find any numeric or keyword answer in the text
            try:
                # Look for IMO scores (0-7)
                score_match = re.search(r'\b([0-7])\b', last_text)
                if score_match:
                    prediction = score_match.group(1)
                    self.log_fn(f"Used regex extraction for score: {prediction}")
            except Exception:
                pass

        return str(prediction), msg_history
