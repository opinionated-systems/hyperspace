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

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles nested JSON objects within the content.
    Includes robust error recovery for malformed JSON.
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
        
        # Try to parse the inner content as JSON
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks within the content
            code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', inner, re.DOTALL)
            if code_block_match:
                try:
                    results.append(json.loads(code_block_match.group(1)))
                except json.JSONDecodeError:
                    pass
            
            # Try to find JSON object directly in the text using JSONDecoder
            try:
                decoder = json.JSONDecoder()
                idx = 0
                while idx < len(inner):
                    try:
                        # Find next opening brace
                        brace_idx = inner.find('{', idx)
                        if brace_idx == -1:
                            break
                        # Try to decode JSON starting at this position
                        obj, end_idx = decoder.raw_decode(inner, brace_idx)
                        if isinstance(obj, dict):
                            results.append(obj)
                            break
                        idx = brace_idx + end_idx
                    except (ValueError, json.JSONDecodeError):
                        idx += 1
                        continue
            except Exception:
                pass
            
            # Last resort: Try to fix common JSON errors
            try:
                # Fix trailing commas before closing braces/brackets
                fixed = re.sub(r',(\s*[}\]])', r'\1', inner)
                # Fix single quotes to double quotes (simple cases)
                fixed = re.sub(r"'([^']*?)':", r'"\1":', fixed)
                fixed = re.sub(r":\s*'([^']*?)'", r': "\1"', fixed)
                # Fix unescaped newlines in strings
                fixed = re.sub(r'("[^"]*?)\n([^"]*?")', r'\1\\n\2', fixed)
                obj = json.loads(fixed)
                if isinstance(obj, dict):
                    results.append(obj)
            except Exception:
                pass
                
    return results or None


def _extract_json_fallback(text: str) -> dict | None:
    """Fallback JSON extraction for non-tagged JSON or markdown code blocks.

    Tries to find JSON in markdown code blocks or raw JSON objects.
    Uses a robust approach to handle nested braces and common JSON errors.
    """
    # First try markdown code blocks
    patterns = [
        r'```json\s*(.*?)\s*```',  # Markdown JSON blocks
        r'```\s*(\{.*?\})\s*```',  # Generic code blocks with JSON
        r'```\s*(\[.*?\])\s*```',  # JSON arrays in code blocks
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, dict):
                    return parsed
                elif isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[-1], dict):
                    # If it's an array of objects, return the last one
                    return parsed[-1]
            except json.JSONDecodeError:
                # Try to fix common JSON errors
                try:
                    fixed = _fix_common_json_errors(match)
                    parsed = json.loads(fixed)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    continue
    
    # Use json.JSONDecoder to find valid JSON objects in text
    decoder = json.JSONDecoder()
    json_candidates = []
    
    idx = 0
    while idx < len(text):
        # Find the next potential JSON start
        try:
            idx = text.find('{', idx)
            if idx == -1:
                break
        except ValueError:
            break
        
        # Try to decode JSON starting at this position
        try:
            obj, end_idx = decoder.raw_decode(text, idx)
            if isinstance(obj, dict):
                json_candidates.append(obj)
            idx += end_idx
        except (json.JSONDecodeError, ValueError):
            idx += 1
    
    # Also try to find JSON arrays that might contain objects
    idx = 0
    while idx < len(text):
        try:
            idx = text.find('[', idx)
            if idx == -1:
                break
        except ValueError:
            break
        
        try:
            obj, end_idx = decoder.raw_decode(text, idx)
            if isinstance(obj, list) and len(obj) > 0:
                # Extract dict objects from the array
                for item in reversed(obj):  # Prefer later items
                    if isinstance(item, dict):
                        json_candidates.append(item)
            idx += end_idx
        except (json.JSONDecodeError, ValueError):
            idx += 1
    
    # Prioritize candidates with expected keys
    priority_keys = ["response", "grade", "score", "answer", "reasoning", "evaluation"]
    for key in priority_keys:
        for candidate in json_candidates:
            if key in candidate:
                return candidate
    
    # Return the first valid candidate if any
    if json_candidates:
        return json_candidates[0]
    
    return None


def _fix_common_json_errors(text: str) -> str:
    """Fix common JSON formatting errors that LLMs produce."""
    # Fix trailing commas before closing braces/brackets
    fixed = re.sub(r',(\s*[}\]])', r'\1', text)
    # Fix single quotes to double quotes (simple cases)
    fixed = re.sub(r"'([^']*?)':", r'"\1":', fixed)
    fixed = re.sub(r":\s*'([^']*?)'", r': "\1"', fixed)
    # Fix unescaped newlines in strings
    fixed = re.sub(r'("[^"]*?)\n([^"]*?")', r'\1\\n\2', fixed)
    # Fix unescaped tabs in strings
    fixed = re.sub(r'("[^"]*?)\t([^"]*?")', r'\1\\t\2', fixed)
    # Fix unescaped backslashes (but not already escaped ones)
    fixed = re.sub(r'(?<!\\)\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', fixed)
    return fixed


def _validate_and_normalize_prediction(prediction: str, grading_guidelines: str) -> str:
    """Validate and normalize the prediction based on grading guidelines.
    
    Ensures the prediction matches expected format from grading guidelines.
    Handles various edge cases and normalizes common variations.
    Includes comprehensive pattern matching for different grading formats.
    """
    if not prediction or prediction.strip() == "":
        return "None"
    
    original = prediction
    prediction = prediction.strip()
    
    # Remove common prefixes that LLMs sometimes add (case-insensitive matching)
    prefixes_to_remove = [
        "the answer is", "answer:", "score:", "grade:", 
        "final answer:", "prediction:", "result:", "output:",
        "the final answer is", "the grade is", "the score is",
        "therefore,", "thus,", "so,", "hence,",
        "i conclude that", "my conclusion is", "in conclusion",
        "the student deserves", "the student should receive",
        "i assign", "assigned grade:", "final grade:",
    ]
    pred_lower = prediction.lower()
    for prefix in prefixes_to_remove:
        if pred_lower.startswith(prefix):
            prediction = prediction[len(prefix):].strip()
            pred_lower = prediction.lower()
    
    # Remove surrounding quotes and markdown formatting
    while (prediction.startswith('"') and prediction.endswith('"')) or \
          (prediction.startswith("'") and prediction.endswith("'")) or \
          (prediction.startswith('`') and prediction.endswith('`')) or \
          (prediction.startswith('*') and prediction.endswith('*')):
        prediction = prediction[1:-1].strip()
        pred_lower = prediction.lower()
    
    # Remove markdown bold/italic markers
    prediction = re.sub(r'\*+', '', prediction).strip()
    pred_lower = prediction.lower()
    
    # Remove trailing punctuation that might be added
    prediction = re.sub(r'[.!?]+$', '', prediction).strip()
    pred_lower = prediction.lower()
    
    # Extract expected score patterns from grading guidelines
    guidelines_lower = grading_guidelines.lower()
    
    # IMO-style 0-7 scoring - look for single digit 0-7
    if re.search(r'\b[0-7]\b', grading_guidelines):
        # First try to find a standalone digit 0-7
        match = re.search(r'\b([0-7])\b', prediction)
        if match:
            return match.group(1)
        # Also check for spelled-out numbers
        number_words = {
            "zero": "0", "one": "1", "two": "2", "three": "3", 
            "four": "4", "five": "5", "six": "6", "seven": "7",
            "0/7": "0", "1/7": "1", "2/7": "2", "3/7": "3",
            "4/7": "4", "5/7": "5", "6/7": "6", "7/7": "7",
        }
        for word, digit in number_words.items():
            if re.search(rf'\b{re.escape(word)}\b', pred_lower):
                return digit
        # Check for patterns like "X out of 7" or "X/7"
        out_of_match = re.search(r'\b([0-7])\s*(?:out\s+of\s+7|/\s*7)\b', pred_lower)
        if out_of_match:
            return out_of_match.group(1)
    
    # Check for "Correct"/"Incorrect" format
    if "correct" in guidelines_lower or "incorrect" in guidelines_lower:
        # Check for explicit correct/incorrect mentions (prioritize negative)
        if re.search(r'\b(incorrect|wrong|false|invalid|error|mistake)\b', pred_lower):
            return "Incorrect"
        elif re.search(r'\b(correct|right|true|valid|accurate)\b', pred_lower):
            return "Correct"
        # Check for partial credit indicators
        if re.search(r'\b(partial|partially|incomplete|some)\b', pred_lower):
            # For binary correct/incorrect, partial usually means incorrect
            if "partial" not in guidelines_lower:
                return "Incorrect"
    
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
    
    # Check for letter grades (A, B, C, D, F)
    if re.search(r'\b[ABCDF][+-]?\b', grading_guidelines):
        letter_match = re.search(r'\b([ABCDF][+-]?)\b', prediction.upper())
        if letter_match:
            return letter_match.group(1)
    
    # Check for percentage format (0-100%)
    if '%' in grading_guidelines or 'percent' in guidelines_lower:
        percent_match = re.search(r'\b(\d{1,3})\s*%', prediction)
        if percent_match:
            val = int(percent_match.group(1))
            if 0 <= val <= 100:
                return f"{val}%"
    
    # Check for numeric ranges in guidelines (e.g., "0-10", "1 to 5")
    range_match = re.search(r'\b(\d+)\s*(?:-|to|–)\s*(\d+)\b', grading_guidelines)
    if range_match:
        min_val, max_val = int(range_match.group(1)), int(range_match.group(2))
        # Look for any number in that range
        numbers_found = re.findall(r'\b(\d+)\b', prediction)
        for num_str in numbers_found:
            val = int(num_str)
            if min_val <= val <= max_val:
                return num_str
    
    # Log normalization for debugging if changed
    if prediction != original.strip():
        logger.debug(f"Normalized prediction from '{original}' to '{prediction}'")
    
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
4. The JSON must be valid and properly escaped (no unescaped newlines or quotes inside strings).
5. Wrap your entire JSON response in <json>...</json> tags.
6. Ensure the JSON is properly formatted with double quotes for all keys and string values.

EXAMPLE CORRECT RESPONSE:
<json>
{{
    "reasoning": "The student correctly identified the key theorem and applied it appropriately. The proof is complete and well-structured.",
    "response": "7"
}}
</json>

EXAMPLE INCORRECT RESPONSE:
<json>
{{
    "reasoning": "The student made some progress but missed the key insight.",
    "response": "The student deserves partial credit of 3 points."
}}
</json>
(This is incorrect because the response field contains explanatory text instead of just the grade value.)"""

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
                # Priority order for response fields
                response_fields = ["response", "grade", "score", "answer", "evaluation", "result", "output"]
                for field in response_fields:
                    if field in last_json:
                        prediction = str(last_json[field])
                        break
                else:
                    # If no recognized field, use the whole JSON as string
                    prediction = json.dumps(last_json)
            else:
                # Second try: fallback extraction for non-tagged JSON
                fallback = _extract_json_fallback(last_text)
                if fallback:
                    extraction_method = "fallback"
                    response_fields = ["response", "grade", "score", "answer", "evaluation", "result", "output"]
                    for field in response_fields:
                        if field in fallback:
                            prediction = str(fallback[field])
                            break
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
                else:
                    # Try to find any numeric value
                    any_num = re.search(r'\b(\d+)\b', last_text)
                    if any_num:
                        prediction = any_num.group(1)
                        self.log_fn(f"Used regex extraction for number: {prediction}")
            except Exception as e2:
                self.log_fn(f"Last resort extraction also failed: {e2}")

        return str(prediction), msg_history
