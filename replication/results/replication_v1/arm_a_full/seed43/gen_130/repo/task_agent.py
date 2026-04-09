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
    Includes robust handling for escaped characters and malformed JSON.
    """
    results = []
    search_from = 0
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            # Also try lowercase variant
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
        except json.JSONDecodeError as e:
            # Try to clean up common JSON issues
            cleaned = _clean_json_string(inner)
            try:
                results.append(json.loads(cleaned))
                continue
            except json.JSONDecodeError:
                pass
            
            # Try to extract JSON from markdown code blocks within the content
            code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', inner, re.DOTALL)
            if code_block_match:
                try:
                    results.append(json.loads(code_block_match.group(1)))
                except json.JSONDecodeError:
                    # Try with cleaned version
                    try:
                        cleaned = _clean_json_string(code_block_match.group(1))
                        results.append(json.loads(cleaned))
                    except json.JSONDecodeError:
                        continue
            else:
                # Try to find JSON object directly in the text using decoder
                try:
                    decoder = json.JSONDecoder()
                    idx = 0
                    while idx < len(inner):
                        try:
                            # Skip whitespace and find next brace
                            while idx < len(inner) and inner[idx].isspace():
                                idx += 1
                            if idx >= len(inner) or inner[idx] != '{':
                                idx += 1
                                continue
                            obj, end_idx = decoder.raw_decode(inner, idx)
                            if isinstance(obj, dict):
                                results.append(obj)
                                break
                            idx += end_idx
                        except (ValueError, json.JSONDecodeError):
                            idx += 1
                            continue
                except Exception:
                    continue
    return results or None


def _clean_json_string(text: str) -> str:
    """Clean up common JSON formatting issues.
    
    Handles:
    - Trailing commas in objects/arrays
    - Unescaped newlines in strings
    - Comments (// and /* */)
    - Control characters
    """
    # Remove comments
    # Remove single-line comments
    text = re.sub(r'//.*?$', '', text, flags=re.MULTILINE)
    # Remove multi-line comments
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    
    # Remove control characters except tab, newline, carriage return
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\t\n\r')
    
    # Remove trailing commas before closing braces/brackets
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    
    # Fix unescaped newlines in strings (simple heuristic)
    # This is a best-effort fix
    result = []
    in_string = False
    escape_next = False
    for char in text:
        if escape_next:
            result.append(char)
            escape_next = False
            continue
        if char == '\\':
            result.append(char)
            escape_next = True
            continue
        if char == '"' and not in_string:
            in_string = True
            result.append(char)
        elif char == '"' and in_string:
            in_string = False
            result.append(char)
        elif char in '\n\r' and in_string:
            # Replace unescaped newlines with escaped version
            result.append('\\n')
        else:
            result.append(char)
    
    return ''.join(result)


def _extract_json_fallback(text: str) -> dict | None:
    """Fallback JSON extraction for non-tagged JSON or markdown code blocks.

    Tries to find JSON in markdown code blocks or raw JSON objects.
    Uses a robust approach to handle nested braces and common formatting issues.
    """
    # First try markdown code blocks with various formats
    patterns = [
        r'```json\s*(.*?)\s*```',  # Markdown JSON blocks
        r'```\s*(\{.*?\})\s*```',  # Generic code blocks with JSON
        r'`(\{.*?\})`',  # Inline code with JSON
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                # Try with cleaned version
                try:
                    cleaned = _clean_json_string(match)
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    continue
    
    # Use json.JSONDecoder to find valid JSON objects in text
    # This is more robust than manual brace counting
    decoder = json.JSONDecoder()
    json_candidates = []
    
    idx = 0
    while idx < len(text):
        # Find the next potential JSON start
        try:
            # Skip whitespace
            while idx < len(text) and text[idx].isspace():
                idx += 1
            if idx >= len(text):
                break
            idx = text.index('{', idx)
        except ValueError:
            break
        
        # Try to decode JSON starting at this position
        try:
            obj, end_idx = decoder.raw_decode(text, idx)
            if isinstance(obj, dict) and len(obj) > 0:
                json_candidates.append(obj)
            idx += end_idx
        except (json.JSONDecodeError, ValueError):
            idx += 1
    
    # Also try to find JSON-like structures that might have minor issues
    # Look for patterns like {"key": "value"} even if not perfect JSON
    if not json_candidates:
        # Try to extract from text that looks like JSON but might have issues
        json_like_pattern = r'\{\s*"[^"]+"\s*:\s*"[^"]+"[^}]*\}'
        matches = re.findall(json_like_pattern, text, re.DOTALL)
        for match in matches:
            try:
                cleaned = _clean_json_string(match)
                obj = json.loads(cleaned)
                if isinstance(obj, dict):
                    json_candidates.append(obj)
            except json.JSONDecodeError:
                continue
    
    # Prioritize candidates with expected keys (in order of priority)
    priority_keys = ["response", "grade", "score", "answer", "reasoning", "evaluation"]
    for key in priority_keys:
        for candidate in json_candidates:
            if key in candidate:
                return candidate
    
    # Return the largest valid candidate (most complete JSON object)
    if json_candidates:
        # Prefer the one with most keys (likely most complete)
        return max(json_candidates, key=lambda x: len(x))
    
    return None


def _validate_and_normalize_prediction(prediction: str, grading_guidelines: str) -> str:
    """Validate and normalize the prediction based on grading guidelines.
    
    Ensures the prediction matches expected format from grading guidelines.
    Handles various edge cases and normalizes common variations.
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
        "the student deserves", "i assign", "assigned grade:",
    ]
    pred_lower = prediction.lower()
    for prefix in prefixes_to_remove:
        if pred_lower.startswith(prefix):
            prediction = prediction[len(prefix):].strip()
            pred_lower = prediction.lower()
            # Continue checking for more prefixes
    
    # Remove surrounding quotes and markdown formatting
    while (prediction.startswith('"') and prediction.endswith('"')) or \
          (prediction.startswith("'") and prediction.endswith("'")) or \
          (prediction.startswith('`') and prediction.endswith('`')) or \
          (prediction.startswith('*') and prediction.endswith('*')) or \
          (prediction.startswith('**') and prediction.endswith('**')):
        prediction = prediction[1:-1].strip() if len(prediction) > 2 else prediction.strip('*`')
        pred_lower = prediction.lower()
    
    # Remove markdown bold/italic markers and other formatting
    prediction = re.sub(r'\*+', '', prediction).strip()
    prediction = re.sub(r'_+', '', prediction).strip()  # Remove underscores
    prediction = re.sub(r'\[|\]', '', prediction).strip()  # Remove brackets
    pred_lower = prediction.lower()
    
    # Extract expected score patterns from grading guidelines
    guidelines_lower = grading_guidelines.lower()
    
    # Common IMO patterns: "7" (full score), "0" (no score), "1-6" (partial)
    if re.search(r'\b[0-7]\b', grading_guidelines):
        # IMO-style 0-7 scoring - look for single digit 0-7
        # Try to find standalone digit first
        match = re.search(r'\b([0-7])\b', prediction)
        if match:
            return match.group(1)
        # Also check for spelled-out numbers
        number_words = {"zero": "0", "one": "1", "two": "2", "three": "3", 
                       "four": "4", "five": "5", "six": "6", "seven": "7",
                       "0/7": "0", "1/7": "1", "2/7": "2", "3/7": "3",
                       "4/7": "4", "5/7": "5", "6/7": "6", "7/7": "7"}
        for word, digit in number_words.items():
            if re.search(rf'\b{re.escape(word)}\b', pred_lower):
                return digit
        # Check for patterns like "score of X" or "grade of X"
        score_match = re.search(r'(?:score|grade)\s+of\s+([0-7])', pred_lower)
        if score_match:
            return score_match.group(1)
    
    # Check for "Correct"/"Incorrect" format
    if "correct" in guidelines_lower or "incorrect" in guidelines_lower:
        # Check for explicit correct/incorrect mentions (prioritize negative)
        if re.search(r'\b(incorrect|wrong|false|invalid|error)\b', pred_lower):
            return "Incorrect"
        elif re.search(r'\b(correct|right|true|valid|accurate)\b', pred_lower):
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
    
    # Check for True/False format
    if re.search(r'\b(true|false)\b', grading_guidelines, re.IGNORECASE):
        if re.search(r'\btrue\b', pred_lower):
            return "True"
        elif re.search(r'\bfalse\b', pred_lower):
            return "False"
    
    # Check for numeric ranges in guidelines (e.g., "0-10", "1-5")
    range_match = re.search(r'(\d+)\s*-\s*(\d+)', grading_guidelines)
    if range_match:
        min_val, max_val = int(range_match.group(1)), int(range_match.group(2))
        # Look for any number in that range
        num_match = re.search(r'\b(\d+)\b', prediction)
        if num_match:
            val = int(num_match.group(1))
            if min_val <= val <= max_val:
                return str(val)
    
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
4. The JSON must be valid and properly escaped.
5. Wrap your entire JSON response in <json>...</json> tags.

IMPORTANT GRADING PRINCIPLES:
- Be objective and consistent with the official solution
- Award partial credit when the student demonstrates understanding of key concepts
- Deduct points for logical errors, missing steps, or incorrect conclusions
- Consider alternative valid approaches that differ from the official solution
- The response field should be a simple, unformatted value matching the expected format in the grading guidelines"""

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
                prediction = _extract_prediction_from_json(last_json)
            else:
                # Second try: fallback extraction for non-tagged JSON
                fallback = _extract_json_fallback(last_text)
                if fallback:
                    extraction_method = "fallback"
                    prediction = _extract_prediction_from_json(fallback)
                    self.log_fn(f"Used fallback JSON extraction: {prediction}")
                else:
                    # Third try: direct text extraction for simple responses
                    extraction_method = "direct"
                    prediction = _extract_prediction_from_text(last_text, grading_guidelines)
                    if prediction != "None":
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
            prediction = _extract_prediction_from_text(last_text, grading_guidelines)
            if prediction != "None":
                self.log_fn(f"Used emergency regex extraction: {prediction}")

        return str(prediction), msg_history


def _extract_prediction_from_json(json_obj: dict) -> str:
    """Extract prediction value from a JSON object.
    
    Tries multiple common field names in order of priority.
    """
    # Priority order for field names
    priority_fields = ["response", "grade", "score", "answer", "evaluation", "result", "prediction"]
    
    for field in priority_fields:
        if field in json_obj:
            value = json_obj[field]
            # Handle different value types
            if isinstance(value, str):
                return value.strip()
            elif isinstance(value, (int, float)):
                return str(value)
            elif isinstance(value, bool):
                return "Correct" if value else "Incorrect"
            else:
                return str(value)
    
    # If no recognized field, return the whole JSON as string
    return json.dumps(json_obj)


def _extract_prediction_from_text(text: str, grading_guidelines: str) -> str:
    """Extract prediction directly from text when JSON extraction fails.
    
    Uses pattern matching based on grading guidelines format.
    """
    guidelines_lower = grading_guidelines.lower()
    text_lower = text.lower()
    
    # Look for IMO scores (0-7)
    if re.search(r'\b[0-7]\b', grading_guidelines):
        # Try to find score patterns
        patterns = [
            r'(?:grade|score)\s*(?:is|:|=)\s*([0-7])',
            r'(?:assigned|give|award)\s+(?:a\s+)?(?:score|grade)\s+of\s+([0-7])',
            r'(?:the\s+)?(?:student\s+)?(?:deserves|gets|receives)\s+(?:a\s+)?([0-7])',
            r'\bfinal\s+(?:grade|score)\s*(?::|=)?\s*([0-7])\b',
            r'\b(?:grade|score)\s*([0-7])\s*(?:/\s*7)?\b',
        ]
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                return match.group(1)
        # Simple digit search as last resort
        match = re.search(r'\b([0-7])\b', text)
        if match:
            return match.group(1)
    
    # Check for Correct/Incorrect
    if "correct" in guidelines_lower or "incorrect" in guidelines_lower:
        if re.search(r'\b(incorrect|wrong|false|invalid|error)\b', text_lower):
            return "Incorrect"
        elif re.search(r'\b(correct|right|true|valid|accurate)\b', text_lower):
            return "Correct"
    
    # Check for Yes/No
    if re.search(r'\b(yes|no)\b', grading_guidelines, re.IGNORECASE):
        if re.search(r'\byes\b', text_lower):
            return "Yes"
        elif re.search(r'\bno\b', text_lower):
            return "No"
    
    # Check for Pass/Fail
    if re.search(r'\b(pass|fail)\b', grading_guidelines, re.IGNORECASE):
        if re.search(r'\bpass\b', text_lower):
            return "Pass"
        elif re.search(r'\bfail\b', text_lower):
            return "Fail"
    
    # Check for True/False
    if re.search(r'\b(true|false)\b', grading_guidelines, re.IGNORECASE):
        if re.search(r'\btrue\b', text_lower):
            return "True"
        elif re.search(r'\bfalse\b', text_lower):
            return "False"
    
    # Look for the last line that might be the answer
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if lines:
        last_line = lines[-1]
        # Check if the last non-empty line looks like a simple answer
        if len(last_line) < 50 and not last_line.startswith('{') and not last_line.startswith('<'):
            return last_line
    
    return "None"
