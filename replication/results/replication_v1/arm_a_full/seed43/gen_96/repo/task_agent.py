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
            # Try multiple recovery strategies
            parsed = False
            
            # Strategy 1: Try to extract JSON from markdown code blocks within the content
            code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', inner, re.DOTALL)
            if code_block_match:
                try:
                    results.append(json.loads(code_block_match.group(1)))
                    parsed = True
                except json.JSONDecodeError:
                    pass
            
            # Strategy 2: Try to find JSON object directly in the text using JSONDecoder
            if not parsed:
                try:
                    decoder = json.JSONDecoder()
                    idx = 0
                    while idx < len(inner):
                        try:
                            idx = inner.index('{', idx)
                            obj, end_idx = decoder.raw_decode(inner, idx)
                            if isinstance(obj, dict):
                                results.append(obj)
                                parsed = True
                                break
                            idx += end_idx
                        except (ValueError, json.JSONDecodeError):
                            idx += 1
                            continue
                except Exception:
                    pass
            
            # Strategy 3: Try to fix common JSON errors and retry
            if not parsed:
                fixed_json = _attempt_json_repair(inner)
                if fixed_json:
                    try:
                        results.append(json.loads(fixed_json))
                        parsed = True
                    except json.JSONDecodeError:
                        pass
    return results or None


def _attempt_json_repair(text: str) -> str | None:
    """Attempt to repair common JSON formatting errors.
    
    Fixes issues like:
    - Trailing commas in objects/arrays
    - Unquoted keys
    - Single quotes instead of double quotes
    - Missing closing braces/brackets
    """
    if not text or not text.strip():
        return None
    
    text = text.strip()
    
    # Remove any text before the first '{' and after the last '}'
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
        return None
    text = text[start_idx:end_idx+1]
    
    # Fix 1: Remove trailing commas before closing braces/brackets
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    
    # Fix 2: Replace single quotes with double quotes (carefully)
    # Only replace quotes that appear to be delimiting strings
    text = re.sub(r"(?<!\\)'", '"', text)
    
    # Fix 3: Try to fix unquoted keys (simple cases only)
    # Match patterns like {key: or ,key: where key is a simple identifier
    text = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*\s*):', r'\1"\2":', text)
    
    # Fix 4: Balance braces if possible
    open_braces = text.count('{') - text.count('}')
    if open_braces > 0:
        text = text + ('}' * open_braces)
    
    open_brackets = text.count('[') - text.count(']')
    if open_brackets > 0:
        text = text + (']' * open_brackets)
    
    return text


def _extract_json_fallback(text: str) -> dict | None:
    """Fallback JSON extraction for non-tagged JSON or markdown code blocks.

    Tries to find JSON in markdown code blocks or raw JSON objects.
    Uses a robust approach to handle nested braces and includes repair attempts.
    """
    # First try markdown code blocks with various patterns
    patterns = [
        r'```json\s*(.*?)\s*```',  # Markdown JSON blocks
        r'```\s*(\{.*?\})\s*```',  # Generic code blocks with JSON
        r'```\s*([\{\[].*?[\}\]])\s*```',  # Any JSON-like structure in code blocks
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                # Try to repair and parse
                fixed = _attempt_json_repair(match)
                if fixed:
                    try:
                        return json.loads(fixed)
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
            idx = text.index('{', idx)
        except ValueError:
            break
        
        # Try to decode JSON starting at this position
        try:
            obj, end_idx = decoder.raw_decode(text, idx)
            if isinstance(obj, dict):
                json_candidates.append(obj)
            idx += end_idx
        except (json.JSONDecodeError, ValueError):
            # Try to repair this section
            section = text[idx:min(idx+2000, len(text))]
            fixed = _attempt_json_repair(section)
            if fixed:
                try:
                    obj = json.loads(fixed)
                    if isinstance(obj, dict):
                        json_candidates.append(obj)
                except json.JSONDecodeError:
                    pass
            idx += 1
    
    # Prioritize candidates with expected keys
    priority_keys = ["response", "grade", "score", "answer", "reasoning"]
    for key in priority_keys:
        for candidate in json_candidates:
            if key in candidate:
                return candidate
    
    # Return the first valid candidate if any
    if json_candidates:
        return json_candidates[0]
    
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
        "the student should receive", "the student deserves",
        "i assign", "assigned grade:", "final grade:",
        "evaluation:", "assessment:", "verdict:",
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
        prediction = prediction[1:-1].strip() if not prediction.startswith('**') else prediction[2:-2].strip()
        pred_lower = prediction.lower()
    
    # Remove markdown bold/italic markers
    prediction = re.sub(r'\*+', '', prediction).strip()
    pred_lower = prediction.lower()
    
    # Remove HTML tags if present
    prediction = re.sub(r'<[^>]+>', '', prediction).strip()
    pred_lower = prediction.lower()
    
    # Extract expected score patterns from grading guidelines
    guidelines_lower = grading_guidelines.lower()
    
    # Common IMO patterns: "7" (full score), "0" (no score), "1-6" (partial)
    if re.search(r'\b[0-7]\b', grading_guidelines):
        # IMO-style 0-7 scoring - look for single digit 0-7
        match = re.search(r'\b([0-7])\b', prediction)
        if match:
            return match.group(1)
        # Also check for spelled-out numbers
        number_words = {"zero": "0", "one": "1", "two": "2", "three": "3", 
                       "four": "4", "five": "5", "six": "6", "seven": "7",
                       "0 points": "0", "1 point": "1", "2 points": "2", 
                       "3 points": "3", "4 points": "4", "5 points": "5",
                       "6 points": "6", "7 points": "7"}
        for word, digit in number_words.items():
            if re.search(rf'\b{re.escape(word)}\b', pred_lower):
                return digit
    
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
    
    # Check for numeric score patterns (0-100, 0-10, etc.)
    numeric_match = re.search(r'\b(\d+(?:\.\d+)?)\s*(?:/\s*\d+)?\b', prediction)
    if numeric_match:
        score = numeric_match.group(1)
        # Check if it's a valid score in context
        if re.search(r'\b' + re.escape(score) + r'\b', grading_guidelines):
            return score
    
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
        extraction_error = None
        
        try:
            # First try: extract from <json> tags
            extracted = _extract_jsons(last_text)
            if extracted:
                extraction_method = "json_tags"
                # Try to get response field, fall back to other common fields
                last_json = extracted[-1]
                
                # Priority order for response fields
                field_priority = ["response", "grade", "score", "answer", "result", "evaluation", "verdict"]
                found_field = None
                for field in field_priority:
                    if field in last_json:
                        prediction = str(last_json[field])
                        found_field = field
                        break
                
                if not found_field:
                    # If no recognized field, use the first string value found
                    for key, value in last_json.items():
                        if isinstance(value, str):
                            prediction = value
                            found_field = key
                            break
                    if not found_field:
                        # Last resort: use the whole JSON as string
                        prediction = json.dumps(last_json)
                        found_field = "full_json"
                
                self.log_fn(f"Extracted from JSON using field '{found_field}': {prediction}")
            else:
                # Second try: fallback extraction for non-tagged JSON
                fallback = _extract_json_fallback(last_text)
                if fallback:
                    extraction_method = "fallback"
                    field_priority = ["response", "grade", "score", "answer", "result", "evaluation", "verdict"]
                    found_field = None
                    for field in field_priority:
                        if field in fallback:
                            prediction = str(fallback[field])
                            found_field = field
                            break
                    
                    if not found_field:
                        for key, value in fallback.items():
                            if isinstance(value, str):
                                prediction = value
                                found_field = key
                                break
                        if not found_field:
                            prediction = json.dumps(fallback)
                            found_field = "full_json"
                    
                    self.log_fn(f"Used fallback JSON extraction with field '{found_field}': {prediction}")
                else:
                    # Third try: direct text extraction for simple responses
                    extraction_method = "direct"
                    prediction = _extract_from_text(last_text, grading_guidelines)
                    if prediction != "None":
                        self.log_fn(f"Used direct text extraction: {prediction}")
            
            # Validate and normalize the prediction
            original_prediction = prediction
            prediction = _validate_and_normalize_prediction(prediction, grading_guidelines)
            
            if original_prediction != prediction:
                self.log_fn(f"Normalized prediction from '{original_prediction}' to '{prediction}'")
            
            self.log_fn(f"Extraction method: {extraction_method}, final prediction: {prediction}")
            
        except Exception as e:
            extraction_error = str(e)
            self.log_fn(f"Error extracting prediction: {e}")
            # Last resort: try to find any numeric or keyword answer in the text
            prediction = _extract_from_text(last_text, grading_guidelines)
            if prediction != "None":
                self.log_fn(f"Used emergency text extraction after error: {prediction}")

        return str(prediction), msg_history


def _extract_from_text(text: str, grading_guidelines: str) -> str:
    """Extract a prediction from raw text when JSON parsing fails.
    
    Uses pattern matching based on grading guidelines to find likely answers.
    """
    if not text:
        return "None"
    
    # Look for lines that might contain the answer
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Check for IMO scores (0-7) - most common pattern
    score_match = re.search(r'\b([0-7])\b', text)
    if score_match:
        return score_match.group(1)
    
    # Check for "Correct"/"Incorrect" format
    if "correct" in grading_guidelines.lower() or "incorrect" in grading_guidelines.lower():
        text_lower = text.lower()
        if "incorrect" in text_lower or "wrong" in text_lower:
            return "Incorrect"
        elif "correct" in text_lower or "right" in text_lower:
            return "Correct"
    
    # Check for Yes/No format
    if re.search(r'\b(yes|no)\b', grading_guidelines, re.IGNORECASE):
        text_lower = text.lower()
        if re.search(r'\bno\b', text_lower):
            return "No"
        elif re.search(r'\byes\b', text_lower):
            return "Yes"
    
    # Check for Pass/Fail format
    if re.search(r'\b(pass|fail)\b', grading_guidelines, re.IGNORECASE):
        text_lower = text.lower()
        if re.search(r'\bfail\b', text_lower):
            return "Fail"
        elif re.search(r'\bpass\b', text_lower):
            return "Pass"
    
    # Look for the last line that might be the answer
    if lines:
        last_line = lines[-1]
        # Check if the last non-empty line looks like a simple answer
        if len(last_line) < 50 and not last_line.startswith('{') and not last_line.startswith('<'):
            return last_line
        
        # Try to find a line that looks like "Answer: X" or "Result: X"
        for line in reversed(lines):
            match = re.search(r'(?:answer|result|grade|score|prediction)[\s:]+(.+)', line, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                if len(answer) < 50:
                    return answer
    
    return "None"
