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
                    continue
            else:
                continue
    return results or None


def _extract_json_fallback(text: str) -> dict | None:
    """Fallback JSON extraction for non-tagged JSON or markdown code blocks.

    Tries to find JSON in markdown code blocks or raw JSON objects.
    Uses a robust approach to handle nested braces.
    """
    # First try markdown code blocks
    patterns = [
        r'```json\s*(.*?)\s*```',  # Markdown JSON blocks
        r'```\s*(\{.*?\})\s*```',  # Generic code blocks with JSON
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    
    # Use a more efficient approach: find all JSON-like substrings and validate
    # Look for patterns that look like JSON objects with expected keys
    json_candidates = []
    text_len = len(text)
    
    # Find all potential JSON object starts (looking for {" or {\n})
    for match in re.finditer(r'\{\s*"', text):
        start = match.start()
        # Use a stack-based approach to find the matching end brace
        # Limit search window to prevent excessive scanning on malformed input
        search_end = min(start + 10000, text_len)
        stack = 0
        in_string = False
        escape_next = False
        
        for i in range(start, search_end):
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
                    stack += 1
                elif char == '}':
                    stack -= 1
                    if stack == 0:
                        # Found a complete JSON object
                        candidate = text[start:i+1]
                        json_candidates.append(candidate)
                        break
                    elif stack < 0:
                        # Unbalanced braces, skip
                        break
    
    # Try to parse each candidate with improved scoring
    best_match = None
    best_score = -1
    priority_keys = ["response", "grade", "score", "answer", "reasoning", "confidence"]
    
    for candidate in json_candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                # Score based on presence of priority keys
                score = sum(2 for key in priority_keys if key in parsed)
                # Bonus for having multiple keys
                if score >= 4:
                    score += 1
                if score > best_score:
                    best_score = score
                    best_match = parsed
                    # Early exit on perfect match
                    if score >= 6:
                        break
        except json.JSONDecodeError:
            continue
    
    return best_match


def _validate_and_normalize_prediction(prediction: str, grading_guidelines: str) -> str:
    """Validate and normalize the prediction based on grading guidelines.
    
    Ensures the prediction matches expected format from grading guidelines.
    Handles various edge cases and normalizes common variations with improved
    robustness for markdown formatting, multi-line responses, and edge cases.
    """
    if not prediction or prediction.strip() == "":
        return "None"
    
    original = prediction
    prediction = prediction.strip()
    
    # Remove markdown code blocks and HTML-like tags
    prediction = re.sub(r'```[\w]*\n?', '', prediction)
    prediction = re.sub(r'```', '', prediction)
    prediction = re.sub(r'<[^>]+>', '', prediction)
    
    # Remove surrounding quotes and brackets
    prediction = re.sub(r'^[\s"\'`\[\{]+', '', prediction)
    prediction = re.sub(r'[\s"\'`\]\}]+$', '', prediction)
    prediction = prediction.strip()
    
    # Remove common prefixes that LLMs sometimes add (case-insensitive)
    prefixes_to_remove = [
        r'^the answer is[\s:]+',
        r'^answer[\s:]*',
        r'^score[\s:]*',
        r'^grade[\s:]*',
        r'^final answer[\s:]*',
        r'^prediction[\s:]*',
        r'^result[\s:]*',
        r'^output[\s:]*',
        r'^response[\s:]*',
    ]
    pred_lower = prediction.lower()
    for pattern in prefixes_to_remove:
        match = re.search(pattern, pred_lower, re.IGNORECASE)
        if match:
            prediction = prediction[match.end():].strip()
            pred_lower = prediction.lower()
    
    # Extract expected score patterns from grading guidelines
    # Common IMO patterns: "7" (full score), "0" (no score), "1-6" (partial)
    if re.search(r'\b[0-7]\b', grading_guidelines):
        # IMO-style 0-7 scoring - look for explicit score mentions first
        score_match = re.search(r'(?:score|grade|rating)[\s:=]+([0-7])\b', prediction, re.IGNORECASE)
        if score_match:
            return score_match.group(1)
        
        # Look for single digit 0-7 with word boundaries
        match = re.search(r'(?:^|\s|\b)([0-7])(?:\s*$|\s*\.|\s*[,;]|\s+)', prediction)
        if match:
            return match.group(1)
        
        # Fallback: any standalone 0-7
        match = re.search(r'\b([0-7])\b', prediction)
        if match:
            return match.group(1)
        
        # Also check for spelled-out numbers
        number_words = {"zero": "0", "one": "1", "two": "2", "three": "3", 
                       "four": "4", "five": "5", "six": "6", "seven": "7"}
        for word, digit in number_words.items():
            if re.search(rf'\b{word}\b', pred_lower):
                return digit
    
    # Check for "Correct"/"Incorrect" format
    if "correct" in grading_guidelines.lower() or "incorrect" in grading_guidelines.lower():
        # Check for explicit incorrect mentions first (more specific)
        if re.search(r'\b(incorrect|wrong|false|error)\b', pred_lower):
            return "Incorrect"
        elif re.search(r'\b(correct|right|true|valid|accurate)\b', pred_lower):
            return "Correct"
    
    # Check for Yes/No format with improved patterns
    if re.search(r'\b(yes|no)\b', grading_guidelines, re.IGNORECASE):
        # More specific matching to avoid partial word matches
        if re.search(r'\byes\b', pred_lower) and not re.search(r'\bno\b', pred_lower):
            return "Yes"
        elif re.search(r'\bno\b', pred_lower) and not re.search(r'\byes\b', pred_lower):
            return "No"
        # If both appear, use the first one
        yes_pos = pred_lower.find("yes") if "yes" in pred_lower else float('inf')
        no_pos = pred_lower.find("no") if "no" in pred_lower else float('inf')
        if yes_pos < no_pos:
            return "Yes"
        elif no_pos < yes_pos:
            return "No"
    
    # Check for Pass/Fail format
    if re.search(r'\b(pass|fail)\b', grading_guidelines, re.IGNORECASE):
        if re.search(r'\bpass\b', pred_lower) and not re.search(r'\bfail\b', pred_lower):
            return "Pass"
        elif re.search(r'\bfail\b', pred_lower) and not re.search(r'\bpass\b', pred_lower):
            return "Fail"
        # If both appear, use the first one
        pass_pos = pred_lower.find("pass") if "pass" in pred_lower else float('inf')
        fail_pos = pred_lower.find("fail") if "fail" in pred_lower else float('inf')
        if pass_pos < fail_pos:
            return "Pass"
        elif fail_pos < pass_pos:
            return "Fail"
    
    # If we stripped everything useful, return the cleaned version or original
    return prediction if prediction else original


def _extract_confidence_score(text: str) -> float | None:
    """Extract confidence score from text if present.
    
    Looks for patterns like "confidence: 0.85" or "confidence score: 85%"
    Returns a float between 0 and 1, or None if not found.
    """
    # Look for confidence patterns
    patterns = [
        r'confidence[:\s]+(\d+\.?\d*)\s*%?',
        r'confidence score[:\s]+(\d+\.?\d*)\s*%?',
        r'confidence[:\s]+(\d+)%',
    ]
    
    text_lower = text.lower()
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            try:
                value = float(match.group(1))
                # Normalize percentage to 0-1 range
                if value > 1.0:
                    value = value / 100.0
                return max(0.0, min(1.0, value))  # Clamp to [0, 1]
            except ValueError:
                continue
    
    return None


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
    "response": "The final grade/score as specified in the grading guidelines",
    "confidence": 0.85
}}
</json>

CRITICAL INSTRUCTIONS:
1. The "response" field must contain ONLY the grade/score value (e.g., "7", "2", "0", "Correct", "Incorrect", etc.) exactly as specified in the grading guidelines.
2. The "confidence" field is OPTIONAL. If provided, it should be a number between 0 and 1 indicating your confidence in the grade.
3. Do NOT add explanations, reasoning, or extra text in the "response" field.
4. Do NOT use markdown formatting, code blocks, or any other formatting inside the JSON.
5. The JSON must be valid and properly escaped.
6. Wrap your entire JSON response in <json>...</json> tags."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        confidence = None
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
                
                # Extract confidence if available
                if "confidence" in last_json:
                    try:
                        confidence = float(last_json["confidence"])
                    except (ValueError, TypeError):
                        pass
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
                    
                    # Extract confidence if available
                    if "confidence" in fallback:
                        try:
                            confidence = float(fallback["confidence"])
                        except (ValueError, TypeError):
                            pass
                    
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
            
            # Try to extract confidence from reasoning text if not in JSON
            if confidence is None:
                confidence = _extract_confidence_score(last_text)
            
            # Validate and normalize the prediction
            original_prediction = prediction
            prediction = _validate_and_normalize_prediction(prediction, grading_guidelines)
            
            if original_prediction != prediction:
                self.log_fn(f"Normalized prediction from '{original_prediction}' to '{prediction}'")
            
            # Log confidence if extracted
            if confidence is not None:
                self.log_fn(f"Extraction method: {extraction_method}, final prediction: {prediction}, confidence: {confidence:.2f}")
            else:
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
