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
    Enhanced to handle markdown code blocks and nested structures.
    Includes detailed logging for debugging extraction failures.
    """
    results = []
    search_from = 0
    extraction_attempts = 0
    
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            logger.warning(f"Found <json> tag at position {start} but no closing </json> tag")
            break
        
        inner = text[start + 6:end].strip()
        search_from = end + 7
        extraction_attempts += 1
        
        # Try to parse the inner content as JSON
        try:
            parsed = json.loads(inner)
            results.append(parsed)
            logger.debug(f"Successfully parsed JSON from <json> block #{extraction_attempts}")
        except json.JSONDecodeError as e:
            logger.debug(f"JSONDecodeError in <json> block #{extraction_attempts}: {e}")
            
            # Try to extract JSON from markdown code blocks within the content
            code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', inner, re.DOTALL)
            if code_block_match:
                try:
                    parsed = json.loads(code_block_match.group(1))
                    results.append(parsed)
                    logger.debug(f"Successfully parsed JSON from markdown code block in <json> block #{extraction_attempts}")
                except json.JSONDecodeError as e2:
                    logger.debug(f"Failed to parse JSON from markdown code block: {e2}")
            
            # Try to find JSON objects using brace matching as fallback
            if not results or not any(r for r in results if r):
                try:
                    json_objects = _extract_all_json_objects(inner)
                    if json_objects:
                        results.extend(json_objects)
                        logger.debug(f"Successfully extracted {len(json_objects)} JSON object(s) using brace matching")
                except Exception as e3:
                    logger.debug(f"Brace matching extraction failed: {e3}")
            
            if not results or not any(r for r in results if r):
                logger.warning(f"Failed to extract valid JSON from <json> block #{extraction_attempts}")
                continue
    
    if extraction_attempts > 0 and not results:
        logger.warning(f"Attempted to extract JSON from {extraction_attempts} <json> block(s) but found no valid JSON")
    elif results:
        logger.info(f"Successfully extracted {len(results)} JSON object(s) from {extraction_attempts} <json> block(s)")
    
    return results or None


def _extract_all_json_objects(text: str) -> list[dict]:
    """Extract all valid JSON objects from text using brace matching.
    
    This is a more robust extractor that finds JSON objects by tracking
    brace depth, handling nested objects and escaped characters properly.
    Includes logging for debugging extraction issues.
    """
    results = []
    i = 0
    n = len(text)
    candidates_found = 0
    
    while i < n:
        # Find the start of a potential JSON object
        if text[i] == '{':
            start = i
            brace_depth = 1
            in_string = False
            escape_next = False
            j = i + 1
            
            while j < n and brace_depth > 0:
                char = text[j]
                
                if escape_next:
                    escape_next = False
                    j += 1
                    continue
                
                if char == '\\':
                    escape_next = True
                    j += 1
                    continue
                
                if char == '"' and not in_string:
                    in_string = True
                elif char == '"' and in_string:
                    in_string = False
                elif not in_string:
                    if char == '{':
                        brace_depth += 1
                    elif char == '}':
                        brace_depth -= 1
                
                j += 1
            
            if brace_depth == 0:
                # Found a complete JSON object
                candidates_found += 1
                candidate = text[start:j]
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        results.append(parsed)
                        logger.debug(f"Successfully parsed JSON object #{candidates_found} using brace matching")
                except json.JSONDecodeError as e:
                    logger.debug(f"Failed to parse JSON candidate #{candidates_found}: {e}")
            else:
                logger.debug(f"Unbalanced braces in JSON candidate starting at position {start}")
            
            i = j
        else:
            i += 1
    
    if candidates_found > 0:
        logger.debug(f"Brace matching: found {candidates_found} candidates, successfully parsed {len(results)}")
    
    return results


def _extract_json_fallback(text: str) -> dict | None:
    """Fallback JSON extraction for non-tagged JSON or markdown code blocks.

    Tries to find JSON in markdown code blocks or raw JSON objects.
    Uses a robust approach to handle nested braces.
    Includes detailed logging for debugging extraction issues.
    """
    logger.debug("Attempting fallback JSON extraction")
    
    # First try markdown code blocks
    patterns = [
        r'```json\s*(.*?)\s*```',  # Markdown JSON blocks
        r'```\s*(\{.*?\})\s*```',  # Generic code blocks with JSON
    ]
    for pattern_idx, pattern in enumerate(patterns):
        matches = re.findall(pattern, text, re.DOTALL)
        logger.debug(f"Pattern {pattern_idx + 1}: found {len(matches)} potential matches")
        for match_idx, match in enumerate(matches):
            try:
                parsed = json.loads(match)
                logger.info(f"Successfully parsed JSON from markdown code block (pattern {pattern_idx + 1}, match {match_idx + 1})")
                return parsed
            except json.JSONDecodeError as e:
                logger.debug(f"Failed to parse match {match_idx + 1} from pattern {pattern_idx + 1}: {e}")
                continue
    
    # Use the robust brace-matching extractor to find all JSON objects
    logger.debug("Attempting brace-matching extraction for fallback")
    all_jsons = _extract_all_json_objects(text)
    
    # Try to find one with expected keys first
    expected_keys = ["response", "grade", "score", "answer", "reasoning"]
    for parsed in all_jsons:
        found_keys = [key for key in expected_keys if key in parsed]
        if found_keys:
            logger.info(f"Found JSON with expected keys {found_keys} using brace matching")
            return parsed
    
    # If no prioritized candidate found, return the first valid one
    if all_jsons:
        logger.info(f"Returning first valid JSON from brace matching ({len(all_jsons)} total found)")
        return all_jsons[0]
    
    logger.warning("Fallback JSON extraction failed: no valid JSON found")
    return None


def _validate_and_normalize_prediction(prediction: str, grading_guidelines: str) -> str:
    """Validate and normalize the prediction based on grading guidelines.
    
    Ensures the prediction matches expected format from grading guidelines.
    Handles various edge cases and normalizes common variations.
    Includes logging for debugging normalization decisions.
    """
    original_prediction = prediction
    
    if not prediction or prediction.strip() == "":
        logger.debug("Prediction is empty, returning 'None'")
        return "None"
    
    prediction = prediction.strip()
    
    # Remove common prefixes that LLMs sometimes add
    prefixes_to_remove = [
        "the answer is", "answer:", "score:", "grade:", 
        "final answer:", "prediction:", "result:", "output:"
    ]
    pred_lower = prediction.lower()
    prefix_removed = None
    for prefix in prefixes_to_remove:
        if pred_lower.startswith(prefix):
            prediction = prediction[len(prefix):].strip()
            pred_lower = prediction.lower()
            prefix_removed = prefix
            break
    
    if prefix_removed:
        logger.debug(f"Removed prefix '{prefix_removed}' from prediction")
    
    # Extract expected score patterns from grading guidelines
    # Common IMO patterns: "7" (full score), "0" (no score), "1-6" (partial)
    if re.search(r'\b[0-7]\b', grading_guidelines):
        # IMO-style 0-7 scoring - look for single digit 0-7
        match = re.search(r'\b([0-7])\b', prediction)
        if match:
            result = match.group(1)
            logger.debug(f"Normalized IMO score: '{original_prediction}' -> '{result}'")
            return result
        # Also check for spelled-out numbers
        number_words = {"zero": "0", "one": "1", "two": "2", "three": "3", 
                       "four": "4", "five": "5", "six": "6", "seven": "7"}
        for word, digit in number_words.items():
            if re.search(rf'\b{word}\b', pred_lower):
                logger.debug(f"Normalized spelled-out number '{word}' to '{digit}'")
                return digit
    
    # Check for "Correct"/"Incorrect" format
    if "correct" in grading_guidelines.lower() or "incorrect" in grading_guidelines.lower():
        pred_lower = prediction.lower()
        # Check for explicit correct/incorrect mentions
        if "incorrect" in pred_lower or "wrong" in pred_lower or "false" in pred_lower:
            logger.debug(f"Normalized to 'Incorrect' from: '{original_prediction}'")
            return "Incorrect"
        elif "correct" in pred_lower or "right" in pred_lower or "true" in pred_lower:
            logger.debug(f"Normalized to 'Correct' from: '{original_prediction}'")
            return "Correct"
    
    # Check for Yes/No format
    if re.search(r'\b(yes|no)\b', grading_guidelines, re.IGNORECASE):
        if re.search(r'\byes\b', pred_lower):
            logger.debug(f"Normalized to 'Yes' from: '{original_prediction}'")
            return "Yes"
        elif re.search(r'\bno\b', pred_lower):
            logger.debug(f"Normalized to 'No' from: '{original_prediction}'")
            return "No"
    
    # Check for Pass/Fail format
    if re.search(r'\b(pass|fail)\b', grading_guidelines, re.IGNORECASE):
        if re.search(r'\bpass\b', pred_lower):
            logger.debug(f"Normalized to 'Pass' from: '{original_prediction}'")
            return "Pass"
        elif re.search(r'\bfail\b', pred_lower):
            logger.debug(f"Normalized to 'Fail' from: '{original_prediction}'")
            return "Fail"
    
    if original_prediction != prediction:
        logger.debug(f"Prediction normalized from '{original_prediction}' to '{prediction}'")
    else:
        logger.debug(f"No normalization needed for prediction: '{prediction}'")
    
    return prediction


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning.
    
    Enhanced with comprehensive logging and robust error handling for production use.
    """

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        logger.info(f"TaskAgent initialized with model: {model}")

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
        
        # Log input summary for debugging
        logger.info(f"Processing problem in domain: {domain[:50] if domain else 'N/A'}...")
        logger.debug(f"Problem length: {len(problem)} chars, Solution length: {len(solution)} chars")
        logger.debug(f"Student answer length: {len(student_answer)} chars")
        logger.debug(f"Grading guidelines length: {len(grading_guidelines)} chars")

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

        logger.info("Sending instruction to LLM for grading...")
        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )
        
        # Log LLM response info
        usage = info.get("usage", {})
        logger.info(f"LLM response received. Tokens: prompt={usage.get('prompt_tokens', 'N/A')}, completion={usage.get('completion_tokens', 'N/A')}")

        # Extract prediction from JSON
        prediction = "None"
        last_text = msg_history[-1]["text"] if msg_history else ""
        extraction_method = "none"
        
        logger.debug(f"Raw LLM response length: {len(last_text)} chars")
        
        try:
            # First try: extract from <json> tags
            logger.debug("Attempting extraction from <json> tags...")
            extracted = _extract_jsons(last_text)
            if extracted:
                extraction_method = "json_tags"
                # Try to get response field, fall back to other common fields
                last_json = extracted[-1]
                logger.debug(f"Extracted JSON keys: {list(last_json.keys())}")
                if "response" in last_json:
                    prediction = str(last_json["response"])
                    logger.info(f"Extracted prediction from 'response' field: {prediction}")
                elif "grade" in last_json:
                    prediction = str(last_json["grade"])
                    logger.info(f"Extracted prediction from 'grade' field: {prediction}")
                elif "score" in last_json:
                    prediction = str(last_json["score"])
                    logger.info(f"Extracted prediction from 'score' field: {prediction}")
                elif "answer" in last_json:
                    prediction = str(last_json["answer"])
                    logger.info(f"Extracted prediction from 'answer' field: {prediction}")
                else:
                    # If no recognized field, use the whole JSON as string
                    prediction = json.dumps(last_json)
                    logger.warning(f"No recognized field found in JSON, using full JSON: {prediction[:100]}...")
            else:
                # Second try: fallback extraction for non-tagged JSON
                logger.debug("<json> tag extraction failed, attempting fallback extraction...")
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
                    logger.info(f"Used fallback JSON extraction: {prediction}")
                else:
                    # Third try: direct text extraction for simple responses
                    extraction_method = "direct"
                    logger.debug("Fallback extraction failed, attempting direct text extraction...")
                    # Look for the last line that might be the answer
                    lines = [line.strip() for line in last_text.split('\n') if line.strip()]
                    if lines:
                        # Check if the last non-empty line looks like a simple answer
                        last_line = lines[-1]
                        if len(last_line) < 50 and not last_line.startswith('{') and not last_line.startswith('<'):
                            prediction = last_line
                            logger.info(f"Used direct text extraction: {prediction}")
                        else:
                            logger.warning(f"Last line not suitable for direct extraction: {last_line[:50]}...")
                    else:
                        logger.warning("No non-empty lines found in response")
            
            # Validate and normalize the prediction
            original_prediction = prediction
            prediction = _validate_and_normalize_prediction(prediction, grading_guidelines)
            
            if original_prediction != prediction:
                logger.info(f"Normalized prediction from '{original_prediction}' to '{prediction}'")
            
            logger.info(f"Extraction method: {extraction_method}, final prediction: {prediction}")
            
        except Exception as e:
            logger.error(f"Error extracting prediction: {e}", exc_info=True)
            # Last resort: try to find any numeric or keyword answer in the text
            try:
                # Look for IMO scores (0-7)
                score_match = re.search(r'\b([0-7])\b', last_text)
                if score_match:
                    prediction = score_match.group(1)
                    logger.info(f"Used regex extraction for score: {prediction}")
            except Exception as e2:
                logger.error(f"Regex extraction also failed: {e2}")

        logger.info(f"TaskAgent forward() completed with prediction: {prediction}")
        return str(prediction), msg_history
