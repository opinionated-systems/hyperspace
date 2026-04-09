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
    Also handles markdown code blocks and bare JSON objects.
    Includes improved handling for nested braces, escaped characters, and 
    common LLM formatting issues like trailing commas and comments.
    
    Enhanced with better error logging and multiple fallback strategies.
    """
    if not text or not isinstance(text, str):
        logger.debug("_extract_jsons: received empty or non-string input")
        return None
    
    # Track extraction attempts for debugging
    extraction_attempts = 0
    results = []
    search_from = 0
    
    # First try to find <json>...</json> blocks
    max_iterations = 100  # Safety limit to prevent infinite loops
    while extraction_attempts < max_iterations:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            logger.debug(f"_extract_jsons: found opening <json> at {start} but no closing tag")
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        extraction_attempts += 1
        
        try:
            results.append(json.loads(inner))
            logger.debug(f"_extract_jsons: successfully parsed JSON from <json> block #{extraction_attempts}")
        except json.JSONDecodeError as e:
            logger.debug(f"_extract_jsons: JSONDecodeError in <json> block #{extraction_attempts}: {e}")
            # Try to extract JSON from within the content using brace matching
            try:
                json_str = _extract_json_with_brace_matching(inner)
                if json_str:
                    results.append(json.loads(json_str))
                    logger.debug(f"_extract_jsons: successfully parsed JSON using brace matching")
            except (json.JSONDecodeError, ValueError) as e2:
                logger.debug(f"_extract_jsons: brace matching failed: {e2}")
                # Try cleaning common LLM formatting issues
                try:
                    cleaned = _clean_json_string(inner)
                    if cleaned:
                        results.append(json.loads(cleaned))
                        logger.debug(f"_extract_jsons: successfully parsed JSON after cleaning")
                except (json.JSONDecodeError, ValueError) as e3:
                    logger.debug(f"_extract_jsons: cleaning failed: {e3}")
                    continue
    
    # If no <json> blocks found, try markdown code blocks
    if not results:
        logger.debug("_extract_jsons: no valid JSON found in <json> blocks, trying markdown code blocks")
        # Try ```json ... ``` blocks
        json_blocks = re.findall(r'```json\s*(.*?)```', text, re.DOTALL)
        logger.debug(f"_extract_jsons: found {len(json_blocks)} markdown json blocks")
        
        for i, block in enumerate(json_blocks):
            try:
                results.append(json.loads(block.strip()))
                logger.debug(f"_extract_jsons: successfully parsed JSON from markdown block #{i+1}")
            except json.JSONDecodeError as e:
                logger.debug(f"_extract_jsons: JSONDecodeError in markdown block #{i+1}: {e}")
                # Try brace matching extraction
                try:
                    json_str = _extract_json_with_brace_matching(block)
                    if json_str:
                        results.append(json.loads(json_str))
                        logger.debug(f"_extract_jsons: successfully parsed markdown JSON using brace matching")
                except (json.JSONDecodeError, ValueError) as e2:
                    logger.debug(f"_extract_jsons: brace matching failed for markdown: {e2}")
                    # Try cleaning common LLM formatting issues
                    try:
                        cleaned = _clean_json_string(block)
                        if cleaned:
                            results.append(json.loads(cleaned))
                            logger.debug(f"_extract_jsons: successfully parsed markdown JSON after cleaning")
                    except (json.JSONDecodeError, ValueError) as e3:
                        logger.debug(f"_extract_jsons: cleaning failed for markdown: {e3}")
                        continue
        
        # Try bare JSON objects as fallback with improved regex
        if not results:
            logger.debug("_extract_jsons: no valid JSON in markdown blocks, trying bare JSON objects")
            # Find JSON-like structures with nested brace support
            potential_jsons = _find_json_objects(text)
            logger.debug(f"_extract_jsons: found {len(potential_jsons)} potential JSON objects")
            
            for i, pj in enumerate(potential_jsons):
                try:
                    results.append(json.loads(pj))
                    logger.debug(f"_extract_jsons: successfully parsed bare JSON object #{i+1}")
                except json.JSONDecodeError as e:
                    logger.debug(f"_extract_jsons: JSONDecodeError in bare object #{i+1}: {e}")
                    # Try cleaning common LLM formatting issues
                    try:
                        cleaned = _clean_json_string(pj)
                        if cleaned:
                            results.append(json.loads(cleaned))
                            logger.debug(f"_extract_jsons: successfully parsed bare JSON after cleaning")
                    except (json.JSONDecodeError, ValueError) as e2:
                        logger.debug(f"_extract_jsons: cleaning failed for bare object: {e2}")
                        continue
    
    if extraction_attempts >= max_iterations:
        logger.warning(f"_extract_jsons: reached max iterations ({max_iterations}), stopping search")
    
    if results:
        logger.debug(f"_extract_jsons: returning {len(results)} JSON object(s)")
    else:
        logger.debug(f"_extract_jsons: no JSON objects found after {extraction_attempts} attempts")
    
    return results or None


def _clean_json_string(text: str) -> str | None:
    """Clean common LLM JSON formatting issues.
    
    Handles:
    - Trailing commas before closing braces/brackets
    - Single quotes instead of double quotes
    - Comments (// and /* */)
    - Extra whitespace and newlines
    - Unescaped newlines in string values
    - Unicode escape sequences
    - BOM (Byte Order Mark) characters
    """
    if not text:
        return None
    
    # First try to extract just the JSON object using brace matching
    json_str = _extract_json_with_brace_matching(text)
    if not json_str:
        json_str = text.strip()
    
    # Remove BOM if present
    json_str = json_str.lstrip('\ufeff')
    
    # Remove comments (// style) - more robust implementation
    lines = []
    for line in json_str.split('\n'):
        # Remove // comments but preserve URLs and quoted strings
        if '//' in line:
            in_string = False
            escape_next = False
            comment_start = -1
            for i, char in enumerate(line):
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
                elif not in_string and char == '/' and i + 1 < len(line) and line[i + 1] == '/':
                    comment_start = i
                    break
            if comment_start >= 0:
                line = line[:comment_start]
        lines.append(line.rstrip())  # Also strip trailing whitespace from each line
    json_str = '\n'.join(lines)
    
    # Remove /* */ comments (multiline)
    while '/*' in json_str and '*/' in json_str:
        start = json_str.find('/*')
        end = json_str.find('*/', start)
        if start >= 0 and end > start:
            json_str = json_str[:start] + json_str[end + 2:]
        else:
            break
    
    # Remove trailing commas before } or ] - handle multiple commas
    json_str = re.sub(r',\s*,\s*}', '}', json_str)  # Multiple trailing commas
    json_str = re.sub(r',\s*,\s*\]', ']', json_str)
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*\]', ']', json_str)
    
    # Replace single quotes with double quotes (carefully)
    # Only replace quotes that are not inside strings
    result = []
    in_string = False
    escape_next = False
    for char in json_str:
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
        elif char == "'" and not in_string:
            result.append('"')
        else:
            result.append(char)
    json_str = ''.join(result)
    
    # Handle unescaped newlines in string values by replacing with \n
    # This is a more aggressive fix for malformed JSON
    result = []
    in_string = False
    escape_next = False
    for char in json_str:
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
        elif in_string and char == '\n':
            result.append('\\n')  # Escape newlines inside strings
        elif in_string and char == '\t':
            result.append('\\t')  # Escape tabs inside strings
        else:
            result.append(char)
    json_str = ''.join(result)
    
    return json_str.strip()


def _extract_json_with_brace_matching(text: str) -> str | None:
    """Extract a JSON object from text using brace counting.
    
    This handles nested braces correctly by counting open/close braces.
    Also handles square brackets for JSON arrays.
    """
    if not text:
        return None
    
    # Find the first { or [ that starts a JSON structure
    brace_start = text.find("{")
    bracket_start = text.find("[")
    
    if brace_start == -1 and bracket_start == -1:
        return None
    
    # Determine which comes first
    if brace_start == -1:
        start = bracket_start
        is_array = True
    elif bracket_start == -1:
        start = brace_start
        is_array = False
    else:
        start = min(brace_start, bracket_start)
        is_array = (start == bracket_start)
    
    brace_count = 0
    bracket_count = 0
    in_string = False
    escape_next = False
    
    for i, char in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        if char == "\\":
            escape_next = True
            continue
        if char == '"' and not in_string:
            in_string = True
        elif char == '"' and in_string:
            in_string = False
        elif not in_string:
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if not is_array and brace_count == 0:
                    return text[start:i+1]
            elif char == "[":
                bracket_count += 1
            elif char == "]":
                bracket_count -= 1
                if is_array and bracket_count == 0 and brace_count == 0:
                    return text[start:i+1]
    
    return None


def _find_json_objects(text: str) -> list[str]:
    """Find all potential JSON objects and arrays in text using brace matching."""
    results = []
    i = 0
    max_iterations = len(text) * 2  # Safety limit based on text length
    iterations = 0
    
    while i < len(text) and iterations < max_iterations:
        iterations += 1
        if text[i] in "{[":
            json_str = _extract_json_with_brace_matching(text[i:])
            if json_str:
                # For objects: must contain at least one quoted string
                # For arrays: must contain at least one element
                if (text[i] == "{" and '"' in json_str) or (text[i] == "[" and len(json_str) > 2):
                    results.append(json_str)
                    i += len(json_str)
                else:
                    i += 1
            else:
                i += 1
        else:
            i += 1
    
    if iterations >= max_iterations:
        logger.warning(f"_find_json_objects: reached max iterations for text of length {len(text)}")
    
    return results


def _extract_grade_from_text(text: str) -> str | None:
    """Extract a grade/assessment from plain text when JSON parsing fails.
    
    Looks for common grading patterns and keywords in the text.
    Returns the most likely grade or None if no grade is found.
    
    Enhanced with better pattern matching and context awareness.
    """
    if not text:
        return None
    
    # Limit text length to prevent performance issues with regex
    max_text_length = 50000
    if len(text) > max_text_length:
        logger.debug(f"_extract_grade_from_text: truncating text from {len(text)} to {max_text_length} chars")
        text = text[:max_text_length]
    
    text_lower = text.lower()
    
    # Define grade patterns with their associated keywords and weights
    grade_patterns = [
        ("Correct", [
            ("correct", 2), ("right", 2), ("accurate", 2), ("valid", 1), 
            ("proper", 1), ("appropriate", 1), ("satisfactory", 1),
            ("well done", 3), ("excellent", 2), ("perfect", 3), ("good", 1)
        ]),
        ("Incorrect", [
            ("incorrect", 3), ("wrong", 2), ("error", 1), ("mistake", 1), 
            ("invalid", 2), ("unsatisfactory", 2), ("fail", 2),
            ("not correct", 3), ("not right", 3), ("does not match", 2),
            ("doesn't match", 2), ("no match", 2)
        ]),
        ("Partially Correct", [
            ("partially correct", 3), ("partial credit", 3), ("partially right", 3), 
            ("some correct", 2), ("incomplete", 2), ("partial", 2),
            ("mostly correct", 2), ("mostly right", 2), ("half correct", 2),
            ("partly correct", 3), ("some errors", 1), ("minor errors", 1)
        ]),
    ]
    
    # Count weighted occurrences of each grade's keywords
    grade_scores = {}
    for grade, keywords in grade_patterns:
        score = 0
        for keyword, weight in keywords:
            # Count occurrences and apply weight
            count = text_lower.count(keyword)
            score += count * weight
        grade_scores[grade] = score
    
    # Look for explicit grade statements with higher weights
    explicit_patterns = [
        (r'\bgrade\s*[:=]\s*["\']?([^"\'\n]+)["\']?', 5),
        (r'\b(?:the\s+)?(?:final\s+)?(?:grade|score|assessment|evaluation|verdict)\s*(?:is|[:=])\s*["\']?([^"\'\n.]+)["\']?', 5),
        (r'\b(?:i\s+)?(?:would\s+)?(?:grade|score|rate|assess)\s*(?:this\s+)?(?:as|at)\s*["\']?([^"\'\n.]+)["\']?', 5),
        (r'\b(?:answer|solution)\s+(?:is\s+)?(["\']?(?:correct|incorrect|partially correct|partial|wrong)["\']?)', 6),
        (r'\b(?:this\s+is|the\s+answer\s+is)\s+(["\']?(?:correct|incorrect|partially correct|partial|wrong)["\']?)', 6),
    ]
    
    # Limit total regex operations to prevent performance issues
    max_regex_matches = 100
    
    for pattern, weight in explicit_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        # Limit matches per pattern
        if len(matches) > max_regex_matches:
            logger.debug(f"_extract_grade_from_text: limiting pattern matches from {len(matches)} to {max_regex_matches}")
            matches = matches[:max_regex_matches]
        for match in matches:
            if isinstance(match, str):
                match_clean = match.strip().strip('"\'').lower()
                for grade, keywords in grade_patterns:
                    # Check if any keyword is in the match
                    for keyword, _ in keywords:
                        if keyword in match_clean:
                            grade_scores[grade] += weight
                            break
    
    # Look for negation patterns (e.g., "not correct" should favor Incorrect)
    negation_patterns = [
        (r'\bnot\s+(correct|right|accurate|valid)\b', "Incorrect", 4),
        (r'\bnot\s+(incorrect|wrong|invalid)\b', "Correct", 4),
    ]
    
    for pattern, grade, weight in negation_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        grade_scores[grade] += len(matches) * weight
    
    # Look for numeric scores (0-100 or 0-10)
    numeric_patterns = [
        r'\b(?:score|grade)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(?:/\s*(\d+))?\b',
        r'\b(\d+(?:\.\d+)?)\s*(?:out\s+of|/\s*)(\d+)\b',
        r'\b(\d+(?:\.\d+)?)\s*points?\b',
    ]
    
    for pattern in numeric_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        # Limit matches per pattern
        if len(matches) > max_regex_matches:
            logger.debug(f"_extract_grade_from_text: limiting numeric pattern matches from {len(matches)} to {max_regex_matches}")
            matches = matches[:max_regex_matches]
        for match in matches:
            if isinstance(match, tuple):
                try:
                    score = float(match[0])
                    total = float(match[1]) if match[1] and match[1].strip() else 100.0
                    if total > 0:
                        percentage = (score / total) * 100
                        if percentage >= 80:
                            grade_scores["Correct"] += 4
                        elif percentage >= 50:
                            grade_scores["Partially Correct"] += 4
                        else:
                            grade_scores["Incorrect"] += 4
                except (ValueError, TypeError):
                    pass
            elif isinstance(match, str):
                try:
                    score = float(match)
                    # Assume 0-10 scale if score is small
                    if score <= 10:
                        if score >= 8:
                            grade_scores["Correct"] += 3
                        elif score >= 5:
                            grade_scores["Partially Correct"] += 3
                        else:
                            grade_scores["Incorrect"] += 3
                    else:
                        # Assume 0-100 scale
                        if score >= 80:
                            grade_scores["Correct"] += 3
                        elif score >= 50:
                            grade_scores["Partially Correct"] += 3
                        else:
                            grade_scores["Incorrect"] += 3
                except ValueError:
                    pass
    
    # Return the grade with the highest score, if any
    if grade_scores:
        best_grade = max(grade_scores, key=grade_scores.get)
        if grade_scores[best_grade] > 0:
            logger.debug(f"_extract_grade_from_text: selected '{best_grade}' with score {grade_scores[best_grade]}")
            return best_grade
    
    logger.debug("_extract_grade_from_text: no grade found")
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning.
    
    Enhanced with better error handling, logging, and multiple fallback strategies
    for extracting grades from LLM responses.
    """

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3  # Increased max retries for better robustness
        self._extraction_stats = {
            "json_extractions": 0,
            "fallback_extractions": 0,
            "failed_extractions": 0,
            "total_attempts": 0
        }

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with chain-of-thought reasoning.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Validate inputs
        if not isinstance(inputs, dict):
            logger.error(f"TaskAgent.forward: expected dict, got {type(inputs)}")
            return "Error: Invalid inputs", []
        
        # Extract fields for better prompting
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Log input sizes for debugging
        logger.debug(f"TaskAgent.forward: problem length={len(problem)}, "
                    f"solution length={len(solution)}, "
                    f"student_answer length={len(student_answer)}")

        instruction = f"""You are an expert {domain} grader evaluating student solutions.

Your task is to grade a student's answer to a problem by comparing it against the official solution and following the grading guidelines.

## Problem Statement:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Instructions:
1. First, analyze the student's answer step by step. Identify what they did correctly and incorrectly.
2. Compare their approach to the official solution.
3. Consider the grading guidelines carefully.
4. Provide your reasoning before giving the final grade.
5. Respond in JSON format with the following schema:

<json>
{{
    "reasoning": "Your detailed analysis and reasoning about the student's answer...",
    "response": "The final grade/assessment (e.g., 'Correct', 'Partially Correct', 'Incorrect', or a numeric score)"
}}
</json>

Important: 
- The JSON must be valid and properly formatted.
- Wrap the JSON in <json>...</json> tags.
- The 'response' field should contain only the final grade/assessment.
- The 'reasoning' field should contain your detailed analysis.
- Do not include any text outside the <json>...</json> tags.

Think carefully and provide a fair assessment based on the official solution and grading guidelines."""

        msg_history = []
        prediction = "None"
        reasoning = ""
        extraction_method = None
        
        for attempt in range(self.max_retries + 1):
            self._extraction_stats["total_attempts"] += 1
            
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=msg_history,
                )
            except Exception as e:
                logger.error(f"TaskAgent.forward: LLM call failed on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries:
                    continue
                else:
                    return "Error: LLM call failed", msg_history

            # Extract prediction from JSON
            extracted = None
            try:
                # Try the last assistant message
                last_msg = msg_history[-1]["text"] if msg_history else ""
                
                if not last_msg:
                    logger.warning(f"TaskAgent.forward: empty last message on attempt {attempt + 1}")
                    if attempt < self.max_retries:
                        continue
                    break
                
                extracted = _extract_jsons(last_msg)
                
                # If not found, try searching all messages
                if not extracted:
                    logger.debug(f"TaskAgent.forward: no JSON in last message, searching all messages")
                    for msg in reversed(msg_history):
                        text = msg.get("text", "")
                        if text:
                            extracted = _extract_jsons(text)
                            if extracted:
                                logger.debug(f"TaskAgent.forward: found JSON in earlier message")
                                break
                
                if extracted:
                    self._extraction_stats["json_extractions"] += 1
                    extraction_method = "json"
                    result = extracted[-1]
                    
                    # Validate result is a dict
                    if not isinstance(result, dict):
                        logger.warning(f"TaskAgent.forward: extracted result is not a dict: {type(result)}")
                        if attempt < self.max_retries:
                            continue
                        break
                    
                    # Try multiple possible keys for the response
                    response_found = False
                    for key in ["response", "grade", "answer", "result", "assessment", "evaluation", "verdict", "conclusion"]:
                        if key in result:
                            prediction = result[key]
                            response_found = True
                            logger.debug(f"TaskAgent.forward: found response in key '{key}': {prediction}")
                            break
                    
                    if not response_found:
                        logger.warning(f"TaskAgent.forward: no response key found in result: {list(result.keys())}")
                    
                    # Extract reasoning if available
                    for key in ["reasoning", "analysis", "thought", "explanation", "rationale", "thinking"]:
                        if key in result:
                            reasoning = result[key]
                            break
                    
                    # Log reasoning if available
                    if reasoning:
                        self.log_fn(f"Reasoning: {reasoning[:200]}...")
                    
                    # Success - break out of retry loop
                    if response_found:
                        break
                else:
                    # Try to extract any text that might be a grade/assessment
                    # This is a fallback for when JSON parsing fails completely
                    fallback_prediction = _extract_grade_from_text(last_msg)
                    if fallback_prediction and fallback_prediction != "None":
                        self._extraction_stats["fallback_extractions"] += 1
                        extraction_method = "fallback"
                        self.log_fn(f"Using fallback grade extraction: {fallback_prediction}")
                        prediction = fallback_prediction
                        break
                    
                if attempt < self.max_retries:
                    # No JSON found - add feedback and retry
                    self.log_fn(f"No JSON found in attempt {attempt + 1}, retrying with feedback...")
                    feedback = (
                        "Your previous response did not contain valid JSON in the required format. "
                        "Please respond with a JSON object wrapped in <json>...</json> tags. "
                        "The JSON must have 'reasoning' and 'response' fields. "
                        "Example format:\n"
                        "<json>\n"
                        '{\n  "reasoning": "The student correctly identified...",\n  "response": "Correct"\n}'
                        "\n</json>"
                    )
                    msg_history.append({"role": "user", "text": feedback})
                    instruction = feedback  # Update instruction for next iteration
                else:
                    self._extraction_stats["failed_extractions"] += 1
                    self.log_fn(f"No JSON found after {self.max_retries + 1} attempts")
                    
            except Exception as e:
                self.log_fn(f"Error extracting prediction (attempt {attempt + 1}): {e}")
                logger.exception("TaskAgent.forward: exception during extraction")
                if attempt < self.max_retries:
                    feedback = (
                        f"Error parsing your response: {e}. "
                        "Please ensure your response contains valid JSON wrapped in <json>...</json> tags. "
                        "The JSON must have 'reasoning' and 'response' fields."
                    )
                    msg_history.append({"role": "user", "text": feedback})
                    instruction = feedback

        # Log extraction stats periodically
        if self._extraction_stats["total_attempts"] % 100 == 0:
            logger.info(f"TaskAgent extraction stats: {self._extraction_stats}")
        
        logger.debug(f"TaskAgent.forward: returning prediction='{prediction}', method={extraction_method}")
        return str(prediction), msg_history
