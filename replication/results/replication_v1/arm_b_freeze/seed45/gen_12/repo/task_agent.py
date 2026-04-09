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
    """
    if not text or not isinstance(text, str):
        return None
        
    results = []
    search_from = 0
    
    # First try to find <json>...</json> blocks
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # Try to extract JSON from within the content using brace matching
            try:
                json_str = _extract_json_with_brace_matching(inner)
                if json_str:
                    results.append(json.loads(json_str))
            except (json.JSONDecodeError, ValueError):
                # Try cleaning common LLM formatting issues
                try:
                    cleaned = _clean_json_string(inner)
                    if cleaned:
                        results.append(json.loads(cleaned))
                except (json.JSONDecodeError, ValueError):
                    continue
    
    # If no <json> blocks found, try markdown code blocks
    if not results:
        # Try ```json ... ``` blocks
        json_blocks = re.findall(r'```json\s*(.*?)```', text, re.DOTALL)
        for block in json_blocks:
            try:
                results.append(json.loads(block.strip()))
            except json.JSONDecodeError:
                # Try brace matching extraction
                try:
                    json_str = _extract_json_with_brace_matching(block)
                    if json_str:
                        results.append(json.loads(json_str))
                except (json.JSONDecodeError, ValueError):
                    # Try cleaning common LLM formatting issues
                    try:
                        cleaned = _clean_json_string(block)
                        if cleaned:
                            results.append(json.loads(cleaned))
                    except (json.JSONDecodeError, ValueError):
                        continue
        
        # Try bare JSON objects as fallback with improved regex
        if not results:
            # Find JSON-like structures with nested brace support
            potential_jsons = _find_json_objects(text)
            for pj in potential_jsons:
                try:
                    results.append(json.loads(pj))
                except json.JSONDecodeError:
                    # Try cleaning common LLM formatting issues
                    try:
                        cleaned = _clean_json_string(pj)
                        if cleaned:
                            results.append(json.loads(cleaned))
                    except (json.JSONDecodeError, ValueError):
                        continue
    
    return results or None


def _clean_json_string(text: str) -> str | None:
    """Clean common LLM JSON formatting issues.
    
    Handles:
    - Trailing commas before closing braces/brackets
    - Single quotes instead of double quotes
    - Comments (// and /* */)
    - Extra whitespace and newlines
    """
    if not text:
        return None
    
    # First try to extract just the JSON object using brace matching
    json_str = _extract_json_with_brace_matching(text)
    if not json_str:
        json_str = text.strip()
    
    # Remove comments (// style)
    lines = []
    for line in json_str.split('\n'):
        # Remove // comments but preserve URLs
        if '//' in line:
            # Simple heuristic: if // is inside quotes, keep it
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
        lines.append(line)
    json_str = '\n'.join(lines)
    
    # Remove /* */ comments
    while '/*' in json_str and '*/' in json_str:
        start = json_str.find('/*')
        end = json_str.find('*/', start) + 2
        if start >= 0 and end > start:
            json_str = json_str[:start] + json_str[end:]
        else:
            break
    
    # Remove trailing commas before } or ]
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
    
    return json_str.strip()


def _extract_json_with_brace_matching(text: str) -> str | None:
    """Extract a JSON object from text using brace counting.
    
    This handles nested braces correctly by counting open/close braces.
    """
    start = text.find("{")
    if start == -1:
        return None
    
    brace_count = 0
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
                if brace_count == 0:
                    return text[start:i+1]
    
    return None


def _find_json_objects(text: str) -> list[str]:
    """Find all potential JSON objects in text using brace matching."""
    results = []
    i = 0
    while i < len(text):
        if text[i] == "{":
            json_str = _extract_json_with_brace_matching(text[i:])
            if json_str and '"' in json_str:  # Must contain at least one quoted string
                results.append(json_str)
                i += len(json_str)
            else:
                i += 1
        else:
            i += 1
    return results


def _extract_grade_from_text(text: str) -> str | None:
    """Extract a grade/assessment from plain text when JSON parsing fails.
    
    Looks for common grading patterns and keywords in the text.
    Returns the most likely grade or None if no grade is found.
    """
    if not text:
        return None
    
    text_lower = text.lower()
    
    # Define grade patterns with their associated keywords
    grade_patterns = [
        ("Correct", ["correct", "right", "accurate", "valid", "proper", "appropriate", "satisfactory"]),
        ("Incorrect", ["incorrect", "wrong", "error", "mistake", "invalid", "unsatisfactory", "fail"]),
        ("Partially Correct", ["partially correct", "partial credit", "partially right", "some correct", "incomplete"]),
    ]
    
    # Count occurrences of each grade's keywords
    grade_scores = {}
    for grade, keywords in grade_patterns:
        score = 0
        for keyword in keywords:
            score += text_lower.count(keyword)
        grade_scores[grade] = score
    
    # Also look for explicit grade statements
    explicit_patterns = [
        (r'grade\s*[:=]\s*["\']?([^"\'\n]+)["\']?', 1),
        (r'(?:the\s+)?(?:final\s+)?(?:grade|score|assessment|evaluation|verdict)\s*(?:is|[:=])\s*["\']?([^"\'\n.]+)["\']?', 1),
        (r'(?:i\s+)?(?:would\s+)?(?:grade|score|rate|assess)\s*(?:this\s+)?(?:as|at)\s*["\']?([^"\'\n.]+)["\']?', 1),
        (r'(?:answer|solution)\s+(?:is\s+)?(["\']?(?:correct|incorrect|partially correct|partial|wrong)["\']?)', 1),
    ]
    
    for pattern, group in explicit_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            if isinstance(match, str):
                match_clean = match.strip().strip('"\'').lower()
                for grade, keywords in grade_patterns:
                    if any(kw in match_clean for kw in keywords):
                        grade_scores[grade] += 5  # Higher weight for explicit statements
    
    # Look for numeric scores (0-100 or 0-10)
    numeric_patterns = [
        r'(?:score|grade)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(?:/\s*(\d+))?',
        r'(\d+(?:\.\d+)?)\s*(?:out\s+of|/\s*)(\d+)',
    ]
    for pattern in numeric_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                try:
                    score = float(match[0])
                    total = float(match[1]) if match[1] and match[1].strip() else 100.0
                    if total > 0:
                        percentage = (score / total) * 100
                        if percentage >= 80:
                            grade_scores["Correct"] += 3
                        elif percentage >= 50:
                            grade_scores["Partially Correct"] += 3
                        else:
                            grade_scores["Incorrect"] += 3
                except (ValueError, TypeError):
                    pass
            elif isinstance(match, str):
                try:
                    score = float(match)
                    if score >= 8:
                        grade_scores["Correct"] += 2
                    elif score >= 5:
                        grade_scores["Partially Correct"] += 2
                    else:
                        grade_scores["Incorrect"] += 2
                except ValueError:
                    pass
    
    # Return the grade with the highest score, if any
    if grade_scores:
        best_grade = max(grade_scores, key=grade_scores.get)
        if grade_scores[best_grade] > 0:
            return best_grade
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3  # Increased max retries for better robustness

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with chain-of-thought reasoning.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract fields for better prompting
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

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

Think carefully and provide a fair assessment based on the official solution and grading guidelines."""

        msg_history = []
        prediction = "None"
        reasoning = ""
        
        for attempt in range(self.max_retries + 1):
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=msg_history,
            )

            # Extract prediction from JSON
            extracted = None
            try:
                # Try the last assistant message
                last_msg = msg_history[-1]["text"] if msg_history else ""
                extracted = _extract_jsons(last_msg)
                
                # If not found, try searching all messages
                if not extracted:
                    for msg in reversed(msg_history):
                        text = msg.get("text", "")
                        extracted = _extract_jsons(text)
                        if extracted:
                            break
                
                if extracted:
                    result = extracted[-1]
                    # Try multiple possible keys for the response
                    for key in ["response", "grade", "answer", "result", "assessment", "evaluation", "verdict", "conclusion"]:
                        if key in result:
                            prediction = result[key]
                            break
                    
                    # Extract reasoning if available
                    for key in ["reasoning", "analysis", "thought", "explanation", "rationale", "thinking"]:
                        if key in result:
                            reasoning = result[key]
                            break
                    
                    # Log reasoning if available
                    if reasoning:
                        self.log_fn(f"Reasoning: {reasoning[:200]}...")
                    
                    # Success - break out of retry loop
                    break
                else:
                    # Try to extract any text that might be a grade/assessment
                    # This is a fallback for when JSON parsing fails completely
                    fallback_prediction = _extract_grade_from_text(last_msg)
                    if fallback_prediction and fallback_prediction != "None":
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
                    self.log_fn(f"No JSON found after {self.max_retries + 1} attempts")
                    
            except Exception as e:
                self.log_fn(f"Error extracting prediction (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries:
                    feedback = (
                        f"Error parsing your response: {e}. "
                        "Please ensure your response contains valid JSON wrapped in <json>...</json> tags. "
                        "The JSON must have 'reasoning' and 'response' fields."
                    )
                    msg_history.append({"role": "user", "text": feedback})
                    instruction = feedback

        return str(prediction), msg_history
