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
    Also handles markdown code blocks with json tag and inline JSON.
    Includes comprehensive JSON repair for common LLM output issues.
    """
    results = []
    search_from = 0
    
    # First try <json>...</json> tags
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        # Try parsing with progressive repair
        parsed = _try_parse_json_with_repair(inner)
        if parsed is not None:
            results.append(parsed)
    
    # Also try markdown code blocks ```json ... ```
    if not results:
        search_from = 0
        while True:
            start = text.find("```json", search_from)
            if start == -1:
                break
            start = start + 7  # Skip past ```json
            end = text.find("```", start)
            if end == -1:
                break
            inner = text[start:end].strip()
            search_from = end + 3
            
            parsed = _try_parse_json_with_repair(inner)
            if parsed is not None:
                results.append(parsed)
    
    # Try ``` ... ``` without json tag
    if not results:
        search_from = 0
        while True:
            start = text.find("```", search_from)
            if start == -1:
                break
            start = start + 3  # Skip past ```
            end = text.find("```", start)
            if end == -1:
                break
            inner = text[start:end].strip()
            search_from = end + 3
            # Only try if it looks like JSON
            if inner.startswith("{") or inner.startswith("["):
                parsed = _try_parse_json_with_repair(inner)
                if parsed is not None:
                    results.append(parsed)
    
    return results or None


def _try_parse_json_with_repair(text: str) -> dict | None:
    """Try to parse JSON with progressive repair strategies.
    
    Attempts multiple repair strategies for common LLM JSON output issues.
    Returns the parsed dict or None if all attempts fail.
    """
    # Strategy 0: Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Strategy 1: Remove trailing commas before closing braces/brackets
    try:
        fixed = re.sub(r',(\s*[}\]])', r'\1', text)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Fix single quotes to double quotes (common LLM mistake)
    try:
        fixed = re.sub(r',(\s*[}\]])', r'\1', text)
        fixed = re.sub(r"'([^']*?)'", r'"\1"', fixed)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Fix unescaped newlines in strings
    try:
        fixed = re.sub(r',(\s*[}\]])', r'\1', text)
        fixed = re.sub(r"'([^']*?)'", r'"\1"', fixed)
        fixed = re.sub(r'(?<!\\)\n', r'\\n', fixed)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Strategy 4: Fix unescaped tabs in strings
    try:
        fixed = re.sub(r',(\s*[}\]])', r'\1', text)
        fixed = re.sub(r"'([^']*?)'", r'"\1"', fixed)
        fixed = re.sub(r'(?<!\\)\n', r'\\n', fixed)
        fixed = re.sub(r'(?<!\\)\t', r'\\t', fixed)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Strategy 5: Fix common escape sequence issues
    try:
        fixed = re.sub(r',(\s*[}\]])', r'\1', text)
        fixed = re.sub(r"'([^']*?)'", r'"\1"', fixed)
        fixed = re.sub(r'(?<!\\)\n', r'\\n', fixed)
        fixed = re.sub(r'(?<!\\)\t', r'\\t', fixed)
        # Fix double-escaped characters that should be single-escaped
        fixed = re.sub(r'\\\\(?!\\)', r'\\', fixed)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Strategy 6: Extract just the object structure, ignoring content issues
    try:
        # Find the outermost braces
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            core = text[start:end+1]
            # Apply all fixes to the core
            fixed = re.sub(r',(\s*[}\]])', r'\1', core)
            fixed = re.sub(r"'([^']*?)'", r'"\1"', fixed)
            fixed = re.sub(r'(?<!\\)\n', r'\\n', fixed)
            fixed = re.sub(r'(?<!\\)\t', r'\\t', fixed)
            return json.loads(fixed)
    except (json.JSONDecodeError, ValueError):
        pass
    
    return None


def _extract_any_json(text: str) -> list[dict] | None:
    """Fallback JSON extraction that looks for any JSON objects in text.
    
    Uses brace counting to find JSON objects and applies comprehensive repair.
    """
    results = []
    # Try to find JSON objects between curly braces
    brace_count = 0
    start_idx = -1
    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                json_str = text[start_idx:i+1]
                # Use the comprehensive repair function
                parsed = _try_parse_json_with_repair(json_str)
                if parsed is not None:
                    results.append(parsed)
                start_idx = -1
    return results or None


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
        # Extract task components for better prompting
        domain = inputs.get("domain", "Unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematical grader for International Mathematical Olympiad (IMO) problems.

Your task is to evaluate a student's answer to a mathematics problem and provide a grade.

DOMAIN: {domain}

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

Think step by step:
1. Analyze what the problem is asking - identify key concepts, theorems, and required proof steps
2. Review the official solution approach - understand the expected reasoning and any alternative valid approaches
3. Compare the student's answer to the official solution:
   - Check mathematical correctness of each step
   - Verify logical flow and rigor of the proof
   - Identify any gaps, errors, or unstated assumptions
   - Note any creative or alternative valid approaches
4. Apply the grading guidelines precisely:
   - Look for specific partial credit criteria mentioned
   - Consider both correctness and completeness
   - Check if errors are conceptual or computational
5. Determine the appropriate grade based on the guidelines

IMPORTANT INSTRUCTIONS:
- Your response MUST be valid JSON wrapped in <json> tags
- Do not include any text outside the JSON tags
- Be objective and consistent with the grading guidelines
- If the guidelines specify numeric scores, use those exactly
- If the guidelines use categorical grades (Correct/Incorrect/Partial), use those exactly

Respond in this exact format:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis. Be specific about: (1) what the student did correctly, (2) any errors or gaps, (3) how the grading guidelines apply, (4) why this grade is appropriate.",
    "response": "The final grade. Use the EXACT format specified in the grading guidelines."
}}
</json>"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with multiple fallback mechanisms
        prediction = self._extract_prediction(msg_history)

        return str(prediction), msg_history

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history with robust fallback mechanisms.
        
        Args:
            msg_history: List of message dicts from LLM conversation
            
        Returns:
            Extracted prediction string or "None" if extraction fails
        """
        if not msg_history:
            self.log_fn("Warning: Empty message history, returning 'None'")
            return "None"
        
        try:
            last_message = msg_history[-1].get("text", "")
            if not last_message:
                self.log_fn("Warning: Last message has no text content")
                return "None"
            
            # Try primary extraction method (tagged JSON blocks)
            extracted = _extract_jsons(last_message)
            self.log_fn(f"Primary JSON extraction found {len(extracted) if extracted else 0} objects")
            
            # Fallback to generic JSON extraction if primary fails
            if not extracted:
                self.log_fn("Primary extraction failed, trying fallback extraction")
                extracted = _extract_any_json(last_message)
                if extracted:
                    self.log_fn(f"Fallback extraction found {len(extracted)} objects")
            
            if extracted:
                prediction = self._get_prediction_from_json(extracted[-1])
                self.log_fn(f"Successfully extracted prediction: '{prediction}'")
                return prediction
            
            # Final fallback: look for common grade patterns in raw text
            self.log_fn("JSON extraction failed, trying raw text extraction")
            raw_prediction = self._extract_from_raw_text(last_message)
            self.log_fn(f"Raw text extraction result: '{raw_prediction}'")
            return raw_prediction
            
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {type(e).__name__}: {e}")
            return "None"

    def _get_prediction_from_json(self, json_obj: dict) -> str:
        """Extract prediction value from a JSON object.
        
        Tries multiple common field names in order of preference.
        """
        # Ordered list of field names to try (most specific first)
        field_names = [
            "response", "grade", "evaluation", "verdict", 
            "result", "answer", "score", "prediction", "output"
        ]
        
        for field in field_names:
            if field in json_obj:
                value = json_obj[field]
                if isinstance(value, str):
                    return value.strip()
                elif isinstance(value, (int, float)):
                    return str(value)
                elif isinstance(value, bool):
                    return "Correct" if value else "Incorrect"
        
        # If no known field, use the first string or numeric value found
        for key, value in json_obj.items():
            if isinstance(value, str):
                return value.strip()
            elif isinstance(value, (int, float)):
                return str(value)
        
        return "None"

    def _extract_from_raw_text(self, text: str) -> str:
        """Extract grade from raw text when JSON parsing fails.
        
        Looks for common grade patterns and keywords with improved accuracy.
        Handles both categorical grades and numeric scores.
        """
        text_lower = text.lower()
        
        # Check for explicit grade statements with more specific patterns
        # These patterns look for "grade: X" or "grade is X" formats
        grade_patterns = [
            (r'final\s+grade[:\s]+([\w\s-]+?)(?:\n|$|\.|\,)', 1),
            (r'grade[:\s]+([\w\s-]+?)(?:\n|$|\.|\,)', 1),
            (r'evaluation[:\s]+([\w\s-]+?)(?:\n|$|\.|\,)', 1),
            (r'response[:\s]+([\w\s-]+?)(?:\n|$|\.|\,)', 1),
            (r'verdict[:\s]+([\w\s-]+?)(?:\n|$|\.|\,)', 1),
            (r'result[:\s]+([\w\s-]+?)(?:\n|$|\.|\,)', 1),
            (r'score[:\s]+([\d]+(?:\.\d+)?)', 1),
            (r'(?:^|\n)\s*([\w\s-]+?)(?:\s*=\s*|\s*is\s+)(?:correct|incorrect|partial)', 1),
            # Look for "The grade is X" patterns
            (r'the\s+grade\s+is\s+([\w\s-]+?)(?:\n|$|\.|\,)', 1),
            (r'the\s+answer\s+is\s+([\w\s-]+?)(?:\n|$|\.|\,)', 1),
        ]
        
        for pattern, group in grade_patterns:
            match = re.search(pattern, text_lower)
            if match:
                result = match.group(group).strip()
                # Clean up common artifacts
                result = re.sub(r'\s+', ' ', result)  # Normalize whitespace
                # Remove common filler words at start/end
                result = re.sub(r'^(the|a|an|this|that)\s+', '', result)
                result = re.sub(r'\s+(the|a|an|this|that)$', '', result)
                if result and result not in ['the', 'a', 'an', 'this', 'that', '']:
                    # Capitalize first letter for consistency
                    return result.capitalize() if result.islower() else result
        
        # Check for numeric scores (0-10 or 0-100 scale)
        score_patterns = [
            r'\bscore[:\s]+(\d+(?:\.\d+)?)\s*/\s*(\d+)\b',
            r'\b(\d+(?:\.\d+)?)\s*/\s*(\d+)\s*(?:points?|score|grade)',
            r'\bgrade[:\s]+(\d+(?:\.\d+)?)\b',
        ]
        for pattern in score_patterns:
            match = re.search(pattern, text_lower)
            if match:
                return match.group(1)
        
        # Check for binary correct/incorrect indicators with context awareness
        # Look for patterns like "is correct", "is incorrect", "answer is correct"
        correct_patterns = [
            r'\b(?:answer|response|solution|grade|evaluation)\s+is\s+correct\b',
            r'\b(?:the\s+)?(?:answer|response|solution)\s+is\s+right\b',
            r'\bcorrect\s+(?:answer|response|solution)\b',
            r'\bthe\s+student\s+(?:is\s+)?correct\b',
            r'\bthis\s+is\s+correct\b',
        ]
        incorrect_patterns = [
            r'\b(?:answer|response|solution|grade|evaluation)\s+is\s+incorrect\b',
            r'\b(?:answer|response|solution|grade|evaluation)\s+is\s+wrong\b',
            r'\bincorrect\s+(?:answer|response|solution)\b',
            r'\bwrong\s+(?:answer|response|solution)\b',
            r'\bthe\s+student\s+(?:is\s+)?incorrect\b',
            r'\bthe\s+student\s+(?:is\s+)?wrong\b',
            r'\bthis\s+is\s+incorrect\b',
            r'\bthis\s+is\s+wrong\b',
        ]
        partial_patterns = [
            r'\b(?:answer|response|solution|grade|evaluation)\s+is\s+partial\b',
            r'\bpartial\s+(?:credit|score|grade)\b',
            r'\bpartially\s+correct\b',
            r'\bthe\s+student\s+(?:is\s+)?partial\b',
            r'\bpartial\s+marks?\b',
        ]
        
        # Count matches for each category
        correct_count = sum(1 for p in correct_patterns if re.search(p, text_lower))
        incorrect_count = sum(1 for p in incorrect_patterns if re.search(p, text_lower))
        partial_count = sum(1 for p in partial_patterns if re.search(p, text_lower))
        
        # Return the most specific match
        if partial_count > 0:
            return "Partial"
        if incorrect_count > correct_count:
            return "Incorrect"
        if correct_count > 0 and incorrect_count == 0:
            return "Correct"
        
        # Last resort: simple keyword matching with context
        has_correct = re.search(r'\bcorrect\b', text_lower) is not None
        has_incorrect = re.search(r'\bincorrect\b', text_lower) is not None
        has_wrong = re.search(r'\bwrong\b', text_lower) is not None
        has_partial = re.search(r'\bpartial\b', text_lower) is not None
        has_right = re.search(r'\bright\b', text_lower) is not None
        
        if has_partial:
            return "Partial"
        if has_incorrect or has_wrong:
            return "Incorrect"
        if has_correct or has_right:
            if not has_incorrect and not has_wrong:
                return "Correct"
        
        return "None"
