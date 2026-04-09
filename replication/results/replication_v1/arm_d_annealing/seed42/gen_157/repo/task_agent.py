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
from typing import Any

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


def _clean_json_string(json_str: str) -> str:
    """Clean up common JSON formatting issues.
    
    Args:
        json_str: Raw JSON string that may have formatting issues
        
    Returns:
        Cleaned JSON string
    """
    # Remove trailing commas before closing braces/brackets
    cleaned = re.sub(r',(\s*[}\]])', r'\1', json_str)
    # Fix single quotes used as JSON delimiters (but not within values)
    cleaned = re.sub(r"'([^']*?)':\s*", r'"\1": ', cleaned)
    cleaned = re.sub(r":\s*'([^']*?)'", r': "\1"', cleaned)
    # Remove comments (both // and /* */ style)
    cleaned = re.sub(r'//[^\n]*', '', cleaned)
    cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
    # Remove control characters (except whitespace)
    cleaned = ''.join(char for char in cleaned if ord(char) >= 32 or char in '\n\r\t')
    # Remove leading/trailing whitespace
    cleaned = cleaned.strip()
    return cleaned


def _try_parse_json(json_str: str) -> dict | None:
    """Try to parse a JSON string, with cleanup on failure.
    
    Args:
        json_str: JSON string to parse
        
    Returns:
        Parsed dict or None if parsing fails
    """
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            cleaned = _clean_json_string(json_str)
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles markdown code blocks with json and inline JSON objects.
    
    Args:
        text: Input text containing JSON objects
        
    Returns:
        List of extracted JSON objects, or None if none found
    """
    results = []
    
    # Strategy 1: Find <json>...</json> blocks
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
        
        parsed = _try_parse_json(inner)
        if parsed and isinstance(parsed, dict):
            results.append(parsed)
    
    # Strategy 2: Find markdown json blocks ```json...``` or ```...```
    if not results:
        # Look for all code blocks and try to parse each
        md_start = 0
        while True:
            start = text.find("```", md_start)
            if start == -1:
                break
            # Check if it's a json block
            block_end = start + 3
            lang = ""
            # Check for optional newline after opening ```
            next_chars = text[block_end:block_end + 5].lower()
            if next_chars.startswith("json"):
                block_end += 4
                lang = "json"
                # Skip newline if present
                if text[block_end:block_end + 1] == "\n":
                    block_end += 1
            elif next_chars.startswith("\n"):
                block_end += 1
            # Find closing ```
            end = text.find("```", block_end)
            if end == -1:
                break
            inner = text[block_end:end].strip()
            md_start = end + 3
            # Try to parse if it's json or if we haven't found anything yet
            if lang == "json" or not results:
                parsed = _try_parse_json(inner)
                if parsed and isinstance(parsed, dict):
                    results.append(parsed)
                    if lang == "json":
                        break  # Found explicit json block, stop searching
    
    # Strategy 3: Find inline JSON objects with brace matching
    if not results:
        brace_start = text.find('{')
        while brace_start != -1:
            # Find matching closing brace using counter with proper string handling
            brace_count = 1
            in_string = False
            escape_next = False
            # Skip the first char (opening brace) since we already counted it
            for i, char in enumerate(text[brace_start + 1:]):
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
                            # i is offset by 1 since we started from brace_start + 1
                            json_str = text[brace_start:brace_start + i + 2]
                            parsed = _try_parse_json(json_str)
                            if parsed and isinstance(parsed, dict):
                                results.append(parsed)
                            break
            brace_start = text.find('{', brace_start + 1)
    
    # Strategy 4: Try to find JSON-like patterns with regex as last resort
    if not results:
        # Look for patterns like {"key": "value"} or {"key": value}
        json_pattern = re.search(r'\{[^{}]*"[^"]+"\s*:\s*[^}]+\}', text)
        if json_pattern:
            parsed = _try_parse_json(json_pattern.group(0))
            if parsed and isinstance(parsed, dict):
                results.append(parsed)
    
    return results if results else None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _build_grading_prompt(self, inputs: dict) -> str:
        """Build a structured prompt for IMO grading with chain-of-thought reasoning."""
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        return f"""You are an expert mathematical grader for the International Mathematical Olympiad (IMO).

Your task is to evaluate a student's solution to a mathematical problem. Follow these steps carefully:

1. **Understand the Problem**: Read the problem statement carefully and identify what is being asked.

2. **Review the Official Solution**: Study the provided official solution to understand the expected approach and key insights.

3. **Analyze the Student's Answer**: Examine the student's solution step by step, checking:
   - Mathematical correctness of each step
   - Logical soundness and justification
   - Completeness (does it address all parts of the problem?)
   - Clarity of presentation

4. **Apply Grading Guidelines**: Use the provided grading guidelines to assess the student's work objectively.

5. **Provide Your Evaluation**: Give a clear, definitive assessment.

---

**Domain**: {domain}

**Problem**:
```
{problem}
```

**Official Solution**:
```
{solution}
```

**Grading Guidelines**:
```
{grading_guidelines}
```

**Student's Answer**:
```
{student_answer}
```

---

**Your Task**:

First, think through your evaluation step by step (chain-of-thought reasoning). Consider:
- Does the student's approach align with the official solution?
- Are the mathematical steps correct?
- Is the logic sound and well-justified?
- Does the student address all parts of the problem?
- What score would you assign based on the grading guidelines?

Then, provide your final assessment in the following JSON format. Be precise and concise:

<json>
{{
    "reasoning": "Your detailed step-by-step evaluation and reasoning process",
    "assessment": "Correct",
    "response": "The final grading decision or score as specified in the guidelines (e.g., 'Score: 2/6' or '0 points')"
}}
</json>

Important: 
- Ensure your response is valid JSON without trailing commas.
- The "assessment" field MUST be exactly one of: 'Correct', 'Partially correct', 'Incorrect', or 'Almost' (case-sensitive).
  - Use 'Correct' for fully correct answers (full marks)
  - Use 'Almost' for answers with minor errors (high partial credit)
  - Use 'Partially correct' for partially correct answers (some credit)
  - Use 'Incorrect' for wrong or invalid answers (no credit)
- The "response" field should contain the final answer including the score (e.g., "Score: 2/6").
- Do not include any text after the closing </json> tag."""

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_grading_prompt(inputs)

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with improved error handling
        prediction = self._extract_prediction(msg_history)
        
        # Normalize prediction to expected format
        prediction = self._normalize_prediction(prediction)
        
        return prediction, msg_history

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history.
        
        Args:
            msg_history: List of message dicts from LLM conversation
            
        Returns:
            Extracted prediction string
        """
        if not msg_history:
            return "None"
        
        # Handle both "text" (paper format) and "content" (OpenAI format) fields
        last_message = msg_history[-1].get("text") or msg_history[-1].get("content", "")
        
        # Try to extract from JSON first
        extracted = _extract_jsons(last_message)
        if extracted:
            return self._extract_from_json(extracted[-1])
        
        # Fallback: extract from plain text
        return self._extract_from_text(last_message)

    def _extract_from_json(self, json_data: dict) -> str:
        """Extract prediction from JSON data.
        
        Priority order: assessment > response > answer > result > conclusion > verdict > reasoning > first string value
        
        Args:
            json_data: Parsed JSON dict
            
        Returns:
            Extracted prediction string
        """
        # Priority fields in order of preference (most specific to least specific)
        priority_fields = [
            "assessment", "evaluation", "verdict", "judgment", "decision",
            "response", "answer", "result", "conclusion", "grade", "score",
            "status", "outcome", "determination", "reasoning", "explanation"
        ]
        
        for field in priority_fields:
            if field in json_data:
                value = json_data[field]
                # Handle both string and list values
                if isinstance(value, str):
                    value = value.strip()
                    if value:
                        return value
                elif isinstance(value, list) and value:
                    # If it's a list, try to get the first string element
                    first_item = value[0]
                    if isinstance(first_item, str):
                        return first_item.strip()
                elif isinstance(value, (int, float, bool)):
                    return str(value)
        
        # Fallback: try any string field
        for key, value in json_data.items():
            if isinstance(value, str) and value.strip():
                return value
            elif isinstance(value, (int, float, bool)):
                return str(value)
        
        return "None"

    def _extract_from_text(self, text: str) -> str:
        """Extract prediction from plain text when JSON parsing fails.
        
        Args:
            text: Raw text to extract prediction from
            
        Returns:
            Extracted prediction string
        """
        if not text or not text.strip():
            return "None"
            
        text_lower = text.lower()
        
        # First, try to extract score from patterns like "Score: X/Y" or "X/Y points"
        score_patterns = [
            r'[Ss]core:\s*(\d+)\s*/\s*(\d+)',
            r'(\d+)\s*/\s*(\d+)\s*(?:points?|pts?)',
            r'(\d+)\s+out\s+of\s+(\d+)',
            r'(?:earned|got|received?)\s+(\d+)\s*/\s*(\d+)',
        ]
        
        for pattern in score_patterns:
            match = re.search(pattern, text)
            if match:
                score = int(match.group(1))
                total = int(match.group(2))
                if total > 0:
                    ratio = score / total
                    if ratio == 1.0:
                        return "Correct"
                    elif ratio >= 0.75:
                        return "Almost"
                    elif ratio >= 0.4:
                        return "Partially correct"
                    else:
                        return "Incorrect"
        
        # Try to extract percentage scores
        percent_match = re.search(r'(\d+)%', text_lower)
        if percent_match:
            percent = int(percent_match.group(1))
            if percent >= 95:
                return "Correct"
            elif percent >= 75:
                return "Almost"
            elif percent >= 40:
                return "Partially correct"
            else:
                return "Incorrect"
        
        # Look for explicit assessment keywords first (most reliable)
        # Check in order of specificity: partial > almost > incorrect > correct
        assessment_patterns = [
            (r'\b(partially correct|partial credit|partially right|mostly correct|incomplete|partial)\b', "Partially correct"),
            (r'\b(almost correct|nearly correct|almost there|minor error|small mistake|close)\b', "Almost"),
            (r'\b(incorrect|wrong|false|no|fail|rejected|invalid|inaccurate|error)\b', "Incorrect"),
            (r'\b(correct|right|true|yes|pass|accepted|valid|accurate)\b', "Correct"),
        ]
        
        for pattern, assessment in assessment_patterns:
            if re.search(pattern, text_lower):
                return assessment
        
        # Try to find text after common markers (case-insensitive)
        markers = [
            "Assessment:", "Response:", "Answer:", "Grade:", "Score:", "Result:",
            "Final Answer:", "Conclusion:", "Verdict:", "Decision:", "Evaluation:",
            "Judgment:", "Rating:", "Status:", "Outcome:", "Determination:",
            "Final Assessment:", "Summary:", "Grade is", "The answer is"
        ]
        
        for marker in markers:
            marker_lower = marker.lower()
            if marker_lower in text_lower:
                # Find the position in original case
                idx = text_lower.find(marker_lower)
                if idx != -1:
                    # Get the text after the marker (preserve original case)
                    after_marker = text[idx + len(marker):].strip()
                    # Get first non-empty line
                    for line in after_marker.split('\n'):
                        candidate = line.strip()
                        if candidate and not candidate.startswith(('{', '[', '<', '`')):
                            # Remove common punctuation at the end
                            candidate = candidate.rstrip('.;,!')
                            if candidate:
                                # Check if this candidate contains an assessment
                                candidate_lower = candidate.lower()
                                for pattern, assessment in assessment_patterns:
                                    if re.search(pattern, candidate_lower):
                                        return assessment
                                return candidate
        
        # Look for assessment in quotes or emphasized text
        quote_patterns = [
            r'["\'](Correct|Partially correct|Incorrect|Almost)["\']',
            r'\*\*(Correct|Partially correct|Incorrect|Almost)\*\*',
            r'\*(Correct|Partially correct|Incorrect|Almost)\*',
            r'`(Correct|Partially correct|Incorrect|Almost)`',
        ]
        for pattern in quote_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).capitalize()
        
        # Fallback: extract the last substantial non-empty line
        lines = text.strip().split('\n')
        for line in reversed(lines):
            stripped = line.strip()
            # Skip lines that are too short, start with special chars, or are just punctuation
            if (stripped and 
                len(stripped) > 2 and 
                not stripped.startswith(('<', '{', '[', '`', '-', '*', '#')) and
                not stripped.endswith(':')):
                # Remove trailing punctuation
                cleaned = stripped.rstrip('.;,!')
                if cleaned:
                    # Check if this contains an assessment
                    cleaned_lower = cleaned.lower()
                    for pattern, assessment in assessment_patterns:
                        if re.search(pattern, cleaned_lower):
                            return assessment
                    return cleaned
        
        return "None"

    def _normalize_prediction(self, prediction: str) -> str:
        """Normalize prediction to standard assessment values.
        
        Args:
            prediction: Raw prediction string
            
        Returns:
            Normalized prediction: 'Correct', 'Partially correct', 'Incorrect', 
            'Almost', or the original prediction if no match found
        """
        prediction = str(prediction).strip()
        prediction_lower = prediction.lower()
        
        # First, try to extract score from patterns like "Score: X/Y" or "X/Y points" or "X out of Y"
        score_patterns = [
            r'[Ss]core:\s*(\d+)\s*/\s*(\d+)',
            r'(\d+)\s*/\s*(\d+)\s*(?:points?|pts?)',
            r'(\d+)\s+out\s+of\s+(\d+)',
            r'(?:earned|got|received?)\s+(\d+)\s*/\s*(\d+)',
        ]
        
        for pattern in score_patterns:
            match = re.search(pattern, prediction)
            if match:
                score = int(match.group(1))
                total = int(match.group(2))
                if total > 0:
                    ratio = score / total
                    if ratio == 1.0:
                        return "Correct"
                    elif ratio >= 0.75:
                        return "Almost"
                    elif ratio >= 0.4:
                        return "Partially correct"
                    else:
                        return "Incorrect"
        
        # Try to extract percentage scores
        percent_match = re.search(r'(\d+)%', prediction_lower)
        if percent_match:
            percent = int(percent_match.group(1))
            if percent >= 95:
                return "Correct"
            elif percent >= 75:
                return "Almost"
            elif percent >= 40:
                return "Partially correct"
            else:
                return "Incorrect"
        
        # Helper to check if a variation appears as a whole word/phrase
        def contains_word(text: str, word: str) -> bool:
            # For multi-word phrases, use simple containment
            if " " in word:
                return word in text
            # For single words, use word boundary regex
            pattern = r'\b' + re.escape(word) + r'\b'
            return re.search(pattern, text) is not None
        
        # Priority: Partially correct > Incorrect > Correct (to avoid misclassification)
        # Check "partially correct" first to avoid matching "correct" within it
        
        partial_variations = [
            "partially correct", "partial credit", "partially right", 
            "mostly correct", "incomplete", "partial"
        ]
        for var in partial_variations:
            if contains_word(prediction_lower, var):
                return "Partially correct"
        
        # Check "almost" and "nearly" for the Almost category
        almost_variations = [
            "almost correct", "nearly correct", "almost there", 
            "minor error", "small mistake", "close"
        ]
        for var in almost_variations:
            if contains_word(prediction_lower, var):
                return "Almost"
        
        # Check "incorrect" before "correct" to avoid substring match issues
        incorrect_variations = [
            "incorrect", "wrong", "false", "no", "fail", "rejected",
            "zero", "0", "invalid", "inaccurate", "error"
        ]
        for var in incorrect_variations:
            if contains_word(prediction_lower, var):
                return "Incorrect"
        
        correct_variations = [
            "correct", "right", "true", "yes", "pass", "accepted",
            "full marks", "full credit", "valid", "accurate"
        ]
        for var in correct_variations:
            if contains_word(prediction_lower, var):
                return "Correct"
        
        return prediction


# Module-level function for backward compatibility
def solve_task(inputs: dict, model: str = EVAL_MODEL) -> tuple[str, list[dict]]:
    """Solve a task using the TaskAgent.
    
    Args:
        inputs: Task inputs dictionary
        model: Model to use for solving
        
    Returns:
        Tuple of (prediction, message_history)
    """
    agent = TaskAgent(model=model)
    return agent.forward(inputs)
