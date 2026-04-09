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
    if not text:
        return None
        
    results = []
    
    # Strategy 1: Find <json>...</json> blocks (highest priority)
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
        # Look for ```json blocks first
        start = text.find("```json")
        if start != -1:
            start += len("```json")
            end = text.find("```", start)
            if end != -1:
                inner = text[start:end].strip()
                parsed = _try_parse_json(inner)
                if parsed and isinstance(parsed, dict):
                    results.append(parsed)
        
        # If still no results, try generic ``` blocks
        if not results:
            start = text.find("```")
            if start != -1:
                start += len("```")
                end = text.find("```", start)
                if end != -1:
                    inner = text[start:end].strip()
                    # Try to parse as JSON
                    parsed = _try_parse_json(inner)
                    if parsed and isinstance(parsed, dict):
                        results.append(parsed)
    
    # Strategy 3: Find inline JSON objects with brace matching
    if not results:
        brace_start = text.find('{')
        while brace_start != -1:
            # Skip if this brace is inside a string (simple check)
            quote_count = text[:brace_start].count('"') - text[:brace_start].count('\\"')
            if quote_count % 2 == 1:
                # Inside a string, skip
                brace_start = text.find('{', brace_start + 1)
                continue
                
            # Find matching closing brace using counter
            brace_count = 0
            in_string = False
            escape_next = False
            for i, char in enumerate(text[brace_start:]):
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
                            json_str = text[brace_start:brace_start + i + 1]
                            parsed = _try_parse_json(json_str)
                            if parsed and isinstance(parsed, dict):
                                # Only add if it looks like a grading result
                                if any(key in parsed for key in ['assessment', 'response', 'reasoning', 'grade', 'score']):
                                    results.append(parsed)
                            break
            brace_start = text.find('{', brace_start + 1)
    
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
    "assessment": "A clear summary: 'Correct', 'Partially correct', or 'Incorrect'",
    "response": "The final grading decision or score as specified in the guidelines"
}}
</json>

Important: 
- Ensure your response is valid JSON without trailing commas.
- The "assessment" field should be one of: 'Correct', 'Partially correct', or 'Incorrect'.
- The "response" field should contain the final answer that will be used for evaluation."""

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
        
        if not last_message or not last_message.strip():
            return "None"
        
        # Try to extract from JSON first
        extracted = _extract_jsons(last_message)
        if extracted:
            # Try each extracted JSON object, starting from the last one
            for json_obj in reversed(extracted):
                result = self._extract_from_json(json_obj)
                if result != "None":
                    return result
        
        # Fallback: extract from plain text
        return self._extract_from_text(last_message)

    def _extract_from_json(self, json_data: dict) -> str:
        """Extract prediction from JSON data.
        
        Priority order: assessment > grade/score > response > reasoning > first string value
        
        Args:
            json_data: Parsed JSON dict
            
        Returns:
            Extracted prediction string
        """
        if not json_data or not isinstance(json_data, dict):
            return "None"
        
        # Priority fields in order of preference
        priority_fields = ["assessment", "grade", "score", "response", "reasoning", "answer", "result"]
        
        for field in priority_fields:
            if field in json_data:
                value = json_data[field]
                # Handle string values
                if isinstance(value, str):
                    value = value.strip()
                    if value:
                        return value
                # Handle numeric values (convert to string)
                elif isinstance(value, (int, float)):
                    return str(value)
                # Handle boolean values
                elif isinstance(value, bool):
                    return "Correct" if value else "Incorrect"
        
        # Fallback: try any string field
        for key, value in json_data.items():
            if isinstance(value, str) and value.strip():
                return value.strip()
            # Also check for nested dicts that might contain assessment
            elif isinstance(value, dict):
                nested = self._extract_from_json(value)
                if nested != "None":
                    return nested
        
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
        
        # Strategy 1: Look for explicit assessment keywords with word boundaries
        # Priority order: Partially correct > Incorrect > Correct
        assessment_patterns = [
            (r'\b(partially correct|partial credit|partially right|mostly correct|incomplete solution|partial solution)\b', "Partially correct"),
            (r'\b(incorrect|wrong|false|no|fail|rejected|invalid|inaccurate|error|not correct)\b', "Incorrect"),
            (r'\b(correct|right|true|yes|pass|accepted|valid|accurate|full marks|full credit)\b', "Correct"),
        ]
        
        for pattern, assessment in assessment_patterns:
            if re.search(pattern, text_lower):
                return assessment
        
        # Strategy 2: Try to find text after common markers (case-insensitive)
        markers = [
            "Assessment:", "Response:", "Answer:", "Grade:", "Score:", "Result:",
            "Final Answer:", "Conclusion:", "Verdict:", "Decision:", "Evaluation:",
            "Judgment:", "Rating:", "Status:", "Outcome:", "Determination:",
            "The student answer is", "I conclude that", "My assessment is"
        ]
        
        for marker in markers:
            marker_lower = marker.lower()
            if marker_lower in text_lower:
                # Find the position in original case
                idx = text_lower.find(marker_lower)
                if idx != -1:
                    # Get the text after the marker (preserve original case)
                    after_marker = text[idx + len(marker):].strip()
                    # Get first non-empty line that contains meaningful content
                    for line in after_marker.split('\n'):
                        candidate = line.strip()
                        # Skip empty lines, JSON markers, and code blocks
                        if (candidate and 
                            len(candidate) > 1 and
                            not candidate.startswith(('{', '[', '<', '`', '-', '*', '#', '|')) and
                            not candidate.endswith(':')):
                            # Remove common punctuation at the end
                            candidate = candidate.rstrip('.;,!')
                            if candidate:
                                # Check if this candidate contains an assessment
                                candidate_lower = candidate.lower()
                                for pattern, assessment in assessment_patterns:
                                    if re.search(pattern, candidate_lower):
                                        return assessment
                                # If no assessment found, return the candidate if it's short enough
                                if len(candidate) < 100:
                                    return candidate
        
        # Strategy 3: Look for assessment in the last few lines
        lines = text.strip().split('\n')
        # Check last 5 lines for assessment keywords
        for line in reversed(lines[-5:] if len(lines) > 5 else lines):
            stripped = line.strip()
            # Skip lines that are too short, start with special chars, or are just punctuation
            if (stripped and 
                len(stripped) > 2 and 
                len(stripped) < 200 and
                not stripped.startswith(('<', '{', '[', '`', '-', '*', '#', '|', '/')) and
                not stripped.endswith(':')):
                stripped_lower = stripped.lower()
                # Check for assessment keywords in this line
                for pattern, assessment in assessment_patterns:
                    if re.search(pattern, stripped_lower):
                        return assessment
        
        # Strategy 4: Fallback - extract the last substantial non-empty line
        for line in reversed(lines):
            stripped = line.strip()
            # Skip lines that are too short, start with special chars, or are just punctuation
            if (stripped and 
                len(stripped) > 2 and 
                len(stripped) < 100 and
                not stripped.startswith(('<', '{', '[', '`', '-', '*', '#', '|', '/')) and
                not stripped.endswith(':')):
                # Remove trailing punctuation
                cleaned = stripped.rstrip('.;,!')
                if cleaned:
                    return cleaned
        
        return "None"

    def _normalize_prediction(self, prediction: str) -> str:
        """Normalize prediction to standard assessment values.
        
        Args:
            prediction: Raw prediction string
            
        Returns:
            Normalized prediction: 'Correct', 'Partially correct', 'Incorrect', or original
        """
        if not prediction:
            return "None"
            
        prediction = str(prediction).strip()
        prediction_lower = prediction.lower()
        
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
            "mostly correct", "incomplete solution", "partial solution",
            "incomplete", "partial", "partially"
        ]
        for var in partial_variations:
            if contains_word(prediction_lower, var):
                return "Partially correct"
        
        # Check "incorrect" before "correct" to avoid substring match issues
        incorrect_variations = [
            "incorrect", "wrong", "false", "no", "fail", "rejected",
            "zero", "0", "invalid", "inaccurate", "error", "not correct",
            "not right", "not valid"
        ]
        for var in incorrect_variations:
            if contains_word(prediction_lower, var):
                return "Incorrect"
        
        correct_variations = [
            "correct", "right", "true", "yes", "pass", "accepted",
            "full marks", "full credit", "valid", "accurate", "perfect",
            "complete solution", "correct solution"
        ]
        for var in correct_variations:
            if contains_word(prediction_lower, var):
                return "Correct"
        
        # If prediction is very short (1-2 words), try to match common grading terms
        words = prediction_lower.split()
        if len(words) <= 2:
            # Check for numeric scores that indicate partial credit
            if re.search(r'\b[1-9]/[1-9]\b', prediction_lower):
                return "Partially correct"
            # Check for "0" or "0/" which indicates incorrect
            if re.search(r'\b0\b|\b0/', prediction_lower):
                return "Incorrect"
            # Check for full score patterns
            if re.search(r'\b(10|100|full|complete)\b', prediction_lower):
                return "Correct"
        
        return prediction
