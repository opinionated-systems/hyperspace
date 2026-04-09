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
    cleaned = json_str.strip()
    
    # Remove markdown code block markers if present
    if cleaned.startswith('```'):
        # Find the end of the opening marker
        first_newline = cleaned.find('\n')
        if first_newline != -1:
            cleaned = cleaned[first_newline:].strip()
        else:
            cleaned = cleaned[3:].strip()
    if cleaned.endswith('```'):
        cleaned = cleaned[:-3].strip()
    
    # Remove trailing commas before closing braces/brackets
    cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
    
    # Fix single quotes used as JSON delimiters (but not within values)
    # Handle key names with single quotes
    cleaned = re.sub(r"'([^']*?)':\s*", r'"\1": ', cleaned)
    # Handle string values with single quotes
    cleaned = re.sub(r":\s*'([^']*?)'([,}\]])", r': "\1"\2', cleaned)
    
    # Fix unquoted keys (common in LLM output)
    # Match word characters followed by colon at start of object or after comma
    cleaned = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', cleaned)
    
    # Remove control characters
    cleaned = re.sub(r'[\x00-\x1F\x7F]', '', cleaned)
    
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
        if parsed:
            results.append(parsed)
    
    # Strategy 2: Find markdown json blocks ```json...``` or ```...```
    if not results:
        # Look for ```json or ``` blocks
        md_start = 0
        while True:
            json_marker = text.find("```json", md_start)
            plain_marker = text.find("```\n", md_start)
            
            if json_marker != -1 and (plain_marker == -1 or json_marker < plain_marker):
                start = json_marker + 7
            elif plain_marker != -1:
                start = plain_marker + 4
            else:
                break
                
            end = text.find("```", start)
            if end == -1:
                break
                
            inner = text[start:end].strip()
            # Try to find JSON within the code block
            if inner.startswith('{'):
                parsed = _try_parse_json(inner)
                if parsed:
                    results.append(parsed)
                    break
            # Try extracting from nested braces
            brace_start = inner.find('{')
            if brace_start != -1:
                parsed = _try_parse_json(inner[brace_start:])
                if parsed:
                    results.append(parsed)
                    break
            md_start = end + 3
    
    # Strategy 3: Find inline JSON objects with brace matching (improved)
    if not results:
        brace_start = text.find('{')
        while brace_start != -1:
            # Skip if this brace is inside a code block we already checked
            # Find matching closing brace using counter
            brace_count = 0
            in_string = False
            escape_next = False
            
            for i, char in enumerate(text[brace_start:]):
                if escape_next:
                    escape_next = False
                    continue
                if char == '\\' and in_string:
                    escape_next = True
                    continue
                if char == '"' and (i == 0 or text[brace_start + i - 1] != '\\'):
                    in_string = not in_string
                    continue
                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_str = text[brace_start:brace_start + i + 1]
                            parsed = _try_parse_json(json_str)
                            if parsed and isinstance(parsed, dict):
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
    "assessment": "One of: Correct, Partially correct, or Incorrect",
    "response": "The final grading decision or score as specified in the guidelines"
}}
</json>

Important: 
- Your response MUST be valid JSON inside the <json>...</json> tags.
- Do NOT include trailing commas in the JSON.
- The "assessment" field MUST be exactly one of: 'Correct', 'Partially correct', or 'Incorrect'.
- The "response" field should contain the final answer that will be used for evaluation.
- Use double quotes for all JSON strings, not single quotes."""

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
        
        Priority order: assessment > response > reasoning > first string value
        
        Args:
            json_data: Parsed JSON dict
            
        Returns:
            Extracted prediction string
        """
        # Priority fields in order of preference
        priority_fields = ["assessment", "response", "answer", "result", "conclusion", "verdict", "reasoning"]
        
        for field in priority_fields:
            if field in json_data and isinstance(json_data[field], str):
                value = json_data[field].strip()
                if value:
                    return value
            # Also check for nested dicts that might contain the field
            elif field in json_data and isinstance(json_data[field], dict):
                nested = json_data[field]
                for nested_key in ["value", "text", "content", "answer", "result"]:
                    if nested_key in nested and isinstance(nested[nested_key], str):
                        value = nested[nested_key].strip()
                        if value:
                            return value
        
        # Fallback: try any string field at top level
        for key, value in json_data.items():
            if isinstance(value, str) and value.strip():
                return value
            # Check for nested structures
            elif isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    if isinstance(nested_value, str) and nested_value.strip():
                        return nested_value
        
        return "None"

    def _extract_from_text(self, text: str) -> str:
        """Extract prediction from plain text when JSON parsing fails.
        
        Args:
            text: Raw text to extract prediction from
            
        Returns:
            Extracted prediction string
        """
        text_lower = text.lower()
        
        # Priority 1: Look for explicit assessment keywords with context
        # Check for "partially correct" patterns first (highest priority to avoid misclassification)
        partial_patterns = [
            r'\bpartially\s+correct\b',
            r'\bpartial\s+credit\b',
            r'\bpartially\s+right\b',
            r'\bmostly\s+correct\b',
            r'\bincomplete\s+but\s+correct\b',
            r'\bpartial\s+marks\b',
            r'\bhalf\s+correct\b',
            r'\bsome\s+credit\b',
        ]
        for pattern in partial_patterns:
            if re.search(pattern, text_lower):
                return "Partially correct"
        
        # Check for "incorrect" patterns
        incorrect_patterns = [
            r'\bincorrect\b',
            r'\bwrong\b',
            r'\bfalse\b',
            r'\bnot\s+correct\b',
            r'\bis\s+incorrect\b',
            r'\bassessment.*incorrect\b',
            r'\bgrade.*incorrect\b',
            r'\bevaluation.*incorrect\b',
            r'\bnot\s+valid\b',
            r'\bnot\s+accurate\b',
            r'\bfail\w*\b',
            r'\breject\w*\b',
            r'\bzero\s+marks\b',
            r'\bno\s+credit\b',
        ]
        for pattern in incorrect_patterns:
            if re.search(pattern, text_lower):
                return "Incorrect"
        
        # Check for "correct" patterns (lowest priority)
        correct_patterns = [
            r'\bcorrect\b',
            r'\bis\s+correct\b',
            r'\bassessment.*correct\b',
            r'\bgrade.*correct\b',
            r'\bevaluation.*correct\b',
            r'\bright\b',
            r'\btrue\b',
            r'\bvalid\s+solution\b',
            r'\baccurate\b',
            r'\bpass\w*\b',
            r'\baccept\w*\b',
            r'\bfull\s+marks\b',
            r'\bfull\s+credit\b',
            r'\ball\s+credit\b',
        ]
        for pattern in correct_patterns:
            if re.search(pattern, text_lower):
                return "Correct"
        
        # Priority 2: Try to find text after common markers (case-insensitive)
        markers = [
            "Assessment:", "Response:", "Answer:", "Grade:", "Score:", "Result:",
            "Final Answer:", "Conclusion:", "Verdict:", "Decision:", "Evaluation:",
            "Judgment:", "Rating:", "Status:", "Outcome:", "Determination:",
            "The student", "Student's answer", "This answer"
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
                                return candidate
        
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
                    return cleaned
        
        return "None"

    def _normalize_prediction(self, prediction: str) -> str:
        """Normalize prediction to standard assessment values.
        
        Args:
            prediction: Raw prediction string
            
        Returns:
            Normalized prediction: 'Correct', 'Partially correct', 'Incorrect', or original
        """
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
            "mostly correct", "incomplete but correct", "partial marks",
            "half correct", "some credit", "partially valid"
        ]
        for var in partial_variations:
            if contains_word(prediction_lower, var):
                return "Partially correct"
        
        # Check "incorrect" before "correct" to avoid substring match issues
        incorrect_variations = [
            "incorrect", "wrong", "false", "not correct", "is incorrect",
            "assessment is incorrect", "grade is incorrect", "evaluation is incorrect",
            "not valid", "not accurate", "failed", "rejected",
            "zero marks", "no credit", "invalid", "inaccurate", "error"
        ]
        for var in incorrect_variations:
            if contains_word(prediction_lower, var):
                return "Incorrect"
        
        correct_variations = [
            "correct", "is correct", "assessment is correct", "grade is correct",
            "evaluation is correct", "right", "true", "valid solution",
            "accurate", "passed", "accepted", "full marks", "full credit",
            "all credit", "completely correct"
        ]
        for var in correct_variations:
            if contains_word(prediction_lower, var):
                return "Correct"
        
        # Additional check: if prediction contains numbers, try to interpret
        # Common grading scales: 0-10, 0-100, etc.
        import re as re_module
        numbers = re_module.findall(r'\d+', prediction)
        if numbers:
            # Try to interpret numeric scores
            for num_str in numbers:
                try:
                    num = int(num_str)
                    # If we see 0, likely incorrect
                    if num == 0:
                        return "Incorrect"
                    # If we see partial scores in common ranges
                    if 1 <= num <= 5:  # Partial credit range
                        return "Partially correct"
                    if num >= 8:  # High score
                        return "Correct"
                except ValueError:
                    continue
        
        return prediction
