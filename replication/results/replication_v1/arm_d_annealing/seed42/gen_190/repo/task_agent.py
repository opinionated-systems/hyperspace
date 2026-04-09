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
    # Remove control characters
    cleaned = ''.join(char for char in cleaned if ord(char) >= 32 or char in '\n\r\t')
    # Normalize whitespace
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
    if not text or not isinstance(text, str):
        return None
        
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
    
    # Strategy 2: Find markdown json blocks ```json...```
    if not results:
        for pattern in ["```json", "```"]:
            start = text.find(pattern)
            if start != -1:
                end = text.find("```", start + len(pattern))
                if end != -1:
                    inner = text[start + len(pattern):end].strip()
                    parsed = _try_parse_json(inner)
                    if parsed and isinstance(parsed, dict):
                        results.append(parsed)
                        break
    
    # Strategy 3: Find inline JSON objects with brace matching
    if not results:
        brace_start = text.find('{')
        while brace_start != -1:
            # Find matching closing brace using counter
            brace_count = 0
            match_end = -1
            for i, char in enumerate(text[brace_start:]):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        match_end = brace_start + i + 1
                        json_str = text[brace_start:match_end]
                        parsed = _try_parse_json(json_str)
                        if parsed and isinstance(parsed, dict):
                            results.append(parsed)
                        break
            # Advance past this brace pair to avoid overlapping matches
            next_start = text.find('{', brace_start + 1)
            if match_end > brace_start and next_start != -1 and next_start < match_end:
                # Next '{' is inside the current JSON object, skip to after match_end
                brace_start = text.find('{', match_end)
            else:
                brace_start = next_start
    
    return results if results else None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _build_grading_prompt(self, inputs: dict) -> str:
        """Build a structured prompt for IMO grading with chain-of-thought reasoning.
        
        Args:
            inputs: Dictionary containing problem data with keys:
                - domain: The mathematical domain
                - problem: The problem statement
                - solution: The official solution
                - grading_guidelines: Guidelines for grading
                - student_answer: The student's submitted answer
                
        Returns:
            A formatted prompt string for the LLM
        """
        # Extract input fields with defaults
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
        
        # Try to extract from JSON first
        extracted = _extract_jsons(last_message)
        if extracted:
            return self._extract_from_json(extracted[-1])
        
        # Fallback: extract from plain text
        return self._extract_from_text(last_message)

    def _extract_from_json(self, json_data: dict) -> str:
        """Extract prediction from JSON data.
        
        Priority order: assessment > response > reasoning > conclusion > result > first string value
        
        Args:
            json_data: Parsed JSON dict
            
        Returns:
            Extracted prediction string
        """
        # Priority fields in order of preference
        priority_fields = [
            "assessment", "response", "reasoning", "conclusion", 
            "result", "grade", "score", "evaluation", "verdict",
            "decision", "answer", "output", "summary"
        ]
        
        for field in priority_fields:
            if field in json_data:
                value = json_data[field]
                # Handle both string and list values
                if isinstance(value, str):
                    stripped = value.strip()
                    if stripped:
                        return stripped
                elif isinstance(value, list) and value:
                    # If it's a list, try to get the first string element
                    for item in value:
                        if isinstance(item, str) and item.strip():
                            return item.strip()
                elif isinstance(value, (int, float)):
                    # Convert numeric values to string
                    return str(value)
        
        # Fallback: try any string field
        for key, value in json_data.items():
            if isinstance(value, str):
                stripped = value.strip()
                if stripped:
                    return stripped
            elif isinstance(value, (int, float)):
                return str(value)
        
        return "None"

    def _extract_from_text(self, text: str) -> str:
        """Extract prediction from plain text when JSON parsing fails.
        
        Args:
            text: Raw text to extract prediction from
            
        Returns:
            Extracted prediction string
        """
        # Try to find text after common markers (case-insensitive)
        markers = [
            "Assessment:", "Response:", "Answer:", "Grade:", "Score:", "Result:",
            "Final Answer:", "Conclusion:", "Verdict:", "Decision:", "Evaluation:",
            "Judgment:", "Ruling:", "Determination:", "Finding:", "Status:"
        ]
        
        text_lower = text.lower()
        for marker in markers:
            marker_lower = marker.lower()
            if marker_lower in text_lower:
                # Find position in original case-sensitive text
                pos = text_lower.find(marker_lower)
                if pos != -1:
                    # Extract after the marker (using original text for case preservation)
                    after_marker = text[pos + len(marker):].strip()
                    # Get first non-empty line
                    for line in after_marker.split('\n'):
                        candidate = line.strip()
                        # Skip empty lines, JSON markers, and code blocks
                        if candidate and not candidate.startswith(('{', '<', '`', '[')):
                            # Remove common prefixes like quotes or dashes
                            candidate = candidate.lstrip('"\'-*• ')
                            if candidate:
                                return candidate
        
        # Fallback: extract the last non-empty line that looks like an assessment
        lines = text.strip().split('\n')
        for line in reversed(lines):
            stripped = line.strip()
            # Skip empty lines, JSON markers, code blocks, and common non-content lines
            if stripped and not stripped.startswith(('<', '{', '`', '[', '-', '*', '•')):
                # Skip lines that are just punctuation or very short
                if len(stripped) > 2 and not stripped.rstrip('.!?').isdigit():
                    # Check if it looks like an assessment (contains key words)
                    lower = stripped.lower()
                    if any(word in lower for word in ['correct', 'incorrect', 'partial', 'right', 'wrong', 'pass', 'fail']):
                        return stripped
        
        # Last resort: return the last substantial line
        for line in reversed(lines):
            stripped = line.strip()
            if stripped and not stripped.startswith(('<', '{', '`', '[')):
                if len(stripped) > 2:
                    return stripped
        
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
            "mostly correct", "incomplete solution", "partial solution",
            "partially solved", "partial success", "partial marks",
            "half correct", "half right", "partial"
        ]
        for var in partial_variations:
            if contains_word(prediction_lower, var):
                return "Partially correct"
        
        # Check "incorrect" before "correct" to avoid substring match issues
        incorrect_variations = [
            "incorrect", "wrong", "false", "no", "fail", "rejected",
            "zero", "0 points", "0/", "/0", "invalid", "inaccurate", "error",
            "not correct", "not right", "not valid", "unsolved",
            "does not solve", "failed", "failure"
        ]
        for var in incorrect_variations:
            if contains_word(prediction_lower, var):
                return "Incorrect"
        
        correct_variations = [
            "correct", "right", "true", "yes", "pass", "accepted",
            "full marks", "full credit", "valid", "accurate", "perfect",
            "complete solution", "solved correctly", "correctly solved",
            "success", "successful"
        ]
        for var in correct_variations:
            if contains_word(prediction_lower, var):
                return "Correct"
        
        # Check for numeric scores that indicate partial credit
        # e.g., "1/2", "0.5", "50%", "3 points out of 7"
        # Pattern for fractions like "1/2", "2/7", etc.
        fraction_match = re.search(r'(\d+)\s*/\s*(\d+)', prediction_lower)
        if fraction_match:
            numerator = int(fraction_match.group(1))
            denominator = int(fraction_match.group(2))
            if denominator > 0:
                ratio = numerator / denominator
                if ratio == 1.0:
                    return "Correct"
                elif ratio == 0.0:
                    return "Incorrect"
                elif 0 < ratio < 1:
                    return "Partially correct"
        
        # Pattern for percentages
        percent_match = re.search(r'(\d+)%', prediction_lower)
        if percent_match:
            percent = int(percent_match.group(1))
            if percent == 100:
                return "Correct"
            elif percent == 0:
                return "Incorrect"
            elif 0 < percent < 100:
                return "Partially correct"
        
        return prediction
