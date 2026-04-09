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
        # Try ```json first, then plain ```
        for pattern in ["```json", "```"]:
            search_pos = 0
            while True:
                start = text.find(pattern, search_pos)
                if start == -1:
                    break
                # Find the closing ```
                end = text.find("```", start + len(pattern))
                if end == -1:
                    break
                inner = text[start + len(pattern):end].strip()
                # Try to parse the content
                parsed = _try_parse_json(inner)
                if parsed and isinstance(parsed, dict):
                    results.append(parsed)
                    # Continue searching for more JSON blocks
                    search_pos = end + 3
                else:
                    search_pos = end + 3
            if results:
                break
    
    # Strategy 3: Find inline JSON objects with brace matching
    if not results:
        brace_start = text.find('{')
        while brace_start != -1:
            # Find matching closing brace using counter
            brace_count = 0
            for i, char in enumerate(text[brace_start:]):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_str = text[brace_start:brace_start + i + 1]
                        parsed = _try_parse_json(json_str)
                        if parsed and isinstance(parsed, dict):
                            # Check if this JSON has assessment-related fields
                            if any(key in parsed for key in ["assessment", "response", "reasoning", "grade", "score", "evaluation"]):
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
        priority_fields = ["assessment", "response", "reasoning"]
        
        for field in priority_fields:
            if field in json_data and isinstance(json_data[field], str):
                value = json_data[field].strip()
                if value:
                    return value
        
        # Fallback: try any string field
        for key, value in json_data.items():
            if isinstance(value, str) and value.strip():
                return value
        
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
        # Check for "partially correct" first to avoid matching "correct" within it
        partial_patterns = [
            r'\bpartially\s+correct\b',
            r'\bpartial\s+credit\b',
            r'\bpartially\s+right\b',
            r'\bmostly\s+correct\b',
            r'\bincomplete\s+(?:solution|answer|work)\b',
            r'\bpartial\s+(?:solution|answer|credit)\b',
        ]
        for pattern in partial_patterns:
            if re.search(pattern, text_lower):
                return "Partially correct"
        
        # Check for incorrect/negative indicators
        incorrect_patterns = [
            r'\bincorrect\b',
            r'\bwrong\b',
            r'\bfalse\b',
            r'\bfail\b',
            r'\brejected\b',
            r'\binvalid\b',
            r'\binaccurate\b',
            r'\berror\b',
            r'\bnot\s+correct\b',
            r'\bnot\s+valid\b',
            r'\bnot\s+accurate\b',
            r'\bzero\s+(?:marks|points|score)\b',
            r'\b0\s+(?:marks|points|score)\b',
        ]
        for pattern in incorrect_patterns:
            if re.search(pattern, text_lower):
                return "Incorrect"
        
        # Check for correct/positive indicators
        correct_patterns = [
            r'\bcorrect\b',
            r'\bright\b',
            r'\btrue\b',
            r'\bpass\b',
            r'\baccepted\b',
            r'\bvalid\b',
            r'\baccurate\b',
            r'\bfull\s+(?:marks|credit|points|score)\b',
            r'\bcomplete\s+(?:solution|answer)\b',
        ]
        for pattern in correct_patterns:
            if re.search(pattern, text_lower):
                return "Correct"
        
        # Priority 2: Try to find text after common markers (case-insensitive)
        markers = [
            "Assessment:", "Response:", "Answer:", "Grade:", "Score:", "Result:",
            "Final Answer:", "Conclusion:", "Verdict:", "Decision:", "Evaluation:",
            "Judgment:", "Rating:", "Status:", "Outcome:", "Determination:",
            "Summary:", "Final Assessment:", "Overall:", "Grade Assessment:",
            "Evaluation Result:", "Grading Decision:", "Final Grade:", "Score Assessment:"
        ]
        
        for marker in markers:
            marker_lower = marker.lower()
            if marker_lower in text_lower:
                # Find the position in original case
                idx = text_lower.find(marker_lower)
                if idx != -1:
                    # Get the text after the marker (preserve original case)
                    after_marker = text[idx + len(marker):].strip()
                    # Get first non-empty line that has meaningful content
                    for line in after_marker.split('\n'):
                        candidate = line.strip()
                        # Skip empty lines, JSON markers, and very short responses
                        if (candidate and 
                            len(candidate) > 1 and
                            not candidate.startswith(('{', '[', '<', '`', '-', '*', '#', '|')) and
                            not candidate.endswith(':')):
                            # Remove common punctuation at the end
                            candidate = candidate.rstrip('.;,!')
                            if candidate:
                                # Normalize the extracted text
                                normalized = self._normalize_prediction(candidate)
                                if normalized in ["Correct", "Partially correct", "Incorrect"]:
                                    return normalized
                                return candidate
        
        # Priority 3: Look for assessment keywords anywhere in text (broader search)
        broad_assessment_patterns = [
            (r'\b(?:the\s+)?(?:student\s+)?(?:answer|solution|response)\s+is\s+(correct|right|valid|accurate|true|accepted)\b', "Correct"),
            (r'\b(?:the\s+)?(?:student\s+)?(?:answer|solution|response)\s+is\s+(incorrect|wrong|invalid|false|rejected)\b', "Incorrect"),
            (r'\b(?:the\s+)?(?:student\s+)?(?:answer|solution|response)\s+is\s+(partially\s+correct|partially\s+right|incomplete)\b', "Partially correct"),
            (r'\bgrade[d]?\s*[:=]?\s*(correct|right|valid|accurate)\b', "Correct"),
            (r'\bgrade[d]?\s*[:=]?\s*(incorrect|wrong|invalid|false)\b', "Incorrect"),
            (r'\bgrade[d]?\s*[:=]?\s*(partially\s+correct|partial)\b', "Partially correct"),
        ]
        
        for pattern, assessment in broad_assessment_patterns:
            match = re.search(pattern, text_lower)
            if match:
                return assessment
        
        # Priority 4: Extract the last substantial non-empty line
        lines = text.strip().split('\n')
        for line in reversed(lines):
            stripped = line.strip()
            # Skip lines that are too short, start with special chars, or are just punctuation
            if (stripped and 
                len(stripped) > 2 and 
                not stripped.startswith(('<', '{', '[', '`', '-', '*', '#', '|', '/', '\\')) and
                not stripped.endswith(':') and
                not stripped.startswith('//') and
                not stripped.startswith('/*')):
                # Remove trailing punctuation
                cleaned = stripped.rstrip('.;,!')
                if cleaned:
                    # Try to normalize
                    normalized = self._normalize_prediction(cleaned)
                    if normalized in ["Correct", "Partially correct", "Incorrect"]:
                        return normalized
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
            "mostly correct", "incomplete but valid", "partially accurate",
            "partial success", "partial solution", "incomplete solution"
        ]
        for var in partial_variations:
            if contains_word(prediction_lower, var):
                return "Partially correct"
        
        # Check "incorrect" before "correct" to avoid substring match issues
        incorrect_variations = [
            "incorrect", "wrong", "false", "fail", "rejected",
            "invalid", "inaccurate", "error", "not correct", "not valid",
            "not accurate", "not right", "not true", "not accepted",
            "zero marks", "0 marks", "zero points", "0 points",
            "zero score", "0 score", "no credit", "failed"
        ]
        for var in incorrect_variations:
            if contains_word(prediction_lower, var):
                return "Incorrect"
        
        # Single word checks that indicate incorrect
        single_word_incorrect = ["no", "zero", "0"]
        for var in single_word_incorrect:
            if prediction_lower == var or prediction_lower.startswith(var + " "):
                return "Incorrect"
        
        correct_variations = [
            "correct", "right", "true", "pass", "accepted",
            "full marks", "full credit", "valid", "accurate",
            "complete solution", "complete answer", "perfect",
            "excellent", "well done", "success", "successful"
        ]
        for var in correct_variations:
            if contains_word(prediction_lower, var):
                return "Correct"
        
        # Single word checks that indicate correct
        single_word_correct = ["yes"]
        for var in single_word_correct:
            if prediction_lower == var:
                return "Correct"
        
        # Check for numeric scores that indicate correctness
        # Common grading scales: 0-10, 0-100, 0-1
        numeric_match = re.search(r'\b(\d+(?:\.\d+)?)\s*(?:/\s*\d+)?\b', prediction)
        if numeric_match:
            try:
                score = float(numeric_match.group(1))
                # If there's a denominator, calculate percentage
                denom_match = re.search(r'/\s*(\d+(?:\.\d+)?)', prediction)
                if denom_match:
                    denom = float(denom_match.group(1))
                    if denom > 0:
                        percentage = score / denom
                        if percentage >= 0.9:
                            return "Correct"
                        elif percentage >= 0.5:
                            return "Partially correct"
                        else:
                            return "Incorrect"
                else:
                    # No denominator - interpret based on common scales
                    if score >= 9 and score <= 10:  # 0-10 scale
                        return "Correct"
                    elif score >= 5 and score < 9:  # 0-10 scale
                        return "Partially correct"
                    elif score < 5 and score >= 0:  # 0-10 scale
                        return "Incorrect"
                    elif score >= 90:  # 0-100 scale
                        return "Correct"
                    elif score >= 50 and score < 90:  # 0-100 scale
                        return "Partially correct"
                    elif score < 50 and score >= 0:  # 0-100 scale
                        return "Incorrect"
            except (ValueError, TypeError):
                pass
        
        return prediction
