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
    if not json_str:
        return json_str
        
    cleaned = json_str.strip()
    
    # Remove markdown code block markers if present
    if cleaned.startswith('```'):
        # Find the first newline after ```
        first_nl = cleaned.find('\n')
        if first_nl != -1:
            cleaned = cleaned[first_nl+1:]
        else:
            cleaned = cleaned[3:]
    if cleaned.endswith('```'):
        cleaned = cleaned[:-3].strip()
    
    # Remove <json> tags if present
    cleaned = cleaned.replace('<json>', '').replace('</json>', '')
    
    # Remove trailing commas before closing braces/brackets
    cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
    
    # Fix single quotes used as JSON delimiters (but not within values)
    # Handle key names with single quotes
    cleaned = re.sub(r"'([^']*?)':\s*", r'"\1": ', cleaned)
    # Handle string values with single quotes (simple cases)
    cleaned = re.sub(r":\s*'([^']*?)'([,}\]])", r': "\1"\2', cleaned)
    
    # Remove comments (both // and /* */)
    cleaned = re.sub(r'//[^\n]*', '', cleaned)
    cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
    
    # Remove control characters
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', cleaned)
    
    # Fix common escape sequence issues - but be careful not to break valid escapes
    # Only fix double-escaped quotes that are clearly errors
    cleaned = re.sub(r'\\"', '"', cleaned)
    
    # Handle newlines and tabs in strings properly
    cleaned = cleaned.replace('\\n', '\n').replace('\\t', '\t')
    
    # Remove any leading/trailing whitespace again after all processing
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
        json_block_start = text.find("```json")
        if json_block_start != -1:
            content_start = json_block_start + 7
            end = text.find("```", content_start)
            if end != -1:
                inner = text[content_start:end].strip()
                parsed = _try_parse_json(inner)
                if parsed and isinstance(parsed, dict):
                    results.append(parsed)
        
        # If no results, try generic code blocks
        if not results:
            start = text.find("```")
            if start != -1:
                content_start = start + 3
                # Skip language identifier if present
                first_newline = text.find('\n', content_start)
                if first_newline != -1 and first_newline < text.find('```', content_start):
                    content_start = first_newline + 1
                end = text.find("```", content_start)
                if end != -1:
                    inner = text[content_start:end].strip()
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
                if char == '"' and not escape_next:
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
                                # Only accept if it has expected fields
                                if any(key in parsed for key in ['assessment', 'response', 'reasoning', 'answer', 'result', 'evaluation', 'grade', 'score', 'decision']):
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
- Your response MUST be valid JSON.
- The "assessment" field MUST be exactly one of: 'Correct', 'Partially correct', or 'Incorrect' (case-sensitive).
- The "response" field should contain the final answer that will be used for evaluation.
- Do not include any text outside the JSON block."""

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
        
        Priority order: assessment > evaluation > grade > score > response > reasoning > decision > first string value
        
        Args:
            json_data: Parsed JSON dict
            
        Returns:
            Extracted prediction string
        """
        # Priority fields in order of preference (most specific to least specific)
        priority_fields = [
            "assessment",      # Primary field for IMO grading
            "evaluation",    # Alternative grading field
            "grade",         # Grade/score field
            "score",         # Numeric or text score
            "verdict",       # Decision field
            "decision",      # Binary decision
            "response",      # General response
            "result",        # Result field
            "reasoning",     # Explanation (last resort)
        ]
        
        for field in priority_fields:
            if field in json_data:
                value = json_data[field]
                # Handle both string and non-string values
                if isinstance(value, str):
                    value = value.strip()
                    if value:
                        return value
                elif isinstance(value, (int, float, bool)):
                    return str(value)
                elif isinstance(value, list) and value:
                    # If it's a list, try to get the first meaningful string element
                    for item in value:
                        if isinstance(item, str) and item.strip():
                            return item.strip()
        
        # Fallback: try any string field
        for key, value in json_data.items():
            if isinstance(value, str) and value.strip():
                return value.strip()
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
        
        # Priority 1: Look for explicit assessment keywords with context
        # Check for "partially correct" patterns first to avoid misclassification
        partial_patterns = [
            r'\bpartially\s+correct\b',
            r'\bpartial\s+credit\b',
            r'\bpartially\s+right\b',
            r'\bmostly\s+correct\b',
            r'\bincomplete\s+(?:solution|answer|work)\b',
            r'\bpartial\s+(?:solution|answer|work)\b',
            r'\bsome\s+credit\b',
            r'\bpartial\s+marks\b',
            r'\bhalf\s+(?:correct|right|marks|credit)\b',
        ]
        for pattern in partial_patterns:
            if re.search(pattern, text_lower):
                return "Partially correct"
        
        # Check for incorrect patterns
        incorrect_patterns = [
            r'\bincorrect\b',
            r'\bwrong\b',
            r'\bfalse\b',
            r'\bfail\w*\b',
            r'\brejected\b',
            r'\binvalid\b',
            r'\binaccurate\b',
            r'\berror\b',
            r'\bzero\s+(?:marks|points|score)\b',
            r'\b0\s+(?:marks|points|score)\b',
            r'\bno\s+credit\b',
            r'\bnot\s+correct\b',
            r'\bnot\s+valid\b',
        ]
        for pattern in incorrect_patterns:
            if re.search(pattern, text_lower):
                return "Incorrect"
        
        # Check for correct patterns
        correct_patterns = [
            r'\bcorrect\b',
            r'\bright\b',
            r'\btrue\b',
            r'\bpass\w*\b',
            r'\baccepted\b',
            r'\bvalid\b',
            r'\baccurate\b',
            r'\bfull\s+(?:marks|credit|points|score)\b',
            r'\bcomplete\s+(?:solution|answer)\b',
            r'\bwell\s+done\b',
            r'\bexcellent\b',
        ]
        for pattern in correct_patterns:
            if re.search(pattern, text_lower):
                return "Correct"
        
        # Priority 2: Try to find text after common markers (case-insensitive)
        markers = [
            "Assessment:", "Evaluation:", "Grade:", "Score:", "Verdict:",
            "Decision:", "Result:", "Response:", "Answer:",
            "Final Answer:", "Conclusion:", "Judgment:", "Rating:",
            "Status:", "Outcome:", "Determination:", "Summary:",
            "Final Assessment:", "Grading Decision:", "Final Grade:",
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
                                # Normalize the extracted candidate
                                normalized = self._normalize_prediction(candidate)
                                if normalized in ["Correct", "Incorrect", "Partially correct"]:
                                    return normalized
                                # If not a standard assessment but looks like a score/grade
                                if any(c.isdigit() for c in candidate):
                                    return candidate
        
        # Priority 3: Look for quoted assessments
        quoted_pattern = r'["\'](Correct|Incorrect|Partially correct)["\']'
        match = re.search(quoted_pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).capitalize()
        
        # Priority 4: Look for standalone grades/scores (e.g., "0/7", "7 points", "Full marks")
        score_patterns = [
            r'\b(\d+/\d+)\b',  # Pattern like "3/7" or "0/7"
            r'\b(\d+\s+points?)\b',
            r'\b(\d+\s+marks?)\b',
            r'\b(score[:\s]+\d+)\b',
        ]
        for pattern in score_patterns:
            match = re.search(pattern, text_lower)
            if match:
                return match.group(1)
        
        # Priority 5: Extract the last substantial non-empty line
        lines = text.strip().split('\n')
        for line in reversed(lines):
            stripped = line.strip()
            # Skip lines that are too short, start with special chars, or are just punctuation
            if (stripped and 
                len(stripped) > 2 and 
                not stripped.startswith(('<', '{', '[', '`', '-', '*', '#', '|')) and
                not stripped.endswith(':') and
                not stripped.lower().startswith(('note:', 'important:', 'warning:', 'hint:'))):
                # Remove trailing punctuation
                cleaned = stripped.rstrip('.;,!')
                if cleaned:
                    # Normalize the extracted text
                    normalized = self._normalize_prediction(cleaned)
                    if normalized in ["Correct", "Incorrect", "Partially correct"]:
                        return normalized
                    # If not a standard assessment, return the cleaned text
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
            "mostly correct", "incomplete", "partial", "some credit",
            "partial marks", "half correct", "half right", "partial solution",
            "incomplete solution", "partial answer", "incomplete answer",
        ]
        for var in partial_variations:
            if contains_word(prediction_lower, var):
                return "Partially correct"
        
        # Check "incorrect" before "correct" to avoid substring match issues
        incorrect_variations = [
            "incorrect", "wrong", "false", "no", "fail", "rejected",
            "zero", "0", "invalid", "inaccurate", "error", "not correct",
            "not valid", "no credit", "not accepted", "unsatisfactory",
        ]
        for var in incorrect_variations:
            if contains_word(prediction_lower, var):
                return "Incorrect"
        
        correct_variations = [
            "correct", "right", "true", "yes", "pass", "accepted",
            "full marks", "full credit", "valid", "accurate", "complete",
            "perfect", "excellent", "well done", "satisfactory", "good",
        ]
        for var in correct_variations:
            if contains_word(prediction_lower, var):
                return "Correct"
        
        # Handle numeric scores (e.g., "7/7" or "0/7")
        score_match = re.search(r'(\d+)\s*/\s*(\d+)', prediction)
        if score_match:
            earned = int(score_match.group(1))
            total = int(score_match.group(2))
            if total > 0:
                ratio = earned / total
                if ratio == 1.0:
                    return "Correct"
                elif ratio == 0.0:
                    return "Incorrect"
                else:
                    return "Partially correct"
        
        # Handle numeric values (0 = incorrect, positive = partially correct)
        if prediction.isdigit():
            num = int(prediction)
            if num == 0:
                return "Incorrect"
            else:
                return "Partially correct"
        
        return prediction
