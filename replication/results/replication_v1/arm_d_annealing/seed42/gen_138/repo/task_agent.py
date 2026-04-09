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
        for pattern in ["```json", "```JSON", "```"]:
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
            # Skip if this brace is inside a string (simple check)
            # Count quotes before this position to see if we're in a string
            quote_count = text[:brace_start].count('"') - text[:brace_start].count('\\"')
            if quote_count % 2 == 1:
                # We're inside a string, skip this brace
                brace_start = text.find('{', brace_start + 1)
                continue
            
            # Find matching closing brace using counter
            brace_count = 0
            in_string = False
            escape_next = False
            found_valid_json = False
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
                                # Check if this looks like our expected format
                                if any(key in parsed for key in ["assessment", "response", "reasoning"]):
                                    results.append(parsed)
                                    found_valid_json = True
                            break
            if found_valid_json:
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
    "assessment": "Correct",
    "response": "The final grading decision or score as specified in the guidelines"
}}
</json>

Important Instructions:
1. The "assessment" field MUST be exactly one of these three values: 'Correct', 'Partially correct', or 'Incorrect' (case-sensitive, no extra words).
2. The "response" field should contain the final answer that will be used for evaluation.
3. Ensure your response is valid JSON without trailing commas.
4. Do not include any text outside the <json>...</json> tags.
5. Be objective and consistent with the grading guidelines provided."""

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
        priority_fields = ["assessment", "response", "reasoning", "evaluation", "grade", "result"]
        
        for field in priority_fields:
            if field in json_data:
                value = json_data[field]
                # Handle string values
                if isinstance(value, str) and value.strip():
                    return value.strip()
                # Handle nested dict (e.g., {"assessment": {"value": "Correct"}})
                elif isinstance(value, dict):
                    # Try common nested keys
                    for nested_key in ["value", "result", "assessment", "grade", "score"]:
                        if nested_key in value and isinstance(value[nested_key], str):
                            return value[nested_key].strip()
                    # If no nested key found, try to get first string value
                    for v in value.values():
                        if isinstance(v, str) and v.strip():
                            return v.strip()
                # Handle list (take first string element)
                elif isinstance(value, list) and value:
                    for item in value:
                        if isinstance(item, str) and item.strip():
                            return item.strip()
        
        # Fallback: try any string field
        for key, value in json_data.items():
            if isinstance(value, str) and value.strip():
                return value.strip()
        
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
            r'\bincomplete\s+but\s+correct\b',
            r'\bpartial\s+(?:solution|answer|credit)\b',
        ]
        for pattern in partial_patterns:
            if re.search(pattern, text_lower):
                return "Partially correct"
        
        # Check for incorrect/ wrong patterns
        incorrect_patterns = [
            r'\bincorrect\b',
            r'\bwrong\b',
            r'\bfalse\b',
            r'\bfail\b',
            r'\brejected\b',
            r'\binvalid\b',
            r'\berror\b',
            r'\bnot\s+correct\b',
            r'\bnot\s+valid\b',
            r'\bzero\s+(?:marks|points|score)\b',
            r'\b0\s+(?:marks|points|score)\b',
        ]
        for pattern in incorrect_patterns:
            if re.search(pattern, text_lower):
                return "Incorrect"
        
        # Check for correct patterns
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
            "Final Assessment:", "Summary:", "Decision:", "Grade Assessment:"
        ]
        
        for marker in markers:
            marker_lower = marker.lower()
            if marker_lower in text_lower:
                # Find the position in original case
                idx = text_lower.find(marker_lower)
                if idx != -1:
                    # Get the text after the marker (preserve original case)
                    after_marker = text[idx + len(marker):].strip()
                    # Get first non-empty line that has substance
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
                                # Normalize and return
                                normalized = self._normalize_prediction(candidate)
                                if normalized != candidate:
                                    return normalized
                                return candidate
        
        # Priority 3: Look for assessment in quotes or emphasized text
        quote_patterns = [
            r'["\'](Correct|Incorrect|Partially correct)["\']',
            r'\*\*(Correct|Incorrect|Partially correct)\*\*',
            r'\*(Correct|Incorrect|Partially correct)\*',
            r'`(Correct|Incorrect|Partially correct)`',
        ]
        for pattern in quote_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return self._normalize_prediction(match.group(1))
        
        # Priority 4: Extract from the last substantial paragraph
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if paragraphs:
            last_para = paragraphs[-1]
            # Check if last paragraph contains an assessment
            for pattern, assessment in [
                (r'\bpartially\s+correct\b', "Partially correct"),
                (r'\bincorrect\b', "Incorrect"),
                (r'\bcorrect\b', "Correct"),
            ]:
                if re.search(pattern, last_para.lower()):
                    return assessment
            
            # Try to extract the last sentence
            sentences = re.split(r'[.!?]+', last_para)
            for sentence in reversed(sentences):
                stripped = sentence.strip()
                if (stripped and 
                    len(stripped) > 3 and 
                    not stripped.startswith(('<', '{', '[', '`', '-', '*', '#'))):
                    normalized = self._normalize_prediction(stripped)
                    if normalized in ["Correct", "Incorrect", "Partially correct"]:
                        return normalized
        
        # Fallback: extract the last substantial non-empty line
        lines = text.strip().split('\n')
        for line in reversed(lines):
            stripped = line.strip()
            # Skip lines that are too short, start with special chars, or are just punctuation
            if (stripped and 
                len(stripped) > 2 and 
                not stripped.startswith(('<', '{', '[', '`', '-', '*', '#', '|', '/')) and
                not stripped.endswith(':') and
                not stripped.startswith('//')):
                # Remove trailing punctuation
                cleaned = stripped.rstrip('.;,!')
                if cleaned:
                    normalized = self._normalize_prediction(cleaned)
                    return normalized
        
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
            "mostly correct", "incomplete but correct", "partial solution",
            "partial answer", "partial"
        ]
        for var in partial_variations:
            if contains_word(prediction_lower, var):
                return "Partially correct"
        
        # Check "incorrect" before "correct" to avoid substring match issues
        # Also check for negations that indicate incorrect
        incorrect_variations = [
            "incorrect", "wrong", "false", "no", "fail", "rejected",
            "zero", "0", "invalid", "inaccurate", "error", "not correct",
            "not valid", "not accurate", "not accepted", "does not pass",
            "fails", "failed"
        ]
        for var in incorrect_variations:
            if contains_word(prediction_lower, var):
                return "Incorrect"
        
        correct_variations = [
            "correct", "right", "true", "yes", "pass", "accepted",
            "full marks", "full credit", "valid", "accurate", "complete",
            "perfect", "excellent", "good"
        ]
        for var in correct_variations:
            if contains_word(prediction_lower, var):
                return "Correct"
        
        # Check for numeric scores that might indicate the assessment
        # Look for patterns like "Score: 0", "0/7", "0 points", etc.
        zero_score_patterns = [
            r'\b0\s*/\s*\d+',
            r'\b0\s+points?\b',
            r'\b0\s+marks?\b',
            r'\bscore\s*[:=]\s*0\b',
            r'\bgrade\s*[:=]\s*0\b',
        ]
        for pattern in zero_score_patterns:
            if re.search(pattern, prediction_lower):
                return "Incorrect"
        
        # Look for full/maximum score patterns
        full_score_patterns = [
            r'\b7\s*/\s*7\b',
            r'\b\d+\s*/\s*\1\b',  # Same number on both sides like 5/5
            r'\bfull\s+(?:marks?|points?|score|credit)\b',
            r'\bmaximum\s+(?:marks?|points?|score)\b',
        ]
        for pattern in full_score_patterns:
            if re.search(pattern, prediction_lower):
                return "Correct"
        
        return prediction
