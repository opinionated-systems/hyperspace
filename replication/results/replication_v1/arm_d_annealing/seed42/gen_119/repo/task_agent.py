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
    Also handles markdown code blocks with json and inline JSON objects.
    """
    results = []
    
    # Helper function to try parsing JSON with cleanup
    def try_parse_json(json_str: str) -> dict | None:
        """Try to parse JSON string, with cleanup on failure."""
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            try:
                cleaned = _clean_json_string(json_str)
                return json.loads(cleaned)
            except json.JSONDecodeError:
                return None
    
    # First, try to find <json>...</json> blocks
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
        parsed = try_parse_json(inner)
        if parsed:
            results.append(parsed)
    
    # Also try to find markdown json blocks ```json...```
    if not results:
        for pattern in ["```json", "```"]:
            start = text.find(pattern)
            if start != -1:
                # Skip past the opening pattern and any following whitespace/newlines
                offset = start + len(pattern)
                # Skip leading whitespace/newlines after the opening ```
                while offset < len(text) and text[offset] in ' \t\n\r':
                    offset += 1
                end = text.find("```", offset)
                if end != -1:
                    inner = text[offset:end].strip()
                    parsed = try_parse_json(inner)
                    if parsed:
                        results.append(parsed)
                        break
    
    # Try to find inline JSON objects as a last resort
    if not results:
        brace_start = text.find('{')
        while brace_start != -1:
            # Try to find matching closing brace
            brace_count = 0
            for i, char in enumerate(text[brace_start:]):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_str = text[brace_start:brace_start + i + 1]
                        parsed = try_parse_json(json_str)
                        if parsed and isinstance(parsed, dict):
                            results.append(parsed)
                        break
            brace_start = text.find('{', brace_start + 1)
    
    # Try to find JSON objects with common field patterns
    if not results:
        # Look for patterns like {"assessment": ...} or {"response": ...}
        pattern = r'\{\s*"(assessment|response|reasoning|answer|result|grade|score)"\s*:\s*"[^"]*"[^}]*\}'
        for match in re.finditer(pattern, text, re.IGNORECASE):
            json_str = match.group(0)
            parsed = try_parse_json(json_str)
            if parsed and isinstance(parsed, dict):
                results.append(parsed)
    
    # Final fallback: try to extract any valid JSON object with relevant keys
    if not results:
        relevant_keys = {'assessment', 'response', 'reasoning', 'answer', 'result', 'grade', 'score'}
        for start in re.finditer(r'\{', text):
            start_pos = start.start()
            # Try to find a valid JSON object starting at this position
            for end_pos in range(start_pos + 2, min(start_pos + 2000, len(text) + 1)):
                candidate = text[start_pos:end_pos]
                if candidate.count('{') == candidate.count('}'):
                    parsed = try_parse_json(candidate)
                    if parsed and isinstance(parsed, dict) and relevant_keys.intersection(parsed.keys()):
                        results.append(parsed)
                        break
    
    return results or None


def _clean_json_string(json_str: str) -> str:
    """Clean up common JSON formatting issues.
    
    Args:
        json_str: Raw JSON string that may have formatting issues
        
    Returns:
        Cleaned JSON string
    """
    cleaned = json_str
    
    # Remove trailing commas before closing braces/brackets
    cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
    
    # Fix single quotes used as JSON delimiters (but not within strings)
    cleaned = re.sub(r"'([^']*?)':", r'"\1":', cleaned)
    cleaned = re.sub(r":\s*'([^']*?)'", r': "\1"', cleaned)
    
    # Remove comments
    cleaned = re.sub(r'//.*?\n', '\n', cleaned)
    cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
    
    # Remove control characters (except whitespace)
    cleaned = ''.join(char for char in cleaned if ord(char) >= 32 or char in '\n\r\t')
    
    # Fix unescaped newlines in string values (common LLM issue)
    # Use a safer approach: find strings and escape newlines within them
    def escape_in_strings(match: re.Match) -> str:
        content = match.group(1)
        # Escape newlines and tabs within the string content
        content = content.replace('\n', '\\n').replace('\t', '\\t').replace('"', '\\"')
        return f'"{content}"'
    
    # Match quoted strings and process them
    cleaned = re.sub(r'"([^"\\]*(?:\\.[^"\\]*)*)"', escape_in_strings, cleaned)
    
    return cleaned


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
    "assessment": "Exactly one of: 'Correct', 'Partially correct', or 'Incorrect'",
    "response": "Exactly one of: 'Correct', 'Partially correct', or 'Incorrect' - must match the assessment field"
}}
</json>

Important: 
- Ensure your response is valid JSON without trailing commas.
- Both "assessment" and "response" fields must contain EXACTLY one of these three values: 'Correct', 'Partially correct', or 'Incorrect'.
- Do not add any other text or explanation in these fields - only the exact label.
- The evaluation system extracts these fields to determine the grade, so they must be exact."""

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
        prediction = _extract_prediction(msg_history)
        
        return prediction, msg_history


def _extract_prediction(msg_history: list[dict]) -> str:
    """Extract and normalize prediction from message history.
    
    Args:
        msg_history: List of message dictionaries from LLM conversation
        
    Returns:
        Normalized prediction string ("Correct", "Partially correct", or "Incorrect")
    """
    if not msg_history:
        return _normalize_prediction(None)
    
    try:
        # Handle both "text" (paper format) and "content" (OpenAI format) fields
        last_message = msg_history[-1].get("text") or msg_history[-1].get("content", "")
        if not last_message:
            return _normalize_prediction(None)
        
        # Try to extract JSON first
        extracted = _extract_jsons(last_message)
        if extracted:
            last_json = extracted[-1]
            # Priority order for extraction: assessment > response > reasoning > any string field
            for key in ["assessment", "response", "reasoning"]:
                if key in last_json and isinstance(last_json[key], str):
                    return _normalize_prediction(last_json[key])
            # Fallback: try any string field
            for key, value in last_json.items():
                if isinstance(value, str) and value.strip():
                    return _normalize_prediction(value)
        
        # If no JSON found, try to extract the last non-empty line as a fallback
        lines = last_message.strip().split('\n')
        for line in reversed(lines):
            stripped = line.strip()
            if stripped and not stripped.startswith(('<', '{', '`')):
                return _normalize_prediction(stripped)
        
        # If still no prediction, try to find any text after common markers
        markers = [
            "Assessment:", "Response:", "Answer:", "Grade:", "Score:", "Result:", 
            "Final Answer:", "Conclusion:", "Verdict:", "Decision:",
            "Evaluation:", "Judgment:", "Rating:", "Status:"
        ]
        for marker in markers:
            if marker in last_message:
                parts = last_message.split(marker, 1)
                if len(parts) > 1:
                    candidate = parts[1].strip().split('\n')[0].strip()
                    if candidate and not candidate.startswith('{'):
                        return _normalize_prediction(candidate)
    except Exception as e:
        logger.warning(f"Error extracting prediction: {e}")
    
    return _normalize_prediction(None)


def _normalize_prediction(prediction: str | None) -> str:
    """Normalize prediction to one of the expected assessment values.
    
    Args:
        prediction: Raw prediction string or None
        
    Returns:
        Normalized prediction: "Correct", "Partially correct", or "Incorrect"
    """
    # Handle None or empty prediction
    if prediction is None:
        return "Incorrect"
    
    prediction = str(prediction).strip()
    if not prediction or prediction.lower() == "none":
        return "Incorrect"
    
    prediction_lower = prediction.lower()
    
    # Exact matches (case-insensitive)
    exact_matches = {
        "correct": "Correct",
        "partially correct": "Partially correct",
        "partially": "Partially correct",
        "partial": "Partially correct",
        "incorrect": "Incorrect",
    }
    if prediction_lower in exact_matches:
        return exact_matches[prediction_lower]
    
    # Check for "partially correct" first (before "correct" and "incorrect")
    # to avoid misclassifying it as either
    if "partially correct" in prediction_lower:
        return "Partially correct"
    
    # Check for "incorrect" variants BEFORE "correct" variants
    # to avoid misclassifying "not correct" or "incorrect" as "correct"
    incorrect_variants = [
        "incorrect", "wrong", "false", "no", "fail", "rejected", 
        "zero", "0", "invalid", "erroneous", "unsound", "flawed",
        "incorrect solution", "wrong answer", "not correct", "not right",
        "not valid", "not accurate", "not proper", "not sound"
    ]
    for variant in incorrect_variants:
        if variant in prediction_lower:
            return "Incorrect"
    
    # Check for "partial" variants (but not "partially correct" which we already checked)
    partial_variants = [
        "partial credit", "partially right", "incomplete", "mostly correct", 
        "some correct", "partial solution", "partially valid", "partial success", 
        "half correct", "partial marks", "partially accurate", "partially sound"
    ]
    for variant in partial_variants:
        if variant in prediction_lower:
            return "Partially correct"
    
    # Check for "correct" variants (but not "partially correct" or "incorrect")
    correct_variants = [
        "correct", "right", "true", "yes", "pass", "accepted", 
        "full marks", "full credit", "valid", "accurate", "proper",
        "sound", "complete solution", "correct solution", "right answer",
        "true solution", "valid solution", "accurate solution"
    ]
    for variant in correct_variants:
        if variant in prediction_lower:
            return "Correct"
    
    # Default to Incorrect for unrecognized predictions
    logger.debug(f"Unrecognized prediction '{prediction}', defaulting to Incorrect")
    return "Incorrect"
