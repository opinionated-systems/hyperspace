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


def _clean_json_string(json_str: str) -> str:
    """Clean common JSON formatting issues.
    
    Args:
        json_str: Raw JSON string that may have formatting issues
        
    Returns:
        Cleaned JSON string
    """
    # Remove trailing commas before closing braces/brackets
    cleaned = re.sub(r',(\s*[}\]])', r'\1', json_str)
    # Fix single quotes used as JSON delimiters
    cleaned = re.sub(r"'([^']*?)':", r'"\1":', cleaned)
    cleaned = re.sub(r":\s*'([^']*?)'", r': "\1"', cleaned)
    # Fix unescaped newlines in strings
    cleaned = re.sub(r'(?<!\\)\n', r'\\n', cleaned)
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
        # Skip nested <json> tags
        next_start = text.find("<json>", start + 6)
        if next_start != -1 and next_start < end:
            search_from = start + 6
            continue
        inner = text[start + 6:end].strip()
        search_from = end + 7
        parsed = _try_parse_json(inner)
        if parsed:
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
                    if parsed:
                        results.append(parsed)
                        break
    
    # Strategy 3: Find inline JSON objects
    if not results:
        brace_start = text.find('{')
        search_count = 0
        max_searches = 50
        while brace_start != -1 and search_count < max_searches:
            search_count += 1
            # Find matching closing brace using state machine
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

def _extract_prediction_from_json(extracted: list[dict]) -> str | None:
    """Extract prediction from parsed JSON objects.
    
    Args:
        extracted: List of parsed JSON dicts
        
    Returns:
        Prediction string or None if not found
    """
    if not extracted:
        return None
    
    last_json = extracted[-1]
    
    # Priority order for extraction
    priority_fields = ["assessment", "response", "reasoning"]
    for field in priority_fields:
        if field in last_json and isinstance(last_json[field], str):
            value = last_json[field].strip()
            if value:
                return value
    
    # Fallback: try any string field
    for key, value in last_json.items():
        if isinstance(value, str) and value.strip():
            return value.strip()
    
    return None


def _extract_prediction_from_text(text: str) -> str | None:
    """Extract prediction from plain text when JSON parsing fails.
    
    Args:
        text: Raw text to extract prediction from
        
    Returns:
        Prediction string or None if not found
    """
    # Try to extract the last non-empty line
    lines = text.strip().split('\n')
    for line in reversed(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith(('<', '{', '`')):
            return stripped
    
    # Try to find text after common markers
    markers = [
        "Assessment:", "Response:", "Answer:", "Grade:", "Score:", "Result:",
        "Final Answer:", "Conclusion:", "Verdict:", "Decision:"
    ]
    for marker in markers:
        if marker in text:
            parts = text.split(marker, 1)
            if len(parts) > 1:
                candidate = parts[1].strip().split('\n')[0].strip()
                if candidate and not candidate.startswith('{'):
                    return candidate
    
    return None


def _normalize_prediction(prediction: str) -> str:
    """Normalize prediction to standard assessment values.
    
    Args:
        prediction: Raw prediction string
        
    Returns:
        Normalized prediction: 'Correct', 'Partially correct', or 'Incorrect'
    """
    prediction = str(prediction).strip()
    prediction_lower = prediction.lower()
    
    # Map common variations to standard values
    correct_variants = [
        "correct", "right", "true", "yes", "pass", "accepted",
        "full marks", "full credit", "valid", "accurate"
    ]
    partial_variants = [
        "partially correct", "partial", "partial credit",
        "partially right", "incomplete", "mostly correct"
    ]
    incorrect_variants = [
        "incorrect", "wrong", "false", "no", "fail", "rejected",
        "zero", "0", "invalid", "error"
    ]
    
    if any(variant in prediction_lower for variant in correct_variants):
        return "Correct"
    elif any(variant in prediction_lower for variant in partial_variants):
        return "Partially correct"
    elif any(variant in prediction_lower for variant in incorrect_variants):
        return "Incorrect"
    
    # Return original if no normalization matched
    return prediction


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

        # Extract prediction from response
        prediction = "None"
        try:
            if msg_history:
                # Handle both "text" (paper format) and "content" (OpenAI format) fields
                last_message = msg_history[-1].get("text") or msg_history[-1].get("content", "")
                
                # Try JSON extraction first
                extracted = _extract_jsons(last_message)
                if extracted:
                    prediction = _extract_prediction_from_json(extracted) or "None"
                
                # Fallback to text extraction if JSON failed
                if prediction == "None":
                    prediction = _extract_prediction_from_text(last_message) or "None"
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Normalize prediction to expected format
        prediction = _normalize_prediction(prediction)
        
        return prediction, msg_history
