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
    """Clean up common JSON formatting issues.
    
    Args:
        json_str: Raw JSON string that may have formatting issues
        
    Returns:
        Cleaned JSON string
    """
    # Remove trailing commas before closing braces/brackets
    cleaned = re.sub(r',(\s*[}\]])', r'\1', json_str)
    # Fix single quotes used as JSON delimiters (keys)
    cleaned = re.sub(r"'([^']*?)':", r'"\1":', cleaned)
    # Fix single quotes used as JSON delimiters (values)
    cleaned = re.sub(r":\s*'([^']*?)'", r': "\1"', cleaned)
    # Handle escaped quotes and newlines
    cleaned = cleaned.replace('\\"', '"').replace('\\n', '\n')
    return cleaned


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles markdown code blocks with json and inline JSON objects.
    
    Args:
        text: The text to extract JSON from
        
    Returns:
        List of parsed JSON objects, or None if no valid JSON found
    """
    results = []
    search_from = 0
    
    # First, try to find <json>...</json> blocks
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # Try to clean up common JSON issues
            try:
                cleaned = _clean_json_string(inner)
                results.append(json.loads(cleaned))
            except json.JSONDecodeError:
                continue
    
    # Also try to find markdown json blocks ```json...```
    if not results:
        search_from = 0
        while True:
            start = text.find("```json", search_from)
            if start == -1:
                # Try without language specifier
                start = text.find("```", search_from)
                if start == -1:
                    break
                end = text.find("```", start + 3)
                if end == -1:
                    break
                inner = text[start + 3:end].strip()
                search_from = end + 3
            else:
                end = text.find("```", start + 7)
                if end == -1:
                    break
                inner = text[start + 7:end].strip()
                search_from = end + 3
            
            # Try to parse the content
            try:
                results.append(json.loads(inner))
                break  # Found valid JSON, stop searching
            except json.JSONDecodeError:
                # Try to clean up common JSON issues
                try:
                    cleaned = _clean_json_string(inner)
                    results.append(json.loads(cleaned))
                    break  # Found valid JSON, stop searching
                except json.JSONDecodeError:
                    # Continue searching for next code block
                    continue
    
    # Try to find inline JSON objects as a last resort
    if not results:
        try:
            # Look for JSON-like structures with curly braces
            brace_start = text.find('{')
            while brace_start != -1:
                # Try to find matching closing brace
                brace_count = 0
                json_str = None
                for i, char in enumerate(text[brace_start:]):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_str = text[brace_start:brace_start + i + 1]
                            break
                
                if json_str:
                    try:
                        parsed = json.loads(json_str)
                        if isinstance(parsed, dict):
                            results.append(parsed)
                            break  # Found valid JSON, stop searching
                    except json.JSONDecodeError:
                        # Try to clean up common JSON issues
                        try:
                            cleaned = _clean_json_string(json_str)
                            parsed = json.loads(cleaned)
                            if isinstance(parsed, dict):
                                results.append(parsed)
                                break  # Found valid JSON, stop searching
                        except json.JSONDecodeError:
                            pass
                
                brace_start = text.find('{', brace_start + 1)
        except Exception:
            pass
    
    return results or None


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
        prediction = "None"
        try:
            if msg_history and len(msg_history) > 0:
                # Handle both "text" (paper format) and "content" (OpenAI format) fields
                last_message = msg_history[-1].get("text") or msg_history[-1].get("content", "")
                extracted = _extract_jsons(last_message)
                if extracted:
                    last_json = extracted[-1]
                    # Try assessment field first (contains categorical label like "Correct", "Partially correct", "Incorrect")
                    if "assessment" in last_json:
                        prediction = last_json["assessment"]
                    # Fallback: try response field if assessment is not available
                    elif "response" in last_json:
                        prediction = last_json["response"]
                    # Fallback: try reasoning field if neither is available
                    elif "reasoning" in last_json:
                        prediction = last_json["reasoning"]
                    # Fallback: try any string field
                    else:
                        for key, value in last_json.items():
                            if isinstance(value, str) and value.strip():
                                prediction = value
                                break
                else:
                    # If no JSON found, try to extract the last non-empty line as a fallback
                    lines = last_message.strip().split('\n')
                    for line in reversed(lines):
                        stripped = line.strip()
                        if stripped and not stripped.startswith('<') and not stripped.startswith('{') and not stripped.startswith('`'):
                            prediction = stripped
                            break
                    # If still no prediction, try to find any text after common markers
                    if prediction == "None":
                        markers = ["Assessment:", "Response:", "Answer:", "Grade:", "Score:", "Result:", 
                                     "Final Answer:", "Conclusion:", "Verdict:", "Decision:"]
                        for marker in markers:
                            if marker in last_message:
                                parts = last_message.split(marker, 1)
                                if len(parts) > 1:
                                    candidate = parts[1].strip().split('\n')[0].strip()
                                    if candidate and not candidate.startswith('{'):
                                        prediction = candidate
                                        break
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Normalize prediction to expected format
        prediction = str(prediction).strip()
        
        # Normalize common variations of assessment values
        prediction_lower = prediction.lower()
        
        # Define assessment patterns with priority (higher = more specific)
        # Format: (priority, pattern, normalized_value)
        assessment_patterns = [
            # Highest priority: exact phrases for "Partially correct" (must check before single words)
            (4, "partially correct", "Partially correct"),
            (4, "partial credit", "Partially correct"),
            (4, "partially right", "Partially correct"),
            (4, "half correct", "Partially correct"),
            (4, "mostly correct", "Partially correct"),
            (4, "incomplete", "Partially correct"),  # Check before "complete"
            # High priority: exact phrases for "Incorrect"
            (3, "incorrect", "Incorrect"),
            (3, "unsatisfactory", "Incorrect"),
            # High priority: exact phrases for "Correct"
            (3, "correct", "Correct"),
            (3, "full marks", "Correct"),
            (3, "full credit", "Correct"),
            # Medium priority: single words that strongly indicate assessment
            (2, "right", "Correct"),
            (2, "true", "Correct"),
            (2, "yes", "Correct"),
            (2, "pass", "Correct"),
            (2, "accepted", "Correct"),
            (2, "perfect", "Correct"),
            (2, "excellent", "Correct"),
            (2, "valid", "Correct"),
            (2, "accurate", "Correct"),
            (2, "complete", "Correct"),
            (2, "wrong", "Incorrect"),
            (2, "false", "Incorrect"),
            (2, "no", "Incorrect"),
            (2, "fail", "Incorrect"),
            (2, "rejected", "Incorrect"),
            (2, "invalid", "Incorrect"),
            (2, "error", "Incorrect"),
            (2, "mistake", "Incorrect"),
            (2, "inaccurate", "Incorrect"),
            (2, "flawed", "Incorrect"),
            (2, "erroneous", "Incorrect"),
            (2, "partial", "Partially correct"),
            # Low priority: numeric scores
            (1, "0", "Incorrect"),
            (1, "zero", "Incorrect"),
        ]
        
        # Sort by priority (descending) so more specific patterns match first
        assessment_patterns.sort(key=lambda x: -x[0])
        
        # Check for pattern matches
        for priority, pattern, normalized_value in assessment_patterns:
            if pattern in prediction_lower:
                prediction = normalized_value
                break
        
        return prediction, msg_history
