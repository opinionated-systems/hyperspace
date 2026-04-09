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
                # Remove trailing commas before closing braces/brackets
                cleaned = re.sub(r',(\s*[}\]])', r'\1', inner)
                # Fix single quotes used as JSON delimiters
                cleaned = re.sub(r"'([^']*?)':", r'"\1":', cleaned)
                cleaned = re.sub(r":\s*'([^']*?)'", r': "\1"', cleaned)
                results.append(json.loads(cleaned))
            except json.JSONDecodeError:
                continue
    
    # Also try to find markdown json blocks ```json...```
    if not results:
        search_from = 0
        while True:
            start = text.find("```json", search_from)
            offset = 7  # Length of "```json"
            if start == -1:
                # Try without language specifier
                start = text.find("```", search_from)
                if start == -1:
                    break
                offset = 3  # Length of "```"
            end = text.find("```", start + offset)
            if end == -1:
                break
            inner = text[start + offset:end].strip()
            search_from = end + 3
            try:
                results.append(json.loads(inner))
                break  # Found valid JSON, stop searching
            except json.JSONDecodeError:
                # Try to clean up common JSON issues
                try:
                    cleaned = re.sub(r',(\s*[}\]])', r'\1', inner)
                    cleaned = re.sub(r"'([^']*?)':", r'"\1":', cleaned)
                    cleaned = re.sub(r":\s*'([^']*?)'", r': "\1"', cleaned)
                    results.append(json.loads(cleaned))
                    break  # Found valid JSON after cleaning, stop searching
                except json.JSONDecodeError:
                    # Continue to next code block if this one wasn't valid JSON
                    continue
    
    # Try to find inline JSON objects as a last resort
    if not results:
        try:
            # Look for JSON-like structures with curly braces
            brace_start = text.find('{')
            while brace_start != -1:
                # Try to find matching closing brace
                brace_count = 0
                found_json = False
                for i, char in enumerate(text[brace_start:]):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_str = text[brace_start:brace_start + i + 1]
                            try:
                                parsed = json.loads(json_str)
                                if isinstance(parsed, dict):
                                    results.append(parsed)
                                    found_json = True
                                    break  # Found valid JSON, stop searching
                            except json.JSONDecodeError:
                                # Try to clean up common JSON issues
                                try:
                                    cleaned = re.sub(r',(\s*[}\]])', r'\1', json_str)
                                    cleaned = re.sub(r"'([^']*?)':", r'"\1":', cleaned)
                                    cleaned = re.sub(r":\s*'([^']*?)'", r': "\1"', cleaned)
                                    # Also handle escaped quotes and newlines
                                    cleaned = cleaned.replace('\\"', '"').replace('\\n', '\n')
                                    parsed = json.loads(cleaned)
                                    if isinstance(parsed, dict):
                                        results.append(parsed)
                                        found_json = True
                                        break  # Found valid JSON, stop searching
                                except json.JSONDecodeError:
                                    pass
                            break
                if found_json:
                    break
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
    "assessment": "One of: 'correct', 'almost', or 'incorrect' (lowercase, matching the expected label format)",
    "response": "The final grading decision or score as specified in the guidelines"
}}
</json>

Important: 
- Ensure your response is valid JSON without trailing commas.
- The "assessment" field MUST be exactly one of these three lowercase values: 'correct', 'almost', or 'incorrect'.
- Use 'correct' for fully correct solutions, 'almost' for partially correct solutions (minor errors, incomplete but mostly right), and 'incorrect' for wrong solutions.
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
        
        # Correct variations -> "correct"
        correct_patterns = [
            "correct", "right", "true", "yes", "pass", "accepted", 
            "full marks", "full credit", "perfect", "excellent", "valid",
            "accurate", "proper", "appropriate", "satisfactory", "complete"
        ]
        # Partially correct / Almost correct variations -> "almost"
        partial_patterns = [
            "partially correct", "partial", "partial credit", "partially right", 
            "incomplete", "mostly correct", "some correct", "partially valid",
            "half correct", "partial success", "partially accurate",
            "almost", "almost correct", "nearly correct", "close", "minor error",
            "small mistake", "mostly right", "near miss"
        ]
        # Incorrect variations -> "incorrect"
        incorrect_patterns = [
            "incorrect", "wrong", "false", "no", "fail", "rejected", 
            "zero", "0", "invalid", "error", "mistake", "unsatisfactory",
            "inaccurate", "improper", "inappropriate", "flawed", "erroneous"
        ]
        
        # Check for exact matches first
        if prediction_lower in correct_patterns:
            prediction = "correct"
        elif prediction_lower in partial_patterns:
            prediction = "almost"
        elif prediction_lower in incorrect_patterns:
            prediction = "incorrect"
        # Check for partial matches (contains) - prioritize longer patterns first
        else:
            # Sort patterns by length (longest first) to avoid partial matching issues
            sorted_correct = sorted(correct_patterns, key=len, reverse=True)
            sorted_partial = sorted(partial_patterns, key=len, reverse=True)
            sorted_incorrect = sorted(incorrect_patterns, key=len, reverse=True)
            
            # Check partial patterns first (they're more specific)
            if any(pattern in prediction_lower for pattern in sorted_partial):
                prediction = "almost"
            elif any(pattern in prediction_lower for pattern in sorted_incorrect):
                prediction = "incorrect"
            elif any(pattern in prediction_lower for pattern in sorted_correct):
                prediction = "correct"
        
        return prediction, msg_history
