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
            try:
                results.append(json.loads(inner))
            except json.JSONDecodeError:
                # Try to clean up common JSON issues
                try:
                    cleaned = re.sub(r',(\s*[}\]])', r'\1', inner)
                    cleaned = re.sub(r"'([^']*?)':", r'"\1":', cleaned)
                    cleaned = re.sub(r":\s*'([^']*?)'", r': "\1"', cleaned)
                    results.append(json.loads(cleaned))
                except json.JSONDecodeError:
                    continue
            break  # Only process first valid JSON block
    
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
                            cleaned = re.sub(r',(\s*[}\]])', r'\1', json_str)
                            cleaned = re.sub(r"'([^']*?)':", r'"\1":', cleaned)
                            cleaned = re.sub(r":\s*'([^']*?)'", r': "\1"', cleaned)
                            # Also handle escaped quotes and newlines
                            cleaned = cleaned.replace('\\"', '"').replace('\\n', '\n')
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
    "assessment": "Correct",
    "response": "The final grading decision or score as specified in the guidelines"
}}
</json>

**CRITICAL INSTRUCTIONS FOR THE assessment FIELD:**
You MUST use EXACTLY one of these three values (case-sensitive):
- "Correct" - if the student's answer is fully correct, complete, and matches the official solution
- "Partially correct" - if the student made partial progress but has errors or is incomplete
- "Incorrect" - if the student's answer is wrong, has major errors, or shows no meaningful progress

The assessment field will be used for automatic evaluation, so it must be exactly one of these three strings.

Important: 
- Ensure your response is valid JSON without trailing commas.
- The "assessment" field MUST be exactly: "Correct", "Partially correct", or "Incorrect" (case-sensitive).
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

        # Extract prediction from JSON
        prediction = self._extract_prediction(msg_history)
        
        # Normalize to expected format
        prediction = self._normalize_prediction(prediction)
        
        return prediction, msg_history
    
    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history.
        
        Tries multiple strategies to extract a valid prediction:
        1. Extract from JSON blocks (assessment, response, or reasoning fields)
        2. Extract from text markers (Assessment:, Response:, etc.)
        3. Use the last non-empty line as fallback
        
        Returns:
            Extracted prediction string or "None" if extraction fails
        """
        if not msg_history:
            return "None"
        
        # Get the last message content
        last_message = msg_history[-1].get("text") or msg_history[-1].get("content", "")
        if not last_message:
            return "None"
        
        # Strategy 1: Try to extract from JSON blocks
        extracted = _extract_jsons(last_message)
        if extracted:
            last_json = extracted[-1]
            # Try fields in order of preference
            for field in ["assessment", "response", "reasoning", "answer", "grade", "result"]:
                if field in last_json and isinstance(last_json[field], str):
                    value = last_json[field].strip()
                    if value:
                        return value
            # Try any string field as last resort
            for key, value in last_json.items():
                if isinstance(value, str) and value.strip():
                    return value.strip()
        
        # Strategy 2: Look for text markers
        markers = [
            "Assessment:", "Response:", "Answer:", "Grade:", "Score:", "Result:",
            "Final Answer:", "Conclusion:", "Verdict:", "Decision:", "Evaluation:"
        ]
        for marker in markers:
            if marker in last_message:
                parts = last_message.split(marker, 1)
                if len(parts) > 1:
                    candidate = parts[1].strip().split('\n')[0].strip()
                    # Remove common punctuation and quotes
                    candidate = candidate.strip('"\'`.,;:!')
                    if candidate and not candidate.startswith('{'):
                        return candidate
        
        # Strategy 3: Use last non-empty line
        lines = last_message.strip().split('\n')
        for line in reversed(lines):
            stripped = line.strip()
            # Skip lines that are likely not predictions
            if stripped and not stripped.startswith(('<', '{', '`', '-', '*', '#')):
                # Remove common punctuation and quotes
                stripped = stripped.strip('"\'`.,;:!')
                if stripped:
                    return stripped
        
        return "None"
    
    def _normalize_prediction(self, prediction: str) -> str:
        """Normalize prediction to one of the expected labels.
        
        Maps various forms of correct/incorrect/partial to standardized labels:
        - "Correct" for fully correct answers
        - "Partially correct" for partially correct answers  
        - "Incorrect" for incorrect answers
        
        Args:
            prediction: Raw prediction string
            
        Returns:
            Normalized prediction string
        """
        if not prediction or prediction == "None":
            return "None"
        
        prediction = str(prediction).strip()
        prediction_lower = prediction.lower()
        
        # Direct exact matches (case-insensitive)
        exact_map = {
            "correct": "Correct",
            "right": "Correct",
            "true": "Correct",
            "yes": "Correct",
            "pass": "Correct",
            "accepted": "Correct",
            "valid": "Correct",
            "accurate": "Correct",
            "perfect": "Correct",
            "excellent": "Correct",
            "full marks": "Correct",
            "full credit": "Correct",
            "100%": "Correct",
            "partially correct": "Partially correct",
            "partial": "Partially correct",
            "partial credit": "Partially correct",
            "incomplete": "Partially correct",
            "mostly correct": "Partially correct",
            "some correct": "Partially correct",
            "half correct": "Partially correct",
            "partial success": "Partially correct",
            "partial marks": "Partially correct",
            "half marks": "Partially correct",
            "almost": "Partially correct",
            "incorrect": "Incorrect",
            "wrong": "Incorrect",
            "false": "Incorrect",
            "no": "Incorrect",
            "fail": "Incorrect",
            "rejected": "Incorrect",
            "invalid": "Incorrect",
            "error": "Incorrect",
            "unsatisfactory": "Incorrect",
            "zero": "Incorrect",
            "0": "Incorrect",
            "0%": "Incorrect",
            "failed": "Incorrect",
            "failure": "Incorrect",
        }
        
        if prediction_lower in exact_map:
            return exact_map[prediction_lower]
        
        # Pattern-based matching for compound phrases
        # Check for "not correct", "not right" patterns first (negative)
        if any(phrase in prediction_lower for phrase in ["not correct", "not right", "not valid"]):
            return "Incorrect"
        
        # Check for partially correct patterns
        if any(phrase in prediction_lower for phrase in ["partially", "partial credit", "incomplete", "mostly", "half"]):
            return "Partially correct"
        
        # Check for incorrect patterns (but not "correct" alone)
        if any(phrase in prediction_lower for phrase in ["wrong", "incorrect", "error", "invalid", "fail", "mistake"]):
            return "Incorrect"
        
        # If contains "correct" and no negation/partial indicators
        if "correct" in prediction_lower and "not" not in prediction_lower:
            return "Correct"
        
        # Return original if no normalization matched
        return prediction
