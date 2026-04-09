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
    
    def _clean_and_parse_json(json_str: str) -> dict | None:
        """Try to clean and parse a JSON string with multiple fallback strategies."""
        json_str = json_str.strip()
        if not json_str:
            return None
            
        # Strategy 1: Direct parse
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Remove trailing commas before closing braces/brackets
        try:
            cleaned = re.sub(r',(\s*[}\]])', r'\1', json_str)
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        
        # Strategy 3: Fix single quotes used as JSON delimiters
        try:
            cleaned = re.sub(r',(\s*[}\]])', r'\1', json_str)
            cleaned = re.sub(r"'([^']*?)':", r'"\1":', cleaned)
            cleaned = re.sub(r":\s*'([^']*?)'", r': "\1"', cleaned)
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        
        # Strategy 4: Handle newlines and extra whitespace in strings
        try:
            cleaned = re.sub(r',(\s*[}\]])', r'\1', json_str)
            cleaned = re.sub(r"'([^']*?)':", r'"\1":', cleaned)
            cleaned = re.sub(r":\s*'([^']*?)'", r': "\1"', cleaned)
            # Replace literal newlines within strings with escaped newlines
            cleaned = re.sub(r'(?<=")\n(?=")', '\\n', cleaned)
            cleaned = re.sub(r'(?<=: )"([^"]*)\n([^"]*)"', lambda m: '"' + m.group(1) + '\\n' + m.group(2) + '"', cleaned)
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        
        # Strategy 5: Handle escaped quotes and control characters
        try:
            cleaned = re.sub(r',(\s*[}\]])', r'\1', json_str)
            # Fix unescaped quotes within strings
            cleaned = re.sub(r'(?<=: )"([^"]*)"([^"]*)"', lambda m: '"' + m.group(1).replace('"', '\\"') + '"' + m.group(2).replace('"', '\\"') + '"', cleaned)
            # Remove control characters
            cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', cleaned)
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        
        # Strategy 6: Extract just the assessment field if full JSON fails
        try:
            # Try to extract assessment value directly
            assessment_match = re.search(r'"assessment"\s*:\s*"([^"]+)"', json_str, re.IGNORECASE)
            if assessment_match:
                return {"assessment": assessment_match.group(1)}
            # Try with single quotes
            assessment_match = re.search(r"'assessment'\s*:\s*'([^']+)'", json_str, re.IGNORECASE)
            if assessment_match:
                return {"assessment": assessment_match.group(1)}
        except Exception:
            pass
        
        return None
    
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
        
        parsed = _clean_and_parse_json(inner)
        if parsed is not None:
            results.append(parsed)
    
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
            
            parsed = _clean_and_parse_json(inner)
            if parsed is not None:
                results.append(parsed)
                break  # Only process first valid JSON block
    
    # Try to find inline JSON objects as a last resort
    if not results:
        try:
            # Look for JSON-like structures with curly braces
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
                            parsed = _clean_and_parse_json(json_str)
                            if parsed is not None and isinstance(parsed, dict):
                                results.append(parsed)
                            break
                brace_start = text.find('{', brace_start + 1)
        except Exception:
            pass
    
    return results or None


def _extract_prediction_from_text(text: str) -> str:
    """Extract prediction from plain text when JSON parsing fails.
    
    Tries to find text after common markers, looks for assessment keywords,
    or falls back to the last non-empty line.
    """
    # First, try to find explicit assessment markers
    assessment_markers = [
        ("Assessment:", ["Correct", "Partially correct", "Incorrect"]),
        ("Grade:", ["Correct", "Partially correct", "Incorrect"]),
        ("Verdict:", ["Correct", "Partially correct", "Incorrect"]),
        ("Decision:", ["Correct", "Partially correct", "Incorrect"]),
    ]
    
    for marker, valid_values in assessment_markers:
        if marker in text:
            parts = text.split(marker, 1)
            if len(parts) > 1:
                candidate = parts[1].strip().split('\n')[0].strip()
                # Check if it matches one of the valid assessment values
                candidate_clean = candidate.strip('"\'').strip()
                for valid in valid_values:
                    if valid.lower() in candidate_clean.lower():
                        return valid
                if candidate and not candidate.startswith(('{', '[')):
                    return candidate
    
    # Try to find text after common markers
    markers = ["Response:", "Answer:", "Score:", "Result:", 
               "Final Answer:", "Conclusion:"]
    for marker in markers:
        if marker in text:
            parts = text.split(marker, 1)
            if len(parts) > 1:
                candidate = parts[1].strip().split('\n')[0].strip()
                if candidate and not candidate.startswith(('{', '[')):
                    return candidate
    
    # Look for assessment keywords anywhere in the text
    text_lower = text.lower()
    if "partially correct" in text_lower:
        return "Partially correct"
    elif "incorrect" in text_lower and "not incorrect" not in text_lower:
        return "Incorrect"
    elif "correct" in text_lower:
        return "Correct"
    
    # Fallback: extract the last non-empty line
    lines = text.strip().split('\n')
    for line in reversed(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith(('<', '{', '[', '`')):
            # Check if the last line contains an assessment
            stripped_lower = stripped.lower()
            if "partially correct" in stripped_lower:
                return "Partially correct"
            elif "incorrect" in stripped_lower:
                return "Incorrect"
            elif "correct" in stripped_lower:
                return "Correct"
            return stripped
    
    return "None"


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

**Assessment Categories**:
- **Correct**: The student's solution is mathematically sound, complete, and correctly solves the problem. The solution demonstrates full understanding and provides a valid proof or answer.
- **Partially correct**: The student made some progress but the solution is incomplete, has minor errors, or only partially addresses the problem. The student shows partial understanding but the solution is not fully correct.
- **Incorrect**: The student's solution is fundamentally flawed, contains major errors, or does not solve the problem. The approach is wrong or the solution is completely invalid.

**Important Instructions**:
1. Ensure your response is valid JSON without trailing commas.
2. The "assessment" field MUST be exactly one of: 'Correct', 'Partially correct', or 'Incorrect' (case-sensitive).
3. The "response" field should contain the final answer that will be used for evaluation.
4. Be objective and consistent in your grading - follow the official solution and grading guidelines closely.
5. If the student's answer contains the correct final answer but the reasoning is flawed, mark it as "Partially correct" or "Incorrect" depending on the severity of the errors."""

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
                    # Priority order for extracting prediction from JSON fields
                    for field in ["assessment", "response", "reasoning"]:
                        if field in last_json:
                            prediction = last_json[field]
                            break
                    else:
                        # Fallback: try any string field
                        for value in last_json.values():
                            if isinstance(value, str) and value.strip():
                                prediction = value
                                break
                else:
                    # If no JSON found, try to extract from text markers or last line
                    prediction = _extract_prediction_from_text(last_message)
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Normalize prediction to expected format
        prediction = str(prediction).strip()
        prediction_lower = prediction.lower()
        
        # Define assessment categories with their variations
        assessment_categories = {
            "Correct": [
                "correct", "right", "true", "yes", "pass", "accepted", 
                "full marks", "full credit", "perfect", "excellent", 
                "complete solution", "valid", "accurate", "properly solved",
                "solved correctly", "correct solution", "correct answer",
                "fully correct", "completely correct", "entirely correct"
            ],
            "Partially correct": [
                "partially correct", "partial", "partial credit", 
                "partially right", "incomplete", "mostly correct",
                "some progress", "partial solution", "half correct",
                "partial marks", "partial success", "partially solved",
                "incomplete solution", "partial answer", "some correct",
                "minor errors", "small errors", "nearly correct",
                "almost correct", "mostly right", "partial understanding"
            ],
            "Incorrect": [
                "incorrect", "wrong", "false", "no", "fail", "rejected", 
                "zero", "0", "invalid", "erroneous", "unsolved",
                "no solution", "incorrect solution", "wrong answer",
                "fundamentally flawed", "does not solve", "not correct",
                "not solved", "failed", "failure", "incorrect approach",
                "wrong approach", "incorrect method", "wrong method",
                "does not work", "not valid", "not accurate"
            ]
        }
        
        # Check for exact matches first, then fuzzy matches
        for category, variations in assessment_categories.items():
            if prediction_lower in variations:
                prediction = category
                break
        else:
            # Handle numeric scores (0-100 scale or 0-1 scale) before fuzzy matching
            # This prevents "100" from being matched as "Incorrect" because it contains "0"
            try:
                # Try to extract numeric value - look for standalone numbers
                numeric_match = re.search(r'(?:^|\s)(\d+(?:\.\d+)?)(?:\s*$|\s)', prediction)
                if numeric_match:
                    score = float(numeric_match.group(1))
                    # Handle 0-100 scale
                    if score > 1 and score <= 100:
                        if score >= 80:
                            prediction = "Correct"
                        elif score >= 40:
                            prediction = "Partially correct"
                        else:
                            prediction = "Incorrect"
                    # Handle 0-1 scale
                    elif score >= 0 and score <= 1:
                        if score >= 0.8:
                            prediction = "Correct"
                        elif score >= 0.4:
                            prediction = "Partially correct"
                        else:
                            prediction = "Incorrect"
                else:
                    # Fuzzy matching: check if any variation is contained in the prediction
                    # Use word boundary matching to avoid "100" matching "0"
                    for category, variations in assessment_categories.items():
                        if any(re.search(r'\b' + re.escape(var) + r'\b', prediction_lower) for var in variations):
                            prediction = category
                            break
            except (ValueError, TypeError):
                # Fuzzy matching fallback
                for category, variations in assessment_categories.items():
                    if any(re.search(r'\b' + re.escape(var) + r'\b', prediction_lower) for var in variations):
                        prediction = category
                        break
        
        return prediction, msg_history
