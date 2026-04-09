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
    
    def _clean_and_parse_json(json_str: str) -> dict | None:
        """Try to clean and parse a JSON string with multiple fallback strategies.
        
        Applies fixes incrementally to avoid redundant operations.
        """
        json_str = json_str.strip()
        if not json_str:
            return None
            
        # Strategy 1: Direct parse
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Remove trailing commas before closing braces/brackets
        cleaned = re.sub(r',(\s*[}\]])', r'\1', json_str)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        
        # Strategy 3: Fix single quotes used as JSON delimiters
        cleaned = re.sub(r"'([^']*?)':", r'"\1":', cleaned)
        cleaned = re.sub(r":\s*'([^']*?)'", r': "\1"', cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        
        # Strategy 4: Handle newlines and extra whitespace in strings
        cleaned = re.sub(r'(?<=")\n(?=")', '\\n', cleaned)
        cleaned = re.sub(r'(?<=: )"([^"]*)\n([^"]*)"', lambda m: '"' + m.group(1) + '\\n' + m.group(2) + '"', cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        
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
        
        parsed = _clean_and_parse_json(inner)
        if parsed is not None:
            results.append(parsed)
    
    # Also try to find markdown json blocks ```json...```
    # Continue searching even if we found <json> blocks to capture all JSON
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
            # Continue searching for more JSON blocks (consistent with <json> handling)
            continue
    
    # Try to find inline JSON objects as a last resort
    if not results:
        try:
            # Look for JSON-like structures with curly braces
            # Use a more robust approach that handles nested braces
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
                
                # Try to parse the extracted JSON
                if json_str:
                    parsed = _clean_and_parse_json(json_str)
                    if parsed is not None and isinstance(parsed, dict):
                        results.append(parsed)
                
                # Move to next potential JSON object
                brace_start = text.find('{', brace_start + 1)
        except Exception:
            pass
    
    return results or None


def _extract_prediction_from_text(text: str) -> str:
    """Extract prediction from plain text when JSON parsing fails.
    
    Tries to find text after common markers, or falls back to the last non-empty line.
    Handles multi-line responses and extracts the most relevant assessment.
    """
    if not text or not text.strip():
        return "None"
    
    text = text.strip()
    
    # Try to find text after common markers (ordered by specificity)
    markers = [
        "Final Assessment:", "Assessment:", "Final Answer:", "Grade:", 
        "Verdict:", "Conclusion:", "Decision:", "Response:", 
        "Answer:", "Score:", "Result:"
    ]
    
    for marker in markers:
        if marker in text:
            parts = text.rsplit(marker, 1)  # Use rsplit to get the last occurrence
            if len(parts) > 1:
                # Get the text after the marker, handling multi-line responses
                after_marker = parts[1].strip()
                # Try to get the first substantial line (at least 3 chars)
                for line in after_marker.split('\n'):
                    candidate = line.strip()
                    # Skip empty lines, JSON markers, code blocks, and common formatting
                    if len(candidate) >= 3 and not candidate.startswith(('{', '[', '<', '`', '-', '*', '|', '#')):
                        # Skip lines that are just punctuation or formatting
                        if candidate not in ['}', ']', '```', '</json>', '---', '***', '']:
                            return candidate
                # If no substantial line found, return first non-empty line
                for line in after_marker.split('\n'):
                    candidate = line.strip()
                    if candidate and not candidate.startswith(('{', '[', '<', '`')):
                        return candidate
    
    # Fallback: extract the last non-empty, substantial line
    lines = text.split('\n')
    for line in reversed(lines):
        stripped = line.strip()
        # Skip empty lines, JSON markers, code blocks, and common formatting
        if stripped and len(stripped) >= 2 and not stripped.startswith(('<', '{', '[', '`', '-', '*', '|', '#')):
            # Skip lines that are just punctuation or formatting
            if stripped not in ['}', ']', '```', '</json>', '---', '***', '']:
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
- **Correct**: The student's solution is mathematically sound, complete, and correctly solves the problem.
- **Partially correct**: The student made some progress but the solution is incomplete, has minor errors, or only partially addresses the problem.
- **Incorrect**: The student's solution is fundamentally flawed, contains major errors, or does not solve the problem.

Important: 
- Ensure your response is valid JSON without trailing commas.
- The "assessment" field MUST be exactly one of: 'Correct', 'Partially correct', or 'Incorrect' (case-sensitive).
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
        
        return prediction, msg_history

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract and normalize the prediction from message history.
        
        Args:
            msg_history: List of message dictionaries from the LLM conversation
            
        Returns:
            Normalized prediction string (one of: Correct, Partially correct, Incorrect)
        """
        if not msg_history:
            return "None"
        
        # Handle both "text" (paper format) and "content" (OpenAI format) fields
        last_message = msg_history[-1].get("text") or msg_history[-1].get("content", "")
        if not last_message:
            return "None"
        
        # Try to extract from JSON first
        extracted = _extract_jsons(last_message)
        if extracted:
            # Look for the JSON object that contains assessment-related fields
            # Prefer the last one that has an "assessment" field, otherwise use the last one
            best_json = None
            for json_obj in reversed(extracted):
                if "assessment" in json_obj:
                    best_json = json_obj
                    break
            if best_json is None:
                best_json = extracted[-1]
            prediction = self._get_prediction_from_json(best_json)
        else:
            # If no JSON found, try to extract from text markers or last line
            prediction = _extract_prediction_from_text(last_message)
        
        return self._normalize_prediction(prediction)

    def _get_prediction_from_json(self, json_data: dict) -> str:
        """Extract prediction value from JSON data.
        
        Priority order: assessment > response > reasoning > first string value
        
        Args:
            json_data: Parsed JSON dictionary
            
        Returns:
            Raw prediction string
        """
        # Priority order for extracting prediction from JSON fields
        for field in ["assessment", "response", "reasoning"]:
            if field in json_data:
                value = json_data[field]
                if isinstance(value, str):
                    return value
                # Handle non-string values (convert to string)
                if value is not None:
                    return str(value)
        
        # Fallback: try any string field that looks like an assessment
        assessment_keywords = ["correct", "incorrect", "partial", "grade", "score", "result"]
        for key, value in json_data.items():
            if isinstance(value, str) and value.strip():
                # Prioritize fields with assessment-related keywords in their names
                key_lower = key.lower()
                if any(keyword in key_lower for keyword in assessment_keywords):
                    return value
        
        # Final fallback: any non-empty string value
        for value in json_data.values():
            if isinstance(value, str) and value.strip():
                return value
        
        return "None"

    def _normalize_prediction(self, prediction: str) -> str:
        """Normalize prediction to one of the expected assessment categories.
        
        Args:
            prediction: Raw prediction string
            
        Returns:
            Normalized prediction (Correct, Partially correct, Incorrect, or None)
        """
        prediction = str(prediction).strip()
        if not prediction:
            return "None"
        
        prediction_lower = prediction.lower()
        
        # Define assessment categories with their variations
        # Order matters: more specific/longer patterns should be checked first
        assessment_categories = {
            "Partially correct": [
                "partially correct", "partially right", "partial credit", 
                "partial solution", "partial marks", "partial success",
                "partial", "incomplete", "mostly correct",
                "some progress", "half correct", "partially valid",
                "partial correctness", "partially accurate"
            ],
            "Correct": [
                "correct", "right", "true", "yes", "pass", "accepted", 
                "full marks", "full credit", "perfect", "excellent", 
                "complete solution", "valid", "accurate", "properly solved",
                "correct solution", "valid solution", "sound solution"
            ],
            "Incorrect": [
                "incorrect", "fundamentally flawed", "does not solve",
                "incorrect solution", "wrong answer", "no solution",
                "wrong", "false", "no", "fail", "rejected", 
                "zero", "0", "invalid", "erroneous", "unsolved",
                "not correct", "not valid", "not accurate"
            ]
        }
        
        # Check for exact matches first, then fuzzy matches
        for category, variations in assessment_categories.items():
            if prediction_lower in variations:
                return category
        
        # Fuzzy matching: check if any variation is contained in the prediction
        # Check longer patterns first to avoid partial matches
        # Skip single-character variations to avoid false positives with numbers
        for category, variations in assessment_categories.items():
            # Sort by length descending to prioritize longer matches
            # Filter out single-character variations to avoid matching digits
            sorted_variations = sorted([v for v in variations if len(v) > 1], key=len, reverse=True)
            if any(var in prediction_lower for var in sorted_variations):
                return category
        
        # Handle numeric scores (0-100 scale or 0-1 scale)
        try:
            # Look for standalone numbers or percentages
            numeric_match = re.search(r'\b(\d+(?:\.\d+)?)\s*%?', prediction)
            if numeric_match:
                score = float(numeric_match.group(1))
                # Determine if it's a 0-1 scale (has decimal and value <= 1.0)
                # or a 0-100 scale (no decimal or value > 1.0)
                has_decimal = '.' in numeric_match.group(1)
                if has_decimal and score <= 1.0:
                    # 0-1 scale, convert to 0-100
                    score = score * 100
                # Map score to category
                if score >= 80:
                    return "Correct"
                elif score >= 40:
                    return "Partially correct"
                else:
                    return "Incorrect"
        except (ValueError, TypeError):
            pass
        
        # If no match found, return the original prediction
        return prediction
