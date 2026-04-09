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
        """Try to clean and parse a JSON string with various fixes."""
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Try to clean up common JSON issues
            try:
                cleaned = json_str
                # Remove trailing commas before closing braces/brackets
                cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
                # Fix single quotes used as JSON delimiters (but not within strings)
                cleaned = re.sub(r"(?<![\\])'([^']*?)'(?=\s*:)", r'"\1"', cleaned)
                cleaned = re.sub(r":\s*'([^']*?)'(?=\s*[,}\]])", r': "\1"', cleaned)
                # Fix unescaped newlines in strings
                cleaned = re.sub(r'(?<=")\n(?=")', r'\\n', cleaned)
                # Remove comments
                cleaned = re.sub(r'//[^\n]*', '', cleaned)
                cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
                return json.loads(cleaned)
            except json.JSONDecodeError:
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


def _extract_assessment_from_text(text: str) -> str | None:
    """Extract assessment from plain text when JSON parsing fails.
    
    Looks for keywords like 'Correct', 'Partially correct', 'Incorrect' in the text.
    Uses a scoring system to determine the most likely assessment.
    """
    text_lower = text.lower()
    
    # Score-based approach for more robust extraction
    scores = {
        'Correct': 0,
        'Partially correct': 0,
        'Incorrect': 0
    }
    
    # Check for explicit assessment patterns (higher weight)
    patterns = {
        'Correct': [
            r'assessment[\s]*:[\s]*"?correct"?',
            r'assessment[\s]*:[\s]*correct\b',
            r'"assessment"[\s]*:[\s]*"correct"',
            r'\bcorrect\b.*\bassessment\b',
            r'\bfull marks?\b',
            r'\bfully correct\b',
            r'\bcompletely correct\b',
        ],
        'Partially correct': [
            r'assessment[\s]*:[\s]*"?partially correct"?',
            r'assessment[\s]*:[\s]*partially correct\b',
            r'"assessment"[\s]*:[\s]*"partially correct"',
            r'\bpartially correct\b.*\bassessment\b',
            r'\bpartial credit\b',
            r'\bpartially right\b',
            r'\bincomplete\b',
        ],
        'Incorrect': [
            r'assessment[\s]*:[\s]*"?incorrect"?',
            r'assessment[\s]*:[\s]*incorrect\b',
            r'"assessment"[\s]*:[\s]*"incorrect"',
            r'\bincorrect\b.*\bassessment\b',
            r'\bwrong\b',
            r'\bfail\b',
            r'\brejected\b',
            r'\bzero\b',
        ]
    }
    
    for assessment, pattern_list in patterns.items():
        for pattern in pattern_list:
            if re.search(pattern, text_lower):
                scores[assessment] += 3  # Higher weight for explicit patterns
    
    # Check for standalone keywords (lower weight)
    if 'partially correct' in text_lower:
        scores['Partially correct'] += 2
    elif 'partial' in text_lower:
        scores['Partially correct'] += 1
        
    if 'incorrect' in text_lower:
        scores['Incorrect'] += 2
    elif 'wrong' in text_lower or 'fail' in text_lower or 'rejected' in text_lower:
        scores['Incorrect'] += 1
        
    if 'correct' in text_lower:
        # Check if it's "not correct" or "incorrect"
        if 'not correct' in text_lower or 'not fully correct' in text_lower:
            scores['Incorrect'] += 1
        else:
            scores['Correct'] += 1
    if 'right' in text_lower or 'pass' in text_lower or 'accepted' in text_lower:
        scores['Correct'] += 1
    
    # Return the assessment with highest score, or None if all scores are 0
    max_score = max(scores.values())
    if max_score == 0:
        return None
    
    # In case of tie, prefer Partially correct > Incorrect > Correct
    if scores['Partially correct'] == max_score:
        return 'Partially correct'
    if scores['Incorrect'] == max_score:
        return 'Incorrect'
    return 'Correct'


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

Then, provide your final assessment in the following JSON format. You MUST wrap your JSON response in <json> and </json> tags:

<json>
{{
    "reasoning": "Your detailed step-by-step evaluation and reasoning process",
    "assessment": "Correct" | "Partially correct" | "Incorrect",
    "response": "The final grading decision or score as specified in the guidelines"
}}
</json>

CRITICAL INSTRUCTIONS:
1. You MUST output the JSON block wrapped in <json>...</json> tags.
2. The "assessment" field MUST be exactly one of: "Correct", "Partially correct", or "Incorrect" (case-sensitive, no extra quotes).
3. Use "Correct" ONLY if the answer is fully correct with proper reasoning.
4. Use "Partially correct" if the student showed some correct work but made errors or had incomplete reasoning.
5. Use "Incorrect" if the answer is wrong or shows fundamental misunderstanding.
6. The "response" field should contain the final answer that will be used for evaluation.
7. Ensure your JSON is valid with no trailing commas.
8. Do not include any text after the closing </json> tag."""

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
        prediction = self._extract_prediction(msg_history, response)
        
        return prediction, msg_history

    def _extract_prediction(self, msg_history: list[dict], response: str) -> str:
        """Extract prediction from message history and response.
        
        Args:
            msg_history: List of message dictionaries
            response: Raw response string from LLM
            
        Returns:
            Normalized prediction string
        """
        prediction = "None"
        last_message = ""
        
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
                    # If no JSON found, try to extract assessment from plain text
                    text_assessment = _extract_assessment_from_text(last_message)
                    if text_assessment:
                        prediction = text_assessment
                    else:
                        # Try to extract the last non-empty line as a fallback
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
            # Try to extract from raw response as last resort
            if response:
                text_assessment = _extract_assessment_from_text(response)
                if text_assessment:
                    prediction = text_assessment

        return self._normalize_prediction(prediction)

    def _normalize_prediction(self, prediction: str) -> str:
        """Normalize prediction to one of the expected assessment values.
        
        Args:
            prediction: Raw prediction string
            
        Returns:
            Normalized prediction: "Correct", "Partially correct", or "Incorrect"
        """
        prediction = str(prediction).strip()
        prediction_lower = prediction.lower()
        
        # Normalize common variations of assessment values
        if prediction_lower in ["correct", "right", "true", "yes", "pass", "accepted", "full marks", "full credit", "accurate", "valid"]:
            return "Correct"
        elif prediction_lower in ["partially correct", "partial", "partial credit", "partially right", "incomplete", "mostly correct", "half correct"]:
            return "Partially correct"
        elif prediction_lower in ["incorrect", "wrong", "false", "no", "fail", "rejected", "zero", "0", "invalid", "error"]:
            return "Incorrect"
        
        # If no match, return original (may need manual review)
        return prediction
