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
    
    Enhanced to handle nested braces, comments, and common JSON formatting issues.
    """
    if not text or not isinstance(text, str):
        return None
        
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
        
        # Skip if the content looks like a placeholder/example
        if inner.startswith("{{") and "..." in inner:
            continue
            
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
            
            # Skip if the content looks like a placeholder/example
            if inner.startswith("{{") and "..." in inner:
                continue
            
            # Try to parse the JSON block
            parsed = None
            try:
                parsed = json.loads(inner)
            except json.JSONDecodeError:
                # Try to clean up common JSON issues
                try:
                    cleaned = _clean_json_string(inner)
                    parsed = json.loads(cleaned)
                except json.JSONDecodeError:
                    pass  # Continue to next block if cleanup fails
            
            if parsed is not None:
                results.append(parsed)
    
    # Try to find inline JSON objects as a last resort
    if not results:
        try:
            # Look for JSON-like structures with curly braces
            brace_start = text.find('{')
            while brace_start != -1:
                # Try to find matching closing brace
                brace_count = 0
                json_end = -1
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
                                json_end = brace_start + i + 1
                                break
                
                if json_end > brace_start:
                    json_str = text[brace_start:json_end]
                    # Skip if it looks like a template/placeholder
                    if "{{" in json_str and "..." in json_str:
                        brace_start = text.find('{', json_end)
                        continue
                        
                    try:
                        parsed = json.loads(json_str)
                        if isinstance(parsed, dict):
                            results.append(parsed)
                    except json.JSONDecodeError:
                        # Try to clean up common JSON issues
                        try:
                            cleaned = _clean_json_string(json_str)
                            parsed = json.loads(cleaned)
                            if isinstance(parsed, dict):
                                results.append(parsed)
                        except json.JSONDecodeError:
                            pass
                    # Continue searching from after this JSON object
                    brace_start = text.find('{', json_end)
                else:
                    # No matching closing brace found, move to next opening brace
                    brace_start = text.find('{', brace_start + 1)
        except Exception:
            pass
    
    return results or None


def _clean_json_string(json_str: str) -> str:
    """Clean up common JSON formatting issues.
    
    Args:
        json_str: Raw JSON string that may have formatting issues
        
    Returns:
        Cleaned JSON string
    """
    # Remove C-style comments
    cleaned = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
    cleaned = re.sub(r'//.*?$', '', cleaned, flags=re.MULTILINE)
    
    # Remove trailing commas before closing braces/brackets
    cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
    
    # Fix single quotes used as JSON delimiters (but not within strings)
    # This is a simplified approach - handle common cases
    cleaned = re.sub(r"(?<=[{\s,])'([^']*?)'(?=\s*:)", r'"\1"', cleaned)
    cleaned = re.sub(r":\s*'([^']*?)'(?=\s*[},])", r': "\1"', cleaned)
    
    # Remove control characters
    cleaned = ''.join(char for char in cleaned if ord(char) >= 32 or char in '\n\r\t')
    
    # Normalize whitespace
    cleaned = cleaned.strip()
    
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
        # Validate inputs
        if not isinstance(inputs, dict):
            logger.error(f"Invalid inputs type: {type(inputs)}")
            return "Incorrect", []
            
        # Check for required fields
        required_fields = ["problem", "student_answer"]
        missing_fields = [f for f in required_fields if f not in inputs or not inputs.get(f)]
        if missing_fields:
            logger.warning(f"Missing required fields: {missing_fields}")
        
        instruction = self._build_grading_prompt(inputs)

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return "Incorrect", []

        # Extract prediction from JSON with improved error handling
        prediction = "None"
        extraction_method = "none"
        confidence = 0.0
        
        try:
            if msg_history and len(msg_history) > 0:
                # Handle both "text" (paper format) and "content" (OpenAI format) fields
                last_message = msg_history[-1].get("text") or msg_history[-1].get("content", "")
                
                if not last_message:
                    logger.warning("Empty last message in history")
                    prediction = "Incorrect"
                    extraction_method = "empty_message"
                else:
                    # Try to extract JSON from the message
                    extracted = _extract_jsons(last_message)
                    if extracted:
                        last_json = extracted[-1]
                        extraction_method = "json"
                        
                        # Try assessment field first (contains categorical label like "Correct", "Partially correct", "Incorrect")
                        if "assessment" in last_json:
                            prediction = last_json["assessment"]
                            extraction_method = "json:assessment"
                            confidence = 1.0
                        # Fallback: try response field if assessment is not available
                        elif "response" in last_json:
                            prediction = last_json["response"]
                            extraction_method = "json:response"
                            confidence = 0.9
                        # Fallback: try reasoning field if neither is available
                        elif "reasoning" in last_json:
                            prediction = last_json["reasoning"]
                            extraction_method = "json:reasoning"
                            confidence = 0.7
                        # Fallback: try any string field
                        else:
                            for key, value in last_json.items():
                                if isinstance(value, str) and value.strip():
                                    prediction = value
                                    extraction_method = f"json:{key}"
                                    confidence = 0.5
                                    break
                    else:
                        # If no JSON found, try to extract the last non-empty line as a fallback
                        lines = last_message.strip().split('\n')
                        for line in reversed(lines):
                            stripped = line.strip()
                            if stripped and not stripped.startswith('<') and not stripped.startswith('{') and not stripped.startswith('`'):
                                prediction = stripped
                                extraction_method = "last_line"
                                confidence = 0.4
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
                                            extraction_method = f"marker:{marker}"
                                            confidence = 0.5
                                            break
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            logger.exception("Prediction extraction failed")

        # Normalize prediction to expected format
        prediction = str(prediction).strip() if prediction else "None"
        
        # Normalize common variations of assessment values
        prediction_lower = prediction.lower()
        if prediction_lower in ["correct", "right", "true", "yes", "pass", "accepted", "full marks", "full credit", "valid", "accurate"]:
            prediction = "Correct"
        elif prediction_lower in ["partially correct", "partial", "partial credit", "partially right", "incomplete", "partially accurate", "mostly correct"]:
            prediction = "Partially correct"
        elif prediction_lower in ["incorrect", "wrong", "false", "no", "fail", "rejected", "zero", "0", "invalid", "inaccurate", "error"]:
            prediction = "Incorrect"
        elif prediction == "None" or not prediction:
            # Default to Incorrect if we couldn't extract anything
            prediction = "Incorrect"
            extraction_method = "default"
        
        # Log extraction details for debugging
        self.log_fn(f"Prediction extracted via {extraction_method} (confidence: {confidence:.2f}): {prediction}")
        
        return prediction, msg_history
