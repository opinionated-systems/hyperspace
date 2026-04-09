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
    
    Args:
        text: The input text containing <json> tags
        
    Returns:
        List of parsed JSON objects, or None if no valid JSON found
    """
    results = []
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
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            continue
    return results or None


def _extract_json_flexible(text: str) -> list[dict] | None:
    """Extract JSON objects with multiple fallback strategies.
    
    Tries multiple patterns in order of reliability:
    1. <json>...</json> tags
    2. ```json code blocks
    3. Raw JSON objects at start/end of text
    4. JSON-like structures with relaxed parsing
    5. Repair common JSON syntax errors
    """
    results = []
    
    # Strategy 1: <json> tags (original)
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
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # Try to repair common JSON errors
            repaired = _repair_json(inner)
            if repaired:
                results.append(repaired)
            continue
    
    # Strategy 2: ```json code blocks
    if not results:
        pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                results.append(json.loads(match.strip()))
            except json.JSONDecodeError:
                repaired = _repair_json(match.strip())
                if repaired:
                    results.append(repaired)
                continue
    
    # Strategy 3: Look for JSON objects directly
    if not results:
        # Try to find JSON objects between curly braces
        pattern = r'\{[^{}]*"[^"]+"[^{}]*\}'
        for match in re.finditer(pattern, text, re.DOTALL):
            try:
                # Expand to capture nested structures
                start = match.start()
                brace_count = 0
                end = start
                for i, char in enumerate(text[start:]):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end = start + i + 1
                            break
                if end > start:
                    obj_str = text[start:end]
                    try:
                        results.append(json.loads(obj_str))
                    except json.JSONDecodeError:
                        repaired = _repair_json(obj_str)
                        if repaired:
                            results.append(repaired)
            except (json.JSONDecodeError, ValueError):
                continue
    
    return results or None


def _repair_json(text: str) -> dict | None:
    """Attempt to repair common JSON syntax errors with enhanced robustness.
    
    This function applies multiple repair strategies to fix malformed JSON:
    
    Fixes applied (in order):
    - Trailing commas before closing braces/brackets
    - Single quotes instead of double quotes
    - Unescaped newlines in strings
    - Missing closing braces/brackets
    - Unescaped quotes within strings
    - Comments in JSON (// and /* */)
    - Control characters
    - Unicode escape sequences
    - Mixed quote styles
    
    Args:
        text: The potentially malformed JSON string
        
    Returns:
        Parsed dict if repair succeeds, None otherwise
    """
    if not text or not text.strip():
        return None
    
    try:
        # First, try parsing as-is
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Remove control characters except tab, newline, carriage return
    repaired = ''.join(char for char in text if char in '\t\n\r' or ord(char) >= 32)
    
    # Remove single-line comments (// ...)
    repaired = re.sub(r'//[^\n]*', '', repaired)
    
    # Remove multi-line comments (/* ... */)
    repaired = re.sub(r'/\*.*?\*/', '', repaired, flags=re.DOTALL)
    
    # Remove trailing commas before } or ]
    repaired = re.sub(r',\s*([}\]])', r'\1', repaired)
    
    # Try to fix single quotes (comprehensive approach)
    # Replace single quotes around keys with double quotes
    repaired = re.sub(r"(?<=[{\s,])'([^']+)'(?=\s*:)", r'"\1"', repaired)
    # Replace single quotes around string values with double quotes
    repaired = re.sub(r":\s*'([^']*?)'(?=\s*[,}\]])", r': "\1"', repaired)
    
    # Handle escaped quotes within strings
    repaired = re.sub(r'\\"', '"', repaired)
    
    # Try parsing after basic repairs
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass
    
    # Advanced: Extract and repair JSON object character by character
    try:
        start = repaired.find('{')
        if start == -1:
            return None
        
        # Use a stack-based parser to find the complete JSON object
        brace_count = 0
        in_string = False
        escape_next = False
        end_pos = -1
        
        for i, char in enumerate(repaired[start:]):
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
                        end_pos = start + i + 1
                        break
        
        if end_pos > start:
            candidate = repaired[start:end_pos]
            # Clean up the candidate
            candidate = re.sub(r',\s*}', '}', candidate)
            candidate = re.sub(r',\s*]', ']', candidate)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                # Try one more time with aggressive cleaning
                candidate = re.sub(r'\n', ' ', candidate)
                candidate = re.sub(r'\r', ' ', candidate)
                candidate = re.sub(r'\t', ' ', candidate)
                candidate = re.sub(r' +', ' ', candidate)
                try:
                    return json.loads(candidate)
                except:
                    pass
    except Exception:
        pass
    
    # Final fallback: Try to extract key-value pairs manually
    try:
        result = {}
        # Look for "key": "value" or "key": value patterns
        pattern = r'"([^"]+)"\s*:\s*("(?:[^"\\]|\\.)*"|\[[^\]]*\]|\{[^}]*\}|[^",\s}]+)'
        matches = re.findall(pattern, repaired)
        for key, value in matches:
            # Try to parse the value
            try:
                if value.startswith('"') and value.endswith('"'):
                    result[key] = value[1:-1]
                elif value.lower() == 'true':
                    result[key] = True
                elif value.lower() == 'false':
                    result[key] = False
                elif value.lower() == 'null':
                    result[key] = None
                elif value.startswith('[') or value.startswith('{'):
                    result[key] = json.loads(value)
                else:
                    # Try as number
                    try:
                        if '.' in value:
                            result[key] = float(value)
                        else:
                            result[key] = int(value)
                    except:
                        result[key] = value
            except:
                result[key] = value
        
        if result:
            return result
    except Exception:
        pass
    
    return None


def _extract_json_robust(text: str) -> list[dict] | None:
    """Most robust JSON extraction with multiple fallback strategies.
    
    This is the primary extraction function that tries multiple
    strategies in order of reliability.
    """
    if not text or not text.strip():
        return None
    
    # Strategy 1: <json> tags (original format)
    results = _extract_jsons(text)
    if results:
        return results
    
    # Strategy 2: Flexible extraction with repair
    results = _extract_json_flexible(text)
    if results:
        return results
    
    # Strategy 3: Look for any JSON-like structure
    # Try to find content between outermost braces
    try:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end+1]
            # Try to repair and parse
            repaired = _repair_json(candidate)
            if repaired:
                return [repaired]
    except Exception:
        pass
    
    # Strategy 4: Extract key-value pairs from malformed JSON
    # Last resort: try to extract reasoning and response fields directly
    try:
        # More flexible pattern matching for various field names
        field_patterns = {
            'reasoning': [
                r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"',
                r'"reasoning"\s*:\s*"([^"]*(?:\.[^"]*)*)"',
                r'"analysis"\s*:\s*"((?:[^"\\]|\\.)*)"',
                r'"explanation"\s*:\s*"((?:[^"\\]|\\.)*)"',
            ],
            'response': [
                r'"response"\s*:\s*"([^"]*)"',
                r'"answer"\s*:\s*"([^"]*)"',
                r'"result"\s*:\s*"([^"]*)"',
                r'"conclusion"\s*:\s*"([^"]*)"',
                r'"verdict"\s*:\s*"([^"]*)"',
                r'"grade"\s*:\s*"([^"]*)"',
                r'"evaluation"\s*:\s*"([^"]*)"',
            ]
        }
        
        extracted = {}
        
        # Try to extract reasoning
        for pattern in field_patterns['reasoning']:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                reasoning = match.group(1).replace('\\n', '\n').replace('\\"', '"').replace('\\t', '\t')
                extracted["reasoning"] = reasoning
                break
        
        # Try to extract response
        for pattern in field_patterns['response']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                extracted["response"] = match.group(1)
                break
        
        # If we have reasoning but no response, try to infer from text
        if "reasoning" in extracted and "response" not in extracted:
            text_lower = text.lower()
            # Look for explicit verdict patterns
            if re.search(r'\bthe\s+answer\s+is\s+(correct|right)\b', text_lower):
                extracted["response"] = "correct"
            elif re.search(r'\bthe\s+answer\s+is\s+(incorrect|wrong)\b', text_lower):
                extracted["response"] = "incorrect"
            elif re.search(r'\bpartial\s+(?:credit|correct)\b', text_lower):
                extracted["response"] = "partial"
            elif "correct" in text_lower and "incorrect" not in text_lower:
                extracted["response"] = "correct"
            elif "incorrect" in text_lower:
                extracted["response"] = "incorrect"
            elif "partial" in text_lower:
                extracted["response"] = "partial"
        
        if extracted:
            return [extracted]
    except Exception:
        pass
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _build_prompt(self, inputs: dict) -> str:
        """Build a structured prompt for IMO grading with chain-of-thought reasoning.
        
        Enhanced prompt with clearer instructions and examples for better grading accuracy.
        """
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        prompt = f"""You are an expert {domain} grader evaluating student solutions to International Mathematical Olympiad (IMO) style competition problems.

Your task is to carefully analyze the student's answer and provide a rigorous evaluation according to the official solution and grading guidelines.

## Problem Statement
{problem}

## Official Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Instructions

Think through this step-by-step:

1. **Understand the Problem**: What is being asked? What are the key concepts and theorems involved?

2. **Analyze the Official Solution**: What is the correct approach? What is the final answer? What are the critical steps that must be present?

3. **Review the Student's Answer**: What approach did the student take? What is their final answer? Did they show all necessary work?

4. **Compare and Evaluate**: Does the student's answer match the official solution? Consider:
   - Is the final answer numerically/algebraically equivalent to the official solution?
   - Did the student demonstrate correct mathematical reasoning?
   - Are there any logical gaps or errors in the student's work?
   - Did the student use appropriate methods and theorems?
   - Is the solution complete or partial?

5. **Assign Grade**: Based on your analysis, provide your evaluation.

## Response Format

You MUST respond in valid JSON format with exactly this structure:

<json>
{{
    "reasoning": "Your detailed step-by-step analysis here. Explain your thought process clearly.",
    "response": "Your final evaluation - must be exactly one of: 'correct', 'incorrect', or 'partial'"
}}
</json>

### Grading Criteria:
- **"correct"**: The answer is fully correct with proper reasoning and matches the official solution
- **"incorrect"**: The answer is wrong, has critical errors, or fundamentally misunderstands the problem
- **"partial"**: The answer has some correct elements but is incomplete, has minor errors, or shows partial understanding

### Important Notes:
- The "response" field MUST contain exactly one of: "correct", "incorrect", or "partial"
- Do not include any other text outside the JSON tags
- Ensure your JSON is valid (proper quotes, no trailing commas)
- Be strict but fair in your evaluation
- Consider both the final answer AND the reasoning process"""
        
        return prompt

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history with multiple fallback strategies.
        
        Enhanced for IMO grading: specifically handles correctness evaluation,
        numeric scores, and partial credit scenarios.
        
        Improved version with better handling of edge cases and more robust
        extraction patterns for various response formats.
        """
        if not msg_history:
            self.log_fn("Warning: Empty message history")
            return "None"
        
        last_text = msg_history[-1].get("text", "")
        if not last_text:
            self.log_fn("Warning: Last message has no text content")
            return "None"
        
        # Try robust extraction first (includes all strategies)
        try:
            extracted = _extract_json_robust(last_text)
            if extracted:
                # Prefer response field, fallback to other common fields
                last_obj = extracted[-1]
                
                # Check for structured grading response first
                for key in ["grade", "score", "evaluation", "response", "answer", "result", "conclusion", "verdict"]:
                    if key in last_obj:
                        value = last_obj[key]
                        # Handle different value types
                        if isinstance(value, (str, int, float, bool)):
                            return str(value)
                        elif isinstance(value, (list, dict)):
                            return json.dumps(value)
                
                # Check for correctness boolean
                if "correct" in last_obj:
                    correct_val = last_obj["correct"]
                    if isinstance(correct_val, bool):
                        return "correct" if correct_val else "incorrect"
                    return str(correct_val)
                
                # Check for points/partial credit
                if "points" in last_obj:
                    return f"points:{last_obj['points']}"
                
                # If no known field, return the whole object as string
                return str(last_obj)
        except Exception as e:
            self.log_fn(f"Flexible extraction failed: {e}")
        
        # Fallback 1: try to find any JSON-like structure with regex
        try:
            # Look for patterns like "response": "..." or 'response': '...'
            patterns = [
                r'"grade"\s*:\s*"([^"]+)"',
                r'"score"\s*:\s*"?([^"},\s]+)"?',
                r'"evaluation"\s*:\s*"([^"]+)"',
                r'"response"\s*:\s*"([^"]+)"',
                r'"answer"\s*:\s*"([^"]+)"',
                r'"verdict"\s*:\s*"([^"]+)"',
                r'"correct"\s*:\s*(true|false)',
            ]
            for pattern in patterns:
                match = re.search(pattern, last_text, re.IGNORECASE)
                if match:
                    return match.group(1).lower() if len(match.groups()) > 0 else match.group(0)
        except Exception as e:
            self.log_fn(f"Regex extraction failed: {e}")
        
        # Fallback 2: Enhanced text-based extraction for grading scenarios
        text_lower = last_text.lower()
        
        # Check for numeric scores (e.g., "score: 7", "7/7", "7 points")
        score_patterns = [
            r'(?:score|grade|points?)\s*[:=]\s*(\d+(?:\.\d+)?)',
            r'(\d+)\s*/\s*\d+\s*(?:points?)?',
            r'(?:awarded|given|assigned)\s+(\d+)\s*(?:points?)?',
            r'(?:score|grade)\s+of\s+(\d+)',
        ]
        for pattern in score_patterns:
            match = re.search(pattern, text_lower)
            if match:
                return f"score:{match.group(1)}"
        
        # Check for correctness indicators with improved logic
        # Look for explicit verdict patterns first
        verdict_patterns = [
            (r'\bthe\s+answer\s+is\s+(correct|right)\b', 1),
            (r'\bthe\s+answer\s+is\s+(incorrect|wrong)\b', 1),
            (r'\bverdict\s*[:=]\s*(correct|incorrect|partial)\b', 1),
            (r'\b(correct|incorrect|partial)\s+answer\b', 1),
            (r'\banswer\s+is\s+(correct|incorrect|partial)\b', 1),
            (r'\bthis\s+is\s+(correct|incorrect|partial)\b', 1),
            (r'\bthe\s+student\s+(?:answer|solution)\s+is\s+(correct|incorrect|partial)\b', 1),
        ]
        for pattern, group in verdict_patterns:
            match = re.search(pattern, text_lower)
            if match:
                verdict = match.group(group).lower()
                if verdict in ["right", "correct"]:
                    return "correct"
                elif verdict in ["wrong", "incorrect"]:
                    return "incorrect"
                elif verdict == "partial":
                    return "partial"
        
        # Check for correctness indicators with context awareness
        # Look for "correct" but check if it's negated
        correct_match = re.search(r'\bcorrect\b', text_lower)
        incorrect_match = re.search(r'\bincorrect\b', text_lower)
        not_correct_match = re.search(r'\bnot\s+correct\b', text_lower)
        
        if incorrect_match or not_correct_match:
            return "incorrect"
        elif correct_match:
            return "correct"
        
        # Check for partial credit indicators
        partial_indicators = ["partial", "partially", "some credit", "incomplete", "partial credit", 
                             "partially correct", "half credit", "partial solution"]
        if any(term in text_lower for term in partial_indicators):
            return "partial"
        
        # Check for other grading terms
        if any(term in text_lower for term in ["wrong", "error", "mistake", "invalid", "rejected"]):
            return "incorrect"
        if any(term in text_lower for term in ["right", "valid", "accepted", "full credit", "complete"]):
            return "correct"
        
        # Fallback 3: return first 200 chars of text as prediction
        if last_text.strip():
            return last_text.strip()[:200]
        
        return "None"

    def _normalize_prediction(self, prediction: str) -> str:
        """Normalize prediction to standard format for evaluation.
        
        Converts various forms of correct/incorrect/partial to standardized values.
        Enhanced to handle more edge cases and variations.
        """
        if not prediction or prediction == "None":
            return "None"
        
        pred_lower = prediction.lower().strip()
        
        # Handle score format (e.g., "score:7", "7/7", "points:5")
        if pred_lower.startswith("score:") or pred_lower.startswith("points:") or "/" in pred_lower:
            return prediction
        
        # Check for numeric-only predictions (assume it's a score)
        if re.match(r'^\d+(\.\d+)?$', pred_lower):
            return f"score:{prediction}"
        
        # Check for boolean values
        if pred_lower == "true":
            return "correct"
        if pred_lower == "false":
            return "incorrect"
        
        # Normalize correct variations - be careful about negations
        correct_terms = ["correct", "right", "true", "valid", "accepted", "accurate", "proper"]
        incorrect_terms = ["incorrect", "wrong", "false", "invalid", "rejected", "error", "mistake", "inaccurate"]
        partial_terms = ["partial", "partially", "incomplete", "some credit", "half credit", "partial credit"]
        
        # Check for negated correct (e.g., "not correct", "not right")
        negation_patterns = [r'\bnot\s+', r'\bisn\'t\s+', r'\bare\'t\s+', r'\bno\s+']
        has_negation = any(re.search(pattern + r'(correct|right|valid)', pred_lower) for pattern in negation_patterns)
        
        if has_negation:
            return "incorrect"
        
        # Check for incorrect terms first (they take precedence)
        if any(term in pred_lower for term in incorrect_terms):
            return "incorrect"
        
        # Check for partial terms
        if any(term in pred_lower for term in partial_terms):
            return "partial"
        
        # Check for correct terms
        if any(term in pred_lower for term in correct_terms):
            return "correct"
        
        # Return original if no normalization applied
        return prediction

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Validate required inputs
        required_keys = ["problem", "solution", "student_answer"]
        missing_keys = [k for k in required_keys if k not in inputs or not inputs[k]]
        if missing_keys:
            error_msg = f"Error: Missing required inputs: {missing_keys}"
            self.log_fn(error_msg)
            return error_msg, [{"role": "assistant", "text": error_msg}]
        
        instruction = self._build_prompt(inputs)
        
        # Log the problem being processed (truncated for privacy)
        problem_preview = inputs.get("problem", "")[:100]
        self.log_fn(f"Processing problem: {problem_preview}...")

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            error_msg = f"Error: LLM call failed: {e}"
            self.log_fn(error_msg)
            return error_msg, [{"role": "assistant", "text": error_msg}]

        # Extract prediction from JSON
        raw_prediction = self._extract_prediction(msg_history)
        
        # Normalize prediction to standard format
        prediction = self._normalize_prediction(raw_prediction)
        
        if prediction == "None":
            self.log_fn(f"Warning: Could not extract valid prediction from response")
            # Log the raw response for debugging
            if msg_history:
                raw_text = msg_history[-1].get('text', '')
                self.log_fn(f"Raw response preview: {raw_text[:500]}...")
            # Try to extract any meaningful text as fallback
            if msg_history and msg_history[-1].get('text', '').strip():
                # Use text-based extraction as last resort
                text = msg_history[-1].get('text', '').lower()
                if 'correct' in text and 'incorrect' not in text:
                    prediction = "correct"
                    self.log_fn("Fallback: extracted 'correct' from text")
                elif 'incorrect' in text:
                    prediction = "incorrect"
                    self.log_fn("Fallback: extracted 'incorrect' from text")
                elif 'partial' in text:
                    prediction = "partial"
                    self.log_fn("Fallback: extracted 'partial' from text")
        else:
            # Log successful extraction
            preview = prediction[:100] if len(prediction) > 100 else prediction
            self.log_fn(f"Extracted prediction: {preview}")

        return str(prediction), msg_history
