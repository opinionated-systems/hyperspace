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
from collections import Counter

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
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
    """Attempt to repair common JSON syntax errors.
    
    Fixes:
    - Trailing commas before closing braces/brackets
    - Single quotes instead of double quotes
    - Unescaped newlines in strings
    - Missing closing braces/brackets
    - Unescaped quotes within strings
    - Comments in JSON (// and /* */)
    - Control characters
    - Unicode escape sequences
    - BOM (Byte Order Mark) characters
    """
    try:
        # First, try parsing as-is
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Remove BOM if present
    if text.startswith('\ufeff'):
        text = text[1:]
    
    # Remove control characters except tab, newline, carriage return
    repaired = ''.join(char for char in text if char in '\t\n\r' or ord(char) >= 32)
    
    # Remove single-line comments (// ...)
    repaired = re.sub(r'//[^\n]*', '', repaired)
    
    # Remove multi-line comments (/* ... */)
    repaired = re.sub(r'/\*.*?\*/', '', repaired, flags=re.DOTALL)
    
    # Remove trailing commas before } or ]
    repaired = re.sub(r',\s*([}\]])', r'\1', repaired)
    
    # Try to fix single quotes (simple cases only)
    # Replace single quotes around keys/values with double quotes
    # This is a best-effort approach
    repaired = re.sub(r"'([^']*?)':", r'"\1":', repaired)
    repaired = re.sub(r":\s*'([^']*?)'([,}\]])", r': "\1"\2', repaired)
    
    # Escape unescaped newlines in strings (but not in already-escaped contexts)
    # Use a more careful approach: find string content and escape newlines within
    def escape_newlines_in_strings(match):
        content = match.group(1)
        # Escape unescaped newlines
        content = re.sub(r'(?<!\\)\n', r'\\n', content)
        content = re.sub(r'(?<!\\)\r', r'\\r', content)
        content = re.sub(r'(?<!\\)\t', r'\\t', content)
        return '"' + content + '"'
    
    # Match string contents (simplified approach)
    repaired = re.sub(r'"((?:[^"\\]|\\.)*?)"', escape_newlines_in_strings, repaired)
    
    # Try to balance braces
    open_braces = repaired.count('{') - repaired.count('}')
    if open_braces > 0:
        repaired += '}' * open_braces
    
    open_brackets = repaired.count('[') - repaired.count(']')
    if open_brackets > 0:
        repaired += ']' * open_brackets
    
    # Try parsing the repaired version
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass
    
    # Second attempt: try to extract just the first complete JSON object
    try:
        # Find the first { and matching }
        start = repaired.find('{')
        if start == -1:
            return None
        
        brace_count = 0
        in_string = False
        escape_next = False
        
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
                        # Found complete object
                        try:
                            return json.loads(repaired[start:start+i+1])
                        except json.JSONDecodeError:
                            # Try repairing just this subset
                            subset = repaired[start:start+i+1]
                            # Remove any trailing commas in this subset
                            subset = re.sub(r',\s*}', '}', subset)
                            try:
                                return json.loads(subset)
                            except:
                                return None
        return None
    except Exception:
        pass
    
    # Third attempt: try to find and repair JSON with common LLM output patterns
    try:
        # Look for JSON-like content that might be wrapped in other text
        json_pattern = re.search(r'\{[\s\S]*?"reasoning"[\s\S]*?"response"[\s\S]*?\}', repaired, re.IGNORECASE)
        if json_pattern:
            candidate = json_pattern.group(0)
            # Clean up the candidate
            candidate = re.sub(r',\s*}', '}', candidate)
            candidate = re.sub(r',\s*\]', ']', candidate)
            try:
                return json.loads(candidate)
            except:
                pass
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
        reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*(?:\.[^"]*)*)"', text, re.DOTALL)
        response_match = re.search(r'"response"\s*:\s*"([^"]*)"', text, re.IGNORECASE)
        
        if reasoning_match or response_match:
            extracted = {}
            if reasoning_match:
                extracted["reasoning"] = reasoning_match.group(1).replace('\\n', '\n').replace('\\"', '"')
            if response_match:
                extracted["response"] = response_match.group(1)
            elif not response_match and reasoning_match:
                # Try to infer response from text patterns
                text_lower = text.lower()
                if "correct" in text_lower and "incorrect" not in text_lower:
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
        """Build a structured prompt for IMO grading with chain-of-thought reasoning."""
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

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis here, including specific observations about the student's work",
    "response": "Your final evaluation here - should be one of: 'correct', 'incorrect', 'partial', or a specific score if applicable"
}}
</json>

The "response" field must contain a clear, concise final determination. Use:
- "correct" if the answer is fully correct with proper reasoning
- "incorrect" if the answer is wrong or has critical errors
- "partial" if the answer has some correct elements but is incomplete or has minor errors
- A specific score (e.g., "7" or "3/7") if the problem uses point-based scoring"""
        
        return prompt

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history with multiple fallback strategies.
        
        Enhanced for IMO grading: specifically handles correctness evaluation,
        numeric scores, and partial credit scenarios.
        
        Improved version with better handling of edge cases and more robust
        extraction patterns for various response formats.
        
        New: Added confidence scoring and multi-source consensus for ambiguous cases.
        """
        if not msg_history:
            self.log_fn("Warning: Empty message history")
            return "None"
        
        last_text = msg_history[-1].get("text", "")
        if not last_text:
            self.log_fn("Warning: Last message has no text content")
            return "None"
        
        # Collect all potential predictions with confidence scores
        predictions = []
        
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
                            pred = str(value).lower().strip()
                            predictions.append((pred, 0.9, f"json:{key}"))
                        elif isinstance(value, (list, dict)):
                            predictions.append((json.dumps(value), 0.8, f"json:{key}:complex"))
                
                # Check for correctness boolean
                if "correct" in last_obj:
                    correct_val = last_obj["correct"]
                    if isinstance(correct_val, bool):
                        pred = "correct" if correct_val else "incorrect"
                        predictions.append((pred, 0.95, "json:correct:bool"))
                    else:
                        predictions.append((str(correct_val).lower(), 0.7, "json:correct"))
                
                # Check for points/partial credit
                if "points" in last_obj:
                    pred = f"points:{last_obj['points']}"
                    predictions.append((pred, 0.85, "json:points"))
        except Exception as e:
            self.log_fn(f"Flexible extraction failed: {e}")
        
        # Fallback 1: try to find any JSON-like structure with regex
        try:
            # Look for patterns like "response": "..." or 'response': '...'
            patterns = [
                (r'"grade"\s*:\s*"([^"]+)"', 0.85, "regex:grade"),
                (r'"score"\s*:\s*"?([^"},\s]+)"?', 0.85, "regex:score"),
                (r'"evaluation"\s*:\s*"([^"]+)"', 0.8, "regex:evaluation"),
                (r'"response"\s*:\s*"([^"]+)"', 0.9, "regex:response"),
                (r'"answer"\s*:\s*"([^"]+)"', 0.75, "regex:answer"),
                (r'"verdict"\s*:\s*"([^"]+)"', 0.85, "regex:verdict"),
                (r'"correct"\s*:\s*(true|false)', 0.9, "regex:correct:bool"),
            ]
            for pattern, conf, source in patterns:
                match = re.search(pattern, last_text, re.IGNORECASE)
                if match:
                    pred = match.group(1).lower().strip() if len(match.groups()) > 0 else match.group(0).lower().strip()
                    if source == "regex:correct:bool":
                        pred = "correct" if pred == "true" else "incorrect"
                    predictions.append((pred, conf, source))
        except Exception as e:
            self.log_fn(f"Regex extraction failed: {e}")
        
        # Fallback 2: Enhanced text-based extraction for grading scenarios
        text_lower = last_text.lower()
        
        # Check for numeric scores (e.g., "score: 7", "7/7", "7 points")
        score_patterns = [
            (r'(?:score|grade|points?)\s*[:=]\s*(\d+(?:\.\d+)?)', 0.8, "text:score"),
            (r'(\d+)\s*/\s*\d+\s*(?:points?)?', 0.75, "text:fraction"),
            (r'(?:awarded|given|assigned)\s+(\d+)\s*(?:points?)?', 0.75, "text:awarded"),
            (r'(?:score|grade)\s+of\s+(\d+)', 0.8, "text:score_of"),
        ]
        for pattern, conf, source in score_patterns:
            match = re.search(pattern, text_lower)
            if match:
                pred = f"score:{match.group(1)}"
                predictions.append((pred, conf, source))
        
        # Check for correctness indicators with improved logic
        # Look for explicit verdict patterns first
        verdict_patterns = [
            (r'\bthe\s+answer\s+is\s+(correct|incorrect|wrong|right)\b', 0.9, "text:verdict_explicit"),
            (r'\bverdict\s*[:=]\s*(correct|incorrect|partial)\b', 0.9, "text:verdict_label"),
            (r'\b(correct|incorrect|partial)\s+answer\b', 0.85, "text:answer_quality"),
            (r'\banswer\s+is\s+(correct|incorrect|partial)\b', 0.85, "text:answer_is"),
        ]
        for pattern, conf, source in verdict_patterns:
            match = re.search(pattern, text_lower)
            if match:
                verdict = match.group(1).lower()
                if verdict in ["right", "correct"]:
                    predictions.append(("correct", conf, source))
                elif verdict in ["wrong", "incorrect"]:
                    predictions.append(("incorrect", conf, source))
                elif verdict == "partial":
                    predictions.append(("partial", conf, source))
        
        # Check for correctness indicators with context analysis
        if "correct" in text_lower:
            # Distinguish between "correct" and "incorrect"
            if "incorrect" in text_lower or "not correct" in text_lower:
                predictions.append(("incorrect", 0.7, "text:correct_negated"))
            else:
                predictions.append(("correct", 0.75, "text:correct_mentioned"))
        
        # Check for partial credit indicators
        partial_terms = ["partial", "partially", "some credit", "incomplete", "partial credit"]
        if any(term in text_lower for term in partial_terms):
            predictions.append(("partial", 0.7, "text:partial_terms"))
        
        # If we have predictions, select the best one based on confidence and consensus
        if predictions:
            # Sort by confidence (descending)
            predictions.sort(key=lambda x: x[1], reverse=True)
            
            # Check for consensus among high-confidence predictions
            high_conf_preds = [p for p in predictions if p[1] >= 0.8]
            if high_conf_preds:
                # Group by prediction value
                from collections import Counter
                pred_values = [p[0] for p in high_conf_preds]
                most_common = Counter(pred_values).most_common(1)[0]
                
                # If there's consensus among high-confidence predictions, use it
                if most_common[1] >= 2:  # At least 2 high-confidence predictions agree
                    consensus_pred = most_common[0]
                    self.log_fn(f"Consensus prediction from {len(high_conf_preds)} high-confidence sources: {consensus_pred}")
                    return consensus_pred
            
            # Return highest confidence prediction
            best_pred, best_conf, best_source = predictions[0]
            self.log_fn(f"Best prediction from {best_source} (confidence: {best_conf}): {best_pred}")
            return best_pred
        
        # Fallback 3: return first 200 chars of text as prediction
        if last_text.strip():
            return last_text.strip()[:200]
        
        return "None"

    def _normalize_prediction(self, prediction: str) -> str:
        """Normalize prediction to standard format for evaluation.
        
        Converts various forms of correct/incorrect/partial to standardized values.
        """
        if not prediction or prediction == "None":
            return "None"
        
        pred_lower = prediction.lower().strip()
        
        # Handle score format (e.g., "score:7", "7/7")
        if pred_lower.startswith("score:") or "/" in pred_lower:
            return prediction
        
        # Normalize correct variations
        if any(term in pred_lower for term in ["correct", "right", "true", "valid", "accepted"]):
            if "incorrect" not in pred_lower and "not correct" not in pred_lower:
                return "correct"
        
        # Normalize incorrect variations
        if any(term in pred_lower for term in ["incorrect", "wrong", "false", "invalid", "rejected", "error"]):
            return "incorrect"
        
        # Normalize partial variations
        if any(term in pred_lower for term in ["partial", "partially", "incomplete", "some credit"]):
            return "partial"
        
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
        else:
            # Log successful extraction
            preview = prediction[:100] if len(prediction) > 100 else prediction
            self.log_fn(f"Extracted prediction: {preview}")

        return str(prediction), msg_history
