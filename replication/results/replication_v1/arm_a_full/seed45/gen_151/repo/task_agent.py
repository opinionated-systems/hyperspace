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
    """Attempt to repair common JSON syntax errors with optimized robustness.
    
    Fixes:
    - Trailing commas before closing braces/brackets
    - Single quotes instead of double quotes
    - Unescaped newlines, tabs, carriage returns in strings
    - Missing closing braces/brackets
    - Unescaped quotes within strings
    - Comments in JSON (// and /* */)
    - Control characters
    """
    # Fast path: try parsing as-is first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Remove control characters except tab, newline, carriage return
    repaired = ''.join(char for char in text if char in '\t\n\r' or ord(char) >= 32)
    
    # Remove comments (single-line and multi-line)
    repaired = re.sub(r'//[^\n]*', '', repaired)
    repaired = re.sub(r'/\*.*?\*/', '', repaired, flags=re.DOTALL)
    
    # Remove trailing commas before } or ]
    repaired = re.sub(r',\s*([}\]])', r'\1', repaired)
    
    # Fix single quotes to double quotes for keys and string values
    repaired = re.sub(r"(?<=[{\[,])\s*'([^']+)'\s*:", r'"\1":', repaired)
    repaired = re.sub(r":\s*'([^']*?)'(?=\s*[,}\]])", r': "\1"', repaired)
    
    # Balance braces and brackets
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
    
    # Final attempt: extract first complete JSON object
    try:
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
                        subset = repaired[start:start+i+1]
                        subset = re.sub(r',\s*}', '}', subset)
                        subset = re.sub(r',\s*]', ']', subset)
                        try:
                            return json.loads(subset)
                        except json.JSONDecodeError:
                            try:
                                return json.loads(subset.replace("'", '"'))
                            except:
                                return None
        return None
    except Exception:
        return None


def _extract_json_robust(text: str) -> list[dict] | None:
    """Optimized JSON extraction with multiple fallback strategies.
    
    Tries strategies in order of reliability and performance.
    """
    if not text or not text.strip():
        return None
    
    # Strategy 1: <json> tags (fastest path)
    results = _extract_jsons(text)
    if results:
        return results
    
    # Strategy 2: Flexible extraction with repair
    results = _extract_json_flexible(text)
    if results:
        return results
    
    # Strategy 3: Look for JSON between outermost braces
    try:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end+1]
            repaired = _repair_json(candidate)
            if repaired:
                return [repaired]
    except Exception:
        pass
    
    # Strategy 4: Extract key-value pairs directly from malformed JSON
    try:
        reasoning_match = re.search(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
        response_match = re.search(r'"response"\s*:\s*"([^"]*)"', text, re.IGNORECASE)
        
        if reasoning_match or response_match:
            extracted = {}
            if reasoning_match:
                reasoning = reasoning_match.group(1)
                reasoning = reasoning.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r').replace('\\"', '"').replace('\\\\', '\\')
                extracted["reasoning"] = reasoning
            if response_match:
                extracted["response"] = response_match.group(1)
            elif reasoning_match and not response_match:
                text_lower = text.lower()
                if "incorrect" in text_lower or "not correct" in text_lower or "wrong" in text_lower:
                    extracted["response"] = "incorrect"
                elif "partial" in text_lower or "partially" in text_lower:
                    extracted["response"] = "partial"
                elif "correct" in text_lower or "right" in text_lower:
                    extracted["response"] = "correct"
            
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
        """Build an optimized structured prompt for IMO grading."""
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        prompt_parts = [
            f"You are an expert {domain} grader for IMO-style problems.",
            "",
            "## Problem",
            problem,
            "",
            "## Official Solution",
            solution,
        ]
        
        if grading_guidelines:
            prompt_parts.extend([
                "",
                "## Grading Guidelines",
                grading_guidelines,
            ])
        
        prompt_parts.extend([
            "",
            "## Student's Answer",
            student_answer,
            "",
            "## Evaluation Instructions",
            "",
            "Analyze the student's answer step-by-step:",
            "1. Understand the problem requirements",
            "2. Compare the student's approach to the official solution",
            "3. Check for correct reasoning, methods, and final answer",
            "4. Identify any errors, gaps, or partial credit",
            "",
            "## Response Format (JSON)",
            "",
            "<json>",
            '{',
            '    "reasoning": "Your detailed analysis of the student work",',
            '    "response": "correct | incorrect | partial | score"',
            '}',
            "</json>",
            "",
            "Response values:",
            '- "correct": fully correct with proper reasoning',
            '- "incorrect": wrong answer or critical errors',
            '- "partial": some correct elements but incomplete',
            '- score format: "7" or "3/7" for point-based problems',
            "",
            "Ensure valid JSON. Escape quotes with backslash. No text outside JSON tags.",
        ])
        
        return "\n".join(prompt_parts)

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history with optimized strategies."""
        if not msg_history:
            return "None"
        
        last_text = msg_history[-1].get("text", "")
        if not last_text:
            return "None"
        
        # Try JSON extraction first
        try:
            extracted = _extract_json_robust(last_text)
            if extracted:
                last_obj = extracted[-1]
                
                # Check priority keys
                for key in ["response", "grade", "score", "evaluation", "verdict", "result"]:
                    if key in last_obj:
                        value = last_obj[key]
                        if isinstance(value, (str, int, float, bool)):
                            return str(value).lower() if isinstance(value, str) else str(value)
                
                # Check for correctness boolean
                if "correct" in last_obj:
                    return "correct" if last_obj["correct"] else "incorrect"
                
                # Check for points
                if "points" in last_obj:
                    return f"points:{last_obj['points']}"
                
                return str(last_obj)
        except Exception:
            pass
        
        # Regex-based extraction
        text_lower = last_text.lower()
        
        # Check for numeric scores
        score_match = re.search(r'(?:score|grade|points?)\s*[:=]\s*(\d+(?:\.\d+)?)', text_lower)
        if score_match:
            return f"score:{score_match.group(1)}"
        
        # Check for explicit verdicts
        verdict_match = re.search(r'"response"\s*:\s*"(correct|incorrect|partial)"', text_lower)
        if verdict_match:
            return verdict_match.group(1)
        
        verdict_match = re.search(r'\banswer\s+is\s+(correct|incorrect|partial|wrong|right)\b', text_lower)
        if verdict_match:
            v = verdict_match.group(1)
            return "correct" if v in ["correct", "right"] else "incorrect" if v in ["incorrect", "wrong"] else v
        
        # Check for correctness boolean
        bool_match = re.search(r'"correct"\s*:\s*(true|false)', text_lower)
        if bool_match:
            return "correct" if bool_match.group(1) == "true" else "incorrect"
        
        # Check for keywords
        if "incorrect" in text_lower or "wrong" in text_lower:
            return "incorrect"
        if "partial" in text_lower:
            return "partial"
        if "correct" in text_lower:
            return "correct"
        
        # Fallback
        stripped = last_text.strip()
        return stripped[:200] if stripped else "None"

    def _normalize_prediction(self, prediction: str) -> str:
        """Normalize prediction to standard format."""
        if not prediction or prediction == "None":
            return "None"
        
        pred_lower = prediction.lower().strip()
        
        # Handle score/points format
        if pred_lower.startswith(("score:", "points:")) or "/" in pred_lower:
            return prediction
        
        # Numeric-only predictions
        if pred_lower.isdigit():
            return f"score:{prediction}"
        
        # Check for negations
        if any(neg in pred_lower for neg in ["not correct", "not right", "not valid"]):
            return "incorrect"
        
        # Check verdicts in priority order
        if any(term in pred_lower for term in ["incorrect", "wrong", "false", "invalid"]):
            return "incorrect"
        
        if any(term in pred_lower for term in ["partial", "partially", "incomplete"]):
            return "partial"
        
        if any(term in pred_lower for term in ["correct", "right", "true", "valid", "accepted"]):
            return "correct"
        
        return prediction

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Validate inputs
        required_keys = ["problem", "solution", "student_answer"]
        missing_keys = [k for k in required_keys if k not in inputs or not inputs[k]]
        if missing_keys:
            error_msg = f"Error: Missing required inputs: {missing_keys}"
            self.log_fn(error_msg)
            return error_msg, [{"role": "assistant", "text": error_msg}]
        
        instruction = self._build_prompt(inputs)
        
        # Log problem preview
        problem_preview = inputs.get("problem", "")[:80].replace('\n', ' ')
        self.log_fn(f"Processing: {problem_preview}...")

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            error_msg = f"Error: LLM call failed: {type(e).__name__}: {e}"
            self.log_fn(error_msg)
            return error_msg, [{"role": "assistant", "text": error_msg}]

        # Extract and normalize prediction
        raw_prediction = self._extract_prediction(msg_history)
        prediction = self._normalize_prediction(raw_prediction)
        
        # Log result
        if prediction == "None":
            self.log_fn("Warning: Could not extract valid prediction")
        else:
            preview = prediction[:80] if len(prediction) > 80 else prediction
            usage_info = info.get('usage', {})
            tokens = usage_info.get('total_tokens', 0)
            self.log_fn(f"Prediction: {preview} (tokens: {tokens})")

        return str(prediction), msg_history
