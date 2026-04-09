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
    """Attempt to repair common JSON syntax errors with improved robustness.
    
    Fixes:
    - Trailing commas before closing braces/brackets
    - Single quotes instead of double quotes
    - Unescaped newlines, tabs, carriage returns in strings
    - Missing closing braces/brackets
    - Unescaped quotes within strings
    - Comments in JSON (// and /* */)
    - Control characters
    - Unicode escape sequences
    """
    # First, try parsing as-is for performance
    try:
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
    
    # Fix single quotes: replace 'key': with "key": and : 'value' with : "value"
    # Use a more robust approach with word boundaries
    repaired = re.sub(r"(?<=[{\[,])\s*'([^']+)'\s*:", r'"\1":', repaired)
    repaired = re.sub(r":\s*'([^']*?)'(?=\s*[,}\]])", r': "\1"', repaired)
    
    # Escape unescaped newlines, tabs, carriage returns in string values
    def escape_special_chars(match):
        content = match.group(1)
        # Escape unescaped special characters
        content = content.replace('\\n', '\x00NEWLINE\x00')  # Temp placeholder
        content = content.replace('\\t', '\x00TAB\x00')
        content = content.replace('\\r', '\x00CR\x00')
        content = content.replace('\\"', '\x00QUOTE\x00')
        
        content = content.replace('\n', '\\n')
        content = content.replace('\t', '\\t')
        content = content.replace('\r', '\\r')
        content = content.replace('"', '\\"')
        
        # Restore already-escaped sequences
        content = content.replace('\x00NEWLINE\x00', '\\n')
        content = content.replace('\x00TAB\x00', '\\t')
        content = content.replace('\x00CR\x00', '\\r')
        content = content.replace('\x00QUOTE\x00', '\\"')
        
        return '"' + content + '"'
    
    # Match string contents more carefully
    repaired = re.sub(r'"((?:[^"\\]|\\.)*?)"', escape_special_chars, repaired)
    
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
    
    # Second attempt: extract the first complete JSON object using state machine
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
                        # Found complete object
                        subset = repaired[start:start+i+1]
                        # Clean up the subset
                        subset = re.sub(r',\s*}', '}', subset)
                        subset = re.sub(r',\s*]', ']', subset)
                        try:
                            return json.loads(subset)
                        except json.JSONDecodeError:
                            # Try one more repair on the subset
                            try:
                                return json.loads(subset.replace("'", '"'))
                            except:
                                return None
        return None
    except Exception:
        return None


def _extract_json_robust(text: str) -> list[dict] | None:
    """Most robust JSON extraction with multiple fallback strategies.
    
    This is the primary extraction function that tries multiple
    strategies in order of reliability, optimized for performance.
    """
    if not text or not text.strip():
        return None
    
    # Strategy 1: <json> tags (original format) - fastest path
    results = _extract_jsons(text)
    if results:
        return results
    
    # Strategy 2: Flexible extraction with repair
    results = _extract_json_flexible(text)
    if results:
        return results
    
    # Strategy 3: Look for any JSON-like structure between outermost braces
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
    
    # Strategy 4: Extract key-value pairs from malformed JSON using regex
    # Last resort: try to extract reasoning and response fields directly
    try:
        # More flexible regex patterns for field extraction
        reasoning_match = re.search(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
        response_match = re.search(r'"response"\s*:\s*"([^"]*)"', text, re.IGNORECASE)
        
        if reasoning_match or response_match:
            extracted = {}
            if reasoning_match:
                reasoning = reasoning_match.group(1)
                # Unescape common escape sequences
                reasoning = reasoning.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r').replace('\\"', '"').replace('\\\\', '\\')
                extracted["reasoning"] = reasoning
            if response_match:
                extracted["response"] = response_match.group(1)
            elif reasoning_match and not response_match:
                # Try to infer response from text patterns
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
        # Cache for similar problems to improve performance
        self._cache: dict[str, tuple[str, list[dict]]] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def _get_cache_key(self, inputs: dict) -> str:
        """Generate a cache key from inputs for similar problem detection.
        
        Uses problem + solution hash to identify similar grading scenarios.
        """
        import hashlib
        # Create key from problem and solution (student answer can vary)
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        key_content = f"{problem}:{solution}"
        return hashlib.md5(key_content.encode()).hexdigest()[:16]
    
    def get_cache_stats(self) -> dict:
        """Return cache statistics for performance monitoring."""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "total": total,
            "hit_rate": hit_rate,
            "cache_size": len(self._cache),
        }
    
    def clear_cache(self) -> None:
        """Clear the cache and reset statistics."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def _build_prompt(self, inputs: dict) -> str:
        """Build an optimized structured prompt for IMO grading with chain-of-thought reasoning."""
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Build prompt with clear structure and explicit instructions
        prompt_parts = [
            f"You are an expert {domain} grader evaluating student solutions to International Mathematical Olympiad (IMO) style competition problems.",
            "",
            "Your task is to carefully analyze the student's answer and provide a rigorous evaluation according to the official solution and grading guidelines.",
            "",
            "## Problem Statement",
            problem,
            "",
            "## Official Solution",
            solution,
        ]
        
        # Only include grading guidelines if they exist
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
            "## Instructions",
            "",
            "Think through this step-by-step:",
            "",
            "1. **Understand the Problem**: What is being asked? What are the key concepts and theorems involved?",
            "",
            "2. **Analyze the Official Solution**: What is the correct approach? What is the final answer? What are the critical steps that must be present?",
            "",
            "3. **Review the Student's Answer**: What approach did the student take? What is their final answer? Did they show all necessary work?",
            "",
            "4. **Compare and Evaluate**: Does the student's answer match the official solution? Consider:",
            "   - Is the final answer numerically/algebraically equivalent to the official solution?",
            "   - Did the student demonstrate correct mathematical reasoning?",
            "   - Are there any logical gaps or errors in the student's work?",
            "   - Did the student use appropriate methods and theorems?",
            "   - Is the solution complete or partial?",
            "",
            "5. **Assign Grade**: Based on your analysis, provide your evaluation.",
            "",
            "## Response Format",
            "",
            "You MUST respond in valid JSON format with exactly this structure:",
            "",
            "<json>",
            '{',
            '    "reasoning": "Your detailed step-by-step analysis here. Include specific observations about what the student did correctly and incorrectly.",',
            '    "response": "Your final evaluation here",',
            '    "confidence": "high|medium|low"',
            '}',
            "</json>",
            "",
            "The 'response' field MUST be exactly one of these values:",
            '- "correct" - if the answer is fully correct with proper reasoning',
            '- "incorrect" - if the answer is wrong or has critical errors',
            '- "partial" - if the answer has some correct elements but is incomplete or has minor errors',
            '- A specific score (e.g., "7" or "3/7") if the problem uses point-based scoring',
            "",
            "The 'confidence' field indicates your certainty:",
            '- "high" - you are very confident in your evaluation',
            '- "medium" - the evaluation is reasonable but there is some ambiguity',
            '- "low" - the evaluation is uncertain, possibly due to unclear student work or ambiguous problem',
            "",
            "IMPORTANT: Ensure your JSON is valid. Escape any quotes within strings with backslash. Do not include any text outside the JSON tags.",
        ])
        
        return "\n".join(prompt_parts)

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history with optimized fallback strategies.
        
        Enhanced for IMO grading: specifically handles correctness evaluation,
        numeric scores, and partial credit scenarios with improved performance.
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
                last_obj = extracted[-1]
                
                # Priority order for response fields
                priority_keys = ["response", "grade", "score", "evaluation", "verdict", "answer", "result", "conclusion"]
                for key in priority_keys:
                    if key in last_obj:
                        value = last_obj[key]
                        if isinstance(value, (str, int, float, bool)):
                            return str(value).lower() if isinstance(value, str) else str(value)
                        elif isinstance(value, (list, dict)):
                            return json.dumps(value)
                
                # Check for correctness boolean
                if "correct" in last_obj:
                    correct_val = last_obj["correct"]
                    if isinstance(correct_val, bool):
                        return "correct" if correct_val else "incorrect"
                    return str(correct_val).lower()
                
                # Check for points/partial credit
                if "points" in last_obj:
                    return f"points:{last_obj['points']}"
                
                # If no known field, return the whole object as string
                return str(last_obj)
        except Exception as e:
            self.log_fn(f"Flexible extraction failed: {e}")
        
        # Fallback 1: Direct regex patterns for common fields
        text_lower = last_text.lower()
        
        # Check for numeric scores first (most specific)
        score_patterns = [
            (r'(?:score|grade|points?)\s*[:=]\s*(\d+(?:\.\d+)?)', 1),
            (r'(\d+)\s*/\s*\d+\s*(?:points?)?', 1),
            (r'(?:awarded|given|assigned)\s+(\d+)\s*(?:points?)?', 1),
            (r'(?:score|grade)\s+of\s+(\d+)', 1),
        ]
        for pattern, group in score_patterns:
            match = re.search(pattern, text_lower)
            if match:
                return f"score:{match.group(group)}"
        
        # Check for explicit verdict patterns (most reliable)
        verdict_patterns = [
            (r'\bthe\s+answer\s+is\s+(correct|incorrect|wrong|right)\b', 1),
            (r'\bverdict\s*[:=]\s*(correct|incorrect|partial)\b', 1),
            (r'\b(correct|incorrect|partial)\s+answer\b', 1),
            (r'\banswer\s+is\s+(correct|incorrect|partial)\b', 1),
            (r'"response"\s*:\s*"(correct|incorrect|partial)"', 1),
            (r'"grade"\s*:\s*"(correct|incorrect|partial)"', 1),
            (r'"evaluation"\s*:\s*"(correct|incorrect|partial)"', 1),
            (r'"verdict"\s*:\s*"(correct|incorrect|partial)"', 1),
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
        
        # Check for correctness boolean in JSON
        bool_match = re.search(r'"correct"\s*:\s*(true|false)', text_lower)
        if bool_match:
            return "correct" if bool_match.group(1) == "true" else "incorrect"
        
        # Check for correctness indicators with negation handling
        has_correct = "correct" in text_lower
        has_incorrect = "incorrect" in text_lower or "not correct" in text_lower or "wrong" in text_lower
        has_partial = any(term in text_lower for term in ["partial", "partially", "some credit", "incomplete", "partial credit"])
        
        if has_incorrect:
            return "incorrect"
        elif has_partial:
            return "partial"
        elif has_correct:
            return "correct"
        
        # Fallback: return first 200 chars of text as prediction
        stripped = last_text.strip()
        if stripped:
            return stripped[:200]
        
        return "None"

    def _normalize_prediction(self, prediction: str) -> str:
        """Normalize prediction to standard format for evaluation.
        
        Converts various forms of correct/incorrect/partial to standardized values.
        Handles edge cases like negations and compound statements.
        """
        if not prediction or prediction == "None":
            return "None"
        
        pred_lower = prediction.lower().strip()
        
        # Handle score format (e.g., "score:7", "7/7", "points:5")
        if pred_lower.startswith(("score:", "points:")) or "/" in pred_lower:
            return prediction
        
        # Check for numeric-only predictions (assume score)
        if pred_lower.isdigit():
            return f"score:{prediction}"
        
        # Check for negations first (e.g., "not correct", "not right")
        negation_patterns = ["not correct", "not right", "not valid", "not accepted", "not true"]
        if any(neg in pred_lower for neg in negation_patterns):
            return "incorrect"
        
        # Check for incorrect/wrong first (more specific than correct)
        incorrect_terms = ["incorrect", "wrong", "false", "invalid", "rejected", "error", "mistake"]
        if any(term in pred_lower for term in incorrect_terms):
            return "incorrect"
        
        # Check for partial credit
        partial_terms = ["partial", "partially", "incomplete", "some credit", "partially correct"]
        if any(term in pred_lower for term in partial_terms):
            return "partial"
        
        # Check for correct/valid
        correct_terms = ["correct", "right", "true", "valid", "accepted", "accurate"]
        if any(term in pred_lower for term in correct_terms):
            return "correct"
        
        # Return original if no normalization applied
        return prediction

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with improved error handling and caching.

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
        
        # Validate input types
        for key in required_keys:
            if not isinstance(inputs.get(key), str):
                error_msg = f"Error: Input '{key}' must be a string"
                self.log_fn(error_msg)
                return error_msg, [{"role": "assistant", "text": error_msg}]
        
        # Check cache for similar problems (same problem/solution, different student answer)
        cache_key = self._get_cache_key(inputs)
        if cache_key in self._cache:
            self._cache_hits += 1
            cached_prediction, cached_history = self._cache[cache_key]
            self.log_fn(f"Cache hit! (hits: {self._cache_hits}, misses: {self._cache_misses})")
            # Still need to evaluate this specific student answer, but we can use cached reasoning
            # For now, return cached result as a hint but still make fresh call for accuracy
        
        self._cache_misses += 1
        
        instruction = self._build_prompt(inputs)
        
        # Log the problem being processed (truncated for privacy)
        problem_preview = inputs.get("problem", "")[:100].replace('\n', ' ')
        self.log_fn(f"Processing problem: {problem_preview}...")

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
            # Log successful extraction with usage info if available
            preview = prediction[:100] if len(prediction) > 100 else prediction
            usage_info = info.get('usage', {})
            if usage_info:
                tokens = usage_info.get('total_tokens', usage_info.get('prompt_tokens', 0) + usage_info.get('completion_tokens', 0))
                self.log_fn(f"Extracted prediction: {preview} (tokens: {tokens})")
            else:
                self.log_fn(f"Extracted prediction: {preview}")
        
        # Cache the result for potential reuse
        self._cache[cache_key] = (str(prediction), msg_history)
        
        # Limit cache size to prevent memory issues
        if len(self._cache) > 1000:
            # Remove oldest entries (simple FIFO)
            oldest_keys = list(self._cache.keys())[:100]
            for key in oldest_keys:
                del self._cache[key]

        return str(prediction), msg_history
