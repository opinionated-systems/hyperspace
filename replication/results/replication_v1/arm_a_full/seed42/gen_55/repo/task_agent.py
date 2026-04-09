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
import time
from typing import Any

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)

# Maximum input size to prevent memory issues
MAX_INPUT_SIZE = 100_000  # characters
# Maximum response length
MAX_RESPONSE_LENGTH = 50_000  # characters
# Required input fields
REQUIRED_FIELDS = {"domain", "problem", "solution", "grading_guidelines", "student_answer"}


def _validate_inputs(inputs: dict) -> tuple[bool, str]:
    """Validate task inputs for required fields and size limits.
    
    Args:
        inputs: Dictionary containing task inputs
        
    Returns:
        (is_valid, error_message)
    """
    # Check required fields
    missing = REQUIRED_FIELDS - set(inputs.keys())
    if missing:
        return False, f"Missing required fields: {', '.join(missing)}"
    
    # Check for None or empty values
    empty_fields = [k for k in REQUIRED_FIELDS if not inputs.get(k)]
    if empty_fields:
        return False, f"Empty required fields: {', '.join(empty_fields)}"
    
    # Check input size
    total_size = sum(len(str(v)) for v in inputs.values() if isinstance(v, str))
    if total_size > MAX_INPUT_SIZE:
        return False, f"Input too large ({total_size} chars, max: {MAX_INPUT_SIZE})"
    
    return True, ""


def _preprocess_inputs(inputs: dict) -> dict:
    """Preprocess and sanitize task inputs.
    
    Args:
        inputs: Raw task inputs
        
    Returns:
        Preprocessed inputs
    """
    processed = {}
    for key, value in inputs.items():
        if isinstance(value, str):
            # Normalize whitespace
            value = value.strip()
            # Remove excessive newlines (more than 3 consecutive)
            value = re.sub(r'\n{4,}', '\n\n\n', value)
            # Truncate very long individual fields
            if len(value) > MAX_INPUT_SIZE // 5:  # 20KB per field
                value = value[:MAX_INPUT_SIZE // 5] + "\n... [truncated due to length]"
        processed[key] = value
    return processed


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
        except json.JSONDecodeError as e:
            logger.debug(f"JSON decode error: {e}, content: {inner[:100]}...")
            continue
    return results or None


def _extract_relevant_text(text: str, max_length: int = 2000) -> str:
    """Extract the most relevant part of text for evaluation.
    
    Tries to find evaluation-related content and returns a meaningful
    portion of the text, prioritizing the end where conclusions often are.
    
    Args:
        text: The raw text to extract from
        max_length: Maximum length of extracted text
        
    Returns:
        The most relevant portion of text
    """
    if len(text) <= max_length:
        return text
    
    # Look for evaluation-related keywords to find relevant sections
    eval_keywords = [
        "evaluation", "grade", "score", "correct", "incorrect", 
        "answer", "solution", "student", "point", "mark",
        "feedback", "assessment", "analysis", "conclusion",
        "partial credit", "full credit", "no credit", "points awarded",
        "explanation", "reasoning", "justification", "therefore",
        "in conclusion", "to summarize", "final answer"
    ]
    
    # Also look for numerical patterns that might indicate scoring
    score_patterns = [
        r'\d+\s*/\s*\d+',  # e.g., "5/10"
        r'\d+\s*points?',  # e.g., "5 points"
        r'score[\s:]+\d+',  # e.g., "score: 5"
        r'grade[\s:]+[A-Fa-f0-9]',  # e.g., "grade: A"
    ]
    
    # Split into paragraphs and score them
    paragraphs = text.split('\n\n')
    scored_paragraphs = []
    
    for i, para in enumerate(paragraphs):
        score = 0
        para_lower = para.lower()
        # Score based on keyword presence
        for keyword in eval_keywords:
            if keyword in para_lower:
                score += 1
        # Score based on regex patterns
        for pattern in score_patterns:
            if re.search(pattern, para, re.IGNORECASE):
                score += 2  # Higher weight for explicit scoring patterns
        # Favor later paragraphs (conclusions often at end)
        score += i * 0.1
        scored_paragraphs.append((score, i, para))
    
    # Sort by score descending
    scored_paragraphs.sort(reverse=True)
    
    # Build result from highest scoring paragraphs
    result_parts = []
    current_length = 0
    
    # Always include the last paragraph (often contains conclusion)
    if paragraphs:
        last_para = paragraphs[-1]
        if len(last_para) <= max_length // 2:
            result_parts.append(last_para)
            current_length += len(last_para) + 2
    
    # Add high-scoring paragraphs
    for score, idx, para in scored_paragraphs:
        if current_length + len(para) + 2 <= max_length:
            if para not in result_parts:
                result_parts.append(para)
                current_length += len(para) + 2
    
    if result_parts:
        return '\n\n'.join(result_parts)
    
    # Fallback: return end portion of text (often most relevant)
    return "..." + text[-(max_length-3):]


def _extract_json_fallback(text: str) -> dict | None:
    """Fallback JSON extraction using regex patterns for common formats.
    
    Attempts to extract JSON from code blocks or raw JSON objects.
    Enhanced to handle nested braces and common LLM output patterns.
    """
    # Try to extract from markdown code blocks (various formats)
    code_block_patterns = [
        r'```(?:json)?\s*\n?(.*?)\n?```',  # Standard markdown
        r'`\s*(\{.*?\})\s*`',  # Inline code with braces
    ]
    for pattern in code_block_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                parsed = json.loads(match.strip())
                if "response" in parsed:
                    return parsed
            except json.JSONDecodeError:
                continue
    
    # Try to find raw JSON objects with improved pattern for nested braces
    # This pattern handles up to 3 levels of nesting
    json_patterns = [
        r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}',  # Nested braces
        r'\{[^{}]*\}',  # Simple objects
    ]
    for pattern in json_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                parsed = json.loads(match)
                if "response" in parsed:
                    return parsed
            except json.JSONDecodeError:
                continue
    
    # Last resort: try to find anything that looks like a response field
    response_pattern = r'"response"\s*:\s*"([^"]*)"'
    match = re.search(response_pattern, text, re.DOTALL)
    if match:
        return {"response": match.group(1)}
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced extraction."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.stats = {
            "total_calls": 0, 
            "json_extracted": 0, 
            "fallback_used": 0, 
            "raw_extracted": 0,
            "failed": 0,
            "validation_errors": 0,
            "avg_response_time": 0.0,
        }
        self._response_times: list[float] = []

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        self.stats["total_calls"] += 1
        start_time = time.time()
        
        # Validate inputs
        is_valid, error_msg = _validate_inputs(inputs)
        if not is_valid:
            self.log_fn(f"Input validation failed: {error_msg}")
            self.stats["validation_errors"] += 1
            return f"Error: {error_msg}", []
        
        # Preprocess inputs
        processed_inputs = _preprocess_inputs(inputs)
        
        instruction = f"""You are an expert grading agent. Analyze the student answer carefully.

Task input:
```
{json.dumps(processed_inputs, indent=2)}
```

Respond ONLY in JSON format with the following schema:
<json>
{{
    "response": "Your detailed evaluation here"
}}
</json>"""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"LLM call failed: {e}")
            self.stats["failed"] += 1
            self._update_response_time(time.time() - start_time)
            return "Error: LLM call failed", []

        # Extract prediction from JSON
        prediction = "None"
        extraction_method = "none"
        
        try:
            if msg_history and len(msg_history) > 0:
                last_message = msg_history[-1].get("text", "")
                
                # Primary extraction method
                extracted = _extract_jsons(last_message)
                if extracted and "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                    extraction_method = "primary"
                    self.stats["json_extracted"] += 1
                else:
                    # Fallback extraction
                    fallback = _extract_json_fallback(last_message)
                    if fallback and "response" in fallback:
                        prediction = fallback["response"]
                        extraction_method = "fallback"
                        self.stats["fallback_used"] += 1
                        self.log_fn(f"Used fallback extraction for response")
                    else:
                        # Last resort: use raw text with intelligent truncation
                        # Try to find the most relevant part of the response
                        prediction = _extract_relevant_text(last_message)
                        extraction_method = "raw"
                        self.stats["raw_extracted"] += 1
                        self.log_fn(f"Using raw text extraction (length: {len(str(prediction))})")
                
                # Truncate if too long
                if len(str(prediction)) > MAX_RESPONSE_LENGTH:
                    prediction = str(prediction)[:MAX_RESPONSE_LENGTH] + "\n... [truncated due to length]"
                        
                self.log_fn(f"Extraction method: {extraction_method}, prediction length: {len(str(prediction))}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            self.stats["failed"] += 1

        self._update_response_time(time.time() - start_time)
        return str(prediction), msg_history
    
    def _update_response_time(self, elapsed: float) -> None:
        """Update average response time tracking."""
        self._response_times.append(elapsed)
        # Keep last 100 measurements
        if len(self._response_times) > 100:
            self._response_times.pop(0)
        self.stats["avg_response_time"] = sum(self._response_times) / len(self._response_times)
    
    def get_stats(self) -> dict[str, Any]:
        """Return extraction statistics."""
        stats = self.stats.copy()
        # Calculate success rate
        if stats["total_calls"] > 0:
            stats["success_rate"] = round(
                (stats["total_calls"] - stats["failed"] - stats["validation_errors"]) / stats["total_calls"] * 100, 2
            )
            stats["extraction_distribution"] = {
                "primary": stats["json_extracted"],
                "fallback": stats["fallback_used"],
                "raw": stats["raw_extracted"],
            }
        return stats
