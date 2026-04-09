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

# Response quality thresholds
MIN_RESPONSE_LENGTH = 50
MAX_RESPONSE_LENGTH = 8000


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
        "partial credit", "full credit", "points awarded",
        "the student", "student's", "work shown", "final answer"
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
        # Favor later paragraphs (conclusions often at end)
        score += i * 0.1
        # Bonus for paragraphs with numbers (likely scores)
        if re.search(r'\d+', para):
            score += 0.5
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


def _validate_response_quality(response: str) -> tuple[bool, str, str]:
    """Validate the quality of an extracted response.
    
    Args:
        response: The extracted response text
        
    Returns:
        Tuple of (is_valid, quality_level, reason)
        - is_valid: Whether the response meets minimum quality standards
        - quality_level: 'high', 'medium', or 'low'
        - reason: Explanation of the quality assessment
    """
    if not response or not isinstance(response, str):
        return False, "low", "Empty or invalid response type"
    
    response = response.strip()
    length = len(response)
    
    # Check minimum length
    if length < MIN_RESPONSE_LENGTH:
        return False, "low", f"Response too short ({length} chars, min: {MIN_RESPONSE_LENGTH})"
    
    # Check for common error patterns
    error_patterns = [
        r"^\s*error\s*[:\-]",
        r"^\s*failed\s*[:\-]",
        r"^\s*none\s*$",
        r"^\s*null\s*$",
        r"^\s*undefined\s*$",
    ]
    for pattern in error_patterns:
        if re.search(pattern, response, re.IGNORECASE):
            return False, "low", f"Response matches error pattern: {pattern}"
    
    # Quality scoring
    quality_score = 0
    quality_indicators = []
    
    # Length scoring
    if length >= 500:
        quality_score += 2
        quality_indicators.append("good_length")
    elif length >= 200:
        quality_score += 1
        quality_indicators.append("adequate_length")
    
    # Content quality indicators
    if re.search(r'\d+', response):  # Contains numbers (likely scores)
        quality_score += 1
        quality_indicators.append("contains_numbers")
    
    if re.search(r'[.!?].{3,}[.!?]', response):  # Multiple sentences
        quality_score += 1
        quality_indicators.append("multiple_sentences")
    
    # Check for evaluation-specific content
    eval_terms = ['grade', 'score', 'point', 'correct', 'incorrect', 'feedback', 'assessment']
    eval_count = sum(1 for term in eval_terms if term in response.lower())
    if eval_count >= 2:
        quality_score += 2
        quality_indicators.append("evaluation_terms")
    elif eval_count >= 1:
        quality_score += 1
        quality_indicators.append("some_evaluation_terms")
    
    # Determine quality level
    if quality_score >= 4:
        quality_level = "high"
    elif quality_score >= 2:
        quality_level = "medium"
    else:
        quality_level = "low"
    
    # Truncate if too long
    if length > MAX_RESPONSE_LENGTH:
        response = response[:MAX_RESPONSE_LENGTH] + "... [truncated]"
        quality_indicators.append("truncated")
    
    is_valid = quality_level in ("high", "medium")
    reason = f"Score: {quality_score}/6, indicators: {', '.join(quality_indicators)}"
    
    return is_valid, quality_level, reason


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
            "quality_high": 0,
            "quality_medium": 0,
            "quality_low": 0,
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
        
        instruction = f"""You are an expert grading agent. Analyze the student answer carefully.

Task input:
```
{json.dumps(inputs, indent=2)}
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
                
                # Validate response quality
                is_valid, quality_level, quality_reason = _validate_response_quality(str(prediction))
                self.stats[f"quality_{quality_level}"] += 1
                
                if not is_valid:
                    self.log_fn(f"Low quality response detected: {quality_reason}")
                    # Try to extract better content from the full message
                    if len(last_message) > len(str(prediction)):
                        alternative = _extract_relevant_text(last_message, max_length=3000)
                        alt_valid, alt_quality, alt_reason = _validate_response_quality(alternative)
                        if alt_quality != "low":
                            prediction = alternative
                            self.log_fn(f"Switched to alternative extraction with quality: {alt_quality}")
                        else:
                            # Keep original but log the issue
                            self.log_fn(f"Keeping original response despite quality issues")
                else:
                    self.log_fn(f"Response quality: {quality_level} ({quality_reason})")
                        
                self.log_fn(f"Extraction method: {extraction_method}, prediction length: {len(str(prediction))}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            self.stats["failed"] += 1

        # Track response time
        elapsed = time.time() - start_time
        self._response_times.append(elapsed)
        self.stats["avg_response_time"] = sum(self._response_times) / len(self._response_times)
        self.log_fn(f"Response time: {elapsed:.2f}s (avg: {self.stats['avg_response_time']:.2f}s)")

        return str(prediction), msg_history
    
    def get_stats(self) -> dict[str, Any]:
        """Return extraction statistics."""
        return self.stats.copy()
