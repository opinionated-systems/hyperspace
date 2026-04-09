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
from typing import Any

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
        except json.JSONDecodeError as e:
            logger.debug(f"JSON decode error: {e}, content: {inner[:100]}...")
            continue
    return results or None


def _extract_relevant_text(text: str, max_length: int = 2000) -> str:
    """Extract the most relevant part of text for evaluation.
    
    Tries to find evaluation-related content and returns a meaningful
    portion of the text, prioritizing the end where conclusions often are.
    Enhanced to better handle code blocks, structured content, and edge cases.
    
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
        "reasoning", "explanation", "justification", "rationale"
    ]
    
    # High-priority patterns that indicate final answers or conclusions
    conclusion_patterns = [
        r"(?:final|overall|total)\s+(?:score|grade|mark|evaluation)",
        r"(?:in\s+conclusion|to\s+summarize|overall)",
        r"(?:the\s+student|student\s+(?:should|earns?|receives?))",
        r"(?:points?|marks?)\s*(?:awarded?|given|assigned)",
    ]
    
    # Split into paragraphs and score them
    paragraphs = text.split('\n\n')
    scored_paragraphs = []
    
    for i, para in enumerate(paragraphs):
        score = 0
        para_lower = para.lower()
        para_stripped = para.strip()
        
        # Score based on keyword presence
        for keyword in eval_keywords:
            if keyword in para_lower:
                score += 1
        
        # Bonus for conclusion patterns
        for pattern in conclusion_patterns:
            if re.search(pattern, para_lower):
                score += 3
        
        # Bonus for paragraphs containing numbers (likely scores)
        if re.search(r'\d+\s*(?:/|out\s+of|points?)', para_lower):
            score += 2
        
        # Bonus for structured content (bullet points, numbered lists)
        if re.search(r'^(?:\d+\.\s+|[-*]\s+)', para_stripped, re.MULTILINE):
            score += 1
        
        # Favor later paragraphs (conclusions often at end)
        score += i * 0.15
        
        # Penalty for very short paragraphs (likely noise)
        if len(para_stripped) < 20:
            score -= 1
        
        # Penalty for code blocks unless they contain evaluation keywords
        if para_stripped.startswith('```') or para_stripped.startswith('    '):
            has_eval_keyword = any(kw in para_lower for kw in eval_keywords)
            if not has_eval_keyword:
                score -= 2
        
        scored_paragraphs.append((score, i, para))
    
    # Sort by score descending, but preserve original order for same scores
    scored_paragraphs.sort(key=lambda x: (-x[0], x[1]))
    
    # Build result from highest scoring paragraphs
    result_parts = []
    current_length = 0
    added_indices = set()
    
    # Always include the last paragraph if it looks like a conclusion
    if paragraphs:
        last_para = paragraphs[-1]
        last_para_lower = last_para.lower()
        # Check if last paragraph is meaningful (not just whitespace or code)
        if len(last_para.strip()) > 30:
            # Check if it contains evaluation-related content
            has_conclusion = any(re.search(p, last_para_lower) for p in conclusion_patterns)
            has_keyword = any(kw in last_para_lower for kw in eval_keywords)
            
            if has_conclusion or has_keyword or len(paragraphs) <= 3:
                if len(last_para) <= max_length // 3:
                    result_parts.append(last_para)
                    added_indices.add(len(paragraphs) - 1)
                    current_length += len(last_para) + 2
    
    # Add high-scoring paragraphs
    for score, idx, para in scored_paragraphs:
        if idx in added_indices:
            continue
        if current_length + len(para) + 2 <= max_length:
            result_parts.append(para)
            added_indices.add(idx)
            current_length += len(para) + 2
    
    # Sort result parts by original index to maintain logical flow
    result_parts_with_idx = [(i, para) for i, para in enumerate(paragraphs) 
                              if para in result_parts]
    result_parts_with_idx.sort(key=lambda x: x[0])
    
    if result_parts_with_idx:
        ordered_parts = [para for _, para in result_parts_with_idx]
        return '\n\n'.join(ordered_parts)
    
    # Fallback: return end portion of text with smart truncation
    # Try to find a good break point (end of sentence or paragraph)
    end_text = text[-(max_length-3):]
    # Look for sentence boundaries
    sentence_end = re.search(r'[.!?]\s+', end_text[:max_length//2])
    if sentence_end and len(end_text) > max_length // 2:
        # Start from a sentence boundary if possible
        start_pos = sentence_end.end()
        return "..." + end_text[start_pos:].strip()
    
    return "..." + end_text


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
        self.stats = {"total_calls": 0, "json_extracted": 0, "fallback_used": 0, "failed": 0}

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        self.stats["total_calls"] += 1
        
        # Validate inputs
        if not isinstance(inputs, dict):
            self.log_fn(f"Invalid inputs type: {type(inputs).__name__}")
            self.stats["failed"] += 1
            return f"Error: Invalid inputs type - expected dict, got {type(inputs).__name__}", []
        
        # Check for required fields
        required_fields = ["domain", "problem", "solution", "grading_guidelines", "student_answer"]
        missing_fields = [f for f in required_fields if f not in inputs]
        if missing_fields:
            self.log_fn(f"Missing required fields: {missing_fields}")
            # Continue anyway - some fields might be optional
        
        # Sanitize inputs to prevent JSON serialization issues
        sanitized_inputs = {}
        for key, value in inputs.items():
            if isinstance(value, str):
                # Truncate very long strings to prevent token limit issues
                if len(value) > 50000:
                    sanitized_inputs[key] = value[:50000] + "... [truncated]"
                else:
                    sanitized_inputs[key] = value
            else:
                sanitized_inputs[key] = value
        
        instruction = f"""You are an expert grading agent. Analyze the student answer carefully.

Task input:
```
{json.dumps(sanitized_inputs, indent=2, default=str)}
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
                        self.log_fn(f"Using raw text extraction (length: {len(str(prediction))})")
                        
                self.log_fn(f"Extraction method: {extraction_method}, prediction length: {len(str(prediction))}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            self.stats["failed"] += 1

        return str(prediction), msg_history
    
    def get_stats(self) -> dict[str, Any]:
        """Return extraction statistics."""
        return self.stats.copy()
