"""
Task agent: solves a given task with a single LLM call.

Reimplemented from facebookresearch/HyperAgents task_agent.py.
Same interface, same JSON output format, same extraction logic.

This is the INITIAL task agent. The meta agent modifies this file
during self-improvement. The evaluation harness loads whatever
task_agent.py exists at the agent's repo path.

Enhanced with:
- Better prompt management with templates
- Improved response extraction and validation
- Response quality scoring
- Better error context
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)

# Required input fields for grading tasks
REQUIRED_INPUT_FIELDS = {"domain", "problem", "solution", "grading_guidelines", "student_answer"}

# Prompt template for grading tasks
GRADING_PROMPT_TEMPLATE = """You are an expert grading agent. Analyze the student answer carefully.

Task Information:
- Domain: {domain}
- Problem: {problem}

Reference Solution:
{solution}

Grading Guidelines:
{grading_guidelines}

Student Answer to Evaluate:
{student_answer}

Instructions:
1. Compare the student answer against the reference solution
2. Apply the grading guidelines strictly
3. Provide a detailed evaluation explaining your reasoning
4. Be fair but rigorous in your assessment

Respond ONLY in JSON format with the following schema:
<json>
{{
    "response": "Your detailed evaluation here. Include specific points about what was correct, what was incorrect, and why."
}}
</json>"""

# Alternative extraction patterns for robustness
EXTRACTION_PATTERNS = [
    # Primary: <json>...</json> blocks
    (r'<json>\s*(.*?)\s*</json>', 'json_block'),
    # Secondary: ```json code blocks
    (r'```(?:json)?\s*\n?(.*?)\n?```', 'code_block'),
    # Tertiary: Raw JSON objects
    (r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', 'raw_json'),
]


def _build_grading_prompt(inputs: dict) -> str:
    """Build a structured grading prompt from inputs.
    
    Args:
        inputs: Dictionary with domain, problem, solution, grading_guidelines, student_answer
    
    Returns:
        Formatted prompt string
    """
    return GRADING_PROMPT_TEMPLATE.format(
        domain=inputs.get("domain", "Unknown"),
        problem=inputs.get("problem", ""),
        solution=inputs.get("solution", ""),
        grading_guidelines=inputs.get("grading_guidelines", ""),
        student_answer=inputs.get("student_answer", ""),
    )


def _score_response_quality(prediction: str) -> dict[str, Any]:
    """Score the quality of an extracted response.
    
    Returns:
        Dictionary with quality metrics
    """
    scores = {
        "length": len(prediction),
        "has_content": bool(prediction.strip()),
        "word_count": len(prediction.split()),
        "has_evaluation_keywords": any(
            kw in prediction.lower() 
            for kw in ["correct", "incorrect", "error", "mistake", "good", "wrong", "right"]
        ),
        "has_reasoning": any(
            kw in prediction.lower()
            for kw in ["because", "since", "therefore", "thus", "reason", "explanation"]
        ),
    }
    
    # Overall quality score (0-100)
    score = 0
    if scores["has_content"]:
        score += 20
    if scores["word_count"] >= 10:
        score += 20
    if scores["has_evaluation_keywords"]:
        score += 30
    if scores["has_reasoning"]:
        score += 30
    
    scores["overall"] = score
    return scores


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


def _extract_json_fallback(text: str) -> dict | None:
    """Fallback JSON extraction using regex patterns for common formats.
    
    Attempts to extract JSON from code blocks or raw JSON objects.
    """
    # Try to extract from markdown code blocks
    code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue
    
    # Try to find raw JSON objects
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, text, re.DOTALL)
    for match in matches:
        try:
            parsed = json.loads(match)
            if "response" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue
    
    return None


def _validate_inputs(inputs: dict) -> tuple[bool, str]:
    """Validate that all required input fields are present.
    
    Returns:
        (is_valid, error_message)
    """
    if not isinstance(inputs, dict):
        return False, f"Expected dict, got {type(inputs).__name__}"
    
    missing = REQUIRED_INPUT_FIELDS - set(inputs.keys())
    if missing:
        return False, f"Missing required fields: {sorted(missing)}"
    
    # Check for empty values
    empty_fields = [k for k in REQUIRED_INPUT_FIELDS if not str(inputs.get(k, "")).strip()]
    if empty_fields:
        return False, f"Empty required fields: {sorted(empty_fields)}"
    
    return True, ""


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
            "quality_scores": [],
        }

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        self.stats["total_calls"] += 1
        
        # Validate inputs
        is_valid, error_msg = _validate_inputs(inputs)
        if not is_valid:
            self.log_fn(f"Input validation failed: {error_msg}")
            self.stats["validation_errors"] += 1
            return f"Error: {error_msg}", []
        
        # Build structured prompt using template
        instruction = _build_grading_prompt(inputs)

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
                
                if not last_message or not last_message.strip():
                    self.log_fn("Empty response from LLM")
                    self.stats["failed"] += 1
                    return "Error: Empty response from LLM", msg_history
                
                # Primary extraction method
                extracted = _extract_jsons(last_message)
                if extracted:
                    # Try to find response in any of the extracted JSONs
                    for item in reversed(extracted):
                        if isinstance(item, dict) and "response" in item:
                            prediction = item["response"]
                            extraction_method = "primary"
                            self.stats["json_extracted"] += 1
                            break
                
                if extraction_method == "none":
                    # Fallback extraction
                    fallback = _extract_json_fallback(last_message)
                    if fallback and isinstance(fallback, dict) and "response" in fallback:
                        prediction = fallback["response"]
                        extraction_method = "fallback"
                        self.stats["fallback_used"] += 1
                        self.log_fn(f"Used fallback extraction for response")
                
                if extraction_method == "none":
                    # Last resort: use raw text (cleaned)
                    cleaned_text = last_message.strip()
                    # Remove common markdown artifacts
                    cleaned_text = re.sub(r'^```[\w]*\n?', '', cleaned_text)
                    cleaned_text = re.sub(r'\n?```$', '', cleaned_text)
                    prediction = cleaned_text[:1000]  # Limit length but allow more context
                    extraction_method = "raw"
                    self.stats["raw_extracted"] += 1
                    self.log_fn(f"Using raw text extraction (limited to 1000 chars)")
                        
                self.log_fn(f"Extraction method: {extraction_method}, prediction length: {len(str(prediction))}")
                
                # Validate prediction is not empty
                if not str(prediction).strip():
                    self.log_fn("Extracted prediction is empty")
                    self.stats["failed"] += 1
                    return "Error: Empty prediction extracted", msg_history
                
                # Score response quality
                quality = _score_response_quality(str(prediction))
                self.stats["quality_scores"].append(quality["overall"])
                self.log_fn(f"Response quality score: {quality['overall']}/100")
                
                # Warn if quality is low
                if quality["overall"] < 50:
                    self.log_fn(f"Warning: Low quality response detected (score: {quality['overall']})")
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            self.stats["failed"] += 1
            return f"Error: Extraction failed - {e}", msg_history

        return str(prediction), msg_history
    
    def get_stats(self) -> dict[str, Any]:
        """Return extraction statistics."""
        stats = self.stats.copy()
        if stats["quality_scores"]:
            stats["avg_quality_score"] = sum(stats["quality_scores"]) / len(stats["quality_scores"])
        else:
            stats["avg_quality_score"] = 0
        return stats
