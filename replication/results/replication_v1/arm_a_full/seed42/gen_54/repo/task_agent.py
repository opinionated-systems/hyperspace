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


def _validate_and_clean_response(text: str) -> str:
    """Validate and clean the extracted response text.
    
    Removes common artifacts and ensures the response is meaningful.
    
    Args:
        text: The raw extracted response text
        
    Returns:
        Cleaned and validated response text
    """
    if not text or not isinstance(text, str):
        return "Error: Invalid response extracted"
    
    # Strip whitespace and common wrapper characters
    text = text.strip()
    
    # Remove common JSON artifacts that might remain
    artifacts = [
        ('"response":', ''),
        ('"response" :', ''),
        ('"response": ', ''),
        ('"response" : ', ''),
    ]
    for artifact, replacement in artifacts:
        if text.startswith(artifact):
            text = text[len(artifact):].strip()
    
    # Remove surrounding quotes if present
    if (text.startswith('"') and text.endswith('"')) or \
       (text.startswith("'") and text.endswith("'")):
        text = text[1:-1]
    
    # Unescape common JSON escape sequences
    text = text.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"').replace("\\'", "'")
    
    # Ensure minimum meaningful content
    if len(text.strip()) < 10:
        return f"Error: Response too short or empty (length: {len(text.strip())})"
    
    return text.strip()


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
        "partially correct", "full credit", "partial credit",
        "error", "mistake", "missing", "omitted"
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
        # Bonus for paragraphs with specific grading language
        if any(phrase in para_lower for phrase in ["the student", "student's", "this answer"]):
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


def _extract_score_from_text(text: str) -> int | None:
    """Extract numerical score from text if present.
    
    Looks for common grading patterns like "Score: X", "Grade: X/Y", etc.
    
    Args:
        text: The text to search for a score
        
    Returns:
        The extracted score as an integer, or None if not found
    """
    import re
    
    # Common score patterns
    patterns = [
        r'[Ss]core\s*[:=]\s*(\d+)',
        r'[Gg]rade\s*[:=]\s*(\d+)',
        r'(\d+)\s*out of\s*\d+',
        r'(\d+)\s*/\s*\d+',
        r'[Ee]arned\s*(\d+)\s*points?',
        r'[Aa]warded\s*(\d+)\s*points?',
        r'(\d+)\s*points?\s*(?:earned|awarded|given)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return int(match.group(1))
            except (ValueError, IndexError):
                continue
    
    return None


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
        
        # Extract key fields for better prompt construction
        domain = inputs.get("domain", "Unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        instruction = f"""You are an expert grading agent for {domain} problems. Your task is to evaluate a student's answer against the correct solution and grading guidelines.

## Problem Statement:
{problem[:2000] if len(problem) > 2000 else problem}

## Correct Solution:
{solution[:2000] if len(solution) > 2000 else solution}

## Grading Guidelines:
{grading_guidelines[:1500] if len(grading_guidelines) > 1500 else grading_guidelines}

## Student's Answer:
{student_answer[:2000] if len(student_answer) > 2000 else student_answer}

## Your Task:
1. Carefully compare the student's answer with the correct solution
2. Check if the student followed the grading guidelines
3. Identify any errors, omissions, or misconceptions
4. Provide a detailed evaluation with specific feedback

## Evaluation Structure (include in your response):
- **Correctness**: State whether the answer is correct, partially correct, or incorrect
- **Score**: If applicable, state the score (e.g., "Score: 7/10" or "Grade: 85%")
- **Key Issues**: List specific errors or omissions found
- **Positive Aspects**: Acknowledge what the student did correctly
- **Improvement Suggestions**: Provide actionable feedback for the student

Respond ONLY in JSON format with the following schema:
<json>
{{
    "response": "Your detailed evaluation here. Include: (1) whether the answer is correct/partially correct/incorrect, (2) specific points earned/lost with reasoning, (3) constructive feedback for the student."
}}
</json>

IMPORTANT: 
- Ensure your response is valid JSON with properly escaped quotes and newlines
- Use \\\\n for newlines within the JSON string
- Use \\\" for quotes within the JSON string
- The response field should contain your complete evaluation"""

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
                    prediction = _validate_and_clean_response(extracted[-1]["response"])
                    extraction_method = "primary"
                    self.stats["json_extracted"] += 1
                else:
                    # Fallback extraction
                    fallback = _extract_json_fallback(last_message)
                    if fallback and "response" in fallback:
                        prediction = _validate_and_clean_response(fallback["response"])
                        extraction_method = "fallback"
                        self.stats["fallback_used"] += 1
                        self.log_fn(f"Used fallback extraction for response")
                    else:
                        # Last resort: use raw text with intelligent truncation
                        # Try to find the most relevant part of the response
                        raw_text = _extract_relevant_text(last_message)
                        prediction = _validate_and_clean_response(raw_text)
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
