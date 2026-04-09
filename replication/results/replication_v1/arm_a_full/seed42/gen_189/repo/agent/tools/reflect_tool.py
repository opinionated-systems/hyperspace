"""
Reflection tool: enables self-analysis and improvement tracking.

This tool allows the agent to reflect on its grading decisions,
identify patterns in errors, and suggest improvements to the grading process.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def tool_info() -> dict:
    return {
        "name": "reflect",
        "description": "Analyze grading decisions and track patterns for self-improvement. Stores reflection data for meta-learning.",
        "input_schema": {
            "type": "object",
            "properties": {
                "problem_id": {
                    "type": "string",
                    "description": "Unique identifier for the problem being graded"
                },
                "predicted_label": {
                    "type": "string",
                    "description": "The label assigned by the agent (correct/almost/partial/incorrect)"
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence score between 0.0 and 1.0"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of why this label was chosen"
                },
                "difficulty_factors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of factors that made this problem difficult to grade"
                },
                "suggested_improvement": {
                    "type": "string",
                    "description": "Suggestion for how grading could be improved"
                }
            },
            "required": ["problem_id", "predicted_label"]
        }
    }


def tool_function(
    problem_id: str,
    predicted_label: str,
    confidence: float = 0.0,
    reasoning: str = "",
    difficulty_factors: list[str] | None = None,
    suggested_improvement: str = ""
) -> str:
    """Record a reflection on a grading decision.
    
    This enables the agent to build up a knowledge base of grading patterns
    that can be used for meta-learning and self-improvement.
    """
    if difficulty_factors is None:
        difficulty_factors = []
    
    reflection_data = {
        "problem_id": problem_id,
        "predicted_label": predicted_label,
        "confidence": confidence,
        "reasoning": reasoning,
        "difficulty_factors": difficulty_factors,
        "suggested_improvement": suggested_improvement,
        "timestamp": None  # Could be added if needed
    }
    
    # Log the reflection for potential external analysis
    logger.info(f"[Reflection] Problem {problem_id}: {predicted_label} (conf: {confidence:.2f})")
    if reasoning:
        logger.info(f"[Reflection] Reasoning: {reasoning[:100]}...")
    if difficulty_factors:
        logger.info(f"[Reflection] Difficulty factors: {difficulty_factors}")
    if suggested_improvement:
        logger.info(f"[Reflection] Suggestion: {suggested_improvement[:100]}...")
    
    # Return structured data that could be stored or analyzed
    return json.dumps({
        "status": "recorded",
        "reflection": reflection_data
    }, indent=2)
