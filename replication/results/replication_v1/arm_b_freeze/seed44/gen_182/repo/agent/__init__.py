"""
Agent package for IMO grading task.

Modules:
- agentic_loop: Main agentic loop with tool calling
- llm_client: LLM client wrapper
- tools: Tool implementations (bash, editor)
- grading_utils: Utilities for grading and answer analysis
"""

from agent.grading_utils import (
    extract_numerical_answer,
    normalize_answer,
    compare_answers,
    analyze_solution_structure,
    calculate_partial_credit,
)

__all__ = [
    "extract_numerical_answer",
    "normalize_answer",
    "compare_answers",
    "analyze_solution_structure",
    "calculate_partial_credit",
]
