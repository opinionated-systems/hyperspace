"""
Agent package for IMO grading task.

This package contains the core agent components:
- llm_client: LLM API client with retry logic and audit logging
- agentic_loop: Tool-calling agent loop with native API support
- tools: Tool implementations (bash, editor, search, thinking)

The task_agent module provides the TaskAgent class for grading IMO problems.
The meta_agent module provides the MetaAgent class for self-improvement.
"""

from __future__ import annotations

__version__ = "1.0.0"
