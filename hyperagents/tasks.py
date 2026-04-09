# -*- coding: utf-8 -*-
"""
Task interface - benchmark abstraction for hyperagent evaluation.

A Task is a callable that takes a prompt and returns a TaskResult with
a score. Concrete tasks are defined in experiments.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TaskResult:
    """Outcome of a single task attempt."""

    score: float  # 0.0 to 1.0, higher is better
    output: str  # the agent's response
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"Score must be in [0, 1], got {self.score}")


class Task(ABC):
    """Abstract benchmark task."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this task."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description (used in meta-prompts)."""

    @abstractmethod
    def evaluate(self, output: str) -> TaskResult:
        """Score an agent's output against this task's criteria."""

    @abstractmethod
    def prompt(self) -> str:
        """Return the task prompt for the agent."""

    def pop_problem(self) -> tuple[str, Any]:
        """Pop next problem, return (prompt, expected_answer).

        Thread-safe pre-pop: call this sequentially before parallel eval.
        Subclasses should override to return the appropriate expected value.
        """
        prompt = self.prompt()
        return prompt, self._get_expected()

    def _get_expected(self) -> Any:
        """Return the expected answer for the most recently prompted problem."""
        # Default: try common attribute names
        if hasattr(self, "_current_grade"):
            return self._current_grade
        if hasattr(self, "_current_answer"):
            return self._current_answer
        raise NotImplementedError("Subclass must implement _get_expected()")

    def score_raw(self, raw_output: Any, expected: Any) -> tuple[float, dict]:
        """Score raw solve() output against expected answer.

        Must be stateless and thread-safe (no reliance on instance state).
        Returns (score, metadata_dict).

        Default: convert raw to int, exact match against expected int.
        Override for non-integer tasks (e.g. Polyglot code generation).
        """
        try:
            got = int(raw_output)
        except (ValueError, TypeError):
            got = -1
        correct = got == expected
        return (
            1.0 if correct else 0.0,
            {"expected": expected, "got": got, "correct": correct},
        )
