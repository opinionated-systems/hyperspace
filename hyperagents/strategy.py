# -*- coding: utf-8 -*-
"""
Strategy representation and local storage.

A Strategy is a named modification to agent behavior - a prompt fragment,
a reasoning approach, a tool usage pattern. Strategies are the unit of
meta-improvement that hyperagents discover and (in collective mode) share
through the mark space.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field


@dataclass(frozen=True)
class StrategyResult:
    """Outcome of applying a strategy to a task."""

    strategy_id: str
    task_name: str
    score: float  # 0.0 to 1.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class Strategy:
    """
    A transferable meta-improvement.

    Strategies have three parts:
    - name: short identifier (e.g., "chain-of-thought", "decompose-first")
    - description: what the strategy does (used for selection)
    - content: the actual modification (prompt fragment, instruction, etc.)

    Strategies accumulate evaluation history as they are tried on tasks.
    """

    name: str
    description: str
    content: str  # task_code (the solve() function)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    author_agent_id: str = ""
    created_at: float = field(default_factory=time.time)
    results: list[StrategyResult] = field(default_factory=list)
    meta_code: str | None = None  # improved meta function, if agent rewrote it

    @property
    def mean_score(self) -> float:
        """Average score across all evaluations, or 0.0 if untested."""
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)

    @property
    def n_evaluations(self) -> int:
        return len(self.results)

    def record(self, task_name: str, score: float) -> StrategyResult:
        """Record an evaluation result and return it."""
        result = StrategyResult(
            strategy_id=self.id,
            task_name=task_name,
            score=score,
        )
        self.results.append(result)
        return result

    def to_dict(self) -> dict:
        """Serialize for mark payload."""
        d = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "content": self.content,
            "author_agent_id": self.author_agent_id,
            "mean_score": self.mean_score,
            "n_evaluations": self.n_evaluations,
        }
        if self.meta_code is not None:
            d["meta_code"] = self.meta_code
        return d

    @classmethod
    def from_dict(cls, data: dict) -> Strategy:
        """Deserialize from mark payload."""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            content=data["content"],
            author_agent_id=data.get("author_agent_id", ""),
            meta_code=data.get("meta_code"),
        )


class StrategyStore:
    """
    Local strategy storage for an individual agent.

    Each agent maintains its own store. In collective mode, the population
    syncs stores through the mark space.
    """

    def __init__(self, max_size: int = 20) -> None:
        self._strategies: dict[str, Strategy] = {}
        self._max_size = max_size

    def add(self, strategy: Strategy) -> None:
        self._strategies[strategy.id] = strategy
        # Prune lowest-scoring strategy when over capacity
        if len(self._strategies) > self._max_size:
            worst_id = min(self._strategies, key=lambda sid: self._strategies[sid].mean_score)
            del self._strategies[worst_id]

    def get(self, strategy_id: str) -> Strategy | None:
        return self._strategies.get(strategy_id)

    def all(self) -> list[Strategy]:
        return list(self._strategies.values())

    def best(self, n: int = 1) -> list[Strategy]:
        """Return top-n strategies by mean score."""
        ranked = sorted(self._strategies.values(), key=lambda s: s.mean_score, reverse=True)
        return ranked[:n]

    def remove(self, strategy_id: str) -> None:
        self._strategies.pop(strategy_id, None)

    def __len__(self) -> int:
        return len(self._strategies)

    def __contains__(self, strategy_id: str) -> bool:
        return strategy_id in self._strategies
