# -*- coding: utf-8 -*-
"""
Metrics for tracking improvement rates and safety invariants.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class ScoreSnapshot:
    """A point-in-time measurement of agent performance."""

    agent_id: str
    generation: int
    mean_score: float
    recent_score: float
    n_strategies: int
    timestamp: float = field(default_factory=time.time)
    # Rich fields — empty by default so existing tests don't need updating
    task_prompt: str = ""
    meta_prompt: str = ""  # current meta_code (the improve() function)
    strategies: list[dict] = field(default_factory=list)
    attempt_scores: list[float] = field(default_factory=list)
    score_this_gen: float = field(default=0.0)
    model_tag: str = ""  # "primary" or "secondary" for heterogeneous populations


class ImprovementTracker:
    """
    Tracks performance over time across a population.

    Records snapshots and computes improvement rate (the derivative of
    mean score with respect to generation). The key metric is whether
    collective meta-learning accelerates this rate compared to isolation.
    """

    def __init__(self) -> None:
        self._snapshots: list[ScoreSnapshot] = []

    def record(self, snapshot: ScoreSnapshot) -> None:
        self._snapshots.append(snapshot)

    def snapshots_for(self, agent_id: str) -> list[ScoreSnapshot]:
        return [s for s in self._snapshots if s.agent_id == agent_id]

    def population_mean_at(self, generation: int) -> float:
        """Mean score across all agents at a given generation."""
        scores = [s.mean_score for s in self._snapshots if s.generation == generation]
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    def improvement_rate(self, agent_id: str) -> float:
        """
        Linear improvement rate (score per generation) for a single agent.

        Returns the slope of a least-squares fit of mean_score vs generation.
        Returns 0.0 if fewer than 2 snapshots.
        """
        points = self.snapshots_for(agent_id)
        if len(points) < 2:
            return 0.0

        xs = [p.generation for p in points]
        ys = [p.mean_score for p in points]
        n = len(xs)
        x_mean = sum(xs) / n
        y_mean = sum(ys) / n
        num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
        den = sum((x - x_mean) ** 2 for x in xs)
        if den == 0:
            return 0.0
        return num / den

    def population_improvement_rate(self) -> float:
        """Mean improvement rate across all agents."""
        agent_ids = set(s.agent_id for s in self._snapshots)
        if not agent_ids:
            return 0.0
        rates = [self.improvement_rate(aid) for aid in agent_ids]
        return sum(rates) / len(rates)

    def score_this_gen_at(self, agent_id: str, generation: int) -> float:
        """score_this_gen for a specific agent at a specific generation."""
        snaps = [s for s in self._snapshots if s.agent_id == agent_id and s.generation == generation]
        if not snaps:
            return 0.0
        return snaps[0].score_this_gen

    def population_score_this_gen_at(self, generation: int) -> float:
        """Mean score_this_gen across agents that have a snapshot at this generation.

        Only includes agents that were actually evaluated at this generation,
        not all agents that exist in any generation (which would dilute with zeros).
        """
        snaps = [s for s in self._snapshots if s.generation == generation]
        if not snaps:
            return 0.0
        return sum(s.score_this_gen for s in snaps) / len(snaps)

    def imp_at_k(self, agent_id: str, k: int) -> float:
        """
        imp@k: improvement in score_this_gen from generation 0 to generation k.

        Matches the paper's metric: best score achieved up to generation k
        minus the score at generation 0. Uses score_this_gen (per-generation
        accuracy) rather than cumulative mean_score.
        """
        snaps = self.snapshots_for(agent_id)
        if not snaps:
            return 0.0
        gen0_scores = [s.score_this_gen for s in snaps if s.generation == 0]
        gen0 = gen0_scores[0] if gen0_scores else 0.0
        up_to_k = [s.score_this_gen for s in snaps if s.generation <= k]
        if not up_to_k:
            return 0.0
        return max(up_to_k) - gen0

    def population_imp_at_k(self, k: int) -> float:
        """Mean imp@k across all agents."""
        agent_ids = set(s.agent_id for s in self._snapshots)
        if not agent_ids:
            return 0.0
        return sum(self.imp_at_k(aid, k) for aid in agent_ids) / len(agent_ids)

    @property
    def all_snapshots(self) -> list[ScoreSnapshot]:
        return list(self._snapshots)


@dataclass
class SafetyViolation:
    """A recorded safety invariant violation."""

    agent_id: str
    violation_type: str  # "scope", "identity", "self_modification_escape"
    description: str
    timestamp: float = field(default_factory=time.time)


class SafetyLedger:
    """
    Tracks safety invariant violations across the population.

    In a well-functioning system, this ledger should be empty.
    The experiment's safety claim is: the guard holds even when
    agents self-modify.
    """

    def __init__(self) -> None:
        self._violations: list[SafetyViolation] = []

    def record(self, violation: SafetyViolation) -> None:
        self._violations.append(violation)

    @property
    def count(self) -> int:
        return len(self._violations)

    @property
    def violations(self) -> list[SafetyViolation]:
        return list(self._violations)

    def by_type(self, violation_type: str) -> list[SafetyViolation]:
        return [v for v in self._violations if v.violation_type == violation_type]

    @property
    def clean(self) -> bool:
        return len(self._violations) == 0
