# -*- coding: utf-8 -*-
"""Tests for population-level coordination."""

import pytest

from hyperagents.agent import Hyperagent
from hyperagents.metrics import ImprovementTracker, SafetyLedger, ScoreSnapshot
from hyperagents.population import (
    META_SCOPE_NAME,
    EVAL_SCOPE_NAME,
    Population,
    PopulationConfig,
    RunMode,
    _build_agent_identity,
    _build_meta_scopes,
)
from hyperagents.strategy import Strategy
from hyperagents.tasks import Task, TaskResult


class _CountingTask(Task):
    """Task that scores based on output length (for predictable testing)."""

    @property
    def name(self) -> str:
        return "counting"

    @property
    def description(self) -> str:
        return "Produce a long response."

    def prompt(self) -> str:
        return "Write something."

    def evaluate(self, output: str) -> TaskResult:
        score = min(len(output) / 100.0, 1.0)
        return TaskResult(score=score, output=output)


class TestMetaScopes:
    def test_build_scopes(self):
        config = PopulationConfig()
        scopes = _build_meta_scopes(config)
        assert len(scopes) == 2
        names = {s.name for s in scopes}
        assert META_SCOPE_NAME in names
        assert EVAL_SCOPE_NAME in names

    def test_strategy_scope_allows_publish(self):
        config = PopulationConfig()
        scopes = _build_meta_scopes(config)
        meta_scope = next(s for s in scopes if s.name == META_SCOPE_NAME)
        assert meta_scope.allows_action_verb("publish_strategy")
        assert not meta_scope.allows_action_verb("delete_strategy")


class TestAgentIdentity:
    def test_build_identity(self):
        ha = Hyperagent(name="test-agent")
        identity = _build_agent_identity(ha)
        assert identity.name == "test-agent"
        from markspace import MarkType
        assert identity.can_write(META_SCOPE_NAME, MarkType.ACTION)
        assert identity.can_write(EVAL_SCOPE_NAME, MarkType.OBSERVATION)


class TestPopulationConfig:
    def test_defaults(self):
        config = PopulationConfig()
        assert config.n_agents == 10
        assert config.mode == RunMode.COLLECTIVE
        assert config.generations == 20

    def test_isolated_mode(self):
        config = PopulationConfig(mode=RunMode.ISOLATED)
        pop = Population(config=config)
        assert pop.space is None
        assert pop.guard is None

    def test_collective_creates_space(self):
        config = PopulationConfig(n_agents=3, mode=RunMode.COLLECTIVE)
        pop = Population(config=config)
        assert pop.space is not None
        assert pop.guard is not None
        assert len(pop.identities) == 3


class TestImprovementTracker:
    def test_improvement_rate(self):
        tracker = ImprovementTracker()
        for gen in range(5):
            tracker.record(
                ScoreSnapshot(
                    agent_id="a1",
                    generation=gen,
                    mean_score=0.2 * gen,
                    recent_score=0.2 * gen,
                    n_strategies=gen,
                )
            )
        rate = tracker.improvement_rate("a1")
        assert rate > 0

    def test_improvement_rate_insufficient_data(self):
        tracker = ImprovementTracker()
        tracker.record(
            ScoreSnapshot(
                agent_id="a1", generation=0, mean_score=0.5,
                recent_score=0.5, n_strategies=0,
            )
        )
        assert tracker.improvement_rate("a1") == 0.0


class TestSafetyLedger:
    def test_starts_clean(self):
        ledger = SafetyLedger()
        assert ledger.clean
        assert ledger.count == 0

    def test_record_violation(self):
        from hyperagents.metrics import SafetyViolation
        ledger = SafetyLedger()
        ledger.record(SafetyViolation(
            agent_id="agent-1",
            violation_type="scope",
            description="Attempted write to forbidden scope",
        ))
        assert not ledger.clean
        assert ledger.count == 1

    def test_by_type_filters_correctly(self):
        from hyperagents.metrics import SafetyViolation
        ledger = SafetyLedger()
        ledger.record(SafetyViolation(agent_id="a1", violation_type="scope", description="s1"))
        ledger.record(SafetyViolation(agent_id="a2", violation_type="identity", description="s2"))
        ledger.record(SafetyViolation(agent_id="a3", violation_type="scope", description="s3"))
        assert len(ledger.by_type("scope")) == 2
        assert len(ledger.by_type("identity")) == 1
        assert len(ledger.by_type("self_modification_escape")) == 0

    def test_scope_violation_recorded_on_guard_rejection(self):
        """Guard rejections during publish_strategy are recorded in the ledger."""
        from markspace import ScopeError
        from unittest.mock import MagicMock, patch
        from hyperagents.population import Population, PopulationConfig, RunMode
        from hyperagents.strategy import Strategy

        config = PopulationConfig(n_agents=1, mode=RunMode.COLLECTIVE)
        pop = Population(config=config)
        agent = pop.agents[0]
        strategy = Strategy(name="test", description="test", content="test prompt")
        strategy.author_agent_id = agent.id

        # Force the guard to raise ScopeError so the ledger catches it
        with patch.object(pop.guard, "write_mark", side_effect=ScopeError("forbidden")):
            pop._publish_strategy(agent, strategy)

        assert pop.safety.count == 1
        assert pop.safety.by_type("scope")[0].agent_id == agent.id
