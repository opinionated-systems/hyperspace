# -*- coding: utf-8 -*-
"""Tests for the Hyperagent core."""

import pytest

from hyperagents.agent import Hyperagent, HyperagentConfig
from hyperagents.strategy import Strategy
from hyperagents.tasks import Task, TaskResult


class _DummyTask(Task):
    """A trivial task for testing."""

    @property
    def name(self) -> str:
        return "dummy"

    @property
    def description(self) -> str:
        return "Return the word 'hello'."

    def prompt(self) -> str:
        return "Say hello."

    def evaluate(self, output: str) -> TaskResult:
        score = 1.0 if "hello" in output.lower() else 0.0
        return TaskResult(score=score, output=output)


def _mock_llm(system: str, user: str) -> str:
    """Mock LLM that always says hello."""
    return "hello world"


def _mock_llm_meta(system: str, user: str) -> str:
    """Mock LLM that returns a valid meta-improvement."""
    return (
        "STRATEGY_NAME: be-polite\n"
        "STRATEGY_DESCRIPTION: Add politeness to responses\n"
        "NEW_PROMPT: You are a polite problem-solver. Say hello nicely."
    )


class TestHyperagent:
    def test_create_default(self):
        agent = Hyperagent()
        assert agent.generation == 0
        assert agent.mean_score == 0.0
        assert len(agent.strategies) == 0

    def test_solve_records_history(self):
        agent = Hyperagent()
        task = _DummyTask()
        result = agent.solve(task, _mock_llm)
        assert result.score == 1.0
        assert len(agent.history) == 1
        assert agent.mean_score == 1.0

    def test_improve_needs_history(self):
        agent = Hyperagent()
        task = _DummyTask()
        # No history yet - should return None
        strategy = agent.improve(task, _mock_llm_meta)
        assert strategy is None

    def test_improve_produces_strategy(self):
        agent = Hyperagent()
        task = _DummyTask()
        # Build up some history
        agent.solve(task, _mock_llm)
        agent.solve(task, _mock_llm)
        strategy = agent.improve(task, _mock_llm_meta)
        assert strategy is not None
        assert strategy.name == "be-polite"
        assert agent.generation == 1
        assert "polite" in agent.task_prompt

    def test_adopt_strategy(self):
        agent = Hyperagent()
        s = Strategy(
            name="external",
            description="From another agent",
            content="You are a focused solver.",
            author_agent_id="other-agent",
        )
        agent.adopt_strategy(s)
        assert agent.task_prompt == "You are a focused solver."
        assert agent.generation == 1
        assert s.id in agent.strategies

    def test_adopt_same_strategy_twice_no_duplicate(self):
        agent = Hyperagent()
        s = Strategy(name="x", description="x", content="x")
        agent.adopt_strategy(s)
        agent.adopt_strategy(s)
        assert len(agent.strategies) == 1
        assert agent.generation == 2  # still counts as a modification

    def test_parse_strategy_invalid_response(self):
        agent = Hyperagent()
        # No STRATEGY_NAME or NEW_PROMPT
        result = agent._parse_strategy("just some random text")
        assert result is None

    def test_parse_strategy_missing_prompt(self):
        result = Hyperagent()._parse_strategy("STRATEGY_NAME: foo\n")
        assert result is None


class TestTaskResult:
    def test_score_bounds(self):
        with pytest.raises(ValueError):
            TaskResult(score=1.5, output="oops")
        with pytest.raises(ValueError):
            TaskResult(score=-0.1, output="oops")
