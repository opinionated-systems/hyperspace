# -*- coding: utf-8 -*-
"""Tests for CodeHyperagent and the new metrics."""

import pytest

from hyperagents.code_agent import CodeHyperagent, CodeHyperagentConfig, INITIAL_TASK_CODE
from hyperagents.metrics import ImprovementTracker, ScoreSnapshot
from hyperagents.strategy import Strategy
from hyperagents.tasks import Task, TaskResult


# ---------------------------------------------------------------------------
# Minimal task for testing — expects the string "42"
# ---------------------------------------------------------------------------
class _FortyTwoTask(Task):
    @property
    def name(self) -> str:
        return "fortytwo"

    @property
    def description(self) -> str:
        return "Return the integer 42."

    def prompt(self) -> str:
        return "What is 6 × 7?"

    def evaluate(self, output: str) -> TaskResult:
        try:
            val = int(output.strip())
            score = 1.0 if val == 42 else 0.0
        except ValueError:
            score = 0.0
        return TaskResult(
            score=score,
            output=output,
            metadata={"expected": 42, "extracted": None, "correct": score == 1.0},
        )


def _llm_returns_42(system: str, user: str) -> str:
    return "42"


def _llm_returns_wrong(system: str, user: str) -> str:
    return "not a number"


def _llm_meta_returns_code(system: str, user: str) -> str:
    return (
        "STRATEGY_NAME: always-42\n"
        "STRATEGY_DESCRIPTION: Hardcode 42 for testing.\n"
        "NEW_CODE:\n"
        "```python\n"
        "def solve(problem: str) -> int:\n"
        "    return 42\n"
        "```\n"
    )


# ---------------------------------------------------------------------------
# CodeHyperagent tests
# ---------------------------------------------------------------------------
class TestCodeHyperagent:
    def test_initial_state(self):
        agent = CodeHyperagent()
        assert agent.generation == 0
        assert agent.mean_score == 0.0
        assert len(agent.strategies) == 0
        assert "def solve" in agent.task_code

    def test_task_prompt_property_returns_code(self):
        agent = CodeHyperagent()
        assert agent.task_prompt == agent.task_code

    def test_solve_with_crashing_code_returns_score_zero(self):
        agent = CodeHyperagent()
        agent.task_code = "def solve(problem):\n    raise RuntimeError('boom')"
        task = _FortyTwoTask()
        result = agent.solve(task, _llm_returns_42)
        assert result.score == 0.0
        assert result.metadata["error"] != ""

    def test_solve_correct_code_scores_one(self):
        agent = CodeHyperagent()
        agent.task_code = "def solve(problem):\n    return 42"
        task = _FortyTwoTask()
        result = agent.solve(task, _llm_returns_42)
        assert result.score == 1.0
        assert result.metadata["got"] == 42

    def test_solve_stores_problem_in_metadata(self):
        agent = CodeHyperagent()
        agent.task_code = "def solve(problem):\n    return 42"
        task = _FortyTwoTask()
        result = agent.solve(task, _llm_returns_42)
        assert "problem" in result.metadata
        assert "6" in result.metadata["problem"]

    def test_solve_llm_injection(self):
        """Code can call llm_call as a global."""
        agent = CodeHyperagent()
        agent.task_code = (
            "def solve(problem):\n"
            "    resp = llm_call('sys', 'user')\n"
            "    return int(resp.strip())\n"
        )
        task = _FortyTwoTask()
        result = agent.solve(task, _llm_returns_42)
        assert result.score == 1.0

    def test_improve_requires_history(self):
        agent = CodeHyperagent()
        task = _FortyTwoTask()
        result = agent.improve(task, _llm_meta_returns_code)
        assert result is None

    def test_improve_updates_code(self):
        agent = CodeHyperagent()
        task = _FortyTwoTask()
        agent.solve(task, _llm_returns_wrong)
        agent.solve(task, _llm_returns_wrong)
        strategy = agent.improve(task, _llm_meta_returns_code)
        assert strategy is not None
        assert strategy.name == "always-42"
        assert "return 42" in agent.task_code
        assert agent.generation == 1

    def test_improve_parses_code_block(self):
        agent = CodeHyperagent()
        task = _FortyTwoTask()
        agent.solve(task, _llm_returns_wrong)
        agent.solve(task, _llm_returns_wrong)
        strategy = agent.improve(task, _llm_meta_returns_code)
        # After improvement, solving should now return 42
        result = agent.solve(task, _llm_returns_wrong)
        assert result.score == 1.0

    def test_improve_invalid_response_returns_none(self):
        agent = CodeHyperagent()
        task = _FortyTwoTask()
        agent.solve(task, _llm_returns_wrong)
        agent.solve(task, _llm_returns_wrong)
        result = agent.improve(task, lambda s, u: "no valid code here")
        assert result is None

    def test_adopt_strategy_updates_code(self):
        agent = CodeHyperagent()
        s = Strategy(
            name="better-parse",
            description="Better parsing",
            content="def solve(problem):\n    return 42",
            author_agent_id="other",
        )
        agent.adopt_strategy(s)
        assert "return 42" in agent.task_code
        assert agent.task_prompt == agent.task_code
        assert agent.generation == 1

    def test_mean_score_across_attempts(self):
        agent = CodeHyperagent()
        task = _FortyTwoTask()
        agent.task_code = "def solve(p):\n    return 42"
        agent.solve(task, _llm_returns_42)
        agent.task_code = "def solve(p):\n    return 0"
        agent.solve(task, _llm_returns_42)
        assert agent.mean_score == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# ScoreSnapshot score_this_gen field
# ---------------------------------------------------------------------------
class TestScoreThisGen:
    def test_score_this_gen_default_zero(self):
        snap = ScoreSnapshot(
            agent_id="a1", generation=0, mean_score=0.0,
            recent_score=0.0, n_strategies=0,
        )
        assert snap.score_this_gen == 0.0

    def test_score_this_gen_set(self):
        snap = ScoreSnapshot(
            agent_id="a1", generation=1, mean_score=0.5,
            recent_score=0.5, n_strategies=1,
            attempt_scores=[1.0, 0.0, 1.0],
            score_this_gen=2 / 3,
        )
        assert snap.score_this_gen == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# ImprovementTracker — new methods
# ---------------------------------------------------------------------------
class TestImprovementTrackerNew:
    def _make_tracker(self) -> ImprovementTracker:
        t = ImprovementTracker()
        # Agent a1: gen0 score=0.0, gen1=0.4, gen2=0.8
        t.record(ScoreSnapshot("a1", 0, 0.0, 0.0, 0, score_this_gen=0.0))
        t.record(ScoreSnapshot("a1", 1, 0.2, 0.4, 1, score_this_gen=0.4))
        t.record(ScoreSnapshot("a1", 2, 0.4, 0.8, 2, score_this_gen=0.8))
        # Agent a2: gen0 score=0.0, gen1=0.2, gen2=0.6
        t.record(ScoreSnapshot("a2", 0, 0.0, 0.0, 0, score_this_gen=0.0))
        t.record(ScoreSnapshot("a2", 1, 0.1, 0.2, 1, score_this_gen=0.2))
        t.record(ScoreSnapshot("a2", 2, 0.2, 0.6, 2, score_this_gen=0.6))
        return t

    def test_score_this_gen_at(self):
        t = self._make_tracker()
        assert t.score_this_gen_at("a1", 0) == pytest.approx(0.0)
        assert t.score_this_gen_at("a1", 2) == pytest.approx(0.8)
        assert t.score_this_gen_at("a2", 1) == pytest.approx(0.2)

    def test_population_score_this_gen_at(self):
        t = self._make_tracker()
        # gen 2: a1=0.8, a2=0.6 → mean=0.7
        assert t.population_score_this_gen_at(2) == pytest.approx(0.7)

    def test_imp_at_k(self):
        t = self._make_tracker()
        # a1: gen0=0.0, best up to gen2=0.8 → imp=0.8
        assert t.imp_at_k("a1", 2) == pytest.approx(0.8)
        # a1: best up to gen1=0.4 → imp=0.4
        assert t.imp_at_k("a1", 1) == pytest.approx(0.4)

    def test_population_imp_at_k(self):
        t = self._make_tracker()
        # a1 imp@2=0.8, a2 imp@2=0.6 → mean=0.7
        assert t.population_imp_at_k(2) == pytest.approx(0.7)

    def test_imp_at_k_no_snapshots(self):
        t = ImprovementTracker()
        assert t.imp_at_k("nobody", 5) == 0.0

    def test_population_score_this_gen_missing(self):
        t = ImprovementTracker()
        assert t.population_score_this_gen_at(0) == 0.0
