# -*- coding: utf-8 -*-
"""Tests for strategy representation and storage."""

import pytest

from hyperagents.strategy import Strategy, StrategyResult, StrategyStore


class TestStrategy:
    def test_create_strategy(self):
        s = Strategy(name="cot", description="Chain of thought", content="Think step by step.")
        assert s.name == "cot"
        assert s.mean_score == 0.0
        assert s.n_evaluations == 0

    def test_record_results(self):
        s = Strategy(name="cot", description="Chain of thought", content="Think step by step.")
        s.record("math", 0.8)
        s.record("math", 0.6)
        assert s.n_evaluations == 2
        assert s.mean_score == pytest.approx(0.7)

    def test_serialization_roundtrip(self):
        s = Strategy(
            name="decompose",
            description="Break into subproblems",
            content="First, decompose the problem.",
            author_agent_id="agent-1",
        )
        s.record("coding", 0.9)
        d = s.to_dict()
        s2 = Strategy.from_dict(d)
        assert s2.name == s.name
        assert s2.content == s.content
        assert s2.author_agent_id == s.author_agent_id
        # Results are not serialized (they are local)
        assert s2.n_evaluations == 0

    def test_to_dict_includes_aggregate_scores(self):
        s = Strategy(name="test", description="test", content="test")
        s.record("a", 0.5)
        s.record("b", 0.7)
        d = s.to_dict()
        assert d["mean_score"] == pytest.approx(0.6)
        assert d["n_evaluations"] == 2


class TestStrategyStore:
    def test_add_and_retrieve(self):
        store = StrategyStore()
        s = Strategy(name="cot", description="Chain of thought", content="Think step by step.")
        store.add(s)
        assert len(store) == 1
        assert store.get(s.id) is s
        assert s.id in store

    def test_best_returns_sorted(self):
        store = StrategyStore()
        s1 = Strategy(name="bad", description="bad", content="bad")
        s1.record("test", 0.2)
        s2 = Strategy(name="good", description="good", content="good")
        s2.record("test", 0.9)
        s3 = Strategy(name="mid", description="mid", content="mid")
        s3.record("test", 0.5)
        store.add(s1)
        store.add(s2)
        store.add(s3)
        best = store.best(2)
        assert len(best) == 2
        assert best[0].name == "good"
        assert best[1].name == "mid"

    def test_remove(self):
        store = StrategyStore()
        s = Strategy(name="cot", description="cot", content="cot")
        store.add(s)
        store.remove(s.id)
        assert len(store) == 0
        assert store.get(s.id) is None

    def test_remove_nonexistent_is_noop(self):
        store = StrategyStore()
        store.remove("nonexistent")  # should not raise
