# -*- coding: utf-8 -*-
"""
hyperspace = hyperagents + markspace.
Cross-agent evolution of self-modifying code via environment-mediated coordination.
"""

from hyperagents.strategy import Strategy, StrategyResult, StrategyStore
from hyperagents.agent import Hyperagent, HyperagentConfig
from hyperagents.population import Population, PopulationConfig, RunMode
from hyperagents.metrics import ImprovementTracker, SafetyLedger
from hyperagents.tasks import Task, TaskResult

__all__ = [
    # Agent
    "Hyperagent",
    "HyperagentConfig",
    # Strategy
    "Strategy",
    "StrategyResult",
    "StrategyStore",
    # Population
    "Population",
    "PopulationConfig",
    "RunMode",
    # Metrics
    "ImprovementTracker",
    "SafetyLedger",
    # Tasks
    "Task",
    "TaskResult",
]
