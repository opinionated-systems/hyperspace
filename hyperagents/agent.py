# -*- coding: utf-8 -*-
"""
Hyperagent - a self-modifying LLM agent with task and meta components.

The task component solves problems. The meta component modifies the task
component's behavior (and can modify itself). This is the core DGM-H
abstraction from Zhang et al. (2026), extended with strategy extraction
and mark-space integration points.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Callable

from hyperagents.strategy import Strategy, StrategyStore
from hyperagents.tasks import Task, TaskResult

logger = logging.getLogger(__name__)

# Type alias for the LLM callable passed to solve/improve.
# Signature: (system_prompt: str, user_prompt: str) -> str
LLMCallable = Callable[[str, str], str]


@dataclass
class HyperagentConfig:
    """Configuration for a hyperagent."""

    # Initial task-solving prompt (the editable program)
    task_prompt: str = (
        "You are a problem-solving agent. "
        "Think step by step and solve the given task."
    )
    # Initial meta prompt (the self-modification mechanism)
    meta_prompt: str = (
        "You are a meta-learning agent. Your job is to improve the task-solving "
        "prompt based on past performance. Analyze what worked and what didn't, "
        "then propose a modified prompt and a named strategy describing the change. "
        "Return your response as:\n"
        "STRATEGY_NAME: <short name>\n"
        "STRATEGY_DESCRIPTION: <what it does>\n"
        "NEW_PROMPT: <the improved task prompt>\n"
    )
    # Maximum strategies to retain locally
    max_strategies: int = 20


class Hyperagent:
    """
    A self-modifying agent combining task execution with meta-improvement.

    The agent maintains:
    - A mutable task prompt (the "program" being improved)
    - A mutable meta prompt (the "improvement mechanism" - also editable)
    - A local strategy store (discovered improvements)
    - A performance history (task scores over time)

    In isolation, this reproduces DGM-H. When connected to a Population,
    the strategy store syncs through marks and the guard enforces scope.
    """

    def __init__(
        self,
        config: HyperagentConfig | None = None,
        agent_id: str | None = None,
        name: str | None = None,
    ) -> None:
        self.id = agent_id or uuid.uuid4().hex[:12]
        self.name = name or f"hyperagent-{self.id[:6]}"
        self.config = config or HyperagentConfig()
        self.task_prompt = self.config.task_prompt
        self.meta_prompt = self.config.meta_prompt
        self.strategies = StrategyStore(max_size=self.config.max_strategies)
        self.history: list[TaskResult] = []
        self._generation = 0  # number of self-modifications applied

    @property
    def generation(self) -> int:
        """How many self-modifications have been applied."""
        return self._generation

    @property
    def mean_score(self) -> float:
        """Average score across all task attempts."""
        if not self.history:
            return 0.0
        return sum(r.score for r in self.history) / len(self.history)

    @property
    def recent_score(self) -> float:
        """Average score over the last 5 attempts."""
        recent = self.history[-5:]
        if not recent:
            return 0.0
        return sum(r.score for r in recent) / len(recent)

    def solve(self, task: Task, llm_call: LLMCallable) -> TaskResult:
        """
        Execute the task component: apply current task_prompt to the task.

        Args:
            task: The benchmark task to solve.
            llm_call: Callable that takes (system_prompt, user_prompt) -> str.

        Returns:
            TaskResult with score and output.
        """
        user_prompt = task.prompt()
        output = llm_call(self.task_prompt, user_prompt)
        result = task.evaluate(output)
        self.history.append(result)
        return result

    def improve(self, task: Task, llm_call: LLMCallable) -> Strategy | None:
        """
        Execute the meta component: analyze performance and modify task_prompt.

        Uses the meta_prompt to reflect on recent performance and generate
        a new strategy. If the meta component produces a valid strategy,
        applies it and returns the Strategy object.

        Args:
            task: The task context (for description).
            llm_call: Callable that takes (system_prompt, user_prompt) -> str.

        Returns:
            The extracted Strategy if self-modification occurred, else None.
        """
        if len(self.history) < 2:
            return None  # need some history to reflect on

        # Build reflection context from recent history
        recent = self.history[-5:]
        history_text = "\n".join(
            f"  Attempt {i+1}: score={r.score:.2f}" for i, r in enumerate(recent)
        )

        reflection_prompt = (
            f"Task: {task.description}\n"
            f"Current task prompt:\n{self.task_prompt}\n\n"
            f"Recent performance:\n{history_text}\n\n"
            f"Current mean score: {self.mean_score:.3f}\n"
            f"Strategies tried: {len(self.strategies)}\n\n"
            "Analyze the pattern and propose an improvement."
        )

        response = llm_call(self.meta_prompt, reflection_prompt)
        strategy = self._parse_strategy(response)

        if strategy is not None:
            strategy.author_agent_id = self.id
            self.strategies.add(strategy)
            self._generation += 1
            logger.info(
                "Agent %s gen %d: applied strategy '%s'",
                self.name,
                self._generation,
                strategy.name,
            )

        return strategy

    def adopt_strategy(self, strategy: Strategy) -> None:
        """
        Adopt a strategy discovered by another agent (collective mode).

        Applies the strategy's content as the new task prompt and records
        it in the local store.
        """
        self.task_prompt = strategy.content
        if strategy.id not in self.strategies:
            self.strategies.add(strategy)
        self._generation += 1
        logger.info(
            "Agent %s adopted strategy '%s' from %s",
            self.name,
            strategy.name,
            strategy.author_agent_id,
        )

    def _parse_strategy(self, response: str) -> Strategy | None:
        """Extract a strategy from the meta component's response."""
        lines = response.strip().split("\n")
        name = ""
        description = ""
        new_prompt_lines: list[str] = []
        in_prompt = False

        for line in lines:
            if line.startswith("STRATEGY_NAME:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("STRATEGY_DESCRIPTION:"):
                description = line.split(":", 1)[1].strip()
            elif line.startswith("NEW_PROMPT:"):
                in_prompt = True
                rest = line.split(":", 1)[1].strip()
                if rest:
                    new_prompt_lines.append(rest)
            elif in_prompt:
                new_prompt_lines.append(line)

        new_prompt = "\n".join(new_prompt_lines).strip()

        if not (name and new_prompt):
            return None

        # Apply the modification
        self.task_prompt = new_prompt

        return Strategy(
            name=name,
            description=description or name,
            content=new_prompt,
        )
