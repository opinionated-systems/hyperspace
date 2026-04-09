# -*- coding: utf-8 -*-
"""
Population - DGM-H algorithm with optional stigmergic coordination.

Paper algorithm (Algorithm 1, arXiv 2603.19461):
    Initialize archive A = {(a^0, s^0)}
    For t = 1 to T:
        P = SelectParents(A)              # sigmoid × novelty weighting
        For each parent a in P:
            a' = a.Modify(a, A)           # metacognitive self-modification
            s' = Evaluate(a', T)          # evaluate modified agent
            If IsValid(a'):
                A = A ∪ {(a', s')}        # add to archive

Parent selection (App. A.2):
    alpha_mid = mean of top-m=3 agents' scores
    s_i = sigmoid(-lambda * (alpha_i - alpha_mid)),  lambda=10
    h_i = 1 / (1 + n_i)  where n_i = compiled children count
    w_i = s_i * h_i,  normalised to categorical distribution

ISOLATED mode: each fork self-improves with no shared state (DGM-H baseline).
COLLECTIVE mode: agents publish/adopt strategies via MarkSpace (our extension).
"""

from __future__ import annotations

import copy
import logging
import math
import random as _random
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from markspace import (
    Agent,
    ConflictPolicy,
    DecayConfig,
    Guard,
    MarkSpace,
    MarkType,
    Observation,
    Scope,
    ScopeError,
    hours,
    minutes,
)

from hyperagents.agent import Hyperagent, LLMCallable
from hyperagents.metrics import ImprovementTracker, SafetyLedger, SafetyViolation, ScoreSnapshot
from hyperagents.strategy import Strategy
from hyperagents.tasks import Task, TaskResult

logger = logging.getLogger(__name__)


class RunMode(str, Enum):
    ISOLATED = "isolated"
    COLLECTIVE = "collective"


@dataclass
class PopulationConfig:
    """Configuration matching the DGM-H paper."""

    max_iterations: int = 80        # total agent modifications (paper: 80 Polyglot, 100 others)
    parallel_workers: int = 10      # parents selected per batch (parallelism)
    n_eval_tasks: int = 50          # task attempts per evaluation (paper uses 50-task subset)
    staged_eval_cutoff: int = 10    # paper: eval on 10 first, expand to n_eval_tasks if score > staged_eval_threshold
    staged_eval_threshold: float = 0.3  # paper uses 0.4 on Polyglot; 0.3 for AIME (easier)
    mode: RunMode = RunMode.COLLECTIVE
    publish_threshold: float = 0.0
    adopt_threshold: float = 0.0
    strategy_half_life_hours: float = 24.0
    # Parent selection hyperparameters (paper App. A.2)
    parent_top_m: int = 3           # top-m agents for alpha_mid
    parent_lambda: float = 10.0     # sigmoid steepness
    parent_selection: str = "dgmh"  # "dgmh" (sigmoid × novelty) or "random"
    improve_mode: str = "directed"  # "directed", "tournament" (directed vs blind), or "annealing" (decaying random exploration)
    annealing_p0: float = 0.5       # initial probability of tournament round (annealing mode)
    annealing_decay: float = 0.9    # multiply p by this after each tournament round


META_SCOPE_NAME = "meta/strategies"
EVAL_SCOPE_NAME = "meta/evaluations"


def _build_meta_scopes(config: PopulationConfig) -> list[Scope]:
    return [
        Scope(
            name=META_SCOPE_NAME,
            allowed_action_verbs=("publish_strategy",),
            allowed_intent_verbs=("will_publish_strategy",),
            observation_topics=("strategy_evaluation",),
            conflict_policy=ConflictPolicy.HIGHEST_CONFIDENCE,
            decay=DecayConfig(
                observation_half_life=hours(config.strategy_half_life_hours),
                warning_half_life=hours(config.strategy_half_life_hours / 2),
                intent_ttl=minutes(30),
            ),
        ),
        Scope(
            name=EVAL_SCOPE_NAME,
            observation_topics=("evaluation_result",),
            conflict_policy=ConflictPolicy.HIGHEST_CONFIDENCE,
            decay=DecayConfig(
                observation_half_life=hours(config.strategy_half_life_hours),
                warning_half_life=hours(config.strategy_half_life_hours / 2),
                intent_ttl=minutes(30),
            ),
        ),
    ]


def _build_agent_identity(agent: Any) -> Agent:
    return Agent(
        name=agent.name,
        scopes={
            META_SCOPE_NAME: ["action", "observation", "intent"],
            EVAL_SCOPE_NAME: ["observation"],
        },
    )


class Population:
    """
    DGM-H archive algorithm.

    Starts from a single initial agent. Each iteration:
      1. Select a parent from the archive (sigmoid × novelty weighting)
      2. Fork the parent → child
      3. Child runs metacognitive self-modification (improve())
      4. Evaluate child on task (n_eval_tasks attempts)
      5. If improve() succeeded (IsValid), add child to archive

    In COLLECTIVE mode, children additionally publish/adopt strategies
    via the shared MarkSpace before/after self-modification.
    """

    def __init__(
        self,
        config: PopulationConfig | None = None,
        initial_agent: Any | None = None,
    ) -> None:
        self.config = config or PopulationConfig()

        # Import here to avoid circular; caller supplies a CodeHyperagent
        from hyperagents.code_agent import CodeHyperagent
        self.initial_agent = initial_agent or CodeHyperagent()

        # Archive: ordered list of agents; scores and child counts tracked separately
        self.archive: list[Any] = []
        self.scores: dict[str, float] = {}          # agent_id -> eval score
        self.n_children: dict[str, int] = defaultdict(int)  # agent_id -> compiled children

        self.tracker = ImprovementTracker()
        self.safety = SafetyLedger()

        if self.config.mode == RunMode.COLLECTIVE:
            scopes = _build_meta_scopes(self.config)
            self.space = MarkSpace(scopes=scopes)
            self.identities: dict[str, Agent] = {}
            self.guard = Guard(space=self.space)
        else:
            self.space = None
            self.identities = {}
            self.guard = None

        # Annealing mode state
        self._annealing_p = self.config.annealing_p0

    # ------------------------------------------------------------------
    # Parent selection — paper App. A.2
    # ------------------------------------------------------------------
    def _select_parents(self, n: int) -> list[Any]:
        """Sample n parents from archive.

        "dgmh": sigmoid × novelty weighting (paper App. A.2).
        "random": uniform random (Test 3: does selection matter?).
        """
        if len(self.archive) <= 1:
            return [self.archive[0]] * n

        if self.config.parent_selection == "random":
            return [_random.choice(self.archive) for _ in range(n)]

        archive_scores = [self.scores[a.id] for a in self.archive]
        top_m = sorted(archive_scores, reverse=True)[:self.config.parent_top_m]
        alpha_mid = sum(top_m) / len(top_m) if top_m else 0.0
        lam = self.config.parent_lambda

        weights = []
        for agent in self.archive:
            alpha_i = self.scores[agent.id]
            s_i = 1.0 / (1.0 + math.exp(-lam * (alpha_i - alpha_mid)))
            n_i = self.n_children[agent.id]
            h_i = 1.0 / (1.0 + n_i)
            weights.append(s_i * h_i)

        total = sum(weights)
        probs = [w / total for w in weights] if total > 0 else None
        return _random.choices(self.archive, weights=probs, k=n)

    # ------------------------------------------------------------------
    # Evaluate an agent on n_eval_tasks attempts
    # ------------------------------------------------------------------
    _MAX_INFRA_RETRIES = 2  # retries per eval slot on infrastructure errors
    _EVAL_PARALLELISM = 10  # parallel eval threads per worker (10 workers × 10 = 100 concurrent max)

    def _evaluate(
        self,
        agent: Any,
        task: Task,
        llm_call: LLMCallable,
        iteration: int,
    ) -> tuple[float, list[float]]:
        """Run eval tasks with staged evaluation and parallel execution.

        Stage 1: evaluate on staged_eval_cutoff problems in parallel.
        Stage 2: if score >= staged_eval_threshold, evaluate remaining in parallel.
        Otherwise stop early — bad agents don't waste compute.

        Problems are pre-popped from the task queue sequentially (thread-safe),
        then each eval thread gets a lightweight single-problem task wrapper
        instead of a full deepcopy of the 1000-problem task.
        """
        cutoff = self.config.staged_eval_cutoff
        total = self.config.n_eval_tasks

        def _run_evals(n: int, offset: int) -> list[float]:
            """Pre-pop n problems, solve in parallel, return scores."""
            # Pop problems sequentially (task queue is not thread-safe)
            problems_and_answers = []
            for _ in range(n):
                prompt, expected = task.pop_problem()
                problems_and_answers.append((prompt, expected))

            def _solve_one(idx: int) -> float:
                prompt, expected = problems_and_answers[idx]
                k = offset + idx
                raw_output: Any = None
                error_str = ""
                is_infra_error = False

                for retry in range(self._MAX_INFRA_RETRIES + 1):
                    try:
                        namespace: dict = {"llm_call": llm_call}
                        exec(agent.task_code, namespace)  # noqa: S102
                        raw_output = namespace["solve"](prompt)
                        error_str = ""
                        is_infra_error = False
                        break
                    except Exception as exc:  # noqa: BLE001
                        error_str = f"{type(exc).__name__}: {exc}"
                        is_infra_error = any(
                            ie in error_str for ie in agent._INFRA_ERRORS
                        )
                        if not is_infra_error or retry >= self._MAX_INFRA_RETRIES:
                            break
                        logger.info(
                            "  %s iter=%d eval=%d/%d infra error, retrying (%d/%d)",
                            agent.name, iteration, k + 1, total,
                            retry + 1, self._MAX_INFRA_RETRIES,
                        )

                # Delegate scoring to task — handles int, str, or any return type
                score, score_meta = task.score_raw(raw_output, expected)
                got = score_meta.get("got", raw_output)
                expected_short = expected if not isinstance(expected, dict) else (
                    expected.get("name") or expected.get("question_title")
                    or expected.get("id") or "?"
                )
                logger.info(
                    "  %s iter=%d eval=%d/%d score=%.2f (got=%s expected=%s%s)",
                    agent.name, iteration, k + 1, total,
                    score, got, expected_short,
                    f" err={error_str!r}" if error_str else "",
                )

                # Append to history (skip infra errors)
                if not is_infra_error:
                    # Don't store full dicts (e.g. Polyglot exercise) in history
                    hist_meta = {k: v for k, v in score_meta.items()
                                 if not isinstance(v, dict)}
                    result = TaskResult(
                        score=score,
                        output=str(raw_output)[:2000],
                        metadata={
                            "problem": prompt[:500],
                            **hist_meta,
                            "error": error_str,
                        },
                    )
                    agent.history.append(result)

                return score

            with ThreadPoolExecutor(max_workers=self._EVAL_PARALLELISM) as pool:
                scores = list(pool.map(_solve_one, range(n)))
            return scores

        # Stage 1
        attempt_scores = _run_evals(cutoff, 0)

        # Check staged threshold
        if cutoff < total:
            stage1_score = sum(attempt_scores) / len(attempt_scores) if attempt_scores else 0.0
            if stage1_score < self.config.staged_eval_threshold:
                logger.info(
                    "  %s iter=%d staged eval: %.3f < %.3f after %d problems, skipping remaining %d",
                    agent.name, iteration, stage1_score,
                    self.config.staged_eval_threshold, cutoff, total - cutoff,
                )
                return stage1_score, attempt_scores

        # Stage 2
        remaining = total - cutoff
        if remaining > 0:
            attempt_scores.extend(_run_evals(remaining, cutoff))

        score = sum(attempt_scores) / len(attempt_scores) if attempt_scores else 0.0
        return score, attempt_scores

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------
    def run(
        self,
        task: Task,
        llm_call: LLMCallable,
        on_snapshot: Callable[[ScoreSnapshot], None] | None = None,
        llm_logger: Any = None,
        on_progress: Callable[[int, int, dict], None] | None = None,
        improve_llm_call: LLMCallable | None = None,
        secondary_llm_call: LLMCallable | None = None,
        secondary_improve_llm_call: LLMCallable | None = None,
    ) -> "PopulationResult":
        """
        Run DGM-H algorithm for max_iterations total agent modifications.

        parallel_workers parents are selected and processed simultaneously
        per batch, giving max_iterations / parallel_workers batches.

        on_progress(completed, total, info): called after each batch with
        iteration count and summary info dict.

        improve_llm_call: separate LLM callable for improve() calls (e.g.
        higher temperature for diversity). Falls back to llm_call if None.

        secondary_llm_call / secondary_improve_llm_call: optional second
        model for heterogeneous populations. Odd-indexed workers use the
        secondary model. This creates genuine capability diversity — the
        condition under which sharing should help.
        """
        _improve_llm_call = improve_llm_call or llm_call
        _secondary_llm = secondary_llm_call
        _secondary_improve_llm = secondary_improve_llm_call or _secondary_llm
        snapshot_lock = threading.Lock()

        def _make_agent_llm(
            agent_name: str, call_type: str, iteration: int,
            use_secondary: bool = False,
        ) -> LLMCallable:
            if use_secondary and _secondary_llm is not None:
                base = _secondary_improve_llm if call_type == "improve" else _secondary_llm
            else:
                base = _improve_llm_call if call_type == "improve" else llm_call
            if llm_logger is not None:
                llm_logger.set_context(agent_name, call_type, generation=iteration)
                return llm_logger.wrap(base)
            return base

        def _snapshot(agent: Any, iteration: int, attempt_scores: list[float]) -> None:
            score = sum(attempt_scores) / len(attempt_scores) if attempt_scores else 0.0
            snap = ScoreSnapshot(
                agent_id=agent.id,
                generation=iteration,
                model_tag=getattr(agent, "_model_tag", ""),
                mean_score=agent.mean_score,
                recent_score=agent.recent_score,
                n_strategies=len(agent.strategies),
                task_prompt=agent.task_prompt,
                meta_prompt=getattr(agent, "meta_code", ""),
                strategies=[s.to_dict() for s in agent.strategies.all()],
                attempt_scores=attempt_scores,
                score_this_gen=score,
            )
            self.tracker.record(snap)
            if on_snapshot is not None:
                with snapshot_lock:
                    on_snapshot(snap)

        # ── Step 0: evaluate initial agent ────────────────────────────
        logger.info("Evaluating initial agent %s", self.initial_agent.name)
        agent0 = self.initial_agent
        if self.config.mode == RunMode.COLLECTIVE:
            self.identities[agent0.id] = _build_agent_identity(agent0)

        agent0_task = copy.deepcopy(task)
        a0_llm = _make_agent_llm(agent0.name, "solve", 0)
        score0, attempts0 = self._evaluate(agent0, agent0_task, a0_llm, 0)

        self.archive.append(agent0)
        self.scores[agent0.id] = score0
        _snapshot(agent0, 0, attempts0)

        logger.info("Initial agent score=%.3f", score0)

        if on_progress is not None:
            on_progress(0, self.config.max_iterations, {
                "archive_size": 1,
                "best_score": score0,
                "batch_valid": 1,
                "batch_size": 1,
                "phase": "init_done",
            })

        # ── Main loop ─────────────────────────────────────────────────
        total_iters = 0

        while total_iters < self.config.max_iterations:
            batch_size = min(
                self.config.parallel_workers,
                self.config.max_iterations - total_iters,
            )
            logger.info(
                "Iterations %d–%d / %d  (archive size=%d)",
                total_iters + 1,
                total_iters + batch_size,
                self.config.max_iterations,
                len(self.archive),
            )

            parents = self._select_parents(batch_size)

            # Collective: adopt strategies into parents before forking
            if self.config.mode == RunMode.COLLECTIVE:
                for parent in set(id(p) for p in parents):
                    pass  # adoption happens on the child after fork (see below)

            def _process(args: tuple) -> Any:
                worker_idx, parent = args
                iter_num = total_iters + worker_idx + 1

                # Stagger workers to avoid API flood (0–2s jitter)
                time.sleep(worker_idx * 0.2)

                # Fork parent → child
                child = copy.deepcopy(parent)
                child.id = uuid.uuid4().hex[:12]
                child.name = f"agent-{iter_num}"

                # Give child its own task queue with deterministic seed
                child_task = copy.deepcopy(task)
                if hasattr(child_task, "_rng") and hasattr(child_task, "_pool"):
                    child_task._rng = _random.Random(iter_num * 997 + worker_idx * 31)
                    child_task._rng.shuffle(child_task._pool)
                    child_task._queue = list(child_task._pool)
                    # Reassign topic bias for subset tasks (different per child)
                    if hasattr(child_task, "_assign_primary"):
                        child_task._assign_primary()

                # Collective: adopt strategies from mark space before self-modification
                if self.config.mode == RunMode.COLLECTIVE:
                    if child.id not in self.identities:
                        self.identities[child.id] = _build_agent_identity(child)
                    self._adopt_strategies(child)

                # Metacognitive self-modification: child rewrites task_code (+ meta_code)
                # Odd workers use secondary model (if provided) for heterogeneous populations
                use_secondary = (worker_idx % 2 == 1) and _secondary_llm is not None
                improve_llm = _make_agent_llm(child.name, "improve", iter_num, use_secondary)
                strategy = child.improve(child_task, improve_llm)

                if strategy is None:
                    logger.info("  %s iter=%d: improve() failed (IsValid=False)", child.name, iter_num)
                    return None  # invalid — not added to archive

                # Tournament/annealing mode: also generate a blind candidate, keep the better one
                do_tournament = False
                if self.config.improve_mode == "tournament":
                    do_tournament = True
                elif self.config.improve_mode == "annealing":
                    r = _random.random()
                    if r < self._annealing_p:
                        do_tournament = True

                if do_tournament:
                    blind_child = copy.deepcopy(parent)
                    blind_child.id = uuid.uuid4().hex[:12]
                    blind_child.name = f"agent-{iter_num}-blind"
                    # Blind improve: LLM writes solve() without seeing current code or failures
                    blind_llm = _make_agent_llm(blind_child.name, "improve", iter_num, use_secondary)
                    blind_resp = blind_llm(
                        "Write a solve() function for this task. Return ONLY the code.",
                        child_task.description,
                    )
                    # Extract and validate
                    from hyperagents.code_agent import parse_improve_response
                    blind_code, _, _, _ = parse_improve_response(
                        f"STRATEGY_NAME: blind\nSTRATEGY_DESCRIPTION: blind generation\n<NEW_CODE>\n{blind_resp}\n</NEW_CODE>"
                    )
                    blind_valid = False
                    if blind_code:
                        try:
                            test_ns = {"llm_call": lambda s, u: "42"}
                            exec(blind_code, test_ns)  # noqa: S102
                            blind_valid = callable(test_ns.get("solve"))
                        except Exception:
                            pass

                    if blind_valid:
                        # Evaluate blind child
                        blind_child.task_code = blind_code
                        blind_task = copy.deepcopy(child_task)
                        blind_solve_llm = _make_agent_llm(blind_child.name, "solve", iter_num, use_secondary)
                        blind_score, blind_attempts = self._evaluate(blind_child, blind_task, blind_solve_llm, iter_num)

                # Evaluate the directed child
                solve_llm = _make_agent_llm(child.name, "solve", iter_num, use_secondary)
                score, attempt_scores = self._evaluate(child, child_task, solve_llm, iter_num)

                # Tournament: pick the winner
                if do_tournament and blind_valid:
                    if blind_score > score:
                        logger.info(
                            "  %s iter=%d TOURNAMENT: blind wins (%.3f > %.3f)",
                            child.name, iter_num, blind_score, score,
                        )
                        child = blind_child
                        score = blind_score
                        attempt_scores = blind_attempts
                        strategy = Strategy(
                            name="blind_winner",
                            description="Blind generation beat directed improve",
                            content=blind_code,
                        )
                    else:
                        logger.info(
                            "  %s iter=%d TOURNAMENT: directed wins (%.3f >= %.3f)",
                            child.name, iter_num, score, blind_score,
                        )
                    # Decay annealing probability after each tournament round
                    if self.config.improve_mode == "annealing":
                        self._annealing_p *= self.config.annealing_decay
                        logger.info("  Annealing p = %.4f", self._annealing_p)

                model_tag = "secondary" if use_secondary else "primary"
                logger.info(
                    "  %s iter=%d score=%.3f strategy='%s'%s [%s]",
                    child.name, iter_num, score, strategy.name,
                    " +meta" if strategy.meta_code else "",
                    model_tag,
                )
                return child, parent, score, strategy, attempt_scores, iter_num, use_secondary

            with ThreadPoolExecutor(max_workers=batch_size) as pool:
                futures = {pool.submit(_process, args): args for args in enumerate(parents)}
                results = []
                try:
                    for future in as_completed(futures, timeout=600):
                        try:
                            results.append(future.result())
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("Worker failed: %s", exc)
                            results.append(None)
                except TimeoutError:
                    done = sum(1 for f in futures if f.done())
                    logger.warning(
                        "Batch timeout: %d/%d workers finished, abandoning stragglers",
                        done, len(futures),
                    )
                    for f in futures:
                        f.cancel()

            # Sequential phase: add valid children to archive, publish, snapshot
            for result in results:
                if result is None:
                    continue
                # Unpack — use_secondary may be absent in older code paths
                if len(result) == 7:
                    child, parent, score, strategy, attempt_scores, iter_num, use_secondary = result
                else:
                    child, parent, score, strategy, attempt_scores, iter_num = result
                    use_secondary = False

                self.archive.append(child)
                self.scores[child.id] = score
                self.n_children[parent.id] += 1

                if self.config.mode == RunMode.COLLECTIVE:
                    if child.id not in self.identities:
                        self.identities[child.id] = _build_agent_identity(child)
                    if score >= self.config.publish_threshold:
                        self._publish_strategy(child, strategy)

                # Tag agent with model info for analysis
                child._model_tag = "secondary" if use_secondary else "primary"
                _snapshot(child, iter_num, attempt_scores)

            total_iters += batch_size

            if on_progress is not None:
                best_score = max(self.scores.values()) if self.scores else 0.0
                on_progress(total_iters, self.config.max_iterations, {
                    "archive_size": len(self.archive),
                    "best_score": best_score,
                    "batch_valid": sum(1 for r in results if r is not None),
                    "batch_size": batch_size,
                })

        return PopulationResult(
            mode=self.config.mode,
            n_agents=len(self.archive),
            max_iterations=self.config.max_iterations,
            tracker=self.tracker,
            safety=self.safety,
            agents=list(self.archive),
        )

    # ------------------------------------------------------------------
    # Mark space: publish / adopt
    # ------------------------------------------------------------------
    def _publish_strategy(self, agent: Any, strategy: Strategy) -> None:
        """Publish a strategy to the mark space.

        Each publication creates a new Observation mark. Old marks decay
        independently. Strategies that are effective get re-discovered and
        re-published by successful children, creating fresh marks that
        replace decaying ones — emergent reinforcement through independent
        validation, not explicit deduplication.
        """
        assert self.space is not None
        assert self.guard is not None
        identity = self.identities[agent.id]
        eval_score = min(self.scores.get(agent.id, 0.0), 1.0)
        payload = strategy.to_dict()
        payload["eval_score"] = eval_score
        mark = Observation(
            agent_id=identity.id,
            scope=META_SCOPE_NAME,
            topic="strategy_evaluation",
            content=payload,
            confidence=eval_score,
        )
        try:
            self.guard.write_mark(identity, mark)
            logger.info(
                "Agent %s published '%s' (score=%.3f)",
                agent.name, strategy.name, eval_score,
            )
        except (ScopeError, ValueError) as e:
            self.safety.record(
                SafetyViolation(
                    agent_id=agent.id,
                    violation_type="scope",
                    description=f"Guard rejected publish: {e}",
                )
            )

    def seed_strategies(self, strategies: list[dict]) -> None:
        """Pre-populate the mark space with external strategies.

        Used for transfer experiments: seed the mark space with strategies
        from a different model's run before starting the current run.
        Each strategy dict must have: name, description, content (task_code),
        eval_score. Optional: meta_code.
        """
        if self.space is None or self.guard is None:
            logger.warning("Cannot seed strategies: mark space not initialised (ISOLATED mode?)")
            return

        # Create a synthetic identity for the transfer source
        transfer_id = "transfer_source"
        transfer_identity = _build_agent_identity(type("FakeAgent", (), {"id": transfer_id, "name": "transfer"})())
        self.identities[transfer_id] = transfer_identity

        for s in strategies:
            strategy = Strategy(
                name=s["name"],
                description=s.get("description", s["name"]),
                content=s["content"],
                meta_code=s.get("meta_code"),
            )
            strategy.author_agent_id = transfer_id
            eval_score = s.get("eval_score", 0.5)
            payload = strategy.to_dict()
            payload["eval_score"] = eval_score
            mark = Observation(
                agent_id=transfer_identity.id,
                scope=META_SCOPE_NAME,
                topic="strategy_evaluation",
                content=payload,
                confidence=eval_score,
            )
            self.guard.write_mark(transfer_identity, mark)
            logger.info(
                "Seeded transfer strategy '%s' (score=%.3f, %d chars)",
                strategy.name, eval_score, len(strategy.content),
            )

    def _adopt_strategies(self, agent: Any) -> None:
        """Adopt the highest-scoring strategy from the mark space.

        Reads all available strategies, filters by threshold, and adopts
        the single best one. The child then runs improve() on top of the
        adopted code. Each mark is a distinct strategy — no identity
        matching or deduplication between strategies.
        """
        assert self.space is not None
        identity = self.identities.get(agent.id)
        if identity is None:
            return
        marks = self.space.read(scope=META_SCOPE_NAME, mark_type=MarkType.OBSERVATION)
        best_mark = None
        best_score = -1.0
        for mark in marks:
            if not isinstance(mark, Observation):
                continue
            payload = mark.content
            if not isinstance(payload, dict) or "id" not in payload:
                continue
            if payload.get("author_agent_id") == agent.id:
                continue
            if payload["id"] in agent.strategies:
                continue
            score = payload.get("eval_score", mark.confidence)
            if score < self.config.adopt_threshold:
                continue
            if score > best_score:
                best_score = score
                best_mark = mark

        if best_mark is not None:
            strategy = Strategy.from_dict(best_mark.content)
            agent.adopt_strategy(strategy)


@dataclass
class PopulationResult:
    mode: RunMode
    n_agents: int
    max_iterations: int
    tracker: ImprovementTracker
    safety: SafetyLedger
    agents: list[Any] = field(default_factory=list)

    @property
    def population_improvement_rate(self) -> float:
        return self.tracker.population_improvement_rate()

    @property
    def safety_violations(self) -> int:
        return self.safety.count

    def summary(self) -> dict[str, Any]:
        last_iter = self.max_iterations
        return {
            "mode": self.mode.value,
            "n_agents_in_archive": self.n_agents,
            "max_iterations": self.max_iterations,
            "improvement_rate": self.population_improvement_rate,
            "imp_at_k": self.tracker.population_imp_at_k(last_iter),
            "score_iter0": self.tracker.population_score_this_gen_at(0),
            "score_final": self.tracker.population_score_this_gen_at(last_iter),
            "safety_violations": self.safety_violations,
            "total_snapshots": len(self.tracker.all_snapshots),
        }
