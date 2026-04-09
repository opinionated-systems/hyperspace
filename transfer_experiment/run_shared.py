"""
Run shared transfer experiment: Claude and Kimi take turns,
exchanging strategies via markspace after each generation.

Round N:
  1. Claude runs generation N (sees Kimi's published strategies)
  2. Kimi runs generation N (sees Claude's published strategies)
  3. Next round

The meta agent — which is itself self-evolving — discovers other agents'
strategies in _repo/strategies/ and decides whether to incorporate them.
No external adoption logic; the agent handles transfer naturally.

Usage:
    python -m transfer_experiment.run_shared --iterations 50 --seed 42
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import random

from markspace import (
    Agent,
    Guard,
    MarkSpace,
    MarkType,
    Observation,
    Scope,
    Source,
    ConflictPolicy,
    DecayConfig,
    hours,
)

from replication.agent.llm_client import META_MODEL, EVAL_MODEL, set_audit_log, cleanup_clients
from replication.generate_loop import _propagate_shared_state, _rewrite_imports
from transfer_experiment.mark_tools import MarkBackedCodebase
from transfer_experiment.mark_loop import (
    _init_codebase_from_dir,
    _evaluate_from_marks,
    _materialize_and_load,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = str(Path(__file__).parent.parent / "replication" / "data")
INITIAL_AGENT_DIR = str(Path(__file__).parent / "initial_agent")


def setup_markspace(agent_names: list[str]):
    """Create markspace with codebase and strategy scopes + telemetry."""
    from markspace.telemetry import InMemorySink

    codebase_scopes = [
        Scope(name=f"codebase/{name}",
              allowed_intent_verbs=("edit", "execute"), allowed_action_verbs=("edited", "executed"),
              conflict_policy=ConflictPolicy.HIGHEST_CONFIDENCE,
              decay=DecayConfig(observation_half_life=hours(24), warning_half_life=hours(12), intent_ttl=hours(1)))
        for name in agent_names
    ]
    strategy_scope = Scope(
        name="strategies",
        allowed_intent_verbs=("publish",), allowed_action_verbs=("published",),
        conflict_policy=ConflictPolicy.HIGHEST_CONFIDENCE,
        decay=DecayConfig(observation_half_life=hours(24), warning_half_life=hours(12), intent_ttl=hours(1)),
    )
    space = MarkSpace(scopes=codebase_scopes + [strategy_scope])
    sink = InMemorySink()
    guard = Guard(space, telemetry=sink)
    return space, guard, sink


def publish_strategy(guard, agent, model_name, val_score, gen, codebase):
    """Publish improved strategy to strategies scope."""
    task_code = codebase.read_file("task_agent.py") or ""
    meta_code = codebase.read_file("meta_agent.py") or ""
    guard.write_mark(agent, Observation(
        scope="strategies", topic=f"{model_name}_gen{gen}",
        content=json.dumps({"model": model_name, "generation": gen,
                            "val_score": val_score, "task_agent": task_code,
                            "meta_agent": meta_code}),
        confidence=val_score, source=Source.FLEET,
    ))
    logger.info("[markspace] %s published gen_%d (val=%.3f)", model_name, gen, val_score)


def snapshot_strategies(space, sink=None) -> list[Observation]:
    """Capture current strategies from markspace.

    Call once at the top of each round so both agents see the same
    state — neither sees the other's output from the current round.
    """
    marks = space.read(scope="strategies", mark_type=MarkType.OBSERVATION)
    obs = [m for m in marks if isinstance(m, Observation)]
    if sink and hasattr(sink, 'record_counter'):
        sink.record_counter("markspace.marks.read", len(obs), {"scope": "strategies"})
        sink.record_gauge("markspace.space.active_marks", len(marks), {"scope": "strategies"})
    return obs


def materialize_strategies(snapshot: list[Observation], target_dir: str, model_name: str):
    """Write other agents' published strategies to a directory the meta agent can see.

    Creates target_dir/{model}_gen{N}/ with task_agent.py, meta_agent.py,
    and a metadata.json containing the val_score.  Only includes strategies
    from *other* models so the meta agent sees what's available from peers.
    """
    # Keep only the highest-scoring strategy per peer model.
    best_per_model: dict[str, Observation] = {}
    for m in snapshot:
        strategy = json.loads(m.content)
        if strategy["model"] == model_name:
            continue
        prev = best_per_model.get(strategy["model"])
        if prev is None or m.confidence > prev.confidence:
            best_per_model[strategy["model"]] = m
    count = 0
    for m in best_per_model.values():
        strategy = json.loads(m.content)
        dirname = f"{strategy['model']}_gen{strategy['generation']}"
        sdir = os.path.join(target_dir, dirname)
        os.makedirs(sdir, exist_ok=True)
        with open(os.path.join(sdir, "task_agent.py"), "w") as f:
            f.write(strategy["task_agent"])
        if strategy.get("meta_agent"):
            with open(os.path.join(sdir, "meta_agent.py"), "w") as f:
                f.write(strategy["meta_agent"])
        with open(os.path.join(sdir, "metadata.json"), "w") as f:
            json.dump({"model": strategy["model"],
                        "generation": strategy["generation"],
                        "val_score": strategy["val_score"]}, f, indent=2)
        count += 1
    if count:
        logger.info("[%s] Materialized %d peer strategies into %s", model_name, count, target_dir)
    return count


def run_one_gen(model_name, meta_model, codebase, output_dir, gen, max_gens,
                strategy_snapshot: list[Observation] | None = None):
    """Run one generation: meta agent modifies code, evaluate."""
    gen_dir = os.path.join(output_dir, f"gen_{gen}")
    os.makedirs(gen_dir, exist_ok=True)
    set_audit_log(os.path.join(gen_dir, "llm_calls.jsonl"))

    train_path = os.path.join(os.path.abspath(DATA_DIR), "train.csv")
    val_path = os.path.join(os.path.abspath(DATA_DIR), "val.csv")

    tmpdir = None
    try:
        # Materialize full codebase for module loading, but create a
        # separate "repo" dir containing ONLY the modifiable files.
        # The meta agent sees only task_agent.py and meta_agent.py.
        # Infrastructure (agent/) is loaded for execution but not shown.
        tmpdir = tempfile.mkdtemp()
        codebase.materialize(tmpdir)

        repo_dir = os.path.join(tmpdir, "_repo")
        os.makedirs(repo_dir, exist_ok=True)
        for fname in ["task_agent.py", "meta_agent.py"]:
            src = os.path.join(tmpdir, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(repo_dir, fname))

        # Load meta agent from the full tmpdir (needs agent/ for imports)
        if tmpdir not in sys.path:
            sys.path.insert(0, tmpdir)
        stale = [k for k in sys.modules if k == "agent" or k.startswith("agent.")]
        for k in stale:
            del sys.modules[k]

        import importlib.util
        meta_path = os.path.join(tmpdir, "meta_agent.py")
        spec = importlib.util.spec_from_file_location("meta_module", meta_path)
        if spec is None or spec.loader is None:
            raise ImportError("No meta_agent.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _propagate_shared_state()

        if not hasattr(mod, "MetaAgent"):
            raise ImportError("No MetaAgent class")

        meta = mod.MetaAgent(model=meta_model, temperature=0.0)
        # Copy eval results into _repo/evals/ so the meta agent can
        # discover them (same as paper copying evals into Docker container).
        # The meta agent can read these but only edit task_agent.py/meta_agent.py.
        evals_dst = os.path.join(repo_dir, "evals")
        if os.path.exists(output_dir):
            for gdir in sorted(glob.glob(os.path.join(output_dir, "gen_*/"))):
                gname = os.path.basename(gdir.rstrip("/"))
                for subdir in ["eval_train", "eval_val"]:
                    src = os.path.join(gdir, subdir)
                    if os.path.exists(src):
                        dst = os.path.join(evals_dst, gname, subdir)
                        if not os.path.exists(dst):
                            shutil.copytree(src, dst)

        # Materialize peer strategies so the meta agent can discover them.
        # These are read-only — the meta agent can browse and compare but
        # only edit its own task_agent.py and meta_agent.py.
        if strategy_snapshot:
            strategies_dir = os.path.join(repo_dir, "strategies")
            materialize_strategies(strategy_snapshot, strategies_dir, model_name)

        sys.modules["agent.tools.bash_tool"].set_allowed_root(repo_dir)
        sys.modules["agent.tools.editor_tool"].set_allowed_root(repo_dir)

        # Meta agent sees task_agent.py, meta_agent.py, evals/, and strategies/
        meta.forward(repo_path=repo_dir, eval_path=os.path.dirname(output_dir),
                     iterations_left=max_gens - gen)

        # Read modified files back into marks
        for fname in ["task_agent.py", "meta_agent.py"]:
            fpath = os.path.join(repo_dir, fname)
            if os.path.exists(fpath):
                new_content = open(fpath).read()
                old_content = codebase.read_file(fname)
                if new_content != old_content:
                    codebase.write_file(fname, new_content)

    except Exception as e:
        logger.warning("[%s] Gen %d: meta agent failed: %s", model_name, gen, e)
        with open(os.path.join(gen_dir, "metadata.json"), "w") as f:
            json.dump({"valid": False, "error": str(e)}, f, indent=2)
        return None, None
    finally:
        repo_bash = sys.modules.get("agent.tools.bash_tool")
        if repo_bash and hasattr(repo_bash, "reset_session"):
            repo_bash.reset_session()
        if tmpdir and os.path.exists(tmpdir):
            shutil.rmtree(tmpdir, ignore_errors=True)
        if tmpdir and tmpdir in sys.path:
            sys.path.remove(tmpdir)

    # Evaluate
    train_result = _evaluate_from_marks(codebase, train_path, os.path.join(gen_dir, "eval_train"), EVAL_MODEL)
    val_result = _evaluate_from_marks(codebase, val_path, os.path.join(gen_dir, "eval_val"), EVAL_MODEL)

    train_raw = train_result[0] if train_result else 0.0
    val_raw = val_result[0] if val_result else 0.0

    # Save snapshot and metadata
    codebase.materialize(os.path.join(gen_dir, "repo"))
    with open(os.path.join(gen_dir, "metadata.json"), "w") as f:
        json.dump({"valid": True, "train_score": train_raw, "val_score": val_raw}, f, indent=2)

    logger.info("[%s] Gen %d: train=%.3f val=%.3f", model_name, gen, train_raw, val_raw)
    return train_raw, val_raw


def _short_name(model_id: str) -> str:
    """Derive a short agent name from a model identifier."""
    # "claude-sonnet-4-6" -> "claude"
    # "accounts/fireworks/routers/kimi-k2p5-turbo" -> "kimi"
    # "gpt-oss-120b" -> "gpt-oss-120b"
    base = model_id.rsplit("/", 1)[-1]
    # Strip provider suffixes like ":free"
    base = base.split(":")[0]
    for prefix in ("claude", "kimi", "gpt", "deepseek", "gemini", "glm", "qwen"):
        if base.startswith(prefix):
            return base.split("-turbo")[0].split("-sonnet")[0].split("-opus")[0]
    return base


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--models", nargs=2, metavar="MODEL",
                   default=["claude-sonnet-4-6", META_MODEL],
                   help="Two model identifiers to pit against each other")
    p.add_argument("--budget", nargs=2, type=int, metavar="N",
                   default=None,
                   help="Per-model iteration budget (defaults to --iterations for both)")
    args = p.parse_args()

    model_a, model_b = args.models
    name_a, name_b = _short_name(model_a), _short_name(model_b)
    if name_a == name_b:
        name_b = name_b + "_b"
    budget_a, budget_b = (args.budget if args.budget
                          else [args.iterations, args.iterations])

    random.seed(args.seed)
    np.random.seed(args.seed)
    cleanup_clients()

    space, guard, telemetry_sink = setup_markspace([name_a, name_b])

    # Create agents — each owns its codebase scope and can read the other's
    agents = {}
    codebases = {}
    for name, peer in [(name_a, name_b), (name_b, name_a)]:
        agents[name] = Agent(
            name=name,
            scopes={f"codebase/{name}": ["intent", "action", "observation"],
                    "strategies": ["intent", "action", "observation"]},
            read_scopes=frozenset({f"codebase/{peer}", "strategies"}),
        )
        codebases[name] = MarkBackedCodebase(space, guard, agents[name], f"codebase/{name}")

    pair_label = f"{name_a}_vs_{name_b}"
    base_output = f"transfer_experiment/results/{pair_label}/seed{args.seed}"
    outputs = {}
    for name in [name_a, name_b]:
        outputs[name] = os.path.join(base_output, name)
        os.makedirs(outputs[name], exist_ok=True)

    # Resume logic: check for existing generations
    import tempfile, shutil as _shutil, platform

    bests = {name_a: 0.0, name_b: 0.0}
    archives = {name_a: [], name_b: []}
    resume_from = 0

    # Check if we can resume from existing data
    existing_gens = {}
    for name in [name_a, name_b]:
        gens = sorted(glob.glob(os.path.join(outputs[name], "gen_*/metadata.json")),
                      key=lambda x: int(x.split("gen_")[1].split("/")[0]))
        existing_gens[name] = []
        for gf in gens:
            g = int(gf.split("gen_")[1].split("/")[0])
            m = json.load(open(gf))
            v = m.get("val_score", 0) or 0
            if m.get("valid", True):
                existing_gens[name].append(g)
                bests[name] = max(bests[name], v)
                archives[name].append(g)

    if existing_gens[name_a] and existing_gens[name_b]:
        resume_from = min(max(existing_gens[name_a]),
                          max(existing_gens[name_b]))
        logger.info("Resuming from gen %d (best: %s=%.3f, %s=%.3f)",
                    resume_from, name_a, bests[name_a], name_b, bests[name_b])

    if resume_from == 0:
        # Fresh start: initialize codebases and evaluate gen 0
        _init_tmp = tempfile.mkdtemp()
        _shutil.copy2(os.path.join(INITIAL_AGENT_DIR, "task_agent.py"), os.path.join(_init_tmp, "task_agent.py"))
        _shutil.copy2(os.path.join(INITIAL_AGENT_DIR, "meta_agent.py"), os.path.join(_init_tmp, "meta_agent.py"))
        _shutil.copytree(os.path.join(INITIAL_AGENT_DIR, "agent"), os.path.join(_init_tmp, "agent"))
        for name in [name_a, name_b]:
            _init_codebase_from_dir(codebases[name], _init_tmp)
        _shutil.rmtree(_init_tmp)

        for name in [name_a, name_b]:
            gen0_dir = os.path.join(outputs[name], "gen_0")
            os.makedirs(gen0_dir, exist_ok=True)
            set_audit_log(os.path.join(gen0_dir, "llm_calls.jsonl"))
            val_result = _evaluate_from_marks(codebases[name], os.path.join(DATA_DIR, "val.csv"),
                                               os.path.join(gen0_dir, "eval_val"), EVAL_MODEL)
            val_raw = val_result[0] if val_result else 0.0
            codebases[name].materialize(os.path.join(gen0_dir, "repo"))
            with open(os.path.join(gen0_dir, "metadata.json"), "w") as f:
                json.dump({"valid": True, "val_score": val_raw, "is_initial": True}, f, indent=2)
            archives[name].append(0)
            logger.info("[%s] Gen 0: val=%.3f", name, val_raw)
    else:
        # Resume: reconstruct codebases and republish best strategies
        for name in [name_a, name_b]:
            last_gen = max(existing_gens[name])
            # Restore codebase from the latest gen's repo
            repo_dir = os.path.join(outputs[name], f"gen_{last_gen}", "repo")
            if os.path.exists(repo_dir):
                _init_codebase_from_dir(codebases[name], repo_dir)
                logger.info("[%s] Restored codebase from gen %d", name, last_gen)

            # Republish the BEST strategy (not latest) so peer can see it
            if bests[name] > 0:
                # Find the gen that achieved the best score
                best_gen = last_gen
                for gf in glob.glob(os.path.join(outputs[name], "gen_*/metadata.json")):
                    g = int(gf.split("gen_")[1].split("/")[0])
                    m = json.load(open(gf))
                    v = m.get("val_score", 0) or 0
                    if v >= bests[name]:
                        best_gen = g
                best_repo = os.path.join(outputs[name], f"gen_{best_gen}", "repo")
                if os.path.exists(best_repo):
                    best_cb = MarkBackedCodebase(space, guard, agents[name], f"codebase/{name}")
                    _init_codebase_from_dir(best_cb, best_repo)
                    publish_strategy(guard, agents[name], name, bests[name],
                                     best_gen, best_cb)
                    logger.info("[%s] Republished best strategy from gen %d (val=%.3f)",
                                name, best_gen, bests[name])

    # Save run config
    for name, model in [(name_a, model_a), (name_b, model_b)]:
        with open(os.path.join(outputs[name], "run_config.json"), "w") as f:
            json.dump({"model": model, "agent_name": name,
                       "seed": args.seed, "iterations": args.iterations,
                       "python_version": platform.python_version()}, f, indent=2)

    # Main loop — alternating rounds
    agent_configs = [
        (name_a, model_a, budget_a),
        (name_b, model_b, budget_b),
    ]
    start_gen = resume_from + 1 if resume_from > 0 else 1
    logger.info("=== Starting alternating rounds: %s vs %s (gen %d-%d) ===",
                name_a, name_b, start_gen, args.iterations)
    for gen in range(start_gen, args.iterations + 1):
        logger.info("--- Round %d ---", gen)

        # Snapshot strategies BEFORE either agent runs this round.
        strat_snapshot = snapshot_strategies(space, sink=telemetry_sink)

        for name, model, budget in agent_configs:
            if gen > budget:
                continue
            train, val = run_one_gen(name, model, codebases[name],
                                     outputs[name], gen, budget,
                                     strategy_snapshot=strat_snapshot)
            if val is not None:
                archives[name].append(gen)
                if val > bests[name]:
                    bests[name] = val
                    publish_strategy(guard, agents[name], name, val, gen, codebases[name])

    # Save summaries
    for name in [name_a, name_b]:
        with open(os.path.join(outputs[name], "summary.json"), "w") as f:
            json.dump({"model": name, "archive_size": len(archives[name]),
                       "best_val": bests[name]}, f, indent=2)

    # Telemetry
    events = telemetry_sink.events
    accepted = sum(1 for e in events if e.verdict == "accepted")
    denied = sum(1 for e in events if e.verdict in ("denied", "blocked"))

    logger.info("")
    logger.info("=== RESULTS ===")
    all_marks = [m for m in space.read(scope="strategies", mark_type=MarkType.OBSERVATION)
                 if isinstance(m, Observation)]
    logger.info("Strategies in markspace: %d", len(all_marks))
    for m in sorted(all_marks, key=lambda x: -x.confidence)[:5]:
        s = json.loads(m.content)
        logger.info("  %s gen_%d: val=%.3f", s["model"], s["generation"], s["val_score"])

    logger.info("")
    logger.info("=== MARKSPACE TELEMETRY ===")
    logger.info("Guard events: %d (accepted=%d, denied=%d)", len(events), accepted, denied)

    # Save telemetry (full event fields)
    with open(os.path.join(base_output, "markspace_telemetry.jsonl"), "w") as f:
        for e in events:
            entry = {
                "timestamp": e.timestamp, "agent_id": str(e.agent_id),
                "operation": e.operation, "scope": e.scope,
                "verdict": e.verdict, "reason": e.reason,
            }
            # Include optional fields if present
            for field in ("mark_type", "conflict_check", "conflict_found",
                          "barrier_restricted", "input_tokens_this_round",
                          "output_tokens_this_round", "budget_remaining_input",
                          "budget_remaining_output", "extra"):
                val = getattr(e, field, None)
                if val is not None:
                    entry[field] = val
            f.write(json.dumps(entry, default=str) + "\n")

    # Save counters and gauges from sink
    if hasattr(telemetry_sink, 'counters'):
        counters = telemetry_sink.counters
        gauges = telemetry_sink.gauges if hasattr(telemetry_sink, 'gauges') else []
        with open(os.path.join(base_output, "markspace_metrics.json"), "w") as f:
            json.dump({
                "counters": counters if isinstance(counters, (list, dict)) else [],
                "gauges": gauges if isinstance(gauges, (list, dict)) else [],
            }, f, indent=2, default=str)

    cleanup_clients()


if __name__ == "__main__":
    main()
