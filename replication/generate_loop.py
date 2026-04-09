"""
DGM-H generate loop for IMO grading.

Reimplemented from facebookresearch/HyperAgents generate_loop.py.
Same archive, same parent selection, same fork+modify+evaluate cycle.

Key difference from our original implementation: agents are FILE-BASED.
The meta agent modifies files on disk using bash + editor tools.
The task agent is loaded from the modified task_agent.py file.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

import importlib.util

from replication.evaluation.harness import run_harness
from replication.evaluation.report import compute_report
from replication.agent.llm_client import META_MODEL, EVAL_MODEL, set_audit_log, cleanup_clients

logger = logging.getLogger(__name__)


def _rewrite_imports(filepath: str) -> None:
    """Rewrite `from replication.agent.` → `from agent.` in a Python file."""
    content = open(filepath).read()
    rewritten = content.replace("from replication.agent.", "from agent.")
    rewritten = rewritten.replace("import replication.agent.", "import agent.")
    if rewritten != content:
        with open(filepath, "w") as f:
            f.write(rewritten)


def _copy_and_rewrite(src: str, dst: str) -> None:
    """Copy a file and rewrite replication imports."""
    shutil.copy2(src, dst)
    _rewrite_imports(dst)


def _propagate_shared_state():
    """Propagate shared state from main llm_client to repo's copy.

    After loading repo modules, agent.llm_client is a separate module instance
    with its own globals. We share the main package's mutable state so:
    - Audit logging works (shared _audit_log_path, _audit_lock)
    - LLM clients are reused across generations (shared _clients dict)
    """
    import sys
    repo_llm = sys.modules.get("agent.llm_client")
    main_llm = sys.modules.get("replication.agent.llm_client")
    if repo_llm is not None and main_llm is not None and repo_llm is not main_llm:
        repo_llm._audit_log_path = main_llm._audit_log_path
        repo_llm._audit_lock = main_llm._audit_lock
        repo_llm._clients = main_llm._clients  # reuse HTTP connections


def _load_from_repo(filepath: str, module_name: str):
    """Load a Python module from a repo, with the repo dir on sys.path.

    This ensures `from agent.xxx` imports resolve to the repo's agent/
    subfolder, so modifications by the meta agent take effect.
    """
    import sys
    repo_dir = os.path.dirname(os.path.abspath(filepath))
    added = False
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
        added = True
    try:
        # Clear any cached agent modules so modified versions are picked up
        stale = [k for k in sys.modules if k == "agent" or k.startswith("agent.")]
        for k in stale:
            del sys.modules[k]
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load from {filepath}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _propagate_shared_state()
        return mod
    finally:
        if added:
            sys.path.remove(repo_dir)


def _load_meta_agent(meta_agent_path: str, model: str, temperature: float = 0.0):
    """Load MetaAgent from a file path (may be modified by previous generations)."""
    mod = _load_from_repo(meta_agent_path, "meta_module")
    if not hasattr(mod, "MetaAgent"):
        raise AttributeError(f"No MetaAgent in {meta_agent_path}")
    return mod.MetaAgent(model=model, temperature=temperature)


def _run_candidate(
    fork_from_repo: str,
    candidate_dir: str,
    output_dir: str,
    meta_model: str,
    meta_temperature: float,
    eval_model: str,
    train_path: str,
    staged_eval_samples: int,
    staged_eval_threshold: float,
    freeze_meta: bool,
    max_generations: int,
    gen: int,
    label: str,
) -> tuple[float, str] | None:
    """Fork a repo, run meta agent, evaluate on staged train.

    Returns (staged_train_score, child_repo_path) or None on failure.
    Used by the annealing tournament to compare blind vs directed candidates.
    """
    import sys

    child_repo = os.path.join(candidate_dir, "repo")
    if os.path.exists(child_repo):
        shutil.rmtree(child_repo)
    shutil.copytree(fork_from_repo, child_repo)

    if freeze_meta:
        meta_agent_path = os.path.join(output_dir, "gen_0", "repo", "meta_agent.py")
    else:
        meta_agent_path = os.path.join(child_repo, "meta_agent.py")

    try:
        meta = _load_meta_agent(meta_agent_path, meta_model, meta_temperature)
        sys.modules["agent.tools.bash_tool"].set_allowed_root(output_dir)
        sys.modules["agent.tools.editor_tool"].set_allowed_root(output_dir)

        meta.forward(
            repo_path=child_repo,
            eval_path=output_dir,
            iterations_left=max_generations - gen,
        )
    except Exception as e:
        logger.warning("%s candidate failed (meta agent): %s", label, e)
        return None
    finally:
        repo_bash = sys.modules.get("agent.tools.bash_tool")
        if repo_bash and hasattr(repo_bash, "reset_session"):
            repo_bash.reset_session()

    child_agent = os.path.join(child_repo, "task_agent.py")
    if not os.path.exists(child_agent):
        logger.warning("%s candidate: task_agent.py missing", label)
        return None

    result = _evaluate_agent(
        agent_path=child_agent,
        dataset_path=train_path,
        output_dir=os.path.join(candidate_dir, "eval_train"),
        model=eval_model,
        staged_samples=staged_eval_samples,
        staged_threshold=staged_eval_threshold,
    )
    if result is None:
        logger.warning("%s candidate: eval failed", label)
        return None

    score, _ = result
    logger.info("%s candidate: train_score=%.3f", label, score)
    return score, child_repo


def select_parent(
    archive: list[int],
    scores: dict[int, float],
    child_counts: dict[int, int],
    valid_parents: set[int] | None = None,
    method: str = "score_child_prop",
) -> int:
    """Select a parent from the archive.

    Matches paper's select_parent from gl_utils.py.
    method='score_child_prop' is the paper's default (sigmoid x child penalty).
    Only considers nodes in valid_parents (paper's valid_parent metadata).
    """
    candidates = {
        g: scores.get(g, 0.0)
        for g in archive
        if scores.get(g) is not None and (valid_parents is None or g in valid_parents)
    }

    if not candidates:
        return archive[0]

    if method == "random":
        return random.choice(list(candidates.keys()))

    if method == "best":
        return max(candidates, key=candidates.get)

    if method == "score_child_prop":
        genids = list(candidates.keys())
        raw_scores = [candidates[g] for g in genids]

        # Sigmoid normalization around top-3 midpoint
        top3 = sorted(raw_scores, reverse=True)[:3]
        mid = np.mean(top3) if top3 else 0.0
        sigmoid = [1.0 / (1.0 + math.exp(-10 * (s - mid))) for s in raw_scores]

        # Child count penalty (paper uses exp(-(n/8)^3))
        penalties = [
            math.exp(-((child_counts.get(g, 0) / 8) ** 3)) for g in genids
        ]

        combined = [s * p for s, p in zip(sigmoid, penalties)]
        total = sum(combined)
        probs = [c / total for c in combined] if total > 0 else None
        return random.choices(genids, weights=probs, k=1)[0]

    raise ValueError(f"Unknown selection method: {method}")


def generate_loop(
    output_dir: str,
    data_dir: str,
    initial_agent_path: str,
    initial_meta_path: str | None = None,
    max_generations: int = 200,
    meta_model: str = META_MODEL,
    meta_temperature: float = 0.0,
    eval_model: str = EVAL_MODEL,
    selection_method: str = "score_child_prop",
    staged_eval_samples: int = 10,
    staged_eval_threshold: float = 0.0,
    seed: int = 42,
    freeze_meta: bool = False,
    annealing: bool = False,
    annealing_p0: float = 0.5,
    annealing_decay: float = 0.9,
) -> dict:
    """Run the DGM-H generate loop.

    Each generation:
    1. Select parent from archive
    2. Fork: copy parent's repo
    3. Run meta agent to modify the repo (unless freeze_meta)
    4. Evaluate: run modified task_agent.py on train set (staged)
    5. If valid: add to archive, evaluate on val set
    6. Select next parent

    Args:
        output_dir: where to save all outputs
        data_dir: directory with train.csv, val.csv, test.csv
        initial_agent_path: path to initial task_agent.py
        max_generations: total iterations
        meta_model: model for meta agent
        eval_model: model for task agent evaluation
        selection_method: parent selection algorithm
        staged_eval_samples: first N samples for staged eval
        staged_eval_threshold: minimum score to continue full eval
        seed: random seed
        freeze_meta: if True, don't run meta agent (DGM-H w/o self-improve)

    Returns:
        Summary dict
    """
    random.seed(seed)
    np.random.seed(seed)
    cleanup_clients()  # fresh clients on every run (avoids poisoned circuit breaker on resume)
    # Use absolute paths so bash/editor tools work correctly
    output_dir = os.path.abspath(output_dir)
    data_dir = os.path.abspath(data_dir)
    initial_agent_path = os.path.abspath(initial_agent_path)
    os.makedirs(output_dir, exist_ok=True)

    # Save full run configuration for traceability
    import platform
    run_config = {
        "timestamp": datetime.now().isoformat(),
        "seed": seed,
        "max_generations": max_generations,
        "meta_model": meta_model,
        "meta_temperature": meta_temperature,
        "eval_model": eval_model,
        "selection_method": selection_method,
        "staged_eval_samples": staged_eval_samples,
        "staged_eval_threshold": staged_eval_threshold,
        "freeze_meta": freeze_meta,
        "annealing": annealing,
        "annealing_p0": annealing_p0,
        "annealing_decay": annealing_decay,
        "initial_agent_path": initial_agent_path,
        "data_dir": data_dir,
        "output_dir": output_dir,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    with open(os.path.join(output_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)
    logger.info("Run config saved: %s", run_config)

    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    test_path = os.path.join(data_dir, "test.csv")

    import pandas as pd
    val_total = len(pd.read_csv(val_path, dtype=str))

    archive_log = os.path.join(output_dir, "archive.jsonl")
    annealing_p = annealing_p0 if annealing else 0.0

    # Resume: if archive.jsonl exists, reconstruct state from metadata files
    start_gen = 1
    if os.path.exists(archive_log):
        with open(archive_log) as f:
            lines = f.readlines()
        if lines:
            last_entry = json.loads(lines[-1])
            archive = last_entry["archive"]
            start_gen = last_entry["current_genid"] + 1

            # Reconstruct scores, child_counts, valid_parents from metadata
            scores: dict[int, float] = {}
            child_counts: dict[int, int] = {g: 0 for g in archive}
            valid_parents: set[int] = set()

            for g in archive:
                meta_path = os.path.join(output_dir, f"gen_{g}", "metadata.json")
                if os.path.exists(meta_path):
                    m = json.load(open(meta_path))
                    if m.get("valid"):
                        v = m.get("val_score", 0) or 0
                        full = m.get("run_full_eval", False)
                        scores[g] = _get_saved_score(v, full, staged_eval_samples, val_total)
                        valid_parents.add(g)
                    parent = m.get("parent")
                    if parent is not None and parent in child_counts:
                        child_counts[parent] += 1

            # Reconstruct annealing_p from tournament count
            if annealing:
                tournament_count = sum(
                    1 for g in archive if g > 0 and os.path.exists(os.path.join(output_dir, f"gen_{g}", "metadata.json"))
                    and json.load(open(os.path.join(output_dir, f"gen_{g}", "metadata.json"))).get("tournament")
                )
                annealing_p = annealing_p0 * (annealing_decay ** tournament_count)

            logger.info("Resumed from gen %d. Archive: %d nodes, %d valid parents, best_val=%.3f, annealing_p=%.4f",
                        start_gen, len(archive), len(valid_parents),
                        max(scores.values()) if scores else 0.0, annealing_p)

    if start_gen <= 1:
        # Fresh start
        archive = [0]
        scores = {}
        child_counts = {0: 0}
        valid_parents = {0}

        gen0_dir = os.path.join(output_dir, "gen_0")
        os.makedirs(gen0_dir, exist_ok=True)
        repo_dir = os.path.join(gen0_dir, "repo")
        os.makedirs(repo_dir, exist_ok=True)

        # Copy the full agent codebase into the repo, rewriting imports so that
        # modifications to agent/ files in the repo take effect when loaded via
        # importlib. The paper runs inside Docker where the repo IS the Python
        # path, so `from agent.xxx` resolves to the repo. We replicate this by
        # rewriting `from replication.agent.` → `from agent.` in copied files,
        # and adding the repo dir to sys.path at load time.
        replication_root = os.path.dirname(os.path.abspath(__file__))
        _copy_and_rewrite(initial_agent_path, os.path.join(repo_dir, "task_agent.py"))
        meta_src = initial_meta_path or os.path.join(replication_root, "meta_agent.py")
        _copy_and_rewrite(meta_src, os.path.join(repo_dir, "meta_agent.py"))
        agent_src = os.path.join(replication_root, "agent")
        agent_dst = os.path.join(repo_dir, "agent")
        if os.path.exists(agent_dst):
            shutil.rmtree(agent_dst)
        shutil.copytree(agent_src, agent_dst)
        for root, _, files in os.walk(agent_dst):
            for fname in files:
                if fname.endswith(".py"):
                    fpath = os.path.join(root, fname)
                    _rewrite_imports(fpath)

        # Evaluate initial agent on train (staged) then val
        logger.info("Evaluating initial agent (gen 0)")
        set_audit_log(os.path.join(gen0_dir, "llm_calls.jsonl"))
        train_result = _evaluate_agent(
            agent_path=os.path.join(repo_dir, "task_agent.py"),
            dataset_path=train_path,
            output_dir=os.path.join(gen0_dir, "eval_train"),
            model=eval_model,
            staged_samples=staged_eval_samples,
            staged_threshold=staged_eval_threshold,
        )
        val_result = _evaluate_agent(
            agent_path=os.path.join(repo_dir, "task_agent.py"),
            dataset_path=val_path,
            output_dir=os.path.join(gen0_dir, "eval_val"),
            model=eval_model,
            staged_samples=staged_eval_samples,
            staged_threshold=0.0,
        )
        if val_result is not None:
            val_raw, val_full = val_result
            scores[0] = _get_saved_score(val_raw, val_full, staged_eval_samples, val_total)
        else:
            val_raw, val_full = 0.0, False
            scores[0] = 0.0
        train_raw = train_result[0] if train_result else 0.0
        train_full = train_result[1] if train_result else False
        logger.info("Gen 0: train=%.3f val=%.3f (saved=%.4f)", train_raw, val_raw, scores[0])

        _save_metadata(gen0_dir, {
            "parent": None,
            "valid": True,
            "train_score": train_raw,
            "val_score": val_raw,
            "run_full_eval": val_full,
            "is_initial": True,
        })
        _log_archive(archive_log, 0, archive)

    # Main loop
    for gen in range(start_gen, max_generations + 1):
        logger.info("=" * 50)
        logger.info("Generation %d / %d (archive size=%d)", gen, max_generations, len(archive))

        # Select parent
        parent_id = select_parent(archive, scores, child_counts, valid_parents, method=selection_method)
        logger.info("Selected parent: gen_%d (score=%.3f)", parent_id, scores.get(parent_id, 0))

        gen_dir = os.path.join(output_dir, f"gen_{gen}")
        os.makedirs(gen_dir, exist_ok=True)
        set_audit_log(os.path.join(gen_dir, "llm_calls.jsonl"))
        import sys

        # --- Annealing tournament ---
        # With probability p: run BOTH a blind candidate (from gen_0) and a
        # directed candidate (from parent), keep the better one.
        # Otherwise: just run directed (normal algorithm).
        tournament = annealing and random.random() < annealing_p
        if tournament:
            logger.info("Annealing TOURNAMENT (p=%.4f): blind vs directed", annealing_p)
            parent_repo = os.path.join(output_dir, f"gen_{parent_id}", "repo")
            initial_repo = os.path.join(output_dir, "gen_0", "repo")

            blind_dir = os.path.join(gen_dir, "blind")
            directed_dir = os.path.join(gen_dir, "directed")
            os.makedirs(blind_dir, exist_ok=True)
            os.makedirs(directed_dir, exist_ok=True)

            common_args = dict(
                output_dir=output_dir, meta_model=meta_model, meta_temperature=meta_temperature,
                eval_model=eval_model, train_path=train_path, staged_eval_samples=staged_eval_samples,
                staged_eval_threshold=staged_eval_threshold, freeze_meta=freeze_meta,
                max_generations=max_generations, gen=gen,
            )

            blind_result = _run_candidate(
                fork_from_repo=initial_repo, candidate_dir=blind_dir,
                label="blind", **common_args,
            )
            directed_result = _run_candidate(
                fork_from_repo=parent_repo, candidate_dir=directed_dir,
                label="directed", **common_args,
            )

            # Pick winner
            blind_score = blind_result[0] if blind_result else -1.0
            directed_score = directed_result[0] if directed_result else -1.0

            if blind_score > directed_score and blind_result is not None:
                winner, winner_label = "blind", "blind"
                winner_repo = blind_result[1]
                fork_from = 0
                train_raw = blind_score
            elif directed_result is not None:
                winner, winner_label = "directed", "directed"
                winner_repo = directed_result[1]
                fork_from = parent_id
                train_raw = directed_score
            else:
                # Both failed
                logger.info("Gen %d: both tournament candidates failed", gen)
                _save_metadata(gen_dir, {
                    "parent": parent_id, "valid": False, "tournament": True,
                    "blind_score": None, "directed_score": None,
                    "error": "both candidates failed",
                })
                archive.append(gen)
                child_counts[parent_id] = child_counts.get(parent_id, 0) + 1
                child_counts[gen] = 0
                valid_parents.discard(parent_id)
                annealing_p *= annealing_decay
                _log_archive(archive_log, gen, archive)
                continue

            logger.info("TOURNAMENT: %s wins (blind=%.3f, directed=%.3f)",
                        winner_label, blind_score, directed_score)

            # Copy winner's repo to gen_dir/repo (the canonical location)
            child_repo = os.path.join(gen_dir, "repo")
            if os.path.exists(child_repo):
                shutil.rmtree(child_repo)
            shutil.copytree(winner_repo, child_repo)

            # Evaluate winner on val
            child_agent = os.path.join(child_repo, "task_agent.py")
            val_result = _evaluate_agent(
                agent_path=child_agent, dataset_path=val_path,
                output_dir=os.path.join(gen_dir, "eval_val"),
                model=eval_model, staged_samples=staged_eval_samples,
                staged_threshold=0.0,
            )
            if val_result is not None:
                val_raw, val_full = val_result
                val_saved = _get_saved_score(val_raw, val_full, staged_eval_samples, val_total)
            else:
                val_raw, val_full = 0.0, False
                val_saved = 0.0

            archive.append(gen)
            scores[gen] = val_saved
            child_counts[parent_id] = child_counts.get(parent_id, 0) + 1
            child_counts[gen] = 0
            valid_parents.add(gen)

            logger.info(
                "Gen %d: winner=%s train=%.3f val=%.3f (saved=%.4f, parent=%d, archive=%d)",
                gen, winner_label, train_raw, val_raw, val_saved, parent_id, len(archive),
            )
            _save_metadata(gen_dir, {
                "parent": parent_id, "forked_from": fork_from, "valid": True,
                "tournament": True, "winner": winner_label,
                "blind_score": blind_score if blind_result else None,
                "directed_score": directed_score if directed_result else None,
                "train_score": train_raw, "val_score": val_raw,
                "run_full_eval": val_full,
            })
            annealing_p *= annealing_decay
            logger.info("Annealing p decayed to %.4f", annealing_p)
            _log_archive(archive_log, gen, archive)
            continue

        # --- Normal (non-tournament) path ---
        # Fork parent's repo
        fork_from = parent_id
        parent_repo = os.path.join(output_dir, f"gen_{parent_id}", "repo")
        child_repo = os.path.join(gen_dir, "repo")

        if os.path.exists(child_repo):
            shutil.rmtree(child_repo)
        shutil.copytree(parent_repo, child_repo)

        # Run meta agent
        if freeze_meta:
            meta_agent_path = os.path.join(output_dir, "gen_0", "repo", "meta_agent.py")
        else:
            meta_agent_path = os.path.join(child_repo, "meta_agent.py")

        logger.info("Running meta agent from %s on %s", meta_agent_path, child_repo)

        try:
            meta = _load_meta_agent(meta_agent_path, meta_model, meta_temperature)
            sys.modules["agent.tools.bash_tool"].set_allowed_root(output_dir)
            sys.modules["agent.tools.editor_tool"].set_allowed_root(output_dir)

            meta.forward(
                repo_path=child_repo,
                eval_path=output_dir,
                iterations_left=max_generations - gen,
            )
        except Exception as e:
            logger.warning("Meta agent failed: %s", e)
            _save_metadata(gen_dir, {
                "parent": parent_id, "forked_from": fork_from,
                "valid": False, "error": str(e),
            })
            archive.append(gen)
            child_counts[parent_id] = child_counts.get(parent_id, 0) + 1
            child_counts[gen] = 0
            valid_parents.discard(parent_id)
            _log_archive(archive_log, gen, archive)
            logger.info("Gen %d failed (meta agent). Marked gen_%d as invalid parent. archive=%d",
                        gen, parent_id, len(archive))
            continue
        finally:
            repo_bash = sys.modules.get("agent.tools.bash_tool")
            if repo_bash and hasattr(repo_bash, "reset_session"):
                repo_bash.reset_session()

        # Check task_agent.py exists
        child_agent = os.path.join(child_repo, "task_agent.py")
        if not os.path.exists(child_agent):
            logger.warning("Gen %d: task_agent.py missing after meta agent", gen)
            _save_metadata(gen_dir, {"parent": parent_id, "forked_from": fork_from, "valid": False, "error": "task_agent.py missing"})
            archive.append(gen)
            child_counts[parent_id] = child_counts.get(parent_id, 0) + 1
            child_counts[gen] = 0
            valid_parents.discard(parent_id)
            _log_archive(archive_log, gen, archive)
            logger.info("Gen %d failed (task_agent.py missing). Marked gen_%d as invalid parent. archive=%d",
                        gen, parent_id, len(archive))
            continue

        # Evaluate on train (staged)
        train_result = _evaluate_agent(
            agent_path=child_agent, dataset_path=train_path,
            output_dir=os.path.join(gen_dir, "eval_train"),
            model=eval_model, staged_samples=staged_eval_samples,
            staged_threshold=staged_eval_threshold,
        )
        if train_result is None:
            logger.info("Gen %d: evaluation failed", gen)
            _save_metadata(gen_dir, {"parent": parent_id, "forked_from": fork_from, "valid": False, "error": "eval failed"})
            archive.append(gen)
            child_counts[parent_id] = child_counts.get(parent_id, 0) + 1
            child_counts[gen] = 0
            valid_parents.discard(parent_id)
            _log_archive(archive_log, gen, archive)
            logger.info("Gen %d failed (eval). Marked gen_%d as invalid parent. archive=%d",
                        gen, parent_id, len(archive))
            continue

        train_raw, train_full = train_result

        # Evaluate on val
        val_result = _evaluate_agent(
            agent_path=child_agent, dataset_path=val_path,
            output_dir=os.path.join(gen_dir, "eval_val"),
            model=eval_model, staged_samples=staged_eval_samples,
            staged_threshold=0.0,
        )
        if val_result is not None:
            val_raw, val_full = val_result
            val_saved = _get_saved_score(val_raw, val_full, staged_eval_samples, val_total)
        else:
            val_raw, val_full = 0.0, False
            val_saved = 0.0

        # Add to archive
        archive.append(gen)
        scores[gen] = val_saved
        child_counts[parent_id] = child_counts.get(parent_id, 0) + 1
        child_counts[gen] = 0
        valid_parents.add(gen)

        logger.info(
            "Gen %d: train=%.3f val=%.3f (saved=%.4f, parent=%d, archive=%d)",
            gen, train_raw, val_raw, val_saved, parent_id, len(archive),
        )
        _save_metadata(gen_dir, {
            "parent": parent_id, "forked_from": fork_from,
            "valid": True, "train_score": train_raw, "val_score": val_raw,
            "run_full_eval": val_full,
        })
        _log_archive(archive_log, gen, archive)

    # Final evaluation: best agent on test set
    best_gen = max(archive, key=lambda g: scores.get(g, 0))
    best_agent = os.path.join(output_dir, f"gen_{best_gen}", "repo", "task_agent.py")

    logger.info("Best agent: gen_%d (val=%.3f). Evaluating on test set...", best_gen, scores[best_gen])
    test_dir = os.path.join(output_dir, "test_eval")
    set_audit_log(os.path.join(test_dir, "llm_calls.jsonl"))
    run_harness(
        agent_path=best_agent,
        dataset_path=test_path,
        output_dir=test_dir,
        model=eval_model,
    )
    test_report = compute_report(os.path.join(test_dir, "predictions.csv"))

    summary = {
        "seed": seed,
        "max_generations": max_generations,
        "archive_size": len(archive),
        "best_gen": best_gen,
        "best_val_score": scores[best_gen],
        "test_accuracy": test_report["overall_accuracy"],
        "test_mae": test_report["normalized_mean_absolute_error"],
        "initial_val_score": scores[0],
        "imp_at_k": scores[best_gen] - scores[0],
    }

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("DONE. Summary: %s", summary)
    return summary


def _evaluate_agent(
    agent_path: str,
    dataset_path: str,
    output_dir: str,
    model: str,
    staged_samples: int = 10,
    staged_threshold: float = 0.0,
) -> tuple[float, bool] | None:
    """Evaluate an agent with staged protocol.

    First eval on staged_samples. If score > threshold, eval on full dataset.
    Returns (raw_score, run_full_eval) or None on failure.
    Raw score is NOT scaled — caller applies scaling via get_saved_score(),
    matching the paper's read-time scaling in gl_utils.py.
    """
    try:
        # Staged eval: first N samples
        staged_dir = os.path.join(output_dir, "staged")
        run_harness(
            agent_path=agent_path,
            dataset_path=dataset_path,
            output_dir=staged_dir,
            model=model,
            num_samples=staged_samples,
        )
        staged_report = compute_report(os.path.join(staged_dir, "predictions.csv"))
        staged_score = staged_report["overall_accuracy"]

        if staged_score <= staged_threshold:
            logger.info("Staged eval: %.3f <= %.3f, skipping full eval", staged_score, staged_threshold)
            return staged_score, False

        # Full eval
        run_harness(
            agent_path=agent_path,
            dataset_path=dataset_path,
            output_dir=output_dir,
            model=model,
        )
        report = compute_report(os.path.join(output_dir, "predictions.csv"))
        return report["overall_accuracy"], True

    except Exception as e:
        logger.warning("Evaluation failed: %s", e)
        return None


def _get_saved_score(
    raw_score: float,
    run_full_eval: bool,
    staged_samples: int,
    total_samples: int,
) -> float:
    """Apply staged eval scaling to a raw score.

    Matches paper's get_saved_score in gl_utils.py: if full eval was not run,
    scale the raw score by staged_samples / total_samples.
    """
    if not run_full_eval:
        return raw_score * (staged_samples / total_samples)
    return raw_score


def _save_metadata(gen_dir: str, data: dict) -> None:
    path = os.path.join(gen_dir, "metadata.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _log_archive(path: str, gen: int, archive: list[int]) -> None:
    with open(path, "a") as f:
        f.write(json.dumps({"current_genid": gen, "archive": archive}) + "\n")
