"""
DGM-H generate loop using mark-backed tools.

The LLM sees normal bash and editor tools. The harness translates all
file operations to markspace mark operations. The Guard enforces scope,
identity, and immutability on every operation.

This replaces generate_loop.py's filesystem operations with mark operations.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import shutil
import sys
import tempfile

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

from replication.agent.llm_client import set_audit_log, cleanup_clients
from replication.generate_loop import _propagate_shared_state
from replication.evaluation.harness import run_harness, load_task_agent
from replication.evaluation.report import compute_report
from transfer_experiment.mark_tools import (
    MarkBackedCodebase,
    make_mark_editor,
    make_mark_bash,
)

logger = logging.getLogger(__name__)


def _init_codebase_from_dir(codebase: MarkBackedCodebase, directory: str):
    """Load all Python files from a directory into a mark-backed codebase."""
    from replication.generate_loop import _rewrite_imports
    _TEXT_EXTENSIONS = {".py", ".txt", ".md", ".json", ".csv", ".cfg", ".toml", ".yaml", ".yml"}
    for root, _, files in os.walk(directory):
        for fname in files:
            if fname.endswith(".pyc") or "__pycache__" in root:
                continue
            if os.path.splitext(fname)[1] not in _TEXT_EXTENSIONS:
                continue
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, directory)
            try:
                content = open(fpath).read()
            except UnicodeDecodeError:
                continue
            # Rewrite imports for repo-local resolution
            content = content.replace("from replication.agent.", "from agent.")
            content = content.replace("import replication.agent.", "import agent.")
            codebase.write_file(rel, content)


def _materialize_and_load(codebase: MarkBackedCodebase, module_name: str, filename: str):
    """Materialize codebase to temp dir, load a module from it."""
    tmpdir = tempfile.mkdtemp()
    codebase.materialize(tmpdir)

    filepath = os.path.join(tmpdir, filename)
    if not os.path.exists(filepath):
        return None, tmpdir

    # Add to sys.path and clear cached agent modules
    if tmpdir not in sys.path:
        sys.path.insert(0, tmpdir)
    stale = [k for k in sys.modules if k == "agent" or k.startswith("agent.")]
    for k in stale:
        del sys.modules[k]

    spec = importlib.util.spec_from_file_location(module_name, filepath)
    if spec is None or spec.loader is None:
        return None, tmpdir
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _propagate_shared_state()
    return mod, tmpdir


def _evaluate_from_marks(
    codebase: MarkBackedCodebase,
    dataset_path: str,
    output_dir: str,
    eval_model: str,
    staged_samples: int = 10,
    staged_threshold: float = 0.0,
) -> tuple[float, bool] | None:
    """Evaluate a task agent from mark-backed codebase."""
    tmpdir = tempfile.mkdtemp()
    try:
        codebase.materialize(tmpdir)
        agent_path = os.path.join(tmpdir, "task_agent.py")
        if not os.path.exists(agent_path):
            return None

        # Staged eval
        staged_dir = os.path.join(output_dir, "staged")
        run_harness(
            agent_path=agent_path,
            dataset_path=dataset_path,
            output_dir=staged_dir,
            model=eval_model,
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
            model=eval_model,
        )
        report = compute_report(os.path.join(output_dir, "predictions.csv"))
        return report["overall_accuracy"], True
    except Exception as e:
        logger.warning("Evaluation failed: %s", e)
        return None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def mark_generate_loop(
    codebase: MarkBackedCodebase,
    output_dir: str,
    data_dir: str,
    max_generations: int,
    meta_model: str,
    eval_model: str,
    seed: int,
) -> dict:
    """Run DGM-H loop with mark-backed codebase.

    All file operations go through marks. The LLM sees normal tools.
    The Guard enforces scope on every operation.
    """
    random.seed(seed)
    np.random.seed(seed)

    output_dir = os.path.abspath(output_dir)
    data_dir = os.path.abspath(data_dir)
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    test_path = os.path.join(data_dir, "test.csv")

    import pandas as pd
    val_total = len(pd.read_csv(val_path, dtype=str))

    archive = [0]
    scores: dict[int, float] = {}
    best_val = 0.0

    # Evaluate initial agent (gen 0)
    logger.info("Evaluating initial agent (gen 0)")
    gen0_dir = os.path.join(output_dir, "gen_0")
    os.makedirs(gen0_dir, exist_ok=True)
    set_audit_log(os.path.join(gen0_dir, "llm_calls.jsonl"))

    train_result = _evaluate_from_marks(
        codebase, train_path, os.path.join(gen0_dir, "eval_train"), eval_model)
    val_result = _evaluate_from_marks(
        codebase, val_path, os.path.join(gen0_dir, "eval_val"), eval_model)

    train_raw = train_result[0] if train_result else 0.0
    if val_result:
        val_raw, val_full = val_result
        frac = 10 / val_total
        scores[0] = val_raw * frac if not val_full else val_raw
    else:
        val_raw = 0.0
        scores[0] = 0.0

    logger.info("Gen 0: train=%.3f val=%.3f", train_raw, val_raw)

    with open(os.path.join(gen0_dir, "metadata.json"), "w") as f:
        json.dump({"parent": None, "valid": True, "train_score": train_raw,
                    "val_score": val_raw, "is_initial": True}, f, indent=2)

    # Main loop
    for gen in range(1, max_generations + 1):
        logger.info("Generation %d / %d", gen, max_generations)
        gen_dir = os.path.join(output_dir, f"gen_{gen}")
        os.makedirs(gen_dir, exist_ok=True)
        set_audit_log(os.path.join(gen_dir, "llm_calls.jsonl"))

        # Load and run meta agent from marks.
        # Only expose task_agent.py and meta_agent.py to the meta agent.
        # Infrastructure (agent/) is loaded for execution but not shown.
        try:
            tmpdir = tempfile.mkdtemp()
            codebase.materialize(tmpdir)

            repo_dir = os.path.join(tmpdir, "_repo")
            os.makedirs(repo_dir, exist_ok=True)
            for fname in ["task_agent.py", "meta_agent.py"]:
                src = os.path.join(tmpdir, fname)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(repo_dir, fname))

            if tmpdir not in sys.path:
                sys.path.insert(0, tmpdir)
            stale = [k for k in sys.modules if k == "agent" or k.startswith("agent.")]
            for k in stale:
                del sys.modules[k]

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
            sys.modules["agent.tools.bash_tool"].set_allowed_root(repo_dir)
            sys.modules["agent.tools.editor_tool"].set_allowed_root(repo_dir)

            meta.forward(repo_path=repo_dir, eval_path=output_dir,
                         iterations_left=max_generations - gen)

            # Read modified files back into marks
            for fname in ["task_agent.py", "meta_agent.py"]:
                fpath = os.path.join(repo_dir, fname)
                if os.path.exists(fpath):
                    new_content = open(fpath).read()
                    old_content = codebase.read_file(fname)
                    if new_content != old_content:
                        codebase.write_file(fname, new_content)

        except Exception as e:
            logger.warning("Gen %d: meta agent failed: %s", gen, e)
            with open(os.path.join(gen_dir, "metadata.json"), "w") as f:
                json.dump({"parent": 0, "valid": False, "error": str(e)}, f, indent=2)
            archive.append(gen)
            continue
        finally:
            repo_bash = sys.modules.get("agent.tools.bash_tool")
            if repo_bash and hasattr(repo_bash, "reset_session"):
                repo_bash.reset_session()
            if tmpdir and os.path.exists(tmpdir):
                shutil.rmtree(tmpdir, ignore_errors=True)
            # Clean sys.path
            if tmpdir in sys.path:
                sys.path.remove(tmpdir)

        # Evaluate
        train_result = _evaluate_from_marks(
            codebase, train_path, os.path.join(gen_dir, "eval_train"), eval_model)
        if train_result is None:
            with open(os.path.join(gen_dir, "metadata.json"), "w") as f:
                json.dump({"parent": 0, "valid": False, "error": "eval failed"}, f, indent=2)
            archive.append(gen)
            continue

        train_raw, _ = train_result
        val_result = _evaluate_from_marks(
            codebase, val_path, os.path.join(gen_dir, "eval_val"), eval_model)

        if val_result:
            val_raw, val_full = val_result
            frac = 10 / val_total
            val_saved = val_raw * frac if not val_full else val_raw
        else:
            val_raw = 0.0
            val_saved = 0.0

        archive.append(gen)
        scores[gen] = val_saved

        logger.info("Gen %d: train=%.3f val=%.3f", gen, train_raw, val_raw)

        with open(os.path.join(gen_dir, "metadata.json"), "w") as f:
            json.dump({"parent": 0, "valid": True, "train_score": train_raw,
                        "val_score": val_raw}, f, indent=2)

        if val_raw > best_val:
            best_val = val_raw

    # Final eval on test
    best_gen = max(archive, key=lambda g: scores.get(g, 0))
    logger.info("Best gen_%d (val=%.3f). Test eval...", best_gen, scores.get(best_gen, 0))
    test_dir = os.path.join(output_dir, "test_eval")
    set_audit_log(os.path.join(test_dir, "llm_calls.jsonl"))

    test_result = _evaluate_from_marks(
        codebase, test_path, test_dir, eval_model)
    test_acc = test_result[0] if test_result else 0.0

    summary = {
        "seed": seed,
        "max_generations": max_generations,
        "archive_size": len(archive),
        "best_gen": best_gen,
        "best_val_score": scores.get(best_gen, 0),
        "test_accuracy": test_acc,
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary
