"""
Evaluation harness for IMO grading.

Reimplemented from facebookresearch/HyperAgents domains/harness.py.
Loads the task agent from a file path, runs it on the dataset, saves predictions.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

QUESTION_ID = "Grading ID"
GROUND_TRUTH_KEY = "Reward"


def format_input_dict(row: dict) -> dict:
    """Format a dataset row into task agent input.

    Matches paper's domains/imo/grading_utils.py exactly.
    """
    return {
        "domain": "imo_grading",
        "problem": row["Problem"],
        "solution": row["Solution"],
        "grading_guidelines": row["Grading guidelines"],
        "student_answer": row["Response"],
    }


def _propagate_shared_state():
    """Propagate shared state from main llm_client to repo's copy."""
    import sys
    repo_llm = sys.modules.get("agent.llm_client")
    main_llm = sys.modules.get("replication.agent.llm_client")
    if repo_llm is not None and main_llm is not None and repo_llm is not main_llm:
        repo_llm._audit_log_path = main_llm._audit_log_path
        repo_llm._audit_lock = main_llm._audit_lock
        repo_llm._clients = main_llm._clients


def load_task_agent(agent_path: str):
    """Load TaskAgent class from a Python file.

    Adds the agent's parent directory to sys.path so that `from agent.xxx`
    imports resolve to the repo's agent/ subfolder (matching paper's Docker
    setup where the repo IS the Python path).
    """
    import sys
    abs_path = os.path.abspath(agent_path)
    repo_dir = os.path.dirname(abs_path)
    added = False
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
        added = True
    try:
        # Clear cached agent modules so modified versions are picked up
        stale = [k for k in sys.modules if k == "agent" or k.startswith("agent.")]
        for k in stale:
            del sys.modules[k]
        spec = importlib.util.spec_from_file_location("agent_module", abs_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load from {abs_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _propagate_shared_state()
        if not hasattr(mod, "TaskAgent"):
            raise AttributeError(f"No TaskAgent in {abs_path}")
        return mod.TaskAgent
    finally:
        if added:
            sys.path.remove(repo_dir)


def _run_agent(TaskAgentClass, model: str, row: dict, evals_dir: str) -> str:
    """Run task agent on a single problem."""
    question_id = row[QUESTION_ID]
    agent = TaskAgentClass(model=model)
    inputs = format_input_dict(row)
    prediction, _ = agent.forward(inputs)
    return prediction


def run_harness(
    agent_path: str,
    dataset_path: str,
    output_dir: str,
    model: str = "gpt-oss-120b",
    num_samples: int = -1,
    num_workers: int = 5,
    save_interval: int = 20,
) -> str:
    """Run evaluation harness on a dataset.

    Args:
        agent_path: path to task_agent.py
        dataset_path: path to CSV (train, val, or test split)
        output_dir: where to save predictions.csv
        model: LLM model for task agent
        num_samples: limit number of samples (-1 for all)
        num_workers: parallel workers
        save_interval: save intermediate predictions every N completions

    Returns:
        Path to output directory
    """
    TaskAgentClass = load_task_agent(agent_path)

    os.makedirs(output_dir, exist_ok=True)
    evals_dir = os.path.join(output_dir, "agent_evals")
    os.makedirs(evals_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "predictions.csv")

    # Load dataset
    df = pd.read_csv(dataset_path, dtype=str)
    if num_samples > 0:
        df = df[:num_samples]

    # Load existing predictions for resume
    completed = set()
    if os.path.exists(output_path):
        existing = pd.read_csv(output_path, dtype=str)
        completed = set(existing[~existing["prediction"].isna()][QUESTION_ID])
        # Merge existing predictions into df so they aren't lost
        existing_map = dict(zip(existing[QUESTION_ID], existing["prediction"]))
        df["prediction"] = df[QUESTION_ID].map(existing_map)
    else:
        df["prediction"] = None

    predictions = df["prediction"].tolist()
    futures = []

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        for i, row in df.iterrows():
            if row[QUESTION_ID] in completed:
                continue
            futures.append((
                i,
                pool.submit(_run_agent, TaskAgentClass, model, row.to_dict(), evals_dir),
            ))

        completed_count = 0
        for idx, future in futures:
            predictions[idx] = future.result()
            completed_count += 1
            if completed_count % save_interval == 0:
                df["prediction"] = predictions
                df.to_csv(output_path, index=False)
                logger.info("Checkpoint: %d/%d saved to %s", completed_count, len(futures), output_path)

    df["prediction"] = predictions
    df.to_csv(output_path, index=False)
    logger.info("Predictions saved to %s", output_path)

    return output_dir
