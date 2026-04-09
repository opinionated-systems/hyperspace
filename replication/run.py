"""
Entry point for DGM-H replication on IMO-GradingBench.

Usage:
    python -m replication.run --iterations 200 --seed 42
    python -m replication.run --iterations 200 --seed 42 --freeze-meta  # w/o self-improve
    python -m replication.run --iterations 200 --seed 42 --selection random
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from replication.generate_loop import generate_loop
from replication.agent.llm_client import META_MODEL, EVAL_MODEL, cleanup_clients

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = str(Path(__file__).parent / "data")
INITIAL_AGENT = str(Path(__file__).parent / "task_agent.py")


def main():
    p = argparse.ArgumentParser(description="DGM-H replication on IMO-GradingBench")
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--meta-model", type=str, default=META_MODEL)
    p.add_argument("--meta-temperature", type=float, default=0.0,
                   help="Temperature for meta model")
    p.add_argument("--eval-model", type=str, default=EVAL_MODEL)
    p.add_argument("--selection", type=str, default="score_child_prop",
                   choices=["score_child_prop", "random", "best"])
    p.add_argument("--freeze-meta", action="store_true",
                   help="DGM-H w/o self-improve: skip meta agent modification")
    p.add_argument("--annealing", action="store_true",
                   help="Annealing mode: with decaying probability, fork from initial agent instead of parent")
    p.add_argument("--annealing-p0", type=float, default=0.5,
                   help="Initial probability of blind exploration (annealing mode)")
    p.add_argument("--annealing-decay", type=float, default=0.9,
                   help="Decay factor for annealing probability after each blind round")
    p.add_argument("--initial-meta", type=str, default=None,
                   help="Path to custom initial meta_agent.py (for transfer experiments)")
    p.add_argument("--experiment", type=str, default=None,
                   help="Experiment name (groups runs under replication/results/<experiment>/)")
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    # Verify data exists
    train = Path(DATA_DIR) / "train.csv"
    if not train.exists():
        logger.error("Data not prepared. Run: python -m replication.data.prepare")
        return

    # Output directory
    if args.output:
        output_dir = args.output
    else:
        if args.annealing:
            arm = "arm_d_annealing"
        elif args.freeze_meta:
            arm = "arm_b_freeze"
        elif args.selection == "random":
            arm = "arm_c_random"
        else:
            arm = "arm_a_full"
        experiment = args.experiment or "default"
        output_dir = f"replication/results/{experiment}/{arm}/seed{args.seed}"

    logger.info("DGM-H replication: %d iterations, seed=%d, meta=%s, eval=%s, selection=%s, freeze=%s, annealing=%s",
                args.iterations, args.seed, args.meta_model, args.eval_model, args.selection, args.freeze_meta, args.annealing)

    try:
        summary = generate_loop(
            output_dir=output_dir,
            data_dir=DATA_DIR,
            initial_agent_path=INITIAL_AGENT,
            initial_meta_path=args.initial_meta,
            max_generations=args.iterations,
            meta_model=args.meta_model,
            meta_temperature=args.meta_temperature,
            eval_model=args.eval_model,
            selection_method=args.selection,
            seed=args.seed,
            freeze_meta=args.freeze_meta,
            annealing=args.annealing,
            annealing_p0=args.annealing_p0,
            annealing_decay=args.annealing_decay,
        )
        logger.info("Final summary: %s", summary)
    finally:
        cleanup_clients()


if __name__ == "__main__":
    main()
