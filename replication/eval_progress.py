"""
Live progress tracker for replication experiments.

Usage:
    python -m replication.eval_progress [--experiment replication_v1] [--watch 30]

Produces replication/results/<experiment>/progress.png with:
- Per-arm best-val-so-far curves
- Paper's result range shaded
"""

from __future__ import annotations

import argparse
import json
import glob
import os
import time

import matplotlib.pyplot as plt
import numpy as np


def load_arm_data(arm_dir: str) -> dict:
    """Load all generation metadata for one arm/seed."""
    gens = sorted(
        glob.glob(os.path.join(arm_dir, "gen_*/metadata.json")),
        key=lambda x: int(x.split("gen_")[1].split("/")[0]),
    )
    data = {"gen": [], "train": [], "val": [], "valid": [], "parent": []}
    for f in gens:
        m = json.load(open(f))
        g = int(f.split("gen_")[1].split("/")[0])
        data["gen"].append(g)
        data["train"].append(m.get("train_score", 0) or 0)
        data["val"].append(m.get("val_score", 0) or 0)
        data["valid"].append(bool(m.get("valid")))
        data["parent"].append(m.get("parent"))
    return data


def compute_best_so_far(vals: list[float]) -> list[float]:
    """Running max of val scores."""
    best = []
    current = 0.0
    for v in vals:
        current = max(current, v)
        best.append(current)
    return best


def plot_progress(experiment_dir: str, output_path: str):
    """Generate progress plot for all arms."""
    arm_dirs = sorted(glob.glob(os.path.join(experiment_dir, "arm_*/seed*")))
    if not arm_dirs:
        print("No data found.")
        return

    arm_labels = {
        "arm_a_full": "A: Full DGM-H",
        "arm_b_freeze": "B: Frozen meta",
        "arm_c_random": "C: Random",
        "arm_d_annealing": "D: Annealing",
    }
    arm_colors = {
        "arm_a_full": "#d62728",
        "arm_b_freeze": "#1f77b4",
        "arm_c_random": "#2ca02c",
        "arm_d_annealing": "#9467bd",
    }
    arm_markers = {
        "arm_a_full": "o",
        "arm_b_freeze": "s",
        "arm_c_random": "^",
        "arm_d_annealing": "D",
    }

    # Include other experiments if they exist (e.g., claude_test)
    extra_experiments = {
        "claude_test": {"label": "A: Full DGM-H (Sonnet)", "color": "#ff7f0e", "marker": "*"},
    }
    for exp_name, style in extra_experiments.items():
        exp_path = os.path.join(os.path.dirname(experiment_dir), exp_name)
        for seed_dir in sorted(glob.glob(os.path.join(exp_path, "arm_*/seed*"))):
            arm_dirs.append(seed_dir)
            parts = seed_dir.replace(exp_path + "/", "").split("/")
            arm_key = f"{exp_name}/{parts[0]}"
            arm_labels[arm_key] = style["label"]
            arm_colors[arm_key] = style["color"]
            arm_markers[arm_key] = style["marker"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax1 = plt.subplots(1, 1, figsize=(14, 7))

    # Paper's result range
    ax1.axhspan(0.510, 0.680, alpha=0.08, color="#888888", zorder=0)
    ax1.axhline(0.610, color="#888888", linestyle="--", linewidth=1, alpha=0.6, zorder=0)
    ax1.text(1, 0.615, "Paper median (0.610)", fontsize=8, color="#888888", va="bottom")
    ax1.text(1, 0.685, "Paper 95% CI", fontsize=8, color="#aaaaaa", va="bottom")

    summary_lines = []
    max_gen = 0

    # Group arm_dirs by arm_key
    from collections import defaultdict
    arms_grouped: dict[str, list[str]] = defaultdict(list)
    for arm_dir in arm_dirs:
        # Check if this is from the main experiment or an extra
        if experiment_dir in arm_dir:
            parts = arm_dir.replace(experiment_dir + "/", "").split("/")
            arm_key = parts[0]
        else:
            # Extra experiment — find matching key from arm_labels
            for key in arm_labels:
                if "/" in key and key.split("/")[0] in arm_dir:
                    arm_key = key
                    break
            else:
                arm_key = os.path.basename(os.path.dirname(os.path.dirname(arm_dir)))
        arms_grouped[arm_key].append(arm_dir)

    for arm_key, seed_dirs in sorted(arms_grouped.items()):
        label = arm_labels.get(arm_key, arm_key)
        color = arm_colors.get(arm_key, "black")
        marker = arm_markers.get(arm_key, "o")

        # Collect best-so-far curves for all seeds
        all_best_curves = []
        all_best_vals = []
        seed_summaries = []

        for arm_dir in sorted(seed_dirs):
            seed = arm_dir.split("/")[-1]
            data = load_arm_data(arm_dir)
            if not data["gen"]:
                continue

            best_so_far = compute_best_so_far(data["val"])
            valid_count = sum(data["valid"])
            failed_count = len(data["valid"]) - valid_count
            scoring = sum(1 for v in data["val"] if v > 0)
            last_gen = data["gen"][-1]
            best_val = max(data["val"]) if data["val"] else 0
            max_gen = max(max_gen, last_gen)

            all_best_curves.append((data["gen"], best_so_far))
            all_best_vals.append(best_val)

            # Plot individual seed as thin line
            alpha = 0.3 if len(seed_dirs) > 1 else 1.0
            width = 1.0 if len(seed_dirs) > 1 else 2.5
            ax1.plot(data["gen"], best_so_far, color=color, linewidth=width,
                     alpha=alpha, zorder=2)

            # Individual gen scores as dots
            ax1.scatter(data["gen"], data["val"], color=color, alpha=0.15, s=12,
                        marker=marker, zorder=1, edgecolors="none")

            seed_summaries.append(
                f"  {seed}: gen {last_gen}/200 | best={best_val:.3f} | "
                f"valid={valid_count} fail={failed_count} | scoring={scoring}/{valid_count}"
            )

        if not all_best_vals:
            continue

        # Aggregate: if multiple seeds, plot median with min/max shaded band
        if len(seed_dirs) > 1:
            # Interpolate all curves to common x-axis
            common_max = min(curve[0][-1] for curve in all_best_curves)
            common_gens = list(range(0, common_max + 1))
            interpolated = []
            for gens, bsf in all_best_curves:
                interp = []
                gi = 0
                for g in common_gens:
                    while gi < len(gens) - 1 and gens[gi + 1] <= g:
                        gi += 1
                    interp.append(bsf[gi] if gi < len(bsf) else bsf[-1])
                interpolated.append(interp)
            arr = np.array(interpolated)
            median_curve = np.median(arr, axis=0)
            min_curve = np.min(arr, axis=0)
            max_curve = np.max(arr, axis=0)
            q25 = np.percentile(arr, 25, axis=0)
            q75 = np.percentile(arr, 75, axis=0)

            cg = np.array(common_gens)
            # Light band: min-max
            ax1.fill_between(cg, min_curve, max_curve, color=color, alpha=0.08, zorder=1)
            # Darker band: IQR (25-75th percentile)
            ax1.fill_between(cg, q25, q75, color=color, alpha=0.2, zorder=2)
            # Median line
            ax1.plot(cg, median_curve, color=color, linewidth=2.5,
                     label=f"{label}: median={np.median(all_best_vals):.3f} (n={len(seed_dirs)})",
                     zorder=3)
        else:
            # Single seed — already plotted with full alpha, just add label
            ax1.plot([], [], color=color, linewidth=2.5,
                     label=f"{label}: {all_best_vals[0]:.3f}")

        # Mark best per seed
        for arm_dir in sorted(seed_dirs):
            data = load_arm_data(arm_dir)
            if not data["val"]:
                continue
            best_val = max(data["val"])
            if best_val > 0:
                best_idx = data["val"].index(best_val)
                ax1.scatter([data["gen"][best_idx]], [best_val], color=color,
                            s=80, marker=marker, zorder=4, edgecolors="white", linewidths=1.5)

        median_best = np.median(all_best_vals)
        summary_lines.append(f"{label}: median={median_best:.3f} (n={len(seed_dirs)}, seeds={[d.split('/')[-1] for d in sorted(seed_dirs)]})")
        for s in seed_summaries:
            summary_lines.append(s)

    ax1.set_xlabel("Generation", fontsize=12)
    ax1.set_ylabel("Val Accuracy", fontsize=12)
    ax1.set_title("DGM-H Replication on IMO-GradingBench", fontsize=14, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax1.set_xlim(left=-0.5, right=max(max_gen + 2, 20))
    ax1.set_ylim(bottom=-0.02, top=0.75)
    ax1.tick_params(labelsize=10)

    # Timestamp
    from datetime import datetime
    ax1.text(0.01, 0.01, f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
             transform=ax1.transAxes, fontsize=7, color="#aaaaaa", va="bottom")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to {output_path}")
    print()
    for line in summary_lines:
        print(line)


def main():
    p = argparse.ArgumentParser(description="Track replication experiment progress")
    p.add_argument("--experiment", type=str, default="replication_v1")
    p.add_argument("--watch", type=int, default=0,
                   help="Refresh interval in seconds (0 = run once)")
    args = p.parse_args()

    experiment_dir = os.path.join("replication/results", args.experiment)
    output_path = os.path.join(experiment_dir, "progress.png")

    if args.watch > 0:
        print(f"Watching every {args.watch}s. Ctrl+C to stop.")
        while True:
            os.system("clear")
            plot_progress(experiment_dir, output_path)
            time.sleep(args.watch)
    else:
        plot_progress(experiment_dir, output_path)


if __name__ == "__main__":
    main()
