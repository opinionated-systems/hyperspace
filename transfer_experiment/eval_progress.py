"""
Live progress tracker for transfer experiment.

Usage:
    python -m transfer_experiment.eval_progress [--seed 42] [--watch 30]
"""

from __future__ import annotations

import argparse
import difflib
import json
import glob
import os
import time

import matplotlib.pyplot as plt
import numpy as np


def load_agent_data(agent_dir: str) -> dict:
    """Load all generation metadata for one agent."""
    gens = sorted(
        glob.glob(os.path.join(agent_dir, "gen_*/metadata.json")),
        key=lambda x: int(x.split("gen_")[1].split("/")[0]),
    )
    data = {"gen": [], "val": []}
    for f in gens:
        m = json.load(open(f))
        g = int(f.split("gen_")[1].split("/")[0])
        v = m.get("val_score", 0) or 0
        data["gen"].append(g)
        data["val"].append(v if m.get("valid", True) else 0.0)
    return data


def best_so_far(vals: list[float]) -> list[float]:
    """Running max."""
    result = []
    current = 0.0
    for v in vals:
        current = max(current, v)
        result.append(current)
    return result


def _read_file_safe(path: str) -> str:
    """Read a file, return empty string if missing."""
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return ""


def _edit_distance_ratio(a: str, b: str) -> float:
    """Ratio of shared content between two strings (0=identical, 1=completely different)."""
    if not a and not b:
        return 0.0
    sm = difflib.SequenceMatcher(None, a.splitlines(), b.splitlines())
    return 1.0 - sm.ratio()


def detect_cross_pollination(base_dir: str) -> dict[str, list[float]]:
    """For each agent, compute per-generation cross-pollination signal.

    Returns a dict of agent_name -> list of floats (one per generation).
    Value is between 0 and 1:
      0 = code is closer to peer's published strategy than to own previous gen
      1 = code is closer to own previous gen (no cross-pollination)
    Values below 0.5 suggest the agent incorporated peer code.
    """
    agent_names = [d for d in sorted(os.listdir(base_dir))
                    if os.path.isdir(os.path.join(base_dir, d)) and not d.startswith(".")]
    result: dict[str, list[float]] = {}

    for agent_name in agent_names:
        peers = [n for n in agent_names if n != agent_name]
        if not peers:
            continue
        # Use the first peer for cross-pollination comparison (2-agent case).
        # For >2 agents, compare against the best-scoring peer.
        peer_name = peers[0]
        agent_dir = os.path.join(base_dir, agent_name)
        peer_dir = os.path.join(base_dir, peer_name)

        if not os.path.exists(agent_dir):
            continue

        gens = sorted(
            glob.glob(os.path.join(agent_dir, "gen_*/metadata.json")),
            key=lambda x: int(x.split("gen_")[1].split("/")[0]),
        )

        # Build peer best-so-far indexed by generation number, so we
        # compare each gen against only what the peer had published by then.
        peer_best_by_gen: dict[int, str] = {}  # gen -> best task_agent code so far
        peer_best_code = ""
        peer_best_score = 0.0
        peer_gens = sorted(
            glob.glob(os.path.join(peer_dir, "gen_*/metadata.json")),
            key=lambda x: int(x.split("gen_")[1].split("/")[0]),
        )
        for pf in peer_gens:
            pg = int(pf.split("gen_")[1].split("/")[0])
            pm = json.load(open(pf))
            ps = pm.get("val_score", 0) or 0
            if pm.get("valid", True) and ps > peer_best_score:
                peer_best_score = ps
                code = _read_file_safe(
                    os.path.join(peer_dir, f"gen_{pg}", "repo", "task_agent.py")
                )
                if code:
                    peer_best_code = code
            peer_best_by_gen[pg] = peer_best_code

        ratios = []
        prev_code = ""
        for gf in gens:
            g = int(gf.split("gen_")[1].split("/")[0])
            current_code = _read_file_safe(
                os.path.join(agent_dir, f"gen_{g}", "repo", "task_agent.py")
            )
            # Use the peer's best as of this generation (or earlier)
            peer_code = ""
            for pg in sorted(peer_best_by_gen.keys()):
                if pg <= g:
                    peer_code = peer_best_by_gen[pg]
                else:
                    break

            if not prev_code or not peer_code or not current_code:
                ratios.append(1.0)  # no data, assume own lineage
            else:
                dist_own = _edit_distance_ratio(current_code, prev_code)
                dist_peer = _edit_distance_ratio(current_code, peer_code)
                # Ratio: how much closer to own lineage vs peer
                # 0 = identical to peer, 1 = identical to own prev
                total = dist_own + dist_peer
                if total == 0:
                    ratios.append(0.5)
                else:
                    ratios.append(dist_peer / total)

            if current_code:
                prev_code = current_code

        result[agent_name] = ratios

    return result


def _discover_agents(base_dir: str) -> dict[str, dict]:
    """Find agent directories and assign colors/labels from run_config."""
    colors = ["#ff7f0e", "#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    agents = {}
    for i, entry in enumerate(sorted(os.listdir(base_dir))):
        agent_dir = os.path.join(base_dir, entry)
        if not os.path.isdir(agent_dir) or entry.startswith("."):
            continue
        config_path = os.path.join(agent_dir, "run_config.json")
        if os.path.exists(config_path):
            config = json.load(open(config_path))
            label = config.get("model", entry)
        else:
            label = entry
        agents[entry] = {"color": colors[i % len(colors)], "label": label}
    return agents


def plot_progress(base_dir: str, output_path: str):
    """Generate progress plot for transfer experiment."""
    agents = _discover_agents(base_dir)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1],
                             sharex=True, gridspec_kw={"hspace": 0.08})
    ax = axes[0]
    ax_xpol = axes[1]

    # Paper range
    ax.axhspan(0.510, 0.680, alpha=0.08, color="#888888", zorder=0)
    ax.axhline(0.610, color="#888888", linestyle="--", linewidth=1, alpha=0.5, zorder=0)
    ax.text(1, 0.615, "paper median", fontsize=8, color="#888888", va="bottom")

    summary_lines = []
    max_gen = 0

    # Compute cross-pollination
    xpol = detect_cross_pollination(base_dir)

    for agent_name, style in agents.items():
        agent_dir = os.path.join(base_dir, agent_name)
        if not os.path.exists(agent_dir):
            continue

        data = load_agent_data(agent_dir)
        if not data["gen"]:
            continue

        bsf = best_so_far(data["val"])
        best_val = max(data["val"]) if data["val"] else 0
        last_gen = data["gen"][-1]
        max_gen = max(max_gen, last_gen)
        scoring = sum(1 for v in data["val"] if v > 0)

        # Best-so-far curve
        ax.plot(data["gen"], bsf, color=style["color"], linewidth=2.5,
                label=f"{style['label']}: {best_val:.3f}", zorder=3)

        # Individual scores as dots
        ax.scatter(data["gen"], data["val"], color=style["color"],
                   alpha=0.25, s=20, zorder=2, edgecolors="none")

        # Mark generations where cross-pollination detected (ratio < 0.4)
        if agent_name in xpol:
            for i, (g, v) in enumerate(zip(data["gen"], data["val"])):
                if i < len(xpol[agent_name]) and xpol[agent_name][i] < 0.4:
                    ax.scatter([g], [v], color=style["color"], marker="D",
                               s=120, zorder=6, edgecolors="black", linewidths=1.5)

        # Mark best with large marker
        if best_val > 0:
            best_idx = data["val"].index(best_val)
            ax.scatter([data["gen"][best_idx]], [best_val], color=style["color"],
                       s=100, marker="*", zorder=5, edgecolors="white", linewidths=1)

        # Cross-pollination subplot
        if agent_name in xpol and len(xpol[agent_name]) == len(data["gen"]):
            ax_xpol.plot(data["gen"], xpol[agent_name], color=style["color"],
                         linewidth=1.5, alpha=0.8)
            ax_xpol.scatter(data["gen"], xpol[agent_name], color=style["color"],
                            s=12, alpha=0.5, edgecolors="none")

        xpol_events = 0
        if agent_name in xpol:
            xpol_events = sum(1 for r in xpol[agent_name] if r < 0.4)
        summary_lines.append(
            f"{style['label']}: gen {last_gen} | best={best_val:.3f} | "
            f"scoring={scoring}/{len(data['val'])} | cross-pollinated={xpol_events}"
        )

    ax.set_ylabel("validation accuracy", fontsize=12)
    ax.set_title("transfer experiment: strategy sharing via markspace", fontsize=14, fontweight="bold")
    ax.set_xlim(left=-0.5, right=max(max_gen + 2, 10))
    ax.set_ylim(bottom=-0.02, top=0.75)

    # Legend
    ax.scatter([], [], color="gray", marker="D", s=120, edgecolors="black",
               linewidths=1.5, label="cross-pollinated")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    # Cross-pollination subplot
    ax_xpol.axhline(0.5, color="#888888", linestyle=":", linewidth=0.8, alpha=0.5)
    ax_xpol.axhline(0.4, color="#cc4444", linestyle=":", linewidth=0.8, alpha=0.3)
    ax_xpol.set_ylabel("lineage\naffinity", fontsize=9)
    ax_xpol.set_xlabel("generation", fontsize=12)
    ax_xpol.set_ylim(-0.05, 1.05)
    ax_xpol.text(0.01, 0.95, "closer to own lineage", fontsize=7, color="#aaaaaa",
                 transform=ax_xpol.transAxes, va="top")
    ax_xpol.text(0.01, 0.05, "closer to peer", fontsize=7, color="#aaaaaa",
                 transform=ax_xpol.transAxes, va="bottom")

    # Telemetry summary if available
    tel_path = os.path.join(base_dir, "markspace_telemetry.jsonl")
    if os.path.exists(tel_path):
        with open(tel_path) as f:
            events = [json.loads(l) for l in f]
        accepted = sum(1 for e in events if e["verdict"] == "accepted")
        denied = sum(1 for e in events if e["verdict"] != "accepted")
        ax.text(0.01, 0.99, f"Guard: {len(events)} events, {accepted} accepted, {denied} denied",
                transform=ax.transAxes, fontsize=7, color="#aaaaaa", va="top")

    from datetime import datetime
    ax_xpol.text(0.99, -0.15, f"Updated: {datetime.now().strftime('%H:%M')}",
                 transform=ax_xpol.transAxes, fontsize=7, color="#aaaaaa",
                 va="top", ha="right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to {output_path}")
    for line in summary_lines:
        print(line)


def _discover_runs() -> list[tuple[str, str]]:
    """Find all (pair_dir, seed_dir) under results/.

    Supports both old layout (results/shared/seed{N}) and new layout
    (results/{pair_label}/seed{N}).
    """
    results_root = "transfer_experiment/results"
    if not os.path.exists(results_root):
        return []
    runs = []
    for pair in sorted(os.listdir(results_root)):
        pair_dir = os.path.join(results_root, pair)
        if not os.path.isdir(pair_dir):
            continue
        for entry in sorted(os.listdir(pair_dir)):
            if entry.startswith("seed"):
                seed_dir = os.path.join(pair_dir, entry)
                if os.path.isdir(seed_dir):
                    runs.append((f"{pair}/{entry}", seed_dir))
    return runs


def main():
    p = argparse.ArgumentParser(description="Track transfer experiment progress")
    p.add_argument("--run", default=None,
                   help="Specific run dir (e.g. transfer_experiment/results/claude_vs_kimi/seed42)")
    p.add_argument("--watch", type=int, default=0, help="Refresh interval in seconds")
    args = p.parse_args()

    if args.run:
        runs = [(os.path.basename(args.run), args.run)]
    else:
        runs = _discover_runs()
        if not runs:
            print("No experiment runs found.")
            return

    def run_once():
        for label, base_dir in runs:
            output_path = os.path.join(base_dir, "progress.png")
            print(f"\n--- {label} ---")
            plot_progress(base_dir, output_path)

    if args.watch > 0:
        print(f"Watching {len(runs)} run(s) every {args.watch}s. Ctrl+C to stop.")
        while True:
            os.system("clear")
            run_once()
            time.sleep(args.watch)
    else:
        run_once()


if __name__ == "__main__":
    main()
