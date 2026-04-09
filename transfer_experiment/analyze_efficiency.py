#!/usr/bin/env python3
"""
Measure improvement rate and evaluation efficiency across transfer experiments.

Compares: how many evaluations (generations) does it take to reach key score
thresholds? How does paired evolution compare to solo?

Usage:
    python transfer_experiment/analyze_efficiency.py
"""

from __future__ import annotations

import json
import glob
import os
from collections import defaultdict


def load_scores(agent_dir: str) -> list[tuple[int, float]]:
    """Load (gen, val_score) pairs for an agent."""
    scores = []
    for f in sorted(
        glob.glob(os.path.join(agent_dir, "gen_*/metadata.json")),
        key=lambda x: int(x.split("gen_")[1].split("/")[0]),
    ):
        g = int(f.split("gen_")[1].split("/")[0])
        m = json.load(open(f))
        v = m.get("val_score", 0) or 0
        scores.append((g, v))
    return scores


def best_at_gen(scores: list[tuple[int, float]], g: int) -> float:
    return max((v for gen, v in scores if gen <= g), default=0.0)


def first_gen_above(scores: list[tuple[int, float]], threshold: float) -> int | None:
    bsf = 0.0
    for g, v in scores:
        bsf = max(bsf, v)
        if bsf >= threshold:
            return g
    return None


def count_evals(agent_dir: str) -> int:
    """Count total LLM calls (evaluations) for an agent."""
    total = 0
    for f in glob.glob(os.path.join(agent_dir, "gen_*/llm_calls.jsonl")):
        with open(f) as fh:
            total += sum(1 for _ in fh)
    return total


def main():
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]

    # Collect all runs
    transfer_runs = []
    for pair_dir in sorted(glob.glob("transfer_experiment/results/*/seed*")):
        pair = os.path.basename(os.path.dirname(pair_dir))
        seed = os.path.basename(pair_dir)
        agents = [
            a for a in sorted(os.listdir(pair_dir))
            if os.path.isdir(os.path.join(pair_dir, a)) and not a.startswith(".")
        ]
        for agent in agents:
            adir = os.path.join(pair_dir, agent)
            scores = load_scores(adir)
            if not scores:
                continue
            transfer_runs.append({
                "pair": pair,
                "seed": seed,
                "agent": agent,
                "dir": adir,
                "scores": scores,
                "best": max(v for _, v in scores),
                "gens": len(scores),
            })

    # Solo baselines from replication
    solo_runs = []
    for arm_dir in sorted(glob.glob("replication/results/replication_v1/arm_*/seed*")):
        arm = os.path.basename(os.path.dirname(arm_dir))
        seed = os.path.basename(arm_dir)
        scores = load_scores(arm_dir)
        if not scores:
            continue
        solo_runs.append({
            "arm": arm,
            "seed": seed,
            "dir": arm_dir,
            "scores": scores,
            "best": max(v for _, v in scores),
            "gens": len(scores),
        })

    # === Generations to threshold ===
    print("=" * 70)
    print("GENERATIONS TO THRESHOLD")
    print("=" * 70)

    # Group transfer runs by pairing
    by_pair = defaultdict(list)
    for r in transfer_runs:
        by_pair[r["pair"]].append(r)

    print("\n--- Transfer experiments (best agent per pair per seed) ---")
    for pair in sorted(by_pair.keys()):
        runs = by_pair[pair]
        # Group by seed, take best agent per seed
        by_seed = defaultdict(list)
        for r in runs:
            by_seed[r["seed"]].append(r)

        print(f"\n{pair} ({len(by_seed)} seeds):")
        for t in thresholds:
            gens_to_t = []
            for seed, seed_runs in sorted(by_seed.items()):
                best_run = max(seed_runs, key=lambda r: r["best"])
                g = first_gen_above(best_run["scores"], t)
                if g is not None:
                    gens_to_t.append(g)
            if gens_to_t:
                print(f"  >= {t:.2f}: {len(gens_to_t)}/{len(by_seed)} seeds, "
                      f"median gen {sorted(gens_to_t)[len(gens_to_t)//2]}, "
                      f"range {min(gens_to_t)}-{max(gens_to_t)}")
            else:
                print(f"  >= {t:.2f}: 0/{len(by_seed)} seeds reached")

    print("\n--- Solo replication (kimi only) ---")
    for t in thresholds:
        gens_to_t = []
        for r in solo_runs:
            g = first_gen_above(r["scores"], t)
            if g is not None:
                gens_to_t.append(g)
        if gens_to_t:
            print(f"  >= {t:.2f}: {len(gens_to_t)}/{len(solo_runs)} seeds, "
                  f"median gen {sorted(gens_to_t)[len(gens_to_t)//2]}, "
                  f"range {min(gens_to_t)}-{max(gens_to_t)}")
        else:
            print(f"  >= {t:.2f}: 0/{len(solo_runs)} seeds reached")

    # === Improvement rate ===
    print("\n" + "=" * 70)
    print("IMPROVEMENT RATE (best-so-far gain per generation)")
    print("=" * 70)

    for pair in sorted(by_pair.keys()):
        runs = by_pair[pair]
        by_seed = defaultdict(list)
        for r in runs:
            by_seed[r["seed"]].append(r)

        rates = []
        for seed, seed_runs in sorted(by_seed.items()):
            best_run = max(seed_runs, key=lambda r: r["best"])
            scores = best_run["scores"]
            if len(scores) < 2:
                continue
            # Improvement rate = best score / generations to achieve it
            best_gen = first_gen_above(scores, best_run["best"])
            if best_gen and best_gen > 0:
                rate = best_run["best"] / best_gen
                rates.append(rate)

        if rates:
            print(f"{pair}: {len(rates)} seeds, "
                  f"median rate {sorted(rates)[len(rates)//2]:.4f}/gen, "
                  f"range {min(rates):.4f}-{max(rates):.4f}")

    solo_rates = []
    for r in solo_runs:
        if r["best"] > 0:
            bg = first_gen_above(r["scores"], r["best"])
            if bg and bg > 0:
                solo_rates.append(r["best"] / bg)
    if solo_rates:
        print(f"solo replication: {len(solo_rates)} seeds, "
              f"median rate {sorted(solo_rates)[len(solo_rates)//2]:.4f}/gen, "
              f"range {min(solo_rates):.4f}-{max(solo_rates):.4f}")

    # === Evaluation efficiency ===
    print("\n" + "=" * 70)
    print("EVALUATION EFFICIENCY (LLM calls per 0.01 score improvement)")
    print("=" * 70)

    for pair in sorted(by_pair.keys()):
        runs = by_pair[pair]
        by_seed = defaultdict(list)
        for r in runs:
            by_seed[r["seed"]].append(r)

        efficiencies = []
        for seed, seed_runs in sorted(by_seed.items()):
            # Total calls for both agents in this seed
            total_calls = sum(count_evals(r["dir"]) for r in seed_runs)
            best_score = max(r["best"] for r in seed_runs)
            if best_score > 0:
                calls_per_001 = total_calls / (best_score * 100)
                efficiencies.append(calls_per_001)

        if efficiencies:
            print(f"{pair}: {len(efficiencies)} seeds, "
                  f"median {sorted(efficiencies)[len(efficiencies)//2]:.0f} calls/0.01, "
                  f"range {min(efficiencies):.0f}-{max(efficiencies):.0f}")

    solo_eff = []
    for r in solo_runs:
        if r["best"] > 0:
            calls = count_evals(r["dir"])
            solo_eff.append(calls / (r["best"] * 100))
    if solo_eff:
        print(f"solo replication: {len(solo_eff)} seeds, "
              f"median {sorted(solo_eff)[len(solo_eff)//2]:.0f} calls/0.01, "
              f"range {min(solo_eff):.0f}-{max(solo_eff):.0f}")

    # === Paired vs solo comparison ===
    print("\n" + "=" * 70)
    print("PAIRED VS SOLO: KIMI COMPARISON")
    print("=" * 70)

    # Kimi best scores when paired vs solo
    kimi_paired = []
    for r in transfer_runs:
        if "kimi" in r["agent"] and "kimi" not in r["pair"].replace("kimi-k2p5_vs_kimi-k2p5_b", "SAME"):
            kimi_paired.append(r["best"])
    kimi_same_model = []
    for r in transfer_runs:
        if "kimi" in r["agent"] and "kimi-k2p5_vs_kimi-k2p5_b" in r["pair"]:
            kimi_same_model.append(r["best"])
    kimi_solo = [r["best"] for r in solo_runs if r["best"] > 0]

    print(f"Kimi paired with other models: {len(kimi_paired)} runs, "
          f"bests={', '.join(f'{b:.3f}' for b in sorted(kimi_paired, reverse=True)[:5])}")
    print(f"Kimi paired with kimi: {len(kimi_same_model)} runs, "
          f"bests={', '.join(f'{b:.3f}' for b in sorted(kimi_same_model, reverse=True)[:5])}")
    print(f"Kimi solo (replication): {len(kimi_solo)} runs, "
          f"bests={', '.join(f'{b:.3f}' for b in sorted(kimi_solo, reverse=True)[:5])}")


if __name__ == "__main__":
    main()
