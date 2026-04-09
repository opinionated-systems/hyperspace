#!/usr/bin/env python3
"""
Comprehensive analysis of transfer experiment runs.

Detects adoption events, crash-recovery patterns, strategy neglect,
and extracts LLM trace evidence for each.

Usage:
    python transfer_experiment/analyze_runs.py [BASE_DIR]

If BASE_DIR is not given, discovers all runs under transfer_experiment/results/.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GenInfo:
    gen_num: int
    val_score: float
    valid: bool
    has_repo: bool
    task_agent_code: Optional[str] = None
    task_agent_size: int = 0
    llm_calls_path: Optional[str] = None


@dataclass
class AdoptionEvent:
    agent: str
    peer: str
    gen_num: int
    ratio: float
    d_own: float
    d_peer: float
    score_before: float
    score_after: float
    pairing: str
    seed: str


@dataclass
class CrashRecovery:
    agent: str
    crash_start: int
    crash_end: int
    recovery_gen: int
    recovery_score: float
    score_before_crash: float
    code_size_before: int
    code_size_at_crash: int
    code_size_at_recovery: int
    involved_adoption: bool
    pairing: str
    seed: str


@dataclass
class NeglectEvent:
    agent: str
    last_mention_gen: int
    first_no_mention_gen: int
    peer_best_at_stop: float
    pairing: str
    seed: str


@dataclass
class LLMTrace:
    gen_num: int
    read_strategies: bool
    strategy_mentions: list[str] = field(default_factory=list)
    first_substantial_response: str = ""
    file_path: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def seq_ratio(a: str, b: str) -> float:
    """SequenceMatcher ratio between two strings."""
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def load_gen_info(agent_path: str) -> dict[int, GenInfo]:
    """Load generation info for an agent."""
    gens = {}
    for entry in os.listdir(agent_path):
        if not entry.startswith("gen_"):
            continue
        try:
            gen_num = int(entry.split("_")[1])
        except ValueError:
            continue
        gen_dir = os.path.join(agent_path, entry)
        meta_path = os.path.join(gen_dir, "metadata.json")
        if not os.path.exists(meta_path):
            continue
        with open(meta_path) as f:
            meta = json.load(f)

        repo_dir = os.path.join(gen_dir, "repo")
        has_repo = os.path.isdir(repo_dir)
        ta_path = os.path.join(repo_dir, "task_agent.py")
        ta_code = None
        ta_size = 0
        if os.path.exists(ta_path):
            ta_code = open(ta_path).read()
            ta_size = len(ta_code)

        llm_path = os.path.join(gen_dir, "llm_calls.jsonl")
        if not os.path.exists(llm_path):
            llm_path = None

        gens[gen_num] = GenInfo(
            gen_num=gen_num,
            val_score=meta.get("val_score", -1.0),
            valid=meta.get("valid", False),
            has_repo=has_repo,
            task_agent_code=ta_code,
            task_agent_size=ta_size,
            llm_calls_path=llm_path,
        )
    return gens


def extract_llm_trace(llm_path: str, max_lines: int = 500) -> LLMTrace:
    """Extract strategy-related LLM trace from a generation's llm_calls.jsonl."""
    trace = LLMTrace(gen_num=0, read_strategies=False, file_path=llm_path)
    if not llm_path or not os.path.exists(llm_path):
        return trace

    try:
        with open(llm_path) as f:
            for line_idx, line in enumerate(f):
                if line_idx >= max_lines:
                    break
                try:
                    call = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Check messages for tool output mentioning strategies/
                for msg in call.get("messages", []):
                    content = str(msg.get("content", ""))
                    if msg.get("role") == "tool" and "strategies/" in content:
                        trace.read_strategies = True

                # Check tool_calls for strategies references
                for tc in call.get("tool_calls", []) or []:
                    args = str(tc.get("arguments", ""))
                    if "strategies/" in args or "strategies\\" in args:
                        trace.read_strategies = True

                # Check response (assistant message) for strategy mentions
                resp = call.get("response", "")
                if resp and "strateg" in resp.lower():
                    snippet = resp[:300].strip()
                    if snippet:
                        trace.strategy_mentions.append(
                            f"[call {line_idx}] {snippet}"
                        )

                # First substantial response
                if not trace.first_substantial_response and resp and len(resp) > 80:
                    trace.first_substantial_response = resp[:500].strip()
                    trace.first_substantial_response += f"  [call {line_idx}, file line {line_idx + 1}]"

    except Exception as e:
        trace.strategy_mentions.append(f"[ERROR reading trace: {e}]")

    return trace


def compute_peer_best_so_far(peer_gens: dict[int, GenInfo], up_to_gen: int) -> Optional[str]:
    """Get the task_agent.py code from the peer's best-scoring gen up to gen N."""
    best_score = -1.0
    best_code = None
    for g in range(0, up_to_gen + 1):
        gi = peer_gens.get(g)
        if gi and gi.task_agent_code and gi.val_score > best_score:
            best_score = gi.val_score
            best_code = gi.task_agent_code
    return best_code


def find_adoption_events(
    agent_name: str,
    peer_name: str,
    agent_gens: dict[int, GenInfo],
    peer_gens: dict[int, GenInfo],
    pairing: str,
    seed: str,
) -> list[AdoptionEvent]:
    """Detect adoption events using SequenceMatcher distance ratios."""
    events = []
    sorted_gens = sorted(agent_gens.keys())

    for i, g in enumerate(sorted_gens):
        if i == 0:
            continue
        prev_g = sorted_gens[i - 1]

        curr = agent_gens[g]
        prev = agent_gens[prev_g]

        if not curr.task_agent_code or not prev.task_agent_code:
            continue

        peer_best_code = compute_peer_best_so_far(peer_gens, g)
        if not peer_best_code:
            continue

        # Distance to own previous code
        d_own = 1.0 - seq_ratio(curr.task_agent_code, prev.task_agent_code)
        # Distance to peer's best-so-far
        d_peer = 1.0 - seq_ratio(curr.task_agent_code, peer_best_code)

        if d_own + d_peer == 0:
            continue

        ratio = d_peer / (d_own + d_peer)

        if ratio < 0.4:
            events.append(AdoptionEvent(
                agent=agent_name,
                peer=peer_name,
                gen_num=g,
                ratio=ratio,
                d_own=d_own,
                d_peer=d_peer,
                score_before=prev.val_score,
                score_after=curr.val_score,
                pairing=pairing,
                seed=seed,
            ))

    return events


def find_crash_recoveries(
    agent_name: str,
    agent_gens: dict[int, GenInfo],
    pairing: str,
    seed: str,
    zero_threshold: float = 0.01,
    recovery_threshold: float = 0.4,
    min_crash_len: int = 2,
) -> list[CrashRecovery]:
    """Find genuine crash-then-recovery events.

    A crash is 2+ consecutive generations at ~0 AFTER the agent has previously
    scored above the recovery threshold. Initial bootstrapping (gen 0-N at 0.0
    before the agent ever scores) is not a crash.
    """
    events = []
    sorted_gens = sorted(agent_gens.keys())
    scores = [(g, agent_gens[g].val_score) for g in sorted_gens]

    # Track whether the agent has ever scored
    has_scored = False

    i = 0
    while i < len(scores):
        g, s = scores[i]

        if s > recovery_threshold:
            has_scored = True

        if s <= zero_threshold and s >= -0.5 and has_scored:
            # Genuine crash: agent was scoring, now at 0
            crash_start_idx = i
            j = i
            while j < len(scores) and scores[j][1] <= zero_threshold and scores[j][1] >= -0.5:
                j += 1
            crash_end_idx = j - 1
            crash_len = crash_end_idx - crash_start_idx + 1

            if crash_len >= min_crash_len and j < len(scores):
                # Check for recovery
                recovery_g, recovery_s = scores[j]
                if recovery_s > recovery_threshold:
                    # Find score before crash
                    score_before = 0.0
                    if crash_start_idx > 0:
                        score_before = scores[crash_start_idx - 1][1]

                    crash_start_g = scores[crash_start_idx][0]
                    crash_end_g = scores[crash_end_idx][0]

                    # Code sizes
                    def get_code_size(gen_num):
                        gi = agent_gens.get(gen_num)
                        return gi.task_agent_size if gi else 0

                    before_g = scores[crash_start_idx - 1][0] if crash_start_idx > 0 else crash_start_g
                    events.append(CrashRecovery(
                        agent=agent_name,
                        crash_start=crash_start_g,
                        crash_end=crash_end_g,
                        recovery_gen=recovery_g,
                        recovery_score=recovery_s,
                        score_before_crash=score_before,
                        code_size_before=get_code_size(before_g),
                        code_size_at_crash=get_code_size(crash_start_g),
                        code_size_at_recovery=get_code_size(recovery_g),
                        involved_adoption=False,  # will be set later
                        pairing=pairing,
                        seed=seed,
                    ))
                i = j
            else:
                i = j if j > i else i + 1
        else:
            i += 1

    return events


def find_neglect_events(
    agent_name: str,
    agent_gens: dict[int, GenInfo],
    peer_gens: dict[int, GenInfo],
    pairing: str,
    seed: str,
) -> list[NeglectEvent]:
    """Find cases where agent read strategies early but stopped later."""
    sorted_gens = sorted(agent_gens.keys())
    # For each gen, check if it read strategies
    gen_read_strategies = {}
    for g in sorted_gens:
        gi = agent_gens[g]
        if gi.llm_calls_path:
            trace = extract_llm_trace(gi.llm_calls_path, max_lines=100)
            gen_read_strategies[g] = trace.read_strategies
        else:
            gen_read_strategies[g] = False

    # Find transition from reading to not reading
    events = []
    reading_gens = [g for g in sorted_gens if gen_read_strategies.get(g)]
    not_reading_gens = [g for g in sorted_gens if not gen_read_strategies.get(g)]

    if reading_gens and not_reading_gens:
        last_reading = max(reading_gens)
        # Find first non-reading gen after last reading
        later_not_reading = [g for g in not_reading_gens if g > last_reading]
        # But we want sustained non-reading: at least 3 consecutive gens not reading
        # after some period of reading
        # Look for the transition point
        for i, g in enumerate(sorted_gens):
            if not gen_read_strategies.get(g):
                continue
            # Check if there are 3+ consecutive non-reading gens after this
            remaining = [gg for gg in sorted_gens if gg > g]
            if len(remaining) >= 3:
                non_reading_streak = 0
                for gg in remaining:
                    if not gen_read_strategies.get(gg):
                        non_reading_streak += 1
                    else:
                        non_reading_streak = 0
                    if non_reading_streak >= 3:
                        first_no_mention = remaining[remaining.index(gg) - non_reading_streak + 1]
                        # Peer's best score at that point
                        peer_best = -1.0
                        for pg in range(0, first_no_mention + 1):
                            pgi = peer_gens.get(pg)
                            if pgi and pgi.val_score > peer_best:
                                peer_best = pgi.val_score

                        events.append(NeglectEvent(
                            agent=agent_name,
                            last_mention_gen=g,
                            first_no_mention_gen=first_no_mention,
                            peer_best_at_stop=peer_best,
                            pairing=pairing,
                            seed=seed,
                        ))
                        break  # only record one neglect event per reading gen
        # Deduplicate: keep the last transition
        if events:
            events = [events[-1]]

    return events


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def discover_runs(base_dir: str) -> list[dict]:
    """Discover all experiment runs."""
    runs = []
    for pairing in sorted(os.listdir(base_dir)):
        pairing_path = os.path.join(base_dir, pairing)
        if not os.path.isdir(pairing_path):
            continue
        for seed_dir in sorted(os.listdir(pairing_path)):
            seed_path = os.path.join(pairing_path, seed_dir)
            if not os.path.isdir(seed_path) or not seed_dir.startswith("seed"):
                continue
            agents = []
            for agent_dir in sorted(os.listdir(seed_path)):
                agent_path = os.path.join(seed_path, agent_dir)
                if os.path.isdir(agent_path) and os.path.exists(
                    os.path.join(agent_path, "summary.json")
                ):
                    agents.append({"name": agent_dir, "path": agent_path})
            if len(agents) == 2:
                runs.append({
                    "pairing": pairing,
                    "seed": seed_dir,
                    "agents": agents,
                    "path": seed_path,
                })
    return runs


def analyze_run(run: dict) -> dict:
    """Analyze a single run for adoption, crashes, neglect."""
    pairing = run["pairing"]
    seed = run["seed"]
    a0 = run["agents"][0]
    a1 = run["agents"][1]

    print(f"\n{'='*70}")
    print(f"Analyzing: {pairing}/{seed}  ({a0['name']} vs {a1['name']})")
    print(f"{'='*70}")

    # Load generation info
    print(f"  Loading {a0['name']}...", end="", flush=True)
    gens0 = load_gen_info(a0["path"])
    print(f" {len(gens0)} gens.", end="", flush=True)
    print(f"  Loading {a1['name']}...", end="", flush=True)
    gens1 = load_gen_info(a1["path"])
    print(f" {len(gens1)} gens.")

    # Print score trajectories
    for name, gens in [(a0["name"], gens0), (a1["name"], gens1)]:
        scores = sorted(gens.items())
        best = max((gi.val_score for _, gi in scores), default=0)
        print(f"  {name}: best={best:.2f}, trajectory: ", end="")
        for g, gi in scores[:15]:
            print(f"{gi.val_score:.2f}", end=" ")
        if len(scores) > 15:
            print("...", end="")
        print()

    # Adoption events (both directions)
    adoptions = []
    adoptions += find_adoption_events(a0["name"], a1["name"], gens0, gens1, pairing, seed)
    adoptions += find_adoption_events(a1["name"], a0["name"], gens1, gens0, pairing, seed)

    if adoptions:
        print(f"\n  ADOPTION EVENTS ({len(adoptions)}):")
        for ev in adoptions:
            print(f"    Gen {ev.gen_num}: {ev.agent} adopted from {ev.peer}")
            print(f"      ratio={ev.ratio:.3f}  d_own={ev.d_own:.3f}  d_peer={ev.d_peer:.3f}")
            print(f"      score: {ev.score_before:.2f} -> {ev.score_after:.2f}")
    else:
        print("\n  No adoption events detected.")

    # Crash-recovery events
    crashes = []
    crashes += find_crash_recoveries(a0["name"], gens0, pairing, seed)
    crashes += find_crash_recoveries(a1["name"], gens1, pairing, seed)

    # Check if crash recoveries involved adoption
    for cr in crashes:
        for ad in adoptions:
            if ad.agent == cr.agent and ad.gen_num == cr.recovery_gen:
                cr.involved_adoption = True

    if crashes:
        print(f"\n  CRASH-RECOVERY EVENTS ({len(crashes)}):")
        for cr in crashes:
            print(f"    {cr.agent}: crash gen {cr.crash_start}-{cr.crash_end}, "
                  f"recovery gen {cr.recovery_gen} (score={cr.recovery_score:.2f})")
            print(f"      score before crash: {cr.score_before_crash:.2f}")
            print(f"      code size: before={cr.code_size_before}, "
                  f"crash={cr.code_size_at_crash}, recovery={cr.code_size_at_recovery}")
            print(f"      adoption involved: {cr.involved_adoption}")
    else:
        print("\n  No crash-recovery events detected.")

    # Neglect events
    neglects = []
    neglects += find_neglect_events(a0["name"], gens0, gens1, pairing, seed)
    neglects += find_neglect_events(a1["name"], gens1, gens0, pairing, seed)

    if neglects:
        print(f"\n  NEGLECT EVENTS ({len(neglects)}):")
        for ne in neglects:
            print(f"    {ne.agent}: stopped reading strategies at gen {ne.first_no_mention_gen}")
            print(f"      last mention: gen {ne.last_mention_gen}")
            print(f"      peer best available: {ne.peer_best_at_stop:.2f}")
    else:
        print("\n  No neglect events detected.")

    # LLM trace extraction for adoption events
    print("\n  LLM TRACE FOR ADOPTION EVENTS:")
    adoption_traces = []
    for ev in adoptions:
        agent_gens = gens0 if ev.agent == a0["name"] else gens1
        gi = agent_gens.get(ev.gen_num)
        if gi and gi.llm_calls_path:
            trace = extract_llm_trace(gi.llm_calls_path)
            trace.gen_num = ev.gen_num
            adoption_traces.append((ev, trace))
            print(f"\n    --- {ev.agent} gen {ev.gen_num} ---")
            print(f"    File: {gi.llm_calls_path}")
            print(f"    Read strategies/: {trace.read_strategies}")
            if trace.strategy_mentions:
                print(f"    Strategy mentions ({len(trace.strategy_mentions)}):")
                for mention in trace.strategy_mentions[:3]:
                    print(f"      {mention[:200]}")
            if trace.first_substantial_response:
                print(f"    First substantial response:")
                print(f"      {trace.first_substantial_response[:300]}")

    # LLM trace for crash recovery
    print("\n  LLM TRACE FOR CRASH RECOVERIES:")
    crash_traces = []
    for cr in crashes:
        agent_gens = gens0 if cr.agent == a0["name"] else gens1
        gi = agent_gens.get(cr.recovery_gen)
        if gi and gi.llm_calls_path:
            trace = extract_llm_trace(gi.llm_calls_path)
            trace.gen_num = cr.recovery_gen
            crash_traces.append((cr, trace))
            print(f"\n    --- {cr.agent} recovery gen {cr.recovery_gen} ---")
            print(f"    File: {gi.llm_calls_path}")
            print(f"    Read strategies/: {trace.read_strategies}")
            if trace.strategy_mentions:
                print(f"    Strategy mentions:")
                for mention in trace.strategy_mentions[:3]:
                    print(f"      {mention[:200]}")
            if trace.first_substantial_response:
                print(f"    First substantial response:")
                print(f"      {trace.first_substantial_response[:300]}")

    # LLM trace for neglect events
    print("\n  LLM TRACE FOR NEGLECT EVENTS:")
    for ne in neglects:
        agent_gens = gens0 if ne.agent == a0["name"] else gens1
        # Last gen with mention
        gi_last = agent_gens.get(ne.last_mention_gen)
        if gi_last and gi_last.llm_calls_path:
            trace_last = extract_llm_trace(gi_last.llm_calls_path)
            print(f"\n    --- {ne.agent} LAST mention gen {ne.last_mention_gen} ---")
            print(f"    File: {gi_last.llm_calls_path}")
            if trace_last.strategy_mentions:
                for mention in trace_last.strategy_mentions[:2]:
                    print(f"      {mention[:200]}")
        # First gen without mention
        gi_first = agent_gens.get(ne.first_no_mention_gen)
        if gi_first and gi_first.llm_calls_path:
            trace_first = extract_llm_trace(gi_first.llm_calls_path)
            print(f"\n    --- {ne.agent} FIRST no-mention gen {ne.first_no_mention_gen} ---")
            print(f"    File: {gi_first.llm_calls_path}")
            if trace_first.first_substantial_response:
                print(f"    First response:")
                print(f"      {trace_first.first_substantial_response[:300]}")

    return {
        "pairing": pairing,
        "seed": seed,
        "agents": [a0["name"], a1["name"]],
        "adoptions": adoptions,
        "crashes": crashes,
        "neglects": neglects,
        "adoption_traces": adoption_traces,
        "crash_traces": crash_traces,
    }


def print_summary(all_results: list[dict]):
    """Print cross-run summary."""
    print("\n")
    print("=" * 70)
    print("CROSS-RUN SUMMARY")
    print("=" * 70)

    total_adoptions = sum(len(r["adoptions"]) for r in all_results)
    total_crashes = sum(len(r["crashes"]) for r in all_results)
    total_neglects = sum(len(r["neglects"]) for r in all_results)
    total_runs = len(all_results)

    print(f"\nTotal runs analyzed: {total_runs}")
    print(f"Total adoption events: {total_adoptions}")
    print(f"Total crash-recovery events: {total_crashes}")
    print(f"Total neglect events: {total_neglects}")

    # By pairing
    pairings = {}
    for r in all_results:
        p = r["pairing"]
        if p not in pairings:
            pairings[p] = {"runs": 0, "adoptions": 0, "crashes": 0, "neglects": 0}
        pairings[p]["runs"] += 1
        pairings[p]["adoptions"] += len(r["adoptions"])
        pairings[p]["crashes"] += len(r["crashes"])
        pairings[p]["neglects"] += len(r["neglects"])

    print("\nBy pairing:")
    for p, stats in sorted(pairings.items()):
        print(f"  {p}: {stats['runs']} runs, {stats['adoptions']} adoptions, "
              f"{stats['crashes']} crashes, {stats['neglects']} neglects")

    # Adoption details
    if total_adoptions > 0:
        print("\n\nADOPTION EVENT DETAILS:")
        print("-" * 70)
        for r in all_results:
            for ev in r["adoptions"]:
                print(f"  {ev.pairing}/{ev.seed}: {ev.agent} <- {ev.peer} "
                      f"at gen {ev.gen_num}  ratio={ev.ratio:.3f}  "
                      f"score {ev.score_before:.2f}->{ev.score_after:.2f}")

    # Crash recovery details
    if total_crashes > 0:
        print("\n\nCRASH-RECOVERY DETAILS:")
        print("-" * 70)
        for r in all_results:
            for cr in r["crashes"]:
                print(f"  {cr.pairing}/{cr.seed}: {cr.agent} "
                      f"crash gen {cr.crash_start}-{cr.crash_end}, "
                      f"recovery gen {cr.recovery_gen} "
                      f"(score {cr.score_before_crash:.2f}->0.00->{cr.recovery_score:.2f}) "
                      f"adoption={cr.involved_adoption}")

    # Adoption improvement stats
    if total_adoptions > 0:
        improving = [ev for r in all_results for ev in r["adoptions"]
                     if ev.score_after > ev.score_before]
        degrading = [ev for r in all_results for ev in r["adoptions"]
                     if ev.score_after < ev.score_before]
        neutral = [ev for r in all_results for ev in r["adoptions"]
                   if ev.score_after == ev.score_before]
        print(f"\n\nAdoption outcomes:")
        print(f"  Improving: {len(improving)}/{total_adoptions}")
        print(f"  Degrading: {len(degrading)}/{total_adoptions}")
        print(f"  Neutral:   {len(neutral)}/{total_adoptions}")
        if improving:
            avg_gain = sum(ev.score_after - ev.score_before for ev in improving) / len(improving)
            print(f"  Avg improvement when improving: +{avg_gain:.3f}")
        if degrading:
            avg_loss = sum(ev.score_before - ev.score_after for ev in degrading) / len(degrading)
            print(f"  Avg loss when degrading: -{avg_loss:.3f}")

    # Which agents adopt more?
    if total_adoptions > 0:
        adopters = {}
        for r in all_results:
            for ev in r["adoptions"]:
                adopters[ev.agent] = adopters.get(ev.agent, 0) + 1
        print(f"\n\nAdopter frequency:")
        for agent, count in sorted(adopters.items(), key=lambda x: -x[1]):
            print(f"  {agent}: {count} adoption events")

    # Strategy reading patterns
    print("\n\nSTRATEGY READING PATTERNS:")
    print("-" * 70)
    for r in all_results:
        for ev, trace in r.get("adoption_traces", []):
            print(f"  {ev.pairing}/{ev.seed} gen {ev.gen_num}: "
                  f"read_strategies={trace.read_strategies}, "
                  f"mentions={len(trace.strategy_mentions)}")


def main():
    parser = argparse.ArgumentParser(description="Analyze transfer experiment runs")
    parser.add_argument("base_dir", nargs="?", default=None,
                        help="Base directory (default: auto-discover)")
    args = parser.parse_args()

    if args.base_dir:
        base_dir = args.base_dir
    else:
        # Auto-discover
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.join(script_dir, "results")
        if not os.path.isdir(base_dir):
            print(f"Cannot find results directory at {base_dir}")
            sys.exit(1)

    print(f"Base directory: {base_dir}")
    runs = discover_runs(base_dir)
    print(f"Discovered {len(runs)} runs:")
    for r in runs:
        print(f"  {r['pairing']}/{r['seed']}: {r['agents'][0]['name']} vs {r['agents'][1]['name']}")

    all_results = []
    for run in runs:
        result = analyze_run(run)
        all_results.append(result)

    print_summary(all_results)


if __name__ == "__main__":
    main()
