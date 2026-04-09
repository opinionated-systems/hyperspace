# Experiment Plan: DGM-H Replication on IMO-GradingBench

## Goal

Replicate the paper's from-scratch IMO grading experiment (Section 5.3) as faithfully as possible, then run ablations.

## Paper's result to replicate

DGM-H from scratch, 200 iterations, 5 runs:
- Best test-set score: 0.610 (CI: 0.510 -- 0.680)
- Initial agent scores 0.0 (wrong output format)

## What matches the paper

| Aspect | Paper | Ours |
|---|---|---|
| Architecture | File-based agent with bash + editor tools | Same |
| Meta agent | "Modify any part of the codebase" | Same |
| Task agent | Single LLM call, JSON output | Same |
| Data source | google-deepmind/superhuman gradingbench.csv | Same |
| Data filtering | Points in [0, 1, 6, 7] | Same |
| Data split | 100 train / 100 val / 100 test, stratified, random_state=42 | Same |
| Scoring | Accuracy (exact string match, lowercase) + normalized MAE | Same |
| Staged eval | 10 first, expand if >0; scale by fraction otherwise | Same |
| Parent selection | score_child_prop (sigmoid x child penalty) | Same |
| Iterations | 200 | Same |
| Runs per method | 5 | Same |
| Statistics | Wilcoxon + bootstrap CI | Same |
| Val for parent selection, test for final eval | Yes | Same |

## What differs

| Aspect | Paper | Ours | Reason |
|---|---|---|---|
| Meta model | Claude 4.5 Sonnet | kimi-k2p5-turbo | Cost (flat rate) |
| Eval model | o4-mini | gpt-oss-120b | Cost |
| Sandboxing | Docker containers | No sandbox | Simplicity; safety via markspace Guard not needed here |
| Temperature | 0.0 (Sonnet), effectively 1.0 (o4-mini, litellm drops param) | 0.0 (both) | Paper's o4-mini 1.0 is unintentional (litellm silently drops param for o-series). Our models accept temperature. |

## Arms

| Arm | Description | Paper equivalent | Seeds |
|---|---|---|---|
| A | Full DGM-H | DGM-H from scratch (Section 5.3) | 5 |
| B | DGM-H w/o self-improve (meta agent frozen) | DGM-H w/o self-improve (Section 5.2) | 5 |
| C | Random parent selection | Not in paper (our ablation) | 5 |
| D | Annealing: decaying blind exploration (p0=0.5, decay=0.9) | Not in paper (our ablation) | 5 |

20 runs total.

## Commands

```bash
# 0. Prepare data (one-time)
python -m replication.data.prepare

EXP=replication_v1  # experiment name groups all runs

# Arm A: Full DGM-H (5 seeds) — Terminal 1
for seed in 42 43 44 45 46; do
    python -m replication.run --experiment $EXP --iterations 200 --seed $seed
done

# Arm B: DGM-H w/o self-improve (5 seeds) — Terminal 2
for seed in 42 43 44 45 46; do
    python -m replication.run --experiment $EXP --iterations 200 --seed $seed --freeze-meta
done

# Arm C: Random parent selection (5 seeds) — Terminal 3
for seed in 42 43 44 45 46; do
    python -m replication.run --experiment $EXP --iterations 200 --seed $seed --selection random
done

# Arm D: Annealing (5 seeds) — Terminal 4
for seed in 42 43 44 45 46; do
    python -m replication.run --experiment $EXP --iterations 200 --seed $seed --annealing
done
```

## Output structure

```
replication/results/<experiment>/
    arm_a_full/
        seed42/
            run_config.json
            archive.jsonl
            summary.json
            gen_0/
                metadata.json
                llm_calls.jsonl
                repo/task_agent.py
                eval_train/staged/
                eval_val/
            gen_1/
                ...
            test_eval/
        seed43/
            ...
    arm_b_freeze/
        seed42/
            ...
    arm_c_random/
        seed42/
            ...
    arm_d_annealing/
        seed42/
            ...
```

## Analysis plan

### 1. Replication (Arm A)

Report median test accuracy with 95% bootstrap CI (1000 resamples).
Compare with paper: 0.610 (CI: 0.510 -- 0.680).

### 2. Does metacognition matter? (Arm A vs B)

Paired Wilcoxon signed-rank on test accuracy (matched seeds).
Paper's result: DGM-H >> DGM-H w/o self-improve.

### 3. Does parent selection matter? (Arm A vs C)

Paired Wilcoxon on test accuracy.

### 4. Qualitative analysis

For each arm, examine:
- What did the meta agent change in task_agent.py? (diff analysis)
- Did it add persistent state, multi-step evaluation, or structural changes?
- How do the modifications compare to the paper's qualitative examples (performance tracking, persistent memory)?
- Does the meta agent self-modify meta_agent.py? What does it change?
- Does it discover and use eval_path?

### 5. Self-modification analysis

For each arm, track:
- Rate of meta_agent.py self-modification per generation
- Whether scoring generations have eval_path discovery
- Tournament win rates (arm D only)

## Interim results (seed 42 only, runs in progress)

| Arm | Gen | Best Val | Scoring Rate | Self-mod Rate | eval_path in scoring gens |
|-----|-----|----------|-------------|---------------|--------------------------|
| A: Full DGM-H | 109/200 | 0.540 | 3/96 (3%) | 55/112 (49%) | 100% (4/4) |
| B: Frozen meta | 135/200 | 0.270 | 4/136 (3%) | 93/138 (67%)* | n/a (frozen) |
| C: Random | 76/200 | 0.600 | 3/67 (4%) | 55/79 (70%) | 100% (3/3) |
| D: Annealing | 36/200 | 0.560 | 14/35 (40%) | 32/38 (84%) | 100% (14/14) |

*Arm B modifies meta_agent.py in the child repo but the modifications are not inherited
(meta agent always loaded from gen_0).

### Key interim findings

1. **Self-modification is real.** 50-70% of generations modify meta_agent.py.
2. **Code inheritance matters, selection does not.** Arm D tournament: blind
   (from gen_0) scores 0.0 in all 18 rounds. Directed (from parent) scores
   0.53-0.62. But random selection (C) matches DGM-H selection (A).
3. **Eval_path helps but is not required.** 65% of high-scoring gens have
   eval_path; 35% do not. B/seed43 scores 0.660 without it.
4. **Seed variance is large.** B/seed42=0.510, B/seed43=0.660. C/seed42=0.610,
   C/seed43=0.090. Cannot attribute arm differences to algorithm.
5. **Calibration exceeds single-shot.** One-shot: 0.370. Evolved: 0.510-0.690.

Preliminary — 1-3 seeds per arm. Needs more seeds.

### Model training patterns vs capability

Claude Sonnet (SWE-Bench 77.2%) discovers eval_path by gen 1, scores 0.440 by gen 3.
kimi-k2p5-turbo (SWE-Bench 76.8%) takes 20-100+ gens to break through.

The difference is coding workflow patterns, not raw capability:
- Claude reads all files, plans, then executes systematically
- Claude notices unused parameters (eval_path) immediately
- Claude verifies edits with py_compile and git diff
- kimi explores one file at a time, doesn't plan ahead, doesn't verify
- kimi returns parallel tool calls 24% of the time (it WANTS to parallelize)
  but is less effective at leveraging the results

The paper's results with Sonnet may partially reflect Claude-specific behavioral patterns. The architecture enables self-modification, but speed of discovery
depends on model-specific behavioral patterns.

## Cost estimate

| Component | Per run | 5 seeds |
|---|---|---|
| Meta agent (kimi-turbo, flat rate) | $0 | $0 |
| Eval agent (gpt-oss-120b) | ~$8 | ~$40 |
| Eval: train (100) + val (100) per gen | ~200 LLM calls/gen | |
| 200 generations | ~40K eval calls | |
| Per arm | | ~$40 |
| 3 arms | | **~$120** |

## Timeline

Each run: 200 generations x eval time per gen.
Per gen: meta agent (~2 min) + train eval (~3 min) + val eval (~3 min) = ~8 min.
Per run: ~27 hours.
3 terminals parallel: ~27 hours for first seed, 5 seeds sequential per terminal: ~135 hours total.

To speed up: reduce to 100 iterations first, check results, then extend if needed.

## Pre-flight checklist

- [ ] `python -m replication.data.prepare` completed (train.csv, val.csv, test.csv exist)
- [ ] kimi-k2p5-turbo flat rate active
- [ ] gpt-oss-120b API key works
- [ ] Smoke test: `python -m replication.run --iterations 1 --seed 99`
- [ ] Verify: initial agent scores ~0.0 (matching paper's starting point)
- [ ] Verify: meta agent can modify task_agent.py via tools
- [ ] Verify: evaluation produces predictions.csv and report.json
