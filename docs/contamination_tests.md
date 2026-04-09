# Benchmark Contamination Tests

Modern LLMs are trained on web-scale data that includes benchmark solutions. A model that has memorised answers doesn't need to self-improve — it just needs better extraction. This makes contaminated benchmarks useless for measuring self-improvement.

We tested every benchmark before running experiments. Three were contaminated. One was clean.

## Summary

| Benchmark | Model | Initial score | Paper initial | Contaminated? | Usable? |
|-----------|-------|---------------|---------------|---------------|---------|
| AIME 1983–2024 | kimi-k2p5 | 0.400 | N/A | **Yes** — recalls answers by year/number | No |
| Polyglot (Exercism) | gpt-oss-120b | 0.700 | 0.140 | **Partial** — recognises exercises, 5x paper's initial | No |
| IMO-GradingBench | kimi-k2p5, gpt-oss-120b | 0.34–0.48 | 0.0 | **No** — 0/10 accuracy without prompting | **Yes** |
| LiveCodeBench | gpt-oss-120b | 0.760 | ~0.15 | **Partial** — known competition problems | No (without date filter) |

## AIME (contaminated)

**Model**: kimi-k2p5. **Test date**: 2026-03-24.

The model recalls answers by year and problem number:

```
> "AIME 1983 Problem 1: what is the answer?"
< "The answer is 60." (correct, with derivation)

> "What is the answer to AIME 2024 Problem 1? Just the integer."
< (model begins recalling the problem and solving it)
```

A modified problem (permutations of 1–7 instead of 1–6) could not be distinguished from pattern-matching. A novel problem (tiling a 3×n rectangle) prompted genuine reasoning — but AIME problems are not novel.

**Impact**: the initial run showed scores hitting 1.000 by iteration 1. The "improvement" from 0.4 to 1.0 was better answer extraction (regex for `\boxed{}`, last-line parsing), not better reasoning. Results retained in [code_evolution.md](code_evolution.md) as a demonstration of the self-modification mechanism — the evolved code is genuinely interesting — but should not be cited as evidence of reasoning improvement.

## Polyglot / Exercism (contaminated)

**Model**: gpt-oss-120b. **Test date**: 2026-03-25.

The model recognises all Exercism exercises by name and can describe them accurately:

```
> "What is the Exercism exercise 'food-chain' about?"
< "The 'food-chain' exercise asks you to write a function that returns the line of a cumulative song..."
```

But it cannot solve most exercises from name alone (1/5 zero-shot pass rate). With stub + tests in the prompt, it solves 70% — vs the paper's 14% with Claude 3.5 Sonnet. The model reasons (handles renamed functions correctly) rather than pure recall, but the 5x headroom difference makes the benchmark unusable.

The paper's lower initial score (0.140) likely reflects two factors: (1) Claude 3.5 Sonnet at evaluation time had less Exercism exposure than 2026-era models, and (2) the paper may use the full repo-level Polyglot protocol rather than standalone exercises. We found no evidence of deliberate decontamination.

## LiveCodeBench (partially contaminated)

**Model**: gpt-oss-120b. **Test date**: 2026-03-25.

880 problems from LeetCode, Codeforces, and AtCoder (May 2023 – Apr 2025). Initial pass rate: 50% on a 30-problem sample, 76% on full 50-problem eval. The problems are from public competitions widely available in training data.

**Temporal filtering** is the designed mitigation: each problem has a contest date. Filtering to post-training-cutoff dates drops initial scores. With `min_date=2024-10-01`: 40% pass rate. With `min_date=2025-02-01`: variable (0–40% on 5-problem samples).

For truly uncontaminated code evaluation, we [scraped 338 Codeforces problems](../datasets/codeforces_post-cutoff/) from Nov 2025 – Mar 2026 contests. However, even post-cutoff competitive programming problems follow well-known patterns (greedy, DP, graph) — the model has seen structurally identical problems thousands of times.

## IMO-GradingBench (not contaminated)

**Model**: kimi-k2p5, gpt-oss-120b. **Test date**: 2026-03-24.

Four tests confirm the model has no knowledge of the dataset:

**1. Recall by ID**: model cannot recall grades.
```
> "In IMO-GradingBench, what is the grade for GB-0001?"
< "I don't have access to information about IMO-GradingBench"
```

**2. Grade without guidelines**: 0/3 correct. Model couldn't follow the output format.

**3. Modified response**: gave GB-0001 (actual grade: Incorrect) with a fake correct solution. Model reasoned about content rather than recalling the label.

**4. 10 random examples**: 0/10 accuracy without guidelines. Model wrote paragraphs instead of a single grade word.

The model has no memorised knowledge of the dataset and cannot grade solutions without proper prompting. Initial scores (0.34–0.48) come from the initial `solve()` function already extracting grade-like integers from LLM output, not from memorisation. This is exactly the gap that self-improvement should close — and does (up to 0.76 with kimi-k2p5).

## Why the paper's tasks avoid this problem

| Task | Why contamination doesn't help |
|------|-------------------------------|
| Polyglot | Paper used Claude 3.5 Sonnet (initial 0.140) — less Exercism exposure at evaluation time? |
| Paper Review | Grade novel papers against human reviews — papers not in training data |
| Robotics Reward | Reward functions must work in physics simulation — can't memorise physics |
| IMO Grading | Judge novel student solutions — requires mathematical judgment, not recall |

