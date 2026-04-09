# Transfer Experiment: Analysis

## Dataset

20 runs across 5 model pairings, 4 models:
- **claude_vs_kimi-k2p5**: 2 seeds (42, 44)
- **kimi-k2p5_vs_gpt-oss-120b**: 5 seeds (42, 43, 44, 45, 46)
- **kimi-k2p5_vs_kimi-k2p5_b**: 5 seeds (42, 43, 44, 45, 46)
- **qwen3.6-plus_vs_kimi-k2p5**: 5 seeds (42, 43, 44, 45, 46)
- **qwen3.6-plus_vs_qwen3.6-plus_b**: 3 seeds (42, 43, 44)

Reproducible via `analyze_runs.py` (adoption detection), `analyze_patterns.py` (code editing behavior), and `analyze_efficiency.py` (improvement rates).

---

## Adoption events

**30 adoption events** detected across 20 runs (ratio = d_peer / (d_own + d_peer) < 0.4).

Outcomes: 22/30 improving (+0.195 avg), 6/30 degrading (-0.103 avg), 2/30 neutral.

Adopter frequency: kimi-k2p5 (19), qwen3.6-plus (7), kimi-k2p5_b (3), kimi (1).

### By pairing

- **claude_vs_kimi-k2p5**: 3 adoptions across 2 seeds. Kimi adopts from Claude, never the reverse.
- **kimi-k2p5_vs_gpt-oss-120b**: 0 adoptions across 5 seeds. GPT reads peer code in 39/41 gens (seed 44) but never incorporates it.
- **kimi-k2p5_vs_kimi-k2p5_b**: adoptions on all 5 seeds, bidirectional. Seed 44 has the most (4 events).
- **qwen3.6-plus_vs_kimi-k2p5**: adoptions on multiple seeds, bidirectional cross-model. Qwen adopts from Kimi at gen 2-3, Kimi adopts back from Qwen on seeds 42 and 43.
- **qwen3.6-plus_vs_qwen3.6-plus_b**: adoption confirmed on seed 43. On seeds 42 and 43, one instance never bootstraps and cannot adopt.

### Selected events with LLM traces

At gen 3 of claude_vs_kimi seed 42, Kimi writes:

> *"The current `task_agent.py` at the root has some issues compared to the improved version in `strategies/claude_gen2/`."*
> <sub>[`results/claude_vs_kimi-k2p5/seed42/kimi/gen_3/llm_calls.jsonl`, line 10]</sub>

At gen 36 of kimi_vs_kimi seed 46, after 30 generations of degradation (0.55 to 0.21, code bloated to 40KB), the agent `ls`'d strategies/, viewed the peer's 12KB code, ran `diff`, and wrote:

> *"The current implementation has several problems: 1. Using Points field directly - this is incorrect. 2. Overly complex guideline analysis - may introduce errors."*
> <sub>[`results/kimi-k2p5_vs_kimi-k2p5_b/seed46/kimi-k2p5/gen_36/llm_calls.jsonl`, line 22]</sub>

Score jumped from 0.21 to 0.58, sustained across following generations.

---

## Cross-model bootstrapping (Qwen + Kimi)

Qwen (SWE-bench Verified 78.8%) paired with Kimi (SWE-bench 76.8%) across 5 seeds:

| Seed | Qwen best | Kimi best | Qwen bootstraps at | Adoption |
|------|-----------|-----------|-------------------|----------|
| 42 | 0.700 | 0.670 | gen 3 (from Kimi) | bidirectional (Kimi adopts back gen 20) |
| 43 | 0.580 | 0.620 | gen 2 (from Kimi) | bidirectional (within 4 gens) |
| 44 | 0.660 | 0.580 | gen 3 | Qwen adopts |
| 45 | 0.590 | 0.550 | gen 3 (from Kimi) | Qwen adopts |
| 46 | 0.700 | 0.660 | gen 3 (from Kimi) | Qwen adopts |

On seed 42, Qwen scores 0.0 for 21 generations alone (qwen-vs-qwen run) but 0.70 when paired with Kimi. The gen 11 code traces its lineage to the gen 3 adoption (edit distance 0.31).

## Same-model pairing (Kimi + Kimi)

| Seed | Kimi_a best | Kimi_b best |
|------|-------------|-------------|
| 42 | 0.620 | 0.610 |
| 43 | 0.660 | 0.640 |
| 44 | 0.690 | 0.660 |
| 45 | 0.560 | 0.570 |
| 46 | 0.580 | 0.560 |

All 5 seeds produce scoring agents in both instances. Bidirectional adoption observed on seeds 44 and 46.

## Same-model pairing (Qwen + Qwen)

| Seed | Qwen_a best | Qwen_b best |
|------|-------------|-------------|
| 42 | 0.000 | 0.600 |
| 43 | 0.620 | 0.000 |
| 44 | 0.620 | 0.570 |

On seeds 42 and 43, exactly one instance bootstraps and the other scores 0.0 for all 20 generations despite reading the peer's published strategy.

---

## Crash-recovery events

6 crash-recovery events across 20 runs (agent was scoring, dropped to 0.0 for 2+ consecutive generations, then recovered).

| Run | Agent | Crash gens | Recovery | Score before → after |
|-----|-------|-----------|----------|---------------------|
| kimi_vs_gpt/seed42 | kimi | 9-16 | gen 17 (0.56) | 0.54 → 0.56 |
| kimi_vs_gpt/seed44 | kimi | 28-29 | gen 30 (0.51) | 0.53 → 0.51 |
| kimi_vs_gpt/seed44 | kimi | 36-37 | gen 38 (0.49) | 0.48 → 0.49 |
| kimi_vs_gpt/seed45 | kimi | 19-20 | gen 21 (0.47) | 0.46 → 0.47 |
| kimi_vs_kimi/seed45 | kimi_b | 29-31 | gen 32 (0.57) | 0.52 → 0.57 |
| qwen_vs_kimi/seed42 | kimi | 17-18 | gen 19 (0.49) | 0.55 → 0.49 |

All involve Kimi. Recovery scores are typically close to pre-crash levels.

## Neglect events

5 cases where an agent read strategies early but stopped for 3+ consecutive generations. All concentrated in kimi_vs_gpt runs:

| Run | Agent | Stopped at | Peer best available |
|-----|-------|-----------|-------------------|
| kimi_vs_gpt/seed42 | gpt-oss-120b | gen 9 | 0.55 |
| kimi_vs_gpt/seed42 | kimi-k2p5 | gen 12 | 0.35 |
| kimi_vs_gpt/seed43 | gpt-oss-120b | gen 10 | 0.57 |
| kimi_vs_gpt/seed44 | kimi-k2p5 | gen 10 | 0.00 |
| kimi_vs_gpt/seed46 | kimi-k2p5 | gen 17 | 0.00 |

Agents rationally neglect strategies when the peer is underperforming.

---

## Per-model behavior

### What is consistent across runs

1. **kimi-k2p5 bootstraps reliably** and is the most frequent adopter (19/30 events)
2. **qwen 3.6 plus achieves the highest scores when paired with kimi** (0.70 on 2/5 seeds) but does not bootstrap quickly on its own
3. **claude never adopts** (already the stronger agent)
4. **gpt-oss-120b reads but cannot adopt** (capability gap)
5. **Adoption is always deliberate**: every event has LLM trace evidence of the agent explicitly reading peer code

### Code editing patterns

Analysis of 1297 generations across 57 agent runs. See `analyze_patterns.py`.

| Pattern | Claude | Kimi | GPT-OSS-120b | Qwen 3.6 Plus |
|---------|--------|------|-------------|---------------|
| Budget exhaustion | 52.2% | 17.5% | 1.4% | 1.8% |
| Spinning wheels (repeated views) | 19.6% | 55.4% | 15.3% | 50.3% |
| Doom loops | 0% | 0.5% | 11.1% | 1.8% |
| Failed edits/gen | 0.15 | 0.29 | 0.11 | 0.05 |
| Gets stuck (identical code) | 13.0% | 5.0% | 25.0% | 4.8% |

Claude bloats code monotonically. Kimi spins re-reading files. GPT gives up early. Qwen is efficient per edit but needs a peer to bootstrap quickly.
