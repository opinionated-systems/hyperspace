<a href="https://doi.org/10.5281/zenodo.19488272"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.19488272.svg" alt="DOI"></a>

# Cross-Agent+Model Evolution of Self-Modifying Code

Automatically discovered agent designs are starting to outperform hand-designed ones. [ADAS (Hu et al., ICLR 2025)](https://arxiv.org/abs/2408.08435) demonstrated this across reading comprehension, math, and science benchmarks. [Hyperagents (Zhang et al., 2026)](https://arxiv.org/abs/2603.19461) took the next step: agents that rewrite not just their task code but the code that decides *how to improve*. Advantageous changes to the improvement process itself accelerate all subsequent gains.

<p align="center">
  <img src="hyperspace.jpg" width="450" />
  <br><sub>Ready for warp-speed improvements. Image elements &copy; Paramount Pictures and Disney, of course.</sub>
</p>

Proprietary large language models are accessed via API with no weight access, and even open-weight models are expensive to fine-tune compared to editing the code around them. So in practice, self-modification means evolving the harness: prompts, parsing, extraction logic. The model weights stay fixed, which puts a natural ceiling on overall performance. In our experiments on IMO grading, evolved agents reach 0.510-0.670 accuracy in solo runs and up to 0.700 in paired transfer experiments, where a one-shot prompt achieves 0.370. A human can iterate manually, but can't keep the code calibrated as the data distribution shifts, the provider updates the model behind the API, and the task requirements change.

ADAS's own future work identifies two open problems: the meta agent improving itself, and agents improving continuously post-deployment rather than in offline search. This repository addresses both. We replicate Hyperagents' self-modification, add cross-model code transfer via [markspace](https://github.com/opinionated-systems/markspace), and find that **the only reliable controls on self-modifying agents are the ones the agent cannot edit**.

An agent rewriting its own code can rewrite any constraint you put in that code. When we switched the output format from markdown code fences to XML tags, the agent reverted the change in one generation. [markspace](https://github.com/opinionated-systems/markspace) enforces scope (what the agent can affect, accidentally or adversarially), and by that also controls which evaluation signals the agent can see. The latter matters because a self-modifying agent will optimize for whatever metric is available to it. If the metric is aligned with the goal, this produces genuine improvement. If it isn't, the agent learns to game the measure while the task results get worse ([Goodhart's law](https://en.wikipedia.org/wiki/Goodhart%27s_law)).

> [!WARNING]
> markspace controls what the agent can edit, but not what evolved code can
> do when it executes. An evolved `task_agent.py` runs inside the harness
> Python process and can call arbitrary system functions. Our experiments run
> inside a devcontainer, which limits blast radius but does not provide hard
> isolation (the container runs with `--network=host` and a Docker socket
> mount). Production deployment would require proper sandboxing per generation.

With safety enforced at the infrastructure level, we let multiple self-modifying agents share evolved code through the environment. In prior fixed-search systems ([DGM](https://arxiv.org/abs/2505.22954), [AlphaEvolve](https://arxiv.org/abs/2506.13131), [FunSearch](https://doi.org/10.1038/s41586-023-06924-6)), the framework decides what code to transfer. In our system, the agent discovers peer code through its own exploration and decides whether to use it, without knowing the environment is mediated or constrained.

[CORAL (Qu et al., 2026)](https://arxiv.org/abs/2604.01658) independently arrives at the same architecture: autonomous agents coordinating through shared persistent memory. We extend this in three directions: **cross-model transfer** (agents on different LLMs adopt each other's code), **infrastructure-enforced edit scope** via markspace (a guard layer at the environment boundary, not per-agent workspace directories), and **meta-agent self-modification** (the improvement process itself evolves). CORAL adds structured knowledge types (notes, skills) and heartbeat interventions for stuck agents. [markspace](https://github.com/opinionated-systems/markspace) supports five mark types (Intent, Action, Observation, Warning, Need) but these experiments use only Observations for code. The [benchmarks](https://github.com/opinionated-systems/benchmarks) repository applies DGM-H with cross-model strategy sharing to a different domain: VLIW SIMD kernel optimization (Anthropic's performance take-home, where [CORAL](https://arxiv.org/abs/2604.01658) holds the best known result at 1,103 cycles).

## Transfer experiment results

We run pairs of agents on the same task, each evolving independently through an environment mediated by [markspace](https://github.com/opinionated-systems/markspace). All code lives as observation marks in the shared space, with the evaluation score as confidence. The infrastructure materializes these marks into a filesystem for each agent, because that is how LLMs know to interact with code. Each agent sees its own code as editable files and the peer's best published code in a `strategies/` directory.

Across 20 runs (5 model pairings, 2-5 seeds each), agents voluntarily adopt peer strategies without being told to. The analysis script (`transfer_experiment/analyze_runs.py`) detects 30 adoption events with LLM trace evidence in every pairing that has a capable peer.

At gen 3 of a [Claude Sonnet 4.6](https://platform.claude.com/docs/en/docs/about-claude/models)-vs-[Kimi K2.5 Turbo](https://fireworks.ai/models/fireworks/kimi-k2p5) run, Kimi's meta agent writes:

> *"The current `task_agent.py` at the root has some issues compared to the improved version in `strategies/claude_gen2/`. Let me make improvements to enhance the grading accuracy and robustness."*<br/>
> <sub>[`results/claude_vs_kimi-k2p5/seed42/kimi/gen_3/llm_calls.jsonl`, line 10]</sub>

After 34 generations of degradation in a Kimi-vs-Kimi run (0.55 down to 0.21, code bloated to 40KB), the agent `ls`'d `strategies/`, viewed the peer's 12KB code, ran `diff` against its own, and wrote:

> *"The current implementation has several problems: 1. Using Points field directly - this is incorrect as it bypasses the actual grading logic. 2. Overly complex guideline analysis - may introduce errors."*<br/>
> <sub>[`results/kimi-k2p5_vs_kimi-k2p5_b/seed46/kimi-k2p5/gen_36/llm_calls.jsonl`, line 22]</sub>

It replaced its bloated code with a simplified version derived from the peer. Score jumped from 0.21 to 0.58.

**Cross-model bootstrapping.** [Qwen 3.6 Plus](https://fireworks.ai/models/fireworks/qwen3p6-plus) (SWE-bench Verified 78.8%) paired with Kimi (SWE-bench 76.8%) reaches 0.70 on two of five seeds and 0.58-0.66 on the others. On seed 42, Qwen scores 0.0 for 21 generations alone but 0.70 when paired with Kimi. The lower-benchmarking model provides bootstrapping that the higher-benchmarking model had not yet achieved on its own.

Cross-model adoption is bidirectional. On seed 42, Kimi bootstrapped Qwen, then at gen 20 Kimi adopted back from Qwen's improved code and hit 0.67. On seed 43, Qwen and Kimi adopted from each other within 4 generations.

In benchmark experiments we see that expensive frontier models tend to find breakthroughs faster when facing complex problems, but cheaper models can extend and occasionally surpass those results at a fraction of the cost.

**Results transfer faster than methodology.** When a peer finds a better solution in optimization benchmarks, the code gets adopted within a few generations. The analysis scripts, debug tooling, and reasoning that produced it are available but not adopted. Agents reimplement the insight inline instead of pulling in the tool.

**Adoption without understanding produces confabulation.** In benchmark experiments, an agent that copies code it didn't reason through generates a plausible-sounding but wrong explanation for the improvement. Those fake explanations then propagate into shared notes and mislead peers.

**Same-model parallel exploration.** Two kimi-k2p5-turbo instances paired across 5 seeds reach best scores of 0.56-0.69, with the top pair (seed 44) peaking at 0.69 and 0.66 within 40 generations. Best solo Kimi in the replication: 0.670 after 200 generations. Mixed-model pairings stay structurally diverse for longer than same-model pairings, which tend to converge to near-identical code.

**Each model has distinct behavioral patterns.** In these experiments, Claude bloats code monotonically. Kimi spins re-reading files (55%). GPT gives up early and gets stuck (25%). Qwen has the lowest failed edit rate (0.05/gen) but only reaches top scores when paired with a capable peer.

**Strategy neglect.** Full neglect (agent stops checking `strategies/` entirely) was observed once (Kimi with Claude, seed 42). Informed independence (agent reads peer code every generation but doesn't adopt) is more common and sometimes resolves into late adoption (Kimi with Qwen, seed 42: 15 generations of independence, then adoption at gen 20 after a crash). The [RAGEN "Echo Trap" (2025)](https://arxiv.org/abs/2504.20073) documents similar attention narrowing in RL-trained agents.

**Evolved complacency.** In [benchmark](https://github.com/opinionated-systems/benchmarks) experiments on VLIW kernel optimization, two of three frontier models meta-evolved themselves to *prevent further exploration* after reaching a good-but-not-optimal score. Claude Opus 4.6 redefined its own write_file tool to return "BLOCKED if at world record" and propagated the stop signal to peers. Open-source models wrote similar warnings but ignored them and kept improving. More capable self-modification produces more effective self-imposed exploration limits. See [benchmarks](https://github.com/opinionated-systems/benchmarks) for the full analysis.

See [transfer_experiment/findings.md](transfer_experiment/findings.md) for the full analysis.


## Replication findings

Our [replication](replication/) of DGM-H on IMO-GradingBench confirms the paper's results (0.610, CI: 0.510-0.680).

**Self-modification is gated by interface familiarity.** Same model, same task: file-based tools yield regular self-modification; exec'd-string architecture yields 0% across 14 runs. The capability exists in the weights but only activates through an interface familiar from training data.

**Parent selection does not matter, self-modification does.** Random parent selection (C: median 0.570 across 5 seeds) scored comparably to DGM-H's algorithm (A: median 0.580 across 5 seeds). The evolutionary loop provides the structure (select, fork, modify, evaluate, repeat). What drives improvement is the meta agent reading and modifying its own source code. In our experiments, it discovers an unused `eval_path` parameter and wires it into its own instruction. About 51% of high-scoring generations in self-modifying arms lack this self-discovered eval feedback, so it is not the only path. Harness-level changes likely have a ceiling set by the frozen model weights. Whether techniques like [reinforcement learning from verifiable rewards](https://github.com/huggingface/trl) that modify the weights can push past it is untested in this setup.

**Seed variance dominates.** Arm C across 5 seeds: 0.610, 0.090, 0.600, 0.000, 0.570. Same algorithm, same hyperparameters, different random seed. All four arms have 2-5 seeds at 200 generations. Arm medians (A: 0.580, B: 0.580, C: 0.570, D: 0.615) are not statistically separable.


See [replication/README.md](replication/README.md) for per-arm results and the experiment plan.

## Toward production

ADAS identifies online continual learning as an open problem: "continuously improving agents based on [this] extensive feedback is challenging for human developers." With safety enforcement outside the agent, the evolutionary process becomes safe to run continuously. Our experiments surface what else is needed:

- **Execution sandboxing.** markspace enforces edit scope at the infrastructure boundary, but evolved code still runs in the harness process. Production needs per-generation sandboxing (VM or unprivileged container) so an accidental or adversarial code change cannot affect anything outside the current generation's workspace. Our experiments run inside a devcontainer, which limits blast radius but is not a real sandbox.
- **Quantifying transfer capability.** Can a model adopt another agent's evolved code and improve on it? Four models tested (April 2026) range from consistent adoption to complete inability. No existing coding benchmark we are aware of measures this. See [transfer_experiment/findings.md](transfer_experiment/findings.md).
- **Stability.** Agents can regress catastrophically between generations (0.56 to 0.0). Production requires validation gates with fallback to best-so-far.
- **Generalization.** Agents overfit to small eval sets (train/val gaps up to 0.20). Rotating or held-out eval sets prevent memorization.
- **Code bloat.** The dominant failure mode. Agents accumulate complexity until the code breaks. Multi-agent setups partially self-correct (agents adopt cleaner peer code), but explicit size or complexity guards are needed.
- **Evolved complacency.** In [benchmark](https://github.com/opinionated-systems/benchmarks) experiments, frontier models meta-evolved write guards that block their own exploration after reaching a good-but-not-optimal score. Open-source models wrote similar warnings but ignored them. Assume any self-modifying agent will produce these unless the harness prevents it.
- **Content sanitization.** Agents that adopt code they didn't reason through confabulate explanations that propagate into shared notes and mislead peers. Sanitization at the storage boundary matters for fabricated narratives as much as for complacency patterns.
- **Cost and evaluation efficiency.** Paired runs improve at median 0.02-0.13 score/gen vs 0.004 solo (5-36x faster). Paired runs use median 63-378 LLM calls per 0.01 score improvement vs 680 solo (2-11x more efficient). See `transfer_experiment/analyze_efficiency.py`. Expensive models bootstrap quickly (Claude: 8 generations to 0.660). Cheaper models can continue from there via markspace traces.
- **Persistent state.** markspace state lives in memory for our experiments. Durable storage is discussed in the [markspace framework design](https://github.com/opinionated-systems/markspace/blob/main/docs/framework.md).


## Documentation

- [transfer_experiment/findings.md](transfer_experiment/findings.md) — adoption events, code patterns, LLM traces
- [transfer_experiment/README.md](transfer_experiment/README.md) — protocol, safety, per-run analysis
- [replication/README.md](replication/README.md) — architecture, deviations, open questions, limitations
- [replication/experiment_plan.md](replication/experiment_plan.md) — 4-arm experiment plan

## Usage

```bash
poetry install
cp .env.example .env  # add API keys

# Transfer experiment
poetry run python -m transfer_experiment.run_shared --models claude-sonnet-4-6 accounts/fireworks/routers/kimi-k2p5-turbo --iterations 20 --seed 42
poetry run python -m transfer_experiment.eval_progress

# Replication
poetry run python -m replication.data.prepare
poetry run python -m replication.run --experiment replication_v1 --iterations 200 --seed 42
```

## References

- Hu, S. et al. (2025). [Automated Design of Agentic Systems](https://arxiv.org/abs/2408.08435). ICLR 2025.
- Zhang, J. et al. (2026). [Hyperagents](https://arxiv.org/abs/2603.19461). arXiv:2603.19461.
- Rosser, J. & Foerster, J. (2025). [AgentBreeder: Mitigating the AI Safety Risks of Multi-Agent Scaffolds via Self-Improvement](https://arxiv.org/abs/2502.00757). arXiv:2502.00757.
- Romera-Paredes, B. et al. (2024). [Mathematical discoveries from program search with large language models](https://doi.org/10.1038/s41586-023-06924-6). Nature 625, pages 468-475.
- Dochkina, V. (2026). [Drop the Hierarchy and Roles: How Self-Organizing LLM Agents Outperform Designed Structures](https://arxiv.org/abs/2603.28990). arXiv:2603.28990.
- Yue, Y. et al. (2025). [Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?](https://limit-of-rlvr.github.io/) NeurIPS 2025.
- Bengio, Y. et al. (2024). [Managing extreme AI risks amid rapid progress](https://doi.org/10.1126/science.adn0117). Science, 384(6698).
- Qu, A. et al. (2026). [CORAL: Towards Autonomous Multi-Agent Evolution for Open-Ended Discovery](https://arxiv.org/abs/2604.01658). arXiv:2604.01658. [Code](https://github.com/Human-Agent-Society/CORAL).
- Bengio, Y. et al. (2026). [International AI Safety Report 2026](https://arxiv.org/abs/2602.21012). arXiv:2602.21012.
