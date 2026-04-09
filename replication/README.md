# DGM-H Replication

Faithful reimplementation of the [Hyperagents paper (Zhang et al., 2026)](https://arxiv.org/abs/2603.19461) on IMO-GradingBench.

Source: [facebookresearch/HyperAgents](https://github.com/facebookresearch/HyperAgents)

## Architecture

File-based agents with bash + editor tools. The meta agent modifies the full agent codebase on disk (task_agent.py, meta_agent.py, agent/ infrastructure). Each generation forks the parent's repo, runs the meta agent, evaluates the modified task agent, and adds to the archive.

Imports are rewritten (`from replication.agent.` -> `from agent.`) so that modifications to any file in the repo take effect when loaded via importlib. Module-level state (LLM clients, audit logging, tool scoping) is shared between the main package and repo copies.

## Differences from paper

| Aspect | Paper | Ours | Reason |
|--------|-------|------|--------|
| LLM client | litellm | markspace | Internal infrastructure |
| Sandboxing | Docker containers | Directory-scoped `_ALLOWED_ROOT` | Simpler, equivalent safety |
| Code capture | Git diffs | Full repo copy (~20KB/gen) | Simpler, negligible disk cost |
| Meta model | [Claude Sonnet 4.5](https://platform.claude.com/docs/en/docs/about-claude/models) | [Kimi K2.5 Turbo](https://fireworks.ai/models/fireworks/kimi-k2p5) | Flat-rate cost |
| Eval model | gpt-o4-mini | [GPT-OSS-120B](https://fireworks.ai/models/fireworks/gpt-oss-120b) | Similar tier on benchmarks; available open-weight via Fireworks |
| Temperature | 0.0 (Sonnet), effectively 1.0 (o4-mini, litellm drops param) | 0.0 (both) | Our models accept temperature; paper's o4-mini 1.0 is unintentional |
| Tool calling method | Text-based `<json>` extraction from response | Native API tool calling (`tools` parameter) | kimi-k2p5-turbo's text-based tool calling is unreliable (premature EOS bug); native API tool calling works 100% reliably |
| Bash session | New session per call (code) / persistent (description) | Persistent | Matches paper's stated intent |

## What matches exactly

- Parent selection: `score_child_prop` (sigmoid x child count penalty)
- Staged eval: 10 samples first, expand if score > 0, scale by fraction
- Score scaling: read-time via `_get_saved_score` (matches `gl_utils.py`)
- `valid_parent` tracking: failed meta agent invalidates parent
- Archive: all nodes added (failed nodes excluded from selection via `valid_parents`)
- Meta agent instruction: `"Modify any part of the codebase at {repo_path}."`
- Task agent: single LLM call, `<json>{"response": ...}</json>` format
- Tools: bash (120s timeout, sentinel-based) + editor (view/create/str_replace/insert/undo_edit)
- Tool use loop: max 40 calls, one tool per response, retry on truncation
- Data pipeline: filter Points [0,1,6,7], shuffle seed=42, stratified split 100/100/100
- Evaluation: exact string match (lowercased), normalized MAE with point mapping

## Key finding: metacognitive self-modification is real and gated by interface familiarity

The paper claims metacognitive self-modification enables the meta agent to discover
`eval_path` (an unused parameter in `meta_agent.forward()`). Our experiments confirm
this — with an important caveat about architecture.

**Self-modification in the file-based architecture (this replication):**
- kimi-k2p5-turbo self-modifies `meta_agent.py` in a variable fraction of generations (seed-dependent)
- It discovers the unused `eval_path` parameter and adds it to the instruction
- Across self-modifying arms, about 49% of high-scoring generations (val >= 0.5) wire eval_path into the instruction; 51% do not. Including the frozen-meta arm, 25% have it
- Low scores (0.03-0.27) routinely occur without eval_path or meta modification
- Strongest outlier: B/seed43 scores 0.670 (val) with frozen meta — first scored
  at gen 9 (val 0.57), best generation was gen 165 (val 0.67)

**Self-modification in the exec'd-string architecture (our prior experiments):**
- The same model (kimi-k2p5-turbo) produced **0 meta updates across 14 runs**
- The mechanism existed (`<NEW_META_CODE>` tag extraction) but was never triggered

**Why the difference:**
The file-based architecture makes self-modification frictionless — the model reads
`meta_agent.py` with the editor tool, sees code it can improve, and edits it with
`str_replace`. This is the same workflow it uses for every other file. The exec'd-string
architecture required the model to explicitly output `<NEW_META_CODE>` tags with a
rewritten Python function as a string — a special action the model never performed.

Same model, same latent capability, different interface, completely different behavior.
This validates the paper's architectural choice and explains why the paper's
results did not replicate with our original exec'd-string approach.

**Model training patterns matter as much as capability:**
Claude Sonnet 4.5 and kimi-k2p5-turbo have near-identical SWE-Bench Verified scores (77.2% vs
76.8%), but Claude reaches scoring within 3 generations while kimi takes 1-101 generations depending on seed.
The difference is not raw capability — it is coding workflow patterns:
- Claude reads all files first, plans explicitly, then executes (observed pattern)
- Claude notices code smells like unused parameters immediately (observed pattern)
- Claude verifies its edits (py_compile, git diff) before finishing (observed pattern)
- Claude uses parallel tool calls (24% of kimi calls are also multi-tool, showing
  kimi wants to parallelize but is slower to leverage it)

The paper's results with Sonnet may partially reflect Claude's coding workflow patterns, not a general property of DGM-H. The architecture enables
self-modification, but how quickly a model discovers and uses it depends on
model-specific behavioral patterns.

**The evolutionary loop finds more than formatting:**
A hand-crafted task agent (written by Claude Opus in one shot with knowledge of the
task) scores 0.370 — well below the evolved agents (0.510-0.670). The hand-crafted
agent massively over-grades (predicts "correct" 89% of the time vs 35% ground truth).
The evolutionary loop discovers calibration patterns that single-shot prompting does
not — whether through eval feedback or iterative code refinement.

**Code inheritance matters, parent selection does not:**
Arm D tournament data: on seed 42, blind candidates (from gen_0) score 0.0 in every
round. On seed 43, blind candidates occasionally score (up to 0.67), suggesting
the initial code can sometimes produce valid output depending on the seed's data split.
Across 5 seeds each, random selection (C: median 0.570) matches DGM-H selection
(A: median 0.580).

**Seed variance is large (best validation scores per seed):**
- Arm A (5 seeds): 0.580, 0.640, 0.510, 0.570, 0.650
- Arm B (5 seeds): 0.510, 0.670, 0.000, 0.580, 0.620
- Arm C (5 seeds): 0.610, 0.090, 0.600, 0.000, 0.570
- Arm D (2 seeds): 0.590, 0.640
- All arms completed at 200 generations. Arm medians (A: 0.580, B: 0.580,
  C: 0.570, D: 0.615) are not statistically separable. The paper uses 5 seeds.

## Differences from paper's implementation

- JSON extraction uses index-based tag matching (paper's lazy `.*?` regex truncates nested braces)
- Tool output escaped via `json.dumps` (paper embeds raw strings, breaks on quotes)
- Eval-level resume: if evaluation crashes mid-run, existing predictions are preserved
- Loop-level resume: re-running the same command resumes from the last completed generation

## Logging

Each run produces:

```
results/<experiment>/
  arm_a_full/
    seed42/
      run_config.json        # All parameters, platform
      archive.jsonl          # Archive state per generation
      summary.json           # Final results
      gen_N/
        metadata.json        # Parent, valid, raw scores, run_full_eval
        llm_calls.jsonl      # Every LLM call: messages, response, thinking, usage
        repo/                # Full agent source code snapshot
        eval_train/staged/   # Staged eval predictions + report
        eval_val/            # Val eval predictions + report
      test_eval/
        llm_calls.jsonl
        predictions.csv
        report.json
    seed43/
      ...
  arm_b_freeze/
    ...
  arm_c_random/
    ...
  arm_d_annealing/
    ...
```

## Run

```bash
# Prepare data (filter + split, only needed once)
python -m replication.data.prepare

EXP=replication_v1  # experiment name groups all runs

# Full DGM-H
python -m replication.run --experiment $EXP --iterations 200 --seed 42

# DGM-H without self-improvement (frozen meta agent)
python -m replication.run --experiment $EXP --iterations 200 --seed 42 --freeze-meta

# Random parent selection (ablation)
python -m replication.run --experiment $EXP --iterations 200 --seed 42 --selection random

# Annealing: decaying blind exploration
python -m replication.run --experiment $EXP --iterations 200 --seed 42 --annealing
```

## Open questions

**Where is the ceiling?** Evolved agents (0.510-0.670) exceed one-shot prompting (0.370). The gap is driven by calibration from eval feedback: novel information in the prompt enabling computation the model couldn't do without it. Whether this represents "new capability" or "better use of existing capability" may be a distinction without a difference. We have not measured the ceiling empirically. The paper claims "given appropriate tasks, the DGM-H has the potential to achieve unbounded open-ended self-improvement over time" but does not address whether improvement is bounded by frozen weights.

**Does the evolutionary algorithm matter?** The paper decomposes DGM-H into three components: (1) metacognitive self-modification, which we tested; (2) open-ended exploration / archive diversity, which the paper finds significant at p<0.05; (3) parent selection, which we tested and found non-significant, consistent with the paper's own appendix. All our arms use the full archive; we did not add an ablation that removes archive diversity to independently measure its contribution.

**What drives improvement if not eval_path?** Only 36% of high-scoring generations in self-modifying arms have eval_path wired into the instruction. The majority improve through iterative code reading and modification alone. B/seed43 reaches 0.670 with frozen meta and no eval_path.

**Does calibration generalize?** We observed calibration on IMO grading only. The paper's results across multiple domains suggest generalization, but we have not replicated those.

## Limitations

- **One task** (IMO grading). Cannot generalize to other domains.
- **Two models in replication** (kimi-k2p5-turbo, Claude Sonnet 4.6). Transfer experiments test four models.
- **2-5 seeds per arm.** Seed variance dominates; insufficient for statistical comparison between arms.
- **No optimal-baseline comparison.** We tested against a one-shot agent, not a human iterating with eval feedback.
- **Architecture confound.** File-based vs exec'd-string is binary. We have not explored intermediate designs.
- **Format-fixing vs capability.** Initial agents score 0.0 due to wrong output format. The 0.370-to-0.670 gap includes both format-fixing AND calibration. The paper does not distinguish these.

## Experiment plan

4 arms x 5 seeds x 200 iterations = 20 runs (17 completed, arm D has 2 seeds):

| Arm | Description | Flag |
|-----|-------------|------|
| A: Full DGM-H | `score_child_prop` selection, self-modifying meta agent | (default) |
| B: Frozen meta | Same selection, meta agent always loaded from gen_0 | `--freeze-meta` |
| C: Random selection | Random parent, self-modifying meta agent | `--selection random` |
| D: Annealing | Decaying probability of forking from initial agent (blind exploration) | `--annealing` |

Arm D tests whether DGM-H's evolutionary search is meaningful. Each tournament round (triggered with probability p, starting at 0.5) runs BOTH a blind candidate (meta agent starts from gen_0) and a directed candidate (meta agent builds on selected parent), evaluates both, and keeps the winner. p decays by 0.9x after each tournament. If blind candidates frequently win, the algorithm's parent selection and code inheritance aren't helping — confirming the scaffolding ceiling.
