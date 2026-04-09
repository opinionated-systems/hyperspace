"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."
"""

from __future__ import annotations

import json
import logging
import os

from agent.agentic_loop import chat_with_agent
from agent.llm_client import META_MODEL

logger = logging.getLogger(__name__)


def _load_eval_summary(eval_path: str) -> str:
    """Load and summarise evaluation results from the most recent eval directory."""
    if not eval_path or not os.path.exists(eval_path):
        return "No previous evaluation results available."

    lines = []

    # Try to load val and train reports
    for split in ("eval_val", "eval_train"):
        report_path = os.path.join(eval_path, split, "report.json")
        if not os.path.exists(report_path):
            continue
        try:
            with open(report_path) as f:
                data = json.load(f)
            acc  = data.get("overall_accuracy", "N/A")
            nmae = data.get("normalized_mean_absolute_error", "N/A")
            lines.append(f"  {split}: accuracy={acc:.3f}, NMAE={nmae:.3f}" if isinstance(acc, float) else f"  {split}: {data}")

            # Per-label breakdown
            by_label = data.get("accuracy_by_label", {})
            for label in ("correct", "almost", "partial", "incorrect"):
                info = by_label.get(label, {})
                prec = info.get("precision", "?")
                rec  = info.get("recall", "?")
                tot  = info.get("total", "?")
                cor  = info.get("correct", "?")
                lines.append(
                    f"    {label:10s}: precision={prec:.2f}, recall={rec:.2f}, "
                    f"correct={cor}/{tot}"
                    if isinstance(prec, float) else
                    f"    {label}: {info}"
                )
        except Exception as e:
            lines.append(f"  {split}: could not parse ({e})")

    # Compute confusion matrix from predictions CSV
    for split in ("eval_val", "eval_train"):
        pred_path = os.path.join(eval_path, split, "predictions.csv")
        if not os.path.exists(pred_path):
            continue
        try:
            import csv
            from collections import Counter
            rows = []
            with open(pred_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(row)
            errors = Counter(
                (r["Reward"].lower(), r["prediction"])
                for r in rows
                if r["Reward"].lower() != r["prediction"]
            )
            if errors:
                lines.append(f"\n  {split} confusion (true→pred, top errors):")
                for (true, pred), cnt in sorted(errors.items(), key=lambda x: -x[1])[:8]:
                    lines.append(f"    true={true:10s} → pred={pred:10s}: {cnt}")
        except Exception as e:
            lines.append(f"  {split} confusion: could not parse ({e})")

    return "\n".join(lines) if lines else "Evaluation results available but could not be parsed."


class MetaAgent:
    """Meta agent that self-improves by modifying the codebase."""

    def __init__(self, model: str = META_MODEL, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self.log_fn = logger.info

    def forward(
        self,
        repo_path: str,
        eval_path: str,
        iterations_left: int | None = None,
    ) -> list[dict]:
        """Run the meta agent to modify the codebase.

        Args:
            repo_path: path to the agent's repository
            eval_path: path to previous evaluation results
            iterations_left: remaining iterations (budget info)

        Returns:
            Message history from the agentic loop
        """
        eval_summary = _load_eval_summary(eval_path)

        instruction = f"""You are a meta-agent tasked with improving a task agent that grades student solutions to competition mathematics problems.

## Repository
`{repo_path}`

## Iterations remaining
{iterations_left if iterations_left is not None else 'unknown'}

## Previous evaluation results
{eval_summary}

## Your task
Modify any part of the codebase at `{repo_path}` to improve the task agent's grading accuracy.
The task agent is evaluated on four labels: **correct**, **almost**, **partial**, **incorrect**.
The primary metric is overall accuracy; secondary metric is NMAE (lower is better).

## Key files
- `task_agent.py` — main grading logic, prompt, and label extraction/post-processing
- `meta_agent.py` — this meta-agent's own implementation

## Diagnosis guide (read the confusion matrix above carefully)

### Common failure modes and fixes

**"almost" recall = 0% (most common)**
- The model outputs "correct" when the (Almost) flaw IS PRESENT.
- Fix: In the prompt, move the (Almost) check BEFORE the gap analysis. Force the model
  to quote evidence for each (Almost) criterion before declaring "correct".
- Fix: In `_post_process_prediction`, ensure that `_almost_flaw_is_present(almost_txt)`
  overrides "correct" → "almost" BEFORE the gaps rule fires.

**"partial" recall too low (true=partial predicted as incorrect)**
- The model is too strict about milestone achievement.
- Fix: Add more explicit "be generous" language in the prompt for (Partial) milestones.
- Fix: In `_post_process_prediction`, Rule 4 (milestone achieved → upgrade to partial)
  should fire even when the milestones field uses softer language like "partially".

**"incorrect" precision too low (true=incorrect predicted as partial)**
- The model awards partial credit too liberally.
- Fix: Tighten the milestone check — require substantive engagement, not just mention.
- Fix: In `_post_process_prediction`, Rule 5 (all NOT ACHIEVED → downgrade to incorrect)
  should be robust to edge cases.

**"almost" predicted as "partial" or "correct"**
- The model identifies the flaw but outputs the wrong label.
- Fix: Add a post-processing rule that upgrades "partial" → "almost" when the
  almost_assessment field says "IS PRESENT" and the reasoning mentions minor/sound.

## Guidelines
- Use the `editor` and `bash` tools to make changes.
- View files before modifying them.
- Make focused, targeted improvements based on the confusion matrix.
- Preserve the existing interface (class names, method signatures, return types).
- The `response` field in the JSON output must be one of: correct, almost, partial, incorrect.

Start by reading `task_agent.py` to understand the current implementation, then make targeted improvements."""

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        return msg_history
