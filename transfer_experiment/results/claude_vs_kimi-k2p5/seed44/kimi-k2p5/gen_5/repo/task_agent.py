"""
Task agent: solves a given task with a single LLM call.

Reimplemented from facebookresearch/HyperAgents task_agent.py.
Same interface, same JSON output format, same extraction logic.

This is the INITIAL task agent. The meta agent modifies this file
during self-improvement. The evaluation harness loads whatever
task_agent.py exists at the agent's repo path.
"""

from __future__ import annotations

import json
import logging
import re

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)

# Valid prediction labels (the harness also uses "almost" as a label,
# which we treat as "partial" since it represents near-correct work).
_VALID_LABELS = {"correct", "partial", "incorrect", "almost"}


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    """
    results = []
    search_from = 0
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            continue
    return results or None


def _normalise_label(raw: str) -> str:
    """Map a raw model response string to one of the canonical labels.

    The evaluation harness accepts: "correct", "partial", "incorrect".
    "almost" (used in some rubrics) is treated as "partial".

    Uses strict priority ordering: incorrect > partial > correct to avoid
    false positives. The word "incorrect" contains "correct" as a substring,
    so we must check for "incorrect" first in all matching strategies.
    """
    raw = raw.strip().lower()
    
    # Remove common punctuation and whitespace variations
    raw = raw.strip(".!?,:;\"'")

    # Exact match first (most reliable)
    if raw in _VALID_LABELS:
        if raw == "almost":
            return "partial"
        return raw

    # Handle common variations and synonyms - check NEGATIVE cases first
    if raw in {"wrong", "false", "error", "invalid", "no", "none", 
               "not correct", "not right", "not valid", "bad"}:
        return "incorrect"
    
    # Check PARTIAL cases second
    if raw in {"almost correct", "nearly correct", "mostly correct", "close",
               "almost", "nearly", "mostly", "incomplete", "partially correct",
               "some progress", "partial credit", "half correct"}:
        return "partial"
    
    # Check POSITIVE cases last
    if raw in {"right", "true", "valid", "yes", "full", "complete", 
               "fully correct", "perfect"}:
        return "correct"

    # Prefix matching - STRICT ORDER: incorrect first, then partial, then correct
    if raw.startswith("incorrect") or raw.startswith("wrong") or raw.startswith("error"):
        return "incorrect"
    if raw.startswith("almost") or raw.startswith("partial") or raw.startswith("nearly"):
        return "partial"
    if raw.startswith("correct") or raw.startswith("right"):
        return "correct"

    # Last-resort keyword scan - check longer/more-specific words first
    # Check for "incorrect" BEFORE "correct" because "incorrect" contains "correct"
    if "incorrect" in raw or "wrong" in raw or "error" in raw or "mistake" in raw:
        return "incorrect"
    if "almost" in raw or "nearly" in raw or "mostly" in raw or "partial" in raw:
        return "partial"
    if "incomplete" in raw or "gap" in raw or "missing" in raw:
        return "partial"
    # Only check for "correct" if no negative indicators found
    if "correct" in raw or "right" in raw or "valid" in raw:
        return "correct"

    # Default to incorrect when uncertain - safer than defaulting to correct
    return "incorrect"


def _extract_prediction(text: str) -> str | None:
    """Try multiple strategies to pull a label out of model output.
    
    Returns the first valid label found, prioritizing structured JSON output.
    """
    if not text or not isinstance(text, str):
        return None
        
    text = text.strip()
    
    # Strategy 1: <json>...</json> block (most reliable)
    extracted = _extract_jsons(text)
    if extracted:
        for obj in reversed(extracted):
            if "response" in obj:
                label = _normalise_label(str(obj["response"]))
                if label in _VALID_LABELS or label == "partial":  # "partial" is our canonical form
                    return label

    # Strategy 2: bare JSON object with "response" field (flexible regex)
    # Try to find JSON objects with response field, handling nested braces
    json_patterns = [
        r'\{[^{}]*"response"\s*:\s*"([^"]+)"[^{}]*\}',
        r"\{[^{}]*'response'\s*:\s*'([^']+)'[^{}]*\}",
        r'\{\s*"response"\s*:\s*"([^"]+)"\s*\}',
        r"\{\s*'response'\s*:\s*'([^']+)'\s*\}",
    ]
    for pattern in json_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            label = _normalise_label(match.group(1))
            if label in {"correct", "partial", "incorrect"}:
                return label

    # Strategy 3: Explicit field patterns (grade:, verdict:, etc.)
    field_patterns = [
        r'(?i)\bgrade\s*[:=]\s*"?(correct|partial|incorrect|almost)"?\b',
        r'(?i)\bverdict\s*[:=]\s*"?(correct|partial|incorrect|almost)"?\b',
        r'(?i)\banswer\s*[:=]\s*"?(correct|partial|incorrect|almost)"?\b',
        r'(?i)\bresult\s*[:=]\s*"?(correct|partial|incorrect|almost)"?\b',
        r'(?i)\bstatus\s*[:=]\s*"?(correct|partial|incorrect|almost)"?\b',
        r'(?i)\bevaluation\s*[:=]\s*"?(correct|partial|incorrect|almost)"?\b',
    ]
    for pattern in field_patterns:
        m = re.search(pattern, text)
        if m:
            return _normalise_label(m.group(1))

    # Strategy 4: Look for standalone labels in specific contexts
    # Check for "incorrect" first (contains "correct" as substring)
    context_patterns = [
        r'(?i)the\s+answer\s+is\s+(incorrect|partial|correct)',
        r'(?i)grade\s+is\s+(incorrect|partial|correct)',
        r'(?i)verdict\s+is\s+(incorrect|partial|correct)',
        r'(?i)therefore\s*[:,-]?\s*(incorrect|partial|correct)',
        r'(?i)thus\s*[:,-]?\s*(incorrect|partial|correct)',
        r'(?i)final\s*[:,-]?\s*(incorrect|partial|correct)',
    ]
    for pattern in context_patterns:
        m = re.search(pattern, text)
        if m:
            return _normalise_label(m.group(1))

    # Strategy 5: Last resort - find any valid label
    # Check in order: incorrect, partial, correct (to avoid substring issues)
    for label in ["incorrect", "partial", "correct"]:
        # Look for the label as a whole word
        pattern = rf'(?i)\b{label}\b'
        if re.search(pattern, text):
            return label
            
    # Strategy 6: Check for "almost" which maps to partial
    if re.search(r'(?i)\balmost\b', text):
        return "partial"

    return None


class TaskAgent:
    """Task agent that grades student answers to competition math problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single grading problem.

        Args:
            inputs: dict with keys:
                - domain: subject area of the problem
                - problem: the competition math problem statement
                - solution: the official/reference solution
                - grading_guidelines: rubric describing what earns credit
                - student_answer: the student's submitted solution attempt

        Returns:
            (prediction, msg_history) where prediction is one of:
                "correct"   – the student's answer is fully correct
                "partial"   – the student's answer is partially correct
                "incorrect" – the student's answer is wrong or missing
        """
        problem        = inputs.get("problem", "")
        solution       = inputs.get("solution", "")
        guidelines     = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        domain         = inputs.get("domain", "")

        instruction = f"""You are a strict but fair expert grader evaluating a student's solution to a competition mathematics problem.

## Your task
Read the problem, the official solution, the grading guidelines, and the student's answer.
Then decide whether the student's answer deserves a grade of **"correct"**, **"partial"**, or **"incorrect"**.

### Grading definitions — read carefully before grading

**CORRECT**: The student's answer is fully correct and complete. 
- Every key step, lemma, and logical connection required by the grading guidelines is present and sound.
- A minor notational slip or trivially-fixable arithmetic error is acceptable.
- **CRITICAL**: Any missing proof step, unjustified claim, or logical gap disqualifies "correct".
- When in doubt between "correct" and "partial", choose **"partial"**.

**PARTIAL**: The student's answer contains meaningful correct progress but is not fully correct.
- The student achieved at least one specific milestone from the grading guidelines (e.g., a correct key lemma, invariant, special case, or structural observation).
- The progress must be substantive — it must address a genuine difficulty of the problem.
- "Almost correct", "nearly correct", or "mostly correct" answers belong here.
- An answer with a small but non-trivial error or gap should be "partial", not "correct".
- If the grading guidelines mention "(Partial)" or "(Almost)" criteria, achieving ANY of these earns "partial".

**INCORRECT**: The student's answer is wrong or makes no meaningful progress.
- The answer is fundamentally flawed, trivially incomplete, or does not engage with the core difficulty.
- Merely restating the problem or making trivial observations earns "incorrect".
- A correct final numerical answer with no supporting argument is **incorrect** unless guidelines explicitly award credit for the answer alone.
- If the student achieved NONE of the specific milestones listed in the grading guidelines, the grade is "incorrect".

### Critical distinctions

**partial vs incorrect**: To earn "partial", the student must have achieved at least one specific milestone from the grading guidelines. Merely attempting the problem or writing relevant-looking math does NOT earn "partial".

**partial vs correct**: If the student's solution has ANY non-trivial gap, unjustified claim, missing case, or logical error — even a small one — the grade is **"partial"**, not "correct".

**almost**: The label "almost" in grading guidelines maps to **"partial"**, never to "correct".

---

## Problem ({domain})
{problem}

---

## Official Solution
{solution}

---

## Grading Guidelines
{guidelines}

---

## Student's Answer
{student_answer}

---

## Grading procedure
Work through the following steps explicitly before giving your verdict:

**Step 1 — Identify milestones:** List each specific milestone from the grading guidelines, including any "(Partial)" or "(Almost)" criteria.

**Step 2 — Milestone achievement check:** For each milestone, state explicitly: ACHIEVED / NOT ACHIEVED / PARTIALLY ACHIEVED, with brief justification.

**Step 3 — Gap analysis:** List any remaining gaps, unjustified claims, missing cases, or logical errors. If any non-trivial gap exists, the answer cannot be "correct".

**Step 4 — Final verdict selection:**
- If ALL milestones achieved AND zero non-trivial gaps → "correct"
- If AT LEAST ONE milestone achieved but gaps remain → "partial"  
- If NO milestones achieved → "incorrect"

**IMPORTANT**: Your response field MUST contain exactly one of: "correct", "partial", or "incorrect" (lowercase, no extra text).

Respond with a JSON block in the following format:
<json>
{{
    "milestones": "<list each guideline milestone and status>",
    "gaps_analysis": "<what is missing or wrong; write 'none' if fully correct>",
    "reasoning": "<one-sentence justification of your verdict>",
    "response": "<correct | partial | incorrect>"
}}
</json>"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from the model's response, trying multiple sources.
        prediction = "incorrect"
        try:
            # Try the last assistant message first, then the raw response string.
            sources = []
            if msg_history:
                sources.append(msg_history[-1].get("text", ""))
            if response:
                sources.append(response)

            for src in sources:
                label = _extract_prediction(src)
                if label is not None:
                    prediction = label
                    break
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return prediction, msg_history
