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
    if not raw or not isinstance(raw, str):
        return "incorrect"
        
    raw = raw.strip().lower()
    
    # Remove common punctuation and whitespace variations
    raw = raw.strip(".!?,:;\"'")
    
    # Handle empty string after stripping
    if not raw:
        return "incorrect"

    # Exact match first (most reliable)
    if raw in _VALID_LABELS:
        if raw == "almost":
            return "partial"
        return raw

    # Handle common variations and synonyms - check NEGATIVE cases first
    if raw in {"wrong", "false", "error", "invalid", "no", "none", 
               "not correct", "not right", "not valid", "bad", "fail",
               "failed", "fails", "unsatisfactory", "unacceptable", 
               "not acceptable", "not satisfactory", "not complete"}:
        return "incorrect"
    
    # Check PARTIAL cases second - expanded list
    if raw in {"almost correct", "nearly correct", "mostly correct", "close",
               "almost", "nearly", "mostly", "incomplete", "partially correct",
               "some progress", "partial credit", "half correct", "partly correct",
               "partially right", "mostly right", "nearly right", "close to correct",
               "minor errors", "small errors", "slight errors", "few errors",
               "significant progress", "substantial progress", "good progress",
               "on the right track", "heading in right direction", "approaching correct",
               "almost there", "nearly there", "mostly there", "largely correct",
               "partially complete", "mostly complete", "nearly complete",
               "very close", "nearly solved", "almost solved",
               "minor gap", "small gap", "trivial gap", "minimal error",
               "partially achieved", "partially met", "mostly achieved"}:
        return "partial"
    
    # Check POSITIVE cases last
    if raw in {"right", "true", "valid", "yes", "full", "complete", 
               "fully correct", "perfect", "excellent", "satisfactory",
               "acceptable", "pass", "passed", "complete solution",
               "fully solved", "correct solution", "valid solution",
               "complete and correct", "correct and complete", "achieved", "met"}:
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
    if "incomplete" in raw or "gap" in raw or "missing" in raw or "unfinished" in raw:
        return "partial"
    # Only check for "correct" if no negative indicators found
    if "correct" in raw or "right" in raw or "valid" in raw or "proper" in raw:
        return "correct"

    # Default to incorrect when uncertain - safer than defaulting to correct
    return "incorrect"


def _extract_prediction(text: str) -> str | None:
    """Try multiple strategies to pull a label out of model output.
    
    Returns the first valid label found, prioritizing structured JSON output.
    Handles the "almost" label by mapping it to "partial".
    """
    if not text or not isinstance(text, str):
        return None
        
    text = text.strip()
    
    # Helper to normalize with explicit "almost" handling
    def _normalize_with_almost(raw: str) -> str | None:
        raw = raw.strip().lower()
        if raw == "almost":
            return "partial"
        label = _normalise_label(raw)
        if label in {"correct", "partial", "incorrect"}:
            return label
        return None
    
    # Strategy 1: <json>...</json> block (most reliable)
    extracted = _extract_jsons(text)
    if extracted:
        for obj in reversed(extracted):
            # First try the "response" field (primary output)
            if "response" in obj:
                label = _normalize_with_almost(str(obj["response"]))
                if label:
                    return label
            # Check "almost_count" and "partial_count" fields for decision logic
            if "almost_count" in obj or "partial_count" in obj:
                almost_count = str(obj.get("almost_count", "0")).lower()
                partial_count = str(obj.get("partial_count", "0")).lower()
                has_gaps = str(obj.get("has_gaps", "")).lower()
                # If has gaps and has any partial/almost achievements -> partial
                if has_gaps in ["yes", "true", "1"] and (almost_count not in ["0", "", "none"] or partial_count not in ["0", "", "none"]):
                    return "partial"
            # Also check "verification" field which may contain the verdict
            if "verification" in obj:
                label = _normalize_with_almost(str(obj["verification"]))
                if label:
                    return label
            # Check "reasoning" field for hints about "almost"
            if "reasoning" in obj:
                reasoning = str(obj["reasoning"]).lower()
                if "almost" in reasoning and "not almost" not in reasoning:
                    # Check if there's a "but" or "however" indicating gaps
                    if any(word in reasoning for word in ["but", "however", "gap", "missing", "error", "flaw"]):
                        return "partial"

    # Strategy 2: bare JSON object with "response" field (flexible regex)
    # Try to find JSON objects with response field, handling nested braces
    json_patterns = [
        r'"response"\s*:\s*"(correct|partial|incorrect|almost)"',
        r"'response'\s*:\s*'(correct|partial|incorrect|almost)'",
        r'"response"\s*:\s*"([^"]+)"',
        r"'response'\s*:\s*'([^']+)'",
    ]
    for pattern in json_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            label = _normalize_with_almost(match.group(1))
            if label:
                return label

    # Strategy 3: Explicit field patterns (grade:, verdict:, etc.)
    field_patterns = [
        r'(?i)\bgrade\s*[:=]\s*"?(correct|partial|incorrect|almost)"?\b',
        r'(?i)\bverdict\s*[:=]\s*"?(correct|partial|incorrect|almost)"?\b',
        r'(?i)\banswer\s*[:=]\s*"?(correct|partial|incorrect|almost)"?\b',
        r'(?i)\bresult\s*[:=]\s*"?(correct|partial|incorrect|almost)"?\b',
        r'(?i)\bstatus\s*[:=]\s*"?(correct|partial|incorrect|almost)"?\b',
        r'(?i)\bevaluation\s*[:=]\s*"?(correct|partial|incorrect|almost)"?\b',
        r'(?i)\bprediction\s*[:=]\s*"?(correct|partial|incorrect|almost)"?\b',
        r'(?i)\blabel\s*[:=]\s*"?(correct|partial|incorrect|almost)"?\b',
    ]
    for pattern in field_patterns:
        m = re.search(pattern, text)
        if m:
            label = _normalize_with_almost(m.group(1))
            if label:
                return label

    # Strategy 4: Look for standalone labels in specific contexts
    # Check for "incorrect" first (contains "correct" as substring)
    context_patterns = [
        r'(?i)the\s+answer\s+is\s+(incorrect|partial|correct|almost)',
        r'(?i)grade\s+is\s+(incorrect|partial|correct|almost)',
        r'(?i)verdict\s+is\s+(incorrect|partial|correct|almost)',
        r'(?i)therefore\s*[:,-]?\s*(incorrect|partial|correct|almost)',
        r'(?i)thus\s*[:,-]?\s*(incorrect|partial|correct|almost)',
        r'(?i)final\s*[:,-]?\s*(incorrect|partial|correct|almost)',
        r'(?i)conclusion\s*[:,-]?\s*(incorrect|partial|correct|almost)',
        r'(?i)assessment\s*[:,-]?\s*(incorrect|partial|correct|almost)',
        r'(?i)final\s+verdict\s*[:,-]?\s*(incorrect|partial|correct|almost)',
        r'(?i)final\s+grade\s*[:,-]?\s*(incorrect|partial|correct|almost)',
    ]
    for pattern in context_patterns:
        m = re.search(pattern, text)
        if m:
            return _normalize_with_almost(m.group(1))

    # Strategy 5: Look for labels in quotes
    quote_patterns = [
        r'"(correct|partial|incorrect|almost)"',
        r"'(correct|partial|incorrect|almost)'",
        r'`(correct|partial|incorrect|almost)`',
    ]
    for pattern in quote_patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            label = _normalize_with_almost(m.group(1))
            if label:
                return label

    # Strategy 6: Look for label at the end of the text (often where final verdict appears)
    # Check the last 500 characters for explicit verdict statements
    tail = text[-500:] if len(text) > 500 else text
    tail_patterns = [
        r'(?i)\bverdict\s*[:=]?\s*"?(correct|partial|incorrect|almost)"?\b',
        r'(?i)\bgrade\s*[:=]?\s*"?(correct|partial|incorrect|almost)"?\b',
        r'(?i)\bresponse\s*[:=]?\s*"?(correct|partial|incorrect|almost)"?\b',
        r'(?i)\bfinal\s+answer\s*[:=]?\s*"?(correct|partial|incorrect|almost)"?\b',
    ]
    for pattern in tail_patterns:
        m = re.search(pattern, tail)
        if m:
            label = _normalize_with_almost(m.group(1))
            if label:
                return label
    
    # Strategy 6b: Check tail for "almost" keyword (maps to partial)
    if re.search(r'(?i)\balmost\b', tail):
        return "partial"

    # Strategy 7: Last resort - find any valid label
    # Check in order: incorrect, partial, correct (to avoid substring issues)
    for label in ["incorrect", "partial", "correct"]:
        # Look for the label as a whole word
        pattern = rf'(?i)\b{label}\b'
        if re.search(pattern, text):
            return label
            
    # Strategy 8: Check for "almost" keyword anywhere (maps to partial)
    if re.search(r'(?i)\balmost\b', text):
        return "partial"
    
    # Strategy 9: Check for "almost" in JSON response field specifically
    # Sometimes the model outputs "almost" in the response field when guidelines have "(Almost)"
    almost_patterns = [
        r'"response"\s*:\s*"almost"',
        r"'response'\s*:\s*'almost'",
        r'"grade"\s*:\s*"almost"',
        r"'grade'\s*:\s*'almost'",
        r'"verdict"\s*:\s*"almost"',
        r"'verdict'\s*:\s*'almost'",
        r'"label"\s*:\s*"almost"',
        r"'label'\s*:\s*'almost'",
    ]
    for pattern in almost_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return "partial"
    
    # Strategy 10: Check for "almost" in other common JSON fields
    almost_json_patterns = [
        r'"result"\s*:\s*"almost"',
        r"'result'\s*:\s*'almost'",
        r'"evaluation"\s*:\s*"almost"',
        r"'evaluation'\s*:\s*'almost'",
        r'"prediction"\s*:\s*"almost"',
        r"'prediction'\s*:\s*'almost'",
    ]
    for pattern in almost_json_patterns:
        if re.search(pattern, text, re.IGNORECASE):
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
- **CRITICAL**: "(Almost)" in the grading guidelines means the student is very close to correct but has errors or gaps preventing full correctness. "(Almost)" ALWAYS maps to **"partial"**, NEVER to "correct".

**INCORRECT**: The student's answer is wrong or makes no meaningful progress.
- The student made no substantive progress toward solving the problem.
- No key insights, lemmas, or structural observations from the grading guidelines are present.
- **CRITICAL**: If the student achieved ZERO milestones from the "(Partial)" or "(Almost)" sections of the grading guidelines, the grade MUST be "incorrect".
- Merely restating the problem or making trivial observations earns "incorrect".
- A correct final numerical answer with no supporting argument is **incorrect** unless guidelines explicitly award credit for the answer alone.

### Critical distinctions

**partial vs incorrect**: To earn "partial", the student must have achieved at least one specific milestone from the grading guidelines. Merely attempting the problem or writing relevant-looking math does NOT earn "partial". If NO milestones are achieved, the grade MUST be "incorrect".

**partial vs correct**: If the student's solution has ANY non-trivial gap, unjustified claim, missing case, or logical error — even a small one — the grade is **"partial"**, not "correct". "Almost complete" or "nearly correct" means **"partial"**.

**almost vs correct**: The label "(Almost)" in the grading guidelines describes work that is very close but NOT fully correct due to errors or gaps. "(Almost)" ALWAYS maps to **"partial"**, NEVER to "correct". Only assign "correct" if the solution is truly complete and gap-free.

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

**Step 1 — Identify milestones:** List each specific milestone from the grading guidelines' "(Partial)" section.

**Step 2 — Identify "(Almost)" criteria:** List each criterion from the "(Almost)" section.

**Step 3 — Milestone achievement check:** For each (Partial) milestone, state: ACHIEVED / NOT ACHIEVED / PARTIALLY ACHIEVED, with brief justification.

**Step 4 — "(Almost)" criteria check:** For each (Almost) criterion, state: MET / NOT MET with justification.

**Step 5 — Count achievements:**
- Count of (Partial) milestones achieved: __
- Count of (Almost) criteria met: __
- Total partial achievements: __

**Step 6 — Gap analysis:** List any remaining gaps, unjustified claims, missing cases, or logical errors. If ANY non-trivial gap exists, the answer CANNOT be "correct".

**Step 7 — Final verdict selection (BE CONSERVATIVE):**
- If solution is 100% complete with ZERO gaps → "correct" 
- If AT LEAST ONE (Partial) milestone OR (Almost) criterion achieved, but has gaps/errors → "partial"
- If ZERO milestones/criteria achieved → "incorrect" (this is the DEFAULT when in doubt)

**Step 8 — Verification check:** Before finalizing, verify your verdict:
- If you selected "correct": Confirm there are NO gaps. If any doubt, change to "partial".
- If you selected "partial": Confirm at least one milestone/criterion was achieved. If NO achievements, change to "incorrect".
- If you selected "incorrect": Confirm NO milestones AND NO criteria were achieved.

**Step 9 — Response field validation (CRITICAL):**
Before outputting your final JSON, verify:
1. The "response" field contains ONLY one of: "correct", "partial", or "incorrect" (lowercase)
2. If you wrote "almost" in the response field, CHANGE IT to "partial"
3. If has_gaps=yes but response="correct", CHANGE response to "partial"
4. If partial_count + almost_count > 0 but response="incorrect", CHANGE response to "partial"
5. If partial_count + almost_count == 0 but response="partial", CHANGE response to "incorrect"

**CRITICAL**: Your response field MUST contain exactly one of: "correct", "partial", or "incorrect" (lowercase, no extra text).
- **DO NOT** output "almost" - if the student meets "(Almost)" criteria, output "partial" instead.
- The "almost" label is NOT a valid output - it must be mapped to "partial".
- Double-check your response field before submitting.

Respond with a JSON block in the following format:
<json>
{{
    "milestones": "<list each (Partial) milestone and status: ACHIEVED/NOT ACHIEVED>",
    "almost_criteria": "<list each (Almost) criterion and status: MET/NOT MET>",
    "partial_count": "<number of (Partial) milestones achieved>",
    "almost_count": "<number of (Almost) criteria met>",
    "gaps_analysis": "<what is missing or wrong; write 'none' if fully correct>",
    "has_gaps": "<yes/no - are there any gaps preventing 'correct'?>",
    "verification": "<confirm: if has_gaps=yes then response must be partial or incorrect; if partial_count+almost_count=0 then response must be incorrect>",
    "reasoning": "<one-sentence justification of your verdict>",
    "response": "<correct | partial | incorrect>"
}}
</json>

**IMPORTANT**: The "response" field MUST be exactly one of: "correct", "partial", or "incorrect". Never use "almost" in the response field."""

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
            
            # Post-processing: validate against JSON fields if available
            for src in sources:
                extracted = _extract_jsons(src)
                if extracted:
                    for obj in reversed(extracted):
                        # Get key fields for validation
                        has_gaps = str(obj.get("has_gaps", "")).lower().strip()
                        resp_field = str(obj.get("response", "")).lower().strip()
                        partial_count_str = str(obj.get("partial_count", "0")).lower().strip()
                        almost_count_str = str(obj.get("almost_count", "0")).lower().strip()
                        gaps_analysis = str(obj.get("gaps_analysis", "")).lower().strip()
                        milestones = str(obj.get("milestones", "")).lower()
                        almost_criteria = str(obj.get("almost_criteria", "")).lower()
                        
                        # Parse counts as integers
                        try:
                            partial_count = int(partial_count_str) if partial_count_str not in ["", "none", "n/a"] else 0
                        except (ValueError, TypeError):
                            partial_count = 0
                        try:
                            almost_count = int(almost_count_str) if almost_count_str not in ["", "none", "n/a"] else 0
                        except (ValueError, TypeError):
                            almost_count = 0
                        
                        total_achievements = partial_count + almost_count
                        
                        # Rule 1: "almost" in response field -> map to partial
                        if resp_field == "almost":
                            prediction = "partial"
                            break
                        
                        # Rule 2: has_gaps=yes but response="correct" -> override to partial
                        if has_gaps in ["yes", "true", "1"] and resp_field == "correct":
                            prediction = "partial"
                            break
                        
                        # Rule 3: has achievements but response="incorrect" -> override to partial
                        if total_achievements > 0 and resp_field == "incorrect":
                            prediction = "partial"
                            break
                        
                        # Rule 4: no achievements but response="partial" -> override to incorrect
                        if total_achievements == 0 and resp_field == "partial":
                            # Check if milestones field shows any achievements
                            has_achievements = False
                            if milestones and milestones != "none":
                                if "achieved" in milestones or "met" in milestones:
                                    has_achievements = True
                            if almost_criteria and almost_criteria != "none":
                                if "met" in almost_criteria:
                                    has_achievements = True
                            if not has_achievements:
                                prediction = "incorrect"
                            break
                        
                        # Rule 5: gaps_analysis indicates gaps but response="correct" -> override to partial
                        if gaps_analysis and gaps_analysis not in ["none", "", "n/a", "no gaps"]:
                            if resp_field == "correct":
                                prediction = "partial"
                                break
                        
                        # Rule 6: Use response field as authoritative if it passes basic validation
                        if resp_field in ["correct", "partial", "incorrect"]:
                            prediction = resp_field
                            break
                    break
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return prediction, msg_history
