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


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.
    Falls back to extracting any JSON object in the text using a regex.
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
            # Try to fix common JSON issues
            try:
                fixed = re.sub(r',(\s*[}\]])', r'\1', inner)
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                continue
    # If no <json> blocks were found, try to find any JSON object using a regex fallback.
    if not results:
        json_candidates = re.findall(r'\{[^{}]*\}', text, flags=re.DOTALL)
        for cand in json_candidates:
            try:
                results.append(json.loads(cand))
            except json.JSONDecodeError:
                continue
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract key fields for better context
        domain = inputs.get('domain', 'Unknown')
        problem = inputs.get('problem', '')
        solution = inputs.get('solution', '')
        grading_guidelines = inputs.get('grading_guidelines', '')
        student_answer = inputs.get('student_answer', '')
        
        instruction = f"""You are an expert mathematical grader evaluating IMO-level solutions. Your task is to classify the student's solution into exactly ONE of these four categories: Correct, Almost, Partial, or Incorrect.

=== PROBLEM INFORMATION ===

DOMAIN: {domain}

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER TO EVALUATE:
{student_answer}

=== CLASSIFICATION DEFINITIONS ===

**Correct**: The solution is complete, rigorous, and correct. ALL claims are proven, ALL cases are handled, and there are NO logical gaps. The student would receive full marks.

**Almost**: The solution has a COMPLETE proof structure with only MINOR issues (e.g., small calculation errors, missing trivial cases, minor notational flaws). The main logical flow is essentially correct - if you fixed the minor issues, you'd have a full solution. This is 90-95% complete. "Essentially correct, just needs polishing."

**Partial**: The solution shows SUBSTANTIAL progress with KEY mathematical insights, but the proof structure is INCOMPLETE. The student found something that actually helps solve the problem - perhaps a useful lemma, the right approach, or a significant non-trivial insight. However, substantial work remains to complete the proof. "Found the key idea, but substantial work remains."

**Incorrect**: The solution shows NO meaningful mathematical progress. The student is just restating the problem, making trivial observations, or going down a completely wrong path. There is no insight that would help solve the problem. Be CONSERVATIVE - if the student only states obvious facts or makes no real attempt, this is INCORRECT.

=== CRITICAL DISTINCTIONS ===

**Partial vs Incorrect** (THIS IS THE MOST IMPORTANT DISTINCTION):
- PARTIAL: Ask yourself: "Did the student find something that actually helps solve this problem?" If YES → Partial. The student demonstrated real mathematical understanding, even if the final proof is incomplete.
- INCORRECT: If NO → Incorrect. The student is just playing with symbols or stating obvious facts without real progress.

**Almost vs Partial**:
- ALMOST: The proof structure is essentially complete. The main logical flow is there.
- PARTIAL: Major sections are missing. The student found something useful but there's still substantial work to do.

**When in doubt**: Choose the LOWER classification (more conservative). Almost is rare - most incomplete solutions are Partial, not Almost.

=== DECISION PROCESS ===

1. First, check if the solution is Correct: Is it a complete, rigorous proof with ALL claims proven and ALL cases handled? If YES → Correct.

2. If not Correct, check if it's Almost: Is the proof structure essentially complete with only minor issues? If YES → Almost.

3. If not Almost, check if it's Partial: Did the student find a KEY INSIGHT that advances the solution? Is there SUBSTANTIAL non-trivial progress? If YES → Partial.

4. Otherwise → Incorrect.

=== OUTPUT FORMAT (STRICT) ===

Output ONLY a JSON block in this exact format:

<json>
{{
    "response": "LABEL: Your brief reasoning here"
}}
</json>

Where LABEL is exactly one of: Correct, Almost, Partial, Incorrect

CRITICAL REQUIREMENTS:
- Start with the exact label followed by a colon
- Wrap in <json>...</json> tags
- NO text outside the JSON block
- Be concise but mention the key insight if Partial

EXAMPLES:
<json>
{{
    "response": "Correct: Complete rigorous proof with all cases handled."
}}
</json>

<json>
{{
    "response": "Almost: Complete proof with only a minor calculation error in Case 2."
}}
</json>

<json>
{{
    "response": "Partial: Found the key invariant (sum of squares) and proved it preserves parity, but didn't complete the induction."
}}
</json>

<json>
{{
    "response": "Incorrect: Only restates the problem and makes trivial observations without any real progress toward the solution."
}}
</json>"""

        msg_history, _ = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
            temperature=0.0,
        )

        # Extract prediction from JSON
        prediction = "None"
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted and "response" in extracted[-1]:
                prediction = extracted[-1]["response"]
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
