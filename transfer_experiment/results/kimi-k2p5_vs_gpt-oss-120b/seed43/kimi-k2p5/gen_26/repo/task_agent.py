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
    Also handles markdown code blocks and plain JSON.
    """
    results = []
    
    # Try <json> tags first
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
    
    # Try markdown code blocks
    if not results:
        code_blocks = re.findall(r'```(?:json)?\s*\n?(.*?)\n?```', text, flags=re.DOTALL | re.IGNORECASE)
        for block in code_blocks:
            try:
                results.append(json.loads(block.strip()))
            except json.JSONDecodeError:
                # Try to fix common JSON issues
                try:
                    fixed = re.sub(r',(\s*[}\]])', r'\1', block.strip())
                    results.append(json.loads(fixed))
                except json.JSONDecodeError:
                    continue
    
    # If no blocks found, try to find any JSON object using a regex fallback
    if not results:
        json_candidates = re.findall(r'\{[^{}]*\}', text, flags=re.DOTALL)
        for cand in json_candidates:
            try:
                results.append(json.loads(cand))
            except json.JSONDecodeError:
                continue
    
    return results or None


def _validate_prediction(prediction: str) -> str:
    """Validate and normalize the prediction to one of the four valid categories."""
    if not prediction:
        return "None"
    
    # Normalize the prediction
    pred = prediction.strip()
    
    # Check for exact matches (case-insensitive)
    valid_labels = ["Correct", "Almost", "Partial", "Incorrect"]
    for label in valid_labels:
        if pred.lower() == label.lower():
            return label
    
    # Check if prediction starts with a valid label followed by colon or space
    for label in valid_labels:
        if pred.lower().startswith(label.lower() + ":") or \
           pred.lower().startswith(label.lower() + " "):
            return label
    
    # Check for common misspellings or variations
    if "correct" in pred.lower() and "almost" not in pred.lower() and "partial" not in pred.lower() and "incorrect" not in pred.lower():
        return "Correct"
    if "almost" in pred.lower():
        return "Almost"
    if "partial" in pred.lower():
        return "Partial"
    if "incorrect" in pred.lower() or "wrong" in pred.lower():
        return "Incorrect"
    
    return "None"


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
- PARTIAL: The student made SUBSTANTIAL NON-TRIVIAL PROGRESS toward the solution. They found a KEY INSIGHT, useful lemma, or significant partial result that genuinely advances toward solving the problem. Even if incomplete, the work shows real mathematical understanding.
- INCORRECT: The student made NO MEANINGFUL PROGRESS. They only restate the problem, make trivial observations, or go down completely wrong paths. There is no insight that would help solve the problem.

KEY TEST for Partial vs Incorrect: If you removed the incomplete parts, would what's left be publishable as a useful lemma or partial result? If YES → Partial. If NO → Incorrect.

**Almost vs Partial**:
- ALMOST: The proof structure is essentially complete (90-95% done). Only minor issues remain (small calculation errors, missing trivial cases). The main logical flow is correct and complete.
- PARTIAL: Major sections are missing (50-70% complete). The student found something useful but substantial work remains.

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
</json>

IMPORTANT: The label must be exactly one of the four words: Correct, Almost, Partial, or Incorrect. Do not use variations like "correct" (lowercase), "Almost correct", "Partially correct", "Mostly incorrect", etc."""

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
                raw_prediction = extracted[-1]["response"]
                prediction = _validate_prediction(raw_prediction)
                if prediction == "None":
                    self.log_fn(f"Could not validate prediction: {raw_prediction}")
            else:
                # Try to extract from the raw text if JSON parsing failed
                raw_text = msg_history[-1]["text"]
                prediction = _validate_prediction(raw_text)
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Try to extract from raw text as fallback
            try:
                raw_text = msg_history[-1]["text"]
                prediction = _validate_prediction(raw_text)
            except:
                pass

        return str(prediction), msg_history
