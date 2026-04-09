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
    """Extract JSON objects from <json>...</json> blocks or standalone JSON.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    """
    results = []
    search_from = 0
    
    # First try to find <json>...</json> blocks
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
            # Try to clean up common formatting issues
            try:
                # Remove trailing commas before closing braces
                cleaned = re.sub(r',\s*}', '}', inner)
                cleaned = re.sub(r',\s*]', ']', cleaned)
                results.append(json.loads(cleaned))
            except json.JSONDecodeError:
                continue
    
    # If no <json> blocks found, try to find standalone JSON objects
    if not results:
        # Look for JSON objects in the text
        json_pattern = r'\{[^{}]*"response"[^{}]*\}'
        matches = re.findall(json_pattern, text, re.IGNORECASE | re.DOTALL)
        for match in matches:
            try:
                results.append(json.loads(match))
            except json.JSONDecodeError:
                continue
    
    return results or None


def _extract_label_from_text(text: str) -> str | None:
    """Extract label directly from text using pattern matching as fallback."""
    text_lower = text.lower()
    
    # Look for explicit label mentions with word boundaries
    # Check for "almost" first - should be treated as partial credit
    if re.search(r'\balmost\b', text_lower):
        return "partial"
    # Check for "partial" 
    if re.search(r'\bpartial\b', text_lower):
        return "partial"
    # Check for "partially correct" or "partial credit" explicitly
    if re.search(r'partially\s+(correct|right)', text_lower) or re.search(r'partial\s+credit', text_lower):
        return "partial"
    # Check for incorrect indicators
    if re.search(r'\bincorrect\b', text_lower):
        return "incorrect"
    if re.search(r'\bwrong\b', text_lower):
        return "incorrect"
    if re.search(r'\berror\b', text_lower):
        return "incorrect"
    if re.search(r'\bflawed\b', text_lower):
        return "incorrect"
    if re.search(r'\binvalid\b', text_lower):
        return "incorrect"
    # Check for "correct" - be strict about this
    if re.search(r'\bcorrect\b', text_lower):
        # Ensure it's not in a context suggesting partial credit or negation
        if not re.search(r'partially\s+correct', text_lower) and \
           not re.search(r'not\s+correct', text_lower) and \
           not re.search(r'incorrect', text_lower):
            return "correct"
    
    return None


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
        # Extract fields from inputs for better prompting
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        instruction = f"""You are an expert mathematical grader for International Mathematical Olympiad (IMO) problems with extensive experience in evaluating competition mathematics solutions.

Your task is to evaluate a student's answer and classify it into exactly one of three categories:
- "correct": The answer is fully correct, complete, and rigorous. The proof/solution is logically sound with no gaps or errors, and arrives at the correct conclusion with proper justification. All cases are covered and the reasoning is complete.
- "incorrect": The answer is fundamentally flawed, contains critical errors, uses incorrect reasoning, or arrives at wrong conclusions. The approach is wrong or the solution is incomplete in ways that make it invalid. No meaningful progress toward the solution.
- "partial": The answer demonstrates meaningful progress toward the solution with valid insights and correct intermediate steps, but has significant gaps, missing cases, or lacks completeness in the final proof. The student shows understanding but hasn't fully solved the problem. This includes "almost" correct solutions with minor mistakes or incomplete proofs.

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER TO EVALUATE:
{student_answer}

Evaluation criteria:
1. Check if the student identified the correct approach and key mathematical insights
2. Verify logical soundness - are the deductions valid and properly justified?
3. Check for completeness - are all cases covered? Is the proof rigorous?
4. Look for computational errors, logical fallacies, or unjustified claims
5. Compare the student's reasoning quality against the official solution
6. "Partial" should be used when there's genuine mathematical progress but insufficient for full credit
7. Be especially careful with "almost" solutions - these are partial, not correct

Important: Be strict about correctness. Only label "correct" if the solution is truly complete and rigorous with no gaps. Label "partial" when there's clear evidence of understanding but incomplete execution, including "almost" correct solutions. Label "incorrect" for fundamentally wrong approaches or solutions with critical flaws.

You must respond with EXACTLY ONE of these three labels: "correct", "incorrect", or "partial".

Respond in JSON format with the following schema:
<json>
{{
    "response": "correct" | "incorrect" | "partial",
    "reasoning": "Brief explanation of why this label was chosen, referencing specific aspects of the student's answer"
}}
</json>"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = None
        try:
            # Get the last assistant message
            last_message = msg_history[-1]["text"] if msg_history else ""
            
            # Try JSON extraction first
            extracted = _extract_jsons(last_message)
            if extracted:
                for obj in extracted:
                    if "response" in obj:
                        prediction = str(obj["response"]).lower().strip()
                        break
            
            # If JSON extraction failed, try text extraction
            if prediction is None:
                prediction = _extract_label_from_text(last_message)
            
            # Normalize the prediction
            if prediction:
                # Clean up the prediction
                prediction = prediction.strip('"\'').strip()
                
                # Map to valid labels - be strict about matching
                prediction_lower = prediction.lower()
                
                # Check for "almost" first - should map to partial
                if "almost" in prediction_lower:
                    prediction = "partial"
                # Check for partial (but not "partially correct" which might be ambiguous)
                elif "partial" in prediction_lower:
                    prediction = "partial"
                # Check for incorrect indicators
                elif any(word in prediction_lower for word in ["incorrect", "wrong", "false", "error", "invalid", "flawed"]):
                    prediction = "incorrect"
                # Check for correct - be very strict
                elif prediction_lower == "correct" or prediction_lower.startswith("correct "):
                    # Only accept exact "correct" or "correct " at start
                    # Reject if contains negating words
                    if not any(neg in prediction_lower for neg in ["not correct", "incorrect", "partially", "almost"]):
                        prediction = "correct"
                    else:
                        prediction = None
                else:
                    # Try to infer from content more carefully
                    if prediction_lower.startswith("partial") or prediction_lower.startswith("almost"):
                        prediction = "partial"
                    elif prediction_lower.startswith("incorrect") or prediction_lower.startswith("wrong"):
                        prediction = "incorrect"
                    elif prediction_lower == "correct":
                        prediction = "correct"
                    else:
                        # Use text extraction as final fallback
                        prediction = _extract_label_from_text(prediction)
                        
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            prediction = None

        # Default to "incorrect" if we couldn't extract a valid prediction
        if prediction not in ["correct", "incorrect", "partial"]:
            prediction = "incorrect"

        return prediction, msg_history
