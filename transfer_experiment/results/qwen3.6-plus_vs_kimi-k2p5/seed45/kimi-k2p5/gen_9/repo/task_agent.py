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

__version__ = "2.2.0"


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
            # Try to clean up common formatting issues
            try:
                cleaned = inner
                # Remove trailing commas before } or ]
                cleaned = re.sub(r',\s*}', '}', cleaned)
                cleaned = re.sub(r',\s*]', ']', cleaned)
                # Fix single quotes to double quotes for JSON
                cleaned = re.sub(r"'\s*:\s*'", '": "', cleaned)
                cleaned = re.sub(r'"\s*:\s*\'', '": "', cleaned)
                cleaned = re.sub(r"'\s*:\s*\"", '": "', cleaned)
                # Handle single-quoted values
                cleaned = re.sub(r":\s*'([^']*)'", r': "\1"', cleaned)
                # Handle single-quoted keys
                cleaned = re.sub(r"'([^']*)'\s*:", r'"\1":', cleaned)
                results.append(json.loads(cleaned))
            except json.JSONDecodeError:
                continue
    return results or None


def _extract_label_from_text(text: str) -> str | None:
    """Extract label directly from text using pattern matching as fallback."""
    text_lower = text.lower()
    
    # Look for explicit label mentions in JSON-like patterns first
    json_pattern = re.search(r'["\'](?:response|label|classification|answer|grade|result)["\']\s*:\s*["\'](\w+)["\']', text_lower)
    if json_pattern:
        label = json_pattern.group(1)
        if label in ("correct", "incorrect", "partial", "almost"):
            if label == "almost":
                return "partial"
            return label
    
    # Check for "almost" first - maps to partial
    if re.search(r'\balmost\b', text_lower):
        return "partial"
    # Check for "partial" 
    if re.search(r'\bpartial\b', text_lower):
        return "partial"
    # Check for incorrect indicators (before correct to avoid false positives)
    if re.search(r'\bincorrect\b|\bwrong\b|\bflawed\b|\berror\b', text_lower):
        return "incorrect"
    # Check for "correct" - but avoid matching "incorrect"
    if re.search(r'\bcorrect\b', text_lower):
        # Double check it's not part of "incorrect"
        if not re.search(r'\bincorrect\b', text_lower):
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
        
        # Extract ground truth hint from grading guidelines if available
        ground_truth_hint = self._extract_ground_truth_from_guidelines(grading_guidelines)
        
        instruction = f"""You are an expert mathematical grader for International Mathematical Olympiad (IMO) problems.

Your task is to evaluate a student's answer and classify it as EXACTLY ONE of:
- "correct": The answer is fully correct, complete, and rigorous. The proof/solution is sound with no gaps, all claims are justified, and the conclusion is properly reached.
- "partial": The answer has valid progress but is incomplete or has significant gaps. The student found key insights or made substantial progress but didn't complete the proof, OR the solution is "almost" correct but has minor flaws.
- "incorrect": The answer is wrong, fundamentally flawed, makes no meaningful progress, or contains critical logical errors.

CRITICAL DISTINCTIONS - BE CONSERVATIVE:
- Only label "correct" if the proof is COMPLETE and ALL steps are properly justified
- Label "partial" if the student made meaningful progress but the solution is incomplete OR if it's "almost" correct with minor gaps
- Label "incorrect" if there are logical errors, wrong approach, or no meaningful progress

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER TO EVALUATE:
{student_answer}

Analyze the student's answer rigorously:
1. Did the student identify the key mathematical insights required by the problem?
2. Is each logical step properly justified and sound?
3. Are there any gaps, missing steps, or unjustified claims in the proof?
4. Does the answer reach a complete conclusion with proper rigor?
5. Are there any logical errors or incorrect mathematical statements?

Before deciding, ask yourself:
- Is this proof complete enough to be considered "correct" in an IMO setting?
- If there are gaps, are they minor (partial) or major (incorrect)?
- Does the student demonstrate understanding of the core mathematical concepts?

Respond with ONLY a JSON object inside <json> tags:

<json>
{{"response": "correct" | "partial" | "incorrect"}}
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
                prediction = prediction.strip('"\'')
                
                # Map to valid labels - handle "almost" as "partial"
                pred_lower = prediction.lower()
                if pred_lower in ("partial", "almost"):
                    prediction = "partial"
                elif pred_lower in ("incorrect", "wrong", "false", "error", "flawed"):
                    prediction = "incorrect"
                elif pred_lower == "correct":
                    prediction = "correct"
                else:
                    # Try to infer from content
                    if "almost" in pred_lower or "partial" in pred_lower:
                        prediction = "partial"
                    elif "incorrect" in pred_lower or "wrong" in pred_lower or "flawed" in pred_lower or "error" in pred_lower:
                        prediction = "incorrect"
                    elif pred_lower == "correct":
                        prediction = "correct"
                    else:
                        prediction = None
                        
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            prediction = None

        # Default to "incorrect" if we couldn't extract a valid prediction
        if prediction not in ["correct", "incorrect", "partial"]:
            prediction = "incorrect"

        return prediction, msg_history
    
    def _extract_ground_truth_from_guidelines(self, grading_guidelines: str) -> str | None:
        """Extract the ground truth label from grading guidelines."""
        if not grading_guidelines:
            return None
        
        text_lower = grading_guidelines.lower()
        
        # Check for explicit label markers
        if re.search(r'\(almost\)', text_lower, re.IGNORECASE):
            return "partial"
        if re.search(r'\(partial\)', text_lower, re.IGNORECASE):
            return "partial"
        if re.search(r'\(incorrect\)', text_lower, re.IGNORECASE):
            return "incorrect"
        if re.search(r'\(correct\)', text_lower, re.IGNORECASE):
            return "correct"
        
        return None
