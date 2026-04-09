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


def _extract_json_flexible(text: str) -> dict | None:
    """Extract JSON using multiple fallback methods.
    
    Tries: <json> tags, ```json blocks, raw JSON objects.
    """
    # Try <json> tags first
    results = _extract_jsons(text)
    if results:
        return results[-1]
    
    # Try ```json code blocks
    json_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(json_block_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue
    
    # Try finding raw JSON objects with curly braces
    # Look for outermost braces
    brace_count = 0
    start_idx = -1
    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                try:
                    return json.loads(text[start_idx:i+1])
                except json.JSONDecodeError:
                    continue
    
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
        # Extract key fields for better prompting
        domain = inputs.get('domain', 'Mathematics')
        problem = inputs.get('problem', '')
        solution = inputs.get('solution', '')
        grading_guidelines = inputs.get('grading_guidelines', '')
        student_answer = inputs.get('student_answer', '')
        
        instruction = f"""You are an expert grader for {domain} problems.

Your task is to evaluate a student's answer to a problem and assign a grade.

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

Based on the problem, official solution, and grading guidelines above, evaluate the student's answer.

GRADE CATEGORIES - Use EXACTLY one of these four labels:

1. "Correct" - The student's answer is completely correct. The proof/solution is valid, all claims are justified, and the conclusion matches the official solution. Minor presentation issues are acceptable. The answer demonstrates full understanding of the problem.

2. "Incorrect" - The student's answer is fundamentally wrong. The proof has critical flaws, key claims are unjustified or false, the approach is invalid, or the conclusion is wrong. Major gaps in reasoning exist. The student made only trivial progress or took a completely wrong approach. ONLY use this if the answer has NO substantial correct mathematical content toward the solution.

3. "Partial" - The student made MEANINGFUL progress but the solution is incomplete. The student found a key invariant, proved a significant lemma, identified the right approach, or made substantial progress toward the solution. There is GENUINE mathematical substance - the student understood important aspects of the problem but didn't complete the proof. This is for answers with REAL progress, not just attempts.

4. "Almost" - The student's solution is NEARLY COMPLETE with only MINOR mistakes. Small errors in calculation, missing a trivial case, or minor gaps that could be easily fixed. The core proof structure is sound and the solution would be "Correct" with minimal fixes. The student essentially solved the problem but made small slips.

CRITICAL DECISION RULES:
- "Incorrect" = No substantial correct mathematical progress (just wrong attempts or trivial observations)
- "Partial" = Real mathematical progress (key insights, significant lemmas, correct approach started but not finished)
- "Almost" = Solution essentially complete, just minor errors
- "Correct" = Fully correct solution

When deciding between "Incorrect" and "Partial":
- If the student found a KEY INSIGHT or INVARIANT → "Partial" (not Incorrect)
- If the student proved a SIGNIFICANT LEMMA → "Partial" (not Incorrect)
- If the student only made TRIVIAL OBSERVATIONS or went down a WRONG PATH → "Incorrect"
- If there's GENUINE MATHEMATICAL SUBSTANCE → "Partial"

When deciding between "Partial" and "Almost":
- If the solution is INCOMPLETE with major pieces missing → "Partial"
- If the solution is COMPLETE except for minor fixes → "Almost"

EXAMPLES OF CATEGORY DISTINCTIONS:

Example 1 - "Partial" (not Incorrect): 
Student found a key invariant (mod 4) and proved necessity, but didn't prove sufficiency.
→ This is "Partial" because finding the invariant is meaningful mathematical progress.

Example 2 - "Incorrect":
Student made some calculations but the approach is fundamentally flawed with no key insights.
→ This is "Incorrect" because there's no substantial correct mathematical content.

Example 3 - "Almost" (not Partial):
Student completed the entire proof but made a small calculation error in one step.
→ This is "Almost" because the solution structure is complete, just minor errors.

Example 4 - "Partial" (not Almost):
Student proved a significant lemma and set up the right framework but didn't finish the main proof.
→ This is "Partial" because major pieces are still missing.

You must respond with a JSON object containing your evaluation in the "response" field.
The response MUST be exactly one of: "Correct", "Incorrect", "Partial", or "Almost".

IMPORTANT: Your response MUST be wrapped in <json> tags like this:
<json>
{{
    "response": "Correct"}}
</json>

Provide only the JSON response, nothing else."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON using flexible extraction
        prediction = "None"
        try:
            last_message = msg_history[-1]["text"]
            extracted = _extract_json_flexible(last_message)
            if extracted and "response" in extracted:
                prediction = extracted["response"]
                self.log_fn(f"Successfully extracted prediction: {prediction}")
            else:
                self.log_fn(f"No 'response' field found in extracted JSON: {extracted}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
