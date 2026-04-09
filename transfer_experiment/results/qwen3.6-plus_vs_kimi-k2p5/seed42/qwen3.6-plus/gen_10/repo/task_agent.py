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


def _extract_json_flexible(text: str) -> list[dict] | None:
    """Extract JSON objects using multiple strategies.

    Tries multiple patterns:
    1. <json>...</json> tags
    2. ```json...``` code blocks
    3. Raw JSON objects with "response" field
    4. JSON-like patterns with flexible whitespace
    """
    results = []

    # Strategy 1: <json>...</json> tags
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

    # Strategy 2: ```json...``` code blocks
    pattern = r'```json\s*(.*?)\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            continue

    # Strategy 3: ```...``` code blocks (without json label)
    pattern = r'```\s*(\{.*?\})\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            continue

    # Strategy 4: Raw JSON objects with "response" field
    # Look for JSON-like structures: {"response": "..."}
    pattern = r'\{\s*"response"\s*:\s*"([^"]+)"\s*\}'
    matches = re.findall(pattern, text)
    for match in matches:
        results.append({"response": match})

    # Strategy 5: Case-insensitive response extraction
    # Look for patterns like "response": "correct" or 'response': 'partial'
    pattern = r'["\']?response["\']?\s*:\s*["\']?(correct|incorrect|partial|almost)["\']?'
    matches = re.findall(pattern, text, re.IGNORECASE)
    for match in matches:
        results.append({"response": match.lower()})

    return results or None


def _extract_response_direct(text: str) -> str | None:
    """Direct extraction of response value from text.

    Looks for explicit mentions of the grade in the text, focusing on
    the final answer/conclusion rather than quoted text from the prompt.
    """
    text_lower = text.lower()

    # First, try to find the JSON block and extract from there
    # This is the most reliable source
    json_match = re.search(r'<json>\s*\{\s*"response"\s*:\s*"([^"]+)"\s*\}\s*</json>', text, re.IGNORECASE)
    if json_match:
        val = json_match.group(1).lower()
        if val in ['correct', 'incorrect', 'partial', 'almost']:
            return val

    # Try code block JSON
    code_match = re.search(r'```(?:json)?\s*\{\s*"response"\s*:\s*"([^"]+)"\s*\}\s*```', text, re.IGNORECASE | re.DOTALL)
    if code_match:
        val = code_match.group(1).lower()
        if val in ['correct', 'incorrect', 'partial', 'almost']:
            return val

    # Look for the grade in the LAST part of the response (after reasoning)
    # Split by common conclusion markers
    conclusion_markers = ['therefore', 'thus', 'hence', 'in conclusion', 'final answer', 'final grade', 'my grade', 'my decision', 'verdict', 'grade:', 'decision:']
    
    # Find the last occurrence of any conclusion marker
    last_conclusion_start = 0
    for marker in conclusion_markers:
        idx = text_lower.rfind(marker)
        if idx > last_conclusion_start:
            last_conclusion_start = idx
    
    # Search for grade in the conclusion portion (or last 500 chars if no marker found)
    if last_conclusion_start > 0:
        search_text = text_lower[last_conclusion_start:]
    else:
        search_text = text_lower[-500:]
    
    # Look for grade patterns in the conclusion
    for grade in ['correct', 'incorrect', 'partial', 'almost']:
        # Pattern: grade in quotes
        if f'"{grade}"' in search_text or f"'{grade}'" in search_text:
            return grade
        # Pattern: grade: <word> or decision: <word>
        pattern = rf'(?:grade|decision|verdict|answer)\s*:\s*{grade}'
        if re.search(pattern, search_text):
            return grade
        # Pattern: is <grade> or is "<grade>"
        pattern = rf'\bis\s+(?:")?{grade}(?:")?'
        if re.search(pattern, search_text):
            return grade

    # Fallback: look for grade keywords in the last few lines
    lines = text_lower.strip().split('\n')
    for line in reversed(lines[-10:]):  # Check last 10 lines
        line = line.strip()
        if line in ['correct', 'incorrect', 'partial', 'almost']:
            return line
        for grade in ['correct', 'incorrect', 'partial', 'almost']:
            if re.search(rf'\b{grade}\b', line) and any(kw in line for kw in ['grade', 'decision', 'verdict', 'answer', 'final']):
                return grade

    return None


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.log_file = log_file
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            logger.addHandler(file_handler)

    def __repr__(self) -> str:
        return f"TaskAgent(model={self.model!r}, log_file={self.log_file!r})"

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single grading problem.

        Args:
            inputs: dict with keys 'problem', 'solution', 'grading_guidelines',
                    and 'student_answer'.

        Returns:
            A tuple of (prediction, msg_history) where prediction is one of
            'correct', 'incorrect', or 'partial'.
        """
        self.log_fn(f"Starting task agent with model: {self.model}")

        # Extract fields from inputs for clearer prompting
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematics grader for International Mathematical Olympiad (IMO) problems.

Your task is to evaluate a student's answer to a mathematics problem and assign one of four grades.

IMO problems are scored on a scale of 0 to 7 points. Your four grades correspond to:
- "correct" (7 points): The student's answer is COMPLETELY correct. The proof is rigorous, all key steps are present, all cases are handled, and there are no logical gaps or errors. The solution would receive full marks from an IMO grader.
- "almost" (6 points): The student's solution is nearly complete with only minor errors or small missing pieces. The core idea is correct and the solution is almost there — it needs only small corrections or a minor case analysis to be complete.
- "partial" (1 point): The student has made SOME meaningful progress but is far from a complete solution. This includes: proving useful lemmas, making correct observations, setting up the right framework, or having a mostly correct approach with significant gaps. The student demonstrates genuine understanding but does not come close to completing the proof.
- "incorrect" (0 points): The student's answer is fundamentally wrong, contains critical logical errors, uses invalid methods, or fails to make ANY meaningful progress toward solving the problem. The solution does not establish any useful intermediate result.

CRITICAL grading principles:
1. Be STRICT. IMO grading is rigorous. A solution with a major gap, missing case, or logical error is NOT "correct".
2. "almost" requires the solution to be nearly complete — the main idea is fully developed and only minor details are missing.
3. "partial" requires the student to have established at least one useful intermediate result (a lemma, a key observation, a correct setup). Merely restating the problem or making trivial observations is NOT partial — it is "incorrect".
4. "incorrect" means the solution has NOT made meaningful progress. If the student's approach is fundamentally flawed, or if they only restate the problem without any substantive work, grade it "incorrect".
5. When in doubt between two adjacent grades, choose the LOWER grade. IMO graders are conservative.

IMPORTANT: Do NOT give "correct" unless the solution is truly complete and rigorous. Do NOT give "partial" unless the student has proven at least one non-trivial intermediate result.

Problem:
{problem}

Official Solution:
{solution}

Grading Guidelines (these specify what earns partial/almost credit):
{grading_guidelines}

Student's Answer:
{student_answer}

First, think through your reasoning step by step. Use the following structured analysis:

**Step 1: Key steps in the official solution**
List the key steps/lemmas required to fully solve the problem.

**Step 2: Check against Partial criteria**
The Grading Guidelines specify what earns "(Partial)" credit. For each criterion listed under "(Partial)", determine whether the student's answer satisfies it. Quote the relevant part of the student's answer if it does, or explain why it doesn't. If the student satisfies NONE of the Partial criteria, the grade must be "incorrect".

**Step 3: Check against Almost criteria**
The Grading Guidelines specify what earns "(Almost)" credit. For each criterion listed under "(Almost)", determine whether the student's answer satisfies it. If the student does not satisfy the Partial criteria, they cannot satisfy the Almost criteria.

**Step 4: Check for completeness**
Is the student's solution complete and rigorous with no gaps? If yes, the grade is "correct". Is it nearly complete with only minor issues? If yes, the grade is "almost".

**Step 5: Self-verification**
Before finalizing your grade, ask yourself:
- If I gave "correct": Is there ANY gap, missing case, or unjustified claim? If yes, downgrade.
- If I gave "almost": Is the solution truly nearly complete, or are there major gaps? If major gaps exist, downgrade.
- If I gave "partial": Can I point to a SPECIFIC criterion from the "(Partial)" section that the student satisfies? If not, downgrade to "incorrect".
- If I gave "incorrect": Have I confirmed that the student satisfies NONE of the "(Partial)" criteria? If they do satisfy one, upgrade to "partial".

**Step 6: Final decision**
Based on the analysis and self-verification above, assign the grade. Remember:
- Be STRICT. When in doubt, choose the LOWER grade.
- "correct" requires a COMPLETE, rigorous proof.
- "almost" requires a nearly complete solution.
- "partial" requires at least one criterion from the "(Partial)" section to be satisfied.
- "incorrect" means no meaningful progress — none of the Partial criteria are met.

Then, respond in JSON format with the following schema:
<json>
{{
    "response": "correct" | "almost" | "partial" | "incorrect"
}}
</json>

Provide your step-by-step reasoning first, followed by the JSON response with your grading decision."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from response using multiple strategies
        prediction = "None"
        response_text = ""

        # Get the response text from msg_history or use the direct response
        try:
            if msg_history and len(msg_history) > 0:
                if isinstance(msg_history[-1], dict) and "text" in msg_history[-1]:
                    response_text = msg_history[-1]["text"]
                elif isinstance(msg_history[-1], str):
                    response_text = msg_history[-1]
            else:
                response_text = response if isinstance(response, str) else str(response)
        except Exception as e:
            self.log_fn(f"Error getting response text: {e}")
            response_text = str(response) if response else ""

        self.log_fn(f"Raw response text: {response_text[:500]}...")

        # Try multiple extraction strategies
        try:
            # Strategy 1: Flexible JSON extraction
            extracted = _extract_json_flexible(response_text)
            if extracted:
                for item in extracted:
                    if isinstance(item, dict) and "response" in item:
                        val = item["response"]
                        if isinstance(val, str) and val.lower() in ["correct", "incorrect", "partial", "almost"]:
                            prediction = val.lower()
                            self.log_fn(f"Extracted prediction via JSON: {prediction}")
                            break
        except Exception as e:
            self.log_fn(f"Error in flexible JSON extraction: {e}")

        # Strategy 2: Direct response extraction if JSON failed
        if prediction == "None":
            try:
                direct = _extract_response_direct(response_text)
                if direct:
                    prediction = direct
                    self.log_fn(f"Extracted prediction via direct extraction: {prediction}")
            except Exception as e:
                self.log_fn(f"Error in direct extraction: {e}")

        # Strategy 3: Legacy extraction as fallback
        if prediction == "None":
            try:
                extracted = _extract_jsons(response_text)
                if extracted and len(extracted) > 0:
                    last = extracted[-1]
                    if isinstance(last, dict) and "response" in last:
                        val = last["response"]
                        if val in ["correct", "incorrect", "partial", "almost"]:
                            prediction = val
                            self.log_fn(f"Extracted prediction via legacy JSON: {prediction}")
            except Exception as e:
                self.log_fn(f"Error in legacy extraction: {e}")

        # Validate the prediction
        if prediction not in ["correct", "incorrect", "partial", "almost"]:
            self.log_fn(f"Invalid or missing prediction: '{prediction}', defaulting to incorrect")
            prediction = "incorrect"

        return str(prediction), msg_history
