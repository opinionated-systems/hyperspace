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
    Also handles markdown code blocks with json.
    """
    results = []
    search_from = 0
    
    # First, try to find <json>...</json> blocks
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
            # Try to clean up common JSON issues
            try:
                # Remove trailing commas before closing braces/brackets
                cleaned = re.sub(r',(\s*[}\]])', r'\1', inner)
                results.append(json.loads(cleaned))
            except json.JSONDecodeError:
                continue
    
    # Also try to find markdown json blocks ```json...```
    if not results:
        search_from = 0
        while True:
            start = text.find("```json", search_from)
            if start == -1:
                break
            end = text.find("```", start + 7)
            if end == -1:
                break
            inner = text[start + 7:end].strip()
            search_from = end + 3
            try:
                results.append(json.loads(inner))
            except json.JSONDecodeError:
                # Try to clean up common JSON issues
                try:
                    cleaned = re.sub(r',(\s*[}\]])', r'\1', inner)
                    results.append(json.loads(cleaned))
                except json.JSONDecodeError:
                    continue
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _build_grading_prompt(self, inputs: dict) -> str:
        """Build a structured prompt for IMO grading with chain-of-thought reasoning."""
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        return f"""You are an expert mathematical grader for the International Mathematical Olympiad (IMO).

Your task is to evaluate a student's solution to a mathematical problem and assign a numeric grade.

GRADE DEFINITIONS (apply strictly, err on the side of lower grades):
- 0 = Incorrect: contains fatal logical errors, false claims, or circular reasoning. No mathematically valid progress toward the solution.
- 1 = Partial: contains genuine, mathematically valid progress (proves a useful lemma, handles critical special cases, or sets up correct framework) but fails to complete the main proof due to major gaps.
- 2 = Almost: the main proof strategy and all key ideas are correct and valid. Only minor gaps, omitted trivial details, or small non-critical errors prevent full rigor. Would receive 6/7 points in IMO.
- 3 = Correct: complete, rigorous, and correct. All non-trivial claims justified. Would receive full marks (7/7) in IMO.

IMPORTANT: Be conservative. Most solutions have errors. Grade 3 requires every step justified with no gaps.

Follow these steps carefully:

1. **Understand the Problem**: Read the problem statement carefully and identify what is being asked.

2. **Review the Official Solution**: Study the provided official solution to understand the expected approach and key insights.

3. **Analyze the Student's Answer**: Examine the student's solution step by step, checking:
   - Mathematical correctness of each step
   - Logical soundness and justification
   - Completeness (does it address all parts of the problem?)
   - Clarity of presentation

4. **Apply Grading Guidelines**: Use the provided grading guidelines to assess the student's work objectively.

5. **Provide Your Evaluation**: Give a clear, definitive assessment with a numeric grade (0, 1, 2, or 3).

---

**Domain**: {domain}

**Problem**:
```
{problem}
```

**Official Solution**:
```
{solution}
```

**Grading Guidelines**:
```
{grading_guidelines}
```

**Student's Answer**:
```
{student_answer}
```

---

**Your Task**:

First, think through your evaluation step by step (chain-of-thought reasoning). Consider:
- Does the student's approach align with the official solution?
- Are the mathematical steps correct?
- Is the logic sound and well-justified?
- Does the student address all parts of the problem?
- What numeric grade (0-3) would you assign based on the grading guidelines?

Then, provide your final assessment in the following JSON format. Be precise and concise:

<json>
{{
    "reasoning": "Your detailed step-by-step evaluation and reasoning process",
    "grade": 0,
    "explanation": "Brief explanation of why this grade was assigned"
}}
</json>

Important: 
- Ensure your response is valid JSON without trailing commas.
- The "grade" field MUST be an integer: 0, 1, 2, or 3 (see grade definitions above).
- Be conservative in grading - most solutions have some errors."""

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_grading_prompt(inputs)

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with improved error handling
        prediction = "None"
        try:
            if msg_history and len(msg_history) > 0:
                # Handle both "text" (paper format) and "content" (OpenAI format) fields
                last_message = msg_history[-1].get("text") or msg_history[-1].get("content", "")
                extracted = _extract_jsons(last_message)
                if extracted:
                    last_json = extracted[-1]
                    # Try grade field first (numeric grade 0-3)
                    if "grade" in last_json:
                        grade_val = last_json["grade"]
                        # Handle both int and string grades
                        if isinstance(grade_val, int) and 0 <= grade_val <= 3:
                            prediction = str(grade_val)
                        elif isinstance(grade_val, str):
                            grade_str = grade_val.strip()
                            if grade_str in ("0", "1", "2", "3"):
                                prediction = grade_str
                    # Fallback: try assessment field (categorical label)
                    elif "assessment" in last_json:
                        assessment = last_json["assessment"]
                        prediction = self._map_assessment_to_grade(assessment)
                    # Fallback: try response field
                    elif "response" in last_json:
                        response = last_json["response"]
                        prediction = self._extract_grade_from_text(str(response))
                else:
                    # If no JSON found, try to extract grade from text
                    prediction = self._extract_grade_from_text(last_message)
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history

    def _map_assessment_to_grade(self, assessment: str) -> str:
        """Map categorical assessment to numeric grade."""
        assessment_lower = str(assessment).lower().strip()
        if "incorrect" in assessment_lower:
            return "0"
        elif "partial" in assessment_lower:
            return "1"
        elif "almost" in assessment_lower:
            return "2"
        elif "correct" in assessment_lower and "incorrect" not in assessment_lower:
            return "3"
        return str(assessment)

    def _extract_grade_from_text(self, text: str) -> str:
        """Extract numeric grade 0-3 from text using multiple strategies."""
        import re
        
        if not text or not text.strip():
            return "None"
        
        text_lower = text.lower()
        
        # Priority 1: Look for "grade": X or "grade": "X" pattern
        grade_matches = list(re.finditer(r'["\']?grade["\']?\s*[:=]\s*["\']?([0-3])["\']?', text_lower))
        if grade_matches:
            return grade_matches[-1].group(1)
        
        # Priority 2: Look for "Grade: X" or "grade is X" pattern
        matches = list(re.finditer(r'[Gg]rade[\s:]+([0-3])\b', text))
        if matches:
            return matches[-1].group(1)
        
        # Priority 3: Keyword match - find the LAST grade keyword in the output
        last_pos = -1
        last_grade = None
        for name, grade in [("incorrect", 0), ("partial", 1), ("almost", 2)]:
            pos = text_lower.rfind(name)
            if pos > last_pos:
                last_pos = pos
                last_grade = grade
        # "correct" must not match "incorrect"
        for m in re.finditer(r'(?<!in)correct', text_lower):
            if m.start() > last_pos:
                last_pos = m.start()
                last_grade = 3
        if last_grade is not None:
            return str(last_grade)
        
        # Priority 4: Last line is a bare digit 0-3
        lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
        if lines:
            last = lines[-1].strip("*_`#$\\boxed{}() .:").strip()
            if last in ("0", "1", "2", "3"):
                return last
        
        # Priority 5: Last digit 0-3 on the last line (but NOT negative numbers)
        if lines:
            m_all = list(re.finditer(r'(?<!\d)(?<!-)([0-3])(?!\d)', lines[-1]))
            if m_all:
                return m_all[-1].group(1)
        
        return "None"
