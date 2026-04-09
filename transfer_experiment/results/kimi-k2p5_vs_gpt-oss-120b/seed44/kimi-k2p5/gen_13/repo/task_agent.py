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
    """Extract JSON objects from <json>...</json> blocks (case-insensitive)."""
    results = []
    text_lower = text.lower()
    search_from = 0
    while True:
        start = text_lower.find("<json>", search_from)
        if start == -1:
            break
        end = text_lower.find("</json>", start)
        if end == -1:
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        # Try multiple parsing strategies
        parsed = None
        
        # Strategy 1: Direct parse
        try:
            parsed = json.loads(inner)
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Remove markdown code block markers
        if parsed is None:
            try:
                cleaned = inner.replace("```json", "").replace("```", "").strip()
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                pass
        
        # Strategy 3: Handle single quotes instead of double quotes
        if parsed is None:
            try:
                cleaned = inner.replace("'", '"').strip()
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                pass
        
        # Strategy 4: Extract just the response field using regex
        if parsed is None:
            try:
                match = re.search(r'["\']response["\']\s*:\s*["\']([^"\']+)["\']', inner)
                if match:
                    val = match.group(1)
                    if val in {"correct", "almost", "partial", "incorrect"}:
                        parsed = {"response": val}
            except Exception:
                pass
        
        if parsed is not None and isinstance(parsed, dict) and "response" in parsed:
            results.append(parsed)
            
    return results or None


def _extract_json_from_raw_text(text: str) -> dict | None:
    """Extract JSON object from raw text without <json> tags."""
    # Strategy 1: Try to find JSON object patterns directly with response field
    json_pattern = re.search(r'\{\s*["\']response["\']\s*:\s*["\']([^"\']+)["\']\s*\}', text, re.IGNORECASE)
    if json_pattern:
        try:
            return json.loads(json_pattern.group(0).replace("'", '"'))
        except json.JSONDecodeError:
            pass
    
    # Strategy 2: Try to find any JSON-like object with response field
    json_pattern = re.search(r'\{[^}]*["\']response["\'][^}]*\}', text, re.DOTALL | re.IGNORECASE)
    if json_pattern:
        try:
            parsed = json.loads(json_pattern.group(0).replace("'", '"'))
            if "response" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass
    
    # Strategy 3: Try to find JSON in markdown code blocks
    code_block_pattern = re.search(r'```(?:json)?\s*\n?([^`]+)```', text, re.DOTALL | re.IGNORECASE)
    if code_block_pattern:
        try:
            parsed = json.loads(code_block_pattern.group(1).strip())
            if "response" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass
    
    # Strategy 4: Look for response field with various quote styles
    response_match = re.search(r'response["\']?\s*[:=]\s*["\']?(correct|almost|partial|incorrect)["\']?', text, re.IGNORECASE)
    if response_match:
        val = response_match.group(1).lower()
        if val in {"correct", "almost", "partial", "incorrect"}:
            return {"response": val}
    
    return None


def _analyze_grading_guidelines(grading_guidelines: str) -> dict:
    """Analyze the grading guidelines to determine the expected grade.
    
    The grading guidelines have a specific structure with sections:
    - (Partial) section: Lists achievements that earn partial credit
    - (Almost) section: Describes what was "almost" achieved but had issues
    
    KEY RULE: If the (Almost) section exists and has content describing issues,
    the expected grade is "almost" (not "correct").
    """
    guidelines_lower = grading_guidelines.lower()
    
    # Check for Almost section and extract its content
    has_almost_section = "(almost)" in guidelines_lower
    almost_section_text = ""
    almost_has_content = False
    
    if has_almost_section:
        almost_start = guidelines_lower.find("(almost)")
        next_section = guidelines_lower.find("(", almost_start + 8)
        if next_section > almost_start:
            almost_section_text = guidelines_lower[almost_start:next_section]
        else:
            almost_section_text = guidelines_lower[almost_start:]
        # Check if Almost section has meaningful content
        content_after_header = almost_section_text.replace("(almost)", "").strip()
        almost_has_content = len(content_after_header) > 5
    
    # Check for Partial section
    has_partial_section = "(partial)" in guidelines_lower
    
    # Determine expected grade based on guideline structure
    expected_grade = None
    if has_almost_section and almost_has_content:
        expected_grade = "almost"
    elif has_partial_section:
        expected_grade = "partial"
    
    return {
        "has_almost_section": has_almost_section,
        "almost_has_content": almost_has_content,
        "has_partial_section": has_partial_section,
        "expected_grade_hint": expected_grade,
    }


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem."""
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Analyze guidelines for hint
        guideline_analysis = _analyze_grading_guidelines(grading_guidelines)
        
        # Build hint based on guideline structure
        grade_hint = ""
        if guideline_analysis["expected_grade_hint"] == "almost":
            grade_hint = '\nHINT: The grading guidelines contain an (Almost) section with specific issues. This strongly suggests the grade should be "almost".\n'
        elif guideline_analysis["expected_grade_hint"] == "partial":
            grade_hint = '\nHINT: The grading guidelines contain a (Partial) section but no (Almost) section. This suggests the grade should be "partial" or "incorrect".\n'
        
        instruction = f"""You are an expert mathematical grader. Evaluate the student's answer and output ONLY a JSON grade.

DOMAIN: {domain}

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}{grade_hint}

STUDENT'S ANSWER:
{student_answer}

---

GRADE DEFINITIONS:

1. "correct" - The solution is fully correct, complete, with no mistakes or gaps.
   Use ONLY if there is NO (Almost) section in the guidelines.

2. "almost" - The solution is nearly correct but has minor issues.
   Use when the (Almost) section exists and describes specific minor problems.
   This is the MOST LIKELY grade when (Almost) section has content.

3. "partial" - The solution has some correct elements but significant gaps remain.
   Use when there's NO (Almost) section but the student made some progress.

4. "incorrect" - The solution is wrong or shows no meaningful progress.
   Use when the student missed key insights from the (Partial) section.

DECISION RULES:
- If (Almost) section exists with content → grade is "almost"
- If NO (Almost) section but some progress → grade is "partial"  
- If NO (Almost) section and no progress → grade is "incorrect"
- If NO (Almost) section and fully correct → grade is "correct"

OUTPUT FORMAT (STRICT):
Respond with EXACTLY ONE of these, nothing else:
<json>{{"response": "correct"}}</json>
<json>{{"response": "almost"}}</json>
<json>{{"response": "partial"}}</json>
<json>{{"response": "incorrect"}}</json>"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction
        prediction = "None"
        response_text = msg_history[-1]["text"]
        
        # Try JSON tag extraction
        try:
            extracted = _extract_jsons(response_text)
            if extracted:
                for item in extracted:
                    if isinstance(item, dict) and "response" in item:
                        val = item["response"]
                        if val in {"correct", "almost", "partial", "incorrect"}:
                            prediction = val
                            break
        except Exception:
            pass
        
        # Try raw text extraction
        if prediction not in {"correct", "almost", "partial", "incorrect"}:
            try:
                raw_extracted = _extract_json_from_raw_text(response_text)
                if raw_extracted and "response" in raw_extracted:
                    val = raw_extracted["response"]
                    if val in {"correct", "almost", "partial", "incorrect"}:
                        prediction = val
            except Exception:
                pass
        
        # Text-based fallback
        if prediction not in {"correct", "almost", "partial", "incorrect"}:
            text = response_text.lower()
            
            # Look for response field patterns
            if '"response": "almost"' in text or "'response': 'almost'" in text:
                prediction = "almost"
            elif '"response": "partial"' in text or "'response': 'partial'" in text:
                prediction = "partial"
            elif '"response": "incorrect"' in text or "'response': 'incorrect'" in text:
                prediction = "incorrect"
            elif '"response": "correct"' in text or "'response': 'correct'" in text:
                prediction = "correct"
            # Look for quoted values
            elif '"almost"' in text or "'almost'" in text:
                prediction = "almost"
            elif '"partial"' in text or "'partial'" in text:
                prediction = "partial"
            elif '"incorrect"' in text or "'incorrect'" in text:
                prediction = "incorrect"
            elif '"correct"' in text or "'correct'" in text:
                prediction = "correct"
            # Final keyword fallback
            elif "almost" in text:
                prediction = "almost"
            elif "partial" in text:
                prediction = "partial"
            elif "incorrect" in text:
                prediction = "incorrect"
            elif "correct" in text:
                prediction = "correct"

        # Default to incorrect if invalid
        if prediction not in {"correct", "almost", "partial", "incorrect"}:
            prediction = "incorrect"

        return str(prediction), msg_history
