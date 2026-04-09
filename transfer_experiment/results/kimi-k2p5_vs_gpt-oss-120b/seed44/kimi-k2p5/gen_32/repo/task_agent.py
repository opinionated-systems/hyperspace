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
        
        # Build an improved instruction with clearer decision criteria
        instruction = f"""You are an expert mathematical grader for competition mathematics (IMO, USAMO, Putnam, etc.). Grade the student's solution using exactly one of four labels.

PROBLEM DOMAIN: {domain}

PROBLEM STATEMENT:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

=== GRADING INSTRUCTIONS ===

Compare the student's answer to the official solution. Assign exactly ONE grade:

**CORRECT** (7/7): Complete, rigorous proof. All steps valid, no gaps, no errors.

**ALMOST** (5-6/7): Nearly complete proof with ONLY minor issues:
- Small arithmetic/algebraic errors that don't affect the main argument
- Typos or notation inconsistencies
- Missing trivial edge cases that are obvious to fix
- The core proof structure is sound and would work with trivial fixes

**PARTIAL** (1-4/7): Has meaningful correct elements but significant issues:
- Valid approach but missing key steps or major gaps
- Shows genuine understanding but fails to complete the proof
- Has correct insights but also major errors
- Incomplete despite good start

**INCORRECT** (0/7): Little to no valid progress:
- Approach is fundamentally wrong
- No significant correct mathematics
- Nonsense, irrelevant, or shows zero understanding

=== DECISION RULES (Apply in order) ===

1. If NO valid mathematics → INCORRECT
2. If fundamentally wrong approach → INCORRECT  
3. If has correct elements but major gaps/errors → PARTIAL
4. If nearly complete with only trivial issues → ALMOST
5. If complete and fully correct → CORRECT

=== OUTPUT FORMAT ===
Respond with EXACTLY ONE JSON object:
<json>{{"response": "correct"}}</json>
OR
<json>{{"response": "almost"}}</json>
OR
<json>{{"response": "partial"}}</json>
OR
<json>{{"response": "incorrect"}}</json>

No other text."""

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
        
        # Text-based fallback - look for the grade words
        if prediction not in {"correct", "almost", "partial", "incorrect"}:
            text = response_text.lower()
            
            # Look for explicit grade mentions with context
            patterns = [
                (r'grade\s+is\s+["\']?incorrect["\']?', "incorrect"),
                (r'grade\s+is\s+["\']?partial["\']?', "partial"),
                (r'grade\s+is\s+["\']?almost["\']?', "almost"),
                (r'grade\s+is\s+["\']?correct["\']?', "correct"),
                (r'["\']response["\']?\s*:\s*["\']?incorrect["\']?', "incorrect"),
                (r'["\']response["\']?\s*:\s*["\']?partial["\']?', "partial"),
                (r'["\']response["\']?\s*:\s*["\']?almost["\']?', "almost"),
                (r'["\']response["\']?\s*:\s*["\']?correct["\']?', "correct"),
                (r'\bincorrect\b', "incorrect"),
                (r'\bpartial\b', "partial"),
                (r'\balmost\b', "almost"),
                (r'\bcorrect\b', "correct"),
            ]
            
            for pattern, label in patterns:
                if re.search(pattern, text):
                    prediction = label
                    break
        
        # Default to incorrect if still not valid
        if prediction not in {"correct", "almost", "partial", "incorrect"}:
            prediction = "incorrect"
        
        return str(prediction), msg_history
