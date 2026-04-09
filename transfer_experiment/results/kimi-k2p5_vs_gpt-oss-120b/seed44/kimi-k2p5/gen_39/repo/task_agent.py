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
    # Use regex to find tags case-insensitively with DOTALL for multiline content
    pattern = re.compile(r"<json>(.*?)</json>", re.IGNORECASE | re.DOTALL)
    for match in pattern.finditer(text):
        inner = match.group(1).strip()
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
                resp_match = re.search(r"[\"']response[\"']\s*[:=]\s*[\"']([^\"']+)[\"']", inner, re.IGNORECASE)
                if resp_match:
                    val = resp_match.group(1).lower()
                    if val in {"correct", "almost", "partial", "incorrect"}:
                        parsed = {"response": val}
            except Exception:
                pass
        
        # Strategy 5: Look for grade values with more flexible patterns
        if parsed is None:
            inner_lower = inner.lower()
            for grade in ["correct", "almost", "partial", "incorrect"]:
                # Match the grade as a standalone value
                if re.search(rf'^[\s"\']*{grade}[\s"\']*$', inner_lower) or \
                   re.search(rf':\s*["\']?{grade}["\']?\s*$', inner_lower) or \
                   re.search(rf'["\']?{grade}["\']?\s*$', inner_lower):
                    parsed = {"response": grade}
                    break
        
        if parsed is not None and isinstance(parsed, dict) and "response" in parsed:
            results.append(parsed)
    
    return results or None


def _extract_json_from_raw_text(text: str) -> dict | None:
    """Extract JSON object from raw text without <json> tags."""
    text_lower = text.lower()
    
    # Strategy 1: Try to find JSON object patterns directly with response field
    json_pattern = re.search(r'\{\s*["\']response["\']\s*:\s*["\']([^"\']+)["\']\s*\}', text, re.IGNORECASE)
    if json_pattern:
        try:
            return json.loads(json_pattern.group(0).replace("'", '"'))
        except json.JSONDecodeError:
            pass
    
    # Strategy 2: Try to find any JSON-like object with response field (multiline)
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
    
    # Strategy 5: Look for standalone grade words at the start or end of text
    text_stripped = text.strip().lower()
    for grade in ["correct", "almost", "partial", "incorrect"]:
        # Check if the entire response is just the grade
        if text_stripped == grade:
            return {"response": grade}
        # Check if the grade appears at the very beginning
        if text_stripped.startswith(grade + "\n") or text_stripped.startswith(grade + " "):
            return {"response": grade}
        # Check if the grade appears at the very end
        if text_stripped.endswith(" " + grade) or text_stripped.endswith("\n" + grade) or text_stripped.endswith(grade):
            return {"response": grade}
    
    # Strategy 6: Look for grade words surrounded by word boundaries
    for grade in ["correct", "almost", "partial", "incorrect"]:
        if re.search(rf'\b{grade}\b', text_lower):
            return {"response": grade}
    
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
        # Key improvements:
        # 1. Few-shot examples showing exact output format
        # 2. Stronger emphasis on the JSON output requirement
        # 3. Explicit instruction to NOT provide explanations
        instruction = f"""You are an expert mathematical grader for competition mathematics (IMO, USAMO, Putnam, etc.). Your task is to grade the student's solution by comparing it to the official solution.

PROBLEM DOMAIN: {domain}

PROBLEM STATEMENT:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

=== GRADING RUBRIC ===

Assign exactly ONE of these four grades:

**CORRECT**: The solution is a complete, rigorous proof that would receive full marks (7/7).
- All logical steps are valid and clearly explained
- No gaps in reasoning
- No errors in calculations or logic
- The proof is essentially the same as or equivalent to the official solution

**ALMOST**: The solution is nearly complete with only minor, easily fixable issues (5-6/7).
- The core proof structure and main ideas are correct
- Small arithmetic/algebraic errors that don't affect the main argument
- Typos or minor notation inconsistencies
- Missing trivial edge cases that are obvious to fix
- The proof would work with trivial corrections
- IMPORTANT: The student demonstrates full understanding of the solution approach

**PARTIAL**: The solution has meaningful correct elements but significant gaps or errors (1-4/7).
- Valid approach identified but missing key steps
- Shows genuine understanding of some concepts but fails to complete the proof
- Has correct insights mixed with major errors
- Incomplete proof despite a good start
- Missing crucial lemmas or intermediate results

**INCORRECT**: Little to no valid mathematical progress (0/7).
- Approach is fundamentally wrong or irrelevant
- No significant correct mathematics
- Nonsense or shows zero understanding of the problem

=== DECISION PROCESS ===

Step 1: Does the solution contain any valid mathematics? If NO → INCORRECT

Step 2: Is the overall approach fundamentally correct? If NO → INCORRECT

Step 3: Does the solution have the correct main ideas and proof structure?
- If YES and the proof is complete with only trivial issues → ALMOST
- If YES but missing key steps or has major gaps → PARTIAL
- If NO (wrong approach) → INCORRECT

Step 4: Is the solution fully rigorous with no issues? If YES → CORRECT

=== KEY DISTINCTION: CORRECT vs ALMOST ===
- CORRECT: The solution is perfect and would receive full marks
- ALMOST: The solution is essentially correct but has minor cosmetic issues

=== KEY DISTINCTION: ALMOST vs PARTIAL ===
- ALMOST: The proof structure is sound; only trivial fixes needed
- PARTIAL: Major gaps exist; significant work needed to complete the proof

=== EXAMPLES OF REQUIRED OUTPUT FORMAT ===

Example 1 - If the solution is perfect:
<json>{{"response": "correct"}}</json>

Example 2 - If the solution has minor issues:
<json>{{"response": "almost"}}</json>

Example 3 - If the solution has significant gaps:
<json>{{"response": "partial"}}</json>

Example 4 - If the solution is wrong:
<json>{{"response": "incorrect"}}</json>

=== CRITICAL INSTRUCTIONS ===
1. Your ENTIRE response must be ONLY the JSON object in the format shown above
2. Do NOT include any explanations, reasoning, or analysis before or after the JSON
3. Do NOT use markdown code blocks (```json) - use ONLY the <json>...</json> tags
4. The response field MUST be one of: "correct", "almost", "partial", or "incorrect"
5. Think about the grading decision, but output ONLY the JSON
6. Be CONSERVATIVE: When in doubt between two grades, choose the LOWER grade

YOUR RESPONSE (only JSON, nothing else):"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction
        prediction = "None"
        response_text = msg_history[-1]["text"]
        
        # Try JSON tag extraction first (most reliable)
        try:
            extracted = _extract_jsons(response_text)
            if extracted:
                for item in extracted:
                    if isinstance(item, dict) and "response" in item:
                        val = item["response"].lower().strip()
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
                    val = raw_extracted["response"].lower().strip()
                    if val in {"correct", "almost", "partial", "incorrect"}:
                        prediction = val
            except Exception:
                pass
        
        # Text-based fallback - look for the grade words with priority ordering
        if prediction not in {"correct", "almost", "partial", "incorrect"}:
            text = response_text.lower()
            
            # Priority order: more specific patterns first
            # Check for explicit grade declarations first
            explicit_patterns = [
                (r'grade\s*(is|should\s+be|assigned|given)\s*["\']?incorrect["\']?', "incorrect"),
                (r'grade\s*(is|should\s+be|assigned|given)\s*["\']?partial["\']?', "partial"),
                (r'grade\s*(is|should\s+be|assigned|given)\s*["\']?almost["\']?', "almost"),
                (r'grade\s*(is|should\s+be|assigned|given)\s*["\']?correct["\']?', "correct"),
            ]
            
            for pattern, label in explicit_patterns:
                if re.search(pattern, text, re.MULTILINE | re.IGNORECASE):
                    prediction = label
                    break
            
            # If still not found, check for JSON-like patterns
            if prediction not in {"correct", "almost", "partial", "incorrect"}:
                json_patterns = [
                    (r'["\']response["\']?\s*[:=]\s*["\']?incorrect["\']?', "incorrect"),
                    (r'["\']response["\']?\s*[:=]\s*["\']?partial["\']?', "partial"),
                    (r'["\']response["\']?\s*[:=]\s*["\']?almost["\']?', "almost"),
                    (r'["\']response["\']?\s*[:=]\s*["\']?correct["\']?', "correct"),
                ]
                
                for pattern, label in json_patterns:
                    if re.search(pattern, text, re.MULTILINE | re.IGNORECASE):
                        prediction = label
                        break
            
            # If still not found, check for standalone grades at line boundaries
            if prediction not in {"correct", "almost", "partial", "incorrect"}:
                standalone_patterns = [
                    (r'(?:^|\n)incorrect(?:\s|$|\n)', "incorrect"),
                    (r'(?:^|\n)partial(?:\s|$|\n)', "partial"),
                    (r'(?:^|\n)almost(?:\s|$|\n)', "almost"),
                    (r'(?:^|\n)correct(?:\s|$|\n)', "correct"),
                ]
                
                for pattern, label in standalone_patterns:
                    if re.search(pattern, text):
                        prediction = label
                        break
            
            # Last resort: word boundaries (but be careful about "correct" appearing in other words)
            if prediction not in {"correct", "almost", "partial", "incorrect"}:
                # Check in order of specificity (almost and partial are more specific than correct)
                for grade in ["incorrect", "partial", "almost", "correct"]:
                    if re.search(rf'\b{grade}\b', text):
                        prediction = grade
                        break
        
        # Default to incorrect if still not valid
        if prediction not in {"correct", "almost", "partial", "incorrect"}:
            prediction = "incorrect"
        
        return str(prediction), msg_history
