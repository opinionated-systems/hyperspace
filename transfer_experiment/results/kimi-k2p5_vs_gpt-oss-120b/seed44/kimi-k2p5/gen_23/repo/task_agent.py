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


def _extract_label_from_reasoning(text: str) -> str | None:
    """Extract the grade label from reasoning text by looking for explicit conclusions."""
    text_lower = text.lower()
    
    # Look for explicit grade statements at the end of the text
    conclusion_patterns = [
        r'grade\s*(?:is|should\s*be|:)\s*["\']?(correct|almost|partial|incorrect)["\']?',
        r'therefore\s*,?\s*(?:the\s*)?(?:grade\s*)?(?:is\s*)?["\']?(correct|almost|partial|incorrect)["\']?',
        r'thus\s*,?\s*(?:the\s*)?(?:grade\s*)?(?:is\s*)?["\']?(correct|almost|partial|incorrect)["\']?',
        r'conclusion\s*:?\s*(?:the\s*)?(?:grade\s*)?(?:is\s*)?["\']?(correct|almost|partial|incorrect)["\']?',
        r'assign\s*(?:the\s*)?grade\s*["\']?(correct|almost|partial|incorrect)["\']?',
        r'(?:this|the)\s*(?:solution|answer|proof)\s*(?:is|should\s*be\s*graded\s*as)\s*["\']?(correct|almost|partial|incorrect)["\']?',
    ]
    
    for pattern in conclusion_patterns:
        match = re.search(pattern, text_lower)
        if match:
            val = match.group(1).lower()
            if val in {"correct", "almost", "partial", "incorrect"}:
                return val
    
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
        
        # Check if guidelines have specific Almost criteria
        guidelines_lower = grading_guidelines.lower()
        has_almost_section = "(almost)" in guidelines_lower
        
        # Build the instruction with clearer structure
        instruction = f"""You are an expert mathematical grader for competition mathematics (IMO, USAMO, Putnam, etc.). Evaluate the student's solution and assign exactly one grade.

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

Assign exactly ONE of these four grades:

1. "correct" - Complete and correct proof
   - All steps are mathematically sound and justified
   - No gaps, no missing cases, no errors
   - Would receive full marks (7/7 in IMO)
   - Only use when confident the solution is 100% correct

2. "almost" - Nearly perfect with ONLY trivial issues
   - The solution is essentially complete and correct
   - Issues are truly minor (typos, insignificant calculation errors, minor notation issues)
   - Would receive near-full marks (6/7 in IMO)
   - The core proof is complete and correct
   - EXTREMELY RARE - use only when the solution is truly excellent
   {"- Check if guidelines have (Almost) section with specific criteria" if has_almost_section else ""}

3. "partial" - Meaningful progress but incomplete or flawed
   - Has valid mathematical reasoning and correct elements
   - Shows understanding of the problem and approach
   - BUT has significant gaps, missing key steps, or major errors
   - Incomplete solutions with good structure fall here
   - Solutions with correct approach but critical flaws fall here
   - "Structured partial progress" solutions fall here
   - DEFAULT for incomplete but reasonable attempts

4. "incorrect" - Little to no meaningful progress
   - Approach is fundamentally wrong or misguided
   - No significant correct mathematical progress
   - Answer is irrelevant, nonsense, or completely wrong
   - Only use if there's truly nothing correct of value

=== DECISION PROCESS ===
Step 1: Does the solution contain meaningful correct mathematics? If NO → "incorrect"
Step 2: Is the solution complete with all steps justified? If YES → "correct"
Step 3: Is the solution nearly complete with only trivial issues? If YES → "almost"
Step 4: Otherwise → "partial" (has good elements but significant issues)

=== CRITICAL RULES ===
- "incorrect" should only be used when there's NO meaningful correct mathematics
- "partial" is for solutions that show understanding but have significant gaps/errors
- "almost" is VERY RARE - only for solutions that are essentially complete
- When uncertain between "partial" and "incorrect", prefer "partial" if there's any valid reasoning
- When uncertain between "almost" and "partial", prefer "partial" unless issues are truly trivial

=== GRADING EXAMPLES ===

Example 1 - "correct":
A complete proof with all steps justified, no gaps, proper handling of all cases, and correct conclusion. The solution would receive full marks.

Example 2 - "almost":
A solution that is essentially complete and correct, but has a minor calculation error or typo that doesn't affect the overall proof structure. Would receive 6/7 marks.

Example 3 - "partial":
A solution with good initial setup, valid approach, and some correct reasoning, but missing a key lemma or having a significant gap that prevents it from being complete. Most "structured partial progress" solutions fall here.

Example 4 - "incorrect":
A solution that is fundamentally wrong, contains nonsense, or shows no understanding of the problem. The approach is completely misguided.

Example 5 - Common mistake to avoid:
Do NOT grade a solution as "incorrect" if it contains any valid mathematical reasoning, even if incomplete. "incorrect" is only for solutions with NO redeeming value.

=== OUTPUT ===
Respond with EXACTLY ONE JSON object:
<json>{{"response": "correct"}}</json>
or
<json>{{"response": "almost"}}</json>
or
<json>{{"response": "partial"}}</json>
or
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
        
        # Try to extract from reasoning text
        if prediction not in {"correct", "almost", "partial", "incorrect"}:
            try:
                reasoning_label = _extract_label_from_reasoning(response_text)
                if reasoning_label:
                    prediction = reasoning_label
            except Exception:
                pass
        
        # Text-based fallback with priority order - BE CONSERVATIVE
        if prediction not in {"correct", "almost", "partial", "incorrect"}:
            text = response_text.lower()
            
            # Priority: look for explicit response field patterns first
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
            # Final keyword fallback - be careful about ordering
            elif " almost" in text or "almost " in text:
                prediction = "almost"
            elif " partial" in text or "partial " in text:
                prediction = "partial"
            elif " incorrect" in text or "incorrect " in text:
                prediction = "incorrect"
            elif " correct" in text or "correct " in text:
                prediction = "correct"

        # Default to partial if invalid - this is safer than incorrect
        if prediction not in {"correct", "almost", "partial", "incorrect"}:
            prediction = "partial"

        return str(prediction), msg_history
