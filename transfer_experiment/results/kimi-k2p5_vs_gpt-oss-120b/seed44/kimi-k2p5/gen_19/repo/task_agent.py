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
    """Analyze the grading guidelines to understand the grading criteria."""
    guidelines_lower = grading_guidelines.lower()
    
    # Check for Almost section and extract its content
    has_almost_section = "(almost)" in guidelines_lower
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
    
    return {
        "has_almost_section": has_almost_section,
        "almost_has_content": almost_has_content,
        "has_partial_section": has_partial_section,
    }


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
        
        # Analyze guidelines to provide context
        guidelines_analysis = _analyze_grading_guidelines(grading_guidelines)
        
        # Build specific guidance based on guidelines analysis
        almost_guidance = ""
        if guidelines_analysis["has_almost_section"] and guidelines_analysis["almost_has_content"]:
            almost_guidance = """
IMPORTANT: The grading guidelines contain an (Almost) section describing minor issues.
Use "almost" ONLY when:
- The student achieved ALL items in the (Partial) section
- AND the issues match the (Almost) section criteria (minor mistakes, small gaps)
- The solution is essentially complete with only small details missing"""
        
        instruction = f"""You are an expert mathematical grader for competition mathematics (IMO, USAMO, Putnam, etc.). Your task is to evaluate a student's solution and assign exactly one of four possible grades.

=== PROBLEM INFORMATION ===
DOMAIN: {domain}

PROBLEM STATEMENT:
{problem}

=== REFERENCE MATERIALS ===
OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}
{almost_guidance}

=== STUDENT WORK ===
STUDENT'S ANSWER:
{student_answer}

=== GRADING RUBRIC ===
You must assign exactly ONE of these four grades:

1. "correct" - The solution is fully correct and complete.
   - All key steps match the official solution
   - No significant errors or gaps in reasoning
   - All cases considered, all claims properly justified
   - The solution would receive full points

2. "almost" - The solution is nearly correct with ONLY minor issues.
   - The solution is essentially complete but has small gaps or minor errors
   - The student demonstrated nearly full understanding
   - Issues are truly minor (e.g., small calculation error, missing one trivial case)
   - IMPORTANT: Only use "almost" when the solution is VERY CLOSE to correct

3. "partial" - The student made meaningful progress but has substantial gaps.
   - Has correct elements but is far from a complete solution
   - Missing key steps or has significant gaps in reasoning
   - Use this when the student has SOME correct progress
   - This is the DEFAULT grade for solutions with partial progress
   - When in doubt between "almost" and "partial", choose "partial"

4. "incorrect" - The solution is wrong or shows no meaningful progress.
   - Approach is fundamentally flawed or missing key insights
   - No significant correct progress toward the solution
   - Answer is completely wrong or irrelevant

=== DECISION FRAMEWORK ===
Follow these steps in order:

Step 1: Is the solution fully correct and complete?
   - All steps correct? All cases covered? No gaps?
   - If YES → grade "correct"
   - If NO → continue to Step 2

Step 2: Does the solution show ANY meaningful correct progress?
   - Any correct key insight? Any valid step toward solution?
   - If NO → grade "incorrect"
   - If YES → continue to Step 3

Step 3: Is the solution VERY CLOSE to complete with only minor issues?
   - Are the issues truly minor (small gaps, not major missing pieces)?
   - Would the solution be correct with just small fixes?
   - If YES and issues are truly minor → grade "almost"
   - Otherwise → grade "partial"

=== KEY PRINCIPLES ===
- "correct" = fully complete, no issues
- "almost" = nearly complete, only tiny issues (use sparingly!)
- "partial" = meaningful progress but significant gaps (most common for incomplete work)
- "incorrect" = no meaningful progress
- When uncertain between two grades, choose the LOWER grade (more conservative)

=== OUTPUT FORMAT ===
You MUST respond with EXACTLY ONE JSON object in this format:
<json>{{"response": "correct"}}</json>
OR
<json>{{"response": "almost"}}</json>
OR
<json>{{"response": "partial"}}</json>
OR
<json>{{"response": "incorrect"}}</json>

Your entire response should be just this JSON object. Do not include any other text."""

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
        
        # Text-based fallback with priority order
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

        # Default to incorrect if invalid
        if prediction not in {"correct", "almost", "partial", "incorrect"}:
            prediction = "incorrect"

        return str(prediction), msg_history
