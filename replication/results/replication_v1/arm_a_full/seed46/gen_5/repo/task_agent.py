"""
Task agent: solves a given task with a single LLM call.

Reimplemented from facebookresearch/HyperAgents task_agent.py.
Same interface, same JSON output format, same extraction logic.

This is the INITIAL task agent. The meta agent modifies this file
during self-improvement. The evaluation harness loads whatever
task_agent.py exists at the agent's repo path.

ENHANCED VERSION: Added confidence scoring, robust JSON extraction,
and structured grading rubric for more reliable IMO grading.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

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
    """Extract JSON objects with multiple fallback strategies.
    
    Tries multiple approaches:
    1. Standard <json>...</json> blocks
    2. Markdown code blocks with json
    3. Raw JSON objects in text
    """
    # Try standard extraction first
    results = _extract_jsons(text)
    if results:
        return results
    
    # Try markdown code blocks
    md_pattern = r'```json\s*(.*?)\s*```'
    matches = re.findall(md_pattern, text, re.DOTALL)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            continue
    
    # Try raw JSON objects (look for { ... } patterns)
    if not results:
        # Find JSON-like structures
        json_pattern = r'\{[^{}]*"[^"]+"[^{}]*\}'
        for match in re.finditer(json_pattern, text, re.DOTALL):
            try:
                candidate = match.group(0)
                # Try to parse, allowing for nested braces by expanding
                for end in range(match.end(), min(match.end() + 10000, len(text))):
                    try:
                        extended = text[match.start():end]
                        parsed = json.loads(extended)
                        results.append(parsed)
                        break
                    except json.JSONDecodeError:
                        continue
            except Exception:
                continue
    
    return results or None


def _normalize_grade(grade: Any) -> str:
    """Normalize various grade formats to a standard string."""
    if grade is None:
        return "None"
    
    grade_str = str(grade).strip()
    
    # Handle numeric grades
    try:
        num = float(grade_str)
        if num == 0:
            return "0"
        elif num >= 7:
            return "7"
        elif num >= 1:
            return str(int(num))
    except ValueError:
        pass
    
    # Handle text grades
    grade_lower = grade_str.lower()
    if any(word in grade_lower for word in ["correct", "full", "complete", "right"]):
        return "7"
    elif any(word in grade_lower for word in ["partial", "half", "some"]):
        return "Partial"
    elif any(word in grade_lower for word in ["incorrect", "wrong", "none", "zero", "fail"]):
        return "0"
    
    return grade_str


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning.
    
    ENHANCED: Now includes confidence scoring, robust JSON extraction,
    and structured grading rubric for more reliable results.
    """

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.stats = {"total": 0, "json_extracted": 0, "fallback_used": 0, "errors": 0}

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with structured reasoning.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        self.stats["total"] += 1
        
        # Extract fields for structured prompting
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematical grader for International Mathematical Olympiad (IMO) problems.

Your task is to evaluate a student's answer by comparing it against the official solution and grading guidelines.

## Problem Domain
{domain}

## Problem Statement
{problem}

## Official Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Instructions
1. First, analyze the student's answer step by step. Identify what they got right and what they got wrong.
2. Compare their reasoning against the official solution.
3. Check if they followed the grading guidelines.
4. Assign points based on the official IMO 0-7 scoring system:
   - 7 points: Complete, correct solution
   - 6 points: Minor flaw in an otherwise correct solution
   - 5 points: Significant progress with minor gaps
   - 3-4 points: Partial progress with substantial gaps
   - 1-2 points: Some meaningful progress
   - 0 points: No meaningful progress or incorrect

Respond in JSON format with the following schema:
<json>
{{
    "analysis": "Your detailed step-by-step analysis of the student's answer...",
    "response": "Your final grade as a number 0-7",
    "confidence": "High|Medium|Low - based on clarity of the answer",
    "reasoning_quality": "Excellent|Good|Fair|Poor - quality of student's reasoning"
}}
</json>"""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.stats["errors"] += 1
            self.log_fn(f"LLM call failed: {e}")
            return "Error", [{"role": "system", "text": f"LLM error: {e}"}]

        # Extract prediction from JSON with multiple fallback strategies
        prediction = "None"
        analysis = ""
        confidence = "Unknown"
        reasoning_quality = "Unknown"
        
        try:
            last_text = msg_history[-1]["text"] if msg_history else ""
            
            # Try standard extraction first
            extracted = _extract_jsons(last_text)
            if extracted:
                self.stats["json_extracted"] += 1
            else:
                # Try flexible extraction
                extracted = _extract_json_flexible(last_text)
                if extracted:
                    self.stats["fallback_used"] += 1
                    self.log_fn("Used fallback JSON extraction")
            
            if extracted:
                last_extract = extracted[-1]
                
                # Extract response/grade with priority order
                for key in ["response", "grade", "score", "result", "points", "mark"]:
                    if key in last_extract:
                        prediction = _normalize_grade(last_extract[key])
                        break
                
                # Extract additional fields
                analysis = last_extract.get("analysis", "")
                confidence = last_extract.get("confidence", "Unknown")
                reasoning_quality = last_extract.get("reasoning_quality", "Unknown")
                
                # Log detailed info
                self.log_fn(f"Grade: {prediction}, Confidence: {confidence}, Quality: {reasoning_quality}")
                if analysis:
                    self.log_fn(f"Analysis: {analysis[:200]}...")
            else:
                self.log_fn("No JSON found in response, attempting text extraction")
                # Fallback: try to extract grade from plain text
                grade_patterns = [
                    r'grade[\s:]+(\d+)',
                    r'score[\s:]+(\d+)',
                    r'(\d+)[\s]*points',
                    r'final[\s:]+(\d+)',
                ]
                for pattern in grade_patterns:
                    match = re.search(pattern, last_text, re.IGNORECASE)
                    if match:
                        prediction = _normalize_grade(match.group(1))
                        self.log_fn(f"Extracted grade from text: {prediction}")
                        break
                        
        except Exception as e:
            self.stats["errors"] += 1
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
    
    def get_stats(self) -> dict:
        """Return extraction statistics for debugging."""
        return self.stats.copy()
