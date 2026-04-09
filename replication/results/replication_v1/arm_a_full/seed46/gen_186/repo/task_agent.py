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
from typing import Callable

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles markdown code blocks (```json...```) as fallback.
    Includes additional fallback for JSON embedded in other text.
    """
    results = []
    search_from = 0
    
    # First try <json>...</json> tags
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
            # Try to extract JSON from within the content if it's wrapped in other text
            try:
                # Find first { and last }
                json_start = inner.find("{")
                json_end = inner.rfind("}")
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    results.append(json.loads(inner[json_start:json_end + 1]))
            except json.JSONDecodeError:
                continue
    
    # Fallback: try markdown code blocks
    if not results:
        search_from = 0
        while True:
            start = text.find("```json", search_from)
            if start == -1:
                # Also try without 'json' specifier
                start = text.find("```", search_from)
                if start == -1:
                    break
                end = text.find("```", start + 3)
            else:
                end = text.find("```", start + 7)
            if end == -1:
                break
            
            if text[start:start+7] == "```json":
                inner = text[start + 7:end].strip()
            else:
                inner = text[start + 3:end].strip()
            search_from = end + 3
            try:
                results.append(json.loads(inner))
            except json.JSONDecodeError:
                continue
    
    # Fallback: try to find JSON objects directly in the text
    if not results:
        # Look for JSON-like structures with curly braces
        brace_start = text.find("{")
        while brace_start != -1:
            # Try to find matching closing brace
            brace_count = 0
            for i, char in enumerate(text[brace_start:]):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        json_str = text[brace_start:brace_start + i + 1]
                        try:
                            parsed = json.loads(json_str)
                            # Only accept if it has expected fields
                            if any(key in parsed for key in ["response", "grade", "score", "analysis", "reasoning", "understanding", "evaluation", "result"]):
                                results.append(parsed)
                        except json.JSONDecodeError:
                            pass
                        break
            brace_start = text.find("{", brace_start + 1)
    
    # Additional fallback: try to find JSON with common field patterns
    if not results:
        # Look for patterns like {"response": ...} or {"grade": ...}
        import re
        json_patterns = [
            r'\{\s*"response"\s*:\s*"[^"]+"[^}]*\}',
            r'\{\s*"grade"\s*:\s*"[^"]+"[^}]*\}',
            r'\{\s*"score"\s*:\s*[^,}]+[^}]*\}',
        ]
        for pattern in json_patterns:
            for match in re.finditer(pattern, text, re.DOTALL):
                try:
                    parsed = json.loads(match.group())
                    results.append(parsed)
                except json.JSONDecodeError:
                    continue
    
    return results or None


def _log_extraction_details(text: str, results: list[dict] | None, logger_fn: Callable) -> None:
    """Log detailed information about JSON extraction for debugging.
    
    Args:
        text: The raw text that was processed
        results: The extracted JSON results (or None if failed)
        logger_fn: Logging function to use
    """
    if results:
        logger_fn(f"Successfully extracted {len(results)} JSON object(s)")
        for i, obj in enumerate(results):
            keys = list(obj.keys())
            logger_fn(f"  Object {i+1}: keys={keys}")
    else:
        # Log hints about why extraction might have failed
        has_json_tags = "<json>" in text and "</json>" in text
        has_code_blocks = "```" in text
        has_braces = "{" in text and "}" in text
        
        logger_fn("JSON extraction failed - diagnostic info:")
        logger_fn(f"  - Contains <json> tags: {has_json_tags}")
        logger_fn(f"  - Contains code blocks: {has_code_blocks}")
        logger_fn(f"  - Contains curly braces: {has_braces}")
        logger_fn(f"  - Text length: {len(text)} chars")
        
        # Show a snippet of the text for debugging
        snippet = text[:200].replace("\n", " ")
        logger_fn(f"  - Text preview: {snippet}...")


def _normalize_grade(grade: str) -> str:
    """Normalize grade to standard format.
    
    Converts various grade formats to a consistent representation.
    Handles both string grades (Correct/Almost/Partial/Incorrect) and numeric scores (0-7).
    
    Mapping:
    - 7 points -> "Correct" (full credit)
    - 5-6 points -> "Almost" (minor issues)
    - 1-4 points -> "Partial" (partial credit)
    - 0 points -> "Incorrect" (no credit)
    """
    if not grade:
        return "None"
    
    # Handle non-string types (int, float, etc.)
    if not isinstance(grade, str):
        grade = str(grade)
    
    grade = grade.strip()
    grade_lower = grade.lower()
    
    # First, check for numeric grades and map to categories
    # 7 = Correct, 5-6 = Almost, 1-4 = Partial, 0 = Incorrect
    if grade_lower == "7":
        return "Correct"
    if grade_lower in ["5", "6"]:
        return "Almost"
    if grade_lower in ["1", "2", "3", "4"]:
        return "Partial"
    if grade_lower == "0":
        return "Incorrect"
    
    # Map common variations to standard grades (using lowercase keys)
    grade_map = {
        "correct": "Correct",
        "right": "Correct",
        "true": "Correct",
        "yes": "Correct",
        "full": "Correct",
        "full credit": "Correct",
        "complete": "Correct",
        "solved": "Correct",
        "valid": "Correct",
        "acceptable": "Correct",
        "perfect": "Correct",
        "fully correct": "Correct",
        "entirely correct": "Correct",
        "correct solution": "Correct",
        "valid solution": "Correct",
        "almost": "Almost",
        "almost correct": "Almost",
        "nearly correct": "Almost",
        "mostly correct": "Almost",
        "minor errors": "Almost",
        "small mistakes": "Almost",
        "essentially correct": "Almost",
        "tiny error": "Almost",
        "nearly solved": "Almost",
        "partial": "Partial",
        "partially correct": "Partial",
        "partial credit": "Partial",
        "partially": "Partial",
        "some credit": "Partial",
        "incomplete": "Partial",
        "unfinished": "Partial",
        "missing": "Partial",
        "mostly": "Partial",
        "significant progress": "Partial",
        "half correct": "Partial",
        "some understanding": "Partial",
        "partially solved": "Partial",
        "incorrect": "Incorrect",
        "wrong": "Incorrect",
        "false": "Incorrect",
        "no": "Incorrect",
        "none": "Incorrect",
        "zero": "Incorrect",
        "invalid": "Incorrect",
        "unacceptable": "Incorrect",
        "error": "Incorrect",
        "completely wrong": "Incorrect",
        "no progress": "Incorrect",
        "no understanding": "Incorrect",
    }
    
    # Check for exact match first (case-insensitive)
    if grade_lower in grade_map:
        return grade_map[grade_lower]
    
    # Try to extract numeric score with context-aware patterns
    # Look for patterns like "Score: 5", "Grade: 3", "5 points", etc.
    numeric_patterns = [
        r'\bscore[\s]*[=:]?\s*([0-7])\b',
        r'\bgrade[\s]*[=:]?\s*([0-7])\b',
        r'\bpoints?[\s]*[=:]?\s*([0-7])\b',
        r'\b([0-7])\s*(?:points?|score|grade)\b',
        r'\bfinal[\s]+(?:grade|score)[\s]*[=:]?\s*([0-7])\b',
        r'\b(?:score|grade)[\s]+is[\s]+([0-7])\b',
    ]
    for pattern in numeric_patterns:
        match = re.search(pattern, grade_lower)
        if match:
            score = match.group(1)
            if score == "7":
                return "Correct"
            elif score in ["5", "6"]:
                return "Almost"
            elif score == "0":
                return "Incorrect"
            else:
                return "Partial"
    
    # Fallback: simple numeric match if no pattern matched
    numeric_match = re.search(r'\b([0-7])\b', grade)
    if numeric_match:
        score = numeric_match.group(1)
        if score == "7":
            return "Correct"
        elif score in ["5", "6"]:
            return "Almost"
        elif score == "0":
            return "Incorrect"
        else:
            return "Partial"
    
    # Scoring system for keyword-based classification
    # This helps with ambiguous cases where multiple keywords might match
    scores = {
        "Correct": 0,
        "Almost": 0,
        "Partial": 0,
        "Incorrect": 0
    }
    
    # Strong indicators for each category (higher weight)
    strong_indicators = {
        "Correct": ["correct", "right", "true", "full marks", "complete solution", "perfect", "fully correct", "entirely correct", "valid solution"],
        "Almost": ["almost", "nearly", "minor error", "small mistake", "tiny error", "essentially correct", "mostly correct", "nearly solved"],
        "Partial": ["partial", "incomplete", "unfinished", "significant progress", "some understanding", "partial credit", "half correct", "partially solved"],
        "Incorrect": ["incorrect", "wrong", "false", "completely wrong", "no progress", "invalid", "unacceptable", "no understanding"]
    }
    
    # Weak indicators for each category (lower weight)
    weak_indicators = {
        "Correct": ["full", "complete", "solved", "valid", "acceptable"],
        "Almost": ["minor", "small error", "slight mistake", "mostly"],
        "Partial": ["missing", "unfinished", "some progress", "incomplete"],
        "Incorrect": ["none", "zero", "error", "no"]
    }
    
    # Calculate scores
    for category, keywords in strong_indicators.items():
        for keyword in keywords:
            if keyword in grade_lower:
                scores[category] += 3  # Strong indicator
    
    for category, keywords in weak_indicators.items():
        for keyword in keywords:
            if keyword in grade_lower:
                scores[category] += 1  # Weak indicator
    
    # Check for negations that might flip the meaning
    negation_patterns = ["not correct", "not right", "not valid", "not acceptable", "not complete", "not solved"]
    for pattern in negation_patterns:
        if pattern in grade_lower:
            # Reduce Correct score if negated
            scores["Correct"] -= 2
            scores["Incorrect"] += 1
    
    # Return the category with highest score if there's a clear winner
    if max(scores.values()) > 0:
        best_category = max(scores, key=scores.get)
        # Only return if there's a clear winner (not a tie)
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) >= 2 and sorted_scores[0] > sorted_scores[1]:
            return best_category
    
    # Check if grade contains keywords (order matters - check specific categories first)
    # Check for "almost" category first (more specific than partial)
    if any(word in grade_lower for word in ["almost", "nearly", "minor", "small mistake", "tiny error"]):
        return "Almost"
    # Check for "partial" category
    if "partial" in grade_lower:
        return "Partial"
    if any(word in grade_lower for word in ["incomplete", "unfinished", "missing", "significant progress", "some understanding"]):
        return "Partial"
    # Check for "correct" category
    if any(word in grade_lower for word in ["correct", "right", "true", "full", "complete", "solved", "valid", "acceptable", "perfect"]):
        return "Correct"
    # Check for "incorrect" category
    if any(word in grade_lower for word in ["incorrect", "wrong", "false", "none", "zero", "error", "invalid", "unacceptable", "completely wrong", "no progress"]):
        return "Incorrect"
    
    # Default: capitalize first letter
    return grade.capitalize()


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _build_grading_prompt(self, problem: str, answer: str, official_solution: str, 
                             rubric: str, student_answer: str) -> str:
        """Build a comprehensive grading prompt with clear instructions."""
        return f"""You are an expert IMO (International Mathematical Olympiad) grader. Your task is to evaluate a student's solution to a mathematics problem.

## Problem Statement
{problem}

## Correct Answer
{answer}

## Official Solution
{official_solution}

## Grading Rubric
{rubric}

## Student's Answer
{student_answer}

## Your Task
Carefully analyze the student's answer and assign a grade using the following structured approach:

### Step 1: Understanding Assessment
- Verify you understand the problem correctly
- Identify the key mathematical concepts required
- Note the expected solution approach

### Step 2: Solution Analysis
- Check if the student's approach is mathematically valid
- Identify any errors, gaps, or incorrect assumptions
- Compare their approach to the official solution
- Check mathematical notation and terminology
- Evaluate the logical flow and reasoning quality

### Step 3: Grade Assignment
Based on the rubric, assign ONE of these four grades:
- **Correct**: Complete, correct solution with clear reasoning (7 points)
  - All steps are logically sound and mathematically valid
  - Final answer is correct
  - Clear explanation of the reasoning process
- **Almost**: Minor errors or gaps, but essentially correct approach (5-6 points)
  - Correct approach with minor computational errors
  - Correct approach with missing minor details
  - Correct approach with unclear notation but valid reasoning
- **Partial**: Significant progress but incomplete or with major errors (1-4 points)
  - Shows understanding of key concepts but incomplete solution
  - Correct initial steps but major errors later
  - Partial understanding demonstrated with some valid work
- **Incorrect**: No meaningful progress or completely wrong approach (0 points)
  - No valid mathematical reasoning
  - Completely wrong approach with no understanding shown
  - Answer is wrong with no supporting work

## Important Rules
1. You MUST use exactly one of these four labels: Correct, Almost, Partial, or Incorrect
2. Be strict but fair - partial credit requires meaningful mathematical progress
3. "Almost" is for solutions that are nearly correct but have minor issues
4. "Partial" is for solutions that show understanding but are incomplete or have significant errors
5. Focus on the mathematical validity of the approach, not just the final answer
6. Consider the rubric carefully - different problems may have different grading criteria

## Response Format
Respond ONLY in JSON format with the following structure:
<json>
{{
    "understanding": "Brief summary of the problem and correct approach",
    "analysis": "Detailed analysis of the student's answer - what they got right and wrong",
    "reasoning": "Step-by-step reasoning for your grade decision",
    "response": "Your final grade: MUST be one of 'Correct', 'Almost', 'Partial', or 'Incorrect'"
}}
</json>

IMPORTANT: The "response" field MUST contain exactly one of: "Correct", "Almost", "Partial", or "Incorrect" (case-sensitive)."""

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with structured reasoning.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract fields for structured prompting
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Try to extract answer from solution if available
        answer = inputs.get("answer", "")
        if not answer and solution:
            # Try to extract final answer from solution
            lines = solution.strip().split('\n')
            for line in reversed(lines):
                if line.strip() and not line.strip().startswith('#'):
                    answer = line.strip()
                    break

        # Use the new prompt builder for better structured prompting
        instruction = self._build_grading_prompt(
            problem=problem,
            answer=answer,
            official_solution=solution,
            rubric=grading_guidelines,
            student_answer=student_answer
        )

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with enhanced error handling
        prediction = "None"
        try:
            # Get the last assistant message
            last_assistant_msg = None
            for msg in reversed(msg_history):
                if msg.get("role") == "assistant" or "text" in msg:
                    last_assistant_msg = msg.get("text", msg.get("content", ""))
                    break
            
            if not last_assistant_msg:
                self.log_fn("Warning: No assistant message found in history")
                return str(prediction), msg_history
            
            extracted = _extract_jsons(last_assistant_msg)
            
            # Log extraction details for debugging
            _log_extraction_details(last_assistant_msg, extracted, self.log_fn)
            
            if extracted:
                # Try to get response field first, fallback to other common fields
                last_extract = extracted[-1]
                
                # Priority order for grade extraction - include reasoning field
                # "response" is the primary field as per the prompt instructions
                grade_fields = ["response", "grade", "score", "result", "final_grade", "evaluation", "reasoning"]
                for field in grade_fields:
                    if field in last_extract:
                        prediction = last_extract[field]
                        # If we found a valid grade in the response field, use it directly
                        if field == "response" and isinstance(prediction, str):
                            pred_lower = prediction.strip().lower()
                            if pred_lower in ["correct", "almost", "partial", "incorrect"]:
                                prediction = pred_lower.capitalize()
                                break
                        break
                
                # Normalize the grade for consistency
                prediction = _normalize_grade(prediction)
                
                # Log detailed analysis for debugging
                if "analysis" in last_extract:
                    self.log_fn(f"Analysis: {last_extract['analysis'][:200]}...")
                if "partial_credit_reasoning" in last_extract:
                    self.log_fn(f"Partial Credit: {last_extract['partial_credit_reasoning'][:200]}...")
                if "reasoning" in last_extract:
                    self.log_fn(f"Reasoning: {last_extract['reasoning'][:200]}...")
                if "understanding" in last_extract:
                    self.log_fn(f"Understanding: {last_extract['understanding'][:200]}...")
                
                self.log_fn(f"Extracted grade: {prediction}")
            else:
                self.log_fn("Warning: No JSON blocks found in response")
                # Try to extract grade directly from text as last resort
                text_lower = last_assistant_msg.lower()
                
                # Look for explicit grade/score declarations
                grade_patterns = [
                    r'(?:grade|score|final grade|final score)[\s]*[=:]\s*["\']?([a-zA-Z0-9\s]+)["\']?(?:\n|$|\.|<)',
                    r'(?:grade|score|final grade|final score)[\s]+is[\s]+["\']?([a-zA-Z0-9\s]+)["\']?(?:\n|$|\.|<)',
                    r'response["\']?\s*[=:]\s*["\']?([a-zA-Z0-9\s]+)["\']?(?:\n|$|\.|<)',
                ]
                for pattern in grade_patterns:
                    match = re.search(pattern, text_lower, re.IGNORECASE)
                    if match:
                        extracted_text = match.group(1).strip()
                        prediction = _normalize_grade(extracted_text)
                        self.log_fn(f"Extracted grade from text pattern: {prediction}")
                        break
                
                # Try to find grade labels directly in text (with word boundaries)
                if prediction == "None":
                    for label in ["Correct", "Almost", "Partial", "Incorrect"]:
                        # Use word boundary to avoid partial matches
                        pattern = r'\b' + label.lower() + r'\b'
                        if re.search(pattern, text_lower):
                            prediction = label
                            self.log_fn(f"Extracted grade from text search: {prediction}")
                            break
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            import traceback
            self.log_fn(f"Traceback: {traceback.format_exc()}")

        return str(prediction), msg_history
