"""
Task agent: solves a given task with a single LLM call.

Reimplemented from facebookresearch/HyperAgents task_agent.py.
Same interface, same JSON output format, same extraction logic.

This is the INITIAL task agent. The meta agent modifies this file
during self-improvement. The evaluation harness loads whatever
task_agent.py exists at the agent's repo path.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from functools import lru_cache

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)

# Simple in-memory cache for grading results to avoid redundant API calls
_response_cache: dict[str, tuple[str, str]] = {}


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles nested JSON objects and common formatting issues.
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
        
        # Try to parse the JSON
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # Try common fixes: remove trailing commas, fix quotes
            fixed = _fix_json_string(inner)
            try:
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                # Try extracting just the outermost JSON object
                try:
                    obj = _extract_outermost_json(inner)
                    if obj:
                        results.append(obj)
                except Exception:
                    continue
    
    # Also try to find JSON in code blocks as a fallback
    if not results:
        code_block_pattern = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if code_block_pattern:
            try:
                results.append(json.loads(code_block_pattern.group(1)))
            except json.JSONDecodeError:
                try:
                    fixed = _fix_json_string(code_block_pattern.group(1))
                    results.append(json.loads(fixed))
                except json.JSONDecodeError:
                    pass
    
    return results or None


def _fix_json_string(text: str) -> str:
    """Apply common JSON fixes."""
    # Remove trailing commas before closing braces/brackets
    import re
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    # Fix single quotes to double quotes (basic)
    text = text.replace("'", '"')
    return text


def _extract_outermost_json(text: str) -> dict | None:
    """Extract the outermost JSON object from text, handling nesting."""
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


def _extract_json_with_regex(text: str) -> list[dict] | None:
    """Fallback JSON extraction using regex for malformed responses."""
    results = []
    
    # Try to find JSON objects in code blocks (markdown format)
    json_pattern = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_pattern:
        try:
            results.append(json.loads(json_pattern.group(1)))
        except json.JSONDecodeError:
            # Try with fixes
            try:
                fixed = _fix_json_string(json_pattern.group(1))
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                pass
    
    # Try to find any JSON-like structure with "response" key
    response_pattern = re.search(r'"response"\s*:\s*"([^"]*)"', text)
    if response_pattern and not results:
        results.append({"response": response_pattern.group(1)})
    
    # Try to find JSON without code blocks (look for { ... } patterns)
    if not results:
        # Find all potential JSON objects with balanced braces
        potential_jsons = []
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
                    potential_jsons.append(text[start_idx:i+1])
                    start_idx = -1
        
        for pj in potential_jsons:
            try:
                obj = json.loads(pj)
                if "response" in obj or "reasoning" in obj:
                    results.append(obj)
            except json.JSONDecodeError:
                try:
                    fixed = _fix_json_string(pj)
                    obj = json.loads(fixed)
                    if "response" in obj or "reasoning" in obj:
                        results.append(obj)
                except json.JSONDecodeError:
                    continue
    
    # Last resort: try to extract just the grade from plain text
    if not results:
        # Look for grade keywords in the text
        text_lower = text.lower()
        for grade in ["correct", "partial", "almost", "incorrect"]:
            # Check if the grade appears as a standalone word
            import re as re_module
            if re_module.search(r'\b' + grade + r'\b', text_lower):
                results.append({"response": grade})
                break
    
    return results or None


def _generate_cache_key(inputs: dict) -> str:
    """Generate a cache key from grading inputs for deduplication."""
    # Create a deterministic key from the essential inputs
    key_data = {
        "problem": inputs.get("problem", ""),
        "solution": inputs.get("solution", ""),
        "student_answer": inputs.get("student_answer", ""),
        "guidelines": inputs.get("grading_guidelines", ""),
    }
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_str.encode()).hexdigest()[:32]


def clear_response_cache() -> int:
    """Clear the response cache and return the number of entries cleared."""
    global _response_cache
    count = len(_response_cache)
    _response_cache.clear()
    return count


def get_cache_stats() -> dict:
    """Get current cache statistics."""
    return {
        "entries": len(_response_cache),
        "size_bytes": sum(len(json.dumps({"p": p, "r": r})) for p, r in _response_cache.values()),
    }


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "", use_cache: bool = True) -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3
        self._extraction_stats = {"success": 0, "fallback": 0, "failed": 0}
        self._use_cache = use_cache
        self._cache_hits = 0
        self._cache_misses = 0

    def _get_domain_specific_guidance(self, domain: str) -> str:
        """Get domain-specific grading guidance."""
        guidance = {
            "math": """For mathematical problems:
- Check if the final answer matches numerically or symbolically
- Verify that the reasoning steps are mathematically sound
- Alternative valid approaches should be accepted
- Partial credit for correct setup but calculation errors
- Check for proper units and significant figures where applicable
- Look for logical gaps, missing steps, or unjustified claims
- A proof with a gap in reasoning may be 'Partial' or 'Almost' depending on severity
- If the student claims to have solved it but the reasoning is incomplete, mark as 'Partial' or 'Almost'
- Be thorough but fair - don't be overly harsh on minor issues""",
            "code": """For programming problems:
- Verify the code produces correct output for the given examples
- Check for proper handling of edge cases
- Code style matters less than correctness, but syntax errors are critical
- Partial credit for correct algorithm but minor implementation bugs
- Time/space complexity should be reasonable but not strictly graded""",
            "logic": """For logic puzzles:
- Verify the reasoning chain is sound and complete
- Check that all constraints are satisfied
- Alternative valid deductions should be accepted
- Partial credit for correct partial deductions
- The final answer must be explicitly stated""",
        }
        return guidance.get(domain.lower(), "")

    def _build_grading_prompt(self, inputs: dict) -> str:
        """Build a structured prompt for the grading task with chain-of-thought."""
        domain = inputs.get("domain", "unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Get domain-specific guidance
        domain_guidance = self._get_domain_specific_guidance(domain)
        
        # Build the prompt
        prompt_parts = [
            f"You are an expert grader for {domain} problems. Your task is to evaluate a student's answer against the correct solution and grading guidelines.",
            "",
            "## Problem Statement",
            problem,
            "",
            "## Correct Solution",
            solution,
        ]
        
        # Add guidelines if present
        if guidelines:
            prompt_parts.extend([
                "",
                "## Grading Guidelines",
                guidelines,
            ])
        
        # Add domain-specific guidance if available
        if domain_guidance:
            prompt_parts.extend([
                "",
                "## Domain-Specific Guidance",
                domain_guidance,
            ])
        
        # Add few-shot examples to guide grading
        prompt_parts.extend([
            "",
            "## Grading Examples",
            "Example 1 - 'Correct':",
            "- Student provides a complete proof with all steps justified",
            "- Final answer matches the solution exactly",
            "- No logical gaps or missing reasoning",
            "- Grade: Correct",
            "",
            "Example 2 - 'Partial':",
            "- Student has the right approach but solution is incomplete",
            "- Some correct reasoning but missing key steps",
            "- Partial progress toward the solution",
            "- Grade: Partial",
            "",
            "Example 3 - 'Almost':",
            "- Student's solution is nearly complete with correct main idea",
            "- Has a significant error or missing key insight",
            "- Would be correct if not for one critical flaw",
            "- Grade: Almost",
            "",
            "Example 4 - 'Incorrect':",
            "- Student uses wrong approach or method",
            "- Major logical errors throughout",
            "- Little to no valid reasoning",
            "- Grade: Incorrect",
        ])
        
        prompt_parts.extend([
            "",
            "## Student's Answer",
            student_answer,
            "",
            "## Instructions",
            "1. First, analyze the student's answer step by step. Compare it against the correct solution.",
            "2. Identify any errors, omissions, or alternative valid approaches.",
            "3. Consider the grading guidelines and domain-specific guidance carefully.",
            "4. Provide your reasoning for the grade you will assign.",
            "5. Finally, provide your grade/assessment in the JSON format below.",
            "",
            "## Grading Rubric",
            "When assigning grades, use these criteria:",
            "",
            "- **Correct**: The answer is COMPLETELY correct with:",
            "  - Correct final answer that matches the solution",
            "  - Sound reasoning with no significant logical gaps or errors",
            "  - All required steps present and correct",
            "  - No major omissions or mistakes",
            "  - Minor typos or notation issues don't disqualify if the reasoning is sound",
            "",
            "- **Partial**: The answer shows meaningful progress but has significant issues:",
            "  - Correct approach but incomplete solution",
            "  - Some calculation errors but correct method",
            "  - Partially correct results with valid reasoning for parts",
            "  - Missing key steps but on the right track",
            "  - This is for answers that are roughly 30-70% complete/correct",
            "",
            "- **Almost**: The answer is nearly correct but has flaws preventing full correctness:",
            "  - Correct main idea but error in execution",
            "  - Almost complete but missing a key insight or step",
            "  - Minor mistakes that prevent full correctness",
            "  - This is for answers that are roughly 70-90% complete but not fully correct",
            "",
            "- **Incorrect**: The answer has fundamental problems:",
            "  - Wrong approach or method",
            "  - Major logical errors or gaps",
            "  - Answer is mostly or completely wrong",
            "  - Little to no valid reasoning",
            "  - This is for answers that are less than 30% correct",
            "",
            "## IMPORTANT GRADING PRINCIPLES",
            "1. Be FAIR and BALANCED - don't be overly harsh or lenient",
            "2. 'Correct' should be used for answers that would receive full or near-full marks",
            "3. Minor issues (typos, notation) don't make an answer 'Partial' if the reasoning is sound",
            "4. Consider the grading guidelines carefully - they often specify what constitutes each grade",
            "5. Look for explicit point values in guidelines (e.g., 7 points = Correct, 3 = Partial, 1 = Almost)",
            "",
            "## Response Format (REQUIRED)",
            "You MUST respond with valid JSON wrapped in <json>...</json> tags. Do not include any text outside the JSON tags.",
            "",
            "<json>",
            '{',
            '    "reasoning": "Your detailed step-by-step analysis and reasoning...",',
            '    "response": "Your final grade: Correct, Partial, Almost, or Incorrect"',
            '}',
            "</json>",
            "",
            "IMPORTANT: Ensure your JSON is valid and properly formatted. The 'response' field should contain only the grade (Correct, Partial, Almost, or Incorrect), not the reasoning.",
        ])
        
        return "\n".join(prompt_parts)

    def _normalize_grade(self, grade: str) -> str:
        """Normalize a grade string to one of the standard labels."""
        if not grade or not isinstance(grade, str):
            return "none"
            
        grade_lower = grade.lower().strip()
        
        # Remove common prefixes/suffixes
        grade_lower = grade_lower.replace("grade:", "").replace("final grade:", "").strip()
        grade_lower = grade_lower.strip('"\'')
        
        # Check for exact matches first
        exact_matches = {
            "correct": "correct",
            "partial": "partial", 
            "almost": "almost",
            "incorrect": "incorrect",
            "wrong": "incorrect",
            "none": "none",
        }
        if grade_lower in exact_matches:
            return exact_matches[grade_lower]
        
        # Check for numeric grades (common in IMO-style grading)
        if grade_lower in ["7", "7/7", "full", "complete", "100%", "full marks"]:
            return "correct"
        elif grade_lower in ["3", "3/7", "half", "50%"]:
            return "partial"
        elif grade_lower in ["1", "1/7", "~1"]:
            return "almost"
        elif grade_lower in ["0", "0/7", "none", "0%", "zero"]:
            return "incorrect"
        
        # Check for partial matches with priority order
        # Priority: incorrect > partial > almost > correct
        # This prevents "not correct" from being matched as "correct"
        if "incorrect" in grade_lower or "wrong" in grade_lower:
            return "incorrect"
        if "partial" in grade_lower:
            return "partial"
        if "almost" in grade_lower:
            return "almost"
        if "correct" in grade_lower:
            return "correct"
        
        # If no match, return "none" to indicate failure
        return "none"

    def _extract_prediction(self, text: str) -> tuple[str, str]:
        """Extract prediction and reasoning from response text.
        
        Returns:
            (prediction, reasoning) tuple
        """
        prediction = "none"
        reasoning = ""
        
        # Try primary extraction method
        extracted = _extract_jsons(text)
        if extracted is not None:
            self._extraction_stats["success"] += 1
            self.log_fn("JSON extraction: primary method succeeded")
        else:
            # Fallback to regex extraction
            extracted = _extract_json_with_regex(text)
            if extracted is not None:
                self._extraction_stats["fallback"] += 1
                self.log_fn("JSON extraction: fallback method succeeded")
            else:
                self._extraction_stats["failed"] += 1
                self.log_fn("JSON extraction: all methods failed")
        
        if extracted:
            # Try to find the best JSON object with a response field
            best_json = None
            for json_obj in extracted:
                if "response" in json_obj:
                    best_json = json_obj
                    break
            
            # If no response field found, use the last one
            if best_json is None:
                best_json = extracted[-1]
            
            if "response" in best_json:
                raw_prediction = str(best_json["response"]).strip()
                prediction = self._normalize_grade(raw_prediction)
                self.log_fn(f"Extracted grade: '{raw_prediction}' -> '{prediction}'")
            if "reasoning" in best_json:
                reasoning = str(best_json["reasoning"]).strip()
        
        return prediction, reasoning

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Check cache first if enabled
        if self._use_cache:
            cache_key = _generate_cache_key(inputs)
            if cache_key in _response_cache:
                self._cache_hits += 1
                cached_prediction, cached_reasoning = _response_cache[cache_key]
                self.log_fn(f"Cache hit! Using cached result: {cached_prediction}")
                # Return a minimal msg_history for cache hits
                return str(cached_prediction), [{"role": "assistant", "text": f"[Cached] {cached_reasoning}"}]
            self._cache_misses += 1
        
        instruction = self._build_grading_prompt(inputs)
        
        msg_history = []
        prediction = "none"
        reasoning = ""
        
        # Retry loop for robust extraction
        for attempt in range(self.max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=msg_history if attempt > 0 else [],
                )
                
                last_text = msg_history[-1]["text"] if msg_history else ""
                prediction, reasoning = self._extract_prediction(last_text)
                
                if prediction != "none":
                    self.log_fn(f"Successfully extracted prediction: {prediction}")
                    if reasoning:
                        self.log_fn(f"Reasoning: {reasoning[:200]}...")
                    break
                else:
                    self.log_fn(f"Attempt {attempt + 1}: Failed to extract prediction, retrying...")
                    # Add feedback for retry with more specific guidance
                    if attempt < self.max_retries - 1:
                        instruction = f"""ERROR: Your previous response did not contain valid JSON with a 'response' field.

Your response was:
---
{last_text[:500]}
---

You MUST respond with ONLY valid JSON wrapped in <json>...</json> tags. Do not include any text before or after the JSON tags.

Correct format:
<json>
{{
    "reasoning": "Your detailed analysis here...",
    "response": "Correct"
}}
</json>

IMPORTANT: 
- The JSON must be valid (no trailing commas, proper quotes)
- Both 'reasoning' and 'response' fields are required
- The 'response' field must be EXACTLY one of: "Correct", "Partial", "Almost", or "Incorrect" (case-sensitive)
- Do NOT include any other text in the 'response' field

Now try again with the original task:

{self._build_grading_prompt(inputs)}"""
                    
            except Exception as e:
                self.log_fn(f"Error in attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    prediction = f"error: {e}"
        
        # Store in cache if successful
        if self._use_cache and prediction not in ("none", "error") and not prediction.startswith("error:"):
            _response_cache[cache_key] = (prediction, reasoning)
            self.log_fn(f"Cached result for key: {cache_key[:16]}...")
        
        # Log extraction statistics periodically
        total = sum(self._extraction_stats.values())
        if total > 0 and total % 10 == 0:
            self.log_fn(f"Extraction stats after {total} attempts: {self._extraction_stats}")
            if self._use_cache:
                cache_total = self._cache_hits + self._cache_misses
                if cache_total > 0:
                    hit_rate = self._cache_hits / cache_total * 100
                    self.log_fn(f"Cache stats: {self._cache_hits} hits, {self._cache_misses} misses ({hit_rate:.1f}% hit rate)")
        
        return str(prediction), msg_history
