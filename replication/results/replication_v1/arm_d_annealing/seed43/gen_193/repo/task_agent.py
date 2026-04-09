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
    
    # Try to find JSON objects in code blocks
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
        # Find all potential JSON objects
        potential_jsons = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
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
- Check for proper units and significant figures where applicable

GRADING CRITERIA FOR MATH:
- **Correct**: Complete proof/solution with all steps justified, no gaps, correct final answer
- **Almost**: Nearly complete proof with correct main idea, but ONE critical gap or error:
  * One key lemma is stated but not proven
  * One significant calculation error in an otherwise correct approach
  * One logical gap that prevents the proof from being complete
  * The structure is correct but one essential component is missing
- **Partial**: Correct approach but incomplete:
  * Multiple gaps in the reasoning
  * Correct setup but doesn't reach the conclusion
  * Some valid intermediate results but missing key steps
  * On the right track but far from a complete solution
- **Incorrect**: Wrong approach, major errors, or little valid reasoning

CRITICAL: Be STRICT with proofs. A proof with ANY unjustified claim is NOT 'Correct'.
If the student claims to have solved it but the reasoning has gaps, use 'Almost' (if one gap) or 'Partial' (if multiple gaps).""",
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
            "Example 1 - 'Correct' (Full marks, 100% complete):",
            "- Student provides a complete proof with all steps justified",
            "- Final answer matches the solution exactly",
            "- No logical gaps or missing reasoning",
            "- All required components present and correct",
            "- Grade: Correct",
            "",
            "Example 2 - 'Almost' (80-95% complete, ONE critical flaw):",
            "- Student's solution is nearly complete with correct main idea throughout",
            "- Correct approach and reasoning for most of the problem",
            "- Has ONE significant error or missing key insight that prevents full marks",
            "- Example: Correct proof structure but one key lemma is unproven",
            "- Example: Correct solution method but final calculation has a significant error",
            "- Would be correct if not for this one critical flaw",
            "- Grade: Almost",
            "",
            "Example 3 - 'Partial' (30-70% complete, incomplete):",
            "- Student has the right approach but solution is incomplete",
            "- Some correct reasoning but missing multiple key steps",
            "- Partial progress toward the solution, far from complete",
            "- Example: Correct setup and first few steps, but doesn't reach conclusion",
            "- Example: Valid intermediate results but missing key components",
            "- Grade: Partial",
            "",
            "Example 4 - 'Incorrect' (<30% correct, wrong approach):",
            "- Student uses wrong approach or method entirely",
            "- Major logical errors throughout",
            "- Little to no valid reasoning",
            "- Grade: Incorrect",
            "",
            "## Decision Framework",
            "When grading, ask yourself these questions in order:",
            "1. Is the answer 100% complete and correct? -> Correct",
            "2. Is it nearly perfect with just ONE critical flaw? -> Almost",
            "3. Is it on the right track but incomplete (30-70%)? -> Partial",
            "4. Is it mostly wrong or using wrong approach? -> Incorrect",
            "",
            "Key distinction: 'Almost' = one flaw away from perfect; 'Partial' = incomplete but promising",
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
            "## Grading Rubric (STRICT - Be Conservative)",
            "When assigning grades, use these STRICT criteria:",
            "",
            "- **Correct**: ONLY if the answer is COMPLETELY correct with:",
            "  - Correct final answer that matches the solution exactly",
            "  - Sound reasoning with no logical gaps or errors",
            "  - All required steps present and correct",
            "  - No significant omissions or mistakes",
            "  - Use this ONLY when you are confident the answer is fully correct",
            "  - This is for answers that would receive 100% or full marks",
            "",
            "- **Almost**: Use when the answer is NEARLY correct but has ONE critical flaw:",
            "  - Correct main idea and approach",
            "  - Almost complete solution with correct reasoning throughout",
            "  - Has ONE significant error or missing key insight that prevents full correctness",
            "  - Minor mistakes that are not negligible but don't invalidate the whole approach",
            "  - This is for answers that are roughly 80-95% complete but have a critical gap",
            "  - Example: Correct proof structure but one key lemma is unproven or wrong",
            "  - Example: Correct solution but final answer has a small but significant error",
            "  - IMPORTANT: 'Almost' is for answers that are VERY CLOSE to correct, not just 'good progress'",
            "",
            "- **Partial**: Use when the answer shows meaningful progress but is incomplete:",
            "  - Correct approach but incomplete solution (missing multiple steps)",
            "  - Some correct reasoning but significant gaps remain",
            "  - Partially correct results with valid reasoning for some parts",
            "  - On the right track but far from complete",
            "  - This is for answers that are roughly 30-70% complete/correct",
            "  - Example: Correct setup and first few steps, but solution doesn't reach conclusion",
            "  - Example: Some valid intermediate results but missing key components",
            "",
            "- **Incorrect**: Use when the answer has fundamental problems:",
            "  - Wrong approach or method entirely",
            "  - Major logical errors or gaps throughout",
            "  - Answer is mostly or completely wrong",
            "  - Little to no valid reasoning",
            "  - This is for answers that are less than 30% correct",
            "",
            "## IMPORTANT GRADING PRINCIPLES",
            "1. When in doubt, be CONSERVATIVE - prefer 'Partial' or 'Incorrect' over 'Correct'",
            "2. 'Correct' should be reserved for answers that would receive full marks",
            "3. If there's ANY significant error, do NOT mark as 'Correct'",
            "4. Consider the grading guidelines carefully - they often specify what constitutes each grade",
            "5. Look for explicit point values in guidelines (e.g., 7 points = Correct, 3 = Partial, 1 = Almost)",
            "6. 'Almost' vs 'Partial': 'Almost' means nearly perfect with one flaw; 'Partial' means incomplete but on track",
            "7. If the answer has multiple gaps or is far from complete, use 'Partial' not 'Almost'",
            "8. 'Almost' should be used sparingly - only when the answer is truly close to correct",
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
        grade_lower = grade.lower().strip()
        
        # Remove common prefixes/suffixes
        grade_lower = grade_lower.replace("grade:", "").replace("final grade:", "").strip()
        grade_lower = grade_lower.strip('"\'')
        
        # Check for exact matches first (most reliable)
        exact_matches = {
            "correct": "correct",
            "almost": "almost", 
            "partial": "partial",
            "incorrect": "incorrect",
            "wrong": "incorrect",
        }
        if grade_lower in exact_matches:
            return exact_matches[grade_lower]
        
        # Check for numeric grades (common in IMO-style grading)
        if grade_lower in ["7", "7/7", "full", "complete", "100%"]:
            return "correct"
        elif grade_lower in ["3", "3/7", "half", "50%"]:
            return "partial"
        elif grade_lower in ["1", "1/7", "~1", "almost correct"]:
            return "almost"
        elif grade_lower in ["0", "0/7", "none", "0%"]:
            return "incorrect"
        
        # Check for partial word matches (be careful with ordering)
        # "almost" should be checked before "correct" to avoid "almost correct" -> "correct"
        if "almost" in grade_lower:
            return "almost"
        elif "partial" in grade_lower:
            return "partial"
        elif "incorrect" in grade_lower or "wrong" in grade_lower:
            return "incorrect"
        elif "correct" in grade_lower:
            return "correct"
        
        # If no match, return as-is (lowercase)
        return grade_lower

    def _extract_prediction(self, text: str) -> tuple[str, str]:
        """Extract prediction and reasoning from response text.
        
        Returns:
            (prediction, reasoning) tuple
        """
        prediction = "None"
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
            last_json = extracted[-1]
            if "response" in last_json:
                raw_prediction = str(last_json["response"]).strip()
                prediction = self._normalize_grade(raw_prediction)
                self.log_fn(f"Extracted grade: '{raw_prediction}' -> '{prediction}'")
            if "reasoning" in last_json:
                reasoning = str(last_json["reasoning"]).strip()
        
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
        prediction = "None"
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
                
                if prediction != "None":
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
                    prediction = f"Error: {e}"
        
        # Store in cache if successful
        if self._use_cache and prediction != "None" and not prediction.startswith("Error:"):
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
