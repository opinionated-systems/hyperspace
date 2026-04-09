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
            "math": """For mathematical proof problems:
- **Correct**: Complete proof with all steps valid, correct reasoning, and proper conclusion. The solution is essentially complete and could be accepted as a full solution.
- **Almost**: Proof is nearly complete with minor gaps or small errors that are NOT negligible. The core approach is correct but needs small fixes to be complete. The student has made substantial progress but fallen just short of a complete solution.
- **Partial**: Significant progress made but missing key steps, or has notable errors in reasoning. The student has made meaningful progress toward the solution but has significant gaps or errors. This is for solutions that show good understanding but are incomplete.
- **Incorrect**: Wrong approach, fundamental errors, or no meaningful progress toward solution

Key considerations:
- Check if the proof structure follows logical steps
- Verify key lemmas and theorems are applied correctly
- Look for gaps in reasoning that need to be filled
- Consider if alternative valid approaches are used
- Check if the conclusion properly follows from the premises

IMPORTANT DISTINCTION - Almost vs Partial:
- "Almost" = Nearly complete, minor non-negligible issues, could become correct with small fixes
- "Partial" = Significant progress but major gaps remain, needs substantial work to be complete""",
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
                "## Grading Guidelines (from problem set)",
                guidelines,
                "",
                "### How to interpret these guidelines:",
                "- The (Partial) section lists what partial credit achievements look like",
                "- The (Almost) section describes what an 'Almost' grade looks like - nearly complete with minor issues",
                "- If the student's answer meets the (Almost) criteria → grade as 'Almost'",
                "- If the student's answer meets some (Partial) criteria but not (Almost) → grade as 'Partial'",
                "- If the student's answer goes beyond (Partial) and (Almost) to be essentially complete → grade as 'Correct'",
                "- If the student's answer doesn't meet (Partial) criteria → grade as 'Incorrect'",
            ])
        
        # Add domain-specific guidance if available
        if domain_guidance:
            prompt_parts.extend([
                "",
                "## Domain-Specific Guidance",
                domain_guidance,
            ])
        
        prompt_parts.extend([
            "",
            "## Student's Answer",
            student_answer,
            "",
            "## Instructions",
            "Follow this structured analysis process:",
            "",
            "### Step 1: Compare Against Correct Solution",
            "- Identify which parts of the student's answer match the correct solution",
            "- Note any deviations, errors, or omissions",
            "- Check if the final answer/conclusion is correct",
            "",
            "### Step 2: Evaluate Completeness",
            "- Are all key steps present in the student's solution?",
            "- Are there gaps in the reasoning chain?",
            "- Is the proof/solution structure sound?",
            "",
            "### Step 3: Assess Errors and Issues",
            "- Are errors minor (fixable with small changes) or major (fundamental flaws)?",
            "- Does the student show understanding of the core concepts?",
            "- Is the approach fundamentally correct or wrong?",
            "",
            "### Step 4: Apply Decision Tree",
            "Use the decision tree in the Grading Rubric section to determine the grade:",
            "- First ask: Is it essentially complete and correct? → Correct",
            "- If not: Is it nearly complete with minor non-negligible issues? → Almost",
            "- If not: Does it show meaningful progress with significant gaps? → Partial",
            "- Otherwise: → Incorrect",
            "",
            "### Step 5: Provide Reasoning and Grade",
            "- Explain your analysis and why you chose this grade",
            "- Reference specific aspects of the student's answer",
            "- Be explicit about which decision tree criteria you applied",
            "",
            "## Grading Rubric",
            "You MUST assign exactly ONE of these four grades:",
            "",
            "- **Correct**: The answer is fully correct with complete reasoning and correct final result. The solution is essentially complete and correct. This is for solutions that could be accepted as full solutions.",
            "- **Almost**: The solution is nearly complete and correct, but has minor mistakes or gaps that are NOT negligible. The core approach is right but needs small fixes. The student has made substantial progress but fallen just short of a complete solution.",
            "- **Partial**: The answer shows some correct reasoning and progress toward the solution, but has significant gaps, missing steps, or notable errors. Partial credit is warranted. The student has made meaningful progress but the solution is incomplete.",
            "- **Incorrect**: The answer contains fundamental errors, uses a wrong approach, or is completely wrong. Little to no correct reasoning.",
            "",
            "## How to Decide - Decision Tree",
            "Follow this decision tree carefully:",
            "",
            "1. Is the answer essentially complete and correct? (All key steps present, reasoning valid, conclusion correct)",
            "   → YES: Grade as **Correct**",
            "   → NO: Continue to step 2",
            "",
            "2. Is the solution nearly complete with only minor non-negligible issues? (Core approach correct, small fixes needed, substantial progress made)",
            "   → YES: Grade as **Almost**",
            "   → NO: Continue to step 3",
            "",
            "3. Does the answer show meaningful progress with correct reasoning but has significant gaps or notable errors? (Good understanding shown but incomplete)",
            "   → YES: Grade as **Partial**",
            "   → NO: Grade as **Incorrect**",
            "",
            "## Key Distinctions",
            "- **Correct vs Almost**: Correct = fully complete; Almost = nearly complete but has minor non-negligible issues",
            "- **Almost vs Partial**: Almost = needs small fixes; Partial = needs substantial work, significant gaps remain",
            "- **Partial vs Incorrect**: Partial = meaningful progress shown; Incorrect = little to no correct reasoning",
            "",
            "## Response Format (REQUIRED)",
            "You MUST respond with valid JSON wrapped in <json>...</json> tags. Do not include any text outside the JSON tags.",
            "",
            "<json>",
            '{',
            '    "reasoning": "Your detailed step-by-step analysis. First explain your comparison with the correct solution, then your completeness evaluation, then your error assessment, then which decision tree criteria you applied, and finally why you chose the specific grade.",',
            '    "response": "Your final grade: must be exactly one of Correct, Almost, Partial, or Incorrect"',
            '}',
            "</json>",
            "",
            "CRITICAL REQUIREMENTS:",
            "1. The 'response' field must contain ONLY one of these exact values: Correct, Almost, Partial, or Incorrect",
            "2. Do NOT add any other text, punctuation, or explanation in the 'response' field",
            "3. The 'reasoning' field should show your complete analysis process",
            "4. Make sure your JSON is valid - no trailing commas, proper quotes, etc.",
        ])
        
        return "\n".join(prompt_parts)

    def _normalize_prediction(self, prediction: str) -> str:
        """Normalize prediction to one of the valid labels."""
        if not prediction:
            return "None"
        
        # Clean up the prediction
        pred = prediction.strip().lower()
        
        # Remove common prefixes/suffixes and punctuation
        pred = pred.replace("grade:", "").replace("final grade:", "").replace("assessment:", "").strip()
        pred = pred.replace(".", "").replace("!", "").replace("?", "").replace('"', "").replace("'", "").strip()
        
        # Map to valid labels
        valid_labels = ["correct", "almost", "partial", "incorrect"]
        
        # Check for exact match first (most reliable)
        for label in valid_labels:
            if pred == label:
                return label.capitalize()
        
        # Check for exact match at word boundaries
        import re
        for label in valid_labels:
            if re.search(r'\b' + label + r'\b', pred):
                return label.capitalize()
        
        # Check for partial matches (less reliable, use with caution)
        for label in valid_labels:
            if label in pred:
                return label.capitalize()
        
        # Check for common variations and synonyms
        if any(word in pred for word in ["full", "right", "complete", "perfect", "exact", "accurate"]):
            return "Correct"
        if any(word in pred for word in ["almost", "nearly", "minor", "close", "nearly complete", "mostly correct"]):
            return "Almost"
        if any(word in pred for word in ["partial", "some", "incomplete", "partly", "half", "fragment"]):
            return "Partial"
        if any(word in pred for word in ["wrong", "error", "fail", "none", "incorrect", "bad", "invalid", "false"]):
            return "Incorrect"
        
        return "None"

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
                prediction = self._normalize_prediction(raw_prediction)
                self.log_fn(f"Raw prediction: '{raw_prediction}' -> Normalized: '{prediction}'")
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
- The 'response' field must be exactly one of: Correct, Almost, Partial, or Incorrect
- Do NOT add any other text, punctuation, or explanation in the 'response' field
- The 'response' field should contain ONLY the grade word

Now try again with the original grading task. Remember to follow the decision tree in the instructions:
1. Is it essentially complete? → Correct
2. Is it nearly complete with minor non-negligible issues? → Almost  
3. Does it show meaningful progress with significant gaps? → Partial
4. Otherwise → Incorrect

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
