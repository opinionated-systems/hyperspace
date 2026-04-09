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

# Valid grading labels
VALID_LABELS = ["correct", "incorrect", "partial"]

# Label priority for conflict resolution (higher = more preferred when uncertain)
# We prefer "incorrect" as the safest default when uncertain
LABEL_PRIORITY = {"incorrect": 3, "partial": 2, "correct": 1}

# Common misspellings and variations mapping
LABEL_ALIASES = {
    "correct": ["correct", "right", "true", "valid", "accurate", "complete"],
    "incorrect": ["incorrect", "wrong", "false", "error", "invalid", "inaccurate", "flawed"],
    "partial": ["partial", "incomplete", "partially", "some", "part", "partial credit"],
}


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks or raw JSON.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also tries to find JSON in code blocks and raw JSON objects.
    Includes robust error handling and multiple fallback strategies.
    """
    if not text or not isinstance(text, str):
        return None
        
    results = []
    search_from = 0
    
    # Try to find <json>...</json> blocks (case insensitive)
    text_lower = text.lower()
    while True:
        start = text_lower.find("<json>", search_from)
        if start == -1:
            break
        end = text_lower.find("</json>", start)
        if end == -1:
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # Try to fix common JSON issues
            try:
                # Replace single quotes with double quotes
                fixed = inner.replace("'", '"')
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                # Try to extract just the response field with more flexible matching
                try:
                    # Match response field with various quote styles
                    match = re.search(r'["\']?response["\']?\s*:\s*["\']?([^"\'\}\n,]+)', inner, re.IGNORECASE)
                    if match:
                        value = match.group(1).strip().lower()
                        if value in VALID_LABELS:
                            results.append({"response": value})
                except Exception:
                    continue
    
    # Also try to find JSON in markdown code blocks
    code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    for match in re.finditer(code_block_pattern, text, re.DOTALL):
        try:
            json_str = match.group(1).strip()
            if json_str:
                results.append(json.loads(json_str))
        except json.JSONDecodeError:
            # Try to fix common JSON issues in code blocks
            try:
                fixed = json_str.replace("'", '"')
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                # Try to extract response field from malformed JSON
                try:
                    resp_match = re.search(r'["\']?response["\']?\s*:\s*["\']?([^"\'\}\n,]+)', json_str, re.IGNORECASE)
                    if resp_match:
                        value = resp_match.group(1).strip().lower()
                        if value in VALID_LABELS:
                            results.append({"response": value})
                except Exception:
                    continue
    
    # Try to find raw JSON objects with "response" field
    json_pattern = r'\{\s*"response"\s*:\s*"([^"]+)"\s*\}'
    for match in re.finditer(json_pattern, text):
        try:
            results.append(json.loads(match.group(0)))
        except json.JSONDecodeError:
            continue
    
    # Try to find JSON objects with single quotes
    json_pattern_single = r"\{\s*'response'\s*:\s*'([^']+)'\s*\}"
    for match in re.finditer(json_pattern_single, text):
        try:
            results.append(json.loads(match.group(0).replace("'", '"')))
        except json.JSONDecodeError:
            continue
    
    # Try to find JSON-like objects with various quote styles
    # Match patterns like {response: "correct"} or {"response": correct}
    flexible_pattern = r'\{\s*["\']?response["\']?\s*:\s*["\']?([^"\'\}\n,]+)["\']?\s*\}'
    for match in re.finditer(flexible_pattern, text, re.IGNORECASE):
        try:
            value = match.group(1).strip().lower()
            if value in VALID_LABELS:
                results.append({"response": value})
        except Exception:
            continue
    
    # Try to find JSON with newlines and extra whitespace (more permissive)
    multiline_pattern = r'\{\s*["\']?response["\']?\s*:\s*["\']?([^"\'\}\n,]+)["\']?\s*\}'
    for match in re.finditer(multiline_pattern, text, re.DOTALL | re.IGNORECASE):
        try:
            value = match.group(1).strip().lower()
            if value in VALID_LABELS:
                results.append({"response": value})
        except Exception:
            continue
    
    # Look for the response field anywhere in the text
    # This handles cases where the LLM outputs the JSON without proper formatting
    loose_pattern = r'["\']?response["\']?\s*[:=]\s*["\']?(correct|incorrect|partial)["\']?'
    for match in re.finditer(loose_pattern, text, re.IGNORECASE):
        try:
            value = match.group(1).strip().lower()
            if value in VALID_LABELS:
                results.append({"response": value})
        except Exception:
            continue
    
    # Look for explicit verdict statements with more patterns
    verdict_patterns = [
        r'(?:the\s+)?(?:final\s+)?verdict\s*(?:is|:)\s*["\']?(correct|incorrect|partial)["\']?',
        r'(?:the\s+)?(?:final\s+)?(?:grade|classification|label)\s*(?:is|:)\s*["\']?(correct|incorrect|partial)["\']?',
        r'(?:i\s+(?:would|will)\s+)?(?:classify|grade|label|mark)\s*(?:this\s+)?(?:as\s+)?["\']?(correct|incorrect|partial)["\']?',
        r'(?:this\s+(?:is|should\s+be))\s*["\']?(correct|incorrect|partial)["\']?',
        r'(?:the\s+answer\s+(?:is|should\s+be))\s*["\']?(correct|incorrect|partial)["\']?',
        r'(?:verdict|decision|assessment)\s*:\s*["\']?(correct|incorrect|partial)["\']?',
        r'(?:therefore|thus|hence)[,:]?\s*(?:the\s+)?(?:answer\s+)?(?:is\s+)?["\']?(correct|incorrect|partial)["\']?',
    ]
    for pattern in verdict_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            try:
                value = match.group(1).strip().lower()
                if value in VALID_LABELS:
                    results.append({"response": value})
            except Exception:
                continue
    
    # Look for labels in backticks, bold, or emphasized
    formatting_patterns = [
        r'`(correct|incorrect|partial)`',
        r'\*\*(correct|incorrect|partial)\*\*',
        r'\*(correct|incorrect|partial)\*',
        r'"(correct|incorrect|partial)"',
        r"'(correct|incorrect|partial)'",
    ]
    for pattern in formatting_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            try:
                value = match.group(1).strip().lower()
                if value in VALID_LABELS:
                    results.append({"response": value})
            except Exception:
                continue
    
    # NEW: Look for standalone labels at the end of the response (common pattern)
    # This catches cases where LLM outputs just the label at the end
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if lines:
        last_line = lines[-1].lower()
        for label in VALID_LABELS:
            # Check if the last line is just the label
            if last_line == label or last_line == f'"{label}"' or last_line == f"'{label}'":
                results.append({"response": label})
                break
            # Check if last line contains the label as the main content
            if re.search(rf'^(?:the\s+)?(?:final\s+)?(?:verdict|grade|classification|label)\s*(?:is|:)?\s*["\']?{label}["\']?$', last_line, re.IGNORECASE):
                results.append({"response": label})
                break
    
    return results if results else None


def _normalize_label(value: str) -> str | None:
    """Normalize a label value to one of the valid labels.
    
    Handles common variations, misspellings, and edge cases.
    """
    if not value or not isinstance(value, str):
        return None
    
    # Clean the value
    clean = value.strip().lower()
    # Remove punctuation and extra whitespace
    clean = re.sub(r'[^a-z]', '', clean)
    
    # Direct match
    if clean in VALID_LABELS:
        return clean
    
    # Check aliases
    for label, aliases in LABEL_ALIASES.items():
        if clean in [a.lower() for a in aliases]:
            return label
    
    # Fuzzy matching for common misspellings
    # Check for partial match with correct (allow shorter matches)
    if 'correct' in clean or clean.startswith('corr') or clean.startswith('right'):
        return 'correct'
    # Check for partial match with incorrect
    if 'incorrect' in clean or clean.startswith('incorr') or clean.startswith('wrong'):
        return 'incorrect'
    # Check for partial match with partial
    if 'partial' in clean or clean.startswith('part') or 'incomplete' in clean:
        return 'partial'
    
    return None


def _extract_label_from_text(text: str) -> str | None:
    """Extract the grading label from raw text using pattern matching.
    
    Looks for exact matches of the three valid labels with priority-based
    matching to find the most likely intended label.
    Uses multiple strategies with increasing fallback tolerance.
    Includes confidence scoring to prefer more explicit mentions.
    """
    if not text or not isinstance(text, str):
        return None
        
    text_lower = text.lower().strip()
    
    # Priority 0: Look for explicit <json> blocks with response field
    json_block_pattern = r'<json>\s*\{\s*"response"\s*:\s*"([^"]+)"\s*\}\s*</json>'
    for match in re.finditer(json_block_pattern, text_lower):
        clean_match = re.sub(r'[^a-z]', '', match.group(1).lower())
        if clean_match in VALID_LABELS:
            return clean_match
    
    # Priority 1: Look for quoted JSON-style values with "response" key
    json_pattern = r'"response"\s*[:=]\s*"([^"]+)"'
    for match in re.finditer(json_pattern, text_lower):
        clean_match = re.sub(r'[^a-z]', '', match.group(1).lower())
        if clean_match in VALID_LABELS:
            return clean_match
    
    # Priority 2: Look for single-quoted JSON-style values
    json_pattern_single = r"'response'\s*[:=]\s*'([^']+)'"
    for match in re.finditer(json_pattern_single, text_lower):
        clean_match = re.sub(r'[^a-z]', '', match.group(1).lower())
        if clean_match in VALID_LABELS:
            return clean_match
    
    # Priority 3: Look for explicit label declarations with colons or equals
    label_alternatives = "correct|incorrect|partial"
    declaration_patterns = [
        rf'(?:is|are|be)\s*[:=]\s*["\']?({label_alternatives})["\']?\b',
        rf'(?:classification|grade|label|verdict|result|evaluation|assessment|decision)\s*[:=]\s*["\']?({label_alternatives})["\']?\b',
        rf'(?:answer|response|prediction|output)\s*[:=]\s*["\']?({label_alternatives})["\']?\b',
        rf'(?:the\s+student\s+(?:answer|response)\s+(?:is|should\s+be))\s*["\']?({label_alternatives})["\']?\b',
        rf'(?:i\s+(?:would|will)\s+(?:classify|grade|label|mark|say))\s*["\']?({label_alternatives})["\']?\b',
        rf'(?:this\s+(?:is|should\s+be|appears\s+to\s+be))\s*["\']?({label_alternatives})["\']?\b',
        rf'(?:the\s+answer\s+is)\s*["\']?({label_alternatives})["\']?\b',
        rf'(?:verdict|conclusion|decision|assessment)\s*[:=]?\s*["\']?({label_alternatives})["\']?\b',
        rf'(?:therefore|thus|hence)[,:]?\s*(?:the\s+)?(?:answer\s+)?(?:is\s+)?["\']?({label_alternatives})["\']?\b',
    ]
    
    for pattern in declaration_patterns:
        match = re.search(pattern, text_lower)
        if match:
            for group in match.groups():
                if group in VALID_LABELS:
                    return group
    
    # Priority 4: Look for labels in code blocks or backticks
    for label in VALID_LABELS:
        # Match backtick-quoted labels
        if re.search(rf'`{label}`', text_lower):
            return label
        # Match bold/italic markdown labels
        if re.search(rf'\*\*{label}\*\*|\*{label}\*', text_lower):
            return label
        # Match quoted labels
        if re.search(rf'"{label}"|\'{label}\'', text_lower):
            return label
    
    # Priority 5: Look for labels at the end of sentences or standalone
    for label in VALID_LABELS:
        # Match at end of sentence (with optional punctuation and whitespace)
        if re.search(rf'\b{label}\b[.!?]*\s*$', text_lower):
            return label
        # Match after "therefore", "thus", "so", "hence", "conclusion"
        if re.search(rf'(?:therefore|thus|so|hence|conclusion|concluding)[,:]?\s+\b{label}\b', text_lower):
            return label
        # Match after "final" or "final answer"
        if re.search(rf'(?:final(?:ly)?|final\s+answer|in\s+conclusion)[,:]?\s+\b{label}\b', text_lower):
            return label
    
    # Priority 6: Look for labels preceded by strong indicators
    strong_indicator_patterns = [
        rf'(?:grade[d]?\s+(?:as|as\s+an)?)\s*["\']?({label_alternatives})["\']?',
        rf'(?:marked\s+(?:as|as\s+an)?)\s*["\']?({label_alternatives})["\']?',
        rf'(?:categorized\s+(?:as|as\s+an)?)\s*["\']?({label_alternatives})["\']?',
    ]
    for pattern in strong_indicator_patterns:
        match = re.search(pattern, text_lower)
        if match:
            for group in match.groups():
                if group in VALID_LABELS:
                    return group
    
    # Priority 7: Look for "verdict:" or "verdict is" patterns
    verdict_patterns = [
        rf'verdict\s*[:=]\s*["\']?({label_alternatives})["\']?',
        rf'verdict\s+is\s*["\']?({label_alternatives})["\']?',
    ]
    for pattern in verdict_patterns:
        match = re.search(pattern, text_lower)
        if match:
            for group in match.groups():
                if group in VALID_LABELS:
                    return group
    
    # Priority 8: Count occurrences of each label as whole words
    label_counts = {}
    for label in VALID_LABELS:
        count = len(re.findall(rf'\b{label}\b', text_lower))
        label_counts[label] = count
    
    if any(label_counts.values()):
        valid_counts = {k: v for k, v in label_counts.items() if v > 0}
        if valid_counts:
            # Use priority to break ties - prefer "incorrect" when uncertain
            max_count = max(valid_counts.values())
            candidates = [k for k, v in valid_counts.items() if v == max_count]
            if len(candidates) == 1:
                return candidates[0]
            else:
                # Tie-break by priority (higher priority wins)
                return max(candidates, key=lambda x: LABEL_PRIORITY.get(x, 0))
    
    # Priority 9: Fuzzy matching - look for labels with minor typos
    for label in VALID_LABELS:
        # Look for substrings that are close to the label
        if label in text_lower:
            return label
    
    # Priority 10: Look at the last line of the text
    # LLMs often put their final answer at the end
    lines = [line.strip() for line in text_lower.split('\n') if line.strip()]
    if lines:
        last_line = lines[-1]
        for label in VALID_LABELS:
            if label in last_line:
                return label
    
    # Priority 11: Look for labels in the last few lines
    if len(lines) >= 2:
        for line in reversed(lines[-3:]):  # Check last 3 lines
            for label in VALID_LABELS:
                if label in line:
                    return label
    
    # NEW Priority 12: Look for standalone labels (just the word on a line)
    for label in VALID_LABELS:
        # Match lines that contain only the label (possibly with quotes)
        standalone_pattern = rf'^\s*["\']?{label}["\']?\s*$'
        for line in lines:
            if re.search(standalone_pattern, line, re.IGNORECASE):
                return label
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract fields from inputs for clearer prompting
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        instruction = f"""You are an expert mathematics grader for International Mathematical Olympiad (IMO) style problems. Your task is to evaluate student answers with the same rigor as an IMO grader.

## YOUR TASK

Read the problem, official solution, grading guidelines, and student answer below. Then classify the student answer into EXACTLY ONE of three categories: "correct", "incorrect", or "partial".

## GRADING CATEGORIES

### "correct" - COMPLETE AND CORRECT SOLUTION
- ALL key mathematical steps are present and logically sound
- The final answer matches the official solution
- No significant gaps or errors in reasoning
- Would receive FULL marks in IMO competition
- BE STRICT: Only use this for truly complete solutions

### "incorrect" - FUNDAMENTALLY FLAWED
- The student has fundamental misunderstandings or critical logical errors
- The approach is wrong, off-track, or demonstrates misconceptions
- The answer shows NO evidence of knowing how to complete the solution
- Would receive 0 or minimal points in IMO grading

### "partial" - MEANINGFUL PROGRESS BUT INCOMPLETE
- The student shows genuine mathematical understanding
- Valid approach has been started with correct key insights
- There are gaps, missing details, or incomplete reasoning
- Would receive partial credit (not 0, not full marks) in IMO

## KEY DECISION RULES

1. Wrong final answer + valid approach = "partial"
2. Wrong final answer + wrong approach = "incorrect"
3. Correct final answer + incomplete reasoning = "partial"
4. Correct final answer + complete reasoning = "correct"
5. Good start + didn't finish = "partial"

## IMO-STYLE FEW-SHOT EXAMPLES

### Example 1: Correct (Complete Proof)
Problem: Prove that there are infinitely many prime numbers.
Official Solution: Assume finitely many primes p₁, ..., pₙ. Consider N = p₁p₂...pₙ + 1. This N is not divisible by any pᵢ, so either N is prime or has a prime factor not in our list. Contradiction.
Student Answer: Suppose there are only finitely many primes: p₁, p₂, ..., pₙ. Let N = p₁ × p₂ × ... × pₙ + 1. When we divide N by any pᵢ, we get remainder 1. So N is either prime itself, or has a prime factor not in our list. Either way, we found a prime not in the original list, contradiction.
Analysis: The student correctly understands the proof by contradiction, constructs N properly, and explains why N isn't divisible by any listed prime. The reasoning is complete and correct.
Verdict: correct

### Example 2: Incorrect (Fundamental Error - Wrong Understanding)
Problem: Find all positive integers n such that n² + n + 41 is prime for all n.
Official Solution: At n = 40: 40² + 40 + 41 = 1681 = 41², which is not prime. So no such n exists (the statement is false).
Student Answer: For n = 1: 1 + 1 + 41 = 43 (prime). For n = 2: 4 + 2 + 41 = 47 (prime). This pattern continues because 41 is prime and n² + n is always even, so n² + n + 41 is always odd and greater than 2, hence prime.
Analysis: The student made a fundamental error - they believe "odd and greater than 2 implies prime" which is false. They don't understand what makes a number prime. They missed the counterexample and made an invalid generalization. This shows a misconception, not just an incomplete solution.
Verdict: incorrect

### Example 3: Incorrect (Wrong Final Answer + Incomplete)
Problem: Find the sum of all positive integers n such that n² - 19n + 99 is a perfect square.
Official Solution: Let n² - 19n + 99 = k². Completing the square: (2n-19)² + 35 = 4k². So 4k² - (2n-19)² = 35. Factor as (2k - (2n-19))(2k + (2n-19)) = 35. The factor pairs of 35 give solutions n = 1, 9, 10, 18. Sum = 38.
Student Answer: Let n² - 19n + 99 = k². Then n² - 19n + (99 - k²) = 0. Using the quadratic formula: n = (19 ± √(361 - 4(99-k²)))/2 = (19 ± √(361 - 396 + 4k²))/2 = (19 ± √(4k² - 35))/2. For n to be integer, 4k² - 35 must be a perfect square. Trying k = 3: 36-35=1, so n = (19±1)/2 = 10 or 9. k = 4: 64-35=29, not square. k = 5: 100-35=65, not square. So the answer is 10 + 9 = 19.
Analysis: The student found two solutions (9 and 10) but missed 1 and 18. The final answer (19) is wrong. They didn't systematically check all possibilities and gave up too early. The approach is partially valid but the execution is incomplete and the final answer is incorrect.
Verdict: incorrect

### Example 4: Partial (Good Progress, Incomplete Proof)
Problem: Prove that for any triangle with sides a, b, c: a² + b² + c² ≥ 4√3 × Area.
Official Solution: Using Heron's formula and AM-GM inequality to show the inequality holds with equality for equilateral triangles.
Student Answer: For an equilateral triangle with side s: Area = (√3/4)s². Then a² + b² + c² = 3s² and 4√3 × Area = 4√3 × (√3/4)s² = 3s². So equality holds for equilateral triangles. For other triangles, the left side grows relative to area.
Analysis: The student correctly verified the equality case and understands the problem. They have the right intuition about the general case but didn't provide a complete proof. They clearly understand what needs to be shown but haven't finished the proof.
Verdict: partial

### Example 5: Partial (Right Approach, Missing Final Calculation)
Problem: Six people sit around a circular table. How many different arrangements are possible if rotations are considered the same?
Official Solution: Fix one person's position to account for rotational symmetry. Arrange remaining 5 linearly. Number of arrangements = 5! = 120.
Student Answer: In circular permutations, we fix one person's position to account for rotational symmetry. Then we arrange the remaining 5 people in the remaining 5 seats. This gives us 5! ways.
Analysis: The student correctly identified the key concept (fixing one person for circular symmetry) and set up the problem properly with 5!. They understand the method but didn't compute the final numerical answer (120). The understanding is clearly present.
Verdict: partial

### Example 6: Incorrect (Completely Wrong Approach)
Problem: Prove that √2 is irrational.
Official Solution: Assume √2 = p/q in lowest terms. Then 2q² = p², so p² is even, so p is even. Let p = 2k. Then 2q² = 4k², so q² = 2k², so q is even. Contradiction since p/q was in lowest terms.
Student Answer: √2 ≈ 1.41421356... The decimal goes on forever without repeating, so √2 cannot be written as a fraction p/q. Therefore it's irrational.
Analysis: The student stated a true fact (non-repeating decimal) but didn't provide a proof. In mathematics, we need logical deduction, not just observation. The student doesn't demonstrate understanding of how to prove irrationality.
Verdict: incorrect

### Example 7: Partial (Correct Key Insight, Missing Details)
Problem: Find all real x such that x³ + 3x² + 3x + 1 = 0.
Official Solution: Recognize (x+1)³ = x³ + 3x² + 3x + 1. So (x+1)³ = 0, giving x = -1 as the only solution.
Student Answer: I notice that x³ + 3x² + 3x + 1 looks like a binomial expansion. This is (x+1)³. So (x+1)³ = 0 means x = -1.
Analysis: The student correctly recognized the binomial pattern and identified the solution. The key insight is present and correct. The solution is essentially complete but could be more explicit about why this is the ONLY solution.
Verdict: partial

### Example 8: Correct (Complete with Slight Variation)
Problem: Solve x² - 5x + 6 = 0.
Official Solution: Factoring: (x-2)(x-3) = 0, so x = 2 or x = 3.
Student Answer: Using the quadratic formula: x = (5 ± √(25-24))/2 = (5 ± 1)/2. So x = 3 or x = 2.
Analysis: The student used a different method (quadratic formula vs factoring) but arrived at the correct answer with correct work. The solution is complete and correct.
Verdict: correct

### Example 9: Incorrect (Wrong Understanding of Problem)
Problem: How many ways to choose 3 people from 10?
Official Solution: C(10,3) = 10!/(3!·7!) = 120.
Student Answer: We can pick the first person in 10 ways, second in 9 ways, third in 8 ways. So 10 × 9 × 8 = 720 ways.
Analysis: The student calculated permutations (ordered) instead of combinations (unordered). They don't understand the difference between choosing a committee (order doesn't matter) versus arranging people in order. This is a fundamental misunderstanding.
Verdict: incorrect

### Example 10: Partial (Good Start, Incomplete)
Problem: Prove by induction that 1 + 2 + ... + n = n(n+1)/2.
Official Solution: Base case n=1: 1 = 1(2)/2 = 1 ✓. Assume true for n=k. Then for n=k+1: 1+...+k+(k+1) = k(k+1)/2 + (k+1) = (k+1)(k/2+1) = (k+1)(k+2)/2. ✓
Student Answer: Base case: n=1, LHS = 1, RHS = 1(2)/2 = 1. Check. Inductive step: Assume 1+...+k = k(k+1)/2. Then for k+1: we add (k+1) to both sides.
Analysis: The student correctly set up the base case and started the inductive step. They understand the structure of proof by induction. However, they didn't complete the algebraic manipulation to show the formula holds for k+1.
Verdict: partial

### Example 11: Partial (Wrong Final Answer but Valid Approach)
Problem: Find all positive integers n such that n² + 3n + 2 is prime.
Official Solution: n² + 3n + 2 = (n+1)(n+2). For this to be prime, one factor must be 1. So n+1 = 1, giving n = 0 (not positive) or n+2 = 1 (impossible for positive n). Thus no such positive integers exist.
Student Answer: n² + 3n + 2 = (n+1)(n+2). If n = 1: (2)(3) = 6, not prime. If n = 2: (3)(4) = 12, not prime. If n = 3: (4)(5) = 20, not prime. It seems like it's never prime for positive n.
Analysis: The student correctly factored the expression and tested several values. They have the right approach (recognizing it's a product) and observed the pattern correctly. They didn't provide a formal proof that no solutions exist, but they clearly understand the problem structure and why there are no solutions. The understanding is present even though the final answer isn't rigorously proven.
Verdict: partial

### Example 12: Incorrect (Wrong Approach Despite Some Correct Work)
Problem: Find the maximum value of x(1-x) for real x.
Official Solution: x(1-x) = x - x² = -(x² - x) = -(x - 1/2)² + 1/4. Maximum is 1/4 at x = 1/2.
Student Answer: x(1-x) = x - x². Taking derivative: 1 - 2x = 0, so x = 1/2. Then x(1-x) = 1/2 × 1/2 = 1/4. But wait, if x = 2, then x(1-x) = 2(-1) = -2. If x = 0, we get 0. So the maximum is 1/4.
Analysis: The student used calculus correctly to find the critical point and computed the value correctly. However, they then got confused and started testing random values unnecessarily. While they got the right answer, the extra confused work shows they don't fully understand why the critical point gives the maximum. This is borderline, but the unnecessary confusion suggests incomplete understanding of the method.
Verdict: incorrect

### Example 13: Partial (Correct Setup, Missing Final Step)
Problem: Find the sum of all positive integers n such that n² - 3n + 2 is a perfect square.
Official Solution: n² - 3n + 2 = (n-1)(n-2). For this to be a perfect square, we need (n-1)(n-2) = k². Let n-2 = m, then (m+1)m = k², so m² + m = k². This gives 4m² + 4m = 4k², so (2m+1)² - 1 = 4k², thus (2m+1)² - 4k² = 1. Let u = 2m+1, then u² - 4k² = 1, or u² - (2k)² = 1. This is a Pell equation. The solutions are u = 1, 3, 17, ... giving m = 0, 1, 8, ... and n = 2, 3, 10, ... The sum is 2 + 3 + 10 = 15.
Student Answer: n² - 3n + 2 = (n-1)(n-2). For this to be a perfect square, we need (n-1)(n-2) = k². Let n-2 = m, then (m+1)m = k². This means m(m+1) is a perfect square. Since m and m+1 are consecutive integers, they are coprime. For their product to be a perfect square, both must be perfect squares. The only consecutive perfect squares are 0 and 1. So m = 0, giving n = 2.
Analysis: The student correctly factored the expression and made a valid observation about consecutive integers. They found one solution (n=2) but missed the other solutions (n=3, n=10, etc.). The approach is valid and shows understanding, but the solution is incomplete. The student didn't explore all cases where m(m+1) could be a perfect square (not just when both factors are perfect squares).
Verdict: partial

### Example 14: Partial (Correct Key Insight, Incomplete Justification)
Problem: Prove that for any positive integer n, the number n³ + 2n is divisible by 3.
Official Solution: Consider n mod 3. If n ≡ 0 (mod 3), then n³ + 2n ≡ 0 + 0 ≡ 0 (mod 3). If n ≡ 1 (mod 3), then n³ + 2n ≡ 1 + 2 ≡ 0 (mod 3). If n ≡ 2 (mod 3), then n³ + 2n ≡ 8 + 4 ≡ 12 ≡ 0 (mod 3). In all cases, n³ + 2n ≡ 0 (mod 3).
Student Answer: We can factor n³ + 2n = n(n² + 2). Now consider cases: if n ≡ 0 (mod 3), then n is divisible by 3, so n(n² + 2) is divisible by 3. If n ≡ 1 (mod 3), then n² ≡ 1 (mod 3), so n² + 2 ≡ 0 (mod 3). If n ≡ 2 (mod 3), then n² ≡ 4 ≡ 1 (mod 3), so n² + 2 ≡ 0 (mod 3). In all cases, the expression is divisible by 3.
Analysis: The student correctly identified the factorization and analyzed all three cases modulo 3. The reasoning is sound and complete. However, they didn't explicitly state the conclusion that n³ + 2n ≡ 0 (mod 3) in each case, though this is implied. The solution is essentially correct but could be more explicit.
Verdict: partial

### Example 15: Incorrect (Fundamental Misunderstanding of Divisibility)
Problem: Prove that for any positive integer n, the number n³ + 2n is divisible by 3.
Official Solution: Consider n mod 3. If n ≡ 0 (mod 3), then n³ + 2n ≡ 0 + 0 ≡ 0 (mod 3). If n ≡ 1 (mod 3), then n³ + 2n ≡ 1 + 2 ≡ 0 (mod 3). If n ≡ 2 (mod 3), then n³ + 2n ≡ 8 + 4 ≡ 12 ≡ 0 (mod 3). In all cases, n³ + 2n ≡ 0 (mod 3).
Student Answer: n³ + 2n = n(n² + 2). For this to be divisible by 3, either n is divisible by 3 or n² + 2 is divisible by 3. If n = 3k, then n is divisible by 3. If n = 3k+1, then n² + 2 = 9k² + 6k + 1 + 2 = 9k² + 6k + 3 = 3(3k² + 2k + 1), which is divisible by 3. If n = 3k+2, then n² + 2 = 9k² + 12k + 4 + 2 = 9k² + 12k + 6 = 3(3k² + 4k + 2), which is divisible by 3. So in all cases, the expression is divisible by 3.
Analysis: The student correctly analyzed all three cases and showed divisibility. The work is complete and correct. Wait - actually this is a complete and correct solution! The student expanded the cases properly and showed the divisibility. This should be "correct" not "partial" or "incorrect". Let me reconsider: the student provided a complete case analysis with proper algebraic expansion. This is a valid complete proof.
Verdict: correct

### Example 16: Partial (Correct Approach, Missing Edge Case)
Problem: Find all integers n such that n² - 1 is divisible by 8.
Official Solution: n² - 1 = (n-1)(n+1). For consecutive even integers (when n is odd), one is divisible by 2 and the other by 4, so product divisible by 8. When n is even, n² is divisible by 4, so n² - 1 ≡ -1 ≡ 3 (mod 4), not divisible by 4, hence not by 8. Thus n must be odd.
Student Answer: n² - 1 = (n-1)(n+1). These are two consecutive even integers when n is odd. One of them is divisible by 2 and the other by 4, so their product is divisible by 8. Therefore, all odd integers n work.
Analysis: The student correctly identified the factorization and the key insight about consecutive even integers. They found that odd n works. However, they didn't prove that even n doesn't work - they only showed that odd n works. The solution is incomplete because it doesn't rule out even n. But the approach is valid and shows understanding.
Verdict: partial

### Example 17: Incorrect (Wrong Method, No Understanding)
Problem: Find all integers n such that n² - 1 is divisible by 8.
Official Solution: n² - 1 = (n-1)(n+1). For consecutive even integers (when n is odd), one is divisible by 2 and the other by 4, so product divisible by 8. When n is even, n² is divisible by 4, so n² - 1 ≡ -1 ≡ 3 (mod 4), not divisible by 4, hence not by 8. Thus n must be odd.
Student Answer: Testing values: n = 1: 1-1 = 0, divisible by 8. n = 2: 4-1 = 3, not divisible. n = 3: 9-1 = 8, divisible. n = 4: 16-1 = 15, not divisible. n = 5: 25-1 = 24, divisible. The pattern is that odd numbers work. So the answer is all odd integers.
Analysis: The student only tested a few values and observed a pattern. They didn't provide any mathematical reasoning for WHY odd integers work or prove that this pattern continues. Testing values is not a proof. The student doesn't demonstrate understanding of the mathematical structure - they just observed a pattern without explanation.
Verdict: incorrect

### Example 18: Partial (Good Start, Missing Completion)
Problem: Prove that if a, b, c are positive integers such that a² + b² = c², then at least one of a, b, c is divisible by 3.
Official Solution: Consider squares mod 3. A square is either 0 or 1 mod 3. If neither a nor b is divisible by 3, then a² ≡ 1 (mod 3) and b² ≡ 1 (mod 3), so a² + b² ≡ 2 (mod 3). But c² ≡ 0 or 1 (mod 3), never 2. Contradiction. So at least one of a, b must be divisible by 3.
Student Answer: Let's look at squares modulo 3. If n ≡ 0 (mod 3), then n² ≡ 0 (mod 3). If n ≡ 1 (mod 3), then n² ≡ 1 (mod 3). If n ≡ 2 (mod 3), then n² ≡ 4 ≡ 1 (mod 3). So squares are either 0 or 1 mod 3. Now if a² + b² = c², and neither a nor b is divisible by 3, then a² ≡ 1 and b² ≡ 1, so a² + b² ≡ 2 (mod 3).
Analysis: The student correctly analyzed squares mod 3 and set up the proof. They identified that if neither a nor b is divisible by 3, then a² + b² ≡ 2 (mod 3). However, they didn't complete the proof by noting that c² can never be 2 (mod 3), which would give the contradiction. The understanding is there, but the proof is incomplete.
Verdict: partial

## CURRENT PROBLEM

PROBLEM:
```
{problem}
```

OFFICIAL SOLUTION:
```
{solution}
```

GRADING GUIDELINES:
```
{grading_guidelines}
```

STUDENT'S ANSWER TO EVALUATE:
```
{student_answer}
```

## CHAIN-OF-THOUGHT ANALYSIS

Analyze this answer step-by-step as an IMO grader would:

### STEP 1: Understanding Check
- Does the student grasp what the problem is asking?
- Do they identify the key mathematical concepts correctly?

### STEP 2: Approach Assessment  
- What approach did the student take?
- Is it mathematically valid and heading in the right direction?

### STEP 3: Key Steps Verification
- Are the critical logical and mathematical steps correct?
- Are there unjustified claims or logical gaps?

### STEP 4: Final Answer Check
- Is the final answer correct? Does it match the official solution?
- **If final answer is wrong**: Check if approach was valid → if yes, "partial"; if no, "incorrect"
- **If final answer is correct but reasoning incomplete**: "partial" (not "correct")
- **If final answer is correct with complete reasoning**: "correct"

### STEP 5: Completeness Evaluation
- Does the answer address ALL parts of the problem?
- Is the proof/solution fully developed or just started?

### STEP 6: THE CRITICAL DECISION - Partial vs Incorrect

**Ask yourself: "Does this student understand how to solve this problem?"**

**Signs the student UNDERSTANDS (→ partial):**
- Correctly identifies what needs to be done
- Starts with a valid, mathematically sound approach
- Shows relevant mathematical knowledge
- Makes genuine progress toward the solution
- Errors are minor, technical, or just incomplete
- **Key test**: If the student had 5 more minutes, could they complete it correctly?

**Signs the student DOESN'T UNDERSTAND (→ incorrect):**
- Wrong approach from the start
- Fundamental misconceptions about the problem
- Invalid mathematical reasoning or wrong theorems
- No meaningful progress toward solution
- **Key test**: Even with more time, would they still be lost?

### STEP 7: Final Classification
- **"correct"**: Complete, correct, all key elements present, would get full marks
- **"incorrect"**: Fundamental flaws, wrong approach, no understanding shown, would get 0
- **"partial"**: Valid approach, good understanding shown, but incomplete, would get partial credit

## OUTPUT FORMAT (CRITICAL - FOLLOW EXACTLY)

After your analysis, you MUST output ONLY a JSON object wrapped in <json> tags. 

The JSON must have exactly this format:
- Key: "response" (with double quotes)
- Value: exactly one of "correct", "incorrect", or "partial" (lowercase, with double quotes)

Example outputs:

<json>
{{
    "response": "correct"
}}
</json>

<json>
{{
    "response": "incorrect"
}}
</json>

<json>
{{
    "response": "partial"
}}
</json>

IMPORTANT:
1. Output ONLY the JSON block wrapped in <json> tags
2. No text before or after the JSON
3. No markdown code blocks (```)
4. Use double quotes for both keys and values
5. The three valid values are: "correct", "incorrect", "partial"

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from response
        prediction = None
        text_to_parse = (response or "").strip()
        
        # Debug logging
        self.log_fn(f"Raw LLM response (first 500 chars): {text_to_parse[:500]}")
        
        try:
            # First try to extract from JSON blocks
            extracted = _extract_jsons(text_to_parse)
            if extracted:
                self.log_fn(f"Extracted {len(extracted)} JSON objects from response")
                for item in extracted:
                    if isinstance(item, dict) and "response" in item:
                        raw_pred = str(item["response"]).strip()
                        self.log_fn(f"Found response field with value: {raw_pred}")
                        # Use normalize function for robust matching
                        normalized = _normalize_label(raw_pred)
                        if normalized:
                            prediction = normalized
                            break
            
            # If JSON extraction failed, try text extraction
            if prediction is None:
                text_pred = _extract_label_from_text(text_to_parse)
                if text_pred:
                    self.log_fn(f"Extracted label from text: {text_pred}")
                    prediction = text_pred
            
            # If still no prediction, check msg_history as last resort
            if prediction is None and msg_history:
                self.log_fn(f"Trying to extract from msg_history ({len(msg_history)} messages)")
                for msg in reversed(msg_history):
                    if isinstance(msg, dict):
                        last_content = msg.get("text") or msg.get("content") or str(msg)
                    else:
                        last_content = str(msg)
                    
                    last_content = last_content.strip()
                    if last_content and last_content != text_to_parse:
                        # Try JSON extraction from history
                        extracted = _extract_jsons(last_content)
                        if extracted:
                            for item in extracted:
                                if isinstance(item, dict) and "response" in item:
                                    raw_pred = str(item["response"]).strip()
                                    normalized = _normalize_label(raw_pred)
                                    if normalized:
                                        prediction = normalized
                                        break
                        
                        if prediction is None:
                            text_pred = _extract_label_from_text(last_content)
                            if text_pred:
                                prediction = text_pred
                                break
                        else:
                            break
            
            # Last resort: look for any valid label in the text with word boundaries
            if prediction is None:
                text_lower = text_to_parse.lower()
                # Look for labels as whole words
                for label in VALID_LABELS:
                    if re.search(rf'\b{label}\b', text_lower):
                        self.log_fn(f"Found label '{label}' as whole word in text")
                        prediction = label
                        break
            
            # Extra fallback: look for common variations or misspellings with confidence scoring
            if prediction is None:
                text_lower = text_to_parse.lower()
                
                # High confidence patterns (explicit statements)
                high_confidence_patterns = [
                    (r'(?:the\s+)?(?:final\s+)?verdict\s*(?:is|:)\s*["\']?incorrect["\']?', 'incorrect'),
                    (r'(?:the\s+)?(?:final\s+)?verdict\s*(?:is|:)\s*["\']?partial["\']?', 'partial'),
                    (r'(?:the\s+)?(?:final\s+)?verdict\s*(?:is|:)\s*["\']?correct["\']?', 'correct'),
                    (r'\bgrade\s*(?:is|:)\s*["\']?incorrect["\']?', 'incorrect'),
                    (r'\bgrade\s*(?:is|:)\s*["\']?partial["\']?', 'partial'),
                    (r'\bgrade\s*(?:is|:)\s*["\']?correct["\']?', 'correct'),
                    (r'(?:decision|assessment|evaluation|rating)\s*(?:is|:)\s*["\']?incorrect["\']?', 'incorrect'),
                    (r'(?:decision|assessment|evaluation|rating)\s*(?:is|:)\s*["\']?partial["\']?', 'partial'),
                    (r'(?:decision|assessment|evaluation|rating)\s*(?:is|:)\s*["\']?correct["\']?', 'correct'),
                    (r'\bclassification\s*(?:is|:)\s*["\']?incorrect["\']?', 'incorrect'),
                    (r'\bclassification\s*(?:is|:)\s*["\']?partial["\']?', 'partial'),
                    (r'\bclassification\s*(?:is|:)\s*["\']?correct["\']?', 'correct'),
                ]
                
                for pattern, label in high_confidence_patterns:
                    if re.search(pattern, text_lower):
                        prediction = label
                        self.log_fn(f"High confidence fallback: detected '{label}' via pattern")
                        break
                
                # Medium confidence patterns (indicator words)
                if prediction is None:
                    # Check for "incorrect" indicators first (higher priority for safety)
                    incorrect_indicators = ['wrong', 'false', 'error', 'invalid', 'flawed', 'fundamental error', 'misunderstanding', 'misconception']
                    partial_indicators = ['incomplete', 'partial credit', 'partially', 'missing', 'unfinished', 'not complete']
                    correct_indicators = ['fully correct', 'completely correct', 'entirely correct', 'perfect solution']
                    
                    for indicator in incorrect_indicators:
                        if indicator in text_lower:
                            prediction = 'incorrect'
                            self.log_fn(f"Medium confidence fallback: detected 'incorrect' via indicator '{indicator}'")
                            break
                    
                    if prediction is None:
                        for indicator in partial_indicators:
                            if indicator in text_lower:
                                prediction = 'partial'
                                self.log_fn(f"Medium confidence fallback: detected 'partial' via indicator '{indicator}'")
                                break
                    
                    if prediction is None:
                        for indicator in correct_indicators:
                            if indicator in text_lower:
                                prediction = 'correct'
                                self.log_fn(f"Medium confidence fallback: detected 'correct' via indicator '{indicator}'")
                                break
            
            # NEW: Look for standalone labels at the very end of the response
            # This handles cases where LLM outputs just the label after chain-of-thought
            if prediction is None:
                lines = [line.strip() for line in text_to_parse.split('\n') if line.strip()]
                if lines:
                    last_line = lines[-1].lower()
                    for label in VALID_LABELS:
                        # Check if the last line is just the label (possibly with quotes or punctuation)
                        if re.search(rf'^["\']?{label}["\']?[.!?]*\s*$', last_line, re.IGNORECASE):
                            prediction = label
                            self.log_fn(f"Found standalone label '{label}' at end of response")
                            break
                            
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            prediction = None
        
        # Final validation - ensure prediction is valid
        if prediction not in VALID_LABELS:
            self.log_fn(f"Invalid prediction '{prediction}', defaulting to 'incorrect'")
            # Default to "incorrect" as the safest fallback
            # This is better than random guessing when we can't determine the label
            prediction = "incorrect"
        else:
            self.log_fn(f"Final prediction: {prediction}")

        return str(prediction), msg_history
