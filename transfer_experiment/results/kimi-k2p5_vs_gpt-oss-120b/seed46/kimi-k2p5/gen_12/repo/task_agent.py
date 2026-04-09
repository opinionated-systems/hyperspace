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
    
    return results if results else None


def _extract_label_from_text(text: str) -> str | None:
    """Extract the grading label from raw text using pattern matching.
    
    Looks for exact matches of the three valid labels with priority-based
    matching to find the most likely intended label.
    Uses multiple strategies with increasing fallback tolerance.
    """
    if not text or not isinstance(text, str):
        return None
        
    text_lower = text.lower().strip()
    
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

## GRADING CATEGORIES - READ CAREFULLY

You MUST classify the answer into EXACTLY ONE of these three categories:

### "correct" - COMPLETE AND CORRECT SOLUTION
- The student's answer is essentially complete and correct
- ALL key mathematical steps are present and logically sound
- The final answer matches the official solution
- No significant gaps or errors in reasoning
- Minor presentation issues or trivial notation differences are OK
- The answer must be a FULL solution, not just a partial one
- When in doubt, it's NOT correct - be STRICT about this category

### "incorrect" - FUNDAMENTALLY FLAWED
- The student has fundamental misunderstandings or critical logical errors
- The approach is wrong, off-track, or demonstrates misconceptions
- Major logical errors invalidate the solution
- The final answer is wrong AND the reasoning is flawed
- The answer shows NO evidence of knowing how to complete the solution
- The student made invalid generalizations or fatal mathematical errors
- Key distinction: The student does NOT understand how to solve the problem

### "partial" - MEANINGFUL PROGRESS BUT INCOMPLETE
- The student shows genuine mathematical understanding
- Valid approach has been started with correct key insights
- The student demonstrates they understand the problem structure
- There are gaps, missing details, or incomplete reasoning
- Minor errors that don't invalidate the main approach
- The student is "on the right track" but hasn't finished
- Key distinction: The student DOES understand how to solve the problem
- USE THIS GENEROUSLY when the student demonstrates competence

## CRITICAL DISTINCTION: PARTIAL vs INCORRECT

This is the MOST IMPORTANT decision. Ask yourself these questions:

1. **Does the student UNDERSTAND the problem?**
   - Can they identify what needs to be solved? 
   - Do they know the relevant mathematical concepts?
   - YES → partial, NO → incorrect

2. **Is the approach VALID?**
   - Did they start with a correct strategy?
   - Are they heading in the right direction?
   - YES → partial, NO → incorrect

3. **What would a tutor say?**
   - "You're on the right track, just finish it" → partial
   - "You need to restart with a different approach" → incorrect

4. **Nature of errors:**
   - Minor technical errors, missing final step → partial
   - Fundamental misconceptions, wrong approach → incorrect

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

1. **Understanding Check**: Does the student grasp what the problem is asking? Do they identify the key mathematical concepts correctly?

2. **Approach Assessment**: What approach did the student take? Is it valid? Are they heading in the right direction?

3. **Key Steps Verification**: Are the critical logical and mathematical steps correct? Look for:
   - Correct theorems or formulas applied
   - Valid logical deductions
   - Proper use of given information
   - No unjustified claims

4. **Final Answer Check**: Is the final answer correct? Does it match the official solution?
   - If final answer is wrong → likely "incorrect"
   - If final answer is correct but reasoning incomplete → could be "partial"
   - If final answer is correct with complete reasoning → "correct"

5. **Completeness Evaluation**: Does the answer address ALL parts of the problem?
   - Missing key components → not "correct"
   - Some progress shown → consider "partial"
   - No meaningful progress → "incorrect"

6. **Grading Guidelines Alignment**: How does this map to the specific criteria provided?

7. **THE CRITICAL DECISION - Partial vs Incorrect**:
   
   Ask yourself: "Does this student understand how to solve this problem?"
   
   Signs the student UNDERSTANDS (→ partial):
   - Correctly identifies what needs to be done
   - Starts with a valid approach
   - Shows relevant mathematical knowledge
   - Makes progress toward the solution
   - Errors are minor or technical
   
   Signs the student DOESN'T UNDERSTAND (→ incorrect):
   - Wrong approach from the start
   - Fundamental misconceptions
   - Invalid mathematical reasoning
   - No meaningful progress
   - Would need to restart completely

8. **Final Classification**:
   - "correct": Complete, correct, all key elements present
   - "incorrect": Fundamental flaws, wrong approach, or no understanding shown
   - "partial": Valid approach, good understanding, but incomplete

## CRITICAL INSTRUCTIONS - READ CAREFULLY

1. You MUST output ONLY a JSON object in the exact format shown below.
2. Do NOT include any text before or after the JSON.
3. Do NOT include markdown formatting (like ```json) around the JSON.
4. The "response" field MUST contain exactly one of: "correct", "incorrect", or "partial" (lowercase, with quotes around the value).
5. Be STRICT about "correct" - incomplete solutions are NOT "correct".
6. "partial" is for answers that show real understanding and valid approach - use this GENEROUSLY when the student demonstrates competence.
7. "incorrect" is for fundamental errors, wrong approaches, or when the student shows no understanding of how to solve the problem.
8. When in doubt between "incorrect" and "partial", ask: "Does this student understand how to solve the problem?" If YES → partial, if NO → incorrect.

## OUTPUT FORMAT (STRICT - ONLY OUTPUT THIS JSON, NOTHING ELSE)

<json>
{{
    "response": "correct"
}}
</json>

OR

<json>
{{
    "response": "incorrect"
}}
</json>

OR

<json>
{{
    "response": "partial"
}}
</json>

Remember: ONLY output the JSON block above. No other text. The JSON must be valid with double quotes around both keys and values."""

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
                        raw_pred = str(item["response"]).strip().lower()
                        self.log_fn(f"Found response field with value: {raw_pred}")
                        # Direct match
                        if raw_pred in VALID_LABELS:
                            prediction = raw_pred
                            break
                        # Handle cases where the value might have extra punctuation
                        clean_pred = re.sub(r'[^a-z]', '', raw_pred)
                        if clean_pred in VALID_LABELS:
                            prediction = clean_pred
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
                                    raw_pred = str(item["response"]).strip().lower()
                                    if raw_pred in VALID_LABELS:
                                        prediction = raw_pred
                                        break
                                    clean_pred = re.sub(r'[^a-z]', '', raw_pred)
                                    if clean_pred in VALID_LABELS:
                                        prediction = clean_pred
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
            
            # Extra fallback: look for common variations or misspellings
            if prediction is None:
                text_lower = text_to_parse.lower()
                # Common variations - be careful not to over-predict "correct"
                if 'wrong' in text_lower or 'false' in text_lower or 'error' in text_lower or 'invalid' in text_lower:
                    prediction = 'incorrect'
                    self.log_fn("Fallback: detected 'incorrect' indicators")
                elif 'incomplete' in text_lower or 'partial credit' in text_lower or 'partially' in text_lower:
                    prediction = 'partial'
                    self.log_fn("Fallback: detected 'partial' indicators")
                # Look for explicit verdict statements in the text
                elif re.search(r'(?:the\s+)?(?:final\s+)?verdict\s*(?:is|:)\s*["\']?incorrect["\']?', text_lower):
                    prediction = 'incorrect'
                    self.log_fn("Fallback: detected 'verdict: incorrect'")
                elif re.search(r'(?:the\s+)?(?:final\s+)?verdict\s*(?:is|:)\s*["\']?partial["\']?', text_lower):
                    prediction = 'partial'
                    self.log_fn("Fallback: detected 'verdict: partial'")
                elif re.search(r'(?:the\s+)?(?:final\s+)?verdict\s*(?:is|:)\s*["\']?correct["\']?', text_lower):
                    prediction = 'correct'
                    self.log_fn("Fallback: detected 'verdict: correct'")
                # Additional patterns for explicit statements
                elif re.search(r'\bgrade\s*(?:is|:)\s*["\']?incorrect["\']?', text_lower):
                    prediction = 'incorrect'
                    self.log_fn("Fallback: detected 'grade: incorrect'")
                elif re.search(r'\bgrade\s*(?:is|:)\s*["\']?partial["\']?', text_lower):
                    prediction = 'partial'
                    self.log_fn("Fallback: detected 'grade: partial'")
                elif re.search(r'\bgrade\s*(?:is|:)\s*["\']?correct["\']?', text_lower):
                    prediction = 'correct'
                    self.log_fn("Fallback: detected 'grade: correct'")
                # Look for decision statements
                elif re.search(r'(?:decision|assessment|evaluation|rating)\s*(?:is|:)\s*["\']?incorrect["\']?', text_lower):
                    prediction = 'incorrect'
                    self.log_fn("Fallback: detected decision: incorrect")
                elif re.search(r'(?:decision|assessment|evaluation|rating)\s*(?:is|:)\s*["\']?partial["\']?', text_lower):
                    prediction = 'partial'
                    self.log_fn("Fallback: detected decision: partial")
                elif re.search(r'(?:decision|assessment|evaluation|rating)\s*(?:is|:)\s*["\']?correct["\']?', text_lower):
                    prediction = 'correct'
                    self.log_fn("Fallback: detected decision: correct")
                            
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
