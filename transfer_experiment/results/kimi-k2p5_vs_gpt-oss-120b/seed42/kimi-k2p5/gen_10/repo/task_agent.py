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


def _normalize_prediction(prediction: str) -> str | None:
    """Normalize a prediction string to one of the allowed categories.
    
    Handles common variations and misspellings of category names.
    Uses a priority-based approach to handle overlapping terms correctly.
    """
    if not prediction:
        return None
    
    # Normalize the input
    pred_lower = prediction.lower().strip()
    
    # Remove common prefixes/suffixes that might interfere
    pred_clean = re.sub(r'^(the\s+|a\s+|an\s+|is\s+|it\s+|this\s+|that\s+)', '', pred_lower)
    pred_clean = re.sub(r'\s+(answer|classification|category|result|grade)$', '', pred_clean)
    
    # Check for exact matches first (most reliable)
    allowed_categories = ["correct", "incorrect", "partial", "almost"]
    if pred_clean in allowed_categories:
        return pred_clean.capitalize()
    
    # Check for exact match with word boundaries (e.g., "The answer is Correct")
    for cat in allowed_categories:
        if re.search(rf'\b{cat}\b', pred_clean):
            return cat.capitalize()
    
    # Map common variations to standard categories
    # Be careful with overlapping terms - use word boundaries
    # Order matters: check more specific patterns first
    
    # Check for "almost" first (more specific than "correct")
    # Use word boundaries to avoid matching "almost" inside other words
    almost_variations = [
        r'\balmost\b', r'\balmost correct\b', r'\bnearly correct\b', 
        r'\bclose\b', r'\bminor\s+(?:error|mistake|issue)\b', r'\btrivial\b', r'\bsmall error\b', 
        r'\bminor mistake\b', r'\bslight\s+(?:error|mistake)\b', r'\btiny error\b',
        r'\bessentially correct\b', r'\bmostly correct\b', r'\bnearly right\b',
        r'\bnearly complete\b', r'\bessentially complete\b', r'\bminor\s+issue',
        r'\btiny\s+(?:error|mistake|issue)\b', r'\bsmall\s+(?:error|mistake|issue)\b',
        r'\bnegligible\b', r'\binsignificant\s+(?:error|mistake|issue)\b'
    ]
    for var in almost_variations:
        if re.search(var, pred_lower):
            return "Almost"
    
    # Check for "partial" (more specific than "incorrect")
    partial_variations = [
        r'\bpartial\b', r'\bpartly\b', r'\bpartially\b', r'\bincomplete\b', 
        r'\bsome progress\b', r'\bhalf\b', r'\bmissing\b', r'\bon the right track\b',
        r'\bgood start\b', r'\bsignificant progress\b', r'\bnot complete\b',
        r'\bunfinished\b', r'\bgood approach\b', r'\bcorrect direction\b',
        r'\bin progress\b', r'\bpartial solution\b', r'\bpartial proof\b',
        r'\bsome\s+(?:steps|work|progress)\b', r'\bstarted\s+(?:well|correctly)\b'
    ]
    for var in partial_variations:
        if re.search(var, pred_lower):
            return "Partial"
    
    # Check for "incorrect"
    incorrect_variations = [
        r'\bincorrect\b', r'\bwrong\b', r'\bfalse\b', r'\binvalid\b', 
        r'\berror\b', r'\bmistake\b', r'\bflawed\b', r'\bfundamental\b',
        r'\bunsalvageable\b', r'\bbroken\b', r'\bnot correct\b',
        r'\bdoes not work\b', r'\bfails\b', r'\bflawed approach\b',
        r'\bwrong approach\b', r'\bincorrect approach\b', r'\bnot valid\b',
        r'\bnot true\b', r'\bfalse\s+(?:statement|claim|conclusion)\b'
    ]
    for var in incorrect_variations:
        if re.search(var, pred_lower):
            return "Incorrect"
    
    # Check for "correct" last (most general)
    correct_variations = [
        r'\bcorrect\b', r'\bright\b', r'\btrue\b', r'\bvalid\b', 
        r'\baccurate\b', r'\bperfect\b', r'\bcomplete\b', r'\bsound\b',
        r'\bwell done\b', r'\bexcellent\b', r'\bgood\b', r'\bproper\b',
        r'\bvalid proof\b', r'\bcorrect proof\b', r'\bcomplete proof\b'
    ]
    for var in correct_variations:
        if re.search(var, pred_lower):
            return "Correct"
    
    return None


def _extract_response_flexible(text: str) -> str | None:
    """Extract response using multiple fallback strategies.
    
    Tries multiple patterns to find the classification:
    1. JSON format with "response" field
    2. JSON format with "classification" field
    3. Direct mention of categories in text
    4. Look for explicit grading statements in the analysis
    5. Markdown code block JSON extraction
    """
    text_lower = text.lower()
    
    # Try JSON extraction first (from <json> tags)
    json_results = _extract_jsons(text)
    if json_results:
        for result in json_results:
            if isinstance(result, dict):
                # Check for common field names
                for key in ["response", "classification", "answer", "result", "grade", "evaluation", "verdict", "category"]:
                    if key in result:
                        val = result[key]
                        if isinstance(val, str):
                            normalized = _normalize_prediction(val.strip())
                            if normalized:
                                return normalized
                        elif isinstance(val, bool):
                            # Handle boolean values
                            return "Correct" if val else "Incorrect"
    
    # Try to find JSON in markdown code blocks (handle nested braces better)
    markdown_pattern = r'```(?:json)?\s*\n?(\{[\s\S]*?\})\n?```'
    for match in re.finditer(markdown_pattern, text, re.DOTALL):
        try:
            json_obj = json.loads(match.group(1))
            if isinstance(json_obj, dict):
                for key in ["response", "classification", "answer", "result", "grade", "evaluation", "verdict", "category"]:
                    if key in json_obj:
                        val = json_obj[key]
                        if isinstance(val, str):
                            normalized = _normalize_prediction(val.strip())
                            if normalized:
                                return normalized
        except json.JSONDecodeError:
            continue
    
    # Try to find JSON-like patterns without tags (single line)
    json_pattern = re.search(r'\{\s*"(?:response|classification|answer|result|grade|evaluation|verdict|category)"\s*:\s*"([^"]+)"\s*\}', text, re.IGNORECASE)
    if json_pattern:
        normalized = _normalize_prediction(json_pattern.group(1).strip())
        if normalized:
            return normalized
    
    # Look for explicit grading statements with more specific patterns
    # These patterns look for the final classification decision in the text
    # Order matters: more specific patterns first
    grading_patterns = [
        # Pattern: "Classification: Almost" - specific category mentions
        r'(?:the\s+)?(?:final\s+)?(?:classification|grade|category|result|evaluation|verdict)\s*[:=]\s*(almost|partial|incorrect|correct)',
        # Pattern: "I classify this as: Almost"
        r'(?:i\s+)?(?:classify|grade|rate|evaluate)\s+(?:this|the\s+(?:answer|solution|response))\s+as\s*(almost|partial|incorrect|correct)',
        # Pattern: "This is classified as: Almost"
        r'(?:this|the\s+(?:answer|solution|response))\s+is\s*(?:therefore\s*)?(?:classified\s+as|graded\s+as)\s*(almost|partial|incorrect|correct)',
        # Pattern: "Therefore, the answer is Almost"
        r'therefore[,;]?\s+(?:the\s+)?(?:answer|classification|result|evaluation|verdict)\s+is\s*(almost|partial|incorrect|correct)',
        # Pattern: "In conclusion, the answer is Almost"
        r'in\s+conclusion[,;]?\s+(?:the\s+)?(?:answer|classification|result|evaluation|verdict)\s+is\s*(almost|partial|incorrect|correct)',
        # Pattern: "The answer is: Almost"
        r'(?:the\s+)?(?:student\s+)?(?:answer|solution|response)\s+is\s*[:=]?\s*(almost|partial|incorrect|correct)',
        # Pattern: "I would classify this as Almost"
        r'i\s+would\s+(?:classify|grade|rate|evaluate)\s+(?:this|it)\s+as\s*(almost|partial|incorrect|correct)',
        # Pattern: "This should be classified as Almost"
        r'this\s+should\s+be\s+(?:classified|graded|rated)\s+as\s*(almost|partial|incorrect|correct)',
        # Pattern: "The appropriate classification is Almost"
        r'the\s+(?:appropriate|correct|proper)\s+(?:classification|grade|category)\s+is\s*(almost|partial|incorrect|correct)',
        # Pattern: "I rate this as Almost"
        r'i\s+(?:rate|judge|assess)\s+(?:this|it|the\s+(?:answer|solution|response))\s+as\s*(almost|partial|incorrect|correct)',
        # Pattern: "This gets a grade of Almost"
        r'(?:this|it)\s+(?:gets?|receives?)\s+(?:a\s+)?(?:grade|score|mark|classification)\s+of\s*(almost|partial|incorrect|correct)',
    ]
    
    for pattern in grading_patterns:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            return match.group(1).capitalize()
    
    # Look for standalone category mentions at the end of sentences or lines
    # This helps catch cases where the model just says "Partial." or "Almost"
    standalone_patterns = [
        # Category at start of line
        r'(?:^|\n)\s*(almost|partial|incorrect|correct)\s*[.!?]?\s*(?:$|\n)',
        # Category with colon
        r'(?:classification|grade|result)\s*[:\-]\s*(almost|partial|incorrect|correct)',
        # Bold markdown
        r'\*\*(almost|partial|incorrect|correct)\*\*',
        # Bold markdown with spaces
        r'\*\*\s*(almost|partial|incorrect|correct)\s*\*\*',
        # Category with explanation in parentheses
        r'\b(almost|partial|incorrect|correct)\b\s*\([^)]*\)',
        # Quoted category
        r'["\'](almost|partial|incorrect|correct)["\']',
        # Category at end of text
        r'\b(almost|partial|incorrect|correct)\b[.!?]?\s*$',
    ]
    for pattern in standalone_patterns:
        match = re.search(pattern, text_lower, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).capitalize()
    
    # Look for direct category mentions with word boundaries
    # Check in order of specificity: Almost, Partial, Incorrect, Correct
    categories = ["Almost", "Partial", "Incorrect", "Correct"]
    text_upper = text.upper()
    
    # Check for explicit category statements with more context
    for category in categories:
        # Look for patterns like "The answer is: Correct" or "Classification: Partial"
        patterns = [
            rf'\b{category.upper()}\b',
            rf'(?:is|are|be)\s*[:\-]?\s*{category.upper()}',
            rf'(?:classification|category|grade|result|answer|evaluation|verdict)\s*[:\-]?\s*{category.upper()}',
        ]
        for pattern in patterns:
            if re.search(pattern, text_upper):
                return category
    
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
        # Extract fields from inputs
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        instruction = f"""You are an expert mathematics grader for International Mathematical Olympiad (IMO) problems. Your task is to evaluate a student's answer against the official solution and grading guidelines.

## Problem Statement:
```
{problem}
```

## Official Solution:
```
{solution}
```

## Grading Guidelines:
```
{grading_guidelines}
```

## Student Answer:
```
{student_answer}
```

## Classification Categories:

1. **Correct**: The student answer is complete and correct, matching the official solution. All key steps are present and logically sound. The proof is rigorous and complete. NO errors, NO missing steps, NO gaps.

2. **Incorrect**: The student answer is wrong or contains fundamental errors that invalidate the solution. The approach is flawed, the conclusion is wrong, or there are critical logical gaps that cannot be fixed. The student does NOT demonstrate understanding of the correct approach.

3. **Partial**: The student made significant progress but the solution is incomplete or has major gaps. Some key ideas are present but critical steps are missing. The student demonstrates understanding of the core concepts and correct approach, but hasn't completed the proof. This is for solutions that need SUBSTANTIAL additional work.

4. **Almost**: The student answer is NEARLY PERFECT with only TRIVIAL, MINOR mistakes. The solution is ESSENTIALLY COMPLETE - the core logic is sound, all major steps are present, and the proof structure is valid. Only tiny fixes needed (small calculation error, minor notation issue, one small edge case). If the fix takes less than 30 seconds to explain, it's Almost.

## Key Distinctions (CRITICAL - READ CAREFULLY):

**ALMOST vs PARTIAL** (THE HARDEST DISTINCTION - READ THIS CAREFULLY):

**ALMOST** means: "This is 90-99% complete. The hard work is done. Just needs a tiny fix."
- The proof structure is complete and valid
- All major logical steps are present
- The student clearly understands the solution
- Error is TRIVIAL: small arithmetic mistake, typo in formula, minor notation confusion
- Examples: 2+2=5, wrote n(n-1)/2 instead of n(n+1)/2, forgot to check n=0 case

**PARTIAL** means: "Good start, but significant work remains. Major gaps exist."
- Missing critical lemmas or proof components
- Incomplete case analysis
- Only proved one direction of an iff statement
- Has the right idea but didn't execute the proof
- Would need 5+ minutes of explanation to complete

**DECISION RULE for Almost vs Partial:**
- Count the number of missing/incorrect elements
- If 1-2 TINY issues → Almost
- If 3+ issues OR any MAJOR gap → Partial

**INCORRECT vs PARTIAL** (CRITICAL DISTINCTION):
- **Incorrect**: Fundamental flaws, wrong approach, conclusion is wrong. The student does NOT demonstrate correct understanding.
  - Key test: "Does the student know the RIGHT way to solve this?" If NO → Incorrect
  - Examples: Wrong method entirely, circular reasoning, only checking examples, claiming false is true
  
- **Partial**: Good approach, correct direction, on the right track. The student DOES demonstrate understanding.
  - Key test: "Does the student know the RIGHT way to solve this?" If YES → Partial
  - Examples: Correct key concepts but proof incomplete, right idea but missing steps

**CORRECT vs ALMOST** (IMPORTANT):
- **Correct**: ZERO errors, ZERO gaps, ZERO issues. Perfect solution.
- **Almost**: Has at least one minor issue, even if tiny. Not perfect.

**WHEN IN DOUBT:**
- Between Partial and Incorrect: "Did they identify the correct key concepts?" Yes→Partial, No→Incorrect
- Between Almost and Partial: "How much work remains?" Trivial fix→Almost, Substantial work→Partial
- Between Correct and Almost: "Is it PERFECT?" Any issue→Almost, No issues→Correct

## Detailed Examples:

**Example 1 - Correct:**
Problem: Prove that the sum of two even numbers is even.
Student: Let a = 2m and b = 2n for integers m, n. Then a + b = 2m + 2n = 2(m+n), which is even.
Classification: Correct (complete proof with proper reasoning)

**Example 2 - Incorrect (Fundamental error):**
Problem: Prove that the sum of two even numbers is even.
Student: Even numbers are divisible by 2, so their sum is divisible by 4.
Classification: Incorrect (fundamental error - sum of two evens is not necessarily divisible by 4)

**Example 3 - Incorrect (Wrong approach):**
Problem: Prove that √2 is irrational.
Student: √2 ≈ 1.414, which is not an integer, so it's irrational.
Classification: Incorrect (completely wrong approach - approximation doesn't prove irrationality)

**Example 4 - Partial (Major gaps - missing synthesis):**
Problem: Prove that for any prime p > 3, p² ≡ 1 (mod 24).
Student: Any prime p > 3 is odd, so p² ≡ 1 (mod 8). Also, p is not divisible by 3, so p² ≡ 1 (mod 3).
Classification: Partial (correctly identified key facts but didn't combine them using CRT to get mod 24 - missing the final synthesis)

**Example 5 - Partial (Incomplete proof):**
Problem: Prove that every integer n > 1 has a prime factor.
Student: If n is prime, we're done. If n is composite, it has factors.
Classification: Partial (started case analysis but didn't complete the proof for composite case - no induction or descent argument)

**Example 6 - Partial (Good direction, incomplete):**
Problem: Prove by induction that 1 + 2 + ... + n = n(n+1)/2.
Student: Base case: 1 = 1(2)/2 = 1 ✓. For the inductive step, assume true for n=k.
Classification: Partial (correctly set up induction but didn't complete the inductive step - missing the algebra to show it holds for k+1)

**Example 7 - Partial (Missing key lemma):**
Problem: Prove that the angle bisectors of a triangle meet at a single point.
Student: Let the bisectors of angles A and B meet at point I. Then I is equidistant from all three sides.
Classification: Partial (correctly identified the incenter but didn't prove that the third bisector also passes through I - missing the key lemma that I lies on the third bisector)

**Example 8 - Almost (Minor arithmetic error):**
Problem: Find the sum of integers from 1 to 100.
Student: Using the formula n(n+1)/2 with n=100: 100×101/2 = 5051.
Classification: Almost (correct approach and formula, but arithmetic error: 100×101/2 = 5050, not 5051)

**Example 9 - Almost (Missing edge case):**
Problem: Find all positive integer solutions to x² - y² = 1.
Student: (x-y)(x+y) = 1, so x-y = 1 and x+y = 1, giving x=1, y=0. But y must be positive, so no solutions.
Classification: Almost (correctly analyzed the factorization but missed that x-y and x+y could both be -1, giving x=-1, y=0 - though still no positive solutions, the reasoning was incomplete)

**Example 10 - Almost (Minor notation issue):**
Problem: Prove the Pythagorean theorem.
Student: [Correct proof using similar triangles, but uses a and b for both legs and hypotenuses in different triangles without clear distinction]
Classification: Almost (mathematically correct but notation could be clearer - doesn't affect validity)

**Example 11 - Almost (Small calculation error in proof):**
Problem: Prove that the area of a triangle with sides 3, 4, 5 is 6.
Student: Using Heron's formula: s = (3+4+5)/2 = 5. Area = √(5(5-3)(5-4)(5-5)) = √0 = 0.
Classification: Almost (correct formula but calculation error: s = 6, not 5. Area should be √(6×3×2×1) = 6)

**Example 12 - Partial (NOT Incorrect - shows good understanding):**
Problem: Prove that n³ - n is divisible by 6 for all integers n.
Student: n³ - n = n(n²-1) = n(n-1)(n+1). This is a product of three consecutive integers. Among any three consecutive integers, one is divisible by 2 and one is divisible by 3.
Classification: Partial (correctly factored and identified the key property, but didn't explicitly complete the proof that the product is divisible by 6 - needs to formally show it's divisible by both 2 and 3)

**Example 13 - Incorrect (NOT Partial - wrong approach):**
Problem: Prove that n³ - n is divisible by 6 for all integers n.
Student: n³ - n = n(n-1)(n+1). For n=2, this is 2×1×3 = 6, which is divisible by 6. For n=3, it's 3×2×4 = 24, divisible by 6. For n=4, it's 4×3×5 = 60, divisible by 6. So it's true.
Classification: Incorrect (only checked specific cases, didn't prove for all n - this is verification by example, not a proof)

**Example 14 - Partial vs Incorrect distinction:**
Problem: Find all pairs of positive integers (x,y) such that a sequence has a limit.
Student: [Provides detailed case analysis, correctly identifies that (1,1) is a solution, correctly shows other cases don't work through number theory arguments, but has a small gap in one case analysis sub-argument]
Classification: Partial (shows excellent understanding of the problem and correct approach, makes significant progress with rigorous reasoning, but has a gap in completeness)

**Example 15 - Incorrect (misleading progress):**
Problem: Find all pairs of positive integers (x,y) such that a sequence has a limit.
Student: [Writes down the formula, checks a few examples, makes some algebraic manipulations, but never actually establishes what the answer should be or proves anything conclusive]
Classification: Incorrect (doesn't demonstrate understanding of the correct solution approach - just manipulates formulas without direction)

**Example 16 - Almost (Typo in formula):**
Problem: Find the sum of first n odd numbers.
Student: The sum is n² + 1. For n=1: 1²+1=2, but should be 1. For n=2: 2²+1=5, but should be 1+3=4.
Classification: Almost (correct approach - recognized pattern is n², but made typo writing n²+1 instead of n². One character error.)

**Example 17 - Almost (Forgot one edge case):**
Problem: Prove that for all n ≥ 1, the Fibonacci number F_n ≤ 2^n.
Student: Base case: F_1 = 1 ≤ 2^1 = 2 ✓. Inductive step: Assume F_k ≤ 2^k and F_{k-1} ≤ 2^{k-1}. Then F_{k+1} = F_k + F_{k-1} ≤ 2^k + 2^{k-1} < 2^k + 2^k = 2^{k+1}.
Classification: Almost (proof is correct and complete for the inductive step, but only checked base case n=1, not n=2. Missing one base case out of the induction.)

**Example 18 - Partial (Missing key step, not Almost):**
Problem: Prove that for all n ≥ 1, the Fibonacci number F_n ≤ 2^n.
Student: We use induction. Base case: F_1 = 1 ≤ 2. Inductive step: Assume it holds for all k ≤ n.
Classification: Partial (started induction correctly but completely missing the inductive step reasoning - no algebra shown, no conclusion drawn. This is a MAJOR gap, not a tiny fix.)

**Example 19 - Almost vs Partial distinction:**
Problem: Prove that the product of two consecutive integers is even.
Student: Let the integers be n and n+1. If n is even, n = 2k, so n(n+1) = 2k(n+1) which is even. If n is odd, n+1 is even, so n+1 = 2k, so n(n+1) = n(2k) = 2(nk) which is even. So the product is always even. QED
Classification: Almost (complete proof with both cases covered, logic is sound. Only issue: "QED" is informal notation, but proof is mathematically complete.)

**Example 20 - Partial (Incomplete case analysis):**
Problem: Prove that the product of two consecutive integers is even.
Student: Let the integers be n and n+1. If n is even, n = 2k, so n(n+1) = 2k(n+1) which is even.
Classification: Partial (only proved one case! Missing the case where n is odd. This is a MAJOR gap - half the proof is missing.)

**Example 21 - Almost (Minor sign error):**
Problem: Solve x² - 5x + 6 = 0.
Student: Using quadratic formula: x = (5 ± √(25-24))/2 = (5 ± 1)/2. So x = 3 or x = 2.
Classification: Almost (wait, that's actually correct... let me recheck... Actually this is CORRECT. But if student wrote x = (-5 ± √1)/2 = -3 or -2, that would be Almost - sign error in formula but correct method.)

**Example 22 - Almost (Specific - Sign error):**
Problem: Solve x² - 5x + 6 = 0.
Student: x = (-5 ± √(25+24))/2 = (-5 ± 7)/2. So x = 1 or x = -6.
Classification: Almost (used wrong sign in discriminant: should be -24 not +24. Correct method, one sign error.)

**Example 23 - Incorrect (Wrong formula entirely):**
Problem: Solve x² - 5x + 6 = 0.
Student: x = -b/a = 5/1 = 5.
Classification: Incorrect (completely wrong formula - that's not the quadratic formula or any valid solution method.)

**Example 24 - Partial (Right idea, incomplete execution):**
Problem: Prove that if n² is even, then n is even.
Student: Suppose n² is even. Then n² = 2k for some integer k. So n = √(2k).
Classification: Partial (correctly started with definition of even, but got stuck at the square root. Has the right idea about parity but didn't complete the proof. Needs to use contrapositive or prime factorization.)

**Example 25 - Almost (Notation confusion, logic correct):**
Problem: Prove that if n² is even, then n is even.
Student: Assume n² = 2k. If n were odd, n = 2m+1, then n² = 4m²+4m+1 = 2(2m²+2m)+1 which is odd. Contradiction. So n is even.
Classification: Almost (proof is logically complete and correct! Only issue: used k in first line but m in second - inconsistent variable naming. Logic is perfect.)

**Example 26 - Partial (Missing conclusion):**
Problem: Prove that if n² is even, then n is even.
Student: Assume n² is even, so n² = 2k. Assume n is odd, so n = 2m+1. Then n² = 4m²+4m+1.
Classification: Partial (set up proof by contradiction correctly, computed n² for odd n, but never compared to show it's odd, never drew conclusion. Missing the final steps.)

**Example 27 - Almost (One number wrong in calculation):**
Problem: Find the remainder when 2^100 is divided by 3.
Student: 2 ≡ -1 (mod 3), so 2^100 ≡ (-1)^100 = 1 (mod 3). The remainder is 1.
Classification: Correct (this is actually correct!)

**Example 28 - Almost (Small arithmetic error):**
Problem: Find the remainder when 2^100 is divided by 5.
Student: 2^4 = 16 ≡ 1 (mod 5). So 2^100 = (2^4)^25 ≡ 1^20 = 1 (mod 5). Wait, 100/4 = 25, so 1^25 = 1. The remainder is 1.
Classification: Almost (correct method using Euler's theorem, but wrote 1^20 instead of 1^25 momentarily. Caught and fixed. If they didn't catch it: Almost.)

**Example 29 - Partial (Incomplete pattern recognition):**
Problem: Find the remainder when 2^100 is divided by 5.
Student: 2^1 = 2, 2^2 = 4, 2^3 = 8 ≡ 3, 2^4 = 16 ≡ 1, 2^5 = 32 ≡ 2... I see a pattern.
Classification: Partial (correctly identified the cyclic pattern but didn't use it to compute the final answer. Missing the application step.)

**Example 30 - Incorrect (Wrong pattern):**
Problem: Find the remainder when 2^100 is divided by 5.
Student: Powers of 2 always end in 2, 4, 8, 6. So 2^100 ends in 6. The remainder is 6.
Classification: Incorrect (confused last digit with remainder mod 5. 6 mod 5 = 1, not 6. Wrong conclusion from correct observation.)

## Analysis Steps (FOLLOW THESE IN ORDER):

**STEP 1: Check for Correctness**
- Does the solution match the official solution exactly?
- Are ALL steps present and correct?
- Is the logic 100% sound with NO errors?
- If YES to all → **Correct**
- If there's ANY error or gap, continue to Step 2

**STEP 2: Check if Incorrect**
- Is the approach fundamentally wrong?
- Does the student fail to demonstrate understanding of the correct method?
- Is there circular reasoning, wrong method, or verification by example?
- If YES to any → **Incorrect**
- If the approach is correct but incomplete, continue to Step 3

**STEP 3: Distinguish Almost vs Partial (THE CRITICAL STEP)**
- Count the issues: How many things are wrong or missing?
- Assess the severity: Are they tiny (typo) or major (missing proof)?
- **ALMOST criteria** (ALL must be true):
  * The proof structure is complete
  * All major logical steps are present
  * Only 1-2 tiny issues (arithmetic error, typo, minor notation)
  * Fix would take <30 seconds to explain
  * If YES → **Almost**
- **PARTIAL criteria** (ANY of these):
  * Missing critical lemmas or proof components
  * Incomplete case analysis
  * 3+ issues OR any major gap
  * Would need 5+ minutes to complete
  * If YES → **Partial**

**STEP 4: Final Verification**
- Re-read the definitions above
- Does your classification match ALL the criteria?
- When uncertain, re-examine the examples provided

**QUICK DECISION TREE:**
```
Is it perfect? → Correct
Is the approach wrong? → Incorrect
Is it 90-99% complete with tiny fix needed? → Almost
Is it 50-80% complete with major gaps? → Partial
```

## Self-Reflection Step (CRITICAL - DO THIS BEFORE RESPONDING):
Before finalizing your classification, ask yourself these verification questions:

1. **If you chose Correct**: Is there truly ZERO errors or issues? Even a tiny typo means it's Almost, not Correct.
2. **If you chose Almost**: Is the fix truly trivial (<30 seconds to explain)? Would the solution be perfect after the fix? If substantial work remains, choose Partial.
3. **If you chose Partial**: Does the student demonstrate understanding of the correct approach? If NO, choose Incorrect instead.
4. **If you chose Incorrect**: Is the approach fundamentally wrong? If the approach is right but incomplete, choose Partial instead.

**Common Mistakes to Avoid:**
- Don't confuse "Almost" with "Partial" - Almost is for tiny fixes, Partial is for major gaps
- Don't confuse "Partial" with "Incorrect" - Partial requires correct understanding, Incorrect means wrong approach
- Don't give "Correct" if there's any error, no matter how small

## Response Format:
You MUST respond with a JSON object in the following format (wrapped in <json> tags):

<json>
{{
    "response": "Correct" | "Incorrect" | "Partial" | "Almost"
}}
</json>

Important: Use exactly one of the four category names (Correct, Incorrect, Partial, Almost) as the value for the "response" field. Be precise in your classification based on the definitions above."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction using flexible extraction
        prediction = "None"
        try:
            response_text = msg_history[-1]["text"]
            extracted = _extract_response_flexible(response_text)
            if extracted:
                prediction = extracted
                self.log_fn(f"Extracted prediction: {prediction}")
            else:
                # Try one more time with normalization on the raw text
                normalized = _normalize_prediction(response_text)
                if normalized:
                    prediction = normalized
                    self.log_fn(f"Normalized prediction: {prediction}")
                else:
                    self.log_fn(f"Could not extract prediction from response: {response_text[:200]}...")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Validate prediction against allowed categories
        allowed_categories = ["Correct", "Incorrect", "Partial", "Almost"]
        if prediction not in allowed_categories:
            self.log_fn(f"Invalid prediction '{prediction}', defaulting to None")
            prediction = "None"

        return str(prediction), msg_history
