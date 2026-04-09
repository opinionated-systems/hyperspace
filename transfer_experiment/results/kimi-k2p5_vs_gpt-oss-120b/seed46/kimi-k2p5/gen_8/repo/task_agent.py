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


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks or raw JSON.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also tries to find JSON in code blocks and raw JSON objects.
    Includes robust error handling and multiple fallback strategies.
    """
    results = []
    search_from = 0
    
    # Try to find <json>...</json> blocks
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
            # Try to fix common JSON issues
            try:
                # Replace single quotes with double quotes
                fixed = inner.replace("'", '"')
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                # Try to extract just the response field
                try:
                    match = re.search(r'"response"\s*:\s*"([^"]+)"', inner)
                    if match:
                        results.append({"response": match.group(1)})
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
    flexible_pattern = r'\{\s*["\']?response["\']?\s*:\s*["\']?([^"\'\}]+)["\']?\s*\}'
    for match in re.finditer(flexible_pattern, text, re.IGNORECASE):
        try:
            value = match.group(1).strip().lower()
            if value in VALID_LABELS:
                results.append({"response": value})
        except Exception:
            continue
    
    # NEW: Try to find JSON with newlines and extra whitespace
    multiline_pattern = r'\{\s*"response"\s*:\s*"?([^"\}\n]+)"?\s*\}'
    for match in re.finditer(multiline_pattern, text, re.DOTALL):
        try:
            value = match.group(1).strip().lower()
            if value in VALID_LABELS:
                results.append({"response": value})
        except Exception:
            continue
    
    return results or None


def _extract_label_from_text(text: str) -> str | None:
    """Extract the grading label from raw text using pattern matching.
    
    Looks for exact matches of the three valid labels with priority-based
    matching to find the most likely intended label.
    Uses multiple strategies with increasing fallback tolerance.
    """
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
        rf'(?:verdict|conclusion)\s*[:=]?\s*["\']?({label_alternatives})["\']?\b',
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
            return max(valid_counts, key=valid_counts.get)
    
    # Priority 9: Fuzzy matching - look for labels with minor typos
    for label in VALID_LABELS:
        # Look for substrings that are close to the label
        if label in text_lower:
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

## GRADING CATEGORIES

You MUST classify the answer into EXACTLY ONE of these three categories:

- **"correct"**: The student's answer is essentially complete and correct. The logic is sound, ALL key mathematical steps are present, the final answer matches the official solution, and there are no significant gaps or errors. Minor presentation issues or trivial notation differences do NOT disqualify an answer from being "correct" if the mathematical content is right. The answer must be a FULL solution, not just a partial one.

- **"incorrect"**: The student's answer has fundamental flaws, demonstrates a critical misunderstanding of the problem, contains major logical errors that invalidate the solution, or misses essential components that make the answer wrong. This includes answers that are completely off-track, contain fatal mathematical errors, fail to address the core question, or have wrong final answers. Be CRITICAL - if the reasoning is flawed or incomplete, mark it "incorrect".

- **"partial"**: The student's answer shows genuine mathematical understanding and meaningful progress toward the solution, but has gaps, missing details, minor errors that don't invalidate the main approach, or incomplete reasoning. The student demonstrates they understand key concepts but hasn't fully solved the problem. This is for answers that are "on the right track" but not complete. Use this SPARINGLY - only when there's clear evidence of understanding but genuine incompleteness.

## IMO-STYLE FEW-SHOT EXAMPLES

### Example 1: Correct (Complete Proof)
Problem: Prove that there are infinitely many prime numbers.
Official Solution: Assume finitely many primes p₁, ..., pₙ. Consider N = p₁p₂...pₙ + 1. This N is not divisible by any pᵢ, so either N is prime or has a prime factor not in our list. Contradiction.
Student Answer: Suppose there are only finitely many primes: p₁, p₂, ..., pₙ. Let N = p₁ × p₂ × ... × pₙ + 1. When we divide N by any pᵢ, we get remainder 1. So N is either prime itself, or has a prime factor not in our list. Either way, we found a prime not in the original list, contradiction.
Analysis: The student correctly understands the proof by contradiction, constructs N properly, and explains why N isn't divisible by any listed prime. The reasoning is complete and correct.
Verdict: correct

### Example 2: Incorrect (Fundamental Error)
Problem: Find all positive integers n such that n² + n + 41 is prime for all n.
Official Solution: At n = 40: 40² + 40 + 41 = 1681 = 41², which is not prime. So no such n exists (the statement is false).
Student Answer: For n = 1: 1 + 1 + 41 = 43 (prime). For n = 2: 4 + 2 + 41 = 47 (prime). This pattern continues because 41 is prime and n² + n is always even, so n² + n + 41 is always odd and greater than 2, hence prime.
Analysis: The student made a fundamental error - they checked small cases but didn't verify their general claim. The reasoning that "odd and greater than 2 implies prime" is false. They missed the counterexample at n = 40 and made an invalid generalization.
Verdict: incorrect

### Example 3: Incorrect (Wrong Final Answer)
Problem: Find the sum of all positive integers n such that n² - 19n + 99 is a perfect square.
Official Solution: Let n² - 19n + 99 = k². Completing the square: (2n-19)² + 35 = 4k². So 4k² - (2n-19)² = 35. Factor as (2k - (2n-19))(2k + (2n-19)) = 35. The factor pairs of 35 give solutions n = 1, 9, 10, 18. Sum = 38.
Student Answer: Let n² - 19n + 99 = k². Then n² - 19n + (99 - k²) = 0. Using the quadratic formula: n = (19 ± √(361 - 4(99-k²)))/2 = (19 ± √(361 - 396 + 4k²))/2 = (19 ± √(4k² - 35))/2. For n to be integer, 4k² - 35 must be a perfect square. Trying k = 3: 36-35=1, so n = (19±1)/2 = 10 or 9. k = 4: 64-35=29, not square. k = 5: 100-35=65, not square. So the answer is 10 + 9 = 19.
Analysis: The student found two solutions (9 and 10) but missed 1 and 18. The final answer (19) is wrong. Even though the approach is partially valid, the answer is incomplete and incorrect.
Verdict: incorrect

### Example 4: Partial (Good Progress, Incomplete)
Problem: Prove that for any triangle with sides a, b, c: a² + b² + c² ≥ 4√3 × Area.
Official Solution: Using Heron's formula and AM-GM inequality to show the inequality holds with equality for equilateral triangles.
Student Answer: For an equilateral triangle with side s: Area = (√3/4)s². Then a² + b² + c² = 3s² and 4√3 × Area = 4√3 × (√3/4)s² = 3s². So equality holds for equilateral triangles. For other triangles, the left side grows relative to area.
Analysis: The student correctly verified the equality case for equilateral triangles and has intuition about the general case, but didn't provide a complete proof for all triangles. They showed understanding but the proof is incomplete.
Verdict: partial

### Example 5: Partial (Right Idea, Missing Details)
Problem: Find all pairs of distinct positive integers (a, b) such that a² + b² is divisible by a + b.
Official Solution: Using the identity a² + b² = (a+b)² - 2ab, we need (a+b) | 2ab. Let d = gcd(a,b), a = dx, b = dy with gcd(x,y) = 1. Then d(x+y) | 2d²xy, so (x+y) | 2dxy. Since gcd(x+y, xy) = 1, we get (x+y) | 2d. The solutions are all pairs where a + b divides 2ab.
Student Answer: We can write a² + b² = (a+b)² - 2ab. So a² + b² ≡ -2ab (mod a+b). For this to be 0, we need a+b to divide 2ab.
Analysis: The student made a key insight using modular arithmetic and the identity, correctly reducing the problem to finding when a+b divides 2ab. However, they stopped there without characterizing all solutions. They showed significant understanding but the solution is incomplete.
Verdict: partial

### Example 6: Incorrect (Incomplete with Fatal Gap)
Problem: Prove that the sum of the first n odd numbers is n².
Official Solution: The sum is 1 + 3 + 5 + ... + (2n-1). This is an arithmetic series with n terms, first term 1, last term (2n-1). Sum = n(1 + 2n - 1)/2 = n(2n)/2 = n².
Student Answer: The first few odd numbers are 1, 3, 5, 7, 9. Their sums are: 1=1, 1+3=4, 1+3+5=9, 1+3+5+7=16, 1+3+5+7+9=25. These are 1², 2², 3², 4², 5². So the pattern shows the sum is n².
Analysis: The student observed a pattern from examples but provided no proof. Pattern observation without proof is not a valid mathematical proof. The answer demonstrates understanding but fails to prove the statement.
Verdict: incorrect

### Example 7: Correct (With Minor Issues)
Problem: Solve the system: x + y = 5, xy = 6.
Official Solution: From x + y = 5 and xy = 6, x and y are roots of t² - 5t + 6 = 0, giving (t-2)(t-3) = 0, so solutions are (2,3) and (3,2).
Student Answer: x and y satisfy t² - 5t + 6 = 0. This factors as (t-2)(t-3) = 0. So t = 2 or t = 3. The solutions are x=2, y=3 or x=3, y=2.
Analysis: The student correctly identified the quadratic, factored it, and found both solutions. The presentation is slightly terse but mathematically complete and correct.
Verdict: correct

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

1. **Understanding**: Does the student demonstrate grasp of the core mathematical concepts? What key ideas are present or missing?

2. **Key Steps**: Are the critical logical and mathematical steps correct? Is the reasoning sound at each stage? Look for logical gaps or unjustified claims.

3. **Final Answer**: Is the final answer mathematically correct? Does it match the official solution's conclusions? If the final answer is wrong, the answer is likely "incorrect".

4. **Completeness**: Does the answer address ALL parts of the problem? What gaps or missing components exist? Incomplete solutions should NOT be marked "correct".

5. **Grading Guidelines Alignment**: How does this answer map to the specific grading criteria provided? What would an experienced competition math grader focus on?

6. **Decision**: Based on your analysis, which category best fits?
   - "correct": ONLY for complete, correct solutions with all key elements present
   - "incorrect": For fundamental flaws, wrong approach, critical errors, or wrong final answers
   - "partial": For genuine progress with valid insights but clear incompleteness

## IMPORTANT INSTRUCTIONS

- You MUST respond with EXACTLY ONE of: "correct", "incorrect", or "partial"
- Be STRICT and CRITICAL - IMO graders are rigorous. An incomplete solution is NOT "correct".
- If the final answer is wrong or missing, mark it "incorrect" (not "partial").
- "partial" is for answers that show real understanding but are genuinely incomplete - use sparingly.
- When in doubt between "incorrect" and "partial", prefer "incorrect" unless there's clear evidence of significant valid progress.
- IMO graders reward CORRECT and COMPLETE mathematics. Incomplete work is not rewarded with "correct".

Respond in this exact JSON format:
<json>
{{
    "response": "correct" | "incorrect" | "partial"
}}
</json>

The value of "response" MUST be exactly one of: "correct", "incorrect", or "partial"."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from response
        prediction = None
        text_to_parse = (response or "").strip()
        
        try:
            # First try to extract from JSON blocks
            extracted = _extract_jsons(text_to_parse)
            if extracted:
                for item in extracted:
                    if isinstance(item, dict) and "response" in item:
                        raw_pred = str(item["response"]).strip().lower()
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
                    prediction = text_pred
            
            # If still no prediction, check msg_history as last resort
            if prediction is None and msg_history:
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
            
            # Last resort: look for any valid label in the text
            if prediction is None:
                text_lower = text_to_parse.lower()
                for label in VALID_LABELS:
                    if label in text_lower:
                        prediction = label
                        break
            
            # Extra fallback: look for common variations or misspellings
            if prediction is None:
                text_lower = text_to_parse.lower()
                # Common variations - be careful not to over-predict "correct"
                if 'wrong' in text_lower or 'false' in text_lower or 'error' in text_lower or 'invalid' in text_lower:
                    prediction = 'incorrect'
                elif 'incomplete' in text_lower or 'partial credit' in text_lower:
                    prediction = 'partial'
                elif 'right' in text_lower or 'valid' in text_lower or 'true' in text_lower:
                    # Only use these as very weak signals
                    pass
                            
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            prediction = None
        
        # Final validation - ensure prediction is valid
        if prediction not in VALID_LABELS:
            # Default to "incorrect" as the safest fallback
            # This is better than random guessing when we can't determine the label
            prediction = "incorrect"

        return str(prediction), msg_history
