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

# Valid prediction labels
VALID_LABELS = {"Correct", "Incorrect", "Partial", "Almost"}

# Label priority order for extraction (longest first to avoid substring issues)
LABEL_PRIORITY = ["Incorrect", "Correct", "Partial", "Almost"]


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
        
        # Try to parse the JSON
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # Try to fix common issues
            # Remove any trailing commas before closing braces
            fixed = re.sub(r',\s*}', '}', inner)
            fixed = re.sub(r',\s*]', ']', fixed)
            # Try again with fixed version
            try:
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                # Try to extract just the response field using regex
                response_match = re.search(r'"response"\s*:\s*"([^"]+)"', inner)
                if response_match:
                    results.append({"response": response_match.group(1)})
                continue
    return results or None


def _extract_json_from_code_blocks(text: str) -> list[dict] | None:
    """Extract JSON objects from ```json...``` code blocks."""
    results = []
    pattern = r'```json\s*(.*?)\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            # Try to fix common issues
            fixed = re.sub(r',\s*}', '}', match.strip())
            fixed = re.sub(r',\s*]', ']', fixed)
            try:
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                # Try to extract just the response field using regex
                response_match = re.search(r'"response"\s*:\s*"([^"]+)"', match)
                if response_match:
                    results.append({"response": response_match.group(1)})
                continue
    return results or None


def _extract_json_raw(text: str) -> list[dict] | None:
    """Extract raw JSON objects from text (objects wrapped in {})."""
    results = []
    # Find all JSON-like structures - improved pattern for nested braces
    pattern = r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            parsed = json.loads(match.strip())
            if isinstance(parsed, dict):
                results.append(parsed)
        except json.JSONDecodeError:
            # Try to fix common issues
            fixed = re.sub(r',\s*}', '}', match.strip())
            fixed = re.sub(r',\s*]', ']', fixed)
            try:
                parsed = json.loads(fixed)
                if isinstance(parsed, dict):
                    results.append(parsed)
            except json.JSONDecodeError:
                # Try to extract just the response field using regex
                response_match = re.search(r'"response"\s*:\s*"([^"]+)"', match)
                if response_match:
                    results.append({"response": response_match.group(1)})
                continue
    
    # Additional fallback: try to find any text between curly braces
    if not results:
        # Look for patterns like {"response": "..."} or { "response" : "..." }
        response_pattern = r'\{\s*"response"\s*:\s*"([^"]+)"\s*\}'
        matches = re.findall(response_pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            results.append({"response": match.strip()})
    
    # Try to find response field with single quotes
    if not results:
        response_pattern = r"\{\s*'response'\s*:\s*'([^']+)'\s*\}"
        matches = re.findall(response_pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            results.append({"response": match.strip()})
    
    # Try to find response field with any quotes and spacing variations
    if not results:
        response_pattern = r'\{\s*["\']?response["\']?\s*:\s*["\']?(Correct|Incorrect|Partial|Almost)["\']?\s*\}'
        matches = re.findall(response_pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            results.append({"response": match.strip()})
    
    return results or None


def _extract_label_from_text(text: str) -> str | None:
    """Extract a valid label directly from text using multiple strategies.
    
    This is a comprehensive fallback that tries multiple approaches:
    1. Look for explicit "Classification: X" or "Answer: X" patterns
    2. Look for the last occurrence of any valid label
    3. Look for labels at the start or end of lines
    """
    text = text.strip()
    text_lower = text.lower()
    
    # Strategy 1: Look for explicit classification patterns
    classification_patterns = [
        r'classification\s*[:\-]\s*(Correct|Incorrect|Partial|Almost)',
        r'answer\s*[:\-]\s*(Correct|Incorrect|Partial|Almost)',
        r'result\s*[:\-]\s*(Correct|Incorrect|Partial|Almost)',
        r'grade\s*[:\-]\s*(Correct|Incorrect|Partial|Almost)',
        r'prediction\s*[:\-]\s*(Correct|Incorrect|Partial|Almost)',
        r'final\s*(?:answer|classification|grade)\s*[:\-]?\s*(Correct|Incorrect|Partial|Almost)',
        r'(?:the\s+)?(?:classification|grade|answer)\s+is\s*[:\-]?\s*(Correct|Incorrect|Partial|Almost)',
        r'I\s+(?:classify|grade|rate)\s+this\s+as\s*[:\-]?\s*(Correct|Incorrect|Partial|Almost)',
        r'this\s+is\s+(?:a\s+)?(Correct|Incorrect|Partial|Almost)',
    ]
    
    for pattern in classification_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            label = match.group(1)
            # Validate it's a proper label
            for valid_label in LABEL_PRIORITY:
                if label.lower() == valid_label.lower():
                    return valid_label
    
    # Strategy 2: Look for labels at the start of lines (common in reasoning)
    lines = text.split('\n')
    for line in lines:
        line_stripped = line.strip()
        for label in LABEL_PRIORITY:
            # Check if line starts with the label
            if re.match(rf'^{re.escape(label)}\b', line_stripped, re.IGNORECASE):
                return label
    
    # Strategy 3: Look for the last occurrence of any valid label in the text
    # This helps when the model reasons through multiple options
    last_pos = -1
    last_label = None
    for label in LABEL_PRIORITY:
        # Use word boundary to avoid matching "correct" inside "incorrect"
        pattern = r'\b' + re.escape(label) + r'\b'
        for match in re.finditer(pattern, text, re.IGNORECASE):
            if match.start() > last_pos:
                last_pos = match.start()
                last_label = label
    
    if last_label:
        return last_label
    
    # Strategy 4: Simple substring check (last resort)
    for label in LABEL_PRIORITY:
        if label.lower() in text_lower:
            return label
    
    return None


def _extract_last_label_occurrence(text: str) -> str | None:
    """Extract the last occurrence of a valid label from text.
    
    This is useful when the model mentions multiple labels during reasoning
    but the final classification is the last one mentioned.
    
    DEPRECATED: Use _extract_label_from_text instead for more comprehensive extraction.
    """
    text_lower = text.lower()
    last_pos = -1
    last_label = None
    
    for label in LABEL_PRIORITY:
        # Use word boundary to avoid matching "correct" inside "incorrect"
        pattern = r'\b' + re.escape(label.lower()) + r'\b'
        for match in re.finditer(pattern, text_lower):
            if match.start() > last_pos:
                last_pos = match.start()
                last_label = label
    
    return last_label


def _extract_direct_label(text: str) -> str | None:
    """Extract label directly from text by looking for valid labels."""
    # First, try to find explicit label declarations (highest priority)
    # Use priority order to avoid substring issues (Incorrect before Correct)
    label_patterns = [
        # JSON-like patterns with response field (most specific first)
        (r'"response"\s*:\s*"(Incorrect|Correct|Partial|Almost)"', True),
        (r"'response'\s*:\s*'(Incorrect|Correct|Partial|Almost)'", True),
        # Explicit declarations with various formats
        (r'["\']?(?:response|label|grade|classification|prediction)["\']?\s*[:=]\s*["\']?\s*(Incorrect|Correct|Partial|Almost)\s*["\']?', True),
        (r'(?:is|would be|should be|classified as)\s+["\']?\s*(Incorrect|Correct|Partial|Almost)\s*["\']?', True),
        (r'(?:grade|classification)\s*[:=]\s*["\']?\s*(Incorrect|Correct|Partial|Almost)\s*["\']?', True),
        # Markdown formatting
        (r'\*\s*(Incorrect|Correct|Partial|Almost)\s*\*', True),
        (r'\b\*\*(Incorrect|Correct|Partial|Almost)\*\*\b', True),
        (r'`(Incorrect|Correct|Partial|Almost)`', True),
        # Sentence patterns
        (r'\bThe\s+(?:answer|classification|grade|final\s+answer)\s+(?:is|would\s+be)\s+["\']?(Incorrect|Correct|Partial|Almost)["\']?', True),
        (r'\bI\s+(?:would|will)\s+(?:classify|grade|label)\s+(?:this|it|the\s+answer)\s+as\s+["\']?(Incorrect|Correct|Partial|Almost)["\']?', True),
        (r'\bTherefore[,]?\s+(?:the\s+answer\s+is\s+)?["\']?(Incorrect|Correct|Partial|Almost)["\']?', True),
        (r'\bFinal\s+(?:answer|classification|grade)[:]\s*["\']?(Incorrect|Correct|Partial|Almost)["\']?', True),
        (r'\bClassification[:]\s*(Incorrect|Correct|Partial|Almost)\b', True),
        (r'\bGrade[:]\s*(Incorrect|Correct|Partial|Almost)\b', True),
        # Additional patterns for issue count statements
        (r'I count \d+ issue\(s\).*\b(Incorrect|Correct|Partial|Almost)\b', True),
        (r'Classification:\s*(Incorrect|Correct|Partial|Almost)', True),
        (r'Grade:\s*(Incorrect|Correct|Partial|Almost)', True),
        # Classification at end of reasoning
        (r'\bclassify\s+this\s+as\s+["\']?(Incorrect|Correct|Partial|Almost)["\']?', True),
        (r'\bthis\s+(?:solution|answer)\s+(?:is|should\s+be)\s+["\']?(Incorrect|Correct|Partial|Almost)["\']?', True),
        (r'\bthe\s+(?:solution|answer)\s+(?:is|should\s+be)\s+["\']?(Incorrect|Correct|Partial|Almost)["\']?', True),
        # Final classification statement patterns
        (r'Classification:\s*(Incorrect|Correct|Partial|Almost)', True),
        (r'Final classification:\s*(Incorrect|Correct|Partial|Almost)', True),
        (r'"Classification":\s*"(Incorrect|Correct|Partial|Almost)"', True),
        (r"'Classification':\s*'(Incorrect|Correct|Partial|Almost)'", True),
        # Additional patterns for reasoning-based conclusions
        (r'\bso\s+the\s+(?:answer|classification|grade)\s+(?:is|would\s+be)\s+["\']?(Incorrect|Correct|Partial|Almost)["\']?', True),
        (r'\bhence[,]?\s+(?:the\s+)?["\']?(Incorrect|Correct|Partial|Almost)["\']?', True),
        (r'\bthus[,]?\s+(?:the\s+)?["\']?(Incorrect|Correct|Partial|Almost)["\']?', True),
        # Additional patterns for explicit statements
        (r'\bI\s+choose\s+["\']?(Incorrect|Correct|Partial|Almost)["\']?', True),
        (r'\bMy\s+(?:classification|grade|answer)\s+(?:is|would\s+be)\s+["\']?(Incorrect|Correct|Partial|Almost)["\']?', True),
        (r'\bThe\s+(?:correct|appropriate)\s+(?:classification|grade)\s+(?:is|would\s+be)\s+["\']?(Incorrect|Correct|Partial|Almost)["\']?', True),
        (r'\bThis\s+(?:is|should\s+be)\s+(?:graded|classified)\s+as\s+["\']?(Incorrect|Correct|Partial|Almost)["\']?', True),
        (r'\bI\s+(?:select|pick|choose)\s+["\']?(Incorrect|Correct|Partial|Almost)["\']?', True),
    ]
    
    for pattern, use_priority in label_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            label = match.group(1)
            # Validate against valid labels (case-insensitive)
            for valid_label in LABEL_PRIORITY:  # Use priority order
                if label.lower() == valid_label.lower():
                    return valid_label
    
    # Look for labels in the last few lines (often where the conclusion is)
    lines = text.split('\n')
    last_lines = lines[-50:] if len(lines) > 50 else lines
    for line in reversed(last_lines):
        line = line.strip()
        # Skip empty lines and common non-content lines
        if not line or line.startswith('```') or line.startswith('<json>') or line.startswith('</json>'):
            continue
        # Check in priority order
        for label in LABEL_PRIORITY:
            # Look for label as a whole word with word boundaries
            pattern = r'\b' + re.escape(label) + r'\b'
            if re.search(pattern, line, re.IGNORECASE):
                # Additional check: make sure it's not part of a larger word
                # and not in a code block or tag
                if not line.startswith('```') and not line.startswith('<'):
                    return label
    
    # Clean up the text - remove punctuation that might interfere
    cleaned_text = re.sub(r'["\'\(\)\[\]\{\}<>]', ' ', text)
    words = cleaned_text.split()
    
    # Check each word (case-insensitive), prioritizing later words
    for word in reversed(words):
        word_stripped = word.strip('.,;:!?')
        # Check in priority order
        for label in LABEL_PRIORITY:
            if word_stripped.lower() == label.lower():
                return label
    
    # Also check for labels as substrings in all lines (last resort)
    for line in reversed(lines):
        line = line.strip()
        if not line or line.startswith('```') or line.startswith('<'):
            continue
        # Check in priority order
        for label in LABEL_PRIORITY:
            pattern = r'\b' + re.escape(label) + r'\b'
            if re.search(pattern, line, re.IGNORECASE):
                return label
    
    # Final fallback: check the very last non-empty line for exact match
    for line in reversed(lines):
        line_clean = line.strip().strip('.,;:!?"\'`*[]{}()<>')
        if line_clean:
            # Check in priority order
            for label in LABEL_PRIORITY:
                if line_clean.lower() == label.lower():
                    return label
    
    # Ultra fallback: search entire text for any occurrence of valid labels
    # This is a last resort when all other methods fail
    # IMPORTANT: Check for longer labels first to avoid substring matches
    # e.g., "incorrect" contains "correct", so check "incorrect" first
    text_lower = text.lower()
    
    # Find all occurrences and return the last one found
    last_pos = -1
    last_label = None
    for label in LABEL_PRIORITY:  # Already in priority order
        # Use word boundary to avoid matching "correct" inside "incorrect"
        pattern = r'\b' + re.escape(label.lower()) + r'\b'
        for match in re.finditer(pattern, text_lower):
            if match.start() > last_pos:
                last_pos = match.start()
                last_label = label
    if last_label:
        return last_label
    
    # Final final fallback: look for labels in any context (very last resort)
    # Still use priority order
    for label in LABEL_PRIORITY:
        if label.lower() in text_lower:
            return label
    
    return None


def extract_prediction(text: str) -> str | None:
    """Extract prediction from text using multiple strategies.
    
    Tries multiple extraction methods in order of preference:
    1. <json>...</json> tags
    2. ```json...``` code blocks
    3. Raw JSON objects
    4. Direct label extraction from text using comprehensive patterns
    5. Last occurrence of any valid label
    
    Returns the "response" field value, or None if extraction fails.
    """
    # Clean the text first - remove common formatting issues
    cleaned_text = text.strip()
    
    # Try <json> tags first
    extracted = _extract_jsons(cleaned_text)
    if extracted:
        for item in extracted:
            if isinstance(item, dict) and "response" in item:
                response = str(item["response"]).strip()
                # Case-insensitive validation
                for label in LABEL_PRIORITY:
                    if response.lower() == label.lower():
                        return label
    
    # Try ```json code blocks
    extracted = _extract_json_from_code_blocks(cleaned_text)
    if extracted:
        for item in extracted:
            if isinstance(item, dict) and "response" in item:
                response = str(item["response"]).strip()
                for label in LABEL_PRIORITY:
                    if response.lower() == label.lower():
                        return label
    
    # Try raw JSON
    extracted = _extract_json_raw(cleaned_text)
    if extracted:
        for item in extracted:
            if isinstance(item, dict) and "response" in item:
                response = str(item["response"]).strip()
                for label in LABEL_PRIORITY:
                    if response.lower() == label.lower():
                        return label
    
    # Fallback 1: comprehensive label extraction from text
    # This handles cases where the model provides reasoning but no JSON
    text_label = _extract_label_from_text(cleaned_text)
    if text_label:
        return text_label
    
    # Fallback 2: direct label extraction (legacy method)
    direct_label = _extract_direct_label(cleaned_text)
    if direct_label:
        return direct_label
    
    # Fallback 3: extract the last occurrence of any valid label
    # This is useful when the model mentions multiple labels during reasoning
    last_label = _extract_last_label_occurrence(cleaned_text)
    if last_label:
        return last_label
    
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
        # Extract fields from inputs for better prompt construction
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        instruction = f"""You are an expert mathematical problem solver and grader for IMO (International Mathematical Olympiad) problems.

Your task is to grade a student's answer to a mathematical problem based on the provided solution and grading guidelines.

## Problem:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

---

## ⚠️ CRITICAL WARNING: DO NOT BE FOOLED BY THE GRADING GUIDELINES ⚠️

The grading guidelines describe what WOULD count as Partial or Almost IF the student actually achieved those things.
**BUT** you must verify that the student ACTUALLY PROVED/DEMONSTRATED those things, not just mentioned or discussed them.

**COMMON MISTAKE**: Seeing that the guidelines mention "Proved X" and the student mentions X, then calling it Partial.
**CORRECT APPROACH**: Did the student actually PROVE X with valid mathematical reasoning, or did they just mention X exists?

---

## CLASSIFICATION RUBRIC - FOLLOW THESE RULES EXACTLY

### The Four Categories:

**Correct**: The answer is 100% complete and correct.
- Every step is fully worked out with no gaps
- No errors, typos, or missing cases
- Would receive full marks in a competition
- The proof is rigorous and complete

**Almost**: The answer is nearly perfect with exactly ONE tiny issue.
- 95-99% complete - essentially a full proof with all major steps present
- Only ONE minor error (e.g., single arithmetic mistake, one typo, one trivial case omitted)
- Can be fixed to perfect in under 30 seconds by a knowledgeable reader
- **CRITICAL**: If there are 2+ issues of ANY kind, CANNOT be "Almost"
- **CRITICAL**: "Almost" is VERY RARE - only use when the proof is essentially complete
- **CRITICAL**: Missing a proof direction (e.g., only proving one way of iff) = Partial, NOT Almost
- **CRITICAL**: Multiple issues = Partial (not Almost)
- **CRITICAL**: "Minor mistakes which are not negligible" = Partial (not Almost)
- **CRITICAL**: Incomplete infinite descent = Partial (not Almost)

**Partial**: The answer shows meaningful progress but has significant gaps.
- 30-84% complete - found key insight but couldn't finish
- Multiple issues OR one major gap in reasoning
- Missing critical cases or directions (e.g., only proved one direction of "if and only if")
- Good start with valid mathematical insight, but needs substantial additional work
- **CRITICAL**: If student found a valid key insight, CANNOT be "Incorrect"
- **CRITICAL**: This is the MOST COMMON category for incomplete work
- **CRITICAL**: Work with 2+ minor issues = Partial (not Almost)
- **CRITICAL**: Missing proof direction = Partial (not Almost)
- **CRITICAL**: "Minor mistakes which are not negligible" = Partial

**Incorrect**: The answer shows little to no valid progress.
- 0-29% complete - no meaningful work toward solution
- Fundamental misunderstanding or completely wrong approach
- No key insight found
- Random guessing or completely irrelevant work
- **CRITICAL**: If ANY valid insight was found, it's at least Partial

---

## DECISION WORKFLOW - FOLLOW STEP BY STEP

**Step 1**: Did the student find ANY valid key insight or make meaningful progress toward the solution?
- Look for: correct approach, valid lemma, key observation, partial proof, correct setup
- **KEY TEST**: Did they actually PROVE/DEMONSTRATE something, or just TALK ABOUT what would be needed?
- If NO valid insight at all → **Incorrect**
- If YES, found something valid → Continue to Step 2

**Step 2**: Is the proof essentially complete (95%+)? Are all major proof directions covered?
- Check: Are all major steps present? Is the logic mostly complete?
- Check: For "if and only if" or "find all" problems, did they prove BOTH directions / find ALL solutions?
- If NO, significant gaps remain OR missing a proof direction → **Partial**
- If YES, essentially a complete proof with all directions covered → Continue to Step 3

**Step 3**: Count the issues carefully. Is there exactly ONE minor issue?
- Minor issues: single arithmetic error, one typo, one trivial case omitted, small calculation mistake
- Major issues: missing proof direction, logical gap, multiple errors, missing case analysis
- If exactly ONE minor issue → **Almost**
- If ZERO issues → **Correct**
- If TWO or more issues (even if minor) → **Partial**
- If ANY major issue → **Partial**

---

## DETAILED EXAMPLES

**Example 1 - Correct:**
Problem: Prove that for all positive integers n, n³ - n is divisible by 6.
Student: Shows n³ - n = n(n-1)(n+1), explains this is product of 3 consecutive integers, 
so divisible by 2 (one even) and 3 (one multiple of 3), hence by 6. All cases covered.
→ Classification: Correct

**Example 2 - Almost (single minor error in complete proof):**
Problem: Find all primes p such that p² + 2 is also prime.
Student: Correctly shows p=3 works (gives 11), proves p=2 gives 6 (not prime), 
shows for p>3, p² ≡ 1 (mod 3) so p²+2 ≡ 0 (mod 3) and composite. 
**BUT** makes one arithmetic error: says 3²+2=10 instead of 11, though conclusion is correct.
→ Classification: Almost (one minor arithmetic error in an otherwise complete proof)

**Example 3 - Almost (one trivial case omitted in complete proof):**
Problem: Prove that for all integers n > 1, n⁴ + 4 is composite.
Student: Correctly factors n⁴ + 4 = (n² + 2n + 2)(n² - 2n + 2) for all n, shows both factors > 1 for n > 1,
**BUT** forgets to check n=2 case separately (though it works: 2⁴+4=20=4×5).
→ Classification: Almost (one trivial case check missing in otherwise complete proof)

**Example 4 - Partial (found insight but gap in reasoning):**
Problem: Prove that if a² + b² + c² = ab + bc + ca for real numbers a, b, c, then a = b = c.
Student: Rearranges to (a-b)² + (b-c)² + (c-a)² = 0 correctly, but then says 
"this implies a=b=c" without explaining why squares summing to zero means each is zero.
Missing the key step justification.
→ Classification: Partial (found key insight but gap in reasoning)

**Example 5 - Partial (missing direction - NOT Almost):**
Problem: Prove that triangle ABC is equilateral if and only if a² + b² + c² = ab + bc + ca.
Student: Only proves the forward direction (equilateral → equation holds).
Does not prove the converse (equation holds → equilateral).
→ Classification: Partial (missing a proof direction = significant gap, NOT Almost)

**Example 6 - Partial (multiple minor issues):**
Problem: Prove that sum of first n odd numbers is n².
Student: Correctly identifies pattern and attempts induction. Base case correct. 
Inductive step has correct structure BUT makes two arithmetic errors in the algebra.
→ Classification: Partial (multiple errors, even though approach is correct)

**Example 7 - Partial (incomplete proof - NOT Almost):**
Problem: Prove that there are infinitely many primes of the form 4k+3.
Student: Correctly sets up proof by contradiction, assumes finite list p₁,...,pₙ,
correctly constructs N = 4p₁...pₙ - 1, shows N ≡ 3 (mod 4), but gets stuck showing N has a prime factor of form 4k+3.
→ Classification: Partial (incomplete proof with valid insight but significant gap remaining)

**Example 8 - Partial (valid insight but incomplete execution):**
Problem: Find all functions f: ℤ → ℤ such that f(f(n)) = n + 1 for all n.
Student: Correctly observes that f must be bijective, finds that f(n+1) = f(f(f(n))) = f(n) + 1,
so f(n) = n + c for some constant c. But then fails to check consistency or find the contradiction.
→ Classification: Partial (found key insight about linear form but didn't complete the proof)

**Example 9 - Incorrect (no valid insight):**
Problem: Prove that √2 is irrational.
Student: Says "Assume √2 = p/q" but then just writes random algebraic manipulations 
that don't lead anywhere, or makes fundamental errors like saying "if p² is even then p is odd".
→ Classification: Incorrect (no valid mathematical progress)

**Example 10 - Incorrect (misunderstanding the problem):**
Problem: Find all positive integers n such that n² + n + 1 is prime.
Student: Tests n=1,2,3, finds they're all prime, concludes "all positive integers work" 
without realizing n=4 gives 21 = 3×7.
→ Classification: Incorrect (fundamental misunderstanding, no valid general approach)

**Example 11 - Partial (NOT Almost - non-negligible mistake):**
Problem: Complex number theory problem.
Grading Guidelines say: "(Almost) minor mistakes which are not negligible"
Student: Has the right approach, found key insight, BUT made a mistake in the modular arithmetic 
that affects the conclusion, or missed a non-trivial case.
→ Classification: Partial (NOT Almost - mistakes are "not negligible" per guidelines)
**WHY NOT ALMOST?** When guidelines explicitly say "minor mistakes which are not negligible" for Almost criteria, 
this means Almost requires verification with ONLY truly minor mistakes. "Not negligible" mistakes make it Partial.

**Example 12 - Partial (NOT Almost - omitted case):**
Problem: Find all pairs (x,y) of positive integers for which the limit of sequence aₙ exists, 
where aₙ = gcd(xⁿ + y, (y-x)(Σyⁱx^(n-i-1) - 1)).
Student: Solution is almost complete, BUT made minor mistakes which are not negligible, 
AND omitted the case when xy+1 doesn't have an odd prime factor.
→ Classification: Partial (omitted case + non-negligible mistakes = NOT Almost)
**WHY NOT ALMOST?** Two reasons: (1) "minor mistakes which are not negligible" = Partial, (2) omitted case = Partial.

**Example 13 - INCORRECT (CRITICAL: Do not be fooled by guidelines!):**
Problem: Complex geometry problem.
Grading Guidelines say: "(Partial) 1. Proved that B, T, P, C lie on a circle. 2. Observed that it suffices to show ratio of powers..."
Student: Writes many observations about the problem setup, discusses radical axes, mentions what "would be sufficient" to prove,
but **NEVER ACTUALLY PROVES** that B, T, P, C lie on a circle, and **NEVER ACTUALLY SHOWS** the ratio of powers property.
→ Classification: **INCORRECT** (discusses what WOULD be needed but provides no actual valid proof/observation)
**WHY NOT PARTIAL?** Because the student didn't actually achieve what the guidelines list. They just talked about the topics.

**Example 14 - INCORRECT (CRITICAL: Do not be fooled by guidelines!):**
Problem: Problem about sequences and number theory.
Grading Guidelines say: "(Partial) 1. Considered a prime p|xy+1."
Student: Restates the problem in different notation, considers various cases, but **never actually analyzes** the divisibility condition
or makes any valid number-theoretic progress. The "consideration" of primes is just restating the problem, not actual work.
→ Classification: **INCORRECT** (restatement and setup without actual mathematical progress)
**WHY NOT PARTIAL?** Because the student didn't actually DO anything with the prime consideration. They just mentioned it.

**Example 15 - INCORRECT (CRITICAL: "Considered" ≠ "Mentioned"):**
Problem: Number theory problem about sequences.
Grading Guidelines say: "(Partial) 1. Considered a prime p|xy+1."
Student: Writes "Let p be a prime dividing xy+1" and then immediately moves on without using this in any proof,
without analyzing properties of p, without using p in any meaningful way. Just mentions p exists.
→ Classification: **INCORRECT** (mere mention without actual use)
**WHY NOT PARTIAL?** "Considered" in the guidelines means "analyzed, used, or applied in a meaningful way" - not just "mentioned in passing."

**Example 16 - INCORRECT (CRITICAL: "Observed" ≠ "Stated"):**
Problem: Geometry problem about arcs and partitions.
Grading Guidelines say: "(Partial) 1. Observed that when an arc with scores smaller than 1 is deleted, the condition still holds."
Student: Writes "Note that deleting such an arc preserves the condition" without proof, without explanation, without using this in any argument.
Just states it as if obvious, doesn't actually establish it mathematically.
→ Classification: **INCORRECT** (unsubstantiated claim, not a valid observation)
**WHY NOT PARTIAL?** "Observed" in the guidelines means "proved/established with mathematical reasoning" - not just "stated without proof."

**Example 17 - INCORRECT (CRITICAL: Setup ≠ Progress):**
Problem: Find all pairs (x,y) such that limit of sequence aₙ exists.
Grading Guidelines say: "(Partial) 1. Analyzed the case x=y. 2. Used the identity for the summation."
Student: Writes "Case 1: x=y" and "The summation is Σyⁱx^(n-i-1)" but never actually analyzes what happens when x=y,
never uses the identity to simplify anything, just states the setup.
→ Classification: **INCORRECT** (case labeling without actual analysis)
**WHY NOT PARTIAL?** Labeling a case and stating a formula is not analysis. The student must actually DO something with them.

---

## CRITICAL DISTINCTION: Partial vs Almost

The key difference is the NUMBER and SEVERITY of issues:

**Almost = Complete proof + exactly ONE tiny issue**
- All proof directions covered
- All major steps present
- Just one small mistake (arithmetic, typo, trivial case)
- The proof would be Correct if that one issue were fixed

**Partial = Valid insight + significant gaps OR multiple issues**
- Missing a proof direction (e.g., only one way of iff)
- Multiple minor issues (2+ typos/arithmetic errors)
- One major logical gap
- Good start but couldn't finish
- Incomplete proof (even if 90% done)

**When in doubt between Partial and Almost, choose Partial.**
"Almost" should be used very sparingly - only for proofs that are 95-99% complete with exactly one tiny issue.

---

## CRITICAL DISTINCTION: Incorrect vs Partial

This is the most important distinction for accurate grading:

**Incorrect = No valid mathematical insight or progress**
- Just restates the problem in different words
- Sets up notation but never uses it meaningfully
- Makes observations that are trivial or just restatements
- Claims to "consider" or "observe" things but never actually proves or demonstrates them
- Writes about what WOULD be sufficient without actually proving anything
- No actual key insight found

**Partial = Has at least ONE valid key insight or meaningful progress**
- Actually proves a non-trivial lemma
- Correctly identifies and applies a key technique
- Makes a valid observation that advances toward the solution
- Has some correct mathematical content that contributes to the solution

**KEY TEST for Incorrect vs Partial:**
Ask: "Did the student actually PROVE or DEMONSTRATE something meaningful, or did they just TALK ABOUT what would be needed?"
- If they just talked about it → Incorrect
- If they actually showed/did it → Partial

---

## ⚠️ MOST COMMON ERRORS TO AVOID ⚠️

1. **DON'T be fooled by grading guidelines**: The guidelines describe what WOULD count as Partial/Almost. The student must actually ACHIEVE those things. Just mentioning the topics from the guidelines doesn't make it Partial.

2. **DON'T overuse "Almost"**: "Almost" should be rare. Most incomplete work is "Partial".

3. **DON'T call work with multiple issues "Almost"**: Two typos = Partial, not Almost.

4. **DON'T call work with missing proof directions "Almost"**: Missing a direction = Partial.

5. **DON'T call incomplete proofs "Almost"**: If the proof is not essentially complete, it's Partial.

6. **DON'T call work with valid insights "Incorrect"**: If they found something meaningful, it's at least Partial.

7. **DO use "Partial" for incomplete proofs with valid insights**: This is the most common category.

8. **DO check both directions** for "if and only if" problems - missing one direction = Partial.

9. **DO apply the key test**: Did they actually prove/demonstrate something, or just talk about it?

10. **DO pay attention to "minor mistakes which are not negligible"**: If guidelines use this phrase for Almost criteria, it means Almost requires verification with ONLY truly minor mistakes. "Not negligible" mistakes make it Partial.

11. **DON'T call incomplete infinite descent "Almost"**: Starting an infinite descent argument but not completing it = Partial, not Almost.

---

## OUTPUT FORMAT (MANDATORY)

You MUST end your response with EXACTLY ONE JSON block in this format:

<json>
{{"response": "Correct"}}
</json>

OR

<json>
{{"response": "Almost"}}
</json>

OR

<json>
{{"response": "Partial"}}
</json>

OR

<json>
{{"response": "Incorrect"}}
</json>

**CRITICAL RULES:**
1. Use ONLY one of: Correct, Almost, Partial, Incorrect (exact spelling, case-sensitive)
2. The JSON block must be the LAST thing in your response
3. No text after the JSON block
4. Use double quotes in the JSON, not single quotes
5. Include the <json> tags exactly as shown

**Example of correct output:**
After analyzing the student's answer, I found they identified the key approach and proved the main direction, but did not prove the converse. This is significant incomplete work with a valid insight but missing a proof direction, so the classification is Partial.

<json>
{{"response": "Partial"}}
</json>"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
            temperature=0.1,
        )

        # Extract prediction from JSON using multiple strategies
        prediction = "None"
        try:
            last_message = msg_history[-1]["text"] if msg_history else ""
            extracted_prediction = extract_prediction(last_message)
            if extracted_prediction is not None:
                prediction = extracted_prediction
                self.log_fn(f"Successfully extracted prediction: {prediction}")
            else:
                self.log_fn(f"Failed to extract prediction from response: {last_message[:500]}...")
                # Try to extract any valid label from the entire message history
                for msg in reversed(msg_history):
                    if "text" in msg:
                        extracted = extract_prediction(msg["text"])
                        if extracted is not None:
                            prediction = extracted
                            self.log_fn(f"Extracted prediction from earlier message: {prediction}")
                            break
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Validate the final prediction
        if prediction not in VALID_LABELS:
            self.log_fn(f"Warning: Extracted prediction '{prediction}' is not a valid label. Defaulting to 'None'.")
            prediction = "None"
        
        # Log detailed information for debugging
        self.log_fn(f"Final prediction: {prediction}")
        self.log_fn(f"Problem type: {inputs.get('domain', 'unknown')}")
        self.log_fn(f"Student answer length: {len(student_answer)} chars")

        return str(prediction), msg_history


# Default to using TaskAgentWithRetry for better reliability
class DefaultTaskAgent(TaskAgent):
    """Default task agent with retry mechanism."""
    
    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent with retry logic."""
        # First attempt
        prediction, msg_history = super().forward(inputs)
        
        # If prediction is valid, return it
        if prediction in VALID_LABELS:
            return prediction, msg_history
        
        # Try retries with a reminder about the format
        for attempt in range(2):
            self.log_fn(f"Retry attempt {attempt + 1}/2 due to invalid prediction: {prediction}")
            
            # Add a reminder message to the history
            retry_msg = """Your previous response did not follow the required format. 

You MUST end your response with EXACTLY ONE JSON block in this format:

<json>
{"response": "Correct"}
</json>

OR

<json>
{"response": "Almost"}
</json>

OR

<json>
{"response": "Partial"}
</json>

OR

<json>
{"response": "Incorrect"}
</json>

**CRITICAL RULES:**
1. Use ONLY one of: Correct, Almost, Partial, Incorrect (exact spelling, case-sensitive)
2. Include the <json> tags exactly as shown
3. The JSON block must be the LAST thing in your response
4. No text after the JSON block
5. Use double quotes in the JSON, not single quotes

**Remember the classification rules:**
- Correct: 100% complete, no issues
- Almost: 95-99% complete, exactly ONE minor issue in an otherwise complete proof (VERY RARE!)
  - Must be essentially complete - incomplete proofs are Partial, not Almost
  - Missing proof direction = Partial (not Almost)
  - Multiple issues = Partial (not Almost)
  - "Minor mistakes which are not negligible" = Partial (not Almost)
  - Incomplete infinite descent = Partial (not Almost)
- Partial: 30-84% complete, found key insight but significant gaps remain (MOST COMMON for incomplete work)
  - Missing proof direction = Partial (not Almost)
  - Multiple issues = Partial (not Almost)
  - Incomplete proof = Partial (not Almost)
  - "Minor mistakes which are not negligible" = Partial
- Incorrect: 0-29% complete, no meaningful progress, no valid insight found

**KEY TEST for Incorrect vs Partial:**
Ask: "Did the student actually PROVE or DEMONSTRATE something meaningful, or did they just TALK ABOUT what would be needed?"
- If they just talked about it → Incorrect
- If they actually showed/did it → Partial

**⚠️ CRITICAL: Do not be fooled by grading guidelines!**
The guidelines describe what WOULD count as Partial/Almost. You must verify the student actually ACHIEVED those things.

**⚠️ CRITICAL: "Minor mistakes which are not negligible"**
If the grading guidelines use this phrase for Almost criteria, then Almost requires verification with ONLY truly minor mistakes. "Not negligible" mistakes make it Partial.

Now provide ONLY the JSON block with your classification."""
            
            try:
                response, new_msg_history, info = get_response_from_llm(
                    msg=retry_msg,
                    model=self.model,
                    msg_history=msg_history,
                    temperature=0.1,
                )
                
                # Combine message histories
                msg_history = msg_history + new_msg_history
                
                # Try to extract prediction
                last_message = msg_history[-1]["text"] if msg_history else ""
                extracted_prediction = extract_prediction(last_message)
                
                if extracted_prediction is not None and extracted_prediction in VALID_LABELS:
                    prediction = extracted_prediction
                    self.log_fn(f"Successfully extracted prediction on retry: {prediction}")
                    break
            except Exception as e:
                self.log_fn(f"Error during retry: {e}")
                continue
        
        return str(prediction), msg_history


class TaskAgentWithRetry(TaskAgent):
    """Task agent with retry mechanism for better reliability."""

    def forward(self, inputs: dict, max_retries: int = 2) -> tuple[str, list[dict]]:
        """Run the task agent with retry logic.
        
        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer
            max_retries: maximum number of retries if extraction fails

        Returns:
            (prediction, msg_history)
        """
        prediction, msg_history = super().forward(inputs)
        
        # If prediction is valid, return it
        if prediction in VALID_LABELS:
            return prediction, msg_history
        
        # Try retries with a reminder about the format
        for attempt in range(max_retries):
            self.log_fn(f"Retry attempt {attempt + 1}/{max_retries} due to invalid prediction: {prediction}")
            
            # Add a reminder message to the history
            retry_msg = """Your previous response did not follow the required format. 

You MUST end your response with EXACTLY ONE JSON block in this format:

<json>
{"response": "Correct"}
</json>

OR

<json>
{"response": "Almost"}
</json>

OR

<json>
{"response": "Partial"}
</json>

OR

<json>
{"response": "Incorrect"}
</json>

**CRITICAL RULES:**
1. Use ONLY one of: Correct, Almost, Partial, Incorrect (exact spelling, case-sensitive)
2. Include the <json> tags exactly as shown
3. The JSON block must be the LAST thing in your response
4. No text after the JSON block
5. Use double quotes in the JSON, not single quotes

**Remember the classification rules:**
- Correct: 100% complete, no issues
- Almost: 95-99% complete, exactly ONE minor issue in an otherwise complete proof (VERY RARE!)
  - Must be essentially complete - incomplete proofs are Partial, not Almost
  - Missing proof direction = Partial (not Almost)
  - Multiple issues = Partial (not Almost)
  - "Minor mistakes which are not negligible" = Partial (not Almost)
  - Incomplete infinite descent = Partial (not Almost)
- Partial: 30-84% complete, found key insight but significant gaps remain (MOST COMMON for incomplete work)
  - Missing proof direction = Partial (not Almost)
  - Multiple issues = Partial (not Almost)
  - Incomplete proof = Partial (not Almost)
  - "Minor mistakes which are not negligible" = Partial
- Incorrect: 0-29% complete, no meaningful progress, no valid insight found

**KEY TEST for Incorrect vs Partial:**
Ask: "Did the student actually PROVE or DEMONSTRATE something meaningful, or did they just TALK ABOUT what would be needed?"
- If they just talked about it → Incorrect
- If they actually showed/did it → Partial

**⚠️ CRITICAL: Do not be fooled by grading guidelines!**
The guidelines describe what WOULD count as Partial/Almost. You must verify the student actually ACHIEVED those things.

**⚠️ CRITICAL: "Minor mistakes which are not negligible"**
If the grading guidelines use this phrase for Almost criteria, then Almost requires verification with ONLY truly minor mistakes. "Not negligible" mistakes make it Partial.

Now provide ONLY the JSON block with your classification."""
            
            try:
                response, new_msg_history, info = get_response_from_llm(
                    msg=retry_msg,
                    model=self.model,
                    msg_history=msg_history,
                    temperature=0.1,
                )
                
                # Combine message histories
                msg_history = msg_history + new_msg_history
                
                # Try to extract prediction
                last_message = msg_history[-1]["text"] if msg_history else ""
                extracted_prediction = extract_prediction(last_message)
                
                if extracted_prediction is not None and extracted_prediction in VALID_LABELS:
                    prediction = extracted_prediction
                    self.log_fn(f"Successfully extracted prediction on retry: {prediction}")
                    break
            except Exception as e:
                self.log_fn(f"Error during retry: {e}")
                continue
        
        return str(prediction), msg_history
