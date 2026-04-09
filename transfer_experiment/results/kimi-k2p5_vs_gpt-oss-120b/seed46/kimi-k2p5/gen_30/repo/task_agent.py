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
VALID_LABELS = ["correct", "incorrect", "partial", "almost"]

# Label priority for conflict resolution (higher = more preferred when uncertain)
# For IMO grading, we want to be conservative - prefer incorrect over false positives
# But we also want to catch partial credit cases - so partial gets high priority
LABEL_PRIORITY = {"incorrect": 4, "almost": 3, "partial": 2, "correct": 1}

# Confidence thresholds for different labels
CONFIDENCE_THRESHOLDS = {
    "correct": 0.85,   # High bar for correct - must be very confident
    "incorrect": 0.7,  # Moderate bar for incorrect
    "partial": 0.5,    # Lower bar for partial - catch valid attempts
    "almost": 0.75,    # High bar for almost - nearly complete
}

# Mapping from labels to their canonical form
LABEL_CANONICAL = {
    "correct": "correct",
    "incorrect": "incorrect", 
    "partial": "partial",
    "almost": "partial",
}

# Common misspellings and variations mapping
LABEL_ALIASES = {
    "correct": ["correct", "right", "true", "valid", "accurate", "complete", "fully correct", "entirely correct", "perfect", "solved", "full marks"],
    "incorrect": ["incorrect", "wrong", "false", "error", "invalid", "inaccurate", "flawed", "mistaken", "fundamentally wrong", "no credit", "zero"],
    "partial": ["partial", "incomplete", "partially", "some", "part", "partial credit", "half"],
    "almost": ["almost", "nearly", "mostly correct", "almost correct", "nearly complete", "almost there", "minor gap", "tiny gap", "small gap", "nearly right"],
}


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks or raw JSON.
    
    Uses multiple strategies with priority ordering to handle various output formats.
    Returns list of dicts with "response" field, or None if no valid JSON found.
    """
    if not text or not isinstance(text, str):
        return None
        
    results = []
    text_lower = text.lower()
    
    # Strategy 0: Look for JSON block at the very end of the text (most reliable for our prompt)
    # This handles cases where the JSON is the last line but may have extra whitespace
    lines = text.split('\n')
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i].strip()
        if not line:
            continue
        # Check if this line contains a complete JSON block
        if '<json>' in line and '</json>' in line:
            start = line.lower().find('<json>')
            end = line.lower().find('</json>') + 7
            if start != -1 and end > start:
                inner = line[start + 6:end - 7].strip()
                parsed = _try_parse_json_with_fallbacks(inner)
                if parsed and "response" in parsed:
                    results.append(parsed)
                    break
    
    # Strategy 1: Try to find <json>...</json> blocks (case insensitive) - HIGHEST PRIORITY
    # Search from the END to get the most recent/final JSON block first
    search_positions = []
    search_from = 0
    while True:
        start = text_lower.find("<json>", search_from)
        if start == -1:
            break
        end = text_lower.find("</json>", start)
        if end == -1:
            break
        search_positions.append((start, end))
        search_from = end + 7
    
    # Process from last to first (most recent JSON is likely the final verdict)
    for start, end in reversed(search_positions):
        inner = text[start + 6:end].strip()
        parsed = _try_parse_json_with_fallbacks(inner)
        if parsed and "response" in parsed:
            results.append(parsed)
    
    # Strategy 2: Try to find JSON in markdown code blocks
    code_block_patterns = [
        r'```(?:json)?\s*\n?(.*?)\n?```',
    ]
    for pattern in code_block_patterns:
        for match in re.finditer(pattern, text, re.DOTALL):
            json_str = match.group(1).strip()
            if json_str:
                parsed = _try_parse_json_with_fallbacks(json_str)
                if parsed and "response" in parsed:
                    results.append(parsed)
    
    # Strategy 3: Try to find raw JSON objects with "response" field
    json_patterns = [
        r'\{\s*"response"\s*:\s*"([^"]+)"\s*\}',
        r"\{\s*'response'\s*:\s*'([^']+)'\s*\}",
        r'\{\s*["\']?response["\']?\s*:\s*["\']?([^"\'\}\n,]+)["\']?\s*\}',
    ]
    for pattern in json_patterns:
        for match in re.finditer(pattern, text, re.DOTALL | re.IGNORECASE):
            parsed = _try_parse_json_with_fallbacks(match.group(0))
            if parsed and "response" in parsed:
                results.append(parsed)
    
    # Strategy 4: Look for explicit verdict statements with strong indicators
    verdict_patterns = [
        (r'(?:the\s+)?(?:final\s+)?verdict\s*(?:is|:)\s*["\']?(correct|incorrect|partial|almost)["\']?', 1),
        (r'(?:the\s+)?(?:final\s+)?(?:grade|classification|label|assessment)\s*(?:is|:)\s*["\']?(correct|incorrect|partial|almost)["\']?', 1),
        (r'(?:i\s+(?:would|will)\s+)?(?:classify|grade|label|mark)\s*(?:this\s+)?(?:as\s+)?["\']?(correct|incorrect|partial|almost)["\']?', 1),
        (r'(?:this\s+(?:is|should\s+be))\s*["\']?(correct|incorrect|partial|almost)["\']?', 1),
        (r'["\']?response["\']?\s*[:=]\s*["\']?(correct|incorrect|partial|almost)["\']?', 1),
    ]
    for pattern, group_idx in verdict_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            try:
                value = match.group(group_idx).strip().lower()
                if value in VALID_LABELS:
                    results.append({"response": value})
            except Exception:
                continue
    
    # Strategy 5: Look for labels in backticks, bold, or emphasized
    formatting_patterns = [
        r'`(correct|incorrect|partial|almost)`',
        r'\*\*(correct|incorrect|partial|almost)\*\*',
        r'\*(correct|incorrect|partial|almost)\*',
        r'"(correct|incorrect|partial|almost)"',
        r"'(correct|incorrect|partial|almost)'",
    ]
    for pattern in formatting_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            try:
                value = match.group(1).strip().lower()
                if value in VALID_LABELS:
                    results.append({"response": value})
            except Exception:
                continue
    
    # Strategy 6: Look for standalone labels at the end (last 5 lines)
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if lines:
        for line in reversed(lines[-5:]):
            line_clean = line.lower().strip('"\'.,!?:;`*[]{}()')
            for label in VALID_LABELS:
                if line_clean == label:
                    results.append({"response": label})
                    break
    
    # Remove duplicates while preserving order
    seen = set()
    unique_results = []
    for r in results:
        resp_val = r.get("response", "")
        if resp_val and resp_val not in seen:
            seen.add(resp_val)
            unique_results.append(r)
    
    return unique_results if unique_results else None


def _try_parse_json_with_fallbacks(json_str: str) -> dict | None:
    """Try to parse JSON string with multiple fallback strategies.
    
    Handles common LLM output issues like:
    - Single vs double quotes
    - Trailing commas
    - Missing quotes around keys/values
    - Extra whitespace and newlines
    - Unicode characters
    """
    if not json_str or not isinstance(json_str, str):
        return None
    
    json_str = json_str.strip()
    
    # Try 1: Direct parsing
    try:
        parsed = json.loads(json_str)
        if isinstance(parsed, dict) and "response" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass
    
    # Try 2: Replace single quotes with double quotes (carefully)
    try:
        # Only replace single quotes that are likely JSON delimiters
        fixed = json_str.replace("'", '"')
        parsed = json.loads(fixed)
        if isinstance(parsed, dict) and "response" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass
    
    # Try 3: Remove trailing commas
    try:
        fixed = re.sub(r',\s*}', '}', json_str)
        fixed = re.sub(r',\s*]', ']', fixed)
        parsed = json.loads(fixed)
        if isinstance(parsed, dict) and "response" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass
    
    # Try 4: Fix common JSON syntax errors
    try:
        # Remove comments
        fixed = re.sub(r'//.*?\n', '\n', json_str)
        fixed = re.sub(r'/\*.*?\*/', '', fixed, flags=re.DOTALL)
        # Fix unquoted keys
        fixed = re.sub(r'(\w+):', r'"\1":', fixed)
        parsed = json.loads(fixed)
        if isinstance(parsed, dict) and "response" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass
    
    # Try 5: Extract response field with regex (more flexible)
    try:
        # Match response field with various quote styles
        match = re.search(r'["\']?response["\']?\s*[:=]\s*["\']?([^"\'\}\n,]+)', json_str, re.IGNORECASE)
        if match:
            value = match.group(1).strip().lower()
            # Clean up trailing punctuation
            value = re.sub(r'[.,;:!?\'"`]+$', '', value)
            if value in VALID_LABELS:
                return {"response": value}
    except Exception:
        pass
    
    # Try 6: Look for any valid label in the string
    try:
        text_lower = json_str.lower()
        for label in VALID_LABELS:
            if re.search(rf'\b{label}\b', text_lower):
                return {"response": label}
    except Exception:
        pass
    
    # Try 7: Handle escaped characters
    try:
        fixed = json_str.encode('utf-8').decode('unicode_escape')
        parsed = json.loads(fixed)
        if isinstance(parsed, dict) and "response" in parsed:
            return parsed
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass
    
    return None


def _normalize_label(value: str) -> str | None:
    """Normalize a label value to one of the valid labels.
    
    Handles various input formats including:
    - Direct label matches
    - Common misspellings and variations
    - Punctuation and whitespace
    - Partial matches and fuzzy matching
    """
    if not value or not isinstance(value, str):
        return None
    
    clean = value.strip().lower()
    
    # Remove common punctuation and whitespace variations
    clean_no_punct = re.sub(r'[^a-z\s]', '', clean).strip()
    clean_single = re.sub(r'[^a-z]', '', clean)
    
    # Direct match (exact)
    if clean_no_punct in VALID_LABELS:
        return LABEL_CANONICAL.get(clean_no_punct, clean_no_punct)
    
    # Check aliases (exact match)
    for label, aliases in LABEL_ALIASES.items():
        for alias in aliases:
            if clean_no_punct == alias.lower():
                return label
    
    # Check for partial matches in aliases
    for label, aliases in LABEL_ALIASES.items():
        for alias in aliases:
            alias_clean = re.sub(r'[^a-z]', '', alias.lower())
            if clean_single == alias_clean or clean_no_punct == alias.lower():
                return label
    
    # Fuzzy matching with priority order
    # Check for "incorrect" first (highest priority for safety)
    if clean_single == 'incorrect' or clean_single.startswith('incorr') or clean_single.startswith('wrong'):
        return 'incorrect'
    if 'notcorrect' in clean_single or 'notright' in clean_single:
        return 'incorrect'
    
    # Check for "almost" (before "partial" since "almost" is more specific)
    # "Almost" indicates a solution that is nearly complete with only tiny gaps
    if clean_single == 'almost' or clean_single.startswith('almost'):
        return 'almost'
    if clean_single == 'nearly' or clean_single.startswith('nearly'):
        return 'almost'
    if 'nearlycomplete' in clean_single or 'almostcomplete' in clean_single:
        return 'almost'
    if 'nearlycorrect' in clean_single or 'almostcorrect' in clean_single:
        return 'almost'
    if 'mostlycorrect' in clean_single:
        return 'almost'
    if 'minorgap' in clean_single or 'tinygap' in clean_single or 'smallgap' in clean_single:
        return 'almost'
    if 'nearlythere' in clean_single or 'almostthere' in clean_single:
        return 'almost'
    
    # Check for "partial"
    if clean_single == 'partial' or clean_single.startswith('partial'):
        return 'partial'
    if clean_single == 'incomplete' or clean_single.startswith('incomplete'):
        return 'partial'
    if 'some' in clean_single and 'somehow' not in clean_single:
        return 'partial'
    if clean_single == 'half' or clean_single.startswith('half'):
        return 'partial'
    if 'partly' in clean_single:
        return 'partial'
    
    # Check for "correct" (but not "incorrect")
    if clean_single == 'correct' or (clean_single.startswith('correct') and 'incorrect' not in clean_single):
        return 'correct'
    if clean_single == 'right' and 'wrong' not in clean_single:
        return 'correct'
    if clean_single == 'valid' and 'invalid' not in clean_single:
        return 'correct'
    
    # Check for negations that might indicate incorrect
    if 'no' in clean_single and ('credit' in clean_single or 'marks' in clean_single):
        return 'incorrect'
    if clean_single == 'zero':
        return 'incorrect'
    
    return None


def _extract_label_from_text(text: str) -> str | None:
    """Extract the grading label from raw text using pattern matching.
    
    Uses multiple priority strategies to find the most likely label:
    1. JSON blocks at the end (most reliable)
    2. Explicit declarations with colons/equals
    3. Formatted labels (backticks, bold)
    4. Labels at end of text
    5. Count-based with priority tie-breaking
    6. Last line matching
    """
    if not text or not isinstance(text, str):
        return None
        
    text_lower = text.lower().strip()
    lines = [line.strip() for line in text_lower.split('\n') if line.strip()]
    
    # Priority 0: Look for JSON block at the very end (most reliable)
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i].strip()
        if not line:
            continue
        # Check for JSON block format
        if '<json>' in line and '</json>' in line:
            start = line.find('<json>')
            end = line.find('</json>') + 7
            if start != -1 and end > start:
                inner = line[start + 6:end - 7].strip()
                parsed = _try_parse_json_with_fallbacks(inner)
                if parsed and "response" in parsed:
                    value = str(parsed["response"]).strip().lower()
                    if value in VALID_LABELS:
                        return LABEL_CANONICAL.get(value, value)
    
    # Priority 1: Look for explicit label declarations with colons
    label_alternatives = "correct|incorrect|partial|almost"
    declaration_patterns = [
        rf'(?:verdict|grade|label|classification|assessment|evaluation)\s*[:=]\s*["\']?({label_alternatives})["\']?\b',
        rf'(?:the\s+answer\s+is)\s*["\']?({label_alternatives})["\']?\b',
        rf'(?:this\s+is)\s*["\']?({label_alternatives})["\']?\b',
        rf'["\']?response["\']?\s*[:=]\s*["\']?({label_alternatives})["\']?\b',
        rf'(?:final\s+)?(?:answer|verdict|grade)\s*[:=]\s*["\']?({label_alternatives})["\']?\b',
    ]
    
    for pattern in declaration_patterns:
        matches = list(re.finditer(pattern, text_lower))
        if matches:
            # Use the LAST match (most likely to be the final verdict)
            last_match = matches[-1]
            for group in last_match.groups():
                if group in VALID_LABELS:
                    return LABEL_CANONICAL.get(group, group)
    
    # Priority 2: Look for labels in code blocks or backticks
    for label in VALID_LABELS:
        if re.search(rf'`{label}`', text_lower):
            return LABEL_CANONICAL.get(label, label)
        if re.search(rf'\*\*{label}\*\*|\*{label}\*', text_lower):
            return LABEL_CANONICAL.get(label, label)
        if re.search(rf'"{label}"|\'{label}\'', text_lower):
            return LABEL_CANONICAL.get(label, label)
    
    # Priority 3: Look for labels at the end of the text
    for label in VALID_LABELS:
        if re.search(rf'\b{label}\b[.!?]*\s*$', text_lower):
            return LABEL_CANONICAL.get(label, label)
    
    # Priority 4: Count occurrences and use priority to break ties
    label_counts = {}
    for label in VALID_LABELS:
        count = len(re.findall(rf'\b{label}\b', text_lower))
        label_counts[label] = count
    
    if any(label_counts.values()):
        valid_counts = {k: v for k, v in label_counts.items() if v > 0}
        if valid_counts:
            max_count = max(valid_counts.values())
            candidates = [k for k, v in valid_counts.items() if v == max_count]
            if len(candidates) == 1:
                return LABEL_CANONICAL.get(candidates[0], candidates[0])
            else:
                # Use priority to break ties
                best = max(candidates, key=lambda x: LABEL_PRIORITY.get(x, 0))
                return LABEL_CANONICAL.get(best, best)
    
    # Priority 5: Look at the last line for any valid label
    if lines:
        last_line = lines[-1]
        for label in VALID_LABELS:
            if label in last_line:
                return LABEL_CANONICAL.get(label, label)
    
    # Priority 6: Check last 3 lines for standalone labels with more flexible matching
    if lines:
        for line in reversed(lines[-3:]):
            line_clean = line.strip('"\'`*[]{}()')
            for label in VALID_LABELS:
                if line_clean == label:
                    return LABEL_CANONICAL.get(label, label)
    
    # Priority 7: Look for label mentions in analysis/verdict sections
    verdict_section_patterns = [
        r'(?:verdict|conclusion|assessment|grade)\s*[:\-]?\s*\n?\s*(correct|incorrect|partial|almost)',
        r'(?:therefore|thus|hence|so)\s*[,:]?\s*(correct|incorrect|partial|almost)',
    ]
    for pattern in verdict_section_patterns:
        matches = list(re.finditer(pattern, text_lower, re.IGNORECASE))
        if matches:
            last_match = matches[-1]
            for group in last_match.groups():
                if group in VALID_LABELS:
                    return LABEL_CANONICAL.get(group, group)
    
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
        
        # Enhanced instruction with chain-of-thought and better few-shot examples
        instruction = f"""You are an expert IMO mathematics grader. Your task is to evaluate a student's solution and classify it into EXACTLY ONE category.

## GRADING CATEGORIES

**"correct"** - Complete, correct solution with all steps valid and logically sound. Full marks.
**"incorrect"** - Fundamental flaw: no valid approach, only pattern matching from examples, or complete misunderstanding. Zero credit.
**"partial"** - Valid approach showing genuine understanding, but incomplete or has execution errors. Partial credit (typically 30-70%).
**"almost"** - Nearly complete solution (85%+), only minor gap or small error. High partial credit (typically 70-90%).

## CRITICAL DISTINCTIONS

### PARTIAL vs INCORRECT (Most Important!)
Ask: "Did the student demonstrate ANY valid mathematical understanding?"

**PARTIAL** (Valid insight demonstrated):
- Uses a correct theorem or formula, even if applied imperfectly
- Shows understanding of the problem structure
- Makes meaningful progress toward solution
- Has calculation errors but right approach
- Incomplete proof but valid technique shown
- **Key test**: Would a tutor say "You're on the right track, just need to fix X"?

**INCORRECT** (No valid understanding):
- Only checks small examples without general argument
- Uses completely wrong method (e.g., wrong formula)
- Fundamental mathematical misconception
- Random guessing or irrelevant work
- **Key test**: Would a tutor need to re-explain the problem from scratch?

**Golden Rule**: When in doubt between "partial" and "incorrect", choose "partial" if ANY valid mathematical thinking is present.

### PARTIAL vs ALMOST (Second Most Important!)
Ask: "How close is the solution to being complete and correct?"

**ALMOST** (85-95% complete, tiny gaps):
- Solution structure is essentially complete
- Main proof technique correctly applied
- Only minor details missing (e.g., edge case, explicit statement of obvious fact)
- Small calculation error that doesn't affect main result
- Missing a single verification step
- **Key test**: Would a tutor say "This is basically right, just needs a small fix"?

**PARTIAL** (30-70% complete, significant gaps):
- Valid approach but incomplete execution
- Missing major parts of the proof
- Significant calculation errors affecting result
- Incomplete verification or justification
- **Key test**: Would a tutor say "Good start, but you need to work on the rest"?

**Golden Rule**: "Almost" is for solutions that are nearly perfect. "Partial" is for solutions with valid ideas but significant work remaining.

## DETAILED FEW-SHOT EXAMPLES

### Example 1 - CORRECT (Complete rigorous proof)
Problem: Prove there are infinitely many primes.
Student: Suppose finitely many primes p₁,...,pₙ. Let N = p₁×...×pₙ + 1. For each pᵢ, N mod pᵢ = 1, so pᵢ doesn't divide N. Thus N has a prime factor not in our list. Contradiction.
Analysis: Complete proof with valid construction and logical conclusion. All steps correct.
Verdict: <json>{{"response": "correct"}}</json>

### Example 2 - INCORRECT (Pattern matching without proof)
Problem: Prove sum of first n odd numbers equals n².
Student: n=1: 1=1². n=2: 1+3=4=2². n=3: 1+3+5=9=3². The pattern holds for all n.
Analysis: Only checked examples, no general proof. No valid proof technique demonstrated. No understanding of WHY it works.
Verdict: <json>{{"response": "incorrect"}}</json>

### Example 3 - PARTIAL (Valid insight, incomplete execution)
Problem: Prove n³-n is divisible by 6 for all integers n.
Student: n³-n = n(n-1)(n+1). This is the product of 3 consecutive integers. One of them must be even, so divisible by 2.
Analysis: Excellent factorization and valid insight about divisibility by 2. Student understands the problem structure and key technique. Only missing: didn't prove divisibility by 3 (also true for 3 consecutive integers). Valid approach, incomplete proof.
Verdict: <json>{{"response": "partial"}}</json>

### Example 4 - INCORRECT (Fundamental mathematical error)
Problem: Find all real x where |x-1| + |x+1| = 2.
Student: |x-1| + |x+1| = |(x-1)+(x+1)| = |2x| = 2. So |x| = 1, meaning x = ±1.
Analysis: The step |a|+|b| = |a+b| is FALSE in general (triangle inequality is |a+b| ≤ |a|+|b|). This is a fundamental misunderstanding of absolute value properties. No valid approach demonstrated.
Verdict: <json>{{"response": "incorrect"}}</json>

### Example 5 - PARTIAL (Right method, execution error)
Problem: Find the number of diagonals in a convex n-gon.
Student: From each vertex, we can draw n-3 diagonals (to all non-adjacent vertices). With n vertices, total = n(n-3). For n=5: 5×2=10. But the actual answer is 5. I forgot to divide by 2 since each diagonal is counted twice.
Analysis: Correct method and formula structure. Valid understanding of the problem. Just a counting error (double counting). Shows genuine competence - the error is recognized and explained.
Verdict: <json>{{"response": "partial"}}</json>

### Example 6 - ALMOST (Nearly complete, tiny gap)
Problem: Prove the AM-GM inequality for two positive numbers.
Student: For a,b > 0, we want (a+b)/2 ≥ √ab. This is equivalent to a+b ≥ 2√ab, then (a+b)² ≥ 4ab, then a²+2ab+b² ≥ 4ab, so a²-2ab+b² ≥ 0, which is (a-b)² ≥ 0. This is always true.
Analysis: Perfect proof structure. Valid algebraic manipulation. Only minor issue: should explicitly state that squaring preserves inequality since a+b > 0, and that equality holds when a=b. Nearly complete, just needs these small additions.
Verdict: <json>{{"response": "almost"}}</json>

### Example 7 - PARTIAL (Valid technique, incomplete verification)
Problem: Find the maximum of f(x) = x(1-x) on [0,1].
Student: Take derivative: f'(x) = 1 - 2x. Set to 0: 1-2x=0, so x=1/2. Maximum is f(1/2) = 1/4.
Analysis: Correct technique (calculus), but student forgot to verify it's a maximum (could be min). Should check second derivative or endpoints. Valid approach, just incomplete verification. This is a significant gap, not a tiny one.
Verdict: <json>{{"response": "partial"}}</json>

### Example 8 - INCORRECT (Wrong approach entirely)
Problem: Prove that √2 is irrational.
Student: √2 ≈ 1.41421356... The decimal goes on forever without repeating, so it can't be written as a fraction. Therefore irrational.
Analysis: This is circular reasoning - the student assumes what they're trying to prove (that √2 has infinite non-repeating decimal). No valid proof technique. The standard proof by contradiction is not used.
Verdict: <json>{{"response": "incorrect"}}</json>

### Example 9 - PARTIAL (Good start, didn't finish)
Problem: Prove by induction that 1+2+...+n = n(n+1)/2.
Student: Base case: n=1, 1 = 1(2)/2 = 1 ✓. Inductive step: Assume true for n=k, so 1+...+k = k(k+1)/2. Then for n=k+1...
Analysis: Student correctly set up induction framework. Base case verified. Inductive hypothesis stated correctly. Just didn't complete the inductive step calculation. Clear understanding of the method, but significant work missing.
Verdict: <json>{{"response": "partial"}}</json>

### Example 10 - ALMOST (Complete proof, tiny omission)
Problem: Prove that for any real numbers a, b, c: a² + b² + c² ≥ ab + bc + ca.
Student: We have (a-b)² ≥ 0, so a² + b² ≥ 2ab. Similarly (b-c)² ≥ 0 gives b² + c² ≥ 2bc, and (c-a)² ≥ 0 gives c² + a² ≥ 2ca. Adding: 2(a²+b²+c²) ≥ 2(ab+bc+ca), so a²+b²+c² ≥ ab+bc+ca.
Analysis: Beautiful proof using sum of squares. All steps valid. Only tiny omission: should explicitly state that squares are always non-negative (≥ 0). The proof is essentially complete.
Verdict: <json>{{"response": "almost"}}</json>

### Example 11 - ALMOST (Correct answer, minor justification gap)
Problem: Find all positive integers n such that n² + 3n + 2 is prime.
Student: n² + 3n + 2 = (n+1)(n+2). For this to be prime, one factor must be 1. Since n+1 < n+2, we need n+1 = 1, so n = 0. But n must be positive, so no solutions.
Analysis: Correct factorization and reasoning. Only minor issue: should explicitly state that for a product to be prime, exactly one factor must equal 1 (and the other must be prime). Also should note that n=0 gives 2 which is prime, but 0 is not positive. Nearly complete.
Verdict: <json>{{"response": "almost"}}</json>

### Example 12 - PARTIAL (Correct approach, significant gap)
Problem: Prove that the sum of angles in a triangle is 180°.
Student: Draw a line through one vertex parallel to the opposite side. The alternate angles are equal, so the three angles at that vertex add to 180°.
Analysis: Correct geometric construction and insight. However, the student didn't fully explain WHY the alternate angles are equal, and didn't explicitly show how the triangle's angles correspond to the angles on the straight line. Valid approach but needs more detailed explanation.
Verdict: <json>{{"response": "partial"}}</json>

## CHAIN-OF-THOUGHT ANALYSIS PROCESS

Analyze the student's solution step by step:

1. **Understanding**: Does the student correctly understand what the problem is asking?
2. **Approach Validity**: Is their overall approach mathematically sound?
3. **Key Insight**: Did they identify the crucial insight or technique needed?
4. **Execution**: Are the calculations and logical steps correct?
5. **Completeness**: Is the solution fully finished, or are there gaps? (Estimate percentage: 30%, 70%, 90%, 100%?)
6. **Classification Decision**: Based on the above, which category fits best?
   - Complete and correct → "correct"
   - Nearly complete (85%+), tiny gap → "almost"
   - Valid approach but significant gaps (30-70%) → "partial"
   - No valid understanding → "incorrect"

After this analysis, make your classification decision.

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

STUDENT'S ANSWER:
```
{student_answer}
```

## YOUR TASK

First, provide your chain-of-thought analysis following the 6 steps above.

Then, output EXACTLY ONE of these JSON blocks (this must be the FINAL line of your response):

<json>{{"response": "correct"}}</json>
<json>{{"response": "incorrect"}}</json>
<json>{{"response": "partial"}}</json>
<json>{{"response": "almost"}}</json>

CRITICAL INSTRUCTIONS:
1. The JSON block MUST be on its own line at the very end
2. Use the EXACT format shown above with <json> and </json> tags
3. Do not add any text after the JSON block
4. Do not use markdown code blocks (```) around the JSON
5. The response field must be one of: correct, incorrect, partial, almost
6. When uncertain between partial and incorrect, prefer "partial" if any valid math is shown
7. When uncertain between partial and almost: "almost" is for 85%+ complete solutions with only tiny gaps; "partial" is for solutions with significant work remaining"""

        # Try up to 3 times to get a valid prediction
        max_retries = 3
        prediction = None
        msg_history = []
        
        for attempt in range(max_retries):
            try:
                # On retry, add a stronger reminder about output format
                msg_to_send = instruction
                if attempt > 0:
                    msg_to_send += '''\n\nCRITICAL: Your previous response did not contain a valid JSON block. You MUST output exactly one of these as the FINAL line:
<json>{"response": "correct"}</json>
<json>{"response": "incorrect"}</json>
<json>{"response": "partial"}</json>
<json>{"response": "almost"}</json>'''
                
                response, msg_history, info = get_response_from_llm(
                    msg=msg_to_send,
                    model=self.model,
                    msg_history=[],
                )
            except Exception as e:
                self.log_fn(f"Attempt {attempt + 1}: LLM call failed: {e}")
                continue

            text_to_parse = (response or "").strip()
            
            # Debug logging - log end of response where JSON should be
            self.log_fn(f"Attempt {attempt + 1}: Raw LLM response (last 500 chars): ...{text_to_parse[-500:]}")
            
            try:
                # PRIORITY 1: Extract from <json> blocks (most reliable)
                extracted = _extract_jsons(text_to_parse)
                if extracted:
                    self.log_fn(f"Attempt {attempt + 1}: Extracted {len(extracted)} JSON objects")
                    # Use the LAST extracted JSON (most likely to be the final verdict)
                    for item in reversed(extracted):
                        if isinstance(item, dict) and "response" in item:
                            raw_pred = str(item["response"]).strip().lower()
                            self.log_fn(f"Attempt {attempt + 1}: Found response field: {raw_pred}")
                            normalized = _normalize_label(raw_pred)
                            if normalized:
                                prediction = LABEL_CANONICAL.get(normalized, normalized)
                                self.log_fn(f"Attempt {attempt + 1}: Normalized to: {prediction}")
                                break
                
                # PRIORITY 2: If JSON extraction failed, try text extraction
                if prediction is None:
                    text_pred = _extract_label_from_text(text_to_parse)
                    if text_pred:
                        self.log_fn(f"Attempt {attempt + 1}: Extracted label from text: {text_pred}")
                        prediction = LABEL_CANONICAL.get(text_pred, text_pred)
                
                # PRIORITY 3: Look for high confidence patterns in raw text
                if prediction is None:
                    text_lower = text_to_parse.lower()
                    
                    # High confidence patterns - exact JSON format
                    high_confidence_patterns = [
                        (r'<json>\s*\{\s*"response"\s*:\s*"correct"\s*\}\s*</json>', 'correct'),
                        (r'<json>\s*\{\s*"response"\s*:\s*"incorrect"\s*\}\s*</json>', 'incorrect'),
                        (r'<json>\s*\{\s*"response"\s*:\s*"partial"\s*\}\s*</json>', 'partial'),
                        (r'<json>\s*\{\s*"response"\s*:\s*"almost"\s*\}\s*</json>', 'almost'),
                        (r'response\s*[=:]\s*"correct"', 'correct'),
                        (r'response\s*[=:]\s*"incorrect"', 'incorrect'),
                        (r'response\s*[=:]\s*"partial"', 'partial'),
                        (r'response\s*[=:]\s*"almost"', 'almost'),
                    ]
                    
                    for pattern, label in high_confidence_patterns:
                        if re.search(pattern, text_lower):
                            prediction = label
                            self.log_fn(f"Attempt {attempt + 1}: High confidence pattern matched: '{label}'")
                            break
                
                # PRIORITY 4: Look for standalone labels at the very end (last 3 lines)
                if prediction is None:
                    lines = [line.strip() for line in text_to_parse.split('\n') if line.strip()]
                    if lines:
                        # Check last 3 lines for standalone labels
                        for line in reversed(lines[-3:]):
                            line_lower = line.lower()
                            for label in VALID_LABELS:
                                # Match label possibly surrounded by quotes, punctuation, or markdown
                                if re.search(rf'^[\s"\'\*\`]*{label}[\s"\'\*\`\.,!?:;]*$', line_lower, re.IGNORECASE):
                                    prediction = label
                                    self.log_fn(f"Attempt {attempt + 1}: Found standalone label '{label}' at end")
                                    break
                            if prediction:
                                break
                
                # PRIORITY 5: Look for the LAST occurrence of any valid label in the text
                if prediction is None:
                    text_lower = text_to_parse.lower()
                    last_positions = {}
                    for label in VALID_LABELS:
                        matches = list(re.finditer(rf'\b{label}\b', text_lower))
                        if matches:
                            last_positions[label] = matches[-1].start()
                    
                    if last_positions:
                        # Get the label that appears last in the text
                        last_label = max(last_positions, key=last_positions.get)
                        prediction = last_label
                        self.log_fn(f"Attempt {attempt + 1}: Found label '{prediction}' as last occurrence")
                
                # PRIORITY 6: Confidence-based voting from multiple signals
                if prediction is None:
                    prediction = _confidence_based_extraction(text_to_parse)
                    if prediction:
                        self.log_fn(f"Attempt {attempt + 1}: Confidence-based extraction: '{prediction}'")
                
                # If we got a valid prediction, break out of retry loop
                if prediction in VALID_LABELS:
                    break
                    
            except Exception as e:
                self.log_fn(f"Attempt {attempt + 1}: Error extracting prediction: {e}")
                prediction = None
        
        # Final validation and canonicalization
        if prediction not in VALID_LABELS:
            self.log_fn(f"Invalid prediction '{prediction}', defaulting to 'incorrect'")
            prediction = "incorrect"
        else:
            # Map "almost" to "partial" for final output
            if prediction == "almost":
                prediction = "partial"
            self.log_fn(f"Final prediction: {prediction}")

        return str(prediction), msg_history


def _confidence_based_extraction(text: str) -> str | None:
    """Use confidence scoring to determine the best label from multiple signals.
    
    This function analyzes multiple signals in the text and assigns confidence
    scores to each label based on:
    - Position in text (end = higher confidence)
    - Formatting (JSON tags = higher confidence)
    - Explicit declarations (= higher confidence)
    - Frequency of mentions
    - Context clues (e.g., "nearly", "minor gap" for almost)
    """
    if not text or not isinstance(text, str):
        return None
    
    text_lower = text.lower()
    lines = text.split('\n')
    
    # Initialize confidence scores
    confidence = {label: 0.0 for label in VALID_LABELS}
    
    # Signal 1: JSON block at the end (highest confidence)
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i].strip()
        if '<json>' in line and '</json>' in line:
            for label in VALID_LABELS:
                if f'"response": "{label}"' in line or f"'response': '{label}'" in line:
                    confidence[label] += 10.0
                    break
            break
    
    # Signal 2: Explicit declarations
    declaration_patterns = [
        (r'(?:final\s+)?verdict\s*[:=]\s*["\']?(correct|incorrect|partial|almost)', 1.5),
        (r'(?:final\s+)?grade\s*[:=]\s*["\']?(correct|incorrect|partial|almost)', 1.5),
        (r'response\s*[:=]\s*["\']?(correct|incorrect|partial|almost)', 1.5),
        (r'classification\s*[:=]\s*["\']?(correct|incorrect|partial|almost)', 1.5),
    ]
    for pattern, weight in declaration_patterns:
        for match in re.finditer(pattern, text_lower):
            label = match.group(1)
            if label in VALID_LABELS:
                confidence[label] += weight
    
    # Signal 3: Labels in formatting (backticks, bold)
    format_patterns = [
        (rf'`(correct|incorrect|partial|almost)`', 1.2),
        (rf'\*\*(correct|incorrect|partial|almost)\*\*', 1.2),
        (rf'\*(correct|incorrect|partial|almost)\*', 1.0),
    ]
    for pattern, weight in format_patterns:
        for match in re.finditer(pattern, text_lower):
            label = match.group(1)
            if label in VALID_LABELS:
                confidence[label] += weight
    
    # Signal 4: Position-based scoring (labels near end get higher scores)
    text_length = len(text_lower)
    for label in VALID_LABELS:
        matches = list(re.finditer(rf'\b{label}\b', text_lower))
        for match in matches:
            # Closer to end = higher score
            position_score = match.start() / text_length
            confidence[label] += 0.5 * position_score
    
    # Signal 5: Frequency bonus
    for label in VALID_LABELS:
        count = len(re.findall(rf'\b{label}\b', text_lower))
        if count > 0:
            confidence[label] += 0.3 * min(count, 5)  # Cap at 5 mentions
    
    # Signal 6: Context clues for "almost" (nearly complete solutions)
    almost_context_patterns = [
        (r'\b(nearly complete|almost complete|mostly complete)\b', 0.8),
        (r'\b(nearly correct|almost correct|mostly correct)\b', 0.8),
        (r'\b(minor gap|tiny gap|small gap|minor issue|tiny issue)\b', 0.7),
        (r'\b(just needs|only needs|simply needs)\b', 0.6),
        (r'\b(essentially complete|basically complete)\b', 0.7),
        (r'\b(85%|90%|95% complete)\b', 0.8),
    ]
    for pattern, weight in almost_context_patterns:
        if re.search(pattern, text_lower):
            confidence['almost'] += weight
    
    # Signal 7: Context clues for "partial" (valid but incomplete)
    partial_context_patterns = [
        (r'\b(valid approach|correct approach|right approach)\b', 0.5),
        (r'\b(good start|good beginning|on the right track)\b', 0.5),
        (r'\b(incomplete|not finished|missing steps)\b', 0.4),
        (r'\b(significant gap|major gap|important part missing)\b', 0.4),
    ]
    for pattern, weight in partial_context_patterns:
        if re.search(pattern, text_lower):
            confidence['partial'] += weight
    
    # Signal 8: Context clues for "incorrect" (fundamental flaws)
    incorrect_context_patterns = [
        (r'\b(fundamental error|fundamental flaw|basic misunderstanding)\b', 0.6),
        (r'\b(circular reasoning|begging the question)\b', 0.7),
        (r'\b(no valid|no correct|no proper)\b', 0.5),
        (r'\b(only pattern matching|just examples|no general proof)\b', 0.6),
    ]
    for pattern, weight in incorrect_context_patterns:
        if re.search(pattern, text_lower):
            confidence['incorrect'] += weight
    
    # Find the label with highest confidence
    if any(confidence.values()):
        best_label = max(confidence, key=confidence.get)
        best_score = confidence[best_label]
        
        # Only return if confidence is above threshold
        if best_score >= 1.0:
            return best_label
    
    return None
