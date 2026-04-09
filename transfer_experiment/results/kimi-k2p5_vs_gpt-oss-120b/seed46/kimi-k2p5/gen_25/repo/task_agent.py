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

# Valid grading labels - includes "almost" which maps to "partial"
VALID_LABELS = ["correct", "incorrect", "partial", "almost"]

# Label priority for conflict resolution (higher = more preferred when uncertain)
# We prefer "incorrect" as the safest default when uncertain
LABEL_PRIORITY = {"incorrect": 4, "partial": 3, "almost": 2, "correct": 1}

# Mapping from labels to their canonical form
# "almost" is mapped to "partial" since it represents nearly complete solutions
LABEL_CANONICAL = {
    "correct": "correct",
    "incorrect": "incorrect", 
    "partial": "partial",
    "almost": "partial",  # almost is a variant of partial
}

# Common misspellings and variations mapping
LABEL_ALIASES = {
    "correct": ["correct", "right", "true", "valid", "accurate", "complete", "fully correct", "entirely correct", "perfect"],
    "incorrect": ["incorrect", "wrong", "false", "error", "invalid", "inaccurate", "flawed", "mistaken", "fundamentally wrong"],
    "partial": ["partial", "incomplete", "partially", "some", "part", "partial credit", "almost", "nearly", "mostly correct", "almost correct", "nearly complete"],
    "almost": ["almost", "nearly", "mostly correct", "almost correct", "nearly complete", "almost there"],
}


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks or raw JSON.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also tries to find JSON in code blocks and raw JSON objects.
    Includes robust error handling and multiple fallback strategies.
    Enhanced to handle more edge cases and malformed JSON.
    """
    if not text or not isinstance(text, str):
        return None
        
    results = []
    search_from = 0
    text_lower = text.lower()
    
    # Strategy 1: Try to find <json>...</json> blocks (case insensitive) - HIGHEST PRIORITY
    while True:
        start = text_lower.find("<json>", search_from)
        if start == -1:
            break
        end = text_lower.find("</json>", start)
        if end == -1:
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        # Try multiple parsing strategies for the content
        parsed = _try_parse_json_with_fallbacks(inner)
        if parsed and "response" in parsed:
            results.append(parsed)
    
    # Strategy 2: Try to find JSON in markdown code blocks
    code_block_patterns = [
        r'```(?:json)?\s*\n?(.*?)\n?```',
        r'```json\s*\n?(.*?)\n?```',
        r'```\s*\n?(.*?)\n?```',
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
        r'\{\s*["\']?response["\']?\s*:\s*["\']?(correct|incorrect|partial|almost)["\']?\s*\}',
    ]
    for pattern in json_patterns:
        for match in re.finditer(pattern, text, re.DOTALL | re.IGNORECASE):
            parsed = _try_parse_json_with_fallbacks(match.group(0))
            if parsed and "response" in parsed:
                results.append(parsed)
    
    # Strategy 4: Look for explicit verdict statements with label extraction
    verdict_patterns = [
        (r'(?:the\s+)?(?:final\s+)?verdict\s*(?:is|:)\s*["\']?(correct|incorrect|partial|almost)["\']?', 1),
        (r'(?:the\s+)?(?:final\s+)?(?:grade|classification|label|assessment)\s*(?:is|:)\s*["\']?(correct|incorrect|partial|almost)["\']?', 1),
        (r'(?:i\s+(?:would|will)\s+)?(?:classify|grade|label|mark)\s*(?:this\s+)?(?:as\s+)?["\']?(correct|incorrect|partial|almost)["\']?', 1),
        (r'(?:this\s+(?:is|should\s+be))\s*["\']?(correct|incorrect|partial|almost)["\']?', 1),
        (r'(?:the\s+answer\s+(?:is|should\s+be))\s*["\']?(correct|incorrect|partial|almost)["\']?', 1),
        (r'(?:therefore|thus|hence)[,:]?\s*(?:the\s+)?(?:answer\s+)?(?:is\s+)?["\']?(correct|incorrect|partial|almost)["\']?', 1),
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
    
    # Strategy 6: Look for standalone labels at the end of the response
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if lines:
        # Check last 3 lines for standalone labels
        for line in reversed(lines[-3:]):
            line_clean = line.lower().strip('"\'.,!?:;')
            for label in VALID_LABELS:
                if line_clean == label:
                    results.append({"response": label})
                    break
                # Check for "verdict: label" pattern
                if re.search(rf'(?:verdict|grade|label|classification)\s*[:\-]\s*["\']?{label}["\']?$', line, re.IGNORECASE):
                    results.append({"response": label})
                    break
    
    # Strategy 7: Look for "response:" patterns (without braces)
    simple_response_pattern = r'\bresponse\s*[:=]\s*["\']?(correct|incorrect|partial|almost)["\']?\b'
    for match in re.finditer(simple_response_pattern, text, re.IGNORECASE):
        try:
            value = match.group(1).strip().lower()
            if value in VALID_LABELS:
                results.append({"response": value})
        except Exception:
            continue
    
    # Strategy 8: Look for JSON with single quotes (common LLM output format)
    single_quote_json_pattern = r"\{\s*'response'\s*:\s*'(correct|incorrect|partial|almost)'\s*\}"
    for match in re.finditer(single_quote_json_pattern, text, re.IGNORECASE):
        try:
            value = match.group(1).strip().lower()
            if value in VALID_LABELS:
                results.append({"response": value})
        except Exception:
            continue
    
    # Strategy 9: Look for response field in various quote formats
    response_field_patterns = [
        r'["\']?response["\']?\s*:\s*["\']?(correct|incorrect|partial|almost)["\']?',
        r'response\s*=\s*["\']?(correct|incorrect|partial|almost)["\']?',
    ]
    for pattern in response_field_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            try:
                value = match.group(1).strip().lower()
                if value in VALID_LABELS:
                    results.append({"response": value})
            except Exception:
                continue
    
    # Strategy 10: Look for labels in parentheses or brackets
    bracket_patterns = [
        r'\((correct|incorrect|partial|almost)\)',
        r'\[(correct|incorrect|partial|almost)\]',
        r'\{(correct|incorrect|partial|almost)\}',
    ]
    for pattern in bracket_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            try:
                value = match.group(1).strip().lower()
                if value in VALID_LABELS:
                    results.append({"response": value})
            except Exception:
                continue
    
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
    
    Returns a dict with "response" field if successful, None otherwise.
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
    
    # Try 2: Replace single quotes with double quotes
    try:
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
    
    # Try 4: Extract response field with regex (more flexible)
    try:
        match = re.search(r'["\']?response["\']?\s*[:=]\s*["\']?([^"\'\}\n,]+)', json_str, re.IGNORECASE)
        if match:
            value = match.group(1).strip().lower()
            # Clean up the value - remove trailing punctuation
            value = re.sub(r'[.,;:!?\'"]+$', '', value)
            if value in VALID_LABELS:
                return {"response": value}
    except Exception:
        pass
    
    # Try 5: Look for any valid label in the string
    try:
        text_lower = json_str.lower()
        for label in VALID_LABELS:
            if re.search(rf'\b{label}\b', text_lower):
                return {"response": label}
    except Exception:
        pass
    
    # Try 6: Handle escaped quotes
    try:
        fixed = json_str.replace('\\"', '"').replace("\\'", "'")
        parsed = json.loads(fixed)
        if isinstance(parsed, dict) and "response" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass
    
    # Try 7: Handle newlines and extra whitespace in JSON
    try:
        fixed = re.sub(r'\s+', ' ', json_str)
        parsed = json.loads(fixed)
        if isinstance(parsed, dict) and "response" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass
    
    # Try 8: Extract from malformed JSON like {"response": correct} (missing quotes)
    try:
        match = re.search(r'["\']?response["\']?\s*:\s*([a-z]+)', json_str, re.IGNORECASE)
        if match:
            value = match.group(1).strip().lower()
            if value in VALID_LABELS:
                return {"response": value}
    except Exception:
        pass
    
    return None


def _normalize_label(value: str) -> str | None:
    """Normalize a label value to one of the valid labels.
    
    Handles common variations, misspellings, and edge cases.
    Maps "almost" and similar near-complete terms to "partial".
    """
    if not value or not isinstance(value, str):
        return None
    
    # Clean the value
    clean = value.strip().lower()
    # Remove punctuation and extra whitespace, but keep spaces for phrase matching
    clean_no_punct = re.sub(r'[^a-z\s]', '', clean).strip()
    
    # Direct match
    if clean_no_punct in VALID_LABELS:
        return LABEL_CANONICAL.get(clean_no_punct, clean_no_punct)
    
    # Check for multi-word phrases first
    for label, aliases in LABEL_ALIASES.items():
        for alias in aliases:
            if clean_no_punct == alias.lower():
                return label
    
    # Check single word aliases
    clean_single = re.sub(r'[^a-z]', '', clean)
    for label, aliases in LABEL_ALIASES.items():
        if clean_single in [a.lower().replace(' ', '') for a in aliases]:
            return label
    
    # Fuzzy matching for common misspellings
    # Check for partial match with correct (allow shorter matches)
    if 'correct' in clean_single or clean_single.startswith('corr') or clean_single.startswith('right'):
        # But make sure it's not "incorrect" or "partially correct"
        if 'incorrect' not in clean_single and 'partial' not in clean_single:
            return 'correct'
    # Check for partial match with incorrect
    if 'incorrect' in clean_single or clean_single.startswith('incorr') or clean_single.startswith('wrong'):
        return 'incorrect'
    # Check for "almost" first (more specific than partial)
    if 'almost' in clean_single or clean_single.startswith('alm') or 'nearly' in clean_single:
        return 'almost'
    # Check for partial match with partial
    if 'partial' in clean_single or clean_single.startswith('part') or 'incomplete' in clean_single:
        return 'partial'
    
    return None


def _extract_label_from_text(text: str) -> str | None:
    """Extract the grading label from raw text using pattern matching.
    
    Looks for exact matches of the valid labels with priority-based
    matching to find the most likely intended label.
    Uses multiple strategies with increasing fallback tolerance.
    Includes confidence scoring to prefer more explicit mentions.
    Maps "almost" and similar terms to "partial".
    """
    if not text or not isinstance(text, str):
        return None
        
    text_lower = text.lower().strip()
    lines = [line.strip() for line in text_lower.split('\n') if line.strip()]
    
    # Priority 0: Look for explicit <json> blocks with response field
    json_block_pattern = r'<json>\s*\{\s*"response"\s*:\s*"([^"]+)"\s*\}\s*</json>'
    for match in re.finditer(json_block_pattern, text_lower):
        clean_match = re.sub(r'[^a-z]', '', match.group(1).lower())
        if clean_match in VALID_LABELS:
            return LABEL_CANONICAL.get(clean_match, clean_match)
    
    # Priority 0b: Look for any JSON-like pattern with response field
    json_flexible_pattern = r'["\']?response["\']?\s*[:=]\s*["\']?([^"\'\}\n,]+)'
    for match in re.finditer(json_flexible_pattern, text_lower):
        clean_match = re.sub(r'[^a-z]', '', match.group(1).lower())
        if clean_match in VALID_LABELS:
            return LABEL_CANONICAL.get(clean_match, clean_match)
    
    # Priority 1: Look for explicit label declarations with colons or equals
    # 4 categories: correct, incorrect, partial, almost
    label_alternatives = "correct|incorrect|partial|almost"
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
                    return LABEL_CANONICAL.get(group, group)
    
    # Priority 4: Look for labels in code blocks or backticks
    for label in VALID_LABELS:
        # Match backtick-quoted labels
        if re.search(rf'`{label}`', text_lower):
            return LABEL_CANONICAL.get(label, label)
        # Match bold/italic markdown labels
        if re.search(rf'\*\*{label}\*\*|\*{label}\*', text_lower):
            return LABEL_CANONICAL.get(label, label)
        # Match quoted labels
        if re.search(rf'"{label}"|\'{label}\'', text_lower):
            return LABEL_CANONICAL.get(label, label)
    
    # Priority 5: Look for labels at the end of sentences or standalone
    for label in VALID_LABELS:
        # Match at end of sentence (with optional punctuation and whitespace)
        if re.search(rf'\b{label}\b[.!?]*\s*$', text_lower):
            return LABEL_CANONICAL.get(label, label)
        # Match after "therefore", "thus", "so", "hence", "conclusion"
        if re.search(rf'(?:therefore|thus|so|hence|conclusion|concluding)[,:]?\s+\b{label}\b', text_lower):
            return LABEL_CANONICAL.get(label, label)
        # Match after "final" or "final answer"
        if re.search(rf'(?:final(?:ly)?|final\s+answer|in\s+conclusion)[,:]?\s+\b{label}\b', text_lower):
            return LABEL_CANONICAL.get(label, label)
    
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
                    return LABEL_CANONICAL.get(group, group)
    
    # Priority 7: Look for "verdict:" or "verdict is" patterns
    verdict_patterns = [
        rf'verdict\s*[:=]\s*["\']?({label_alternatives})["\']?',
        rf'verdict\s+is\s*["\']?({label_alternatives})["\']?',
        rf'analysis\s*[:=]\s*["\']?({label_alternatives})["\']?',
        rf'assessment\s*[:=]\s*["\']?({label_alternatives})["\']?',
    ]
    for pattern in verdict_patterns:
        match = re.search(pattern, text_lower)
        if match:
            for group in match.groups():
                if group in VALID_LABELS:
                    return LABEL_CANONICAL.get(group, group)
    
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
                return LABEL_CANONICAL.get(candidates[0], candidates[0])
            else:
                # Tie-break by priority (higher priority wins)
                best = max(candidates, key=lambda x: LABEL_PRIORITY.get(x, 0))
                return LABEL_CANONICAL.get(best, best)
    
    # Priority 9: Fuzzy matching - look for labels with minor typos
    for label in VALID_LABELS:
        # Look for substrings that are close to the label
        if label in text_lower:
            return LABEL_CANONICAL.get(label, label)
    
    # Priority 10: Look at the last line of the text
    # LLMs often put their final answer at the end
    if lines:
        last_line = lines[-1]
        for label in VALID_LABELS:
            if label in last_line:
                return LABEL_CANONICAL.get(label, label)
    
    # Priority 11: Look for labels in the last few lines
    if len(lines) >= 2:
        for line in reversed(lines[-3:]):  # Check last 3 lines
            for label in VALID_LABELS:
                if label in line:
                    return LABEL_CANONICAL.get(label, label)
    
    # Priority 12: Look for standalone labels (just the word on a line)
    for label in VALID_LABELS:
        # Match lines that contain only the label (possibly with quotes)
        standalone_pattern = rf'^\s*["\']?{label}["\']?\s*$'
        for line in lines:
            if re.search(standalone_pattern, line, re.IGNORECASE):
                return LABEL_CANONICAL.get(label, label)
    
    # NEW Priority 13: Look for "Verdict:" or "Analysis:" followed by label on same line
    for line in lines:
        for label in VALID_LABELS:
            # Match patterns like "Verdict: correct" or "Analysis: incorrect"
            if re.search(rf'(?:verdict|analysis|conclusion|assessment|grade|label)\s*[:\-]\s*["\']?{label}["\']?\b', line, re.IGNORECASE):
                return LABEL_CANONICAL.get(label, label)
    
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
        
        instruction = f"""You are an expert IMO (International Mathematical Olympiad) mathematics grader. Your task is to evaluate student answers and classify them into EXACTLY ONE category: "correct", "incorrect", "partial", or "almost".

## GRADING CATEGORIES (DECISION TREE)

### "correct" - COMPLETE AND CORRECT SOLUTION
**Criteria (ALL must be met):**
- Final answer is mathematically correct and matches the official solution
- ALL key mathematical steps are present and logically sound
- Complete reasoning with no gaps or missing justifications
- Valid proof or derivation from start to finish

### "incorrect" - FUNDAMENTALLY FLAWED
**Criteria (ANY of these indicate "incorrect"):**
- Critical logical errors or fundamental misconceptions
- Wrong approach from the very start (no valid mathematical insight)
- No valid mathematical reasoning demonstrated
- Final answer is wrong AND the approach is invalid
- Misunderstanding of the problem statement

**Key distinction:** The student shows NO genuine mathematical understanding of how to solve the problem.

### "partial" - MEANINGFUL PROGRESS BUT INCOMPLETE
**Criteria (ANY of these indicate "partial"):**
- Valid approach started with correct key insights
- Shows genuine mathematical understanding of the problem
- Significant gaps remain, missing steps, or incomplete reasoning
- WRONG final answer BUT valid approach and understanding
- Good start but didn't complete the solution
- Correct approach but calculation errors in execution

**Key distinction:** The student demonstrates UNDERSTANDING of how to solve the problem, even if they didn't finish or made execution errors.

### "almost" - NEARLY COMPLETE (8-9 points worth)
- Solution is very close to complete, just minor gaps
- Correct approach, nearly all steps present
- Small missing detail or minor error in final step
- Could be fixed with a small addition

## CRITICAL DECISION RULES

**Rule 1:** If student shows UNDERSTANDING (valid approach) → "partial" or "almost", NOT "incorrect"
**Rule 2:** "incorrect" is ONLY for fundamental flaws, not execution errors or incomplete work
**Rule 3:** When in doubt between "partial" and "incorrect", choose "partial" if any valid insight exists
**Rule 4:** "almost" is for solutions that are 90%+ complete with only tiny gaps
**Rule 5:** "partial" is for solutions with valid approach but significant gaps (50-90% complete)

## FEW-SHOT EXAMPLES

### Example 1: Correct - Complete Proof
Problem: Prove infinitely many primes.
Official: Assume finitely many primes p₁,...,pₙ. Let N = p₁...pₙ + 1. N not divisible by any pᵢ, contradiction.
Student: Suppose finitely many primes p₁,...,pₙ. Let N = p₁×...×pₙ + 1. N mod pᵢ = 1 for all i, so N has a prime factor not in list. Contradiction.
Analysis: Valid approach, complete reasoning, correct conclusion.
Verdict: correct

### Example 2: Incorrect - Fundamental Misconception
Problem: Find n where n²+n+41 is always prime.
Official: At n=40: 40²+40+41=1681=41², not prime.
Student: For n=1: 43 (prime). n=2: 47 (prime). Pattern continues since 41 is prime and n²+n is even, so result is odd and prime.
Analysis: Pattern matching without proof. No valid approach to check all cases. Fundamental error.
Verdict: incorrect

### Example 3: Partial - Valid Approach, Incomplete
Problem: Prove a²+b²+c² ≥ 4√3 × Area for any triangle.
Official: Uses Heron's formula and AM-GM.
Student: For equilateral triangle side s: Area = (√3/4)s². LHS = 3s², RHS = 4√3 × (√3/4)s² = 3s². Equality holds. For other triangles, LHS grows relative to area.
Analysis: Valid insight (checking special case), but no general proof. Shows understanding but incomplete.
Verdict: partial

### Example 4: Incorrect - Wrong Approach
Problem: Prove √2 is irrational.
Official: Assume √2 = p/q lowest terms. Then 2q² = p², so p even, p=2k, then q even. Contradiction.
Student: √2 ≈ 1.41421356... The decimal never repeats, so √2 cannot be written as p/q. Therefore irrational.
Analysis: No valid proof structure. "Never repeats" is not a proof. Wrong approach.
Verdict: incorrect

### Example 5: Partial - Right Approach, Missing Completion
Problem: Circular arrangements of 6 people (rotations same).
Official: Fix one person, arrange 5 others: 5! = 120.
Student: Fix one person's position for circular symmetry. Arrange remaining 5 people: 5! ways.
Analysis: Correct approach, valid reasoning, but answer not computed (5! not evaluated). Shows understanding.
Verdict: partial

### Example 6: Correct - Different Valid Method
Problem: Solve x² - 5x + 6 = 0.
Official: (x-2)(x-3) = 0, so x = 2 or 3.
Student: Using quadratic formula: x = (5 ± √(25-24))/2 = (5 ± 1)/2. So x = 3 or 2.
Analysis: Different but valid method. Complete and correct.
Verdict: correct

### Example 7: Incorrect - Misunderstanding Problem
Problem: Choose 3 people from 10.
Official: C(10,3) = 120.
Student: Pick first in 10 ways, second in 9, third in 8: 10×9×8 = 720.
Analysis: Confuses permutations with combinations. Fundamental misunderstanding of "choose" (order doesn't matter).
Verdict: incorrect

### Example 8: Partial - Incomplete Induction
Problem: Prove 1+2+...+n = n(n+1)/2 by induction.
Official: Base n=1: 1=1(2)/2 ✓. Assume for n=k. For n=k+1: 1+...+k+(k+1) = k(k+1)/2 + (k+1) = (k+1)(k+2)/2. ✓
Student: Base: n=1, LHS=1, RHS=1. Check. Inductive step: Assume 1+...+k = k(k+1)/2. Then for k+1: add (k+1) to both sides.
Analysis: Correct setup for induction, but inductive step not completed. Shows understanding of method.
Verdict: partial

### Example 9: Almost - Nearly Complete
Problem: Find maximum of sin(x) + cos(x).
Official: sin(x)+cos(x) = √2 sin(x+π/4). Max is √2.
Student: sin(x)+cos(x) = √2[sin(x)/√2 + cos(x)/√2] = √2 sin(x+π/4). Max of sin is 1, so max is √2.
Analysis: Complete solution with all steps. Correct.
Verdict: correct

### Example 10: Partial - Wrong Answer, Valid Approach
Problem: Find sum of n where n²-19n+99 is perfect square.
Official: Solutions n=1,9,10,18. Sum=38.
Student: Let n²-19n+99=k². Using quadratic formula: n=(19±√(4k²-35))/2. For k=3: n=10,9. k=4,5: not squares. Answer: 19.
Analysis: Valid approach (setting up equation), good method (quadratic formula), but missed some solutions and wrong final sum. Shows understanding.
Verdict: partial

### Example 11: Partial - Good Insight, Wrong Execution
Problem: Find all primes p where p+2 and p+6 are also prime.
Official: p=5 (5,7,11 all prime). Check p=3: 5,9 (9 not prime). p=7: 9,13 (9 not prime). Only p=5.
Student: For p=5: 7 and 11 are prime. For p>3, p mod 3 is 1 or 2. If p≡1 mod 3, p+2≡0 mod 3. If p≡2 mod 3, p+4≡0 mod 3. So only p=3,5 possible. p=3 gives 5,9 (not prime). So p=5.
Analysis: Valid modular arithmetic insight, but error in logic (said p+4 instead of p+6). Shows understanding but execution error.
Verdict: partial

### Example 12: Incorrect - No Valid Approach
Problem: Prove sum of first n odd numbers is n².
Official: 1+3+5+...+(2n-1) = n². By induction or pairing: 1+(2n-1)=2n, 3+(2n-3)=2n, etc. n pairs each sum to 2n, total n².
Student: n=1: 1=1². n=2: 1+3=4=2². n=3: 1+3+5=9=3². Pattern holds.
Analysis: Only checked examples, no proof. No valid approach for general case.
Verdict: incorrect

### Example 13: Partial vs Incorrect - Critical Distinction
Problem: Prove that for any positive integer n, n³ - n is divisible by 6.
Official: n³ - n = n(n-1)(n+1). Product of 3 consecutive integers. Among any 3 consecutive integers: one divisible by 3, at least one even (divisible by 2). So divisible by 6.
Student A (PARTIAL): n³ - n = n(n²-1) = n(n-1)(n+1). This is product of 3 consecutive integers. One must be even, so divisible by 2. [Stops here - didn't prove divisibility by 3]
Analysis: Valid approach, correct factorization, good insight about consecutive integers. Missing the divisibility by 3 argument. Shows understanding.
Verdict: partial

Student B (INCORRECT): n³ - n = n(n²-1). For n=2: 2(3)=6, divisible by 6. For n=3: 3(8)=24, divisible by 6. For n=4: 4(15)=60, divisible by 6. Pattern holds for all n.
Analysis: Only checked examples, no general proof. No valid approach to prove for all n.
Verdict: incorrect

### Example 14: Partial - Valid Approach, Wrong Final Answer
Problem: Find the number of ways to arrange 5 people in a line where A and B are not adjacent.
Official: Total arrangements = 5! = 120. Arrangements with A,B adjacent = 2 × 4! = 48. Answer = 120 - 48 = 72.
Student: Total arrangements = 5! = 120. Treat A and B as one unit, so 4 units total. Arrangements = 4! = 24. But A and B can swap, so 24 × 2 = 48. Answer: 48.
Analysis: Valid approach (complementary counting), correct method for adjacent arrangements, but forgot to subtract from total. Shows understanding of the method.
Verdict: partial

### Example 15: Partial - Correct Setup, Incomplete Execution
Problem: Find the sum of all positive integers n such that n² + 3n + 2 is a perfect square.
Official: n² + 3n + 2 = (n+1)(n+2). For this to be a perfect square, need (n+1)(n+2) = k². Since gcd(n+1,n+2)=1, both n+1 and n+2 must be perfect squares. Let n+1=a², n+2=b². Then b²-a²=1, so (b-a)(b+a)=1. Only solution: b=1,a=0, giving n=-1 (not positive). No solutions.
Student: n² + 3n + 2 = (n+1)(n+2). For this to be a perfect square, since n+1 and n+2 are consecutive, their product is rarely a square. Check small cases: n=1: 6 (not square), n=2: 12 (not square), n=7: 72 (not square). Seems like no solutions.
Analysis: Valid insight about factorization, checking small cases shows understanding. But no proof that no solutions exist. Incomplete.
Verdict: partial

### Example 16: Partial - Valid Approach, Significant Progress
Problem: Prove that for any triangle with sides a, b, c: a² + b² + c² ≥ 4√3 × Area.
Official: Uses Heron's formula and AM-GM inequality to prove.
Student: For equilateral triangle with side s: Area = (√3/4)s². LHS = 3s², RHS = 4√3 × (√3/4)s² = 3s². Equality holds. For isosceles triangle with sides a, a, b: Area = (b/4)√(4a²-b²). Need to show 2a² + b² ≥ √3 × b × √(4a²-b²). This is true by AM-GM.
Analysis: Valid approach checking special cases, shows understanding of the inequality. Uses proper mathematical techniques (AM-GM). Significant progress but not a complete general proof.
Verdict: partial

### Example 17: Incorrect - Pattern Matching Without Proof
Problem: Prove that n⁵ - n is divisible by 30 for all positive integers n.
Official: n⁵ - n = n(n⁴-1) = n(n²-1)(n²+1) = n(n-1)(n+1)(n²+1). Among n-1, n, n+1: one divisible by 3, at least one even. Also n⁵ ≡ n (mod 5) by Fermat's Little Theorem. So divisible by 2×3×5 = 30.
Student: n=1: 0, divisible. n=2: 30, divisible. n=3: 240, divisible. n=4: 1020, divisible. Pattern continues for all n.
Analysis: Only checked examples, no general proof. No valid approach to prove for all n.
Verdict: incorrect

### Example 18: Partial - Correct Method, Wrong Calculation
Problem: Find the number of diagonals in a convex n-gon.
Official: From each vertex, can draw n-3 diagonals (to all non-adjacent vertices). Total = n(n-3)/2 (divide by 2 to avoid double counting).
Student: From each vertex, n-3 diagonals. Total = n(n-3). For n=5: 5×2=10. But actual is 5. Forgot to divide by 2.
Analysis: Correct method and understanding, but calculation error (forgot division by 2). Shows understanding of the concept.
Verdict: partial

### Example 19: Incorrect - Fundamental Misunderstanding
Problem: Find all real x such that |x-1| + |x+1| = 2.
Official: Case analysis: x ≤ -1: -(x-1)-(x+1)=2 → -2x=2 → x=-1. -1 < x ≤ 1: -(x-1)+(x+1)=2 → 2=2, all x in [-1,1] work. x > 1: (x-1)+(x+1)=2 → 2x=2 → x=1. Solution: [-1,1].
Student: |x-1| + |x+1| = |x-1+x+1| = |2x| = 2. So |x| = 1, meaning x = ±1.
Analysis: Fundamental error: |a| + |b| ≠ |a+b| in general. Misunderstands absolute value properties.
Verdict: incorrect

### Example 20: Partial - Good Start, Missing Completion
Problem: Prove that if a, b, c are sides of a triangle, then a/(b+c) + b/(c+a) + c/(a+b) < 2.
Official: Uses substitution and Nesbitt's inequality variations.
Student: By triangle inequality, b+c > a, so a/(b+c) < 1. Similarly b/(c+a) < 1 and c/(a+b) < 1. So sum < 3. Need to show it's actually < 2.
Analysis: Valid insight using triangle inequality, correct approach. Shows understanding but didn't complete the proof to show sum < 2.
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

Work through these questions in order:

1. **Understanding**: Does the student correctly understand what the problem is asking?
   - If NO → likely "incorrect"
   - If YES → continue

2. **Approach Validity**: Is the student's overall approach mathematically valid?
   - If NO valid approach at all → "incorrect"
   - If YES, valid approach exists → continue

3. **Key Insights**: Does the student demonstrate genuine mathematical insight?
   - If YES, has good ideas but incomplete → "partial" or "almost"
   - If NO insights, just guessing → "incorrect"

4. **Execution**: Are the calculations/proof steps correct?
   - If errors in execution but approach valid → "partial"
   - If correct → continue

5. **Completeness**: Is the solution fully complete?
   - If complete and correct → "correct"
   - If 90%+ complete, minor gap → "almost"
   - If significant gaps but valid work → "partial"

6. **Final Answer**: Is the final answer correct?
   - Correct answer + complete reasoning → "correct"
   - Correct answer + incomplete reasoning → "partial" or "almost"
   - Wrong answer + valid approach → "partial"
   - Wrong answer + no valid approach → "incorrect"

## PARTIAL vs INCORRECT - CRITICAL DISTINCTION

The most important distinction is between "partial" and "incorrect":

**Choose "partial" when:**
- Student uses a valid mathematical technique (induction, contradiction, case analysis, etc.)
- Student shows understanding of the problem structure
- Student makes progress toward solution but doesn't finish
- Student has calculation errors but correct method
- Student proves special cases but not general case
- Student sets up correct equations but doesn't solve them

**Choose "incorrect" when:**
- Student only checks examples without proof (pattern matching)
- Student uses completely wrong technique for the problem type
- Student misunderstands what the problem is asking
- Student makes claims without any mathematical justification
- Student's reasoning is logically flawed at a fundamental level

**Golden Rule:** If you're unsure, ask: "Does this student show they know HOW to solve this problem, even if they didn't finish?" If YES → "partial". If NO → "incorrect".

## OUTPUT FORMAT (CRITICAL - FOLLOW EXACTLY)

You MUST output ONLY a JSON object wrapped in <json> tags. No other text before or after.

<json>
{{
    "response": "correct" | "incorrect" | "partial" | "almost"
}}
</json>

STRICT REQUIREMENTS:
1. ONLY output the JSON block shown above
2. Use double quotes for the JSON
3. The value must be exactly one of: "correct", "incorrect", "partial", "almost"
4. The JSON block must be the VERY LAST thing in your response
5. Do not include any explanation, analysis, or other text after the JSON
6. "almost" will be mapped to "partial" for final grading

Your verdict: <json>{{"response": "..."}}</json>"""

        # Try up to 3 times to get a valid prediction
        max_retries = 3
        prediction = None
        msg_history = []
        
        for attempt in range(max_retries):
            try:
                # On retry, add a reminder about output format
                msg_to_send = instruction
                if attempt > 0:
                    msg_to_send += "\n\nIMPORTANT: Remember to output ONLY the JSON block in <json> tags. No other text."
                
                response, msg_history, info = get_response_from_llm(
                    msg=msg_to_send,
                    model=self.model,
                    msg_history=[],
                )
            except Exception as e:
                self.log_fn(f"Attempt {attempt + 1}: LLM call failed: {e}")
                continue

            text_to_parse = (response or "").strip()
            
            # Debug logging
            self.log_fn(f"Attempt {attempt + 1}: Raw LLM response (first 500 chars): {text_to_parse[:500]}")
            
            try:
                # First try to extract from JSON blocks
                extracted = _extract_jsons(text_to_parse)
                if extracted:
                    self.log_fn(f"Attempt {attempt + 1}: Extracted {len(extracted)} JSON objects from response")
                    for item in extracted:
                        if isinstance(item, dict) and "response" in item:
                            raw_pred = str(item["response"]).strip()
                            self.log_fn(f"Attempt {attempt + 1}: Found response field with value: {raw_pred}")
                            # Use normalize function for robust matching
                            normalized = _normalize_label(raw_pred)
                            if normalized:
                                prediction = LABEL_CANONICAL.get(normalized, normalized)
                                break
                
                # If JSON extraction failed, try text extraction
                if prediction is None:
                    text_pred = _extract_label_from_text(text_to_parse)
                    if text_pred:
                        self.log_fn(f"Attempt {attempt + 1}: Extracted label from text: {text_pred}")
                        prediction = LABEL_CANONICAL.get(text_pred, text_pred)
                
                # If still no prediction, check msg_history as last resort
                if prediction is None and msg_history:
                    self.log_fn(f"Attempt {attempt + 1}: Trying to extract from msg_history ({len(msg_history)} messages)")
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
                                            prediction = LABEL_CANONICAL.get(normalized, normalized)
                                            break
                            
                            if prediction is None:
                                text_pred = _extract_label_from_text(last_content)
                                if text_pred:
                                    prediction = LABEL_CANONICAL.get(text_pred, text_pred)
                                    break
                            else:
                                break
                
                # Last resort: look for any valid label in the text with word boundaries
                if prediction is None:
                    text_lower = text_to_parse.lower()
                    # Look for labels as whole words
                    for label in VALID_LABELS:
                        if re.search(rf'\b{label}\b', text_lower):
                            self.log_fn(f"Attempt {attempt + 1}: Found label '{label}' as whole word in text")
                            prediction = label
                            break
                
                # Extra fallback: look for common variations or misspellings with confidence scoring
                if prediction is None:
                    text_lower = text_to_parse.lower()
                    
                    # High confidence patterns (explicit statements)
                    high_confidence_patterns = [
                        (r'(?:the\s+)?(?:final\s+)?verdict\s*(?:is|:)\s*["\']?incorrect["\']?', 'incorrect'),
                        (r'(?:the\s+)?(?:final\s+)?verdict\s*(?:is|:)\s*["\']?partial["\']?', 'partial'),
                        (r'(?:the\s+)?(?:final\s+)?verdict\s*(?:is|:)\s*["\']?almost["\']?', 'almost'),
                        (r'(?:the\s+)?(?:final\s+)?verdict\s*(?:is|:)\s*["\']?correct["\']?', 'correct'),
                        (r'\bgrade\s*(?:is|:)\s*["\']?incorrect["\']?', 'incorrect'),
                        (r'\bgrade\s*(?:is|:)\s*["\']?partial["\']?', 'partial'),
                        (r'\bgrade\s*(?:is|:)\s*["\']?almost["\']?', 'almost'),
                        (r'\bgrade\s*(?:is|:)\s*["\']?correct["\']?', 'correct'),
                        (r'(?:decision|assessment|evaluation|rating)\s*(?:is|:)\s*["\']?incorrect["\']?', 'incorrect'),
                        (r'(?:decision|assessment|evaluation|rating)\s*(?:is|:)\s*["\']?partial["\']?', 'partial'),
                        (r'(?:decision|assessment|evaluation|rating)\s*(?:is|:)\s*["\']?almost["\']?', 'almost'),
                        (r'(?:decision|assessment|evaluation|rating)\s*(?:is|:)\s*["\']?correct["\']?', 'correct'),
                        (r'\bclassification\s*(?:is|:)\s*["\']?incorrect["\']?', 'incorrect'),
                        (r'\bclassification\s*(?:is|:)\s*["\']?partial["\']?', 'partial'),
                        (r'\bclassification\s*(?:is|:)\s*["\']?almost["\']?', 'almost'),
                        (r'\bclassification\s*(?:is|:)\s*["\']?correct["\']?', 'correct'),
                    ]
                    
                    for pattern, label in high_confidence_patterns:
                        if re.search(pattern, text_lower):
                            prediction = label
                            self.log_fn(f"Attempt {attempt + 1}: High confidence pattern matched: '{label}'")
                            break
                    
                    # Medium confidence patterns (indicators in text)
                    if prediction is None:
                        incorrect_indicators = ['fundamentally wrong', 'wrong approach', 'misconception', 'does not understand', 'no understanding', 'fundamental error', 'completely wrong', 'not correct']
                        partial_indicators = ['incomplete', 'partial credit', 'partially correct', 'missing steps', 'not finished', 'incomplete solution', 'partial solution', 'not complete']
                        almost_indicators = ['almost', 'nearly', 'almost correct', 'nearly complete', 'almost there', 'minor gap']
                        correct_indicators = ['fully correct', 'completely correct', 'entirely correct', 'perfect solution']
                        
                        for indicator in incorrect_indicators:
                            if indicator in text_lower:
                                prediction = 'incorrect'
                                self.log_fn(f"Attempt {attempt + 1}: Medium confidence fallback: detected 'incorrect' via indicator '{indicator}'")
                                break
                        
                        if prediction is None:
                            for indicator in almost_indicators:
                                if indicator in text_lower:
                                    prediction = 'almost'
                                    self.log_fn(f"Attempt {attempt + 1}: Medium confidence fallback: detected 'almost' via indicator '{indicator}'")
                                    break
                        
                        if prediction is None:
                            for indicator in partial_indicators:
                                if indicator in text_lower:
                                    prediction = 'partial'
                                    self.log_fn(f"Attempt {attempt + 1}: Medium confidence fallback: detected 'partial' via indicator '{indicator}'")
                                    break
                        
                        if prediction is None:
                            for indicator in correct_indicators:
                                if indicator in text_lower:
                                    prediction = 'correct'
                                    self.log_fn(f"Attempt {attempt + 1}: Medium confidence fallback: detected 'correct' via indicator '{indicator}'")
                                    break
                
                # Look for standalone labels at the very end of the response
                if prediction is None:
                    lines = [line.strip() for line in text_to_parse.split('\n') if line.strip()]
                    if lines:
                        last_line = lines[-1].lower()
                        for label in VALID_LABELS:
                            # Check if the last line is just the label (possibly with quotes or punctuation)
                            if re.search(rf'^["\']?{label}["\']?[.!?]*\s*$', last_line, re.IGNORECASE):
                                prediction = label
                                self.log_fn(f"Attempt {attempt + 1}: Found standalone label '{label}' at end of response")
                                break
                
                # Look for labels in the last few non-empty lines (broader search)
                if prediction is None:
                    lines = [line.strip() for line in text_to_parse.split('\n') if line.strip()]
                    # Check last 5 lines for any valid label
                    for line in reversed(lines[-5:] if len(lines) >= 5 else lines):
                        line_lower = line.lower()
                        for label in VALID_LABELS:
                            # Look for label as a standalone word
                            if re.search(rf'\b{label}\b', line_lower):
                                prediction = label
                                self.log_fn(f"Attempt {attempt + 1}: Found label '{label}' in recent line: {line[:50]}")
                                break
                        if prediction:
                            break
                
                # Look for verdict/analysis section patterns
                if prediction is None:
                    text_lower = text_to_parse.lower()
                    # Look for "verdict:" or "analysis:" followed by a label
                    verdict_section = re.search(r'(?:verdict|analysis|conclusion|assessment)\s*[:\-]\s*([a-z]+)', text_lower, re.IGNORECASE)
                    if verdict_section:
                        potential_label = verdict_section.group(1).strip().lower()
                        if potential_label in VALID_LABELS:
                            prediction = potential_label
                            self.log_fn(f"Attempt {attempt + 1}: Found label '{prediction}' in verdict section")
                
                # Strategy: Look for the LAST occurrence of any valid label in the text
                # LLMs often put their final answer at the end after reasoning
                if prediction is None:
                    text_lower = text_to_parse.lower()
                    last_positions = {}
                    for label in VALID_LABELS:
                        # Find all occurrences and track the last one
                        for match in re.finditer(rf'\b{label}\b', text_lower):
                            last_positions[label] = match.start()
                    
                    if last_positions:
                        # Get the label that appears last in the text
                        last_label = max(last_positions, key=last_positions.get)
                        prediction = last_label
                        self.log_fn(f"Attempt {attempt + 1}: Found label '{prediction}' as last occurrence in text")
                
                # Strategy: Look for labels in the final sentence of the response
                if prediction is None:
                    # Split by sentence endings and get the last sentence
                    sentences = re.split(r'[.!?]+', text_to_parse)
                    if sentences:
                        last_sentence = sentences[-1].lower()
                        for label in VALID_LABELS:
                            if re.search(rf'\b{label}\b', last_sentence):
                                prediction = label
                                self.log_fn(f"Attempt {attempt + 1}: Found label '{prediction}' in final sentence")
                                break
                
                # If we got a valid prediction, break out of retry loop
                if prediction in ["correct", "incorrect", "partial", "almost"]:
                    break
                            
            except Exception as e:
                self.log_fn(f"Attempt {attempt + 1}: Error extracting prediction: {e}")
                prediction = None
        
        # Final validation - ensure prediction is valid
        CANONICAL_LABELS = ["correct", "incorrect", "partial", "almost"]
        if prediction not in CANONICAL_LABELS:
            self.log_fn(f"Invalid prediction '{prediction}', defaulting to 'incorrect'")
            # Default to "incorrect" as the safest fallback
            prediction = "incorrect"
        else:
            # Map "almost" to "partial" for final output
            if prediction == "almost":
                prediction = "partial"
            self.log_fn(f"Final prediction: {prediction}")

        return str(prediction), msg_history
