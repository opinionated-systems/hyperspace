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
    multiline_pattern = r'\{\s*["\']?response["\']?\s*:\s*["\']?(correct|incorrect|partial|almost)["\']?\s*\}'
    for match in re.finditer(multiline_pattern, text, re.DOTALL | re.IGNORECASE):
        try:
            value = match.group(1).strip().lower()
            if value in VALID_LABELS:
                results.append({"response": value})
        except Exception:
            continue
    
    # Try to find JSON with escaped quotes or special characters
    escaped_pattern = r'\{\s*"response"\s*:\s*"([^"]*(?:\\"[^"]*)*)"\s*\}'
    for match in re.finditer(escaped_pattern, text):
        try:
            results.append(json.loads(match.group(0)))
        except json.JSONDecodeError:
            continue
    
    # Try to find JSON with trailing commas (common LLM error)
    trailing_comma_pattern = r'\{\s*"response"\s*:\s*"([^"]+)"\s*,?\s*\}'
    for match in re.finditer(trailing_comma_pattern, text):
        try:
            # Remove trailing comma if present
            json_str = match.group(0).replace(',}', '}')
            results.append(json.loads(json_str))
        except json.JSONDecodeError:
            continue
    
    # Try to find JSON objects with unquoted keys (response without quotes)
    unquoted_key_pattern = r'\{\s*response\s*:\s*["\']?([^"\'\}\n,]+)["\']?\s*\}'
    for match in re.finditer(unquoted_key_pattern, text, re.IGNORECASE):
        try:
            value = match.group(1).strip().lower()
            if value in VALID_LABELS:
                results.append({"response": value})
        except Exception:
            continue
    
    # Look for the response field anywhere in the text
    # This handles cases where the LLM outputs the JSON without proper formatting
    loose_pattern = r'["\']?response["\']?\s*[:=]\s*["\']?(correct|incorrect|partial|almost)["\']?'
    for match in re.finditer(loose_pattern, text, re.IGNORECASE):
        try:
            value = match.group(1).strip().lower()
            if value in VALID_LABELS:
                results.append({"response": value})
        except Exception:
            continue
    
    # Look for explicit verdict statements with more patterns
    # 4 categories: correct, incorrect, partial, almost
    verdict_patterns = [
        r'(?:the\s+)?(?:final\s+)?verdict\s*(?:is|:)\s*["\']?(correct|incorrect|partial|almost)["\']?',
        r'(?:the\s+)?(?:final\s+)?(?:grade|classification|label)\s*(?:is|:)\s*["\']?(correct|incorrect|partial|almost)["\']?',
        r'(?:i\s+(?:would|will)\s+)?(?:classify|grade|label|mark)\s*(?:this\s+)?(?:as\s+)?["\']?(correct|incorrect|partial|almost)["\']?',
        r'(?:this\s+(?:is|should\s+be))\s*["\']?(correct|incorrect|partial|almost)["\']?',
        r'(?:the\s+answer\s+(?:is|should\s+be))\s*["\']?(correct|incorrect|partial|almost)["\']?',
        r'(?:verdict|decision|assessment)\s*:\s*["\']?(correct|incorrect|partial|almost)["\']?',
        r'(?:therefore|thus|hence)[,:]?\s*(?:the\s+)?(?:answer\s+)?(?:is\s+)?["\']?(correct|incorrect|partial|almost)["\']?',
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
    
    # Look for standalone labels at the end of the response (common pattern)
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
    
    # NEW: Look for "Verdict:" or "Analysis:" followed by label on same line
    for line in lines:
        line_lower = line.lower()
        for label in VALID_LABELS:
            # Match patterns like "Verdict: correct" or "Analysis: incorrect"
            if re.search(rf'(?:verdict|analysis|conclusion|assessment|grade|label)\s*[:\-]\s*["\']?{label}["\']?\b', line_lower, re.IGNORECASE):
                results.append({"response": label})
                break
    
    # NEW: Look for JSON-like patterns with extra whitespace or newlines
    # Handle cases like: {\n  "response": "correct"\n}
    json_with_whitespace = r'\{\s*["\']?response["\']?\s*:\s*["\']?(correct|incorrect|partial|almost)["\']?\s*\}'
    for match in re.finditer(json_with_whitespace, text, re.DOTALL | re.IGNORECASE):
        try:
            value = match.group(1).strip().lower()
            if value in VALID_LABELS:
                results.append({"response": value})
        except Exception:
            continue
    
    # NEW: Look for patterns like "response: correct" (without braces)
    simple_response_pattern = r'\bresponse\s*:\s*["\']?(correct|incorrect|partial|almost)["\']?\b'
    for match in re.finditer(simple_response_pattern, text, re.IGNORECASE):
        try:
            value = match.group(1).strip().lower()
            if value in VALID_LABELS:
                results.append({"response": value})
        except Exception:
            continue
    
    return results if results else None


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
    
    # Priority 1: Look for quoted JSON-style values with "response" key
    json_pattern = r'"response"\s*[:=]\s*"([^"]+)"'
    for match in re.finditer(json_pattern, text_lower):
        clean_match = re.sub(r'[^a-z]', '', match.group(1).lower())
        if clean_match in VALID_LABELS:
            return LABEL_CANONICAL.get(clean_match, clean_match)
    
    # Priority 2: Look for single-quoted JSON-style values
    json_pattern_single = r"'response'\s*[:=]\s*'([^']+)'"
    for match in re.finditer(json_pattern_single, text_lower):
        clean_match = re.sub(r'[^a-z]', '', match.group(1).lower())
        if clean_match in VALID_LABELS:
            return LABEL_CANONICAL.get(clean_match, clean_match)
    
    # Priority 3: Look for explicit label declarations with colons or equals
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
        
        instruction = f"""You are an expert IMO (International Mathematical Olympiad) mathematics grader. Evaluate the student answer and classify it into EXACTLY ONE category: "correct", "incorrect", "partial", or "almost".

## GRADING CATEGORIES

### "correct" - COMPLETE AND CORRECT (10/10 points)
- ALL key mathematical steps present and logically sound
- Final answer matches the official solution exactly
- No gaps, no errors, no missing justifications

### "incorrect" - FUNDAMENTALLY FLAWED (0-2 points)
- Critical logical errors or fundamental misconceptions
- Wrong approach from the start
- No valid mathematical reasoning demonstrated
- WRONG final answer + no valid approach

### "partial" - MEANINGFUL PROGRESS BUT INCOMPLETE (3-7 points)
- Valid approach started with correct key insights
- Shows genuine mathematical understanding
- Significant gaps remain, missing steps, or incomplete reasoning
- WRONG final answer BUT valid approach = "partial" (not "incorrect")

### "almost" - NEARLY COMPLETE (8-9 points)
- Solution is very close to complete, just minor gaps
- Correct approach, nearly all steps present
- Small missing detail or minor error in final step

## CRITICAL RULES

1. **CORRECT final answer + complete reasoning** = "correct"
2. **CORRECT final answer + incomplete reasoning** = "partial" or "almost"
3. **WRONG final answer + valid approach + understanding** = "partial"
4. **WRONG final answer + incomplete work** = "incorrect"
5. **Nearly complete, minor gap** = "almost"
6. **Good start but significant missing work** = "partial"

## FEW-SHOT EXAMPLES

### Example 1: Correct
Problem: Prove infinitely many primes.
Official: Assume finitely many primes p₁,...,pₙ. Let N = p₁...pₙ + 1. N not divisible by any pᵢ, contradiction.
Student: Suppose finitely many primes p₁,...,pₙ. Let N = p₁×...×pₙ + 1. N mod pᵢ = 1 for all i, so N has a prime factor not in list. Contradiction.
Verdict: correct

### Example 2: Incorrect (Fundamental Error)
Problem: Find n where n²+n+41 is always prime.
Official: At n=40: 40²+40+41=1681=41², not prime.
Student: For n=1: 43 (prime). n=2: 47 (prime). Pattern continues since 41 is prime and n²+n is even, so result is odd and prime.
Verdict: incorrect

### Example 3: Partial (Valid Approach, Incomplete)
Problem: Prove a²+b²+c² ≥ 4√3 × Area for any triangle.
Official: Uses Heron's formula and AM-GM.
Student: For equilateral triangle side s: Area = (√3/4)s². LHS = 3s², RHS = 4√3 × (√3/4)s² = 3s². Equality holds. For other triangles, LHS grows relative to area.
Verdict: partial

### Example 4: Incorrect (Wrong Approach)
Problem: Prove √2 is irrational.
Official: Assume √2 = p/q lowest terms. Then 2q² = p², so p even, p=2k, then q even. Contradiction.
Student: √2 ≈ 1.41421356... The decimal never repeats, so √2 cannot be written as p/q. Therefore irrational.
Verdict: incorrect

### Example 5: Partial (Right Approach, Missing Completion)
Problem: Circular arrangements of 6 people (rotations same).
Official: Fix one person, arrange 5 others: 5! = 120.
Student: Fix one person's position for circular symmetry. Arrange remaining 5 people: 5! ways.
Verdict: partial

### Example 6: Correct (Different Method)
Problem: Solve x² - 5x + 6 = 0.
Official: (x-2)(x-3) = 0, so x = 2 or 3.
Student: Using quadratic formula: x = (5 ± √(25-24))/2 = (5 ± 1)/2. So x = 3 or 2.
Verdict: correct

### Example 7: Incorrect (Misunderstanding)
Problem: Choose 3 people from 10.
Official: C(10,3) = 120.
Student: Pick first in 10 ways, second in 9, third in 8: 10×9×8 = 720.
Verdict: incorrect

### Example 8: Partial (Incomplete Induction)
Problem: Prove 1+2+...+n = n(n+1)/2 by induction.
Official: Base n=1: 1=1(2)/2 ✓. Assume for n=k. For n=k+1: 1+...+k+(k+1) = k(k+1)/2 + (k+1) = (k+1)(k+2)/2. ✓
Student: Base: n=1, LHS=1, RHS=1. Check. Inductive step: Assume 1+...+k = k(k+1)/2. Then for k+1: add (k+1) to both sides.
Verdict: partial

### Example 9: Almost (Nearly Complete)
Problem: Find maximum of sin(x) + cos(x).
Official: sin(x)+cos(x) = √2 sin(x+π/4). Max is √2.
Student: sin(x)+cos(x) = √2[sin(x)/√2 + cos(x)/√2] = √2 sin(x+π/4). Max of sin is 1, so max is √2.
Verdict: almost

### Example 10: Partial (Wrong Answer, Valid Approach)
Problem: Find sum of n where n²-19n+99 is perfect square.
Official: Solutions n=1,9,10,18. Sum=38.
Student: Let n²-19n+99=k². Using quadratic formula: n=(19±√(4k²-35))/2. For k=3: n=10,9. k=4,5: not squares. Answer: 19.
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

## ANALYSIS INSTRUCTIONS

Think step-by-step:
1. Does the student understand the problem?
2. Is their approach mathematically valid?
3. Are the key steps correct?
4. Is the final answer correct?
5. Is the solution complete?

**Key Decision Rule**: If the student shows understanding (valid approach) but has wrong answer or incomplete work → "partial" not "incorrect".

## OUTPUT FORMAT (CRITICAL)

Output ONLY a JSON object in <json> tags:

<json>
{{
    "response": "correct" | "incorrect" | "partial" | "almost"
}}
</json>

Requirements:
- ONLY the JSON block, no other text
- Use double quotes
- "almost" maps to "partial" for final grading
- JSON must be the LAST thing in your response
"""

        # Try up to 3 times to get a valid prediction
        max_retries = 3
        prediction = None
        msg_history = []
        
        for attempt in range(max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
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
