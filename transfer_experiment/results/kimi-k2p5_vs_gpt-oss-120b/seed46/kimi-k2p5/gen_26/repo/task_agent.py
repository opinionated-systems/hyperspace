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
LABEL_PRIORITY = {"incorrect": 4, "partial": 3, "almost": 2, "correct": 1}

# Mapping from labels to their canonical form
LABEL_CANONICAL = {
    "correct": "correct",
    "incorrect": "incorrect", 
    "partial": "partial",
    "almost": "partial",
}

# Common misspellings and variations mapping
LABEL_ALIASES = {
    "correct": ["correct", "right", "true", "valid", "accurate", "complete", "fully correct", "entirely correct", "perfect"],
    "incorrect": ["incorrect", "wrong", "false", "error", "invalid", "inaccurate", "flawed", "mistaken", "fundamentally wrong"],
    "partial": ["partial", "incomplete", "partially", "some", "part", "partial credit", "almost", "nearly", "mostly correct", "almost correct", "nearly complete"],
    "almost": ["almost", "nearly", "mostly correct", "almost correct", "nearly complete", "almost there"],
}


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks or raw JSON."""
    if not text or not isinstance(text, str):
        return None
        
    results = []
    search_from = 0
    text_lower = text.lower()
    
    # Strategy 1: Try to find <json>...</json> blocks (case insensitive)
    while True:
        start = text_lower.find("<json>", search_from)
        if start == -1:
            break
        end = text_lower.find("</json>", start)
        if end == -1:
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
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
    
    # Strategy 4: Look for explicit verdict statements
    verdict_patterns = [
        (r'(?:the\s+)?(?:final\s+)?verdict\s*(?:is|:)\s*["\']?(correct|incorrect|partial|almost)["\']?', 1),
        (r'(?:the\s+)?(?:final\s+)?(?:grade|classification|label|assessment)\s*(?:is|:)\s*["\']?(correct|incorrect|partial|almost)["\']?', 1),
        (r'(?:i\s+(?:would|will)\s+)?(?:classify|grade|label|mark)\s*(?:this\s+)?(?:as\s+)?["\']?(correct|incorrect|partial|almost)["\']?', 1),
        (r'(?:this\s+(?:is|should\s+be))\s*["\']?(correct|incorrect|partial|almost)["\']?', 1),
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
    
    # Strategy 6: Look for standalone labels at the end
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if lines:
        for line in reversed(lines[-3:]):
            line_clean = line.lower().strip('"\'.,!?:;')
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
    """Try to parse JSON string with multiple fallback strategies."""
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
    
    # Try 4: Extract response field with regex
    try:
        match = re.search(r'["\']?response["\']?\s*[:=]\s*["\']?([^"\'\}\n,]+)', json_str, re.IGNORECASE)
        if match:
            value = match.group(1).strip().lower()
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
    
    return None


def _normalize_label(value: str) -> str | None:
    """Normalize a label value to one of the valid labels."""
    if not value or not isinstance(value, str):
        return None
    
    clean = value.strip().lower()
    clean_no_punct = re.sub(r'[^a-z\s]', '', clean).strip()
    
    # Direct match
    if clean_no_punct in VALID_LABELS:
        return LABEL_CANONICAL.get(clean_no_punct, clean_no_punct)
    
    # Check aliases
    for label, aliases in LABEL_ALIASES.items():
        for alias in aliases:
            if clean_no_punct == alias.lower():
                return label
    
    # Fuzzy matching
    clean_single = re.sub(r'[^a-z]', '', clean)
    if 'incorrect' in clean_single or clean_single.startswith('incorr') or clean_single.startswith('wrong'):
        return 'incorrect'
    if 'correct' in clean_single and 'incorrect' not in clean_single and 'partial' not in clean_single:
        return 'correct'
    if 'almost' in clean_single or 'nearly' in clean_single:
        return 'almost'
    if 'partial' in clean_single or 'incomplete' in clean_single:
        return 'partial'
    
    return None


def _extract_label_from_text(text: str) -> str | None:
    """Extract the grading label from raw text using pattern matching."""
    if not text or not isinstance(text, str):
        return None
        
    text_lower = text.lower().strip()
    lines = [line.strip() for line in text_lower.split('\n') if line.strip()]
    
    # Priority 1: Look for explicit label declarations with colons
    label_alternatives = "correct|incorrect|partial|almost"
    declaration_patterns = [
        rf'(?:verdict|grade|label|classification)\s*[:=]\s*["\']?({label_alternatives})["\']?\b',
        rf'(?:the\s+answer\s+is)\s*["\']?({label_alternatives})["\']?\b',
        rf'(?:this\s+is)\s*["\']?({label_alternatives})["\']?\b',
    ]
    
    for pattern in declaration_patterns:
        match = re.search(pattern, text_lower)
        if match:
            for group in match.groups():
                if group in VALID_LABELS:
                    return LABEL_CANONICAL.get(group, group)
    
    # Priority 2: Look for labels in code blocks or backticks
    for label in VALID_LABELS:
        if re.search(rf'`{label}`', text_lower):
            return LABEL_CANONICAL.get(label, label)
        if re.search(rf'\*\*{label}\*\*|\*{label}\*', text_lower):
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
                best = max(candidates, key=lambda x: LABEL_PRIORITY.get(x, 0))
                return LABEL_CANONICAL.get(best, best)
    
    # Priority 5: Look at the last line
    if lines:
        last_line = lines[-1]
        for label in VALID_LABELS:
            if label in last_line:
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
        # Extract fields from inputs
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Simplified, focused instruction
        instruction = f"""You are an expert IMO mathematics grader. Evaluate the student answer and classify it into EXACTLY ONE category.

## GRADING CATEGORIES

**"correct"** - Complete, correct solution with all steps valid.
**"incorrect"** - Fundamental flaw: no valid approach, only pattern matching from examples, or complete misunderstanding.
**"partial"** - Valid approach and shows understanding, but incomplete or has execution errors.
**"almost"** - Nearly complete (90%+), minor gap only. (Maps to "partial" for final grading)

## KEY RULE: PARTIAL vs INCORRECT

This is the most critical distinction:

- **PARTIAL**: Student shows they KNOW HOW to solve the problem (valid technique, good setup, real insight) even if unfinished or with errors.
- **INCORRECT**: Student shows NO valid understanding (just checking examples, wrong method entirely, fundamental misconception).

**Golden Rule**: If the student demonstrates genuine mathematical insight or uses a valid technique → "partial". If no valid approach at all → "incorrect".

## FEW-SHOT EXAMPLES

Example 1 - CORRECT (Complete proof):
Problem: Prove infinitely many primes.
Student: Suppose finitely many primes p₁,...,pₙ. Let N = p₁×...×pₙ + 1. N mod pᵢ = 1 for all i, so N has a prime factor not in list. Contradiction.
Verdict: correct

Example 2 - INCORRECT (Pattern matching, no proof):
Problem: Prove sum of first n odd numbers is n².
Student: n=1: 1=1². n=2: 1+3=4=2². n=3: 1+3+5=9=3². Pattern holds.
Verdict: incorrect

Example 3 - PARTIAL (Valid approach, incomplete):
Problem: Prove n³-n divisible by 6 for all n.
Student: n³-n = n(n-1)(n+1). Product of 3 consecutive integers. One must be even, so divisible by 2.
Verdict: partial (valid factorization and insight, but didn't prove divisibility by 3)

Example 4 - INCORRECT (Fundamental error):
Problem: Find all x where |x-1| + |x+1| = 2.
Student: |x-1| + |x+1| = |2x| = 2. So |x| = 1, meaning x = ±1.
Verdict: incorrect (|a|+|b|≠|a+b|; fundamental misunderstanding)

Example 5 - PARTIAL (Right method, calculation error):
Problem: Find diagonals in n-gon.
Student: From each vertex, n-3 diagonals. Total = n(n-3). For n=5: 10. But actual is 5. Forgot to divide by 2.
Verdict: partial (correct method, execution error)

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

## YOUR ANALYSIS

Step 1 - Understanding: Does student understand the problem?
Step 2 - Approach: Is their approach mathematically valid?
Step 3 - Insight: Do they show genuine mathematical insight?
Step 4 - Execution: Are calculations/proof steps correct?
Step 5 - Completeness: Is the solution complete?

Based on this analysis, provide your verdict.

## OUTPUT FORMAT (CRITICAL)

Output ONLY this JSON block. No other text.

<json>{{"response": "correct"}}</json>
OR
<json>{{"response": "incorrect"}}</json>
OR
<json>{{"response": "partial"}}</json>
OR
<json>{{"response": "almost"}}</json>

The JSON block MUST be the very last thing in your response."""

        # Try up to 3 times to get a valid prediction
        max_retries = 3
        prediction = None
        msg_history = []
        
        for attempt in range(max_retries):
            try:
                # On retry, add a reminder about output format
                msg_to_send = instruction
                if attempt > 0:
                    msg_to_send += "\n\nIMPORTANT: Output ONLY the JSON block in <json> tags. No other text."
                
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
                    self.log_fn(f"Attempt {attempt + 1}: Extracted {len(extracted)} JSON objects")
                    for item in extracted:
                        if isinstance(item, dict) and "response" in item:
                            raw_pred = str(item["response"]).strip()
                            self.log_fn(f"Attempt {attempt + 1}: Found response field: {raw_pred}")
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
                
                # If still no prediction, look for high confidence patterns
                if prediction is None:
                    text_lower = text_to_parse.lower()
                    
                    # High confidence patterns
                    high_confidence_patterns = [
                        (r'<json>\s*\{\s*"response"\s*:\s*"correct"\s*\}\s*</json>', 'correct'),
                        (r'<json>\s*\{\s*"response"\s*:\s*"incorrect"\s*\}\s*</json>', 'incorrect'),
                        (r'<json>\s*\{\s*"response"\s*:\s*"partial"\s*\}\s*</json>', 'partial'),
                        (r'<json>\s*\{\s*"response"\s*:\s*"almost"\s*\}\s*</json>', 'almost'),
                    ]
                    
                    for pattern, label in high_confidence_patterns:
                        if re.search(pattern, text_lower):
                            prediction = label
                            self.log_fn(f"Attempt {attempt + 1}: High confidence pattern matched: '{label}'")
                            break
                    
                    # Medium confidence patterns
                    if prediction is None:
                        incorrect_indicators = ['fundamentally wrong', 'wrong approach', 'no understanding', 'fundamental error', 'completely wrong']
                        partial_indicators = ['incomplete', 'partial credit', 'partially correct', 'missing steps', 'not finished', 'incomplete solution']
                        almost_indicators = ['almost', 'nearly', 'almost correct', 'nearly complete']
                        correct_indicators = ['fully correct', 'completely correct', 'perfect solution']
                        
                        for indicator in incorrect_indicators:
                            if indicator in text_lower:
                                prediction = 'incorrect'
                                self.log_fn(f"Attempt {attempt + 1}: Detected 'incorrect' via indicator")
                                break
                        
                        if prediction is None:
                            for indicator in almost_indicators:
                                if indicator in text_lower:
                                    prediction = 'almost'
                                    self.log_fn(f"Attempt {attempt + 1}: Detected 'almost' via indicator")
                                    break
                        
                        if prediction is None:
                            for indicator in partial_indicators:
                                if indicator in text_lower:
                                    prediction = 'partial'
                                    self.log_fn(f"Attempt {attempt + 1}: Detected 'partial' via indicator")
                                    break
                        
                        if prediction is None:
                            for indicator in correct_indicators:
                                if indicator in text_lower:
                                    prediction = 'correct'
                                    self.log_fn(f"Attempt {attempt + 1}: Detected 'correct' via indicator")
                                    break
                
                # Look for standalone labels at the very end
                if prediction is None:
                    lines = [line.strip() for line in text_to_parse.split('\n') if line.strip()]
                    if lines:
                        last_line = lines[-1].lower()
                        for label in VALID_LABELS:
                            if re.search(rf'^["\']?{label}["\']?[.!?]*\s*$', last_line, re.IGNORECASE):
                                prediction = label
                                self.log_fn(f"Attempt {attempt + 1}: Found standalone label '{label}' at end")
                                break
                
                # Look for the LAST occurrence of any valid label
                if prediction is None:
                    text_lower = text_to_parse.lower()
                    last_positions = {}
                    for label in VALID_LABELS:
                        for match in re.finditer(rf'\b{label}\b', text_lower):
                            last_positions[label] = match.start()
                    
                    if last_positions:
                        last_label = max(last_positions, key=last_positions.get)
                        prediction = last_label
                        self.log_fn(f"Attempt {attempt + 1}: Found label '{prediction}' as last occurrence")
                
                # If we got a valid prediction, break out of retry loop
                if prediction in ["correct", "incorrect", "partial", "almost"]:
                    break
                        
            except Exception as e:
                self.log_fn(f"Attempt {attempt + 1}: Error extracting prediction: {e}")
                prediction = None
        
        # Final validation
        if prediction not in VALID_LABELS:
            self.log_fn(f"Invalid prediction '{prediction}', defaulting to 'incorrect'")
            prediction = "incorrect"
        else:
            if prediction == "almost":
                prediction = "partial"
            self.log_fn(f"Final prediction: {prediction}")

        return str(prediction), msg_history
