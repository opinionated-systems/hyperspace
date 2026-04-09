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
    "correct": ["correct", "right", "true", "valid", "accurate", "complete", "fully correct", "entirely correct", "perfect", "solved", "full marks"],
    "incorrect": ["incorrect", "wrong", "false", "error", "invalid", "inaccurate", "flawed", "mistaken", "fundamentally wrong", "no credit", "zero"],
    "partial": ["partial", "incomplete", "partially", "some", "part", "partial credit", "almost", "nearly", "mostly correct", "almost correct", "nearly complete", "half"],
    "almost": ["almost", "nearly", "mostly correct", "almost correct", "nearly complete", "almost there", "minor gap"],
}


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks or raw JSON."""
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
    search_from = 0
    while True:
        start = text_lower.find("<json>", search_from)
        if start == -1:
            break
        end = text_lower.find("</json>", start)
        if end == -1:
            break
        # Use original text (not lower) for content extraction
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
        rf'(?:verdict|grade|label|classification)\s*[:=]\s*["\']?({label_alternatives})["\']?\b',
        rf'(?:the\s+answer\s+is)\s*["\']?({label_alternatives})["\']?\b',
        rf'(?:this\s+is)\s*["\']?({label_alternatives})["\']?\b',
        rf'["\']?response["\']?\s*[:=]\s*["\']?({label_alternatives})["\']?\b',
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
**"partial"** - Valid approach showing genuine understanding, but incomplete or has execution errors. Partial credit.
**"almost"** - Nearly complete solution (90%+), only minor gap. (Maps to "partial" for final grading)

## CRITICAL DISTINCTION: PARTIAL vs INCORRECT

This is THE most important decision:

- **PARTIAL**: Student demonstrates they UNDERSTAND how to solve the problem. They use a valid technique, show real insight, or make meaningful progress. Even if unfinished or with errors, they show mathematical competence.
- **INCORRECT**: Student shows NO valid understanding. They just check examples, use wrong methods, or have fundamental misconceptions. No valid approach demonstrated.

**Golden Rule**: If there's ANY valid mathematical insight or technique → "partial". Only use "incorrect" when there's truly no valid approach.

## DETAILED FEW-SHOT EXAMPLES

### Example 1 - CORRECT (Complete rigorous proof)
Problem: Prove there are infinitely many primes.
Student: Suppose finitely many primes p₁,...,pₙ. Let N = p₁×...×pₙ + 1. For each pᵢ, N mod pᵢ = 1, so pᵢ doesn't divide N. Thus N has a prime factor not in our list. Contradiction.
Analysis: Complete proof with valid construction and logical conclusion.
Verdict: <json>{{"response": "correct"}}</json>

### Example 2 - INCORRECT (Pattern matching without proof)
Problem: Prove sum of first n odd numbers equals n².
Student: n=1: 1=1². n=2: 1+3=4=2². n=3: 1+3+5=9=3². The pattern holds for all n.
Analysis: Only checked examples, no general proof. No valid proof technique demonstrated.
Verdict: <json>{{"response": "incorrect"}}</json>

### Example 3 - PARTIAL (Valid insight, incomplete execution)
Problem: Prove n³-n is divisible by 6 for all integers n.
Student: n³-n = n(n-1)(n+1). This is the product of 3 consecutive integers. One of them must be even, so divisible by 2.
Analysis: Excellent factorization and valid insight about divisibility by 2. But student didn't prove divisibility by 3 (which is also true for 3 consecutive integers). Valid approach, incomplete proof.
Verdict: <json>{{"response": "partial"}}</json>

### Example 4 - INCORRECT (Fundamental mathematical error)
Problem: Find all real x where |x-1| + |x+1| = 2.
Student: |x-1| + |x+1| = |(x-1)+(x+1)| = |2x| = 2. So |x| = 1, meaning x = ±1.
Analysis: The step |a|+|b| = |a+b| is FALSE in general. This is a fundamental misunderstanding of absolute value properties. No valid approach.
Verdict: <json>{{"response": "incorrect"}}</json>

### Example 5 - PARTIAL (Right method, execution error)
Problem: Find the number of diagonals in a convex n-gon.
Student: From each vertex, we can draw n-3 diagonals (to all non-adjacent vertices). With n vertices, total = n(n-3). For n=5: 5×2=10. But the actual answer is 5. I forgot to divide by 2 since each diagonal is counted twice.
Analysis: Correct method and formula structure. Valid understanding of the problem. Just a counting error (double counting). Shows genuine competence.
Verdict: <json>{{"response": "partial"}}</json>

### Example 6 - ALMOST (Nearly complete, tiny gap)
Problem: Prove the AM-GM inequality for two positive numbers.
Student: For a,b > 0, we want (a+b)/2 ≥ √ab. This is equivalent to a+b ≥ 2√ab, then (a+b)² ≥ 4ab, then a²+2ab+b² ≥ 4ab, so a²-2ab+b² ≥ 0, which is (a-b)² ≥ 0. This is always true.
Analysis: Perfect proof structure. Valid algebraic manipulation. Only minor issue: should explicitly state that squaring preserves inequality since a+b > 0, and that equality holds when a=b. Nearly complete.
Verdict: <json>{{"response": "almost"}}</json>

## CHAIN-OF-THOUGHT ANALYSIS PROCESS

Analyze the student's solution step by step:

1. **Understanding**: Does the student correctly understand what the problem is asking?
2. **Approach Validity**: Is their overall approach mathematically sound?
3. **Key Insight**: Did they identify the crucial insight or technique needed?
4. **Execution**: Are the calculations and logical steps correct?
5. **Completeness**: Is the solution fully finished, or are there gaps?

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

First, provide your chain-of-thought analysis following the 5 steps above.

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
5. The response field must be one of: correct, incorrect, partial, almost"""

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
