"""
Task agent: solves a given task with a single LLM call.
Reimplemented from facebookresearch/HyperAgents task_agent.py.
"""

from __future__ import annotations

import json
import logging
import re

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks."""
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
    """Normalize a prediction string to one of the allowed categories."""
    if not prediction:
        return None
    
    pred_lower = prediction.lower().strip()
    
    # Remove common prefixes/suffixes
    pred_clean = re.sub(r'^(the\s+|a\s+|an\s+|is\s+|it\s+|this\s+|that\s+)', '', pred_lower)
    pred_clean = re.sub(r'\s+(answer|classification|category|result|grade)$', '', pred_clean)
    
    # Check for exact matches first
    allowed_categories = ["correct", "incorrect", "partial", "almost"]
    if pred_clean in allowed_categories:
        return pred_clean.capitalize()
    
    # Check for compound patterns first (before single word patterns)
    # These are phrases that indicate "Almost" category
    almost_compound_patterns = [
        r'\balmost\s+correct\b', r'\bnearly\s+correct\b', r'\balmost\s+perfect\b',
        r'\bclose\s+to\s+correct\b', r'\bessentially\s+correct\b',
        r'\bmostly\s+correct\b', r'\bcorrect\s+except\s+for\b',
        r'\bwould\s+be\s+correct\b',
    ]
    for pattern in almost_compound_patterns:
        if re.search(pattern, pred_lower):
            return "Almost"
    
    # Now check single word patterns in priority order
    # Check Almost first (most specific to avoid being caught by other patterns)
    if re.search(r'\balmost\b', pred_lower):
        return "Almost"
    if re.search(r'\bnearly\b', pred_lower):
        return "Almost"
    if re.search(r'\bminor\s+(?:error|mistake|issue)', pred_lower):
        return "Almost"
    if re.search(r'\bsmall\s+(?:error|mistake|issue)', pred_lower):
        return "Almost"
    if re.search(r'\btiny\s+(?:error|mistake|issue)', pred_lower):
        return "Almost"
    
    # Check Partial patterns
    if re.search(r'\bpartial\b', pred_lower):
        return "Partial"
    if re.search(r'\bpartly\b', pred_lower):
        return "Partial"
    if re.search(r'\bincomplete\b', pred_lower):
        return "Partial"
    if re.search(r'\bon\s+the\s+right\s+track\b', pred_lower):
        return "Partial"
    if re.search(r'\bgood\s+start\b', pred_lower):
        return "Partial"
    
    # Check Incorrect patterns
    if re.search(r'\bincorrect\b', pred_lower):
        return "Incorrect"
    if re.search(r'\bwrong\b', pred_lower):
        return "Incorrect"
    if re.search(r'\bfalse\b', pred_lower):
        return "Incorrect"
    if re.search(r'\binvalid\b', pred_lower):
        return "Incorrect"
    if re.search(r'\bnot\s+correct\b', pred_lower):
        return "Incorrect"
    
    # Check Correct patterns last (least specific)
    if re.search(r'\bcorrect\b', pred_lower):
        return "Correct"
    if re.search(r'\bright\b', pred_lower):
        return "Correct"
    if re.search(r'\bvalid\b', pred_lower):
        return "Correct"
    if re.search(r'\btrue\b', pred_lower):
        return "Correct"
    if re.search(r'\bperfect\b', pred_lower):
        return "Correct"
    
    return None


def _extract_response_flexible(text: str) -> str | None:
    """Extract the classification from model response using multiple strategies."""
    if not text:
        return None
    
    text_lower = text.lower()
    text_upper = text.upper()
    
    # Strategy 1: Look for explicit classification statements with high confidence
    # Check "Almost" first as it's the most commonly missed
    explicit_patterns = [
        # Direct classification statements
        (r'\bclassification\s*[:=]\s*almost\b', "Almost"),
        (r'\bgrade\s*[:=]\s*almost\b', "Almost"),
        (r'\bcategory\s*[:=]\s*almost\b', "Almost"),
        (r'\bclassify\s+(?:this|it)\s+as\s+almost\b', "Almost"),
        (r'\bis\s+almost\b', "Almost"),
        (r'\bthis\s+is\s+almost\b', "Almost"),
        (r'\bthe\s+answer\s+is\s+almost\b', "Almost"),
        (r'\bresponse["\']?\s*[:=]\s*["\']?\s*almost\b', "Almost"),
        (r'\bshould\s+be\s+almost\b', "Almost"),
        (r'\bwould\s+be\s+almost\b', "Almost"),
        
        (r'\bclassification\s*[:=]\s*partial\b', "Partial"),
        (r'\bgrade\s*[:=]\s*partial\b', "Partial"),
        (r'\bcategory\s*[:=]\s*partial\b', "Partial"),
        (r'\bclassify\s+(?:this|it)\s+as\s+partial\b', "Partial"),
        
        (r'\bclassification\s*[:=]\s*incorrect\b', "Incorrect"),
        (r'\bgrade\s*[:=]\s*incorrect\b', "Incorrect"),
        (r'\bcategory\s*[:=]\s*incorrect\b', "Incorrect"),
        
        (r'\bclassification\s*[:=]\s*correct\b', "Correct"),
        (r'\bgrade\s*[:=]\s*correct\b', "Correct"),
        (r'\bcategory\s*[:=]\s*correct\b', "Correct"),
    ]
    
    for pattern, category in explicit_patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return category
    
    # Strategy 2: Try JSON extraction from <json> tags
    json_results = _extract_jsons(text)
    if json_results:
        for result in json_results:
            if isinstance(result, dict):
                for key in ["response", "classification", "answer", "result", "grade", "evaluation", "verdict", "category"]:
                    if key in result:
                        val = result[key]
                        if isinstance(val, str):
                            normalized = _normalize_prediction(val.strip())
                            if normalized:
                                return normalized
                        elif isinstance(val, bool):
                            return "Correct" if val else "Incorrect"
    
    # Strategy 3: Try to find JSON in markdown code blocks
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
    
    # Strategy 4: Look for standalone category mentions at end of text
    # Check Almost first
    almost_standalone = [
        r'(?:^|\n)\s*almost\s*[.!?]?\s*(?:$|\n)',
        r'\*\*almost\*\*',
        r'\balmost\b[.!?]?\s*$',
    ]
    for pattern in almost_standalone:
        if re.search(pattern, text_lower, re.IGNORECASE | re.MULTILINE):
            return "Almost"
    
    # Then other categories
    standalone_patterns = [
        (r'(?:^|\n)\s*partial\s*[.!?]?\s*(?:$|\n)', "Partial"),
        (r'(?:^|\n)\s*incorrect\s*[.!?]?\s*(?:$|\n)', "Incorrect"),
        (r'(?:^|\n)\s*correct\s*[.!?]?\s*(?:$|\n)', "Correct"),
        (r'\*\*partial\*\*', "Partial"),
        (r'\*\*incorrect\*\*', "Incorrect"),
        (r'\*\*correct\*\*', "Correct"),
    ]
    for pattern, category in standalone_patterns:
        if re.search(pattern, text_lower, re.IGNORECASE | re.MULTILINE):
            return category
    
    # Strategy 5: Look for direct category mentions with word boundaries
    # Check Almost first
    if re.search(r'\bALMOST\b', text_upper):
        return "Almost"
    
    # Then other categories in order
    for category in ["Partial", "Incorrect", "Correct"]:
        if re.search(rf'\b{category.upper()}\b', text_upper):
            return category
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem."""
        # Extract fields from inputs
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        instruction = f"""You are an expert mathematics grader for International Mathematical Olympiad (IMO) problems. Evaluate the student's answer and classify it into exactly one of four categories.

## The Four Categories (CHOOSE EXACTLY ONE):

1. **Correct**: The solution is 100% perfect. No errors, no gaps, no issues whatsoever. The proof is complete and rigorous.

2. **Almost**: The solution is 90-99% complete with only 1-2 TINY issues (arithmetic error, typo, minor notation issue, one small edge case). The fix would take less than 30 seconds to explain. The core logic is sound and complete.

3. **Partial**: The solution shows good understanding and correct approach, but has MAJOR gaps or missing components. The student is on the right track but significant work remains (5+ minutes to complete). Multiple issues or missing critical steps.

4. **Incorrect**: The approach is fundamentally wrong, or the student fails to demonstrate understanding of the correct method. Wrong method, circular reasoning, or verification by example.

## CRITICAL DECISION RULES:

**Rule 1 - Correct vs Almost**: 
- If there's ANY flaw (even a tiny typo) → **Almost**
- Only if 100% perfect → **Correct**

**Rule 2 - Almost vs Partial (MOST IMPORTANT)**:
- Count the issues: 1-2 tiny issues → **Almost**
- 3+ issues OR any major gap → **Partial**
- Key test: "Would this be perfect with a 30-second fix?" Yes→Almost, No→Partial

**Rule 3 - Partial vs Incorrect**:
- Does student understand the correct approach? Yes→Partial, No→Incorrect
- Key test: "Are they on the right track?" Yes→Partial, No→Incorrect

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

## Examples:

**Correct**: Complete, rigorous proof with all steps correct. No errors.

**Almost**: 
- "2+2=5" (one arithmetic error)
- "Correct formula but calculated 100×101/2=5051 instead of 5050"
- "Complete proof but missed n=0 case"
- "Correct approach, one sign error"

**Partial**:
- "Started induction correctly but didn't complete the inductive step"
- "Identified key property but didn't complete the proof"
- "Correct method but missing multiple key steps"

**Incorrect**:
- "Proved by checking examples n=1,2,3"
- "Used completely wrong method"
- "Circular reasoning"

## Your Task:
1. Compare the student answer to the official solution
2. Identify any errors, gaps, or issues
3. Count the issues (1-2 tiny = Almost, 3+ or major = Partial)
4. Determine if the approach is correct (Partial) or wrong (Incorrect)
5. Output EXACTLY ONE category: Correct, Almost, Partial, or Incorrect

## Response Format (REQUIRED):
You MUST respond with a JSON object in this exact format:

<json>
{{
    "response": "Correct" | "Almost" | "Partial" | "Incorrect"
}}
</json>

**CRITICAL**: Before responding, verify:
- Did I find ANY error? If yes, cannot be "Correct" → use "Almost"
- Are there 1-2 tiny issues? → "Almost" (NOT Partial)
- Is the approach right but incomplete? → "Partial" (NOT Incorrect)
- Is the approach wrong? → "Incorrect"

Output your classification now:"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction using flexible extraction
        prediction = "None"
        response_text = ""
        try:
            response_text = msg_history[-1]["text"] if msg_history else ""
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
        
        # Post-processing: Check if the response text contains strong indicators for "Almost"
        if prediction in ["Correct", "Partial", "None"] and response_text:
            response_lower = response_text.lower()
            strong_almost_indicators = [
                r'only\s+(?:a\s+)?(?:one|single|1)\s+(?:minor|small|tiny)\s+(?:error|mistake|issue|typo)',
                r'just\s+(?:a\s+)?(?:one|single|1)\s+(?:minor|small|tiny)\s+(?:error|mistake|issue|typo)',
                r'(?:one|single|1)\s+(?:minor|small|tiny)\s+(?:error|mistake|issue|typo)',
                r'(?:one|single|1)\s+(?:arithmetic|calculation|computation|sign)\s+(?:error|mistake)',
                r'sign\s+(?:error|mistake)',
                r'typo\b',
                r'essentially\s+correct',
                r'nearly\s+correct',
                r'almost\s+correct',
                r'95%\s+correct',
                r'99%\s+correct',
                r'correct\s+except\s+for\s+(?:a\s+)?(?:minor|small|tiny)',
                r'would\s+be\s+correct\s+(?:if|with|after)',
                r'would\s+be\s+perfect',
                r'just\s+needs?\s+(?:a\s+)?(?:minor|small|tiny)\s+fix',
                r'only\s+needs?\s+(?:a\s+)?(?:minor|small|tiny)\s+fix',
                r'off\s+by\s+(?:one|1)',
                r'forgot\s+(?:to|the)\s+(?:check|include|mention)',
                r'missing\s+(?:just|only)\s+(?:a\s+)?(?:minor|small|tiny)',
                r'missing\s+(?:one|a\s+single)\s+(?:minor|small|tiny)',
                r'(?:arithmetic|calculation|computation)\s+(?:error|mistake)\s+(?:only|just)',
                r'(?:small|minor|tiny)\s+(?:arithmetic|calculation|computation)\s+(?:error|mistake)',
                r'(?:just|only)\s+(?:one|a\s+single)\s+(?:error|mistake|issue)',
                r'(?:forgot|missed)\s+(?:to\s+)?(?:check|include|verify|prove)',
                r'(?:would|could)\s+be\s+(?:correct|right|valid)\s+with',
                r'(?:minor|small|tiny)\s+(?:correction|adjustment|change)',
                r'(?:simple|trivial)\s+(?:error|mistake|issue|fix)',
                r'(?:easily|quickly)\s+(?:fixed|corrected|remedied)',
                r'(?:one|a)\s+(?:small|minor|tiny)\s+(?:step|thing|detail|part)',
                r'(?:nearly|almost|practically)\s+(?:complete|finished|done|perfect)',
                r'(?:just|only)\s+(?:a\s+)?(?:bit|little|slightly)\s+(?:off|wrong|incorrect)',
                r'(?:small|minor)\s+(?:oversight|omission|lapse)',
                r'(?:one|single)\s+(?:exception|edge\s+case|special\s+case)',
                r'(?:forgot|missed)\s+(?:the|to)\s+(?:case|condition|constraint)',
                r'(?:notation|variable|symbol)\s+(?:confusion|error|issue|inconsistency)',
                r'(?:sign|plus|minus)\s+(?:error|mistake|confusion)',
                r'off\s+by\s+(?:a\s+)?(?:factor|sign)',
                r'(?:should|needs\s+to)\s+be\s+(?:negative|positive|the\s+opposite)',
            ]
            for pattern in strong_almost_indicators:
                if re.search(pattern, response_lower, re.IGNORECASE):
                    self.log_fn(f"Post-processing: Found strong 'Almost' indicator, changing from '{prediction}' to 'Almost'")
                    prediction = "Almost"
                    break
        
        # Post-processing: Check if "Incorrect" should be "Partial"
        if prediction == "Incorrect" and response_text:
            response_lower = response_text.lower()
            strong_partial_indicators = [
                r'(?:good|correct)\s+(?:start|approach|direction|idea)',
                r'on\s+the\s+right\s+track',
                r'right\s+idea',
                r'correct\s+method',
                r'correct\s+approach',
                r'good\s+understanding',
                r'understands\s+the\s+problem',
                r'(?:significant|substantial|good)\s+progress',
                r'partial\s+(?:solution|proof|answer)',
                r'incomplete\s+but\s+(?:correct|valid|good)',
            ]
            for pattern in strong_partial_indicators:
                if re.search(pattern, response_lower, re.IGNORECASE):
                    self.log_fn(f"Post-processing: Found strong 'Partial' indicator, changing from 'Incorrect' to 'Partial'")
                    prediction = "Partial"
                    break
        
        # Post-processing: Check if "Partial" should be "Almost"
        if prediction == "Partial" and response_text:
            response_lower = response_text.lower()
            strong_almost_from_partial = [
                r'only\s+(?:a\s+)?(?:one|single|1)\s+(?:minor|small|tiny)\s+(?:error|mistake|issue)',
                r'just\s+(?:a\s+)?(?:one|single|1)\s+(?:minor|small|tiny)\s+(?:error|mistake|issue)',
                r'(?:one|single|1)\s+(?:minor|small|tiny)\s+(?:error|mistake|issue|typo)',
                r'(?:one|single|1)\s+(?:arithmetic|calculation|computation|sign)\s+(?:error|mistake)',
                r'(?:just|only)\s+(?:a\s+)?(?:typo|sign\s+error)',
                r'(?:small|minor|tiny)\s+(?:arithmetic|calculation)\s+(?:error|mistake)',
                r'essentially\s+(?:correct|complete)',
                r'nearly\s+(?:correct|complete|perfect)',
                r'almost\s+(?:correct|perfect)',
                r'95%\s+(?:correct|complete)',
                r'99%\s+(?:correct|complete)',
                r'correct\s+except\s+for\s+(?:a\s+)?(?:minor|small|tiny)',
                r'would\s+be\s+(?:correct|perfect)\s+(?:if|with|after)',
                r'(?:just|only)\s+needs?\s+(?:a\s+)?(?:minor|small|tiny)\s+fix',
                r'(?:one|single)\s+(?:character|letter|digit|sign)\s+(?:error|mistake)',
                r'(?:simple|trivial)\s+(?:error|mistake|fix)',
                r'off\s+by\s+(?:one|1)',
                r'(?:sign|plus|minus)\s+(?:error|confusion)',
            ]
            for pattern in strong_almost_from_partial:
                if re.search(pattern, response_lower, re.IGNORECASE):
                    self.log_fn(f"Post-processing: Found strong 'Almost' indicator (was Partial), changing to 'Almost'")
                    prediction = "Almost"
                    break

        return str(prediction), msg_history
