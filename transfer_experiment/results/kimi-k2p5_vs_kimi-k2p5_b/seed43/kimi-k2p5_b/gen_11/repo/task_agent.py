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


def _extract_json_from_text(text: str) -> dict | None:
    """Extract JSON object from text using multiple strategies.
    
    Tries multiple extraction methods in order of reliability:
    1. <json>...</json> tags
    2. ```json...``` code blocks
    3. ```...``` code blocks with JSON content
    4. Raw JSON objects with "response" field
    5. Relaxed JSON parsing with common fixes
    """
    text = text.strip()
    
    # Strategy 1: <json>...</json> tags
    json_tag_pattern = r'<json>\s*(.*?)\s*</json>'
    matches = re.findall(json_tag_pattern, text, re.DOTALL | re.IGNORECASE)
    for match in matches:
        try:
            data = json.loads(match.strip())
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            # Try to fix common JSON issues
            fixed = _fix_json(match.strip())
            if fixed:
                try:
                    data = json.loads(fixed)
                    if isinstance(data, dict):
                        return data
                except json.JSONDecodeError:
                    pass
            continue
    
    # Strategy 2: ```json...``` code blocks
    code_block_pattern = r'```json\s*(.*?)\s*```'
    matches = re.findall(code_block_pattern, text, re.DOTALL | re.IGNORECASE)
    for match in matches:
        try:
            data = json.loads(match.strip())
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            # Try to fix common JSON issues
            fixed = _fix_json(match.strip())
            if fixed:
                try:
                    data = json.loads(fixed)
                    if isinstance(data, dict):
                        return data
                except json.JSONDecodeError:
                    pass
            continue
    
    # Strategy 3: ```...``` code blocks (without json label) containing JSON-like content
    code_block_pattern2 = r'```\s*(\{[\s\S]*?\})\s*```'
    matches = re.findall(code_block_pattern2, text, re.DOTALL | re.IGNORECASE)
    for match in matches:
        try:
            data = json.loads(match.strip())
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            # Try to fix common JSON issues
            fixed = _fix_json(match.strip())
            if fixed:
                try:
                    data = json.loads(fixed)
                    if isinstance(data, dict):
                        return data
                except json.JSONDecodeError:
                    pass
            continue
    
    # Strategy 4: Raw JSON objects with "response" or "classification" field
    # Look for complete JSON objects with balanced braces
    json_pattern = r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}'
    matches = re.findall(json_pattern, text, re.DOTALL)
    for match in matches:
        try:
            data = json.loads(match)
            if isinstance(data, dict) and ("response" in data or "classification" in data or "grade" in data or "result" in data):
                return data
        except json.JSONDecodeError:
            # Try to fix common JSON issues
            fixed = _fix_json(match)
            if fixed:
                try:
                    data = json.loads(fixed)
                    if isinstance(data, dict) and ("response" in data or "classification" in data or "grade" in data or "result" in data):
                        return data
                except json.JSONDecodeError:
                    pass
            continue
    
    # Strategy 5: Try to find any JSON-like structure with response field
    # More aggressive pattern matching for simple key-value pairs
    response_pattern = r'"response"\s*:\s*"([^"]+)"'
    match = re.search(response_pattern, text, re.IGNORECASE)
    if match:
        return {"response": match.group(1)}
    
    classification_pattern = r'"classification"\s*:\s*"([^"]+)"'
    match = re.search(classification_pattern, text, re.IGNORECASE)
    if match:
        return {"classification": match.group(1)}
    
    # Strategy 6: Direct pattern matching for all four labels
    # Check for "almost" first as it's the most specific
    almost_pattern = r'"almost"'
    if re.search(almost_pattern, text, re.IGNORECASE):
        return {"response": "almost"}
    
    return None


def _fix_json(text: str) -> str | None:
    """Try to fix common JSON formatting issues."""
    if not text or not text.strip():
        return None
    
    text = text.strip()
    
    # Fix 1: Remove trailing commas before closing braces
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)
    
    # Fix 2: Replace single quotes with double quotes (carefully)
    # Only do this for simple cases to avoid breaking apostrophes in text
    if "'" in text and '"' not in text:
        text = text.replace("'", '"')
    
    # Fix 3: Ensure the text starts with { and ends with }
    if not text.startswith('{'):
        # Find the first {
        start = text.find('{')
        if start != -1:
            text = text[start:]
    
    if not text.endswith('}'):
        # Find the last }
        end = text.rfind('}')
        if end != -1:
            text = text[:end+1]
    
    # Fix 4: Handle unescaped newlines in string values
    # This is a common issue with LLM outputs
    def escape_newlines_in_json(match):
        # Keep the outer quotes and content, but escape newlines
        content = match.group(1)
        # Escape unescaped newlines and tabs
        content = content.replace('\\n', '\n').replace('\n', '\\n')
        content = content.replace('\\t', '\t').replace('\t', '\\t')
        return '"' + content + '"'
    
    # Try to fix unescaped characters in string values
    # This is a simplified approach - match strings and fix them
    try:
        # Find all string values and try to fix them
        string_pattern = r'"((?:[^"\\]|\\.)*)"'
        text = re.sub(string_pattern, escape_newlines_in_json, text)
    except Exception:
        pass
    
    return text if text.startswith('{') and text.endswith('}') else None


def _extract_response_direct(text: str) -> str | None:
    """Extract response by looking for keywords directly in text."""
    text_lower = text.lower()
    
    # Look for explicit classification statements with response/classification keywords
    # These patterns look for explicit field assignments in JSON-like or natural language
    patterns = [
        # JSON-style field assignments (highest priority)
        r'"response"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"classification"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"grade"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"result"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"answer"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"outcome"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"evaluation"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"verdict"\s*:\s*"(correct|incorrect|partial|almost)"',
        # Natural language classification statements
        r'classification[\s:]+"?(correct|incorrect|partial|almost)"?',
        r'classify[\s:]+"?(correct|incorrect|partial|almost)"?',
        r'response[\s:]+"?(correct|incorrect|partial|almost)"?',
        r'grade[\s:]+"?(correct|incorrect|partial|almost)"?',
        r'result[\s:]+"?(correct|incorrect|partial|almost)"?',
        r'outcome[\s:]+"?(correct|incorrect|partial|almost)"?',
        r'evaluation[\s:]+"?(correct|incorrect|partial|almost)"?',
        r'verdict[\s:]+"?(correct|incorrect|partial|almost)"?',
        # Explicit statements about the classification
        r'"?(correct|incorrect|partial|almost)"?\s*is\s*the\s*correct\s*classification',
        r'the\s*answer\s*is\s*"?(correct|incorrect|partial|almost)"?',
        r'should\s*be\s*classified\s*as\s*"?(correct|incorrect|partial|almost)"?',
        r'classify\s*as\s*"?(correct|incorrect|partial|almost)"?',
        r'classified\s*as\s*"?(correct|incorrect|partial|almost)"?',
        r'i\s*classify\s*this\s*as\s*"?(correct|incorrect|partial|almost)"?',
        r'this\s*is\s*"?(correct|incorrect|partial|almost)"?',
        r'the\s*student[\'\']?s\s*answer\s*is\s*"?(correct|incorrect|partial|almost)"?',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            result = match.group(1).lower().strip()
            if result in ["correct", "incorrect", "partial", "almost"]:
                return result
    
    # Count occurrences of each label (only count standalone words)
    # Exclude occurrences within the reasoning field by looking for the response field area
    # Try to find the response field specifically
    response_section_match = re.search(r'"response"\s*:\s*"([^"]*(?:correct|incorrect|partial|almost)[^"]*)"', text_lower)
    if response_section_match:
        response_section = response_section_match.group(1)
        if "almost" in response_section:
            return "almost"
        if "correct" in response_section and "incorrect" not in response_section:
            return "correct"
        elif "partial" in response_section:
            return "partial"
        elif "incorrect" in response_section:
            return "incorrect"
    
    # Count standalone occurrences
    correct_count = len(re.findall(r'\bcorrect\b', text_lower))
    incorrect_count = len(re.findall(r'\bincorrect\b', text_lower))
    partial_count = len(re.findall(r'\bpartial\b', text_lower))
    almost_count = len(re.findall(r'\balmost\b', text_lower))
    
    # If only one appears, use that
    if correct_count > 0 and incorrect_count == 0 and partial_count == 0 and almost_count == 0:
        return "correct"
    if incorrect_count > 0 and correct_count == 0 and partial_count == 0 and almost_count == 0:
        return "incorrect"
    if partial_count > 0 and correct_count == 0 and incorrect_count == 0 and almost_count == 0:
        return "partial"
    if almost_count > 0 and correct_count == 0 and incorrect_count == 0 and partial_count == 0:
        return "almost"
    
    # If multiple appear, use priority: almost > incorrect > partial > correct
    # This prioritizes more specific/less common classifications
    if almost_count > 0:
        return "almost"
    if incorrect_count > 0:
        return "incorrect"
    if partial_count > 0:
        return "partial"
    if correct_count > 0:
        return "correct"
    
    return None


def _normalize_prediction(prediction: str) -> str:
    """Normalize prediction to one of the four valid labels."""
    pred_str = str(prediction).lower().strip()
    
    # Direct match
    if pred_str in ["correct", "incorrect", "partial", "almost"]:
        return pred_str
    
    # Check for almost first (most specific and least common)
    if "almost" in pred_str:
        return "almost"
    
    # Check for partial (more specific than correct/incorrect)
    if "partial" in pred_str:
        return "partial"
    
    # Check for incorrect/wrong/false/error (more specific than correct)
    if "incorrect" in pred_str or "wrong" in pred_str or "false" in pred_str or "error" in pred_str:
        return "incorrect"
    
    # Check for correct (but not incorrect)
    if "correct" in pred_str and "incorrect" not in pred_str:
        return "correct"
    
    # Check for complete/full
    if "complete" in pred_str or "full" in pred_str:
        return "correct"
    
    # Default fallback - use "incorrect" as it's more conservative
    # when we can't determine the classification
    return "incorrect"


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
        # Extract key fields from inputs for better prompting
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        instruction = f"""You are an expert mathematical olympiad grader. Your task is to evaluate a student's answer to an IMO-level problem.

## Problem Statement:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Classification Guidelines:

**"correct"** - The student provided a complete, correct solution with proper reasoning. The answer:
- Arrives at the correct final answer
- Contains valid mathematical reasoning throughout
- May have minor notation issues or small gaps that don't affect correctness
- Awarded full points (6-7 points)

**"almost"** - The student's solution is nearly complete with only minor verification errors or small gaps. The answer:
- Has the correct final answer or very close to it
- Contains valid reasoning with only minor mistakes in verification or small details
- Shows essentially complete understanding with minor technical errors
- The student would receive 5-6 points (high partial credit, very close to full)
- KEY DISTINCTION: The solution is "almost correct" - the main ideas are right but there are minor technical flaws

**"partial"** - The student made meaningful progress but the solution has significant gaps or errors. The answer:
- Contains valid and significant progress toward the solution
- Has correct lemmas, propositions, or intermediate steps
- May have the right approach but incomplete execution or significant errors
- Shows understanding of key concepts even if final answer is wrong or missing
- The student would receive 1-4 points (partial credit)
- KEY DISTINCTION: There is genuine mathematical progress but major gaps remain

**"incorrect"** - The student's answer is wrong or fundamentally flawed. The answer:
- Has an incorrect final answer with no valid path to solution
- Contains critical logical or mathematical errors
- Shows no meaningful progress toward the solution
- Is incomplete with no useful partial results
- The student would receive 0 points
- KEY DISTINCTION: The answer is fundamentally wrong or trivial

## CRITICAL DECISION FRAMEWORK - FOLLOW THESE STEPS IN ORDER:

**STEP 1: Analyze the Grading Guidelines Carefully**
The grading guidelines describe what achievements WOULD merit partial or full credit. However, you must verify whether the student ACTUALLY achieved these criteria:
- Look at what the guidelines list as achievements (e.g., "Proved that X", "Observed that Y")
- Check if the student's answer ACTUALLY contains these achievements
- The presence of "(Partial)" or "(Almost)" in guidelines does NOT automatically mean the student gets that grade - you must verify they actually did what is described

**STEP 2: Compare Student Answer to Official Solution**
- Does the student arrive at the same final answer as the official solution?
- Does the student's reasoning align with the official solution's approach?
- Are there gaps or errors in the student's reasoning?

**STEP 3: Check for "incorrect" FIRST**
Classify as "incorrect" if ANY of the following are true:
- The student's final answer is completely wrong AND there's no valid mathematical path to the solution
- The student's approach is fundamentally misguided or based on false premises
- The answer contains only trivial or nonsensical statements
- There is NO meaningful mathematical progress toward the official solution
- The student would receive 0 points

**STEP 4: Check for "almost"**
Classify as "almost" if ALL of the following are true:
- The solution is very close to being correct (not just "has some progress")
- The final answer is correct or nearly correct
- The reasoning is mostly valid with only minor verification errors or small gaps
- The errors are technical/verification issues, not fundamental misunderstandings
- The student would receive 5-6 points (high partial, very close to full credit)
- KEY: This is for solutions that are "almost correct" - the main structure is right

**STEP 5: Check for "partial"**
Classify as "partial" if:
- The student made GENUINE mathematical progress (proved at least one correct lemma, identified a key insight, or made significant progress)
- BUT there are significant gaps, major errors, or the final answer is wrong/missing
- The student showed understanding of key concepts with valid mathematical content
- The student would receive 1-4 points
- KEY: There must be SUBSTANTIAL valid content, not just trivial observations

**STEP 6: Check for "correct"**
Only classify as "correct" if ALL of the following are true:
- Final answer matches the official solution
- Reasoning is valid throughout (or has only trivial errors)
- Solution is essentially complete
- The student would receive 6-7 points

## Your Task:
First, think through your analysis step by step. Then classify the student's answer into EXACTLY ONE category: "correct", "almost", "partial", or "incorrect".

**IMPORTANT GUIDELINES**:
1. The grading guidelines describe POTENTIAL achievements - you must verify the student ACTUALLY achieved them
2. Be STRICT about "incorrect": If the answer shows NO meaningful progress, classify as "incorrect"
3. Be STRICT about "almost": Only use "almost" if the solution is VERY CLOSE to correct (5-6 points worth)
4. "Partial" requires GENUINE mathematical progress - not just random attempts
5. When in doubt between "partial" and "incorrect", choose "incorrect" if there's no substantial valid mathematical content
6. Only classify as "correct" if the solution is truly complete and correct
7. Be conservative with "correct" - a solution with significant gaps should be "partial" or "almost"
8. Use "almost" for solutions that are very close to correct but have minor verification errors

**DECISION TIPS**:
- If the student has the right final answer but major gaps in reasoning → "partial" (not "almost")
- If the student has wrong final answer but proved some valid lemmas → "partial"
- If the student is close to correct but has minor technical errors → "almost"
- If the student shows essentially no valid progress → "incorrect"

You MUST respond with a JSON object in this exact format:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis here. Explain what the student got right, what they got wrong, and why you chose this classification.",
    "response": "correct"
}}
</json>

OR

<json>
{{
    "reasoning": "Your detailed step-by-step analysis here. Explain what the student got right, what they got wrong, and why you chose this classification.",
    "response": "almost"
}}
</json>

OR

<json>
{{
    "reasoning": "Your detailed step-by-step analysis here. Explain what the student got right, what they got wrong, and why you chose this classification.",
    "response": "partial"
}}
</json>

OR

<json>
{{
    "reasoning": "Your detailed step-by-step analysis here. Explain what the student got right, what they got wrong, and why you chose this classification.",
    "response": "incorrect"
}}
</json>

Replace the "response" value with your classification. Include detailed reasoning in the "reasoning" field. Do not include any other text outside the JSON block."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from response
        prediction = "incorrect"  # Default fallback
        try:
            if msg_history and len(msg_history) > 0:
                response_text = msg_history[-1].get("text", "")
                
                if response_text:
                    # First try: extract from JSON
                    json_data = _extract_json_from_text(response_text)
                    
                    if json_data and isinstance(json_data, dict):
                        # Priority order for JSON fields: response > classification > grade > result
                        # Within each field, prioritize "almost" > "incorrect" > "partial" > "correct"
                        found_value = None
                        
                        # Check primary fields in priority order
                        for field in ["response", "classification", "grade", "result"]:
                            if field in json_data:
                                value = str(json_data[field]).lower().strip()
                                # Check for exact matches first
                                if value in ["correct", "incorrect", "partial", "almost"]:
                                    found_value = value
                                    break
                                # Check for partial matches
                                if "almost" in value:
                                    found_value = "almost"
                                    break
                                elif "incorrect" in value:
                                    found_value = "incorrect"
                                    break
                                elif "partial" in value:
                                    found_value = "partial"
                                    break
                                elif "correct" in value:
                                    found_value = "correct"
                                    break
                        
                        if found_value:
                            prediction = found_value
                        else:
                            # Try to find any value that looks like our labels
                            for key, value in json_data.items():
                                if isinstance(value, str):
                                    val_lower = value.lower()
                                    if val_lower in ["correct", "incorrect", "partial", "almost"]:
                                        prediction = val_lower
                                        break
                                    elif "almost" in val_lower:
                                        prediction = "almost"
                                        break
                                    elif "incorrect" in val_lower:
                                        prediction = "incorrect"
                                        break
                                    elif "partial" in val_lower:
                                        prediction = "partial"
                                        break
                                    elif "correct" in val_lower:
                                        prediction = "correct"
                                        break
                    else:
                        # Try direct extraction from text
                        direct = _extract_response_direct(response_text)
                        if direct:
                            prediction = direct
                        else:
                            self.log_fn(f"No valid response found in: {response_text[:200]}...")
                else:
                    self.log_fn("Empty response text in msg_history")
            else:
                self.log_fn("Empty msg_history")
                
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            import traceback
            self.log_fn(traceback.format_exc())

        return str(prediction), msg_history
