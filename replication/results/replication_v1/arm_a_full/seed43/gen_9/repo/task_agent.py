"""
Task agent: solves a given task with chain-of-thought reasoning.

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
    Also handles nested JSON objects and escaped characters.
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
        
        # Try to parse the inner content as JSON
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # Try to handle common JSON formatting issues
            # 1. Remove trailing commas before closing braces/brackets
            cleaned = re.sub(r',(\s*[}\]])', r'\1', inner)
            # 2. Try to extract JSON from markdown code blocks within the json tags
            code_block_match = re.search(r'```(?:json)?\s*(.*?)\s*```', cleaned, re.DOTALL)
            if code_block_match:
                cleaned = code_block_match.group(1).strip()
            try:
                results.append(json.loads(cleaned))
            except json.JSONDecodeError:
                # 3. Try to find the first valid JSON object by progressively parsing
                for i in range(len(cleaned), 0, -1):
                    try:
                        partial = cleaned[:i]
                        # Ensure we end on a valid closing brace
                        if partial.rstrip().endswith('}'):
                            results.append(json.loads(partial))
                            break
                    except json.JSONDecodeError:
                        continue
                continue
    return results or None


def _extract_json_fallback(text: str) -> dict | None:
    """Fallback JSON extraction for non-tagged JSON or markdown code blocks.

    Tries to find JSON in markdown code blocks or raw JSON objects.
    """
    patterns = [
        r'```json\s*(.*?)\s*```',  # Markdown JSON blocks
        r'```\s*(\{.*?\})\s*```',  # Generic code blocks with JSON
        r'\{\s*"reasoning".*?\}',  # Raw JSON with reasoning field
        r'\{\s*"response".*?\}',   # Raw JSON with response field
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    return None


def _validate_and_normalize_prediction(prediction: str, grading_guidelines: str) -> str:
    """Validate and normalize the prediction based on grading guidelines.
    
    Ensures the prediction matches expected format from grading guidelines.
    Handles various edge cases including partial credit, score ranges, and
    multiple valid response formats.
    """
    if not prediction or prediction.strip() == "":
        return "None"
    
    prediction = prediction.strip()
    guidelines_lower = grading_guidelines.lower()
    
    # Extract expected score patterns from grading guidelines
    # Common IMO patterns: "7" (full score), "0" (no score), "1-6" (partial)
    
    # Check for IMO-style 0-7 scoring (most common for IMO problems)
    if re.search(r'\b[0-7]\b', grading_guidelines):
        # Look for single digit 0-7 in the prediction
        match = re.search(r'\b([0-7])\b', prediction)
        if match:
            return match.group(1)
        # Also check for spelled-out numbers
        number_words = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3',
            'four': '4', 'five': '5', 'six': '6', 'seven': '7'
        }
        pred_lower = prediction.lower()
        for word, digit in number_words.items():
            if re.search(r'\b' + word + r'\b', pred_lower):
                return digit
    
    # Check for "Correct"/"Incorrect" format
    if "correct" in guidelines_lower or "incorrect" in guidelines_lower:
        pred_lower = prediction.lower()
        # Check for explicit "correct" or "incorrect" mentions
        if re.search(r'\bcorrect\b', pred_lower) and not re.search(r'\bincorrect\b', pred_lower):
            return "Correct"
        elif re.search(r'\bincorrect\b', pred_lower) or re.search(r'\bwrong\b', pred_lower):
            return "Incorrect"
        # Check for yes/no patterns
        if re.search(r'\byes\b', pred_lower) and not re.search(r'\bno\b', pred_lower):
            return "Correct"
        elif re.search(r'\bno\b', pred_lower) and not re.search(r'\byes\b', pred_lower):
            return "Incorrect"
    
    # Check for percentage-based scoring (0-100%)
    if '%' in grading_guidelines or 'percent' in guidelines_lower:
        match = re.search(r'(\d+)%', prediction)
        if match:
            return match.group(0)
        match = re.search(r'\b(\d{1,3})\b', prediction)
        if match:
            val = int(match.group(1))
            if 0 <= val <= 100:
                return f"{val}%"
    
    # Check for fractional scores (e.g., "1/2", "3/4")
    frac_match = re.search(r'\b(\d+)/(\d+)\b', prediction)
    if frac_match:
        return frac_match.group(0)
    
    # Check for decimal scores (e.g., "0.5", "3.5")
    decimal_match = re.search(r'\b(\d+\.\d+)\b', prediction)
    if decimal_match:
        return decimal_match.group(1)
    
    # If no specific format matched, return the cleaned prediction
    # but truncate if it's too long (likely not a valid grade)
    if len(prediction) > 100:
        return prediction[:100] + "..."
    
    return prediction


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

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
        # Extract fields for structured prompting
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert grader for the International Mathematical Olympiad (IMO). Your task is to evaluate a student's answer to a mathematical problem with precision and consistency.

## Domain
{domain}

## Problem Statement
{problem}

## Official Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Your Task

Please evaluate the student's answer following this structured approach:

### Step 1: Problem Analysis
- Summarize what the problem is asking
- Identify key mathematical concepts and required techniques
- Note any critical assumptions or constraints

### Step 2: Solution Mapping
- Break down the official solution into key milestones
- Identify which milestones are essential vs. optional
- Note common alternative valid approaches

### Step 3: Student Answer Evaluation
- Map the student's work to solution milestones
- Identify correct steps with clear justification
- Flag any errors, gaps, or logical flaws
- Check for partial credit opportunities per guidelines

### Step 4: Grade Determination
- Apply grading guidelines systematically
- Consider: correctness, completeness, and clarity
- Assign the most appropriate grade/score

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis following the 4 steps above...",
    "response": "The final grade/score as specified in the grading guidelines"
}}
</json>

IMPORTANT: The "response" field must contain ONLY the grade/score (e.g., "7", "2", "0", "Correct", "Incorrect", etc.) exactly as specified in the grading guidelines. Do not add explanations or extra text in this field."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        last_text = msg_history[-1]["text"]
        extraction_method = "none"
        
        try:
            # First try: extract from <json> tags
            extracted = _extract_jsons(last_text)
            if extracted:
                extraction_method = "json_tags"
                # Try to get response field, fall back to other common fields
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = str(last_json["response"])
                elif "grade" in last_json:
                    prediction = str(last_json["grade"])
                elif "score" in last_json:
                    prediction = str(last_json["score"])
                elif "answer" in last_json:
                    prediction = str(last_json["answer"])
                elif "result" in last_json:
                    prediction = str(last_json["result"])
                elif "evaluation" in last_json:
                    prediction = str(last_json["evaluation"])
                else:
                    # If no recognized field, use the whole JSON as string
                    prediction = json.dumps(last_json)
            else:
                # Second try: fallback extraction for non-tagged JSON
                fallback = _extract_json_fallback(last_text)
                if fallback:
                    extraction_method = "fallback"
                    if "response" in fallback:
                        prediction = str(fallback["response"])
                    elif "grade" in fallback:
                        prediction = str(fallback["grade"])
                    elif "score" in fallback:
                        prediction = str(fallback["score"])
                    elif "answer" in fallback:
                        prediction = str(fallback["answer"])
                    elif "result" in fallback:
                        prediction = str(fallback["result"])
                    elif "evaluation" in fallback:
                        prediction = str(fallback["evaluation"])
                    else:
                        prediction = json.dumps(fallback)
                    self.log_fn(f"Used fallback JSON extraction: {prediction}")
                else:
                    # Third try: direct text extraction for simple responses
                    # Look for patterns like "Score: 7" or "Grade: Correct"
                    direct_patterns = [
                        r'[Ss]core[:\s]+([0-7]|Correct|Incorrect)',
                        r'[Gg]rade[:\s]+([0-7]|Correct|Incorrect)',
                        r'[Ff]inal[\s\w]*[:\s]+([0-7]|Correct|Incorrect)',
                        r'[Ee]valuation[:\s]+([0-7]|Correct|Incorrect)',
                    ]
                    for pattern in direct_patterns:
                        match = re.search(pattern, last_text)
                        if match:
                            extraction_method = "direct_text"
                            prediction = match.group(1)
                            self.log_fn(f"Used direct text extraction: {prediction}")
                            break
            
            # Validate and normalize the prediction
            original_prediction = prediction
            prediction = _validate_and_normalize_prediction(prediction, grading_guidelines)
            
            if prediction != original_prediction:
                self.log_fn(f"Normalized prediction from '{original_prediction}' to '{prediction}'")
            
            if extraction_method != "none":
                self.log_fn(f"Extraction method: {extraction_method}, prediction: {prediction}")
            else:
                self.log_fn(f"Warning: Could not extract prediction from response. Raw text: {last_text[:200]}...")
            
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Try to extract any numeric or keyword from the text as last resort
            try:
                # Look for standalone digits 0-7
                match = re.search(r'\b([0-7])\b', last_text)
                if match:
                    prediction = match.group(1)
                    self.log_fn(f"Recovered prediction from raw text: {prediction}")
            except:
                pass

        return str(prediction), msg_history
