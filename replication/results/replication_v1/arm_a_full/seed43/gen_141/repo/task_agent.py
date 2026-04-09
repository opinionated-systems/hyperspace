"""
Task agent: solves a given task with chain-of-thought reasoning.

Reimplemented from facebookresearch/HyperAgents task_agent.py.
Same interface, same JSON output format, same extraction logic.

This agent is designed for Problem Source Classification - identifying the origin
of mathematical problems (USAMO 2025, Novel Problem, Modified IMO 2024 problems, etc.)
"""

from __future__ import annotations

import json
import logging
import re

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)

# Valid problem source labels for classification task
VALID_PROBLEM_SOURCES = [
    "USAMO 2025",
    "Novel Problem",
    "(Modified) IMO 2024 P1",
    "(Modified) IMO 2024 P2",
    "(Modified) IMO 2024 P3",
    "(Modified) IMO 2024 P4",
    "(Modified) IMO 2024 P5",
    "(Modified) IMO 2024 P6",
]


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
            # Try to extract JSON from markdown code blocks within the content
            code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', inner, re.DOTALL)
            if code_block_match:
                try:
                    results.append(json.loads(code_block_match.group(1)))
                except json.JSONDecodeError:
                    continue
            else:
                continue
    return results or None


def _extract_json_fallback(text: str) -> dict | None:
    """Fallback JSON extraction for non-tagged JSON or markdown code blocks."""
    # First try markdown code blocks
    patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(\{.*?\})\s*```',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    
    # Find all potential JSON object starts
    json_candidates = []
    for match in re.finditer(r'\{\s*"', text):
        start = match.start()
        stack = []
        in_string = False
        escape_next = False
        
        for i in range(start, len(text)):
            char = text[i]
            
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\':
                escape_next = True
                continue
            
            if char == '"' and not in_string:
                in_string = True
            elif char == '"' and in_string:
                in_string = False
            elif not in_string:
                if char == '{':
                    stack.append('{')
                elif char == '}':
                    if stack:
                        stack.pop()
                        if not stack:
                            candidate = text[start:i+1]
                            json_candidates.append(candidate)
                            break
                    else:
                        break
    
    best_match = None
    for candidate in json_candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                if any(key in parsed for key in ["response", "problem_source", "source", "answer"]):
                    return parsed
                if best_match is None:
                    best_match = parsed
        except json.JSONDecodeError:
            continue
    
    return best_match


def _normalize_problem_source(prediction: str) -> str:
    """Normalize a prediction to one of the valid problem source labels."""
    if not prediction:
        return prediction
        
    prediction_clean = prediction.strip()
    prediction_lower = prediction_clean.lower()
    
    # Direct match
    for source in VALID_PROBLEM_SOURCES:
        if prediction_clean == source:
            return source
    
    # Case-insensitive match
    for source in VALID_PROBLEM_SOURCES:
        if prediction_clean.lower() == source.lower():
            return source
    
    # Partial match for IMO problems
    if "imo 2024 p1" in prediction_lower or ("modified" in prediction_lower and "p1" in prediction_lower):
        return "(Modified) IMO 2024 P1"
    if "imo 2024 p2" in prediction_lower or ("modified" in prediction_lower and "p2" in prediction_lower):
        return "(Modified) IMO 2024 P2"
    if "imo 2024 p3" in prediction_lower or ("modified" in prediction_lower and "p3" in prediction_lower):
        return "(Modified) IMO 2024 P3"
    if "imo 2024 p4" in prediction_lower or ("modified" in prediction_lower and "p4" in prediction_lower):
        return "(Modified) IMO 2024 P4"
    if "imo 2024 p5" in prediction_lower or ("modified" in prediction_lower and "p5" in prediction_lower):
        return "(Modified) IMO 2024 P5"
    if "imo 2024 p6" in prediction_lower or ("modified" in prediction_lower and "p6" in prediction_lower):
        return "(Modified) IMO 2024 P6"
    
    # Match for USAMO
    if "usamo" in prediction_lower:
        return "USAMO 2025"
    
    # Match for Novel Problem
    if "novel" in prediction_lower or "new problem" in prediction_lower:
        return "Novel Problem"
    
    return prediction_clean


def _extract_problem_source_from_text(text: str) -> str | None:
    """Extract problem source directly from text using pattern matching."""
    text_lower = text.lower()
    
    # Look for explicit mentions of problem sources
    patterns = [
        (r'\bUSAMO\s*2025\b', 'USAMO 2025'),
        (r'\bNovel\s*Problem\b', 'Novel Problem'),
        (r'\(Modified\)\s*IMO\s*2024\s*P1', '(Modified) IMO 2024 P1'),
        (r'\(Modified\)\s*IMO\s*2024\s*P2', '(Modified) IMO 2024 P2'),
        (r'\(Modified\)\s*IMO\s*2024\s*P3', '(Modified) IMO 2024 P3'),
        (r'\(Modified\)\s*IMO\s*2024\s*P4', '(Modified) IMO 2024 P4'),
        (r'\(Modified\)\s*IMO\s*2024\s*P5', '(Modified) IMO 2024 P5'),
        (r'\(Modified\)\s*IMO\s*2024\s*P6', '(Modified) IMO 2024 P6'),
        (r'Modified\s*IMO\s*2024\s*P1', '(Modified) IMO 2024 P1'),
        (r'Modified\s*IMO\s*2024\s*P2', '(Modified) IMO 2024 P2'),
        (r'Modified\s*IMO\s*2024\s*P3', '(Modified) IMO 2024 P3'),
        (r'Modified\s*IMO\s*2024\s*P4', '(Modified) IMO 2024 P4'),
        (r'Modified\s*IMO\s*2024\s*P5', '(Modified) IMO 2024 P5'),
        (r'Modified\s*IMO\s*2024\s*P6', '(Modified) IMO 2024 P6'),
        (r'IMO\s*2024\s*P1', '(Modified) IMO 2024 P1'),
        (r'IMO\s*2024\s*P2', '(Modified) IMO 2024 P2'),
        (r'IMO\s*2024\s*P3', '(Modified) IMO 2024 P3'),
        (r'IMO\s*2024\s*P4', '(Modified) IMO 2024 P4'),
        (r'IMO\s*2024\s*P5', '(Modified) IMO 2024 P5'),
        (r'IMO\s*2024\s*P6', '(Modified) IMO 2024 P6'),
    ]
    
    for pattern, source in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return source
    
    return None


class TaskAgent:
    """Task agent for Problem Source Classification."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer
                    Note: For this task, we need to predict "Problem Source" which is
                    the origin of the problem (USAMO 2025, Novel Problem, etc.)

        Returns:
            (prediction, msg_history)
        """
        # Extract fields
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # The actual label to predict is in "Problem Source" field
        # This is a classification task, not a grading task

        instruction = f"""You are an expert in mathematical competition problems. Your task is to identify the ORIGIN/SOURCE of a given mathematical problem.

## Domain
{domain}

## Problem Statement
{problem}

## Official Solution
{solution}

## Grading Guidelines (describes what the student achieved)
{grading_guidelines}

## Student's Answer
{student_answer}

## Your Task

Analyze the problem and determine its SOURCE. The problem source indicates where this problem originated from.

### Analysis Steps:
1. **Problem Structure Analysis**: Examine the mathematical structure, notation, and problem type
2. **Difficulty Assessment**: Evaluate the complexity and sophistication level
3. **Style Analysis**: Look for characteristic features of known competitions
4. **Comparison**: Compare against known patterns from USAMO, IMO, and novel problems

### Classification Categories:
You MUST classify the problem into ONE of these exact categories:

1. **"USAMO 2025"** - Problems from the 2025 USA Mathematical Olympiad
   - Characteristics: Advanced high school competition level, proof-based
   
2. **"Novel Problem"** - Original problems not from standard competitions
   - Characteristics: Unique structure, may combine multiple techniques innovatively
   
3. **"(Modified) IMO 2024 P1"** - Modified version of IMO 2024 Problem 1
4. **"(Modified) IMO 2024 P2"** - Modified version of IMO 2024 Problem 2
5. **"(Modified) IMO 2024 P3"** - Modified version of IMO 2024 Problem 3
6. **"(Modified) IMO 2024 P4"** - Modified version of IMO 2024 Problem 4
7. **"(Modified) IMO 2024 P5"** - Modified version of IMO 2024 Problem 5
8. **"(Modified) IMO 2024 P6"** - Modified version of IMO 2024 Problem 6
   - Characteristics: Based on actual IMO 2024 problems but with modifications

Respond ONLY in JSON format wrapped in <json> tags with the following exact schema:
<json>
{{
    "reasoning": "Your detailed analysis of why this problem matches a specific source...",
    "problem_source": "The exact source label from the list above",
    "confidence": 0.85
}}
</json>

CRITICAL INSTRUCTIONS:
1. The "problem_source" field MUST contain ONLY ONE of these exact values:
   "USAMO 2025", "Novel Problem", "(Modified) IMO 2024 P1", "(Modified) IMO 2024 P2", 
   "(Modified) IMO 2024 P3", "(Modified) IMO 2024 P4", "(Modified) IMO 2024 P5", 
   "(Modified) IMO 2024 P6"
2. Do NOT use grading labels like "Correct", "Partial", "Incorrect" - this is NOT a grading task
3. Do NOT output numeric scores - this is a classification task
4. The "confidence" field is OPTIONAL (0-1 range)
5. Wrap your entire JSON response in <json>...</json> tags."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        last_text = msg_history[-1]["text"] if msg_history else ""
        extraction_method = "none"
        
        try:
            # First try: extract from <json> tags
            extracted = _extract_jsons(last_text)
            if extracted:
                extraction_method = "json_tags"
                last_json = extracted[-1]
                
                # Look for problem_source field first
                if "problem_source" in last_json:
                    prediction = str(last_json["problem_source"])
                elif "source" in last_json:
                    prediction = str(last_json["source"])
                elif "response" in last_json:
                    prediction = str(last_json["response"])
                elif "answer" in last_json:
                    prediction = str(last_json["answer"])
                else:
                    prediction = json.dumps(last_json)
            else:
                # Second try: fallback extraction for non-tagged JSON
                fallback = _extract_json_fallback(last_text)
                if fallback:
                    extraction_method = "fallback"
                    if "problem_source" in fallback:
                        prediction = str(fallback["problem_source"])
                    elif "source" in fallback:
                        prediction = str(fallback["source"])
                    elif "response" in fallback:
                        prediction = str(fallback["response"])
                    elif "answer" in fallback:
                        prediction = str(fallback["answer"])
                    else:
                        prediction = json.dumps(fallback)
                    
                    self.log_fn(f"Used fallback JSON extraction: {prediction}")
                else:
                    # Third try: direct text extraction
                    extraction_method = "direct"
                    lines = [line.strip() for line in last_text.split('\n') if line.strip()]
                    if lines:
                        last_line = lines[-1]
                        if len(last_line) < 100 and not last_line.startswith('{') and not last_line.startswith('<'):
                            prediction = last_line
                            self.log_fn(f"Used direct text extraction: {prediction}")
            
            # Fourth try: Look for problem source patterns in the text
            if prediction == "None" or prediction == json.dumps({}):
                source_match = _extract_problem_source_from_text(last_text)
                if source_match:
                    prediction = source_match
                    extraction_method = "source_regex"
                    self.log_fn(f"Used source regex extraction: {prediction}")
            
            # Normalize the prediction to valid problem source
            original_prediction = prediction
            prediction = _normalize_problem_source(prediction)
            
            if original_prediction != prediction:
                self.log_fn(f"Normalized prediction from '{original_prediction}' to '{prediction}'")
            
            # Validate that prediction is one of the valid sources
            if prediction not in VALID_PROBLEM_SOURCES:
                self.log_fn(f"Warning: Prediction '{prediction}' is not a valid problem source")
                # Try one more time to extract from the full text
                final_attempt = _extract_problem_source_from_text(last_text)
                if final_attempt:
                    prediction = final_attempt
                    self.log_fn(f"Final extraction attempt succeeded: {prediction}")
            
            self.log_fn(f"Extraction method: {extraction_method}, final prediction: {prediction}")
            
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Last resort: try to find any problem source in the text
            try:
                source_match = _extract_problem_source_from_text(last_text)
                if source_match:
                    prediction = source_match
                    self.log_fn(f"Used emergency source extraction: {prediction}")
            except Exception:
                pass

        return str(prediction), msg_history
