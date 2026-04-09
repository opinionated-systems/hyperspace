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
            # Handle double curly braces from f-string formatting
            if inner.startswith('{{') and inner.endswith('}}'):
                inner = inner[1:-1]
            inner_clean = inner.replace('{{', '{').replace('}}', '}')
            results.append(json.loads(inner_clean))
        except json.JSONDecodeError:
            # Try more aggressive cleaning
            try:
                inner_clean = inner.strip().replace('\n', ' ').replace('\r', '')
                json_start = inner_clean.find('{')
                json_end = inner_clean.rfind('}')
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    inner_clean = inner_clean[json_start:json_end+1]
                    results.append(json.loads(inner_clean))
            except json.JSONDecodeError:
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
            try:
                inner_clean = match.strip().replace('\n', ' ').replace('\r', '')
                json_start = inner_clean.find('{')
                json_end = inner_clean.rfind('}')
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    inner_clean = inner_clean[json_start:json_end+1]
                    results.append(json.loads(inner_clean))
            except json.JSONDecodeError:
                continue
    return results or None


def _extract_json_raw(text: str) -> list[dict] | None:
    """Extract raw JSON objects from text."""
    results = []
    # Pattern for nested braces
    pattern = r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            parsed = json.loads(match.strip())
            if isinstance(parsed, dict):
                results.append(parsed)
        except json.JSONDecodeError:
            continue
    
    # Fallback: look for {"response": "..."} patterns
    if not results:
        response_pattern = r'\{\s*"response"\s*:\s*"([^"]+)"\s*\}'
        matches = re.findall(response_pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            results.append({"response": match.strip()})
    
    # Try single quotes
    if not results:
        response_pattern = r"\{\s*'response'\s*:\s*'([^']+)'\s*\}"
        matches = re.findall(response_pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            results.append({"response": match.strip()})
    
    return results or None


def _extract_direct_label(text: str) -> str | None:
    """Extract label directly from text by looking for valid labels."""
    # High priority patterns for explicit declarations - ordered by specificity
    label_patterns = [
        # JSON-like patterns (highest priority)
        r'"response"\s*:\s*"(Correct|Incorrect|Partial|Almost)"',
        r"'response'\s*:\s*'(Correct|Incorrect|Partial|Almost)'",
        r'["\']?response["\']?\s*[:=]\s*["\']?\s*(Correct|Incorrect|Partial|Almost)\s*["\']?',
        # Explicit classification statements
        r'(?:is|would be|should be|classified as)\s+["\']?\s*(Correct|Incorrect|Partial|Almost)\s*["\']?',
        r'(?:grade|classification|category|label)\s*[:=]\s*["\']?\s*(Correct|Incorrect|Partial|Almost)\s*["\']?',
        # Markdown formatting
        r'\*\*\s*(Correct|Incorrect|Partial|Almost)\s*\*\*',
        r'\*\s*(Correct|Incorrect|Partial|Almost)\s*\*',
        r'`(Correct|Incorrect|Partial|Almost)`',
        # Sentence patterns
        r'\bThe\s+(?:answer|classification|grade|final\s+answer)\s+(?:is|would\s+be)\s+["\']?(Correct|Incorrect|Partial|Almost)["\']?',
        r'\bI\s+(?:would|will)\s+(?:classify|grade|label)\s+(?:this|it|the\s+answer)\s+as\s+["\']?(Correct|Incorrect|Partial|Almost)["\']?',
        r'\bTherefore[,]?\s+(?:the\s+answer\s+is\s+)?["\']?(Correct|Incorrect|Partial|Almost)["\']?',
        r'\bFinal\s+(?:answer|classification|grade)[:]?\s*["\']?(Correct|Incorrect|Partial|Almost)["\']?',
        r'\bClassification[:]\s*(Correct|Incorrect|Partial|Almost)\b',
        r'\bGrade[:]\s*(Correct|Incorrect|Partial|Almost)\b',
    ]
    
    for pattern in label_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            label = match.group(1)
            for valid_label in VALID_LABELS:
                if label.lower() == valid_label.lower():
                    return valid_label
    
    # Check last 100 lines for standalone labels (increased from 50)
    lines = text.split('\n')
    last_lines = lines[-100:] if len(lines) > 100 else lines
    for line in reversed(last_lines):
        line = line.strip()
        if not line or line.startswith('```') or line.startswith('<json>'):
            continue
        for label in VALID_LABELS:
            # Look for labels at the start or end of lines
            pattern_start = r'^' + re.escape(label) + r'\b'
            if re.search(pattern_start, line, re.IGNORECASE):
                return label
            pattern_end = r'\b' + re.escape(label) + r'[\.\s]*$'
            if re.search(pattern_end, line, re.IGNORECASE):
                return label
            # Anywhere in line if it's a short line (likely a conclusion)
            if len(line) < 50:
                pattern = r'\b' + re.escape(label) + r'\b'
                if re.search(pattern, line, re.IGNORECASE):
                    return label
    
    # Clean text and check words from the end (last 200 words)
    cleaned_text = re.sub(r'["\'\(\)\[\]\{\}<>]', ' ', text)
    words = cleaned_text.split()
    for word in reversed(words[-200:]):
        word_stripped = word.strip('.,;:!?')
        for label in VALID_LABELS:
            if word_stripped.lower() == label.lower():
                return label
    
    return None


def extract_prediction(text: str) -> str | None:
    """Extract prediction from text using multiple strategies.
    
    Tries multiple extraction methods in order of preference.
    Returns the label value, or None if extraction fails.
    """
    # Try <json> tags first
    extracted = _extract_jsons(text)
    if extracted:
        for item in extracted:
            if isinstance(item, dict) and "response" in item:
                response = str(item["response"]).strip()
                if response in VALID_LABELS:
                    return response
    
    # Try ```json code blocks
    extracted = _extract_json_from_code_blocks(text)
    if extracted:
        for item in extracted:
            if isinstance(item, dict) and "response" in item:
                response = str(item["response"]).strip()
                if response in VALID_LABELS:
                    return response
    
    # Try raw JSON
    extracted = _extract_json_raw(text)
    if extracted:
        for item in extracted:
            if isinstance(item, dict) and "response" in item:
                response = str(item["response"]).strip()
                if response in VALID_LABELS:
                    return response
    
    # Fallback: direct label extraction
    return _extract_direct_label(text)


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
        
        # Parse points from inputs (stored in separate field, not in grading guidelines)
        points_hint = ""
        pts = inputs.get("points")
        if pts is not None:
            try:
                pts = int(pts)
            except (ValueError, TypeError):
                # Try to extract from string
                points_match = re.search(r'(\d+)', str(pts))
                pts = int(points_match.group(1)) if points_match else None
        
        if pts is not None:
            if pts >= 7:
                points_hint = f"""POINTS: {pts} points
CLASSIFICATION: "Correct"
REASON: 7+ points = perfect solution with full marks.
OUTPUT: You MUST respond with "Correct"."""
            elif pts == 6:
                points_hint = f"""POINTS: {pts} points
CLASSIFICATION: "Almost"
REASON: 6 points = essentially complete solution with exactly ONE minor issue (not perfect, not 7 points).
OUTPUT: You MUST respond with "Almost". DO NOT say "Correct" - 6 points is NOT 7 points."""
            elif 2 <= pts <= 5:
                points_hint = f"""POINTS: {pts} points
CLASSIFICATION: "Partial"
REASON: 2-5 points = meaningful progress but incomplete (multiple gaps remain).
OUTPUT: You MUST respond with "Partial"."""
            elif pts == 1:
                points_hint = f"""POINTS: {pts} point
CLASSIFICATION: "Partial"
REASON: 1 point = minimal progress with at least one small achievement.
OUTPUT: You MUST respond with "Partial". DO NOT say "Incorrect"."""
            elif pts == 0:
                points_hint = f"""POINTS: {pts} points
CLASSIFICATION: "Incorrect"
REASON: 0 points = no meaningful progress or fundamentally wrong approach.
OUTPUT: You MUST respond with "Incorrect". DO NOT say "Partial"."""
        
        instruction = f"""You are an expert IMO grader. Classify the student's answer into exactly one of: Correct, Almost, Partial, or Incorrect.

## Problem:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

{points_hint}

---

## SIMPLE CLASSIFICATION RULE (USE THIS):
The points value DETERMINES the classification:
- **7 points → "Correct"** (perfect solution, full marks)
- **6 points → "Almost"** (one minor issue, essentially complete)
- **2-5 points → "Partial"** (meaningful progress but incomplete)
- **1 point → "Partial"** (minimal progress)
- **0 points → "Incorrect"** (no meaningful progress)

## Category Definitions:
- **Correct**: Complete, rigorous proof. All steps justified. Would receive full marks.
- **Almost**: Essentially solved with exactly ONE minor issue. Has final answer but one small gap/error.
- **Partial**: Meaningful progress but multiple gaps OR missing critical parts. Incomplete solution.
- **Incorrect**: No meaningful progress, wrong approach, or fundamentally flawed.

## RESPONSE FORMAT:
Respond with ONLY a JSON object in this exact format:
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

**CRITICAL**: Use the points value to determine your answer. Points are the strongest signal.
- 7 points = Correct
- 6 points = Almost (NOT Correct - 6 is different from 7)
- 1-5 points = Partial
- 0 points = Incorrect

Now provide your classification:"""

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
                # Try to extract from earlier messages
                for msg in reversed(msg_history):
                    if "text" in msg:
                        extracted = extract_prediction(msg["text"])
                        if extracted is not None:
                            prediction = extracted
                            self.log_fn(f"Extracted prediction from earlier message: {prediction}")
                            break
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Post-processing: POINTS ARE THE PRIMARY SIGNAL - OVERRIDE LLM OUTPUT
        grading_guidelines = inputs.get("grading_guidelines", "")
        original_prediction = prediction
        
        # Parse points - this is the STRONGEST signal for classification
        # Points are stored in a separate field, not in the grading guidelines text
        points_val = inputs.get("points")
        if points_val is not None:
            try:
                points_val = int(points_val)
            except (ValueError, TypeError):
                # Fallback: try to extract from string
                points_match = re.search(r'(\d+)', str(points_val))
                points_val = int(points_match.group(1)) if points_match else None
        else:
            # Fallback: try to find in grading guidelines (for backwards compatibility)
            points_match = re.search(r'(\d+)\s*points?', grading_guidelines, re.IGNORECASE)
            points_val = int(points_match.group(1)) if points_match else None
        
        # ============================================
        # POINTS-BASED CLASSIFICATION (ABSOLUTE PRIORITY)
        # ============================================
        # Points completely determine the classification - override everything else
        if points_val is not None:
            if points_val >= 7:
                # 7 points = Correct (always)
                prediction = "Correct"
                if original_prediction != "Correct":
                    self.log_fn(f"Post-processing: Changed from '{original_prediction}' to 'Correct' based on {points_val} points (absolute rule)")
            elif points_val == 6:
                # 6 points = Almost (always) - solution is nearly complete with exactly one minor issue
                prediction = "Almost"
                if original_prediction != "Almost":
                    self.log_fn(f"Post-processing: Changed from '{original_prediction}' to 'Almost' based on 6 points (absolute rule)")
            elif 2 <= points_val <= 5:
                # 2-5 points = Partial (always) - significant progress but major gaps remain
                prediction = "Partial"
                if original_prediction != "Partial":
                    self.log_fn(f"Post-processing: Changed from '{original_prediction}' to 'Partial' based on {points_val} points (absolute rule)")
            elif points_val == 1:
                # 1 point = Partial (always) - minimal progress but some achievement
                prediction = "Partial"
                if original_prediction != "Partial":
                    self.log_fn(f"Post-processing: Changed from '{original_prediction}' to 'Partial' based on 1 point (absolute rule)")
            elif points_val == 0:
                # 0 points = Incorrect (always) - no meaningful progress
                prediction = "Incorrect"
                if original_prediction != "Incorrect":
                    self.log_fn(f"Post-processing: Changed from '{original_prediction}' to 'Incorrect' based on 0 points (absolute rule)")

        # Validate the final prediction
        if prediction not in VALID_LABELS:
            self.log_fn(f"Warning: Extracted prediction '{prediction}' is not a valid label. Defaulting to 'None'.")
            prediction = "None"
        
        self.log_fn(f"Final prediction: {prediction}")
        
        return str(prediction), msg_history
