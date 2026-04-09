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


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.
    
    Enhanced version with better handling of nested structures and edge cases.
    Also supports markdown code blocks (```json...```) as a fallback.
    """
    results = []
    search_from = 0
    
    # First, try to find <json>...</json> blocks
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            break
        
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        # Try direct parsing first
        try:
            results.append(json.loads(inner))
            continue
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON objects using brace counting
        brace_count = 0
        json_start = -1
        found_in_block = []
        in_string = False
        escape_next = False
        
        for i, char in enumerate(inner):
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
                
            if char == '{':
                if brace_count == 0:
                    json_start = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and json_start != -1:
                    try:
                        obj = json.loads(inner[json_start:i+1])
                        found_in_block.append(obj)
                    except json.JSONDecodeError:
                        pass
                    json_start = -1
        
        if found_in_block:
            results.extend(found_in_block)
    
    # If no <json> blocks found, try markdown code blocks
    if not results:
        md_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        for match in re.finditer(md_pattern, text, re.DOTALL):
            try:
                content = match.group(1).strip()
                if content:
                    results.append(json.loads(content))
            except json.JSONDecodeError:
                pass
    
    return results if results else None


def _extract_any_json(text: str) -> list[dict] | None:
    """Fallback JSON extraction that looks for any JSON objects in text.
    
    Enhanced with proper string handling to avoid false positives.
    """
    results = []
    brace_count = 0
    start_idx = -1
    in_string = False
    escape_next = False
    
    for i, char in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if char == '\\':
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
            
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                try:
                    obj = json.loads(text[start_idx:i+1])
                    results.append(obj)
                except json.JSONDecodeError:
                    pass
                start_idx = -1
    
    return results if results else None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3  # Increased retries for better reliability

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract task components for better prompting
        domain = inputs.get("domain", "Unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematical grader for International Mathematical Olympiad (IMO) problems.

Your task is to evaluate a student's answer to a mathematics problem and provide a grade with detailed reasoning.

DOMAIN: {domain}

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

Think step by step and provide a thorough analysis:
1. PROBLEM ANALYSIS: Identify the key mathematical concepts, theorems, and techniques required
2. SOLUTION REVIEW: Analyze the official solution's approach, key steps, and expected answer format
3. STUDENT WORK ANALYSIS: 
   - Identify what approach the student took
   - Note any correct steps or valid insights
   - Identify errors, gaps, or misconceptions
4. GRADING CRITERIA CHECK:
   - Verify if the student met each criterion in the grading guidelines
   - Note partial credit for incomplete but valid reasoning
5. FINAL DETERMINATION: Assign grade based on completeness, correctness, and adherence to guidelines

Respond ONLY in the following JSON format. Do not include any text outside the JSON tags:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis covering all 5 steps above",
    "response": "The final grade/prediction (must be one of: 'Correct', 'Incorrect', or 'Partial')",
    "confidence": "High/Medium/Low - your confidence in this grading decision"
}}
</json>

Important guidelines:
- Be objective and consistent in your grading
- Award partial credit when the student shows valid reasoning even if the final answer is incorrect
- The response field MUST be exactly one of: 'Correct', 'Incorrect', or 'Partial'
- Ensure your JSON is valid and properly formatted"""

        # Retry loop for robustness
        for attempt in range(self.max_retries + 1):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                break
            except Exception as e:
                self.log_fn(f"LLM call failed (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries:
                    return "Error: LLM call failed", []
                import time
                time.sleep(2 ** attempt)

        # Extract prediction from JSON with fallback mechanisms
        prediction = "None"
        confidence = "Unknown"
        reasoning = ""
        
        try:
            last_message = msg_history[-1]["text"]
            
            # Try primary extraction method
            extracted = _extract_jsons(last_message)
            self.log_fn(f"Primary extraction found {len(extracted) if extracted else 0} JSON objects")
            
            # Fallback to generic JSON extraction if primary fails
            if extracted is None:
                extracted = _extract_any_json(last_message)
                self.log_fn(f"Fallback extraction found {len(extracted) if extracted else 0} JSON objects")
            
            if extracted:
                # Prefer response field, but accept other common field names
                last_json = extracted[-1]
                self.log_fn(f"Using JSON with keys: {list(last_json.keys())}")
                
                # Extract reasoning for better logging
                if "reasoning" in last_json:
                    reasoning = last_json["reasoning"][:200]  # Truncate for log
                    self.log_fn(f"Reasoning preview: {reasoning}...")
                
                # Priority order for prediction fields
                priority_fields = ["response", "grade", "answer", "result", "evaluation", "prediction", "score"]
                found = False
                for field in priority_fields:
                    if field in last_json:
                        prediction = last_json[field]
                        found = True
                        self.log_fn(f"Found prediction in field '{field}': {prediction}")
                        break
                
                if not found:
                    # If no known field, use the first string value found
                    for key, value in last_json.items():
                        if isinstance(value, str) and value.strip():
                            prediction = value
                            self.log_fn(f"Using first string value from key '{key}': {prediction}")
                            break
                        elif isinstance(value, (int, float)):
                            prediction = str(value)
                            self.log_fn(f"Using first numeric value from key '{key}': {prediction}")
                            break
                
                # Extract confidence if available
                if "confidence" in last_json:
                    confidence = last_json["confidence"]
                    self.log_fn(f"Confidence level: {confidence}")
            else:
                self.log_fn("No JSON objects found in response")
                # Try to extract any meaningful text as prediction
                lines = last_message.strip().split('\n')
                for line in reversed(lines):
                    line = line.strip()
                    if line and not line.startswith('<') and not line.startswith('{') and len(line) < 100:
                        prediction = line
                        self.log_fn(f"Using last non-JSON line as prediction: {prediction}")
                        break
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Validate and normalize prediction format
        prediction = self._normalize_prediction(str(prediction))
        self.log_fn(f"Final prediction: {prediction}")
        return prediction, msg_history

    def _normalize_prediction(self, prediction: str) -> str:
        """Normalize prediction to standard format.
        
        Maps various common variations to the three standard categories:
        'Correct', 'Incorrect', or 'Partial'.
        """
        prediction = prediction.strip()
        
        # Handle empty or None predictions
        if not prediction or prediction.lower() in ['none', 'null', 'nan', '']:
            return 'Incorrect'  # Default to Incorrect for missing predictions
        
        # Handle common variations for CORRECT
        correct_variations = [
            'correct', 'right', 'true', 'yes', '1', 'full', 'pass', 'passed',
            'solved', 'solution correct', 'answer correct', 'valid', 'true positive'
        ]
        
        # Handle common variations for INCORRECT  
        incorrect_variations = [
            'incorrect', 'wrong', 'false', 'no', '0', 'none', 'fail', 'failed',
            'error', 'invalid', 'not correct', 'not solved', 'false positive',
            'false negative', 'unsolved', 'incomplete'
        ]
        
        # Handle common variations for PARTIAL
        partial_variations = [
            'partial', 'partially correct', 'half', 'partial credit', 'partially',
            'incomplete but valid', 'some correct', 'partially solved', 'partial solution',
            'partially right', 'mostly correct', 'mostly right', 'partial pass'
        ]
        
        lower_pred = prediction.lower()
        
        # Check for exact matches first
        if lower_pred in correct_variations:
            return 'Correct'
        elif lower_pred in incorrect_variations:
            return 'Incorrect'
        elif lower_pred in partial_variations:
            return 'Partial'
        
        # Check for partial matches (contains)
        for var in correct_variations:
            if var in lower_pred:
                return 'Correct'
        for var in incorrect_variations:
            if var in lower_pred:
                return 'Incorrect'
        for var in partial_variations:
            if var in lower_pred:
                return 'Partial'
        
        # Handle numeric scores
        try:
            num_val = float(prediction)
            if num_val >= 0.8:
                return 'Correct'
            elif num_val >= 0.4:
                return 'Partial'
            else:
                return 'Incorrect'
        except ValueError:
            pass
        
        # Default: if prediction is short and capitalized, keep it
        if len(prediction) <= 20 and prediction[0].isupper():
            return prediction
        
        # Final fallback
        return 'Incorrect'
