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

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    
    Args:
        text: The text containing <json> tags to parse.
        
    Returns:
        A list of parsed JSON objects, or None if no valid JSON found.
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
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            continue
    return results or None


def _extract_json_flexible(text: str) -> list[dict] | None:
    """Extract JSON objects with multiple fallback strategies.
    
    Tries multiple patterns in order of reliability:
    1. <json>...</json> tags
    2. ```json code blocks
    3. Raw JSON objects at start/end of text
    4. JSON-like structures with relaxed parsing
    5. Repair common JSON syntax errors
    
    Args:
        text: The text to extract JSON from.
        
    Returns:
        A list of parsed JSON objects, or None if no valid JSON found.
    """
    if not text or not text.strip():
        return None
        
    results = []
    
    # Strategy 1: <json> tags (original)
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
            # Try to repair common JSON errors
            repaired = _repair_json(inner)
            if repaired:
                results.append(repaired)
            continue
    
    # Strategy 2: ```json code blocks
    if not results:
        pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                results.append(json.loads(match.strip()))
            except json.JSONDecodeError:
                repaired = _repair_json(match.strip())
                if repaired:
                    results.append(repaired)
                continue
    
    # Strategy 3: Look for JSON objects directly using brace matching
    if not results:
        # Find all potential JSON object starts
        start_indices = [m.start() for m in re.finditer(r'\{', text)]
        
        for start in start_indices:
            try:
                # Use brace counting to find the matching end
                brace_count = 0
                end = -1
                in_string = False
                escape_next = False
                
                for i, char in enumerate(text[start:]):
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
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end = start + i + 1
                                break
                
                if end > start:
                    obj_str = text[start:end]
                    try:
                        results.append(json.loads(obj_str))
                    except json.JSONDecodeError:
                        repaired = _repair_json(obj_str)
                        if repaired:
                            results.append(repaired)
            except (json.JSONDecodeError, ValueError):
                continue
    
    return results or None


def _repair_json(text: str) -> dict | None:
    """Attempt to repair common JSON syntax errors.
    
    Fixes:
    - Trailing commas before closing braces/brackets
    - Single quotes instead of double quotes
    - Unescaped newlines in strings
    - Missing closing braces/brackets
    - Comments in JSON (// and /* */)
    - Control characters in strings
    
    Args:
        text: The potentially malformed JSON string to repair.
        
    Returns:
        A parsed JSON dict if repair succeeds, None otherwise.
    """
    if not text or not text.strip():
        return None
    
    try:
        # First, try parsing as-is
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Remove comments (// and /* */)
    repaired = re.sub(r'//.*?$', '', text, flags=re.MULTILINE)
    repaired = re.sub(r'/\*.*?\*/', '', repaired, flags=re.DOTALL)
    
    # Remove trailing commas before } or ]
    repaired = re.sub(r',\s*([}\]])', r'\1', repaired)
    
    # Try to fix single quotes (simple cases only)
    # Replace single quotes around keys/values with double quotes
    repaired = re.sub(r"'([^']*?)':", r'"\1":', repaired)
    repaired = re.sub(r":\s*'([^']*?)'([,}\]])", r': "\1"\2', repaired)
    
    # Escape unescaped newlines in strings (but not already escaped ones)
    repaired = re.sub(r'(?<!\\)\n', r'\\n', repaired)
    
    # Escape unescaped tabs in strings
    repaired = re.sub(r'(?<!\\)\t', r'\\t', repaired)
    
    # Escape unescaped carriage returns
    repaired = re.sub(r'(?<!\\)\r', r'\\r', repaired)
    
    # Remove control characters (0x00-0x1F except allowed ones: \n, \t, \r)
    # These are invalid in JSON strings
    allowed_controls = {'\n', '\t', '\r'}
    cleaned = []
    for char in repaired:
        if ord(char) < 32 and char not in allowed_controls:
            continue  # Skip control characters
        cleaned.append(char)
    repaired = ''.join(cleaned)
    
    # Try to balance braces
    open_braces = repaired.count('{') - repaired.count('}')
    if open_braces > 0:
        repaired += '}' * open_braces
    elif open_braces < 0:
        # Remove extra closing braces from the end
        for _ in range(-open_braces):
            last_brace = repaired.rfind('}')
            if last_brace > repaired.rfind('{'):
                repaired = repaired[:last_brace] + repaired[last_brace+1:]
    
    open_brackets = repaired.count('[') - repaired.count(']')
    if open_brackets > 0:
        repaired += ']' * open_brackets
    elif open_brackets < 0:
        # Remove extra closing brackets from the end
        for _ in range(-open_brackets):
            last_bracket = repaired.rfind(']')
            if last_bracket > repaired.rfind('['):
                repaired = repaired[:last_bracket] + repaired[last_bracket+1:]
    
    # Try parsing the repaired version
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass
    
    # Last resort: try to extract just the first complete JSON object
    try:
        start = repaired.find('{')
        if start == -1:
            return None
        
        brace_count = 0
        for i, char in enumerate(repaired[start:]):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    # Found complete object
                    return json.loads(repaired[start:start+i+1])
        return None
    except Exception:
        return None


def _extract_json_robust(text: str) -> list[dict] | None:
    """Most robust JSON extraction with multiple fallback strategies.
    
    This is the primary extraction function that tries multiple
    strategies in order of reliability.
    
    Args:
        text: The text to extract JSON from.
        
    Returns:
        A list of parsed JSON objects, or None if no valid JSON found.
    """
    if not text or not text.strip():
        return None
    
    # Strategy 1: <json> tags (original format)
    results = _extract_jsons(text)
    if results:
        return results
    
    # Strategy 2: Flexible extraction with repair
    results = _extract_json_flexible(text)
    if results:
        return results
    
    # Strategy 3: Look for any JSON-like structure
    # Try to find content between outermost braces with proper string handling
    try:
        start = text.find('{')
        if start == -1:
            return None
            
        # Use proper brace counting with string awareness
        brace_count = 0
        end = -1
        in_string = False
        escape_next = False
        
        for i, char in enumerate(text[start:]):
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
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end = start + i + 1
                        break
        
        if end > start:
            candidate = text[start:end]
            # Try to repair and parse
            repaired = _repair_json(candidate)
            if repaired:
                return [repaired]
    except Exception:
        pass
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _build_prompt(self, inputs: dict) -> str:
        """Build a structured prompt for IMO grading with enhanced chain-of-thought reasoning.
        
        Improvements:
        - Better structured reasoning sections
        - Explicit handling of edge cases
        - Clearer partial credit guidelines
        - Improved JSON format instructions
        """
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Truncate very long inputs to prevent context overflow
        max_len = 8000
        problem = problem[:max_len] if len(problem) > max_len else problem
        solution = solution[:max_len] if len(solution) > max_len else solution
        student_answer = student_answer[:max_len] if len(student_answer) > max_len else student_answer
        
        # Handle missing grading guidelines gracefully
        guidelines_section = f"\n## Grading Guidelines\n{grading_guidelines}" if grading_guidelines else "\n## Grading Guidelines\nNo specific guidelines provided. Use standard mathematical rigor and completeness as criteria."
        
        prompt = f"""You are an expert {domain} grader evaluating student solutions to International Mathematical Olympiad (IMO) style competition problems.

Your task is to carefully analyze the student's answer and provide a rigorous evaluation according to the official solution and grading guidelines.

## Problem Statement
{problem}

## Official Solution
{solution}
{guidelines_section}

## Student's Answer
{student_answer}

## Instructions

Think through this step-by-step, documenting your reasoning clearly:

### Step 1: Problem Analysis
- What is the core question being asked?
- What are the key mathematical concepts, theorems, and techniques required?
- What would constitute a complete, rigorous solution?

### Step 2: Official Solution Analysis
- What is the canonical approach to solving this problem?
- What is the final answer (numerical, algebraic, or descriptive)?
- What are the critical proof steps or logical deductions that must be present?
- Are there alternative valid approaches?

### Step 3: Student Answer Analysis
- What approach did the student attempt?
- What is their stated or implied final answer?
- What work did they show? What did they omit?
- Did they use valid mathematical techniques?

### Step 4: Detailed Comparison
Evaluate the student's answer against the official solution:

**Correctness of Final Answer:**
- Is the student's final answer numerically/algebraically equivalent to the official solution?
- If different, is it a minor computational error or a fundamental misunderstanding?

**Validity of Reasoning:**
- Did the student demonstrate correct mathematical reasoning?
- Are there logical gaps, circular arguments, or unsupported claims?
- Did they apply theorems correctly with proper conditions?

**Completeness:**
- Does the solution cover all required cases or conditions?
- Are there missing steps that are essential to the proof?
- Did they justify each step with clear reasoning?

**Partial Credit Assessment (if applicable):**
- Which specific steps or insights were correct?
- What percentage of the solution is correct vs incorrect?
- Is the error minor (computation) or major (conceptual)?

### Step 5: Grade Assignment
Based on your thorough analysis, assign the appropriate grade.

## Grading Rubric

**CORRECT** - Use when:
- Final answer matches the official solution exactly
- All reasoning is mathematically sound and rigorous
- All steps are properly justified
- No logical gaps or errors present
- Complete solution with all required components

**INCORRECT** - Use when:
- Final answer is fundamentally wrong or doesn't match
- Critical conceptual errors in the approach
- Missing essential components that make the solution invalid
- Incorrect theorems or methods applied
- No meaningful progress toward the solution

**PARTIAL** - Use when:
- Some correct elements but incomplete solution
- Minor errors that don't invalidate the overall approach
- Correct approach but missing some justifications
- Partial progress with some valid insights
- Answer is close but has minor computational errors
- Some correct steps mixed with incorrect ones

## Response Format

You MUST respond in this exact JSON format:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis here. Be thorough - mention specific mathematical steps, theorems used, errors found, and your reasoning for the grade assigned. This should be a comprehensive evaluation.",
    "response": "correct" | "incorrect" | "partial"
}}
</json>

CRITICAL RULES - FOLLOW EXACTLY:
1. The "response" field MUST contain ONLY one of these three exact lowercase values: "correct", "incorrect", or "partial"
2. NO other text, explanations, or formatting allowed in the "response" field
3. Use "partial" when the student has some correct work but not a complete solution
4. Use "incorrect" when the answer is fundamentally wrong or missing critical components
5. Use "correct" ONLY when the solution is fully correct with proper justification
6. Ensure valid JSON syntax - check for proper quotes, commas, and braces
7. The reasoning field should be detailed but the response field must be exactly one word"""
        
        return prompt

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history with multiple fallback strategies.
        
        Enhanced for IMO grading: specifically handles correctness evaluation,
        numeric scores, and partial credit scenarios.
        
        Improvements:
        - Better handling of nested JSON structures
        - Improved regex patterns for field extraction
        - Enhanced score parsing with fraction support
        - Better handling of boolean values
        - Improved error handling and logging
        """
        if not msg_history:
            self.log_fn("Warning: Empty message history")
            return "None"
        
        last_text = msg_history[-1].get("text", "")
        if not last_text:
            self.log_fn("Warning: Last message has no text content")
            return "None"
        
        # Try robust extraction first (includes all strategies)
        try:
            extracted = _extract_json_robust(last_text)
            if extracted:
                # Prefer response field, fallback to other common fields
                last_obj = extracted[-1]
                
                # Check for structured grading response first - prioritize "response" field
                for key in ["response", "grade", "score", "evaluation", "answer", "result", "conclusion", "verdict"]:
                    if key in last_obj:
                        value = last_obj[key]
                        # Handle different value types
                        if isinstance(value, bool):
                            return "correct" if value else "incorrect"
                        elif isinstance(value, (str, int, float)):
                            return str(value)
                        elif isinstance(value, (list, dict)):
                            return json.dumps(value)
                
                # Check for correctness boolean
                if "correct" in last_obj:
                    correct_val = last_obj["correct"]
                    if isinstance(correct_val, bool):
                        return "correct" if correct_val else "incorrect"
                    elif isinstance(correct_val, str):
                        return correct_val.lower()
                    return str(correct_val)
                
                # Check for points/partial credit with better type handling
                if "points" in last_obj:
                    points_val = last_obj["points"]
                    if isinstance(points_val, (int, float)):
                        return f"points:{points_val}"
                    return f"points:{points_val}"
                
                # Check for fraction scores (e.g., "3/7")
                if "fraction" in last_obj or "ratio" in last_obj:
                    frac_val = last_obj.get("fraction") or last_obj.get("ratio")
                    if frac_val:
                        return f"score:{frac_val}"
                
                # If no known field, return the whole object as string
                return str(last_obj)
        except Exception as e:
            self.log_fn(f"Flexible extraction failed: {e}")
        
        # Fallback 1: try to find any JSON-like structure with regex
        try:
            # Look for patterns like "response": "..." or 'response': '...'
            # Prioritize response field, support both single and double quotes
            patterns = [
                (r'["\']response["\']\s*:\s*["\']([^"\']+)["\']', 1),
                (r'["\']grade["\']\s*:\s*["\']([^"\']+)["\']', 1),
                (r'["\']score["\']\s*:\s*["\']?([^"\'},\s]+)["\']?', 1),
                (r'["\']evaluation["\']\s*:\s*["\']([^"\']+)["\']', 1),
                (r'["\']answer["\']\s*:\s*["\']([^"\']+)["\']', 1),
                (r'["\']correct["\']\s*:\s*(true|false)', 1),
                (r'["\']verdict["\']\s*:\s*["\']([^"\']+)["\']', 1),
            ]
            for pattern, group in patterns:
                match = re.search(pattern, last_text, re.IGNORECASE)
                if match:
                    return match.group(group).lower()
        except Exception as e:
            self.log_fn(f"Regex extraction failed: {e}")
        
        # Fallback 2: Look for standalone correct/incorrect/partial in the text
        try:
            text_lower = last_text.lower()
            
            # Check for exact matches first
            for verdict in ["correct", "incorrect", "partial"]:
                # Look for the word as a standalone token
                pattern = rf'\b{verdict}\b'
                if re.search(pattern, text_lower):
                    return verdict
        except Exception as e:
            self.log_fn(f"Standalone word extraction failed: {e}")
        
        # Fallback 3: Enhanced text-based extraction for grading scenarios
        text_lower = last_text.lower()
        
        # Check for numeric scores with improved patterns
        score_patterns = [
            r'(?:score|grade|points?)\s*[:=]\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)',  # "score: 3/7"
            r'(?:score|grade|points?)\s*[:=]\s*(\d+(?:\.\d+)?)',  # "score: 7"
            r'(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*(?:points?)?',  # "3/7" or "3/7 points"
            r'(?:awarded|given|assigned)\s+(\d+(?:\.\d+)?)\s*(?:out of|/\s*)?(\d+(?:\.\d+)?)?\s*(?:points?)?',
            r'(?:received|earned|got)\s+(\d+(?:\.\d+)?)\s*(?:out of|/\s*)?(\d+(?:\.\d+)?)?\s*(?:points?)?',
        ]
        for pattern in score_patterns:
            match = re.search(pattern, text_lower)
            if match:
                if len(match.groups()) >= 2 and match.group(2):
                    return f"score:{match.group(1)}/{match.group(2)}"
                return f"score:{match.group(1)}"
        
        # Check for fraction patterns like "3 out of 7"
        fraction_match = re.search(r'(\d+)\s+out\s+of\s+(\d+)', text_lower)
        if fraction_match:
            return f"score:{fraction_match.group(1)}/{fraction_match.group(2)}"
        
        # Check for correctness indicators - check for "incorrect" first to avoid false positives
        # Use word boundaries to avoid matching substrings
        incorrect_patterns = [
            r'\bincorrect\b',
            r'\bnot correct\b',
            r'\bwrong\b',
            r'\bnot right\b',
            r'\bfalse\b',
            r'\binvalid\b',
            r'\brejected\b',
            r'\bnot valid\b',
            r'\berroneous\b',
        ]
        for pattern in incorrect_patterns:
            if re.search(pattern, text_lower):
                return "incorrect"
        
        # Check for correct (after checking for "not correct")
        correct_patterns = [
            r'\bcorrect\b',
            r'\bright\b',
            r'\btrue\b',
            r'\bvalid\b',
            r'\baccepted\b',
            r'\bcomplete\b',
            r'\bfully correct\b',
            r'\bentirely correct\b',
        ]
        for pattern in correct_patterns:
            if re.search(pattern, text_lower):
                return "correct"
        
        # Check for partial credit indicators
        partial_patterns = [
            r'\bpartial\b',
            r'\bpartially\b',
            r'\bincomplete\b',
            r'\bsome credit\b',
            r'\bhalf\b',
            r'\bpartial credit\b',
            r'\bpartially correct\b',
            r'\bmostly correct\b',
            r'\bsomewhat correct\b',
        ]
        for pattern in partial_patterns:
            if re.search(pattern, text_lower):
                return "partial"
        
        # Fallback 3: return first 200 chars of text as prediction
        if last_text.strip():
            return last_text.strip()[:200]
        
        return "None"

    def _normalize_prediction(self, prediction: str) -> str:
        """Normalize prediction to standard format for evaluation.
        
        Converts various forms of correct/incorrect/partial to standardized values.
        Uses a priority-based approach to handle ambiguous cases.
        
        Improvements:
        - Better handling of compound phrases
        - Improved numeric score interpretation
        - Better handling of negations
        - Support for more grading terminology
        """
        if not prediction or prediction == "None":
            return "None"
        
        pred_lower = prediction.lower().strip()
        
        # Remove common punctuation and whitespace variations
        pred_clean = pred_lower.strip('"\'.,;: ')
        
        # Exact matches first (highest priority)
        exact_matches = {
            "correct": "correct",
            "incorrect": "incorrect", 
            "partial": "partial",
            "true": "correct",
            "false": "incorrect",
            "right": "correct",
            "wrong": "incorrect",
            "valid": "correct",
            "invalid": "incorrect",
            "accepted": "correct",
            "rejected": "incorrect",
            "incomplete": "partial",
            "partially correct": "partial",
            "partially incorrect": "partial",
            "mostly correct": "partial",
            "mostly incorrect": "partial",
            "some correct": "partial",
            "full marks": "correct",
            "no marks": "incorrect",
            "zero": "incorrect",
            "full credit": "correct",
            "no credit": "incorrect",
        }
        
        if pred_clean in exact_matches:
            return exact_matches[pred_clean]
        
        # Handle score format (e.g., "score:7", "7/7", "points:5")
        if pred_lower.startswith(("score:", "points:", "grade:")):
            return prediction
        
        # Check for numeric patterns that might indicate scoring
        numeric_match = re.search(r'(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)', pred_lower)
        if numeric_match:
            try:
                num, denom = float(numeric_match.group(1)), float(numeric_match.group(2))
                if denom == 0:
                    return prediction  # Avoid division by zero
                ratio = num / denom
                if ratio >= 0.95:  # 95% or higher = correct
                    return "correct"
                elif ratio <= 0.05:  # 5% or lower = incorrect
                    return "incorrect"
                else:
                    return "partial"
            except (ValueError, ZeroDivisionError):
                pass
        
        # Check for standalone numbers (assume out of some max like 7 for IMO)
        standalone_num = re.search(r'^\s*(\d+(?:\.\d+)?)\s*$', pred_clean)
        if standalone_num:
            try:
                num = float(standalone_num.group(1))
                # Assume IMO-style scoring where 7 is max
                if num >= 6.5:  # Close to max
                    return "correct"
                elif num <= 0.5:  # Close to zero
                    return "incorrect"
                else:
                    return "partial"
            except ValueError:
                pass
        
        # Check for percentage scores
        percent_match = re.search(r'(\d+(?:\.\d+)?)\s*%', pred_lower)
        if percent_match:
            try:
                percent = float(percent_match.group(1))
                if percent >= 95:
                    return "correct"
                elif percent <= 5:
                    return "incorrect"
                else:
                    return "partial"
            except ValueError:
                pass
        
        # Priority-based keyword detection with word boundaries
        # Check for negations first (highest priority for incorrect)
        negation_patterns = [
            r'\bnot correct\b',
            r'\bnot right\b',
            r'\bnot valid\b',
            r'\bnot accepted\b',
            r'\bnot true\b',
            r'\bisn\'t correct\b',
            r'\bisn\'t right\b',
            r'\bdoes not match\b',
            r'\bdoesn\'t match\b',
        ]
        for pattern in negation_patterns:
            if re.search(pattern, pred_lower):
                return "incorrect"
        
        # Check for incorrect patterns
        incorrect_patterns = [
            r'\bincorrect\b',
            r'\bwrong\b',
            r'\bfalse\b',
            r'\binvalid\b',
            r'\brejected\b',
            r'\berror\b',
            r'\bfailed\b',
            r'\bno\b',  # standalone "no" in grading context
            r'\bnone\b',
            r'\bzero\b',
            r'\b0\b',
        ]
        for pattern in incorrect_patterns:
            if re.search(pattern, pred_lower):
                return "incorrect"
        
        # Check for partial patterns
        partial_patterns = [
            r'\bpartial\b',
            r'\bpartially\b',
            r'\bincomplete\b',
            r'\bsome credit\b',
            r'\bhalf\b',
            r'\bpartial credit\b',
            r'\bpartially correct\b',
            r'\bmostly\b',
            r'\bsomewhat\b',
            r'\bfair\b',
            r'\bmoderate\b',
            r'\blimited\b',
        ]
        for pattern in partial_patterns:
            if re.search(pattern, pred_lower):
                return "partial"
        
        # Check for correct patterns (after checking for negations)
        correct_patterns = [
            r'\bcorrect\b',
            r'\bright\b',
            r'\btrue\b',
            r'\bvalid\b',
            r'\baccepted\b',
            r'\bfull credit\b',
            r'\bcomplete\b',
            r'\bfully correct\b',
            r'\bentirely correct\b',
            r'\bperfect\b',
            r'\bexcellent\b',
            r'\byes\b',  # standalone "yes" in grading context
            r'\b1\b',  # binary correct indicator
        ]
        for pattern in correct_patterns:
            if re.search(pattern, pred_lower):
                return "correct"
        
        # Return original if no normalization applied
        return prediction

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Validate required inputs
        required_keys = ["problem", "solution", "student_answer"]
        missing_keys = [k for k in required_keys if k not in inputs or not inputs[k]]
        if missing_keys:
            error_msg = f"Error: Missing required inputs: {missing_keys}"
            self.log_fn(error_msg)
            return error_msg, [{"role": "assistant", "text": error_msg}]
        
        # Validate input types
        for key in required_keys:
            if not isinstance(inputs.get(key), str):
                error_msg = f"Error: Input '{key}' must be a string"
                self.log_fn(error_msg)
                return error_msg, [{"role": "assistant", "text": error_msg}]
        
        instruction = self._build_prompt(inputs)
        
        # Log the problem being processed (truncated for privacy)
        problem_preview = inputs.get("problem", "")[:100]
        self.log_fn(f"Processing problem: {problem_preview}...")

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            error_msg = f"Error: LLM call failed: {e}"
            self.log_fn(error_msg)
            return error_msg, [{"role": "assistant", "text": error_msg}]

        # Extract prediction from JSON
        raw_prediction = self._extract_prediction(msg_history)
        
        # Normalize prediction to standard format
        prediction = self._normalize_prediction(raw_prediction)
        
        if prediction == "None":
            self.log_fn(f"Warning: Could not extract valid prediction from response")
            # Log the raw response for debugging
            if msg_history:
                raw_text = msg_history[-1].get('text', '')
                self.log_fn(f"Raw response preview: {raw_text[:500]}...")
        else:
            # Log successful extraction
            preview = prediction[:100] if len(prediction) > 100 else prediction
            self.log_fn(f"Extracted prediction: {preview}")

        return str(prediction), msg_history
