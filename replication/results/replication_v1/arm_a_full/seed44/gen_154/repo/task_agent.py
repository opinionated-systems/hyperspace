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
    Also handles markdown code blocks as a fallback.
    Includes additional heuristics for malformed JSON and nested structures.
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
        
        # Try multiple parsing strategies
        parsed = _try_parse_json(inner)
        if parsed:
            results.append(parsed)
            continue
        
        # Try to extract just the response field if full JSON fails
        try:
            response_match = re.search(r'"response"\s*:\s*"([^"]*)"', inner)
            if response_match:
                results.append({"response": response_match.group(1)})
                continue
        except Exception:
            pass
        
        # Try to find any key-value pair that looks like a score
        try:
            score_match = re.search(r'"(\w+)"\s*:\s*"(\d+)"', inner)
            if score_match:
                results.append({score_match.group(1): score_match.group(2)})
                continue
        except Exception:
            pass
    
    # If no <json> blocks found, try markdown code blocks
    if not results:
        # Look for ```json ... ``` blocks
        md_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        for match in re.finditer(md_pattern, text, re.DOTALL):
            inner = match.group(1).strip()
            parsed = _try_parse_json(inner)
            if parsed:
                results.append(parsed)
                continue
            
            # Try to find a JSON object within the text
            try:
                json_start = inner.find('{')
                json_end = inner.rfind('}')
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    parsed = _try_parse_json(inner[json_start:json_end+1])
                    if parsed:
                        results.append(parsed)
            except Exception:
                continue
    
    # Final fallback: try to find any JSON-like structure in the text
    if not results:
        try:
            # Look for patterns like {"response": "7"} or {"score": "5"}
            json_pattern = r'\{\s*"\w+"\s*:\s*"[^"]*"\s*\}'
            for match in re.finditer(json_pattern, text):
                parsed = _try_parse_json(match.group())
                if parsed:
                    results.append(parsed)
        except Exception:
            pass
    
    # Ultra fallback: try to find just a number in the last line
    if not results:
        try:
            lines = text.strip().split('\n')
            for line in reversed(lines):
                line = line.strip()
                # Look for standalone numbers
                num_match = re.search(r'^\s*(\d+(?:\.\d+)?)\s*$', line)
                if num_match:
                    results.append({"response": num_match.group(1)})
                    break
                # Look for "Score: X" or similar patterns
                score_line_match = re.search(r'(?:score|grade|mark)[\s:]*(\d+(?:\.\d+)?)', line, re.IGNORECASE)
                if score_line_match:
                    results.append({"response": score_line_match.group(1)})
                    break
        except Exception:
            pass
    
    return results or None


def _try_parse_json(text: str) -> dict | None:
    """Try to parse JSON with multiple fallback strategies.
    
    Returns the parsed dict or None if all strategies fail.
    """
    text = text.strip()
    
    # Strategy 1: Direct parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Fix common JSON issues and retry
    try:
        # Fix trailing commas
        fixed = re.sub(r',\s*}', '}', text)
        fixed = re.sub(r',\s*]', ']', fixed)
        # Fix single quotes (convert to double)
        fixed = fixed.replace("'", '"')
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Extract just the outermost JSON object
    try:
        # Find the first { and last }
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end+1])
    except json.JSONDecodeError:
        pass
    
    # Strategy 4: Handle unquoted keys (common LLM mistake)
    try:
        # Add quotes around unquoted keys
        fixed = re.sub(r'(\w+)(\s*:)', r'"\1"\2', text)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Strategy 5: Handle newlines in strings
    try:
        # Replace literal newlines with escaped newlines within string values
        fixed = re.sub(r'("[^"]*?)\n([^"]*?")', r'\1\\n\2', text)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    return None


# Few-shot examples for IMO grading
_FEW_SHOT_EXAMPLES = """
Example 1:
Problem: Find all positive integers n such that n^2 + 3n + 2 is divisible by 4.
Solution: We can factor n^2 + 3n + 2 = (n+1)(n+2). For this to be divisible by 4, either (n+1) or (n+2) must be even, and at least one must be divisible by 2. Since n+1 and n+2 are consecutive integers, one is even. For the product to be divisible by 4, we need the even number to be divisible by 4, or both to be even (impossible for consecutive integers). So we need n+1 ≡ 0 (mod 4) or n+2 ≡ 0 (mod 4), meaning n ≡ 3 (mod 4) or n ≡ 2 (mod 4).
Grading Guidelines: Award 7 points for correct answer (n ≡ 2 or 3 mod 4). Award partial credit: 2 points for correct factorization, 2 points for recognizing need for divisibility by 4, 3 points for correct modular analysis.
Student Answer: n^2 + 3n + 2 = (n+1)(n+2). Since these are consecutive, one is even. For divisibility by 4, we need n ≡ 2 or 3 (mod 4).
Score: 7

Example 2:
Problem: Prove that for any positive integer n, n^3 + 2n is divisible by 3.
Solution: We use induction. Base case: n=1, 1^3 + 2(1) = 3, divisible by 3. Inductive step: assume k^3 + 2k divisible by 3. Then (k+1)^3 + 2(k+1) = k^3 + 3k^2 + 3k + 1 + 2k + 2 = k^3 + 2k + 3(k^2 + k + 1). By induction hypothesis, k^3 + 2k divisible by 3, and 3(k^2 + k + 1) clearly divisible by 3. Thus divisible by 3.
Grading Guidelines: Award 7 points for complete proof. Deduct 2 points for missing base case, 3 points for errors in inductive step, 2 points for lack of clarity.
Student Answer: By induction. Base case n=1 works. Assume true for k, then (k+1)^3 + 2(k+1) = k^3 + 3k^2 + 3k + 1 + 2k + 2 = (k^3 + 2k) + 3(k^2 + k + 1), both terms divisible by 3.
Score: 7

Example 3:
Problem: Let ABC be a triangle with AB = AC. Let D be the midpoint of BC. Prove that AD is perpendicular to BC.
Solution: Since AB = AC, triangle ABC is isosceles with apex A. D is the midpoint of BC. In an isosceles triangle, the median from the apex to the base is also the altitude. Therefore AD ⊥ BC. Alternatively, using coordinates: place D at origin, B at (-a, 0), C at (a, 0). Then A is at (0, h) for some h > 0. Vector AD = (0, -h), vector BC = (2a, 0). Their dot product is 0, so AD ⊥ BC.
Grading Guidelines: Award 7 points for complete proof. Award 4 points for stating the isosceles triangle property without full proof. Award 2 points for setting up coordinate system correctly. Award 5 points for correct coordinate proof with minor errors.
Student Answer: Triangle ABC is isosceles with AB = AC. The line from A to the midpoint D of BC is perpendicular to BC because in isosceles triangles, this line is the axis of symmetry.
Score: 4
"""


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _build_prompt(self, inputs: dict) -> str:
        """Build a structured prompt for the grading task."""
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        prompt = f"""You are an expert mathematics grader for International Mathematical Olympiad (IMO) problems. Your task is to evaluate student answers based on the official solution and grading guidelines.

Your responsibilities:
1. Carefully read the problem, official solution, and grading guidelines
2. Evaluate the student's answer against the official solution
3. Assign a score based on the grading guidelines (typically 0-7 points for IMO problems)
4. Provide your evaluation in the specified JSON format

{_FEW_SHOT_EXAMPLES}

Now evaluate the following:

DOMAIN: {domain}

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT ANSWER TO EVALUATE:
{student_answer}

Evaluation Instructions:
1. Carefully compare the student answer to the official solution
2. Identify what parts are correct, partially correct, or incorrect
3. Consider the grading guidelines for partial credit
4. Look for key insights, correct methods, and valid reasoning
5. Award partial credit for incomplete but correct approaches
6. Be fair but rigorous in your assessment

Scoring Guidelines:
- 7 points: Complete, correct solution with clear reasoning
- 5-6 points: Correct solution with minor gaps or presentation issues
- 3-4 points: Significant progress with some correct elements
- 1-2 points: Some relevant work but major errors or omissions
- 0 points: No valid mathematical work or completely incorrect

IMPORTANT: You must respond in the exact JSON format shown below. The response field must contain ONLY the numerical score as a string (e.g., "7", "5", "0", etc.).

<json>
{{
    "response": "<numerical_score>"
}}
</json>

Example valid responses:
- "response": "7" (for full marks)
- "response": "4" (for partial credit)
- "response": "0" (for no credit)"""

        return prompt

    def _validate_score(self, prediction: str) -> str:
        """Validate and normalize the score prediction.
        
        Handles various formats including:
        - Plain numbers: "7", "5"
        - Score prefixes: "Score: 7", "score: 5"
        - Fractions: "7/7", "5/7"
        - With units: "7 points", "5 pts"
        - Decimal scores: "6.5", "3.5" (rounded to nearest int)
        - Negative or out-of-range scores are clamped to valid range
        - Handles edge cases like "0/7", "full marks", "no credit"
        """
        if prediction is None:
            return "0"
        
        # Convert to string and strip whitespace
        prediction = str(prediction).strip()
        
        if not prediction:
            return "0"
        
        # Handle special text cases first (highest priority)
        prediction_lower = prediction.lower()
        if any(phrase in prediction_lower for phrase in ["full marks", "full credit", "perfect", "complete solution", "entirely correct"]):
            return "7"
        if any(phrase in prediction_lower for phrase in ["no credit", "no marks", "zero", "entirely incorrect", "completely wrong", "no valid work"]):
            return "0"
        
        # Try to extract a number from the prediction
        # First, try to find a fraction pattern like "5/7" and extract numerator
        fraction_match = re.search(r'(\d+(?:\.\d+)?)\s*/\s*\d+', prediction)
        if fraction_match:
            try:
                score_float = float(fraction_match.group(1))
                score_int = int(round(score_float))
                return str(max(0, min(7, score_int)))
            except ValueError:
                pass
        
        # Try to find a decimal or integer number with score prefix
        # Look for patterns like "Score: 6.5", "7 points", "score: 5", etc.
        number_match = re.search(r'(?:score|grade|mark)[:\s]*(\d+(?:\.\d+)?)', prediction, re.IGNORECASE)
        if number_match:
            try:
                score_float = float(number_match.group(1))
                score_int = int(round(score_float))
                # Clamp to valid IMO score range (0-7)
                return str(max(0, min(7, score_int)))
            except ValueError:
                pass
        
        # Try to find any decimal or integer number
        decimal_match = re.search(r'(\d+(?:\.\d+)?)', prediction)
        if decimal_match:
            try:
                score_float = float(decimal_match.group(1))
                score_int = int(round(score_float))
                # Clamp to valid IMO score range (0-7)
                return str(max(0, min(7, score_int)))
            except ValueError:
                pass
        
        # Fallback: try any digit sequence
        digit_match = re.search(r'\d+', prediction)
        if digit_match:
            try:
                score_int = int(digit_match.group())
                return str(max(0, min(7, score_int)))
            except ValueError:
                pass
        
        # If we can't extract a valid number, return "0"
        return "0"

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Validate required inputs
        required_keys = ["problem", "solution", "grading_guidelines", "student_answer"]
        for key in required_keys:
            if key not in inputs:
                self.log_fn(f"Warning: Missing required input key: {key}")
        
        instruction = self._build_prompt(inputs)

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"Error calling LLM: {e}")
            # Return default prediction with minimal history
            return "0", [{"role": "user", "text": instruction}, {"role": "assistant", "text": f"Error: {e}"}]

        # Extract prediction from JSON
        prediction = "0"
        extraction_method = "default"
        
        try:
            if msg_history and len(msg_history) > 0:
                last_message = msg_history[-1]
                last_text = last_message.get("text", "")
                
                if last_text:
                    extracted = _extract_jsons(last_text)
                    if extracted and "response" in extracted[-1]:
                        prediction = extracted[-1]["response"]
                        extraction_method = "json_response"
                    elif extracted:
                        # If no "response" key, try to use the last JSON object
                        last_extracted = extracted[-1]
                        if isinstance(last_extracted, dict):
                            # Try common score keys
                            for key in ["response", "score", "grade", "mark", "points"]:
                                if key in last_extracted:
                                    prediction = str(last_extracted[key])
                                    extraction_method = f"json_{key}"
                                    break
                            else:
                                prediction = str(last_extracted)
                                extraction_method = "json_fallback"
                        else:
                            prediction = str(last_extracted)
                            extraction_method = "json_string"
                    else:
                        # No JSON found, try direct number extraction
                        number_match = re.search(r'\d+', last_text)
                        if number_match:
                            prediction = number_match.group()
                            extraction_method = "regex_fallback"
                        
                        self.log_fn(f"Warning: No JSON found in response, using {extraction_method}")
                else:
                    self.log_fn("Warning: Empty response from LLM")
            else:
                self.log_fn("Warning: Empty message history from LLM")
                
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Try to extract any number from the response as fallback
            try:
                if msg_history and len(msg_history) > 0:
                    last_text = msg_history[-1].get("text", "")
                    number_match = re.search(r'\d+', last_text)
                    if number_match:
                        prediction = number_match.group()
                        extraction_method = "exception_fallback"
            except Exception:
                pass

        # Log extraction method for debugging
        self.log_fn(f"Score extraction method: {extraction_method}, raw prediction: {prediction}")

        # Validate and normalize the score
        validated_prediction = self._validate_score(prediction)
        
        if prediction != validated_prediction:
            self.log_fn(f"Score normalized: {prediction} -> {validated_prediction}")

        return str(validated_prediction), msg_history
