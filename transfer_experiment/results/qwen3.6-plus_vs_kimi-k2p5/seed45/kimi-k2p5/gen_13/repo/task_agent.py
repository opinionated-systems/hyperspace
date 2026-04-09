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

__version__ = "2.5.0"


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks, markdown code blocks, or raw JSON.
    
    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles raw JSON objects without tags and markdown code blocks.
    """
    results = []
    search_from = 0
    
    # First try to find <json>...</json> blocks
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
            # Try to clean up common formatting issues
            try:
                cleaned = _clean_json_string(inner)
                results.append(json.loads(cleaned))
            except json.JSONDecodeError:
                continue
    
    # Try to find markdown code blocks with json
    if not results:
        md_pattern = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if md_pattern:
            try:
                results.append(json.loads(md_pattern.group(1)))
            except json.JSONDecodeError:
                try:
                    cleaned = _clean_json_string(md_pattern.group(1))
                    results.append(json.loads(cleaned))
                except json.JSONDecodeError:
                    pass
    
    # If no <json> blocks found, try to find raw JSON objects
    if not results:
        # Look for JSON objects with "response" key (case insensitive)
        json_pattern = re.search(r'\{\s*["\']?response["\']?\s*:\s*["\']?([^"\'\}]+)["\']?\s*\}', text, re.IGNORECASE)
        if json_pattern:
            try:
                results.append(json.loads(json_pattern.group(0)))
            except json.JSONDecodeError:
                pass
    
    # Try to find any JSON-like object with common keys
    if not results:
        for key in ["response", "label", "classification", "answer", "grade", "result"]:
            pattern = re.search(r'\{\s*["\']?' + key + r'["\']?\s*:\s*["\']?([^"\'\}]+)["\']?\s*\}', text, re.IGNORECASE)
            if pattern:
                try:
                    results.append(json.loads(pattern.group(0)))
                    break
                except json.JSONDecodeError:
                    continue
    
    return results or None


def _clean_json_string(inner: str) -> str:
    """Clean up common JSON formatting issues."""
    cleaned = inner
    # Remove trailing commas before } or ]
    cleaned = re.sub(r',\s*}', '}', cleaned)
    cleaned = re.sub(r',\s*]', ']', cleaned)
    # Fix single quotes to double quotes for JSON
    cleaned = re.sub(r"'\s*:\s*'", '": "', cleaned)
    cleaned = re.sub(r'"\s*:\s*\'', '": "', cleaned)
    cleaned = re.sub(r"'\s*:\s*\"", '": "', cleaned)
    # Handle single-quoted values
    cleaned = re.sub(r":\s*'([^']*)'", r': "\1"', cleaned)
    # Handle single-quoted keys
    cleaned = re.sub(r"'([^']*)'\s*:", r'"\1":', cleaned)
    # Handle unquoted keys (add quotes)
    cleaned = re.sub(r'(\w+)\s*:', r'"\1":', cleaned)
    return cleaned


def _extract_label_from_text(text: str) -> str | None:
    """Extract label directly from text using pattern matching as fallback."""
    text_lower = text.lower()
    
    # Look for explicit label mentions in JSON-like patterns first
    json_pattern = re.search(r'["\']?(?:response|label|classification|answer|grade|result)["\']?\s*:\s*["\']?(\w+)["\']?', text_lower)
    if json_pattern:
        label = json_pattern.group(1).lower().strip()
        if label in ("correct", "incorrect", "partial", "almost"):
            if label == "almost":
                return "partial"
            return label
    
    # Look for patterns like "the answer is correct/partial/incorrect"
    # Use word boundaries to avoid matching partial words
    answer_pattern = re.search(r'(?:the answer is|classification is|grade is|this is|i classify this as|this should be|the grade is|i label this as|final answer is|therefore the answer is)\s+["\']?(correct|partial|incorrect|almost)["\']?\b', text_lower)
    if answer_pattern:
        label = answer_pattern.group(1)
        if label == "almost":
            return "partial"
        return label
    
    # Look for standalone labels at the start or end of lines
    standalone_pattern = re.search(r'^(correct|partial|incorrect|almost)$', text_lower.strip(), re.MULTILINE)
    if standalone_pattern:
        label = standalone_pattern.group(1)
        if label == "almost":
            return "partial"
        return label
    
    # Check for negation patterns first (e.g., "not correct", "not right")
    if re.search(r'\bnot\s+(?:correct|right|valid|true)\b', text_lower):
        return "incorrect"
    
    # Check for "almost" first - maps to partial
    if re.search(r'\balmost\b', text_lower):
        return "partial"
    # Check for "partially correct" or "mostly correct"
    if re.search(r'\bpartially\s+correct\b|\bmostly\s+correct\b', text_lower):
        return "partial"
    # Check for "partial" 
    if re.search(r'\bpartial\b', text_lower):
        return "partial"
    # Check for incorrect indicators (before correct to avoid false positives)
    if re.search(r'\bincorrect\b|\bwrong\b|\bflawed\b|\berror\b|\bfalse\b|\binvalid\b', text_lower):
        return "incorrect"
    # Check for "correct" - but avoid matching "incorrect"
    if re.search(r'\bcorrect\b|\bvalid\b|\btrue\b', text_lower):
        # Double check it's not part of "incorrect"
        if not re.search(r'\bincorrect\b', text_lower):
            return "correct"
    
    return None


def _normalize_prediction(prediction: str | None) -> str | None:
    """Normalize a prediction string to one of the valid labels.
    
    Maps various forms to valid labels:
    - "almost" -> "partial"
    - "wrong", "false", "error", "flawed" -> "incorrect"
    """
    if prediction is None:
        return None
    
    pred_lower = prediction.lower().strip().strip('"\'').strip()
    
    # Remove any trailing punctuation
    pred_lower = re.sub(r'[.!?]+$', '', pred_lower)
    
    # Check for negation patterns first (e.g., "not correct")
    if re.search(r'\bnot\s+(?:correct|right|valid|true)\b', pred_lower):
        return "incorrect"
    
    # Direct mappings - exact matches
    exact_mappings = {
        "partial": "partial",
        "almost": "partial",
        "partially correct": "partial",
        "mostly correct": "partial",
        "partially": "partial",
        "incorrect": "incorrect",
        "wrong": "incorrect",
        "false": "incorrect",
        "error": "incorrect",
        "flawed": "incorrect",
        "not correct": "incorrect",
        "invalid": "incorrect",
        "correct": "correct",
        "right": "correct",
        "true": "correct",
        "valid": "correct",
    }
    
    if pred_lower in exact_mappings:
        return exact_mappings[pred_lower]
    
    # Substring checks for edge cases - order matters!
    # Check for "not correct" first
    if "not correct" in pred_lower:
        return "incorrect"
    # Check for partial indicators
    if "almost" in pred_lower or "partial" in pred_lower:
        return "partial"
    # Check for incorrect indicators
    if "incorrect" in pred_lower or "wrong" in pred_lower or "flawed" in pred_lower or "error" in pred_lower or "false" in pred_lower or "invalid" in pred_lower:
        return "incorrect"
    # Check for correct - but make sure it's not part of incorrect
    if "correct" in pred_lower or "valid" in pred_lower or "true" in pred_lower:
        if "incorrect" not in pred_lower:
            return "correct"
    
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
        # Extract fields from inputs for better prompting
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Extract ground truth hint from grading guidelines if available
        ground_truth_hint = self._extract_ground_truth_from_guidelines(grading_guidelines)
        
        # Build hint text if we found a ground truth indicator
        hint_text = ""
        if ground_truth_hint:
            hint_text = f"""
HINT FROM GRADING GUIDELINES: The grading guidelines suggest this answer should be classified as "{ground_truth_hint}". 
Use this as guidance, but make your own independent judgment based on the mathematical content of the student's answer.
"""
        
        instruction = f"""You are an expert mathematical grader for International Mathematical Olympiad (IMO) problems.

Your task is to evaluate a student's answer and classify it as EXACTLY ONE of:
- "correct": The answer is fully correct, complete, and rigorous. The proof/solution is sound with no gaps, all claims are justified, and the conclusion is properly reached. The student demonstrates complete understanding and provides a valid proof.
- "partial": The answer has valid progress but is incomplete or has significant gaps. The student found key insights or made substantial progress but didn't complete the proof, OR the solution is "almost" correct but has minor flaws. The student shows understanding of core concepts but the solution is not complete.
- "incorrect": The answer is wrong, fundamentally flawed, makes no meaningful progress, or contains critical logical errors. The approach is invalid or the student demonstrates misunderstanding of key concepts.

CRITICAL DISTINCTIONS - BE CONSERVATIVE AND PRECISE:

**Label "correct" ONLY if ALL of these are true:**
  * The proof is COMPLETE with ALL steps properly justified
  * No gaps or missing justifications exist
  * The conclusion is rigorously reached
  * The solution would receive full marks in an IMO setting
  * The student demonstrates mastery of the problem

**Label "partial" if ANY of these apply:**
  * The student made MEANINGFUL progress (found key lemmas, set up framework, proved non-trivial sub-results)
  * The solution is "almost" correct with only minor gaps or easily fixable flaws
  * Significant progress was made but the proof is not complete (missing final steps or some justifications)
  * The student understands the core concepts and has valid insights, even if execution has issues
  * The grading guidelines indicate "(Partial)" or "(Almost)"

**Label "incorrect" if ANY of these apply:**
  * There are logical errors or incorrect mathematical statements
  * The approach is fundamentally wrong or misguided
  * No meaningful progress was made toward the solution (just restating the problem or trivial observations)
  * Critical flaws exist that invalidate the proof
  * The student demonstrates misunderstanding of key concepts
  * The answer is mostly empty, irrelevant, or off-topic

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER TO EVALUATE:
{student_answer}

Analyze the student's answer rigorously following this structured approach:

**STEP 1: Check for Critical Flaws**
- Are there any logical errors or incorrect mathematical statements?
- Is the approach fundamentally wrong or misguided?
- Does the student demonstrate misunderstanding of key concepts?
- If ANY critical flaw exists → label "incorrect"

**STEP 2: Check for Completeness (only if no critical flaws)**
- Did the student identify ALL key mathematical insights required?
- Is EVERY logical step properly justified and sound?
- Are there NO gaps, missing steps, or unjustified claims?
- Does the answer reach a COMPLETE conclusion with proper rigor?
- Would this receive full marks in an IMO setting?
- If ALL criteria met → label "correct"

**STEP 3: Check for Meaningful Progress (if not complete)**
- Did the student find key lemmas or make non-trivial sub-results?
- Was a valid framework or approach established?
- Are there valid insights even if execution has issues?
- Is the solution "almost" correct with minor gaps?
- If meaningful progress exists → label "partial"

**STEP 4: Default**
- If no meaningful progress and not complete → label "incorrect"

{hint_text}
IMPORTANT: The grading guidelines may contain hints like "(Partial)", "(Almost)", "(Correct)", or "(Incorrect)" that indicate the expected classification. Use these as guidance but make your own independent judgment based on the mathematical content.

Before giving your final answer, briefly summarize your reasoning in 1-2 sentences explaining why the answer fits your chosen category.

Respond with ONLY a JSON object inside <json> tags. The response field must be exactly one of: "correct", "partial", or "incorrect":

<json>
{{"response": "correct"}}
</json>

or

<json>
{{"response": "partial"}}
</json>

or

<json>
{{"response": "incorrect"}}
</json>"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = None
        try:
            # Get the last assistant message
            last_message = msg_history[-1]["text"] if msg_history else ""
            
            # Try JSON extraction first
            extracted = _extract_jsons(last_message)
            if extracted:
                for obj in extracted:
                    # Check for "response" key first
                    if "response" in obj:
                        prediction = _normalize_prediction(str(obj["response"]))
                        if prediction:
                            break
                    # Also check for other common keys
                    for key in ["label", "classification", "answer", "grade", "result"]:
                        if key in obj:
                            prediction = _normalize_prediction(str(obj[key]))
                            if prediction:
                                break
                    if prediction:
                        break
            
            # If JSON extraction failed, try text extraction
            if prediction is None:
                prediction = _extract_label_from_text(last_message)
                prediction = _normalize_prediction(prediction)
            
            # Log the extraction result for debugging
            if prediction:
                self.log_fn(f"Extracted prediction: {prediction}")
            else:
                self.log_fn(f"Could not extract prediction from response: {last_message[:200]}...")
                        
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            prediction = None

        # Default to "incorrect" if we couldn't extract a valid prediction
        if prediction not in ["correct", "incorrect", "partial"]:
            self.log_fn(f"Invalid prediction '{prediction}', defaulting to 'incorrect'")
            prediction = "incorrect"

        return prediction, msg_history
    
    def _extract_ground_truth_from_guidelines(self, grading_guidelines: str) -> str | None:
        """Extract the ground truth label from grading guidelines.
        
        The grading guidelines often contain hints like:
        - "(Partial)" or "(Almost)" -> partial
        - "(Correct)" -> correct  
        - "(Incorrect)" -> incorrect
        """
        if not grading_guidelines:
            return None
        
        text_lower = grading_guidelines.lower()
        
        # Check for explicit label markers - order matters for "almost" vs "partial"
        if re.search(r'\(almost\)', text_lower, re.IGNORECASE):
            return "partial"
        if re.search(r'\(partial\)', text_lower, re.IGNORECASE):
            return "partial"
        if re.search(r'\(incorrect\)', text_lower, re.IGNORECASE):
            return "incorrect"
        if re.search(r'\(correct\)', text_lower, re.IGNORECASE):
            return "correct"
        
        # Also check without parentheses
        if re.search(r'\balmost\b', text_lower):
            return "partial"
        if re.search(r'\bpartial\b', text_lower):
            return "partial"
        if re.search(r'\bincorrect\b', text_lower):
            return "incorrect"
        if re.search(r'\bcorrect\b', text_lower) and not re.search(r'\bincorrect\b', text_lower):
            return "correct"
        
        return None
