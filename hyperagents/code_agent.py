# -*- coding: utf-8 -*-
"""
CodeHyperagent - hyperagent whose editable program is Python code.

Both the task code (solve() function) AND the meta code (improve() function)
are editable Python programs that get rewritten each generation.

This is the key property of DGM-H: metacognitive self-modification.
The meta agent can rewrite not just the task solver, but also the procedure
by which it generates future improvements.

llm_call is injected as a global into both exec namespaces.
"""

from __future__ import annotations

import inspect
import logging
import re
import uuid
from dataclasses import dataclass, field

from hyperagents.strategy import Strategy, StrategyStore
from hyperagents.tasks import Task, TaskResult

logger = logging.getLogger(__name__)

LLMCallable = __import__("hyperagents.agent", fromlist=["LLMCallable"]).LLMCallable


def parse_improve_response(response: str) -> tuple[str | None, str | None, str, str]:
    """Parse an improve() LLM response into (task_code, meta_code, name, description).

    Supports two formats:
    - XML tags: <NEW_CODE>...</NEW_CODE> and <NEW_META_CODE>...</NEW_META_CODE>
      (preferred — unambiguous, can't be confused with code content)
    - Markdown fences: NEW_CODE: ```python...``` and NEW_META_CODE: ```python...```
      (fallback for backward compat — breaks when code contains ``` in strings)
    """
    # Parse strategy name and description
    name = ""
    description = ""
    for line in response.split("\n"):
        if line.startswith("STRATEGY_NAME:"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("STRATEGY_DESCRIPTION:"):
            description = line.split(":", 1)[1].strip()

    # Extract task code: FIRST <NEW_CODE> to LAST </NEW_CODE> before meta section.
    # Evolved code often contains these tags inside string literals, so we use
    # index/rindex to find the outermost pair rather than regex.
    new_task_code = None
    if "<NEW_CODE>" in response and "</NEW_CODE>" in response:
        # Isolate task section: cut off everything from first <NEW_META_CODE>
        task_response = response
        if "<NEW_META_CODE>" in task_response:
            task_response = task_response[:task_response.index("<NEW_META_CODE>")]
        try:
            first_open = task_response.index("<NEW_CODE>") + len("<NEW_CODE>")
            last_close = task_response.rindex("</NEW_CODE>")
            new_task_code = task_response[first_open:last_close].strip()
        except ValueError:
            pass
    if new_task_code is None:
        # Markdown fallback
        code_section = response.split("NEW_CODE:", 1)[-1] if "NEW_CODE:" in response else ""
        if code_section:
            for marker in ["NEW_META_CODE:", "<NEW_META_CODE>"]:
                if marker in code_section:
                    code_section = code_section.split(marker, 1)[0]
            cm = re.search(r"```python\s*\n(.*)```", code_section, re.DOTALL)
            if cm:
                new_task_code = cm.group(1).strip()

    # Extract meta code: FIRST <NEW_META_CODE> to LAST </NEW_META_CODE>.
    # Greedy: the evolved meta_code often contains these tags inside prompt
    # template strings. index/rindex finds the outermost pair.
    new_meta_code = None
    xml_meta = None
    if "<NEW_META_CODE>" in response and "</NEW_META_CODE>" in response:
        try:
            first_open = response.index("<NEW_META_CODE>") + len("<NEW_META_CODE>")
            last_close = response.rindex("</NEW_META_CODE>")
            new_meta_code = response[first_open:last_close].strip()
            xml_meta = True  # flag to skip fallback
        except ValueError:
            pass
    if xml_meta is None:
        # Markdown fallback
        meta_match = re.search(
            r"NEW_META_CODE:\s*```python\s*\n(.*)```",
            response, re.DOTALL,
        )
        new_meta_code = meta_match.group(1).strip() if meta_match else None

    return new_task_code, new_meta_code, name, description


# ---------------------------------------------------------------------------
# Initial task code — intentionally weak starting point.
# The meta agent's job: improve this AND improve how it improves.
# ---------------------------------------------------------------------------
INITIAL_TASK_CODE = """\
import re

def solve(problem: str) -> int:
    \"\"\"Read the problem, call the LLM, extract an integer from the response.\"\"\"
    response = llm_call(
        "You are an expert assistant. Read the task carefully and follow "
        "the instructions. Put ONLY your final integer answer on the last line.",
        problem,
    )
    # Try to extract the last integer on the last non-empty line
    lines = [ln.strip() for ln in response.strip().splitlines() if ln.strip()]
    if lines:
        last = lines[-1].strip("*_`#$\\\\boxed{}() ").strip()
        m = re.match(r'^(\\d+)$', last)
        if m:
            return int(m.group(1))
    # Fallback: last integer anywhere in response
    nums = re.findall(r'\\b(\\d+)\\b', response)
    return int(nums[-1]) if nums else -1
"""

# ---------------------------------------------------------------------------
# Initial meta code — the improve() function, also editable.
#
# Signature: improve(task_code, history, llm_call, task_description,
#                    mean_score, recent_score, generation, current_meta_code)
#            -> (new_task_code, new_meta_code, strategy_name, strategy_description)
#
# new_task_code: str | None  — None means keep current
# new_meta_code: str | None  — None means keep current (no metacognitive update)
# strategy_name: str
# strategy_description: str
#
# The meta agent CAN return NEW_META_CODE to rewrite this very function.
# This is metacognitive self-modification: improving the improvement procedure.
# ---------------------------------------------------------------------------
INITIAL_META_CODE = '''\
import re

def improve(task_code, history, llm_call, task_description,
            mean_score, recent_score, generation, current_meta_code):
    """
    Meta-improvement function.

    Returns (new_task_code, new_meta_code, strategy_name, strategy_description).
    Set new_task_code = None to keep current task code unchanged.
    Set new_meta_code = None to keep this meta function unchanged.
    current_meta_code is the source of THIS function — use it to see what you are rewriting.
    """

    meta_prompt = (
        "You are a meta-learning agent for a self-improving AI system.\\n\\n"
        "You have TWO jobs:\\n"
        "1. Improve the task solver (solve() function) based on failure analysis.\\n"
        "2. Optionally improve THIS meta function itself (the improve() function).\\n\\n"
        "The task solver calls llm_call(system, user) -> str and returns an int.\\n"
        "The meta function can also call llm_call.\\n"
        "Standard library (re, math, collections, etc.) available in both.\\n\\n"
        "Common task solver improvements: better prompting, answer extraction,\\n"
        "chain-of-thought, self-verification, problem-type detection.\\n\\n"
        "Common meta function improvements: richer failure analysis, tracking\\n"
        "patterns across generations, adaptive prompting strategy, memory of\\n"
        "what has and hasn\'t worked, smarter reflection prompts.\\n\\n"
        "REQUIRED output format (use EXACTLY these XML tags, NOT markdown fences):\\n"
        "STRATEGY_NAME: <short name>\\n"
        "STRATEGY_DESCRIPTION: <what changed and why>\\n"
        "<NEW_CODE>\\n"
        "complete solve() function here\\n"
        "</NEW_CODE>\\n"
        "<NEW_META_CODE>\\n"
        "complete improve() function here (optional — only if genuine improvement)\\n"
        "</NEW_META_CODE>\\n"
        "CRITICAL: The improve() function signature MUST be exactly:\\n"
        "  def improve(task_code, history, llm_call, task_description, mean_score, recent_score, generation, current_meta_code)\\n"
        "It MUST return a 4-tuple: (new_task_code, new_meta_code, name, description).\\n"
        "Do NOT rename, remove, or reorder parameters. The caller passes all 8 positionally.\\n"
        "Do NOT wrap code in markdown fences (```). Use ONLY the XML tags above.\\n"
    )

    recent = history[-5:]
    lines = []
    for r in recent:
        problem = str(r.get("problem", "?"))[:120]
        expected = r.get("expected", "?")
        got = r.get("got", "?")
        error = r.get("error", "")
        status = "PASS" if r.get("score", 0) >= 1.0 else "FAIL"
        line = f"  [{status}] Problem: {repr(problem)}\\n        Expected={expected}, Got={got}"
        if error:
            line += f", Error={repr(error)}"
        lines.append(line)
    attempts_text = "\\n".join(lines) if lines else "No attempts yet."

    # Include adopted strategy as reference (if available)
    adopted = globals().get("_adopted_strategy")
    adopted_text = ""
    if adopted and isinstance(adopted, dict) and adopted.get("task_code"):
        adopted_text = (
            f"\\n\\nREFERENCE: A high-scoring strategy from another agent:\\n"
            f"Strategy name: {adopted.get('name', '?')}\\n"
            f"Description: {adopted.get('description', '?')}\\n"
            f"Their solve() code:\\n```python\\n{adopted['task_code']}\\n```\\n"
            f"You may borrow ideas from this strategy, but write your OWN improved "
            f"version based on YOUR current code and failure patterns above.\\n"
        )

    reflection_prompt = (
        f"Task: {task_description}\\n\\n"
        f"Current solve() code:\\n```python\\n{task_code}\\n```\\n\\n"
        f"Current improve() code (this function):\\n```python\\n{current_meta_code}\\n```\\n\\n"
        f"Recent attempts:\\n{attempts_text}\\n\\n"
        f"Current mean score: {mean_score:.3f} (last-5 mean: {recent_score:.3f})\\n"
        f"Versions tried: {generation + 1}\\n\\n"
        f"{adopted_text}"
        "Identify failure patterns. Write improved solve() code.\\n"
        "Also consider whether to improve the meta improve() function itself."
    )

    response = llm_call(meta_prompt, reflection_prompt)

    # Use the factored-out parser (injected into namespace by the caller)
    _parse = globals().get("_parse_improve_response")
    if _parse:
        return _parse(response)

    # Inline fallback (if parser not injected — shouldn't happen in normal flow)
    import re as _re
    name, description = "", ""
    for line in response.split("\\n"):
        if line.startswith("STRATEGY_NAME:"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("STRATEGY_DESCRIPTION:"):
            description = line.split(":", 1)[1].strip()
    new_task_code = None
    if "<NEW_CODE>" in response and "</NEW_CODE>" in response:
        _tr = response
        if "<NEW_META_CODE>" in _tr:
            _tr = _tr[:_tr.index("<NEW_META_CODE>")]
        try:
            _fo = _tr.index("<NEW_CODE>") + len("<NEW_CODE>")
            _lc = _tr.rindex("</NEW_CODE>")
            new_task_code = _tr[_fo:_lc].strip()
        except ValueError:
            pass
    new_meta_code = None
    if "<NEW_META_CODE>" in response and "</NEW_META_CODE>" in response:
        try:
            _fo = response.index("<NEW_META_CODE>") + len("<NEW_META_CODE>")
            _lc = response.rindex("</NEW_META_CODE>")
            new_meta_code = response[_fo:_lc].strip()
        except ValueError:
            pass
    if xml_meta:
        new_meta_code = xml_meta.group(1).strip()
    return new_task_code, new_meta_code, name, description
'''


_META_IMPROVE_NPARAMS = 8  # task_code, history, llm_call, task_description, mean_score, recent_score, generation, current_meta_code


def _validate_meta_code(code: str) -> tuple[bool, str]:
    """Validate that meta_code defines improve() with the correct 8-arg signature."""
    try:
        ns: dict = {}
        exec(code, ns)  # noqa: S102
    except Exception as exc:  # noqa: BLE001
        return False, f"exec failed: {exc}"

    fn = ns.get("improve")
    if not callable(fn):
        return False, "no callable improve()"

    try:
        sig = inspect.signature(fn)
        params = [
            p for p in sig.parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        if len(params) < _META_IMPROVE_NPARAMS:
            return False, f"improve() takes {len(params)} required args, need {_META_IMPROVE_NPARAMS}"
    except (ValueError, TypeError):
        pass  # can't inspect builtins etc — accept if callable

    return True, ""


@dataclass
class CodeHyperagentConfig:
    """Configuration for a code-evolving hyperagent."""

    max_strategies: int = 20
    exec_timeout_s: float = 30.0


class CodeHyperagent:
    """
    A hyperagent whose editable program is Python code.

    Both task_code (solve()) and meta_code (improve()) are editable.
    Each generation:
      - solve() is exec'd to attempt tasks
      - improve() is exec'd to reflect and rewrite both solve() AND improve()

    This is metacognitive self-modification: the improvement procedure
    can improve itself.
    """

    def __init__(
        self,
        config: CodeHyperagentConfig | None = None,
        agent_id: str | None = None,
        name: str | None = None,
    ) -> None:
        self.id = agent_id or uuid.uuid4().hex[:12]
        self.name = name or f"code-agent-{self.id[:6]}"
        self.config = config or CodeHyperagentConfig()
        self.task_code: str = INITIAL_TASK_CODE
        self.meta_code: str = INITIAL_META_CODE
        self.strategies = StrategyStore(max_size=self.config.max_strategies)
        self.history: list[TaskResult] = []
        self._generation = 0

    @property
    def task_prompt(self) -> str:
        return self.task_code

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def mean_score(self) -> float:
        if not self.history:
            return 0.0
        return sum(r.score for r in self.history) / len(self.history)

    @property
    def recent_score(self) -> float:
        recent = self.history[-5:]
        if not recent:
            return 0.0
        return sum(r.score for r in recent) / len(recent)

    # ------------------------------------------------------------------
    # solve — execute current task_code and score against task
    # ------------------------------------------------------------------
    # Infrastructure errors that should not count as wrong answers
    _INFRA_ERRORS = ("ReadTimeout", "ConnectTimeout", "TimeoutException", "ConnectionError")

    def solve(self, task: Task, llm_call: LLMCallable) -> TaskResult:
        problem = task.prompt()

        raw_output: Any = None
        error_str: str = ""
        is_infra_error = False
        try:
            namespace: dict = {"llm_call": llm_call}
            exec(self.task_code, namespace)  # noqa: S102
            raw_output = namespace["solve"](problem)
        except Exception as exc:  # noqa: BLE001
            error_str = f"{type(exc).__name__}: {exc}"
            is_infra_error = any(ie in error_str for ie in self._INFRA_ERRORS)
            logger.warning("Agent %s solve error%s: %s", self.name,
                           " (infra)" if is_infra_error else "", error_str)

        base = task.evaluate(str(raw_output) if raw_output is not None else "-1")
        result = TaskResult(
            score=base.score,
            output=base.output,
            metadata={
                **base.metadata,
                "problem": problem[:500],
                "got": base.metadata.get("got", raw_output),
                "error": error_str,
                "infra_error": is_infra_error,
            },
        )
        # Don't pollute history with infrastructure failures —
        # improve() would try to "fix" problems that aren't in the code
        if not is_infra_error:
            self.history.append(result)
        return result

    # ------------------------------------------------------------------
    # improve — exec meta_code and call improve(), update both codes
    # ------------------------------------------------------------------
    def improve(self, task: Task, llm_call: LLMCallable) -> Strategy | None:
        if len(self.history) < 2:
            return None

        history_dicts = [
            {
                "score": r.score,
                "problem": r.metadata.get("problem", ""),
                "expected": r.metadata.get("expected", "?"),
                "got": r.metadata.get("got", "?"),
                "error": r.metadata.get("error", ""),
            }
            for r in self.history
        ]

        try:
            adopted = getattr(self, "_adopted_strategy", None)
            namespace: dict = {
                "llm_call": llm_call,
                "_adopted_strategy": adopted,
                "_parse_improve_response": parse_improve_response,
            }
            exec(self.meta_code, namespace)  # noqa: S102
            result = namespace["improve"](
                self.task_code,
                history_dicts,
                llm_call,
                task.description,
                self.mean_score,
                self.recent_score,
                self._generation,
                self.meta_code,
            )
            new_task_code, new_meta_code, name, description = result
        except Exception as exc:  # noqa: BLE001
            logger.warning("Agent %s meta error: %s", self.name, exc)
            return None
        finally:
            # Clear adopted strategy after use
            self._adopted_strategy = None

        if not name or not new_task_code:
            return None

        # IsValid: validate new_task_code defines a callable solve() before accepting
        try:
            test_ns: dict = {"llm_call": lambda s, u: "42"}
            exec(new_task_code, test_ns)  # noqa: S102
            if not callable(test_ns.get("solve")):
                logger.warning("Agent %s: new_task_code has no callable solve(), rejecting", self.name)
                return None
        except Exception as exc:  # noqa: BLE001
            logger.warning("Agent %s: new_task_code failed validation: %s", self.name, exc)
            return None

        # Apply task code update
        self.task_code = new_task_code

        # Apply metacognitive update — the improve() function rewrote itself
        # Skip if freeze_meta is set (DGM-H w/o self-improve ablation)
        meta_updated = False
        if new_meta_code and not getattr(self, "freeze_meta", False):
            valid, reason = _validate_meta_code(new_meta_code)
            if valid:
                self.meta_code = new_meta_code
                meta_updated = True
                logger.info(
                    "Agent %s gen %d: METACOGNITIVE UPDATE — meta_code rewritten",
                    self.name, self._generation + 1,
                )
            else:
                logger.warning(
                    "Agent %s: new_meta_code rejected: %s", self.name, reason
                )

        strategy = Strategy(
            name=name,
            description=description or name,
            content=new_task_code,
            meta_code=new_meta_code if meta_updated else None,
        )
        strategy.author_agent_id = self.id
        self.strategies.add(strategy)
        self._generation += 1
        logger.info(
            "Agent %s gen %d: strategy='%s'%s",
            self.name,
            self._generation,
            strategy.name,
            " [+meta]" if meta_updated else "",
        )
        return strategy

    # ------------------------------------------------------------------
    # adopt_strategy — receive a strategy from the mark space
    # ------------------------------------------------------------------
    def adopt_strategy(self, strategy: Strategy) -> None:
        """Store adopted strategy as context for improve(), not as overwrite.

        The adopted strategy is passed to improve() via the exec namespace
        as _adopted_strategy — a dict with 'task_code', 'meta_code', 'name',
        'description', and 'score'. The improve() function can reference it
        in its reflection prompt but the agent's own task_code remains the base.
        """
        self._adopted_strategy: dict | None = {
            "task_code": strategy.content,
            "meta_code": strategy.meta_code,
            "name": strategy.name,
            "description": strategy.description,
        }
        if strategy.id not in self.strategies:
            self.strategies.add(strategy)
        logger.info(
            "Agent %s: adopted strategy '%s' as context (not overwriting code)",
            self.name, strategy.name,
        )
        self._generation += 1
        logger.info(
            "Agent %s adopted strategy '%s' from %s",
            self.name, strategy.name, strategy.author_agent_id,
        )

    def _format_attempts(self, results: list[TaskResult]) -> str:
        lines = []
        for i, r in enumerate(results, 1):
            problem = r.metadata.get("problem", "?")[:120]
            expected = r.metadata.get("expected", "?")
            got = r.metadata.get("got", "?")
            error = r.metadata.get("error", "")
            status = "PASS" if r.score >= 1.0 else "FAIL"
            line = f"  [{status}] Problem: {problem!r}\n        Expected={expected}, Got={got}"
            if error:
                line += f", Error={error!r}"
            lines.append(line)
        return "\n".join(lines)
