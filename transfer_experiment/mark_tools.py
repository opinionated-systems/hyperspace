"""
Mark-backed tool implementations.

The LLM sees the same bash and editor tool interface. The harness replaces
the tool functions with these implementations that translate all file
operations to markspace mark operations.

The LLM does not know about markspace.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
import threading

from markspace import (
    Agent,
    Guard,
    MarkSpace,
    MarkType,
    Observation,
    Source,
)

logger = logging.getLogger(__name__)


class MarkBackedCodebase:
    """A virtual codebase backed by markspace marks.

    All file operations go through the Guard. The LLM sees a normal
    filesystem but nothing touches disk except sandboxed bash execution.
    """

    def __init__(
        self,
        space: MarkSpace,
        guard: Guard,
        agent: Agent,
        scope: str,
    ):
        self.space = space
        self.guard = guard
        self.agent = agent
        self.scope = scope
        self._files: dict[str, str] = {}  # resource -> latest content
        self._lock = threading.Lock()

    def read_file(self, resource: str) -> str | None:
        """Read the latest version of a file from marks."""
        marks = self.space.read(
            scope=self.scope,
            topic=resource,
            mark_type=MarkType.OBSERVATION,
        )
        if marks:
            obs = [m for m in marks if isinstance(m, Observation)]
            latest = max(obs, key=lambda m: m.created_at)
            return latest.content
        with self._lock:
            return self._files.get(resource)

    def write_file(self, resource: str, content: str) -> str:
        """Write a file as a mark through the Guard."""
        try:
            self.guard.write_mark(
                self.agent,
                Observation(
                    scope=self.scope,
                    topic=resource,
                    content=content,
                    confidence=1.0,
                    source=Source.FLEET,
                ),
            )
            with self._lock:
                self._files[resource] = content
            return f"File written: {resource}"
        except Exception as e:
            return f"Error: access denied — {e}"

    def list_files(self) -> list[str]:
        """List all files in the codebase."""
        marks = self.space.read(scope=self.scope, mark_type=MarkType.OBSERVATION)
        resources = set()
        for m in marks:
            if isinstance(m, Observation):
                resources.add(m.topic)
        with self._lock:
            resources.update(self._files.keys())
        return sorted(resources)

    def init_from_directory(self, directory: str):
        """Initialize codebase from a directory, writing each file as a mark."""
        for root, _, files in os.walk(directory):
            for fname in files:
                if fname.endswith(".pyc") or "__pycache__" in root:
                    continue
                fpath = os.path.join(root, fname)
                rel = os.path.relpath(fpath, directory)
                content = open(fpath).read()
                self.write_file(rel, content)

    def materialize(self, target_dir: str):
        """Write all current files to a directory for bash execution."""
        os.makedirs(target_dir, exist_ok=True)
        for resource in self.list_files():
            content = self.read_file(resource)
            if content is None:
                continue
            fpath = os.path.join(target_dir, resource)
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            with open(fpath, "w") as f:
                f.write(content)


def make_mark_editor(codebase: MarkBackedCodebase, allowed_root: str):
    """Create an editor tool function backed by marks.

    Same interface as editor_tool.tool_function — the LLM sees no difference.
    """

    def tool_function(
        command: str,
        path: str,
        file_text: str | None = None,
        view_range: list[int] | None = None,
        old_str: str | None = None,
        new_str: str | None = None,
        insert_line: int | None = None,
    ) -> str:
        # Normalize path to resource name
        if path.startswith(allowed_root):
            resource = os.path.relpath(path, allowed_root)
        else:
            resource = path.lstrip("/")

        if command == "view":
            # Try as file first
            content = codebase.read_file(resource)
            if content is not None:
                if view_range:
                    lines = content.split("\n")
                    start, end = view_range
                    if end == -1:
                        end = len(lines)
                    content = "\n".join(lines[start - 1:end])
                    numbered = [f"{i + start:6}\t{line}" for i, line in enumerate(content.split("\n"))]
                else:
                    numbered = [f"{i + 1:6}\t{line}" for i, line in enumerate(content.split("\n"))]
                return f"Here's the result of running `cat -n` on {path}:\n" + "\n".join(numbered) + "\n"
            # Not a file — try as directory prefix
            matching = [f for f in codebase.list_files() if f.startswith(resource) or resource in (".", "")]
            if matching:
                return f"Files in {path}:\n" + "\n".join(os.path.join(path, f) for f in matching)
            return f"Error: {path} does not exist."

        elif command == "create":
            if not file_text:
                return "Error: file_text required for create."
            existing = codebase.read_file(resource)
            if existing is not None:
                return f"Error: {path} already exists."
            return codebase.write_file(resource, file_text)

        elif command == "str_replace":
            if old_str is None:
                return "Error: old_str required."
            content = codebase.read_file(resource)
            if content is None:
                return f"Error: {path} does not exist."
            content = content.expandtabs()
            old_str = old_str.expandtabs()
            new_str_expanded = (new_str or "").expandtabs()
            count = content.count(old_str)
            if count == 0:
                return f"Error: old_str not found in {path}"
            if count > 1:
                return f"Error: old_str appears {count} times. Make it unique."
            new_content = content.replace(old_str, new_str_expanded)
            result = codebase.write_file(resource, new_content)
            line_num = content.split(old_str)[0].count("\n")
            start = max(0, line_num - 4)
            end = line_num + 4 + new_str_expanded.count("\n")
            snippet = "\n".join(new_content.split("\n")[start:end + 1])
            numbered = [f"{i + start + 1:6}\t{line}" for i, line in enumerate(snippet.split("\n"))]
            return f"File {path} edited. Here's the result of running `cat -n` on snippet of {path}:\n" + "\n".join(numbered) + "\n"

        elif command == "insert":
            if insert_line is None or new_str is None:
                return "Error: insert_line and new_str required."
            content = codebase.read_file(resource)
            if content is None:
                return f"Error: {path} does not exist."
            lines = content.expandtabs().split("\n")
            new_lines = lines[:insert_line] + new_str.expandtabs().split("\n") + lines[insert_line:]
            return codebase.write_file(resource, "\n".join(new_lines))

        elif command == "undo_edit":
            return "Error: undo not supported with mark-backed codebase."

        return f"Error: unknown command {command}"

    return tool_function


def make_mark_bash(codebase: MarkBackedCodebase, allowed_root: str):
    """Create a bash tool function that runs through the Guard in a sandbox.

    1. Guard checks the command against the agent's scope
    2. If allowed, materializes codebase from marks to a temp directory
    3. Runs command in that temp directory
    4. Temp directory is destroyed — no writes persist
    """
    _timeout = 120.0

    def _execute_in_sandbox(command: str) -> str:
        """Run a command in a temp sandbox assembled from marks."""
        with tempfile.TemporaryDirectory() as sandbox:
            codebase.materialize(sandbox)
            try:
                result = subprocess.run(
                    ["bash", "--norc", "--noprofile", "-c", command],
                    capture_output=True,
                    text=True,
                    timeout=_timeout,
                    cwd=sandbox,
                    env={**os.environ, "HOME": sandbox},
                )
                output = result.stdout.strip()
                if result.stderr:
                    output += "\n" + result.stderr.strip()
                return output if output else "(no output)"
            except subprocess.TimeoutExpired:
                return f"Error: Timed out after {_timeout}s. Do NOT retry the same command."
            except Exception as e:
                return f"Error: {e}"

    def tool_function(command: str) -> str:
        # Guard checks: is this agent allowed to execute in this scope?
        decision, result = codebase.guard.execute(
            codebase.agent,
            scope=codebase.scope,
            resource="bash",
            intent_action="execute",
            result_action="executed",
            tool_fn=lambda: _execute_in_sandbox(command),
        )
        if hasattr(decision, 'verdict'):
            from markspace import GuardVerdict
            if decision.verdict != GuardVerdict.ALLOW:
                return f"Error: command blocked by Guard ({decision.verdict})"
        return result if isinstance(result, str) else str(result)

    return tool_function
