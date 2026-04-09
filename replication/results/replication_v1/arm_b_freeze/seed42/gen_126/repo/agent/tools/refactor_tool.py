"""
Code refactoring tool: extract methods, rename variables, and restructure code.

Provides automated refactoring capabilities to improve code quality,
maintainability, and readability.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any


def tool_info() -> dict:
    return {
        "name": "refactor",
        "description": (
            "Code refactoring tool for improving code structure. "
            "Supports: extract_method (pull code into new function), "
            "rename_variable (rename variables consistently), "
            "remove_duplicates (find and suggest deduplication). "
            "Works with Python files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["extract_method", "rename_variable", "remove_duplicates", "analyze_complexity"],
                    "description": "The refactoring command to run.",
                },
                "path": {
                    "type": "string",
                    "description": "Absolute path to the Python file.",
                },
                "start_line": {
                    "type": "integer",
                    "description": "Start line for extract_method (1-indexed).",
                },
                "end_line": {
                    "type": "integer",
                    "description": "End line for extract_method (1-indexed).",
                },
                "method_name": {
                    "type": "string",
                    "description": "Name for extracted method.",
                },
                "old_name": {
                    "type": "string",
                    "description": "Old variable name for rename_variable.",
                },
                "new_name": {
                    "type": "string",
                    "description": "New variable name for rename_variable.",
                },
            },
            "required": ["command", "path"],
        },
    }


def tool_function(
    command: str,
    path: str,
    start_line: int | None = None,
    end_line: int | None = None,
    method_name: str | None = None,
    old_name: str | None = None,
    new_name: str | None = None,
) -> str:
    """Execute a refactoring command."""
    try:
        p = Path(path)

        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."

        if not p.exists():
            return f"Error: {p} does not exist."

        if not str(p).endswith(".py"):
            return f"Error: refactor tool only works with Python files (.py)."

        content = p.read_text()
        lines = content.split("\n")

        if command == "extract_method":
            if start_line is None or end_line is None or method_name is None:
                return "Error: start_line, end_line, and method_name required for extract_method."
            return _extract_method(lines, start_line, end_line, method_name, path)

        elif command == "rename_variable":
            if old_name is None or new_name is None:
                return "Error: old_name and new_name required for rename_variable."
            return _rename_variable(content, old_name, new_name, path)

        elif command == "remove_duplicates":
            return _find_duplicates(content, path)

        elif command == "analyze_complexity":
            return _analyze_complexity(content, path)

        else:
            return f"Error: unknown command {command}"

    except Exception as e:
        return f"Error: {e}"


def _extract_method(
    lines: list[str], start_line: int, end_line: int, method_name: str, path: str
) -> str:
    """Extract a block of code into a new method."""
    if start_line < 1 or end_line > len(lines) or start_line > end_line:
        return f"Error: Invalid line range [{start_line}, {end_line}]. File has {len(lines)} lines."

    # Extract the code block
    code_block = lines[start_line - 1 : end_line]
    indent = len(code_block[0]) - len(code_block[0].lstrip())

    # Find variables used in the block (simple heuristic)
    var_pattern = re.compile(r"\b([a-z_][a-z0-9_]*)\b", re.IGNORECASE)
    all_vars = set()
    for line in code_block:
        all_vars.update(var_pattern.findall(line))

    # Filter out keywords and builtins
    keywords = {
        "if", "else", "elif", "for", "while", "def", "class", "return",
        "import", "from", "as", "try", "except", "finally", "with", "in",
        "not", "and", "or", "is", "None", "True", "False", "pass",
        "break", "continue", "lambda", "yield", "raise", "assert",
        "del", "global", "nonlocal", "print", "len", "range", "enumerate",
        "zip", "map", "filter", "sum", "min", "max", "int", "str", "float",
        "list", "dict", "set", "tuple", "bool", "type", "isinstance",
    }
    potential_params = sorted(all_vars - keywords)

    # Build the new method
    method_lines = [f"\n{' ' * indent}def {method_name}({', '.join(potential_params)}):"]
    for line in code_block:
        # Add extra indent
        method_lines.append("    " + line)

    # Build the replacement call
    call_line = f"{' ' * indent}{method_name}({', '.join(potential_params)})"

    result = f"Suggested refactoring for {path}:\n\n"
    result += f"1. Add this new method (insert after line {end_line}):\n"
    result += "\n".join(method_lines) + "\n\n"
    result += f"2. Replace lines {start_line}-{end_line} with:\n"
    result += call_line + "\n\n"
    result += f"Note: Review the parameter list and remove unused variables: {potential_params}"

    return result


def _rename_variable(content: str, old_name: str, new_name: str, path: str) -> str:
    """Rename a variable throughout the file."""
    # Count occurrences
    pattern = re.compile(rf"\b{re.escape(old_name)}\b")
    count = len(pattern.findall(content))

    if count == 0:
        return f"Error: Variable '{old_name}' not found in {path}"

    # Check if new_name already exists
    if re.search(rf"\b{re.escape(new_name)}\b", content):
        return f"Warning: '{new_name}' already exists in the file. This may cause conflicts."

    # Perform replacement
    new_content = pattern.sub(new_name, content)

    # Show a preview
    lines = content.split("\n")
    preview_lines = []
    for i, line in enumerate(lines, 1):
        if pattern.search(line):
            new_line = pattern.sub(new_name, line)
            preview_lines.append(f"  {i}: {line}")
            preview_lines.append(f"  {i}: {new_line}")
            if len(preview_lines) >= 10:
                preview_lines.append("  ...")
                break

    result = f"Suggested rename in {path}:\n\n"
    result += f"Replace '{old_name}' with '{new_name}' ({count} occurrences)\n\n"
    result += "Preview of changes:\n"
    result += "\n".join(preview_lines) + "\n\n"
    result += "To apply this change, use the editor tool with str_replace."

    return result


def _find_duplicates(content: str, path: str) -> str:
    """Find potential code duplicates."""
    lines = content.split("\n")

    # Find duplicate lines (ignoring empty lines and comments)
    line_counts: dict[str, list[int]] = {}
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            line_counts.setdefault(stripped, []).append(i)

    duplicates = {line: nums for line, nums in line_counts.items() if len(nums) > 1}

    if not duplicates:
        return f"No obvious duplicate lines found in {path}."

    result = f"Potential duplicates found in {path}:\n\n"
    for line, nums in sorted(duplicates.items(), key=lambda x: -len(x[1]))[:10]:
        result += f"Line appears {len(nums)} times: {line[:60]}\n"
        result += f"  At lines: {', '.join(map(str, nums))}\n\n"

    result += "Consider extracting these into helper functions or constants."
    return result


def _analyze_complexity(content: str, path: str) -> str:
    """Analyze code complexity metrics."""
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        return f"Error parsing {path}: {e}"

    functions = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Count lines
            start = node.lineno
            end = node.end_lineno or start
            lines = end - start + 1

            # Count branches (if, for, while, except, with, comprehensions)
            branches = 0
            for child in ast.walk(node):
                if isinstance(
                    child,
                    (ast.If, ast.For, ast.While, ast.ExceptHandler, ast.With,
                     ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
                ):
                    branches += 1

            # Simple cyclomatic complexity approximation
            complexity = branches + 1

            functions.append({
                "name": node.name,
                "lines": lines,
                "branches": branches,
                "complexity": complexity,
                "start": start,
            })

    if not functions:
        return f"No functions found in {path}."

    # Sort by complexity
    functions.sort(key=lambda x: -x["complexity"])

    result = f"Complexity analysis for {path}:\n\n"
    result += f"{'Function':<30} {'Lines':>8} {'Branches':>10} {'Complexity':>12}\n"
    result += "-" * 62 + "\n"

    for func in functions:
        complexity_indicator = ""
        if func["complexity"] > 10:
            complexity_indicator = " ⚠️ HIGH"
        elif func["complexity"] > 5:
            complexity_indicator = " ⚡ MEDIUM"

        result += f"{func['name']:<30} {func['lines']:>8} {func['branches']:>10} {func['complexity']:>12}{complexity_indicator}\n"

    result += "\n"
    result += "Complexity levels: 1-5 (low), 6-10 (medium), 11+ (high - consider refactoring)\n"
    result += f"Most complex function: {functions[0]['name']} at line {functions[0]['start']}"

    return result
