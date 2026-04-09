"""
Code refactoring tool: automated code transformations for self-improvement.

Provides common refactoring operations to improve code quality:
- Extract function/method
- Rename variable/function
- Add type hints
- Simplify expressions
- Remove dead code
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any


def tool_info() -> dict:
    return {
        "name": "code_refactor",
        "description": (
            "Automated code refactoring tool for common transformations. "
            "Operations: extract_function, add_type_hints, simplify_expressions, "
            "remove_unused_imports, format_docstrings."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": [
                        "extract_function",
                        "add_type_hints",
                        "simplify_expressions",
                        "remove_unused_imports",
                        "format_docstrings",
                        "analyze_complexity",
                    ],
                    "description": "The refactoring operation to perform.",
                },
                "path": {
                    "type": "string",
                    "description": "Absolute path to the Python file to refactor.",
                },
                "function_name": {
                    "type": "string",
                    "description": "Target function name (for extract_function).",
                },
                "start_line": {
                    "type": "integer",
                    "description": "Start line of code block to extract (1-indexed).",
                },
                "end_line": {
                    "type": "integer",
                    "description": "End line of code block to extract (1-indexed).",
                },
                "new_function_name": {
                    "type": "string",
                    "description": "Name for the extracted function.",
                },
            },
            "required": ["operation", "path"],
        },
    }


def tool_function(
    operation: str,
    path: str,
    function_name: str | None = None,
    start_line: int | None = None,
    end_line: int | None = None,
    new_function_name: str | None = None,
) -> str:
    """Execute a code refactoring operation."""
    try:
        p = Path(path)
        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."
        if not p.exists():
            return f"Error: {path} does not exist."
        if p.suffix != ".py":
            return f"Error: Only Python files (.py) are supported, got {p.suffix}"

        content = p.read_text()

        if operation == "extract_function":
            if start_line is None or end_line is None or new_function_name is None:
                return "Error: extract_function requires start_line, end_line, and new_function_name."
            return _extract_function(p, content, start_line, end_line, new_function_name)
        
        elif operation == "add_type_hints":
            return _add_type_hints(p, content)
        
        elif operation == "simplify_expressions":
            return _simplify_expressions(p, content)
        
        elif operation == "remove_unused_imports":
            return _remove_unused_imports(p, content)
        
        elif operation == "format_docstrings":
            return _format_docstrings(p, content)
        
        elif operation == "analyze_complexity":
            return _analyze_complexity(p, content)
        
        else:
            return f"Error: Unknown operation '{operation}'"
            
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


def _extract_function(
    p: Path, 
    content: str, 
    start_line: int, 
    end_line: int, 
    new_name: str
) -> str:
    """Extract a block of code into a new function."""
    lines = content.split("\n")
    
    if start_line < 1 or end_line > len(lines) or start_line >= end_line:
        return f"Error: Invalid line range [{start_line}, {end_line}] for file with {len(lines)} lines."
    
    # Extract the code block
    code_block = lines[start_line - 1:end_line]
    
    # Detect indentation
    base_indent = len(code_block[0]) - len(code_block[0].lstrip())
    
    # Remove base indentation from code block
    dedented_block = []
    for line in code_block:
        if line.strip():
            if len(line) >= base_indent:
                dedented_block.append(line[base_indent:])
            else:
                dedented_block.append(line.lstrip())
        else:
            dedented_block.append("")
    
    # Build the new function
    new_function = [f"def {new_name}():"]
    for line in dedented_block:
        if line.strip():
            new_function.append("    " + line)
        else:
            new_function.append("")
    new_function.append("")
    
    # Replace the original code block with a function call
    call_line = " " * base_indent + f"{new_name}()"
    new_lines = lines[:start_line - 1] + [call_line] + lines[end_line:]
    
    # Insert the new function before the current function
    # Find the function definition that contains this code
    func_start = 0
    for i in range(start_line - 1, -1, -1):
        if lines[i].strip().startswith("def "):
            func_start = i
            break
    
    # Insert new function before the containing function
    final_lines = new_lines[:func_start] + new_function + new_lines[func_start:]
    
    new_content = "\n".join(final_lines)
    p.write_text(new_content)
    
    return (
        f"Extracted lines {start_line}-{end_line} into function '{new_name}'.\n"
        f"New function added with {len(dedented_block)} lines.\n"
        f"Original code replaced with '{new_name}()' call."
    )


def _add_type_hints(p: Path, content: str) -> str:
    """Add basic type hints to function signatures."""
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        return f"Error: Cannot parse file - {e}"
    
    lines = content.split("\n")
    modifications = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Check if function already has type hints
            has_hints = False
            if node.returns or any(arg.annotation for arg in node.args.args):
                has_hints = True
            
            if not has_hints and node.args.args:
                # Add Any type hints for parameters without annotations
                line_idx = node.lineno - 1
                line = lines[line_idx]
                
                # Simple heuristic: add -> Any return type if missing
                if not node.returns and "->" not in line:
                    # Find the end of the parameter list
                    if ")" in line:
                        # Add return type before colon
                        new_line = line.rstrip()
                        if new_line.endswith(":"):
                            new_line = new_line[:-1] + " -> Any:"
                        elif ")" in new_line:
                            # Insert before the colon
                            idx = new_line.rfind(":")
                            if idx > 0:
                                new_line = new_line[:idx] + " -> Any" + new_line[idx:]
                        
                        if new_line != line:
                            modifications.append((line_idx, line, new_line))
    
    if not modifications:
        return "No type hints added - functions already have hints or none found."
    
    # Apply modifications in reverse order to preserve line numbers
    for line_idx, old_line, new_line in reversed(modifications):
        lines[line_idx] = new_line
    
    new_content = "\n".join(lines)
    p.write_text(new_content)
    
    return f"Added type hints to {len(modifications)} function(s)."


def _simplify_expressions(p: Path, content: str) -> str:
    """Simplify common expression patterns."""
    original_content = content
    simplifications = []
    
    # Pattern 1: Replace list() with []
    new_content = re.sub(r'\blist\(\s*\)', '[]', content)
    if new_content != content:
        simplifications.append("list() -> []")
        content = new_content
    
    # Pattern 2: Replace dict() with {}
    new_content = re.sub(r'\bdict\(\s*\)', '{}', content)
    if new_content != content:
        simplifications.append("dict() -> {}")
        content = new_content
    
    # Pattern 3: Replace set() with set() - no change needed
    
    # Pattern 4: Simplify x == True to x (in if statements)
    new_content = re.sub(r'if\s+(\w+)\s*==\s*True\s*:', r'if \1:', content)
    if new_content != content:
        simplifications.append("x == True -> x in if statements")
        content = new_content
    
    # Pattern 5: Simplify x == False to not x
    new_content = re.sub(r'if\s+(\w+)\s*==\s*False\s*:', r'if not \1:', content)
    if new_content != content:
        simplifications.append("x == False -> not x in if statements")
        content = new_content
    
    # Pattern 6: Replace len(x) == 0 with not x
    new_content = re.sub(r'if\s+len\((\w+)\)\s*==\s*0\s*:', r'if not \1:', content)
    if new_content != content:
        simplifications.append("len(x) == 0 -> not x in if statements")
        content = new_content
    
    if content == original_content:
        return "No simplifications applied - no matching patterns found."
    
    p.write_text(content)
    
    return f"Applied {len(simplifications)} simplification(s):\n" + "\n".join(f"  - {s}" for s in simplifications)


def _remove_unused_imports(p: Path, content: str) -> str:
    """Remove unused import statements."""
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        return f"Error: Cannot parse file - {e}"
    
    # Find all imports
    imports = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports[alias.asname or alias.name] = node
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                name = alias.asname or alias.name
                imports[name] = node
    
    # Find all names used in the file
    used_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            used_names.add(node.id)
        elif isinstance(node, ast.Attribute):
            # For module.attribute, check the base module name
            if isinstance(node.value, ast.Name):
                used_names.add(node.value.id)
    
    # Find unused imports
    unused = set(imports.keys()) - used_names
    
    if not unused:
        return "No unused imports found."
    
    # Remove unused imports from content
    lines = content.split("\n")
    lines_to_remove = set()
    
    for name in unused:
        node = imports[name]
        lines_to_remove.add(node.lineno - 1)
    
    # Remove lines in reverse order
    new_lines = [line for i, line in enumerate(lines) if i not in lines_to_remove]
    
    new_content = "\n".join(new_lines)
    p.write_text(new_content)
    
    return f"Removed {len(unused)} unused import(s): {', '.join(sorted(unused))}"


def _format_docstrings(p: Path, content: str) -> str:
    """Standardize docstring formatting."""
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        return f"Error: Cannot parse file - {e}"
    
    lines = content.split("\n")
    modifications = []
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            # Check if function has a docstring
            if (
                node.body 
                and isinstance(node.body[0], ast.Expr) 
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                docstring = node.body[0].value.value
                # Check if it needs formatting
                if not docstring.startswith('"""') or not docstring.endswith('"""'):
                    # Standardize to triple double quotes
                    stripped = docstring.strip('"\'\n ')
                    new_docstring = '"""' + stripped + '"""'
                    
                    # Find the line with the docstring
                    line_idx = node.body[0].lineno - 1
                    line = lines[line_idx]
                    
                    # Simple replacement (may not handle all edge cases)
                    if '"' in line or "'" in line:
                        # Extract indentation
                        indent = len(line) - len(line.lstrip())
                        new_line = " " * indent + new_docstring
                        modifications.append((line_idx, line, new_line))
    
    if not modifications:
        return "No docstring formatting needed - all docstrings are properly formatted."
    
    # Apply modifications in reverse order
    for line_idx, old_line, new_line in reversed(modifications):
        lines[line_idx] = new_line
    
    new_content = "\n".join(lines)
    p.write_text(new_content)
    
    return f"Formatted {len(modifications)} docstring(s)."


def _analyze_complexity(p: Path, content: str) -> str:
    """Analyze code complexity metrics."""
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        return f"Error: Cannot parse file - {e}"
    
    results = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Calculate cyclomatic complexity (simplified)
            complexity = 1  # Base complexity
            
            for child in ast.walk(node):
                if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                    complexity += 1
                elif isinstance(child, ast.BoolOp):
                    complexity += len(child.values) - 1
            
            # Count lines
            if node.body:
                end_line = node.end_lineno or node.lineno
                line_count = end_line - node.lineno + 1
            else:
                line_count = 1
            
            results.append({
                "name": node.name,
                "complexity": complexity,
                "lines": line_count,
                "lineno": node.lineno,
            })
    
    if not results:
        return "No functions found to analyze."
    
    # Sort by complexity
    results.sort(key=lambda x: x["complexity"], reverse=True)
    
    output_lines = [f"Complexity analysis for {p.name}:", ""]
    output_lines.append(f"{'Function':<30} {'Complexity':>10} {'Lines':>8} {'Line #':>8}")
    output_lines.append("-" * 60)
    
    for r in results:
        complexity_warning = " ⚠️" if r["complexity"] > 10 else ""
        output_lines.append(
            f"{r['name']:<30} {r['complexity']:>10} {r['lines']:>8} {r['lineno']:>8}{complexity_warning}"
        )
    
    output_lines.append("")
    output_lines.append("Complexity scale: 1-5 (low), 6-10 (medium), 11+ (high)")
    
    return "\n".join(output_lines)
