"""
Search tool: find files and search content within files.

Provides grep-like functionality and file finding capabilities
to help the meta agent locate code patterns and files efficiently.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for files and content. "
            "Commands: find_files (glob patterns), grep (search content), "
            "find_in_files (search within file contents with context). "
            "Useful for locating code patterns and files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["find_files", "grep", "find_in_files"],
                    "description": "The search command to run.",
                },
                "path": {
                    "type": "string",
                    "description": "Absolute path to search directory or file.",
                },
                "pattern": {
                    "type": "string",
                    "description": "Search pattern (glob for find_files, regex for grep/find_in_files).",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Optional file glob pattern to filter (e.g., '*.py').",
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Number of context lines for grep/find_in_files (default: 2).",
                },
            },
            "required": ["command", "path", "pattern"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _check_path_allowed(path: str) -> tuple[bool, str]:
    """Check if path is within allowed root. Returns (allowed, error_message)."""
    if _ALLOWED_ROOT is None:
        return True, ""
    resolved = os.path.abspath(path)
    if not resolved.startswith(_ALLOWED_ROOT):
        return False, f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"
    return True, ""


def _truncate_output(output: str, max_chars: int = 10000) -> str:
    """Truncate output if too long."""
    if len(output) > max_chars:
        return output[:max_chars // 2] + "\n... [output truncated] ...\n" + output[-max_chars // 2:]
    return output


def tool_function(
    command: str,
    path: str,
    pattern: str,
    file_pattern: str | None = None,
    context_lines: int = 2,
) -> str:
    """Execute a search command."""
    try:
        # Validate path
        allowed, error_msg = _check_path_allowed(path)
        if not allowed:
            return error_msg

        p = Path(path)
        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."

        if command == "find_files":
            return _find_files(p, pattern)
        elif command == "grep":
            return _grep(p, pattern, file_pattern, context_lines)
        elif command == "find_in_files":
            return _find_in_files(p, pattern, file_pattern, context_lines)
        else:
            return f"Error: unknown command {command}"
    except Exception as e:
        return f"Error: {e}"


def _find_files(directory: Path, glob_pattern: str) -> str:
    """Find files matching glob pattern."""
    if not directory.exists():
        return f"Error: {directory} does not exist."
    
    if not directory.is_dir():
        return f"Error: {directory} is not a directory."

    matches = list(directory.rglob(glob_pattern))
    
    # Limit results
    if len(matches) > 100:
        truncated = matches[:100]
        result = "\n".join(str(m) for m in truncated)
        return f"Found {len(matches)} files (showing first 100):\n{result}\n... and {len(matches) - 100} more"
    
    if not matches:
        return f"No files found matching '{glob_pattern}' in {directory}"
    
    result = "\n".join(str(m) for m in matches)
    return f"Found {len(matches)} files:\n{result}"


def _check_ripgrep_available() -> bool:
    """Check if ripgrep (rg) is available on the system."""
    try:
        subprocess.run(
            ["rg", "--version"],
            capture_output=True,
            timeout=5,
        )
        return True
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return False


def _grep(
    path: Path,
    pattern: str,
    file_pattern: str | None,
    context_lines: int,
) -> str:
    """Search for pattern in file(s) using ripgrep (if available) or grep.
    
    Automatically selects the best available tool:
    - ripgrep (rg): Faster, respects .gitignore, better for large codebases
    - grep: Universal fallback available on all systems
    """
    if not path.exists():
        return f"Error: {path} does not exist."

    # Prefer ripgrep if available for better performance
    use_rg = _check_ripgrep_available()
    
    if use_rg:
        # Build ripgrep command
        cmd = ["rg", "-n", "--color=never"]
        
        if context_lines > 0:
            cmd.extend(["-C", str(context_lines)])
        
        if file_pattern:
            cmd.extend(["-g", file_pattern])
        
        cmd.extend([pattern, str(path)])
        
        tool_name = "ripgrep"
    else:
        # Fallback to standard grep
        cmd = ["grep", "-n", "-r", "-E", "--include"]
        
        if file_pattern:
            cmd.append(file_pattern)
        else:
            cmd.append("*")
        
        if context_lines > 0:
            cmd.extend(["-C", str(context_lines)])
        
        cmd.extend([pattern, str(path)])
        
        tool_name = "grep"

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode == 0:
            output = _truncate_output(result.stdout)
            return f"Matches found (using {tool_name}):
{output}"
        elif result.returncode == 1:
            return f"No matches found for pattern '{pattern}'"
        else:
            return f"Error: {result.stderr}"
    except subprocess.TimeoutExpired:
        return "Error: Search timed out after 30 seconds. Try a more specific pattern."
    except Exception as e:
        return f"Error running {tool_name}: {e}"


def _find_in_files(
    directory: Path,
    pattern: str,
    file_pattern: str | None,
    context_lines: int,
) -> str:
    """Search for pattern within file contents with detailed results."""
    if not directory.exists():
        return f"Error: {directory} does not exist."
    
    if not directory.is_dir():
        return f"Error: {directory} is not a directory."

    results = []
    total_matches = 0
    files_searched = 0
    
    # Determine which files to search
    if file_pattern:
        files_to_search = list(directory.rglob(file_pattern))
    else:
        files_to_search = [f for f in directory.rglob("*") if f.is_file()]
    
    # Limit files to search
    if len(files_to_search) > 1000:
        files_to_search = files_to_search[:1000]
    
    for file_path in files_to_search:
        if not file_path.is_file():
            continue
        
        # Skip binary files and very large files
        try:
            if file_path.stat().st_size > 1024 * 1024:  # Skip files > 1MB
                continue
            
            content = file_path.read_text(errors="ignore")
            files_searched += 1
            
            if pattern in content:
                lines = content.split("\n")
                file_matches = []
                
                for i, line in enumerate(lines, 1):
                    if pattern in line:
                        total_matches += 1
                        # Get context lines
                        start = max(0, i - context_lines - 1)
                        end = min(len(lines), i + context_lines)
                        context = lines[start:end]
                        
                        # Format with line numbers
                        numbered = []
                        for j, ctx_line in enumerate(context, start + 1):
                            prefix = ">>> " if j == i else "    "
                            numbered.append(f"{prefix}{j:4}: {ctx_line}")
                        
                        file_matches.append("\n".join(numbered))
                
                if file_matches:
                    results.append(f"\n=== {file_path} ===\n" + "\n---\n".join(file_matches))
        except Exception:
            continue
        
        # Limit total results
        if total_matches > 50:
            break
    
    if not results:
        return f"No matches found for '{pattern}' in {directory} (searched {files_searched} files)"
    
    output = "\n".join(results)
    if total_matches > 50:
        output += f"\n\n[Results truncated after 50 matches]"
    
    return _truncate_output(output, 15000)
