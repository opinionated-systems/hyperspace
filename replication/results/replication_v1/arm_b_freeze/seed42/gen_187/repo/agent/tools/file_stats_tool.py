"""
File stats tool: analyze code files and provide detailed statistics.

Provides metrics like:
- Total lines, code lines, comment lines, blank lines
- Language detection
- File size and modification time
- Code-to-comment ratio
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "file_stats",
        "description": (
            "Analyze a file and return detailed statistics including "
            "line counts (total, code, comments, blank), language detection, "
            "file size, and modification time. Useful for understanding "
            "code structure and complexity."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file to analyze.",
                }
            },
            "required": ["path"],
        },
    }


# Language detection by extension
_LANGUAGE_MAP = {
    ".py": "Python",
    ".js": "JavaScript",
    ".ts": "TypeScript",
    ".jsx": "JSX",
    ".tsx": "TSX",
    ".java": "Java",
    ".c": "C",
    ".cpp": "C++",
    ".h": "C/C++ Header",
    ".hpp": "C++ Header",
    ".go": "Go",
    ".rs": "Rust",
    ".rb": "Ruby",
    ".php": "PHP",
    ".swift": "Swift",
    ".kt": "Kotlin",
    ".scala": "Scala",
    ".r": "R",
    ".m": "Objective-C/MATLAB",
    ".sh": "Shell",
    ".bash": "Bash",
    ".zsh": "Zsh",
    ".ps1": "PowerShell",
    ".pl": "Perl",
    ".lua": "Lua",
    ".sql": "SQL",
    ".html": "HTML",
    ".htm": "HTML",
    ".css": "CSS",
    ".scss": "SCSS",
    ".sass": "Sass",
    ".less": "Less",
    ".xml": "XML",
    ".json": "JSON",
    ".yaml": "YAML",
    ".yml": "YAML",
    ".toml": "TOML",
    ".ini": "INI",
    ".cfg": "Config",
    ".md": "Markdown",
    ".rst": "reStructuredText",
    ".txt": "Text",
    ".dockerfile": "Dockerfile",
    ".makefile": "Makefile",
    ".cmake": "CMake",
}

# Comment patterns by language
_COMMENT_PATTERNS = {
    "Python": (r"^\s*#", r"^\s*\"\"\"|^\s*'''"),
    "JavaScript": (r"^\s*//", r"^\s*/\*"),
    "TypeScript": (r"^\s*//", r"^\s*/\*"),
    "JSX": (r"^\s*//", r"^\s*/\*"),
    "TSX": (r"^\s*//", r"^\s*/\*"),
    "Java": (r"^\s*//", r"^\s*/\*"),
    "C": (r"^\s*//", r"^\s*/\*"),
    "C++": (r"^\s*//", r"^\s*/\*"),
    "C/C++ Header": (r"^\s*//", r"^\s*/\*"),
    "C++ Header": (r"^\s*//", r"^\s*/\*"),
    "Go": (r"^\s*//", r"^\s*/\*"),
    "Rust": (r"^\s*//", r"^\s*/\*"),
    "Ruby": (r"^\s*#", r"^\s*=begin"),
    "PHP": (r"^\s*//|^\s*#", r"^\s*/\*"),
    "Swift": (r"^\s*//", r"^\s*/\*"),
    "Kotlin": (r"^\s*//", r"^\s*/\*"),
    "Scala": (r"^\s*//", r"^\s*/\*"),
    "R": (r"^\s*#", None),
    "Shell": (r"^\s*#", None),
    "Bash": (r"^\s*#", None),
    "Zsh": (r"^\s*#", None),
    "PowerShell": (r"^\s*#", r"^\s*<#"),
    "Perl": (r"^\s*#", r"^\s*="),
    "Lua": (r"^\s*--", r"^\s*--\[\["),
    "SQL": (r"^\s*--", r"^\s*/\*"),
    "HTML": (None, r"^\s*<!--"),
    "CSS": (None, r"^\s*/\*"),
    "SCSS": (r"^\s*//", r"^\s*/\*"),
    "Sass": (r"^\s*//", r"^\s*/\*"),
    "Less": (r"^\s*//", r"^\s*/\*"),
    "XML": (None, r"^\s*<!--"),
    "YAML": (r"^\s*#", None),
    "TOML": (r"^\s*#", None),
    "INI": (r"^\s*;|^\s*#", None),
    "Config": (r"^\s*;|^\s*#", None),
    "Dockerfile": (r"^\s*#", None),
    "Makefile": (r"^\s*#", None),
}


def _detect_language(path: str) -> str:
    """Detect programming language from file extension."""
    ext = Path(path).suffix.lower()
    basename = Path(path).name.lower()
    
    # Check for special filenames
    if basename == "dockerfile" or basename.startswith("dockerfile."):
        return "Dockerfile"
    if basename in ("makefile", "gnumakefile"):
        return "Makefile"
    if basename == "cmakelists.txt":
        return "CMake"
    
    return _LANGUAGE_MAP.get(ext, "Unknown")


def _count_lines(content: str, language: str) -> dict:
    """Count different types of lines in code."""
    lines = content.splitlines()
    total = len(lines)
    
    blank = sum(1 for line in lines if not line.strip())
    
    # Get comment patterns for this language
    patterns = _COMMENT_PATTERNS.get(language, (None, None))
    single_pattern = patterns[0]
    multi_pattern = patterns[1]
    
    comment = 0
    in_multiline = False
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        
        # Check for multiline comment end
        if in_multiline:
            comment += 1
            if multi_pattern and re.search(r"\*/|-->|=end|--\]\]", stripped):
                in_multiline = False
            continue
        
        # Check for multiline comment start
        if multi_pattern and re.search(multi_pattern, stripped):
            comment += 1
            if not re.search(r"\*/.*\*/|<!--.*-->|<#.*#>", stripped):
                in_multiline = True
            continue
        
        # Check for single-line comment
        if single_pattern and re.search(single_pattern, stripped):
            comment += 1
            continue
    
    code = total - blank - comment
    
    return {
        "total": total,
        "code": max(0, code),
        "comment": comment,
        "blank": blank,
    }


def _format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def tool_function(path: str) -> str:
    """Analyze a file and return detailed statistics.
    
    Returns a formatted string with:
    - File information (name, path, language)
    - Size and modification time
    - Line counts (total, code, comments, blank)
    - Code metrics (code-to-comment ratio, average line length)
    """
    if not os.path.exists(path):
        return f"Error: File not found: {path}"
    
    if not os.path.isfile(path):
        return f"Error: Not a file: {path}"
    
    try:
        # Basic file info
        stat = os.stat(path)
        size_bytes = stat.st_size
        mtime = datetime.fromtimestamp(stat.st_mtime)
        
        language = _detect_language(path)
        
        # Read content
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception as e:
            return f"Error reading file: {e}"
        
        # Count lines
        counts = _count_lines(content, language)
        
        # Calculate metrics
        lines = content.splitlines()
        avg_line_length = sum(len(line) for line in lines) / max(1, len(lines))
        
        code_to_comment_ratio = "N/A"
        if counts["comment"] > 0:
            ratio = counts["code"] / counts["comment"]
            code_to_comment_ratio = f"{ratio:.1f}:1"
        elif counts["code"] > 0:
            code_to_comment_ratio = "∞ (no comments)"
        
        # Build output
        output = []
        output.append(f"File: {os.path.basename(path)}")
        output.append(f"Path: {path}")
        output.append(f"Language: {language}")
        output.append(f"Size: {_format_size(size_bytes)} ({size_bytes:,} bytes)")
        output.append(f"Modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        output.append("")
        output.append("Line Counts:")
        output.append(f"  Total:    {counts['total']:,}")
        output.append(f"  Code:     {counts['code']:,}")
        output.append(f"  Comments: {counts['comment']:,}")
        output.append(f"  Blank:    {counts['blank']:,}")
        output.append("")
        output.append("Metrics:")
        output.append(f"  Code/Comment Ratio: {code_to_comment_ratio}")
        output.append(f"  Avg Line Length: {avg_line_length:.1f} chars")
        
        # Complexity estimate
        if counts["code"] > 0:
            comment_pct = (counts["comment"] / counts["total"]) * 100
            output.append(f"  Comment Coverage: {comment_pct:.1f}%")
        
        return "\n".join(output)
        
    except Exception as e:
        return f"Error analyzing file: {e}"
