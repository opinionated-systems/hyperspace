"""
Tests for the editor tool.

Tests all editor commands: view, view_line, create, str_replace, insert, undo_edit.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.tools.editor_tool import tool_function, set_allowed_root


def test_view_directory():
    """Test viewing a directory."""
    result = tool_function("view", "/workspaces/hyperagents/replication/results/replication_v1/arm_c_random/seed45/gen_170/repo")
    assert "Files in" in result
    assert "agent" in result or "tests" in result


def test_view_file():
    """Test viewing a file."""
    result = tool_function("view", "/workspaces/hyperagents/replication/results/replication_v1/arm_c_random/seed45/gen_170/repo/agent/tools/editor_tool.py")
    assert "def tool_info" in result
    assert "def tool_function" in result


def test_view_file_range():
    """Test viewing a file with line range."""
    result = tool_function(
        "view",
        "/workspaces/hyperagents/replication/results/replication_v1/arm_c_random/seed45/gen_170/repo/agent/tools/editor_tool.py",
        view_range=[1, 20]
    )
    assert "tool_info" in result
    # Should only show first 20 lines
    lines = result.split("\n")
    # Count lines that have line numbers (they start with spaces and a number)
    numbered_lines = [l for l in lines if l.strip() and l[0:6].strip().isdigit()]
    assert len(numbered_lines) <= 20


def test_view_line():
    """Test viewing a specific line with context."""
    result = tool_function(
        "view_line",
        "/workspaces/hyperagents/replication/results/replication_v1/arm_c_random/seed45/gen_170/repo/agent/tools/editor_tool.py",
        line_number=50,
        context_lines=3
    )
    assert "Line 50" in result
    assert ">>>" in result  # Highlight marker
    # Should show context lines
    assert len(result.split("\n")) >= 5  # header + at least 3 context lines


def test_view_line_out_of_range():
    """Test viewing a line that's out of range."""
    result = tool_function(
        "view_line",
        "/workspaces/hyperagents/replication/results/replication_v1/arm_c_random/seed45/gen_170/repo/agent/tools/editor_tool.py",
        line_number=99999
    )
    assert "Error" in result
    assert "out of range" in result


def test_create_and_replace():
    """Test creating a file and replacing content."""
    with tempfile.TemporaryDirectory() as tmpdir:
        set_allowed_root(tmpdir)
        test_file = os.path.join(tmpdir, "test.txt")
        
        # Create file
        result = tool_function("create", test_file, file_text="Hello World")
        assert "created" in result
        
        # Verify file exists
        assert Path(test_file).exists()
        
        # Replace content
        result = tool_function(
            "str_replace",
            test_file,
            old_str="Hello World",
            new_str="Hello Universe"
        )
        assert "edited" in result
        
        # Verify content changed
        content = Path(test_file).read_text()
        assert content == "Hello Universe"


def test_insert():
    """Test inserting content."""
    with tempfile.TemporaryDirectory() as tmpdir:
        set_allowed_root(tmpdir)
        test_file = os.path.join(tmpdir, "test.txt")
        
        # Create file with multiple lines
        tool_function("create", test_file, file_text="Line 1\nLine 2\nLine 3")
        
        # Insert after line 1
        result = tool_function("insert", test_file, insert_line=1, new_str="Inserted line")
        assert "edited" in result
        
        # Verify insertion
        content = Path(test_file).read_text()
        lines = content.split("\n")
        assert lines[1] == "Inserted line"


def test_undo_edit():
    """Test undoing an edit."""
    with tempfile.TemporaryDirectory() as tmpdir:
        set_allowed_root(tmpdir)
        test_file = os.path.join(tmpdir, "test.txt")
        
        # Create file
        tool_function("create", test_file, file_text="Original")
        
        # Replace content
        tool_function("str_replace", test_file, old_str="Original", new_str="Modified")
        
        # Undo
        result = tool_function("undo_edit", test_file)
        assert "undone" in result
        
        # Verify content restored
        content = Path(test_file).read_text()
        assert content == "Original"


def test_error_not_absolute_path():
    """Test error for non-absolute path."""
    result = tool_function("view", "relative/path.txt")
    assert "Error" in result
    assert "absolute path" in result


def test_error_file_not_exist():
    """Test error for non-existent file."""
    # Reset allowed root to allow any path
    set_allowed_root("/")
    result = tool_function("view", "/nonexistent/path/file.txt")
    assert "Error" in result
    assert "does not exist" in result


def test_error_create_existing():
    """Test error when creating existing file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        set_allowed_root(tmpdir)
        test_file = os.path.join(tmpdir, "test.txt")
        
        # Create file
        tool_function("create", test_file, file_text="content")
        
        # Try to create again
        result = tool_function("create", test_file, file_text="new content")
        assert "Error" in result
        assert "already exists" in result


def test_error_str_replace_not_found():
    """Test error when old_str not found."""
    with tempfile.TemporaryDirectory() as tmpdir:
        set_allowed_root(tmpdir)
        test_file = os.path.join(tmpdir, "test.txt")
        
        # Create file
        tool_function("create", test_file, file_text="Hello World")
        
        # Try to replace non-existent string
        result = tool_function("str_replace", test_file, old_str="NonExistent", new_str="Replacement")
        assert "Error" in result
        assert "not found" in result


if __name__ == "__main__":
    print("Running editor tool tests...")
    
    tests = [
        test_view_directory,
        test_view_file,
        test_view_file_range,
        test_view_line,
        test_view_line_out_of_range,
        test_create_and_replace,
        test_insert,
        test_undo_edit,
        test_error_not_absolute_path,
        test_error_file_not_exist,
        test_error_create_existing,
        test_error_str_replace_not_found,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            print(f"  ✓ {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__}: {e}")
            failed += 1
    
    print(f"\n{passed} passed, {failed} failed")
