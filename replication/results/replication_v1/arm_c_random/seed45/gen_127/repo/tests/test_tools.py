"""
Tests for the agent tools.
"""

import sys
import os
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.tools.bash_tool import tool_function as bash, tool_info as bash_info, reset_session
from agent.tools.editor_tool import tool_function as editor, tool_info as editor_info
from agent.tools.search_tool import tool_function as search, tool_info as search_info


def test_bash_info():
    """Test bash tool info structure."""
    info = bash_info()
    assert info["name"] == "bash"
    assert "description" in info
    assert "input_schema" in info
    assert "command" in info["input_schema"]["properties"]


def test_editor_info():
    """Test editor tool info structure."""
    info = editor_info()
    assert info["name"] == "editor"
    assert "description" in info
    assert "input_schema" in info
    assert "command" in info["input_schema"]["properties"]
    assert "view" in info["input_schema"]["properties"]["command"]["enum"]


def test_search_info():
    """Test search tool info structure."""
    info = search_info()
    assert info["name"] == "search"
    assert "description" in info
    assert "input_schema" in info
    assert "pattern" in info["input_schema"]["properties"]


def test_bash_echo():
    """Test basic bash echo command."""
    reset_session()
    result = bash("echo 'Hello World'")
    assert "Hello World" in result


def test_bash_pwd():
    """Test bash pwd command."""
    reset_session()
    result = bash("pwd")
    assert result.strip()  # Should return a path


def test_editor_create_and_view():
    """Test editor create and view commands."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "test.txt")
        
        # Create file
        result = editor("create", test_file, file_text="Hello World")
        assert "created" in result.lower() or "File created" in result
        
        # View file
        result = editor("view", test_file)
        assert "Hello World" in result


def test_editor_str_replace():
    """Test editor str_replace command."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "test.txt")
        
        # Create file
        editor("create", test_file, file_text="Hello World")
        
        # Replace text
        result = editor("str_replace", test_file, old_str="World", new_str="Universe")
        assert "edited" in result.lower() or "File" in result
        
        # Verify replacement
        result = editor("view", test_file)
        assert "Hello Universe" in result


def test_editor_insert():
    """Test editor insert command."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "test.txt")
        
        # Create file with multiple lines
        editor("create", test_file, file_text="Line 1\nLine 2\nLine 3")
        
        # Insert after line 1
        result = editor("insert", test_file, insert_line=1, new_str="Inserted Line")
        assert "edited" in result.lower() or "File" in result
        
        # Verify insertion
        result = editor("view", test_file)
        assert "Inserted Line" in result


def test_editor_view_range():
    """Test editor view with range."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "test.txt")
        
        # Create file with multiple lines
        lines = [f"Line {i}" for i in range(1, 21)]
        editor("create", test_file, file_text="\n".join(lines))
        
        # View specific range
        result = editor("view", test_file, view_range=[5, 10])
        assert "Line 5" in result
        assert "Line 10" in result
        # Should not have lines outside range (check for numbered lines)
        # Line 1 might appear in the header, so we check for "     1\tLine 1"
        assert "     1\tLine 1" not in result
        assert "Line 20" not in result


def test_editor_relative_path_error():
    """Test that relative paths are rejected."""
    result = editor("view", "relative/path.txt")
    assert "Error" in result
    assert "absolute path" in result.lower()


def test_search_basic():
    """Test basic search functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        test_file1 = os.path.join(tmpdir, "file1.py")
        test_file2 = os.path.join(tmpdir, "file2.txt")
        
        with open(test_file1, "w") as f:
            f.write("def hello():\n    return 'world'\n")
        with open(test_file2, "w") as f:
            f.write("Just text\n")
        
        # Search for pattern
        result = search("hello", tmpdir)
        assert "hello" in result.lower() or "file1.py" in result


def test_search_with_extension():
    """Test search with file extension filter."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        test_file1 = os.path.join(tmpdir, "file1.py")
        test_file2 = os.path.join(tmpdir, "file2.txt")
        
        with open(test_file1, "w") as f:
            f.write("test content\n")
        with open(test_file2, "w") as f:
            f.write("test content\n")
        
        # Search with extension filter
        result = search("test", tmpdir, file_extension=".py")
        assert "file1.py" in result
        # Should not include .txt file
        assert "file2.txt" not in result


def test_search_no_matches():
    """Test search with no matches."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "file.txt")
        with open(test_file, "w") as f:
            f.write("some content\n")
        
        result = search("nonexistent_pattern", tmpdir)
        assert "No matches" in result or "not found" in result.lower()


if __name__ == "__main__":
    # Run all tests
    test_functions = [
        test_bash_info,
        test_editor_info,
        test_search_info,
        test_bash_echo,
        test_bash_pwd,
        test_editor_create_and_view,
        test_editor_str_replace,
        test_editor_insert,
        test_editor_view_range,
        test_editor_relative_path_error,
        test_search_basic,
        test_search_with_extension,
        test_search_no_matches,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            test_func()
            print(f"✓ {test_func.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_func.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__}: Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
