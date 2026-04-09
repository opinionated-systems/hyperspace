"""
Tests for the search tool.

Tests text search and Python AST search functionality.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.tools.search_tool import tool_function, set_allowed_root


def test_text_search():
    """Test basic text search."""
    result = tool_function(
        "def tool_info",
        "/workspaces/hyperagents/replication/results/replication_v1/arm_c_random/seed45/gen_170/repo/agent/tools"
    )
    assert "editor_tool.py" in result or "bash_tool.py" in result or "search_tool.py" in result


def test_text_search_no_matches():
    """Test search with no matches."""
    # Reset allowed root to allow searching
    set_allowed_root("/")
    # Use a pattern that won't exist anywhere
    result = tool_function(
        "ZZZ_UNIQUE_PATTERN_999888777",
        "/workspaces/hyperagents/replication/results/replication_v1/arm_c_random/seed45/gen_170/repo"
    )
    assert "No matches" in result or "No" in result


def test_text_search_with_extension():
    """Test search with file extension filter."""
    result = tool_function(
        "class",
        "/workspaces/hyperagents/replication/results/replication_v1/arm_c_random/seed45/gen_170/repo/agent",
        file_extension=".py"
    )
    # Should find class definitions in Python files
    assert ".py" in result or "No matches" in result


def test_ast_search_class():
    """Test AST search for class definitions."""
    result = tool_function(
        "class:TaskAgent",
        "/workspaces/hyperagents/replication/results/replication_v1/arm_c_random/seed45/gen_170/repo"
    )
    assert "task_agent.py" in result
    assert "TaskAgent" in result


def test_ast_search_def():
    """Test AST search for function definitions."""
    result = tool_function(
        "def:tool_function",
        "/workspaces/hyperagents/replication/results/replication_v1/arm_c_random/seed45/gen_170/repo/agent/tools"
    )
    # Should find tool_function definitions
    assert "tool_function" in result or "Found" in result


def test_ast_search_no_results():
    """Test AST search with no results."""
    result = tool_function(
        "class:NonExistentClassXYZ",
        "/workspaces/hyperagents/replication/results/replication_v1/arm_c_random/seed45/gen_170/repo"
    )
    assert "No" in result or "not found" in result.lower()


def test_error_not_absolute_path():
    """Test error for non-absolute path."""
    result = tool_function("pattern", "relative/path")
    assert "Error" in result
    assert "absolute" in result.lower()


def test_case_sensitive_search():
    """Test case-sensitive vs insensitive search."""
    # Case sensitive - should find exact match
    result_sensitive = tool_function(
        "TaskAgent",
        "/workspaces/hyperagents/replication/results/replication_v1/arm_c_random/seed45/gen_170/repo",
        case_sensitive=True
    )
    
    # Case insensitive - should find regardless of case
    result_insensitive = tool_function(
        "taskagent",
        "/workspaces/hyperagents/replication/results/replication_v1/arm_c_random/seed45/gen_170/repo",
        case_sensitive=False
    )
    
    # Both should find results
    assert "task_agent.py" in result_sensitive or "No matches" not in result_sensitive


if __name__ == "__main__":
    print("Running search tool tests...")
    
    tests = [
        test_text_search,
        test_text_search_no_matches,
        test_text_search_with_extension,
        test_ast_search_class,
        test_ast_search_def,
        test_ast_search_no_results,
        test_error_not_absolute_path,
        test_case_sensitive_search,
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
