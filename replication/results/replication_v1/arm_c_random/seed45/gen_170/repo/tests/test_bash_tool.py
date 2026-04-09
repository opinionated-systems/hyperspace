"""
Tests for the bash tool.

Tests bash command execution and session persistence.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.tools.bash_tool import tool_function, reset_session, set_allowed_root


def test_basic_command():
    """Test basic command execution."""
    result = tool_function("echo Hello World")
    assert "Hello World" in result


def test_pwd():
    """Test pwd command."""
    result = tool_function("pwd")
    assert "/" in result  # Should contain a path


def test_empty_command():
    """Test empty command handling."""
    result = tool_function("")
    assert "empty" in result.lower()


def test_session_persistence_cd():
    """Test that cd persists across calls."""
    reset_session()
    
    # Change directory
    tool_function("cd /tmp")
    
    # Check that we're still in /tmp
    result = tool_function("pwd")
    assert "/tmp" in result
    
    reset_session()


def test_session_persistence_env():
    """Test that environment variables persist."""
    reset_session()
    
    # Set environment variable
    tool_function("export TEST_VAR=hello")
    
    # Check that it's still set
    result = tool_function("echo $TEST_VAR")
    assert "hello" in result
    
    reset_session()


def test_ls():
    """Test listing directory contents."""
    result = tool_function("ls /workspaces/hyperagents/replication/results/replication_v1/arm_c_random/seed45/gen_170/repo")
    assert "agent" in result or "tests" in result or "task_agent.py" in result


def test_sed_line_range():
    """Test sed for viewing line ranges."""
    result = tool_function("sed -n 1,5p /workspaces/hyperagents/replication/results/replication_v1/arm_c_random/seed45/gen_170/repo/agent/tools/bash_tool.py")
    assert "bash" in result.lower() or "Bash" in result
    # Should show line numbers or content from first 5 lines
    lines = result.split("\n")
    assert len(lines) <= 7  # Should be around 5 lines plus some output


def test_error_invalid_command():
    """Test error handling for invalid command."""
    result = tool_function("this_command_does_not_exist_12345")
    # Should return error message
    assert "error" in result.lower() or "not found" in result.lower() or "command" in result.lower()


if __name__ == "__main__":
    print("Running bash tool tests...")
    
    tests = [
        test_basic_command,
        test_pwd,
        test_empty_command,
        test_session_persistence_cd,
        test_session_persistence_env,
        test_ls,
        test_sed_line_range,
        test_error_invalid_command,
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
