#!/usr/bin/env python3
"""
Simple test script to verify agent functionality.

Run with: python test_agent.py
"""

import sys
import os

# Add repo to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    from agent import (
        get_response_from_llm,
        get_response_from_llm_with_tools,
        set_audit_log,
        cleanup_clients,
        META_MODEL,
        EVAL_MODEL,
        chat_with_agent,
        DEFAULT_LLM_CONFIG,
        DEFAULT_AGENT_CONFIG,
        LLMConfig,
        AgentConfig,
        truncate_text,
        sanitize_filename,
        format_json_compact,
        count_tokens_approx,
        safe_get,
        retry_with_backoff,
        memoize_with_ttl,
        parse_json_safe,
        format_error_message,
        chunk_text,
        validate_path,
    )
    
    from agent.tools import load_tools, bash, bash_info, editor, editor_info, search, search_info
    from agent.tools.registry import load_tools as registry_load_tools
    from agent.tools.bash_tool import tool_function as bash_tool_function, tool_info as bash_tool_info
    from agent.tools.editor_tool import tool_function as editor_tool_function, tool_info as editor_tool_info, view_lines, get_file_summary
    from agent.tools.search_tool import tool_function as search_tool_function, tool_info as search_tool_info
    
    from task_agent import TaskAgent, _extract_jsons, _extract_response_fallback, _validate_inputs
    from meta_agent import MetaAgent
    
    print("✓ All imports successful")
    return True


def test_utils():
    """Test utility functions."""
    print("\nTesting utilities...")
    
    from agent.utils import (
        truncate_text,
        sanitize_filename,
        format_json_compact,
        count_tokens_approx,
        safe_get,
        parse_json_safe,
        format_error_message,
        chunk_text,
        validate_path,
    )
    
    # truncate_text
    assert truncate_text("hello world", 5) == "he..."
    assert truncate_text("hi", 10) == "hi"
    
    # sanitize_filename
    assert sanitize_filename("file<name>.txt") == "file_name_.txt"
    assert sanitize_filename("  .hidden  ") == "hidden"
    
    # format_json_compact
    assert format_json_compact({"a": 1, "b": 2}) == '{"a":1,"b":2}'
    
    # count_tokens_approx
    assert count_tokens_approx("") == 0
    assert count_tokens_approx("hello world") > 0
    
    # safe_get
    data = {"a": {"b": {"c": 1}}}
    assert safe_get(data, "a", "b", "c") == 1
    assert safe_get(data, "a", "x", default="default") == "default"
    
    # parse_json_safe
    assert parse_json_safe('{"key": "value"}') == {"key": "value"}
    assert parse_json_safe("invalid", default="fallback") == "fallback"
    
    # format_error_message
    error = ValueError("test")
    assert "ValueError" in format_error_message(error)
    assert "Context" in format_error_message(error, context="Context")
    
    # chunk_text
    text = "a" * 10000
    chunks = chunk_text(text, chunk_size=1000, overlap=100)
    assert len(chunks) > 1
    
    # validate_path
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "test.txt")
        with open(test_file, "w") as f:
            f.write("test content")
        
        # Test absolute path check
        is_valid, error = validate_path("relative/path")
        assert not is_valid
        assert "absolute path" in error
        
        # Test allowed root check
        is_valid, error = validate_path(test_file, allowed_root="/nonexistent")
        assert not is_valid
        assert "access denied" in error
        
        # Test must_exist check
        is_valid, error = validate_path(os.path.join(tmpdir, "nonexistent.txt"), must_exist=True)
        assert not is_valid
        assert "does not exist" in error
        
        # Test must_be_file check
        is_valid, error = validate_path(tmpdir, must_be_file=True)
        assert not is_valid
        assert "not a file" in error
        
        # Test must_be_dir check
        is_valid, error = validate_path(test_file, must_be_dir=True)
        assert not is_valid
        assert "not a directory" in error
        
        # Test valid file
        is_valid, error = validate_path(test_file, allowed_root=tmpdir, must_exist=True, must_be_file=True)
        assert is_valid
        assert error == ""
    
    print("✓ All utility tests passed")
    return True


def test_tools():
    """Test tool functions."""
    print("\nTesting tools...")
    
    from agent.tools.bash_tool import tool_function as bash
    from agent.tools.editor_tool import tool_function as editor, view_lines, get_file_summary
    from agent.tools.search_tool import tool_function as search
    
    # Test bash tool validation
    assert "Empty command" in bash("")
    assert "dangerous" in bash("rm -rf /").lower()
    assert "hello" in bash("echo hello")
    
    # Test editor tool validation
    assert "absolute path" in editor("view", "relative/path")
    
    # Test search tool validation
    assert "command is required" in search("", "pattern", "/tmp")
    assert "pattern is required" in search("grep", "", "/tmp")
    assert "absolute path" in search("grep", "test", "relative")
    
    # Test view_lines helper
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "test.txt")
        with open(test_file, "w") as f:
            f.write("line1\nline2\nline3\nline4\nline5\n")
        
        result = view_lines(test_file, 2, 4)
        assert "line2" in result
        assert "line3" in result
        assert "line4" in result
        
        # Test get_file_summary
        summary = get_file_summary(test_file)
        assert "File:" in summary
        assert "Size:" in summary
        assert "Lines:" in summary
        assert "line1" in summary
        # Note: file has 6 lines due to trailing newline
    
    print("✓ All tool tests passed")
    return True


def test_task_agent():
    """Test task agent functions."""
    print("\nTesting task agent...")
    
    from task_agent import _extract_jsons, _extract_response_fallback, _validate_inputs
    
    # Test _extract_jsons
    text = '<json>{"response": "test"}</json>'
    result = _extract_jsons(text)
    assert result == [{"response": "test"}]
    
    # Test _validate_inputs
    valid = {
        "domain": "math",
        "problem": "2+2=?",
        "solution": "4",
        "grading_guidelines": "Correct if 4",
        "student_answer": "4"
    }
    is_valid, error = _validate_inputs(valid)
    assert is_valid == True
    
    invalid = {"domain": "math"}
    is_valid, error = _validate_inputs(invalid)
    assert is_valid == False
    assert "Missing" in error
    
    print("✓ All task agent tests passed")
    return True


def test_agentic_loop():
    """Test agentic loop functions."""
    print("\nTesting agentic loop...")
    
    from agent.agentic_loop import _to_openai_tools, _execute_tool
    
    # Test _to_openai_tools
    tool_infos = [{
        "name": "test",
        "description": "Test tool",
        "input_schema": {"type": "object"}
    }]
    result = _to_openai_tools(tool_infos)
    assert result[0]["type"] == "function"
    
    # Test _execute_tool
    tools_dict = {
        "mock": {"function": lambda x: f"Result: {x}"}
    }
    result = _execute_tool(tools_dict, "mock", {"x": "test"})
    assert "Result: test" in result
    
    result = _execute_tool(tools_dict, "nonexistent", {})
    assert "not found" in result
    
    print("✓ All agentic loop tests passed")
    return True


def test_config():
    """Test configuration."""
    print("\nTesting configuration...")
    
    from agent.config import LLMConfig, AgentConfig, DEFAULT_LLM_CONFIG, DEFAULT_AGENT_CONFIG
    
    # Test LLMConfig
    config = LLMConfig()
    assert config.max_tokens > 0
    assert config.temperature >= 0
    
    # Test AgentConfig
    config = AgentConfig()
    assert config.max_tool_calls > 0
    assert config.bash_timeout > 0
    
    print("✓ All config tests passed")
    return True


def main():
    """Run all tests."""
    print("=" * 50)
    print("Running Agent Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_utils,
        test_tools,
        test_task_agent,
        test_agentic_loop,
        test_config,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
