"""
Tests for the agent utils module.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.utils import (
    truncate_text,
    sanitize_filename,
    format_json_compact,
    count_tokens_approx,
    safe_get,
)


def test_truncate_text_short():
    """Test truncate_text with short text."""
    text = "Short text"
    result = truncate_text(text, max_length=100)
    assert result == text


def test_truncate_text_long():
    """Test truncate_text with long text."""
    text = "A" * 200
    result = truncate_text(text, max_length=50)
    assert len(result) <= 50
    assert result.endswith("...")


def test_truncate_text_custom_suffix():
    """Test truncate_text with custom suffix."""
    text = "A" * 200
    result = truncate_text(text, max_length=50, suffix=" [more]")
    assert len(result) <= 50
    assert result.endswith(" [more]")


def test_sanitize_filename_basic():
    """Test sanitize_filename with valid name."""
    result = sanitize_filename("valid_filename.txt")
    assert result == "valid_filename.txt"


def test_sanitize_filename_invalid_chars():
    """Test sanitize_filename with invalid characters."""
    result = sanitize_filename('file<name>:with/invalid\\chars.txt')
    assert '<' not in result
    assert '>' not in result
    assert ':' not in result
    assert '/' not in result
    assert '\\' not in result


def test_sanitize_filename_empty():
    """Test sanitize_filename with empty string."""
    result = sanitize_filename("")
    assert result == "unnamed"


def test_format_json_compact():
    """Test format_json_compact."""
    data = {"key": "value", "number": 42}
    result = format_json_compact(data)
    assert '"key":"value"' in result
    assert '"number":42' in result
    # Should be compact (no spaces after separators)
    assert ': ' not in result


def test_count_tokens_approx():
    """Test count_tokens_approx."""
    # Roughly 4 chars per token
    text = "A" * 100
    result = count_tokens_approx(text)
    assert result == 25  # 100 // 4


def test_safe_get_nested():
    """Test safe_get with nested keys."""
    data = {"a": {"b": {"c": "value"}}}
    result = safe_get(data, "a", "b", "c")
    assert result == "value"


def test_safe_get_missing():
    """Test safe_get with missing key."""
    data = {"a": {"b": {}}}
    result = safe_get(data, "a", "b", "c", default="default")
    assert result == "default"


def test_safe_get_not_dict():
    """Test safe_get when intermediate value is not a dict."""
    data = {"a": "not a dict"}
    result = safe_get(data, "a", "b", default="default")
    assert result == "default"


def test_safe_get_default_none():
    """Test safe_get with default=None."""
    data = {}
    result = safe_get(data, "missing")
    assert result is None


if __name__ == "__main__":
    # Run all tests
    test_functions = [
        test_truncate_text_short,
        test_truncate_text_long,
        test_truncate_text_custom_suffix,
        test_sanitize_filename_basic,
        test_sanitize_filename_invalid_chars,
        test_sanitize_filename_empty,
        test_format_json_compact,
        test_count_tokens_approx,
        test_safe_get_nested,
        test_safe_get_missing,
        test_safe_get_not_dict,
        test_safe_get_default_none,
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
