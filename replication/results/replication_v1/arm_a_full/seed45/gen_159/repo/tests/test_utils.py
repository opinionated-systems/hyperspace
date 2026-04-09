"""Tests for utility functions."""

import pytest
from agent.utils import (
    truncate_text,
    sanitize_filename,
    count_tokens_approx,
    format_error_message,
    safe_get,
    is_valid_json_key,
    normalize_whitespace,
    extract_code_blocks,
    validate_required_keys,
)


def test_truncate_text_no_truncation():
    """Test that short text is not truncated."""
    text = "Hello, world!"
    result = truncate_text(text, max_len=100)
    assert result == text


def test_truncate_text_truncation():
    """Test that long text is truncated with indicator."""
    text = "a" * 1000
    result = truncate_text(text, max_len=100)
    assert "... [truncated] ..." in result
    assert len(result) <= 100


def test_truncate_text_empty():
    """Test that empty text is handled."""
    assert truncate_text("") == ""
    assert truncate_text(None) is None


def test_sanitize_filename():
    """Test filename sanitization."""
    assert sanitize_filename("test.txt") == "test.txt"
    assert sanitize_filename("test<file>.txt") == "test_file_.txt"
    assert sanitize_filename("  .hidden  ") == ".hidden"
    assert sanitize_filename("") == "unnamed"


def test_count_tokens_approx():
    """Test token counting approximation."""
    assert count_tokens_approx("") == 0
    assert count_tokens_approx("test") == 1  # 4 chars = 1 token
    assert count_tokens_approx("a" * 100) == 25  # 100 chars = 25 tokens


def test_format_error_message():
    """Test error message formatting."""
    error = ValueError("Something went wrong")
    result = format_error_message(error, "test_function")
    assert "test_function" in result
    assert "ValueError" in result
    assert "Something went wrong" in result


def test_safe_get():
    """Test safe dictionary access."""
    data = {"key": "value", "num": 42}
    
    assert safe_get(data, "key") == "value"
    assert safe_get(data, "missing") is None
    assert safe_get(data, "missing", default="default") == "default"
    assert safe_get(data, "num", expected_type=int) == 42
    assert safe_get(data, "num", expected_type=str) == "42"


def test_is_valid_json_key():
    """Test JSON key validation."""
    assert is_valid_json_key("valid_key") is True
    assert is_valid_json_key("") is False
    assert is_valid_json_key(123) is False
    assert is_valid_json_key("key\x00with\x01control") is False


def test_normalize_whitespace():
    """Test whitespace normalization."""
    assert normalize_whitespace("  hello   world  ") == "hello world"
    assert normalize_whitespace("hello\n\n\tworld") == "hello world"
    assert normalize_whitespace("") == ""


def test_extract_code_blocks():
    """Test code block extraction."""
    text = """
Some text
```python
print("hello")
```
More text
```json
{"key": "value"}
```
"""
    blocks = extract_code_blocks(text)
    assert len(blocks) == 2
    assert 'print("hello")' in blocks[0]
    
    python_blocks = extract_code_blocks(text, language="python")
    assert len(python_blocks) == 1


def test_validate_required_keys():
    """Test required key validation."""
    data = {"a": 1, "b": 2}
    
    is_valid, missing = validate_required_keys(data, ["a", "b"])
    assert is_valid is True
    assert missing == []
    
    is_valid, missing = validate_required_keys(data, ["a", "b", "c"])
    assert is_valid is False
    assert "c" in missing


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
