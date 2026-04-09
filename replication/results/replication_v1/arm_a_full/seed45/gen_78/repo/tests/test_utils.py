"""
Tests for agent utilities.
"""

import pytest
from agent.utils import (
    truncate_string,
    safe_json_loads,
    format_dict_for_logging,
    validate_required_keys,
    merge_dicts,
)


def test_truncate_string_no_truncate():
    """Test that short strings are not truncated."""
    text = "Hello, World!"
    result = truncate_string(text, max_length=100)
    assert result == text


def test_truncate_string_truncate():
    """Test that long strings are truncated."""
    text = "a" * 1000
    result = truncate_string(text, max_length=100, indicator="...")
    assert len(result) <= 100
    assert "..." in result
    assert result.startswith("a")
    assert result.endswith("a")


def test_safe_json_loads_valid():
    """Test parsing valid JSON."""
    json_str = '{"key": "value", "number": 42}'
    result = safe_json_loads(json_str)
    assert result == {"key": "value", "number": 42}


def test_safe_json_loads_invalid():
    """Test parsing invalid JSON returns default."""
    json_str = '{invalid json}'
    result = safe_json_loads(json_str, default={})
    assert result == {}


def test_safe_json_loads_default_none():
    """Test parsing invalid JSON returns None by default."""
    json_str = '{invalid json}'
    result = safe_json_loads(json_str)
    assert result is None


def test_format_dict_for_logging():
    """Test formatting dictionary for logging."""
    data = {"key": "value", "number": 42}
    result = format_dict_for_logging(data)
    assert "key" in result
    assert "value" in result
    assert "42" in result


def test_format_dict_for_logging_truncation():
    """Test that long dictionaries are truncated."""
    data = {"key": "x" * 1000}
    result = format_dict_for_logging(data, max_length=100)
    assert len(result) <= 100


def test_validate_required_keys_all_present():
    """Test validation when all keys are present."""
    data = {"a": 1, "b": 2, "c": 3}
    missing = validate_required_keys(data, ["a", "b"])
    assert missing == []


def test_validate_required_keys_missing():
    """Test validation when keys are missing."""
    data = {"a": 1}
    missing = validate_required_keys(data, ["a", "b", "c"])
    assert "b" in missing
    assert "c" in missing
    assert "a" not in missing


def test_validate_required_keys_none_values():
    """Test that None values are treated as missing."""
    data = {"a": 1, "b": None}
    missing = validate_required_keys(data, ["a", "b"])
    assert "b" in missing


def test_merge_dicts():
    """Test merging dictionaries."""
    base = {"a": 1, "b": 2}
    override = {"b": 3, "c": 4}
    result = merge_dicts(base, override)
    assert result == {"a": 1, "b": 3, "c": 4}


def test_merge_dicts_no_mutation():
    """Test that merge doesn't mutate original dictionaries."""
    base = {"a": 1}
    override = {"b": 2}
    result = merge_dicts(base, override)
    assert base == {"a": 1}
    assert override == {"b": 2}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
