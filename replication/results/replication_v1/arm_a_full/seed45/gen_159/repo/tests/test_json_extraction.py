"""Tests for JSON extraction functions in task_agent."""

import pytest
import json
from task_agent import (
    _extract_jsons,
    _extract_json_flexible,
    _repair_json,
    _extract_json_robust,
)


def test_extract_jsons_basic():
    """Test basic JSON extraction from tags."""
    text = '<json>{"key": "value"}</json>'
    result = _extract_jsons(text)
    assert result is not None
    assert len(result) == 1
    assert result[0] == {"key": "value"}


def test_extract_jsons_multiple():
    """Test extraction of multiple JSON objects."""
    text = '<json>{"a": 1}</json> some text <json>{"b": 2}</json>'
    result = _extract_jsons(text)
    assert result is not None
    assert len(result) == 2
    assert result[0] == {"a": 1}
    assert result[1] == {"b": 2}


def test_extract_jsons_no_tags():
    """Test extraction when no tags present."""
    text = '{"key": "value"}'
    result = _extract_jsons(text)
    assert result is None


def test_extract_jsons_invalid_json():
    """Test handling of invalid JSON in tags."""
    text = '<json>{invalid json}</json>'
    result = _extract_jsons(text)
    assert result is None


def test_extract_json_flexible_code_blocks():
    """Test extraction from markdown code blocks."""
    text = '```json\n{"key": "value"}\n```'
    result = _extract_json_flexible(text)
    assert result is not None
    assert len(result) == 1
    assert result[0] == {"key": "value"}


def test_extract_json_flexible_raw():
    """Test extraction from raw JSON."""
    text = 'Some text {"key": "value"} more text'
    result = _extract_json_flexible(text)
    assert result is not None
    assert len(result) >= 1


def test_repair_json_trailing_comma():
    """Test repairing trailing commas."""
    text = '{"a": 1, "b": 2,}'
    result = _repair_json(text)
    assert result is not None
    assert result == {"a": 1, "b": 2}


def test_repair_json_single_quotes():
    """Test repairing single quotes."""
    text = "{'key': 'value'}"
    result = _repair_json(text)
    assert result is not None
    assert result == {"key": "value"}


def test_repair_json_unescaped_newlines():
    """Test repairing unescaped newlines."""
    text = '{"key": "value\nwith newline"}'
    result = _repair_json(text)
    assert result is not None
    assert result["key"] == "value\\nwith newline"


def test_repair_json_missing_braces():
    """Test repairing missing closing braces."""
    text = '{"key": "value"'
    result = _repair_json(text)
    assert result is not None
    assert result == {"key": "value"}


def test_repair_json_empty():
    """Test handling of empty input."""
    result = _repair_json("")
    assert result is None
    result = _repair_json(None)
    assert result is None


def test_extract_json_robust_comprehensive():
    """Test comprehensive robust extraction."""
    # Test with valid JSON
    text = '<json>{"response": "correct", "reasoning": "test"}</json>'
    result = _extract_json_robust(text)
    assert result is not None
    assert len(result) == 1
    assert result[0]["response"] == "correct"


def test_extract_json_robust_with_repair():
    """Test robust extraction with repair."""
    text = 'Some response with broken JSON: {"response": "correct",}'
    result = _extract_json_robust(text)
    assert result is not None
    assert len(result) >= 1


def test_extract_json_robust_empty():
    """Test robust extraction with empty input."""
    result = _extract_json_robust("")
    assert result is None
    result = _extract_json_robust(None)
    assert result is None


def test_nested_json():
    """Test extraction of nested JSON structures."""
    nested = {"outer": {"inner": "value"}}
    text = f'<json>{json.dumps(nested)}</json>'
    result = _extract_jsons(text)
    assert result is not None
    assert result[0] == nested


def test_unicode_in_json():
    """Test handling of unicode in JSON."""
    text = '<json>{"key": "café 🎉"}</json>'
    result = _extract_jsons(text)
    assert result is not None
    assert result[0]["key"] == "café 🎉"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
