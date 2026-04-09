"""
Tests for agent utilities.
"""

import time

import pytest
from agent.utils import (
    truncate_string,
    safe_json_loads,
    format_dict_for_logging,
    validate_required_keys,
    merge_dicts,
    memoize_with_ttl,
    batch_process,
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


def test_memoize_with_ttl_basic():
    """Test basic memoization with TTL."""
    call_count = 0
    
    @memoize_with_ttl(maxsize=10, ttl_seconds=60.0)
    def expensive_function(x):
        nonlocal call_count
        call_count += 1
        return x * 2
    
    # First call should execute the function
    result1 = expensive_function(5)
    assert result1 == 10
    assert call_count == 1
    
    # Second call with same arg should use cache
    result2 = expensive_function(5)
    assert result2 == 10
    assert call_count == 1  # No additional call
    
    # Different arg should execute the function
    result3 = expensive_function(10)
    assert result3 == 20
    assert call_count == 2


def test_memoize_with_ttl_expiration():
    """Test that cache entries expire after TTL."""
    call_count = 0
    
    @memoize_with_ttl(maxsize=10, ttl_seconds=0.1)  # 100ms TTL
    def quick_function(x):
        nonlocal call_count
        call_count += 1
        return x * 3
    
    # First call
    result1 = quick_function(5)
    assert result1 == 15
    assert call_count == 1
    
    # Immediate second call should use cache
    result2 = quick_function(5)
    assert result2 == 15
    assert call_count == 1
    
    # Wait for TTL to expire
    time.sleep(0.15)
    
    # Call after TTL should re-execute
    result3 = quick_function(5)
    assert result3 == 15
    assert call_count == 2


def test_batch_process():
    """Test batch processing."""
    def double_batch(batch):
        return [x * 2 for x in batch]
    
    items = [1, 2, 3, 4, 5]
    results = batch_process(items, batch_size=2, processor=double_batch)
    assert results == [2, 4, 6, 8, 10]


def test_batch_process_single_batch():
    """Test batch processing with single batch."""
    def sum_batch(batch):
        return [sum(batch)]
    
    items = [1, 2, 3, 4, 5]
    results = batch_process(items, batch_size=10, processor=sum_batch)
    assert results == [15]


def test_batch_process_empty():
    """Test batch processing with empty list."""
    def process_batch(batch):
        return batch
    
    items = []
    results = batch_process(items, batch_size=2, processor=process_batch)
    assert results == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
