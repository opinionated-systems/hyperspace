"""
Tests for the LLM client module.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.llm_client import (
    _get_cache_key,
    get_cache_stats,
    clear_cache,
    set_cache_enabled,
)


def test_cache_key_deterministic():
    """Test that cache keys are deterministic for identical inputs."""
    key1 = _get_cache_key("model1", [{"role": "user", "content": "hello"}], 0.5)
    key2 = _get_cache_key("model1", [{"role": "user", "content": "hello"}], 0.5)
    assert key1 == key2


def test_cache_key_different_inputs():
    """Test that different inputs produce different cache keys."""
    key1 = _get_cache_key("model1", [{"role": "user", "content": "hello"}], 0.5)
    key2 = _get_cache_key("model1", [{"role": "user", "content": "world"}], 0.5)
    assert key1 != key2


def test_cache_key_different_models():
    """Test that different models produce different cache keys."""
    key1 = _get_cache_key("model1", [{"role": "user", "content": "hello"}], 0.5)
    key2 = _get_cache_key("model2", [{"role": "user", "content": "hello"}], 0.5)
    assert key1 != key2


def test_cache_key_different_temperatures():
    """Test that different temperatures produce different cache keys."""
    key1 = _get_cache_key("model1", [{"role": "user", "content": "hello"}], 0.5)
    key2 = _get_cache_key("model1", [{"role": "user", "content": "hello"}], 0.7)
    assert key1 != key2


def test_cache_stats_initial():
    """Test initial cache stats."""
    clear_cache()
    stats = get_cache_stats()
    assert stats["enabled"] == True
    assert stats["hits"] == 0
    assert stats["misses"] == 0
    assert stats["size"] == 0
    assert stats["hit_rate"] == 0.0


def test_cache_enable_disable():
    """Test enabling and disabling cache."""
    # Start enabled
    set_cache_enabled(True)
    stats = get_cache_stats()
    assert stats["enabled"] == True
    
    # Disable
    set_cache_enabled(False)
    stats = get_cache_stats()
    assert stats["enabled"] == False
    
    # Re-enable
    set_cache_enabled(True)
    stats = get_cache_stats()
    assert stats["enabled"] == True


def test_clear_cache():
    """Test clearing the cache."""
    # First disable and re-enable to ensure clean state
    set_cache_enabled(True)
    clear_cache()
    
    stats = get_cache_stats()
    assert stats["hits"] == 0
    assert stats["misses"] == 0
    assert stats["size"] == 0


if __name__ == "__main__":
    # Run all tests
    test_functions = [
        test_cache_key_deterministic,
        test_cache_key_different_inputs,
        test_cache_key_different_models,
        test_cache_key_different_temperatures,
        test_cache_stats_initial,
        test_cache_enable_disable,
        test_clear_cache,
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
