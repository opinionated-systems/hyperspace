"""
Simple in-memory cache for LLM responses and other data.

Provides TTL-based caching to avoid redundant computations.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from typing import Any, Callable


class TTLCache:
    """Thread-safe in-memory cache with TTL support."""
    
    def __init__(self, default_ttl: float = 3600.0, max_size: int = 1000):
        """Initialize cache.
        
        Args:
            default_ttl: Default time-to-live in seconds
            max_size: Maximum number of entries (LRU eviction)
        """
        self._default_ttl = default_ttl
        self._max_size = max_size
        self._cache: dict[str, tuple[Any, float]] = {}
        self._lock = threading.RLock()
        self._access_times: dict[str, float] = {}
    
    def _make_key(self, *args, **kwargs) -> str:
        """Create a cache key from arguments."""
        key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Any | None:
        """Get value from cache if not expired."""
        with self._lock:
            if key not in self._cache:
                return None
            
            value, expiry = self._cache[key]
            if time.time() > expiry:
                del self._cache[key]
                del self._access_times[key]
                return None
            
            self._access_times[key] = time.time()
            return value
    
    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Set value in cache with optional custom TTL."""
        with self._lock:
            # Evict oldest if at capacity
            if len(self._cache) >= self._max_size and key not in self._cache:
                oldest = min(self._access_times, key=self._access_times.get)
                del self._cache[oldest]
                del self._access_times[oldest]
            
            expiry = time.time() + (ttl or self._default_ttl)
            self._cache[key] = (value, expiry)
            self._access_times[key] = time.time()
    
    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
    
    def invalidate(self, key: str) -> bool:
        """Remove a specific key from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                del self._access_times[key]
                return True
            return False
    
    def cached(self, ttl: float | None = None) -> Callable:
        """Decorator to cache function results.
        
        Args:
            ttl: Custom TTL for this function's cache entries
        """
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                # Create key from function name and arguments
                key_data = json.dumps({
                    "func": func.__name__,
                    "args": args,
                    "kwargs": kwargs
                }, sort_keys=True, default=str)
                key = hashlib.sha256(key_data.encode()).hexdigest()
                
                # Try cache first
                cached_value = self.get(key)
                if cached_value is not None:
                    return cached_value
                
                # Compute and cache
                result = func(*args, **kwargs)
                self.set(key, result, ttl)
                return result
            
            # Attach cache methods for introspection
            wrapper.cache_clear = lambda: self.clear()
            wrapper.cache_info = lambda: {
                "size": len(self._cache),
                "max_size": self._max_size,
                "default_ttl": self._default_ttl
            }
            
            return wrapper
        return decorator


# Global cache instance
_response_cache = TTLCache(default_ttl=300.0, max_size=500)  # 5 min TTL for LLM responses


def get_response_cache() -> TTLCache:
    """Get the global response cache."""
    return _response_cache


def cached_response(ttl: float = 300.0):
    """Decorator to cache LLM responses.
    
    Usage:
        @cached_response(ttl=600)
        def get_llm_response(prompt):
            ...
    """
    return _response_cache.cached(ttl)
