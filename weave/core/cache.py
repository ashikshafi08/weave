# weave/core/cache.py

from typing import Any, Dict, Optional
from datetime import datetime, timedelta

class Cache:
    """
    Simple cache implementation for the Weave framework.
    """

    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        """
        Initialize the cache.

        Args:
            max_size (int): Maximum number of items in cache
            ttl (int): Time to live in seconds
        """
        self.max_size = max_size
        self.ttl = ttl
        self._cache: Dict[str, Dict[str, Any]] = {}

    def get(self, key: str) -> Optional[Any]:
        """
        Get item from cache.

        Args:
            key (str): Cache key

        Returns:
            Optional[Any]: Cached value if exists and not expired
        """
        if key not in self._cache:
            return None

        item = self._cache[key]
        if datetime.now() > item['expires']:
            del self._cache[key]
            return None

        return item['value']

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set item in cache.

        Args:
            key (str): Cache key
            value (Any): Value to cache
            ttl (Optional[int]): Custom TTL in seconds
        """
        if len(self._cache) >= self.max_size:
            # Remove oldest item
            oldest = min(self._cache.items(), key=lambda x: x[1]['expires'])
            del self._cache[oldest[0]]

        expires = datetime.now() + timedelta(seconds=ttl or self.ttl)
        self._cache[key] = {
            'value': value,
            'expires': expires
        }

    def clear(self) -> None:
        """Clear all items from cache."""
        self._cache.clear()