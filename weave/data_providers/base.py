from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class DataProvider(ABC):
    @abstractmethod
    async def fetch(self, **kwargs) -> Optional[Dict[str, Any]]:
        """Fetch a single data point."""
        pass

    @abstractmethod
    async def fetch_batch(self, batch_size: int, **kwargs) -> List[Dict[str, Any]]:
        """Fetch a batch of data points."""
        pass

    @abstractmethod
    def get_data_type(self) -> str:
        """Return the type of data this provider handles."""
        pass

    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict[str, Any]) -> 'DataProvider':
        """Create a data provider instance from a configuration dictionary."""
        pass