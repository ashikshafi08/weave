from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, List

class DataGenerator(ABC):
    @abstractmethod
    def generate(self, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Generate a data point and its context."""
        pass

    @abstractmethod
    def get_supported_types(self) -> List[str]:
        """Return a list of supported data types."""
        pass