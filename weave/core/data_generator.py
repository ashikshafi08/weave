# weave/weave/core/data_generator.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, List, Optional
from weave.core.plugin import PluginRegistry

class DataGenerator(ABC):
    @abstractmethod
    def generate(self, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Generate a data point and its context."""
        pass

    @abstractmethod
    def get_supported_types(self) -> List[str]:
        """Return a list of supported data types."""
        pass

    @abstractmethod
    def load_dataset(self, dataset_path: str) -> None:
        """Load a dataset from a given path."""
        pass

    @abstractmethod
    def sample(self, n: int) -> List[Tuple[Any, Dict[str, Any]]]:
        """Sample n data points from the dataset."""
        pass

    @abstractmethod
    def augment(self, data: Any, context: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """Augment a data point and its context."""
        pass

data_generator_registry = PluginRegistry()