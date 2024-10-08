from typing import Dict, Type
from .base import DataProvider

class DataProviderRegistry:
    def __init__(self):
        self._providers: Dict[str, Type[DataProvider]] = {}

    def register(self, name: str, provider_class: Type[DataProvider]):
        self._providers[name] = provider_class

    def get(self, name: str) -> Type[DataProvider]:
        if name not in self._providers:
            raise ValueError(f"No data provider registered with name: {name}")
        return self._providers[name]

    def list_providers(self) -> List[str]:
        return list(self._providers.keys())

data_provider_registry = DataProviderRegistry()