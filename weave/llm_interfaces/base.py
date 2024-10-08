# weave/weave/llm_interfaces/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from weave.core.plugin import PluginRegistry

class BaseLLMProvider(ABC):
    @abstractmethod
    async def generate(self, prompt: str) -> str:
        pass

    @abstractmethod
    async def evaluate(self, data_point: Dict[str, Any], criteria: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_supported_criteria(self) -> List[str]:
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        pass

    @classmethod
    def get_provider(cls, name: str):
        return llm_provider_registry.get(name)

llm_provider_registry = PluginRegistry()