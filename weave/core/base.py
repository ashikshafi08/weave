from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

class WeaveComponent(ABC):
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        pass

class DataGenerator(WeaveComponent):
    @abstractmethod
    async def generate(self, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        pass

    @abstractmethod
    def get_supported_types(self) -> List[str]:
        pass

    @abstractmethod
    async def load_dataset(self, dataset_path: str) -> None:
        pass

    @abstractmethod
    async def sample(self, n: int) -> List[Tuple[Any, Dict[str, Any]]]:
        pass

    @abstractmethod
    async def augment(self, data: Any, context: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        pass

class TaskCreator(WeaveComponent):
    @abstractmethod
    async def create_task(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_supported_task_types(self) -> List[str]:
        pass

class LLMProvider(WeaveComponent):
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        pass

    @abstractmethod
    async def evaluate(self, data: Dict[str, Any], criteria: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_supported_criteria(self) -> List[str]:
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        pass