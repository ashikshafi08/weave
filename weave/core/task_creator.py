# weave/weave/core/task_creator.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from weave.core.plugin import PluginRegistry


class TaskCreator(ABC):
    @abstractmethod
    def create_task(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create a task from a data point and its context."""
        pass

    @abstractmethod
    def get_supported_task_types(self) -> List[str]:
        """Return a list of supported task types."""
        pass


class LLMTaskCreator(TaskCreator):
    def __init__(self, llm_provider):
        self.llm_provider = llm_provider

    def create_task(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self.generate_prompt(data, context)
        response = self.llm_provider.generate(prompt)
        return self.parse_response(response)

    @abstractmethod
    def generate_prompt(self, data: Any, context: Dict[str, Any]) -> str:
        """Generate a prompt for the LLM based on the data and context."""
        pass

    @abstractmethod
    def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response into a task dictionary."""
        pass


task_creator_registry = PluginRegistry()
