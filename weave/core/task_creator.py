# weave/weave/core/task_creator.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from weave.core.plugin import PluginRegistry
from weave.prompts.template import PromptTemplateManager

class TaskCreator(ABC):
    def __init__(self, prompt_manager: PromptTemplateManager):
        self.prompt_manager = prompt_manager

    @abstractmethod
    async def create_task(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create a task from a data point and its context."""
        pass

    @abstractmethod
    def get_supported_task_types(self) -> List[str]:
        """Return a list of supported task types."""
        pass

class LLMTaskCreator(TaskCreator):
    def __init__(self, llm_provider, prompt_manager: PromptTemplateManager):
        super().__init__(prompt_manager)
        self.llm_provider = llm_provider

    async def create_task(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self.prompt_manager.render_template("task_creation", data=data, context=context)
        response = await self.llm_provider.generate(prompt)
        return self.parse_response(response)

    @abstractmethod
    def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response into a task dictionary."""
        pass

task_creator_registry = PluginRegistry()