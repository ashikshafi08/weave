from abc import ABC, abstractmethod
from typing import Any, Dict, List
from weave.prompts.prompt_manager import PromptManager

class BaseLLMProvider(ABC):
    def __init__(self):
        self.prompt_manager = PromptManager()

    def set_prompt_template(self, prompt_type: str, template_string: str):
        self.prompt_manager.set_template(prompt_type, template_string)

    @abstractmethod
    async def generate_question(self, answer: Any, context: Dict[str, Any]) -> str:
        pass

    @abstractmethod
    async def validate_answer(self, question: str, proposed_answer: Any, correct_answer: Any) -> bool:
        pass

    @abstractmethod
    async def evaluate(self, context: Dict[str, Any], criteria: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def explain(self, evaluation_result: Dict[str, Any]) -> str:
        pass

    @abstractmethod
    def get_supported_criteria(self) -> List[str]:
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        pass