from abc import ABC, abstractmethod
from typing import Any, Dict, List


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, task: Dict[str, Any], response: Any, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a response to a task based on given criteria."""
        pass


class LLMEvaluator(Evaluator):
    def __init__(self, llm_provider):
        self.llm_provider = llm_provider

    def evaluate(self, task: Dict[str, Any], response: Any, criteria: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self.generate_evaluation_prompt(task, response, criteria)
        llm_response = self.llm_provider.generate(prompt)
        return self.parse_evaluation_response(llm_response)

    @abstractmethod
    def generate_evaluation_prompt(self, task: Dict[str, Any], response: Any, criteria: Dict[str, Any]) -> str:
        """Generate a prompt for the LLM to evaluate the response."""
        pass

    @abstractmethod
    def parse_evaluation_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM's evaluation response into a structured format."""
        pass
