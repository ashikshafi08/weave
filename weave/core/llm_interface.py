from abc import ABC, abstractmethod
from typing import Any, Dict

class LLMInterface(ABC):
    @abstractmethod
    def generate_question(self, answer: Any, context: Dict[str, Any]) -> str:
        """Generate a question based on the answer and context."""
        pass

    @abstractmethod
    def validate_answer(self, question: str, proposed_answer: Any, correct_answer: Any) -> bool:
        """Validate if the proposed answer matches the correct answer for the given question."""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return information about the LLM model being used."""
        pass