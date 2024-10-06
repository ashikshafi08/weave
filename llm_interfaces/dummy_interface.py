from weave.core.llm_interface import LLMInterface
from typing import Any, Dict

class DummyInterface(LLMInterface):
    def generate_question(self, answer: Any, context: Dict[str, Any]) -> str:
        if context["operation"] == "addition":
            return f"What is {context['operands'][0]} plus {context['operands'][1]}?"
        return "Unsupported question type"

    def validate_answer(self, question: str, proposed_answer: Any, correct_answer: Any) -> bool:
        return proposed_answer == correct_answer

    def get_model_info(self) -> Dict[str, Any]:
        return {"name": "DummyInterface", "version": "0.1"}