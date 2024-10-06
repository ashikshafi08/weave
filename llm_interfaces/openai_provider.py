import openai
from typing import Any, Dict, List
from .base import BaseLLMProvider

class OpenAIProvider(BaseLLMProvider):
    def __init__(self, model: str, api_key: str):
        super().__init__()
        self.model = model
        openai.api_key = api_key

    async def generate_question(self, answer: Any, context: Dict[str, Any]) -> str:
        prompt = self.prompt_manager.render("question_generation", answer=answer, context=context)
        response = await openai.ChatCompletion.acreate(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    async def validate_answer(self, question: str, proposed_answer: Any, correct_answer: Any) -> bool:
        prompt = self.prompt_manager.render("answer_validation", question=question, proposed_answer=proposed_answer, correct_answer=correct_answer)
        response = await openai.ChatCompletion.acreate(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip().lower() == "yes"

    async def evaluate(self, context: Dict[str, Any], criteria: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self.prompt_manager.render("evaluation", context=context, criteria=criteria)
        response = await openai.ChatCompletion.acreate(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return {"evaluation": response.choices[0].message.content}

    async def explain(self, evaluation_result: Dict[str, Any]) -> str:
        prompt = self.prompt_manager.render("explanation", evaluation_result=evaluation_result)
        response = await openai.ChatCompletion.acreate(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    def get_supported_criteria(self) -> List[str]:
        return ["grammar", "coherence", "relevance", "creativity"]

    def get_model_info(self) -> Dict[str, Any]:
        return {"name": self.model, "provider": "OpenAI"}