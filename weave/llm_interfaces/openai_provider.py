import openai
from typing import Any, Dict, List
from weave.core.base import LLMProvider
from weave.core.registry import llm_provider_registry

class OpenAIProvider(LLMProvider):
    def __init__(self, model: str, api_key: str):
        self.model = model
        openai.api_key = api_key

    async def generate(self, prompt: str, **kwargs) -> str:
        response = await openai.ChatCompletion.acreate(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.choices[0].message.content

    async def evaluate(self, data: Dict[str, Any], criteria: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"Evaluate the following data based on these criteria: {criteria}\n\nData: {data}"
        response = await self.generate(prompt)
        return {"evaluation": response}

    def get_supported_criteria(self) -> List[str]:
        return ["grammar", "coherence", "relevance", "creativity"]

    def get_model_info(self) -> Dict[str, Any]:
        return {"name": self.model, "provider": "OpenAI"}

    def initialize(self, config: Dict[str, Any]) -> None:
        self.model = config.get('model', self.model)
        openai.api_key = config.get('api_key', openai.api_key)

llm_provider_registry.register("openai", OpenAIProvider)