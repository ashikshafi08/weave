from .core import LLMJudge
from typing import Any, Dict, List
import openai

class OpenAIAdapter(LLMJudge):
    def __init__(self, model: str, api_key: str):
        self.model = model
        openai.api_key = api_key

    async def evaluate(self, context: Dict[str, Any], criteria: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"Context: {context}\nCriteria: {criteria}\nEvaluation:"
        response = await openai.ChatCompletion.acreate(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return {"evaluation": response.choices[0].message.content}

    async def explain(self, evaluation_result: Dict[str, Any]) -> str:
        prompt = f"Evaluation result: {evaluation_result}\nExplanation:"
        response = await openai.ChatCompletion.acreate(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    def get_supported_criteria(self) -> List[str]:
        return ["grammar", "coherence", "relevance", "creativity"]

# Add more adapters for other LLM providers as needed