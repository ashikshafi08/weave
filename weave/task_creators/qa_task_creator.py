from typing import Any, Dict, List
from weave.core.base import TaskCreator, LLMProvider
from weave.core.registry import task_creator_registry

class QATaskCreator(TaskCreator):
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider

    async def create_task(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"Generate a question based on the following text: {data}"
        question = await self.llm_provider.generate(prompt)
        return {"question": question, "answer": data, "context": context}

    def get_supported_task_types(self) -> List[str]:
        return ["question_answering"]

    def initialize(self, config: Dict[str, Any]) -> None:
        # Any additional initialization can be done here
        pass

task_creator_registry.register("qa", QATaskCreator)