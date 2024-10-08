# weave/task_creators/qa_task_creator.py
from typing import Any, Dict, List
from weave.core.task_creator import LLMTaskCreator, task_creator_registry

class QATaskCreator(LLMTaskCreator):
    def generate_prompt(self, data: Any, context: Dict[str, Any]) -> str:
        return f"Generate a question based on the following text: {data}"

    def parse_response(self, response: str) -> Dict[str, Any]:
        return {"question": response, "answer": data}

    def get_supported_task_types(self) -> List[str]:
        return ["question_answering"]

task_creator_registry.register("qa", QATaskCreator)