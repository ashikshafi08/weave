from typing import Any, Dict, List
from weave.core.task_creator import LLMTaskCreator, task_creator_registry

@task_creator_registry.register("qa")
class QATaskCreator(LLMTaskCreator):
    def __init__(self, llm_provider, prompt_manager):
        super().__init__(llm_provider, prompt_manager)
        self.prompt_manager.add_template(
            "task_creation",
            "Generate a question-answer pair based on the following text:\n{{data}}\n\nContext: {{context}}"
        )

    def parse_response(self, response: str) -> Dict[str, Any]:
        # Simple parsing, assuming the response is in the format "Q: ... A: ..."
        parts = response.split("A:")
        question = parts[0].replace("Q:", "").strip()
        answer = parts[1].strip() if len(parts) > 1 else ""
        return {"question": question, "answer": answer}

    def get_supported_task_types(self) -> List[str]:
        return ["question_answering"]