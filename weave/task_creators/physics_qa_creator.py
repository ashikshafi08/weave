from typing import Any, Dict, List
from weave.core.task_creator import LLMTaskCreator, task_creator_registry

@task_creator_registry.register("physics_qa")
class PhysicsQATaskCreator(LLMTaskCreator):
    def __init__(self, llm_provider, prompt_manager):
        super().__init__(llm_provider, prompt_manager)
        self.prompt_manager.add_template(
            "task_creation",
            """
            Based on the following physics-related text, generate a graduate-level question and its corresponding answer:

            Text: {{data}}

            Context: {{context}}

            Generate a question that requires deep understanding and analysis of advanced physics concepts. The question should be suitable for graduate-level physics students.

            Format your response as follows:
            Q: [Your generated question]
            A: [Your detailed answer]
            """
        )

    def parse_response(self, response: str) -> Dict[str, Any]:
        parts = response.split("A:")
        question = parts[0].replace("Q:", "").strip()
        answer = parts[1].strip() if len(parts) > 1 else ""
        return {"question": question, "answer": answer}

    def get_supported_task_types(self) -> List[str]:
        return ["physics_question_answering"]