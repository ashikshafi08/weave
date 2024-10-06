from abc import ABC, abstractmethod
from typing import Any, Dict, List

class LLMJudge(ABC):
    @abstractmethod
    async def evaluate(self, context: Dict[str, Any], criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the given context based on the specified criteria."""
        pass

    @abstractmethod
    async def explain(self, evaluation_result: Dict[str, Any]) -> str:
        """Provide an explanation for the evaluation result."""
        pass

    @abstractmethod
    def get_supported_criteria(self) -> List[str]:
        """Return a list of supported evaluation criteria."""
        pass

class EvaluationPipeline:
    def __init__(self, judge: LLMJudge, preprocessors: List[callable] = None, postprocessors: List[callable] = None):
        self.judge = judge
        self.preprocessors = preprocessors or []
        self.postprocessors = postprocessors or []

    async def run(self, context: Dict[str, Any], criteria: Dict[str, Any]) -> Dict[str, Any]:
        for preprocessor in self.preprocessors:
            context, criteria = preprocessor(context, criteria)

        result = await self.judge.evaluate(context, criteria)

        for postprocessor in self.postprocessors:
            result = postprocessor(result)

        return result