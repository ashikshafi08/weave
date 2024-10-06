from vllm import LLM, SamplingParams
from typing import Any, Dict, List
from .base import BaseLLMProvider
import logging

logger = logging.getLogger(__name__)

class VLLMProvider(BaseLLMProvider):
    def __init__(self, model_name: str):
        super().__init__()
        try:
            self.llm = LLM(model=model_name)
            self.sampling_params = SamplingParams(temperature=0.7, top_p=0.95)
        except Exception as e:
            logger.error(f"Failed to initialize vLLM with model {model_name}: {str(e)}")
            raise

    async def generate_question(self, answer: Any, context: Dict[str, Any]) -> str:
        try:
            prompt = self.prompt_manager.render("question_generation", answer=answer, context=context)
            outputs = self.llm.generate([prompt], self.sampling_params)
            return outputs[0].outputs[0].text.strip()
        except Exception as e:
            logger.error(f"Error generating question: {str(e)}")
            raise

    async def validate_answer(self, question: str, proposed_answer: Any, correct_answer: Any) -> bool:
        try:
            prompt = self.prompt_manager.render("answer_validation", question=question, proposed_answer=proposed_answer, correct_answer=correct_answer)
            outputs = self.llm.generate([prompt], self.sampling_params)
            response = outputs[0].outputs[0].text.strip()
            return response.lower() == "yes"
        except Exception as e:
            logger.error(f"Error validating answer: {str(e)}")
            raise

    async def evaluate(self, context: Dict[str, Any], criteria: Dict[str, Any]) -> Dict[str, Any]:
        try:
            prompt = self.prompt_manager.render("evaluation", context=context, criteria=criteria)
            outputs = self.llm.generate([prompt], self.sampling_params)
            evaluation = outputs[0].outputs[0].text.strip()
            return {"evaluation": evaluation}
        except Exception as e:
            logger.error(f"Error evaluating: {str(e)}")
            raise

    async def explain(self, evaluation_result: Dict[str, Any]) -> str:
        try:
            prompt = self.prompt_manager.render("explanation", evaluation_result=evaluation_result)
            outputs = self.llm.generate([prompt], self.sampling_params)
            return outputs[0].outputs[0].text.strip()
        except Exception as e:
            logger.error(f"Error explaining: {str(e)}")
            raise

    def get_supported_criteria(self) -> List[str]:
        return ["grammar", "coherence", "relevance", "creativity"]

    def get_model_info(self) -> Dict[str, Any]:
        return {"name": self.llm.model_name, "provider": "vLLM"}