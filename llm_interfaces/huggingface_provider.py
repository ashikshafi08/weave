import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any, Dict, List
from .base import BaseLLMProvider
import logging

logger = logging.getLogger(__name__)

class HuggingFaceProvider(BaseLLMProvider):
    def __init__(self, model_name: str):
        super().__init__()
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise

    async def generate_question(self, answer: Any, context: Dict[str, Any]) -> str:
        try:
            prompt = self.prompt_manager.render("question_generation", answer=answer, context=context)
            inputs = self.tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=50)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Error generating question: {str(e)}")
            raise

    async def validate_answer(self, question: str, proposed_answer: Any, correct_answer: Any) -> bool:
        try:
            prompt = self.prompt_manager.render("answer_validation", question=question, proposed_answer=proposed_answer, correct_answer=correct_answer)
            inputs = self.tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=5)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response.strip().lower() == "yes"
        except Exception as e:
            logger.error(f"Error validating answer: {str(e)}")
            raise

    async def evaluate(self, context: Dict[str, Any], criteria: Dict[str, Any]) -> Dict[str, Any]:
        try:
            prompt = self.prompt_manager.render("evaluation", context=context, criteria=criteria)
            inputs = self.tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=100)
            evaluation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return {"evaluation": evaluation}
        except Exception as e:
            logger.error(f"Error evaluating: {str(e)}")
            raise

    async def explain(self, evaluation_result: Dict[str, Any]) -> str:
        try:
            prompt = self.prompt_manager.render("explanation", evaluation_result=evaluation_result)
            inputs = self.tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=100)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Error explaining: {str(e)}")
            raise

    def get_supported_criteria(self) -> List[str]:
        return ["grammar", "coherence", "relevance", "creativity"]

    def get_model_info(self) -> Dict[str, Any]:
        return {"name": self.model.config.name_or_path, "provider": "Hugging Face"}