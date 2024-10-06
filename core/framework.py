from typing import List, Dict, Any
from weave.core.data_generator import DataGenerator
from weave.llm_interfaces.base import BaseLLMProvider
from weave.prompts.prompt_manager import PromptManager
import logging

logger = logging.getLogger(__name__)

class SyntheticDataFramework:
    def __init__(self, data_generator: DataGenerator, llm_provider: BaseLLMProvider):
        self.data_generator = data_generator
        self.llm_provider = llm_provider
        self.prompt_manager = PromptManager()
        logger.info(f"Initialized SyntheticDataFramework with {data_generator.__class__.__name__} and {llm_provider.__class__.__name__}")

    def set_prompt_template(self, prompt_type: str, template: str):
        self.prompt_manager.set_template(prompt_type, template)
        self.llm_provider.set_prompt_template(prompt_type, template)
        logger.info(f"Set new prompt template for {prompt_type}")

    async def generate_dataset(self, num_samples: int, **kwargs) -> List[Dict[str, Any]]:
        logger.info(f"Generating dataset with {num_samples} samples")
        data_points = []
        for i in range(num_samples):
            try:
                answer, context = self.data_generator.generate(**kwargs)
                question = await self.llm_provider.generate_question(answer, context)
                data_points.append({"question": question, "answer": answer, "context": context})
                logger.debug(f"Generated sample {i+1}/{num_samples}")
            except Exception as e:
                logger.error(f"Error generating sample {i+1}: {str(e)}")
        logger.info(f"Dataset generation complete. Generated {len(data_points)} samples.")
        return data_points

    async def evaluate_dataset(self, dataset: List[Dict[str, Any]], criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        logger.info(f"Evaluating dataset with {len(dataset)} samples")
        evaluations = []
        for i, data_point in enumerate(dataset):
            try:
                evaluation = await self.llm_provider.evaluate(data_point, criteria)
                evaluations.append(evaluation)
                logger.debug(f"Evaluated sample {i+1}/{len(dataset)}")
            except Exception as e:
                logger.error(f"Error evaluating sample {i+1}: {str(e)}")
        logger.info("Dataset evaluation complete")
        return evaluations

    async def validate_dataset(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, bool]]:
        logger.info(f"Validating dataset with {len(dataset)} samples")
        validations = []
        for i, data_point in enumerate(dataset):
            try:
                is_valid = await self.llm_provider.validate_answer(
                    data_point['question'], data_point['answer'], data_point['answer']
                )
                validations.append({"question": data_point['question'], "is_valid": is_valid})
                logger.debug(f"Validated sample {i+1}/{len(dataset)}")
            except Exception as e:
                logger.error(f"Error validating sample {i+1}: {str(e)}")
        logger.info("Dataset validation complete")
        return validations

    def get_supported_criteria(self) -> List[str]:
        return self.llm_provider.get_supported_criteria()

    def get_model_info(self) -> Dict[str, Any]:
        return self.llm_provider.get_model_info()