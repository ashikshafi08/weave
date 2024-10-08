# weave/core/framework.py
from typing import List, Dict, Any
from weave.core.base import TaskCreator, LLMProvider
from weave.core.config import Config
from weave.core.pipeline import Pipeline
from weave.data_providers.factory import create_data_provider
import logging

logger = logging.getLogger(__name__)

class SyntheticDataFramework:
    def __init__(self, config: Config):
        self.config = config
        self.data_provider = create_data_provider(config.get('data_provider'))
        self.task_creator = self._load_task_creator()
        self.llm_provider = self._load_llm_provider()
        self.pipeline = Pipeline(config, self.data_provider, self.task_creator, self.llm_provider)

    def _load_task_creator(self) -> TaskCreator:
        task_creator_name = self.config.get('task_creator.name')
        return task_creator_registry.get(task_creator_name)(self.llm_provider)

    def _load_llm_provider(self) -> LLMProvider:
        llm_provider_name = self.config.get('llm_provider.name')
        llm_provider_class = llm_provider_registry.get(llm_provider_name)
        return llm_provider_class(**self.config.get('llm_provider.params', {}))

    async def generate_dataset(self, num_samples: int) -> List[Dict[str, Any]]:
        logger.info(f"Generating dataset with {num_samples} samples")
        self.config.set('num_samples', num_samples)
        return await self.pipeline.run()

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

    def get_supported_criteria(self) -> List[str]:
        return self.llm_provider.get_supported_criteria()

    def get_model_info(self) -> Dict[str, Any]:
        return self.llm_provider.get_model_info()

    def add_pipeline_stage(self, stage_func):
        self.pipeline.add_stage(stage_func)

    def on_error(self, callback):
        self.pipeline.hook_manager.register_hook('on_error', callback)

    def after_data_generation(self, callback):
        self.pipeline.hook_manager.register_hook('after_data_generation', callback)

    def after_task_creation(self, callback):
        self.pipeline.hook_manager.register_hook('after_task_creation', callback)