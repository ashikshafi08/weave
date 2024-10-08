# weave/weave/core/pipeline.py
from typing import List, Dict, Any
from weave.core.data_generator import DataGenerator
from weave.core.task_creator import TaskCreator
from weave.llm_interfaces.base import BaseLLMProvider
from weave.core.config import Config
import logging

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, config: Config, data_generator: DataGenerator, task_creator: TaskCreator,
                 llm_provider: BaseLLMProvider):
        self.config = config
        self.data_generator = data_generator
        self.task_creator = task_creator
        self.llm_provider = llm_provider
        self.stages = []

    def add_stage(self, stage_func):
        self.stages.append(stage_func)

    async def run(self, num_samples: int) -> List[Dict[str, Any]]:
        dataset = []
        for _ in range(num_samples):
            try:
                data_point = await self._process_data_point()
                dataset.append(data_point)
            except Exception as e:
                logger.error(f"Error processing data point: {str(e)}")
        return dataset

    async def _process_data_point(self) -> Dict[str, Any]:
        data, context = self.data_generator.generate()
        for stage in self.stages:
            data, context = await stage(data, context)
        task = self.task_creator.create_task(data, context)
        return task
