# weave/weave/core/pipeline.py
import asyncio
from typing import List, Dict, Any, Callable
from weave.core.base import DataGenerator, TaskCreator, LLMProvider
from weave.core.config import WeaveConfig
from weave.core.hooks import HookManager

class Pipeline:
    def __init__(self, config: WeaveConfig, data_generator: DataGenerator, task_creator: TaskCreator, llm_provider: LLMProvider):
        self.config = config
        self.data_generator = data_generator
        self.task_creator = task_creator
        self.llm_provider = llm_provider
        self.hook_manager = HookManager()
        self.stages: List[Callable] = []

    def add_stage(self, stage_func: Callable):
        self.stages.append(stage_func)

    async def run(self) -> List[Dict[str, Any]]:
        dataset = []
        async for data_point in self._generate_data_points():
            dataset.append(data_point)
        return dataset

    async def _generate_data_points(self):
        for _ in range(self.config.num_samples):
            try:
                data_point = await self._process_data_point()
                yield data_point
            except Exception as e:
                self.hook_manager.call_hook('on_error', error=e)

    async def _process_data_point(self) -> Dict[str, Any]:
        data, context = await self.data_generator.generate()
        self.hook_manager.call_hook('after_data_generation', data=data, context=context)

        for stage in self.stages:
            data, context = await stage(data, context)
            self.hook_manager.call_hook('after_stage', stage=stage, data=data, context=context)

        task = await self.task_creator.create_task(data, context)
        self.hook_manager.call_hook('after_task_creation', task=task)

        return task