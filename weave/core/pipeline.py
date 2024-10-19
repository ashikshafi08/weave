# weave/core/pipeline.py

import asyncio
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from .plugin_manager import PluginManager
from .exceptions import PipelineConfigurationError, PipelineExecutionError
from .data_source import DataSource
from .data_processor import DataProcessor
from .data_generator import DataGenerator
from .data_validator import DataValidator
from .llm_interface import LLMInterface
from .prompt_manager import PromptManager

class Pipeline(ABC):
    """
    Abstract base class for data generation pipelines.
    """

    def __init__(self, config: Dict[str, Any], plugin_manager: PluginManager):
        """
        Initialize the pipeline.

        Args:
            config (Dict[str, Any]): Configuration for the pipeline.
            plugin_manager (PluginManager): Plugin manager instance.

        Raises:
            PipelineConfigurationError: If the pipeline configuration is invalid.
        """
        try:
            self.config = config
            self.plugin_manager = plugin_manager
            self.stages: List[Any] = []
            self.prompt_manager: Optional[PromptManager] = None
            self.llm_interface: Optional[LLMInterface] = None
            self._create_stages()
            self._configure_prompt_manager()
            self._configure_llm_interface()
        except Exception as e:
            raise PipelineConfigurationError(f"Failed to initialize pipeline: {str(e)}")

    @abstractmethod
    def _create_stages(self) -> None:
        """
        Create the processing stages for the pipeline.

        Raises:
            PipelineConfigurationError: If the stage configuration is invalid.
        """
        pass

    def _configure_prompt_manager(self) -> None:
        """
        Configure the prompt manager for the pipeline.

        Raises:
            PipelineConfigurationError: If the prompt manager configuration is invalid.
        """
        try:
            prompt_manager_config = self.config.get('prompt_manager', {})
            prompt_manager_type = prompt_manager_config.get('type', 'default')
            prompt_manager_class = self.plugin_manager.get_component(f"{prompt_manager_type}_prompt_manager")
            self.prompt_manager = prompt_manager_class(prompt_manager_config)
        except Exception as e:
            raise PipelineConfigurationError(f"Failed to configure prompt manager: {str(e)}")

    def _configure_llm_interface(self) -> None:
        """
        Configure the LLM interface for the pipeline.

        Raises:
            PipelineConfigurationError: If the LLM interface configuration is invalid.
        """
        try:
            llm_config = self.config.get('llm_interface', {})
            llm_type = llm_config.get('type', 'default')
            llm_class = self.plugin_manager.get_component(f"{llm_type}_llm")
            self.llm_interface = llm_class(llm_config)
        except Exception as e:
            raise PipelineConfigurationError(f"Failed to configure LLM interface: {str(e)}")

    async def run(self, num_samples: int) -> List[Dict[str, Any]]:
        """
        Run the pipeline to generate data.

        Args:
            num_samples (int): Number of samples to generate.

        Returns:
            List[Dict[str, Any]]: Generated data.

        Raises:
            PipelineExecutionError: If an error occurs during pipeline execution.
        """
        try:
            data = await self._run_initial_stage(num_samples)
            total_stages = len(self.stages)
            
            for i, stage in enumerate(self.stages[1:], start=1):
                data = await self._run_stage(stage, data)
                self._report_progress(i, total_stages)
            
            return data
        except Exception as e:
            raise PipelineExecutionError(f"Pipeline execution failed: {str(e)}")

    async def _run_initial_stage(self, num_samples: int) -> List[Dict[str, Any]]:
        """
        Run the initial stage of the pipeline.

        Args:
            num_samples (int): Number of samples to generate.

        Returns:
            List[Dict[str, Any]]: Data generated by the initial stage.

        Raises:
            PipelineExecutionError: If the initial stage execution fails.
        """
        try:
            if isinstance(self.stages[0], DataSource):
                return await self.stages[0].fetch(num_samples)
            elif isinstance(self.stages[0], DataGenerator):
                return await self.stages[0].generate(num_samples)
            else:
                raise PipelineExecutionError("Invalid initial stage type")
        except Exception as e:
            raise PipelineExecutionError(f"Initial stage execution failed: {str(e)}")

    async def _run_stage(self, stage: Any, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run a single stage of the pipeline.

        Args:
            stage (Any): The stage to run.
            data (List[Dict[str, Any]]): Input data for the stage.

        Returns:
            List[Dict[str, Any]]: Processed data from the stage.

        Raises:
            PipelineExecutionError: If the stage execution fails.
        """
        try:
            if isinstance(stage, DataProcessor):
                return await stage.process(data)
            elif isinstance(stage, DataValidator):
                return [item for item in data if await stage.validate(item)]
            elif isinstance(stage, LLMInterface):
                return await asyncio.gather(*[self._process_llm_stage(item) for item in data])
            else:
                raise PipelineExecutionError(f"Unknown stage type: {type(stage)}")
        except Exception as e:
            raise PipelineExecutionError(f"Stage execution failed: {str(e)}")

    async def _process_llm_stage(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single item through the LLM stage.

        Args:
            item (Dict[str, Any]): Input item for the LLM.

        Returns:
            Dict[str, Any]: Processed item from the LLM.
        """
        prompt = self.prompt_manager.get_prompt("llm_stage", **item)
        llm_output = await self.llm_interface.generate(prompt)
        item['llm_output'] = llm_output
        return item

    def add_stage(self, stage: Any) -> None:
        """
        Add a new stage to the pipeline.

        Args:
            stage (Any): The stage to add.

        Raises:
            PipelineConfigurationError: If the stage cannot be added.
        """
        try:
            self.stages.append(stage)
        except Exception as e:
            raise PipelineConfigurationError(f"Failed to add stage: {str(e)}")

    def remove_stage(self, index: int) -> None:
        """
        Remove a stage from the pipeline.

        Args:
            index (int): The index of the stage to remove.

        Raises:
            PipelineConfigurationError: If the stage cannot be removed.
        """
        try:
            del self.stages[index]
        except IndexError:
            raise PipelineConfigurationError(f"Invalid stage index: {index}")
        except Exception as e:
            raise PipelineConfigurationError(f"Failed to remove stage: {str(e)}")

    def _report_progress(self, current_stage: int, total_stages: int) -> None:
        """
        Report the progress of the pipeline execution.

        Args:
            current_stage (int): The current stage number.
            total_stages (int): The total number of stages.
        """
        progress = (current_stage / total_stages) * 100
        print(f"Pipeline progress: {progress:.2f}% ({current_stage}/{total_stages} stages completed)")

class DefaultPipeline(Pipeline):
    def _create_stages(self) -> None:
        """
        Create the processing stages for the default pipeline.

        Raises:
            PipelineConfigurationError: If the stage configuration is invalid.
        """
        try:
            for stage_config in self.config.get('stages', []):
                stage_type = stage_config.get('type')
                stage_class = self.plugin_manager.get_component(stage_type)
                self.stages.append(stage_class(stage_config))
        except Exception as e:
            raise PipelineConfigurationError(f"Failed to create pipeline stages: {str(e)}")

async def main():
    plugin_manager = PluginManager()
    config = {
        'stages': [
            {'type': 'data_source', 'config': {}},
            {'type': 'data_processor', 'config': {}},
            {'type': 'llm_interface', 'config': {}},
            {'type': 'data_validator', 'config': {}},
            {'type': 'data_generator', 'config': {}}
        ],
        'prompt_manager': {'type': 'default', 'config': {}},
        'llm_interface': {'type': 'default', 'config': {}}
    }
    pipeline = DefaultPipeline(config, plugin_manager)
    
    try:
        result = await pipeline.run(num_samples=10)
        print(f"Generated {len(result)} valid samples")
    except Exception as e:
        print(f"Error running pipeline: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())