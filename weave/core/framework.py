# weave/core/framework.py

import asyncio
import logging
from typing import Dict, Any, List, Optional
from .pipeline import Pipeline
from .plugin_manager import PluginManager
from .data_validator import DataValidator
from .exceptions import ConfigurationError, PluginError, ValidationError
from .cache import Cache

class WeaveFramework:
    """
    Main class for the Weave framework, orchestrating the synthetic data generation process.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Weave framework.

        Args:
            config (Dict[str, Any]): Configuration dictionary for the framework.

        Raises:
            ConfigurationError: If the configuration is invalid.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        try:
            self.config = config
            self.plugin_manager = PluginManager()
            self.pipeline: Optional[Pipeline] = None
            self.validator: Optional[DataValidator] = None
            self.cache = Cache()
            self._configure()
        except Exception as e:
            self.logger.error(f"Failed to initialize WeaveFramework: {str(e)}")
            raise ConfigurationError(f"Failed to initialize WeaveFramework: {str(e)}")

    def _configure(self) -> None:
        """
        Configure the framework based on the provided configuration.

        Raises:
            ConfigurationError: If the configuration is invalid.
        """
        try:
            self.pipeline = self._create_pipeline()
            self.validator = self._create_validator()
            self.logger.info("WeaveFramework configured successfully")
        except Exception as e:
            self.logger.error(f"Failed to configure WeaveFramework: {str(e)}")
            raise ConfigurationError(f"Failed to configure WeaveFramework: {str(e)}")

    def _create_pipeline(self) -> Pipeline:
        """
        Create the data generation pipeline based on the configuration.

        Returns:
            Pipeline: The created pipeline instance.

        Raises:
            ConfigurationError: If the pipeline configuration is invalid.
        """
        try:
            pipeline_config = self.config.get('pipeline', {})
            pipeline_type = pipeline_config.get('type', 'default')
            cached_pipeline = self.cache.get(f"pipeline_{pipeline_type}")
            if cached_pipeline:
                self.logger.info(f"Using cached pipeline of type {pipeline_type}")
                return cached_pipeline
            
            pipeline_class = self.plugin_manager.get_pipeline(pipeline_type)
            pipeline = pipeline_class(pipeline_config, self.plugin_manager)
            self.cache.set(f"pipeline_{pipeline_type}", pipeline)
            self.logger.info(f"Created new pipeline of type {pipeline_type}")
            return pipeline
        except Exception as e:
            self.logger.error(f"Failed to create pipeline: {str(e)}")
            raise ConfigurationError(f"Failed to create pipeline: {str(e)}")

    def _create_validator(self) -> DataValidator:
        """
        Create the data validator based on the configuration.

        Returns:
            DataValidator: The created validator instance.

        Raises:
            ConfigurationError: If the validator configuration is invalid.
        """
        try:
            validator_config = self.config.get('validator', {})
            validator_type = validator_config.get('type', 'default')
            cached_validator = self.cache.get(f"validator_{validator_type}")
            if cached_validator:
                self.logger.info(f"Using cached validator of type {validator_type}")
                return cached_validator
            
            validator_class = self.plugin_manager.get_validator(validator_type)
            validator = validator_class(validator_config)
            self.cache.set(f"validator_{validator_type}", validator)
            self.logger.info(f"Created new validator of type {validator_type}")
            return validator
        except Exception as e:
            self.logger.error(f"Failed to create validator: {str(e)}")
            raise ConfigurationError(f"Failed to create validator: {str(e)}")

    async def generate_dataset(self, num_samples: int) -> List[Dict[str, Any]]:
        """
        Generate a synthetic dataset.

        Args:
            num_samples (int): Number of samples to generate.

        Returns:
            List[Dict[str, Any]]: Generated dataset.

        Raises:
            RuntimeError: If the pipeline or validator is not configured.
            ValidationError: If the generated data fails validation.
        """
        if not self.pipeline or not self.validator:
            self.logger.error("Pipeline or validator not configured")
            raise RuntimeError("Pipeline or validator not configured. Call set_config() first.")

        try:
            self.logger.info(f"Generating dataset with {num_samples} samples")
            raw_dataset = await self.pipeline.run(num_samples)
            validated_dataset = []

            async def validate_data(data: Dict[str, Any]) -> None:
                if await self.validator.validate(data):
                    validated_dataset.append(data)
                else:
                    self.logger.warning("Data sample failed validation")

            await asyncio.gather(*[validate_data(data) for data in raw_dataset])

            if len(validated_dataset) < num_samples:
                self.logger.error(f"Only {len(validated_dataset)} out of {num_samples} samples passed validation")
                raise ValidationError(f"Only {len(validated_dataset)} out of {num_samples} samples passed validation.")

            self.logger.info(f"Successfully generated and validated {len(validated_dataset)} samples")
            return validated_dataset
        except Exception as e:
            self.logger.error(f"Failed to generate dataset: {str(e)}")
            raise RuntimeError(f"Failed to generate dataset: {str(e)}")

    def add_plugin(self, plugin_name: str, plugin_instance: Any) -> None:
        """
        Add a plugin to the framework.

        Args:
            plugin_name (str): Name of the plugin.
            plugin_instance (Any): Instance of the plugin.

        Raises:
            PluginError: If the plugin cannot be added.
        """
        try:
            self.plugin_manager.register_plugin(plugin_name, plugin_instance)
            self.logger.info(f"Successfully added plugin: {plugin_name}")
        except Exception as e:
            self.logger.error(f"Failed to add plugin {plugin_name}: {str(e)}")
            raise PluginError(f"Failed to add plugin {plugin_name}: {str(e)}")

    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Update the framework configuration.

        Args:
            config (Dict[str, Any]): New configuration dictionary.

        Raises:
            ConfigurationError: If the new configuration is invalid.
        """
        try:
            self.config = config
            self._configure()
            self.logger.info("Successfully updated framework configuration")
        except Exception as e:
            self.logger.error(f"Failed to set new configuration: {str(e)}")
            raise ConfigurationError(f"Failed to set new configuration: {str(e)}")

# Example usage
async def main():
    logging.basicConfig(level=logging.INFO)
    config = {
        'pipeline': {'type': 'default'},
        'validator': {'type': 'default'}
    }
    framework = WeaveFramework(config)
    
    # Add custom plugins if needed
    # framework.add_plugin('custom_pipeline', CustomPipeline)
    
    try:
        dataset = await framework.generate_dataset(num_samples=100)
        print(f"Generated {len(dataset)} valid samples")
    except Exception as e:
        print(f"Error generating dataset: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())