from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import json
import re
from .exceptions import DataGenerationError

class DataGenerator(ABC):
    """
    Abstract base class for data generators.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data generator.

        Args:
            config (Dict[str, Any]): Configuration for the data generator.
        """
        self.config = config
        self.schema = config.get('schema', {})

    @abstractmethod
    def generate(self, llm_output: str) -> Dict[str, Any]:
        """
        Generate structured data from LLM output.

        Args:
            llm_output (str): Output from the LLM.

        Returns:
            Dict[str, Any]: Generated structured data.

        Raises:
            DataGenerationError: If data generation fails.
        """
        pass

    def validate_schema(self, data: Dict[str, Any]) -> bool:
        """
        Validate generated data against the schema.

        Args:
            data (Dict[str, Any]): Generated data.

        Returns:
            bool: True if valid, False otherwise.
        """
        for key, value_type in self.schema.items():
            if key not in data or not isinstance(data[key], value_type):
                return False
        return True

class JSONDataGenerator(DataGenerator):
    """
    Data generator for JSON-formatted LLM output.
    """

    def generate(self, llm_output: str) -> Dict[str, Any]:
        try:
            data = json.loads(llm_output)
            if not self.validate_schema(data):
                raise DataGenerationError("Generated data does not match the schema")
            return data
        except json.JSONDecodeError as e:
            raise DataGenerationError(f"Failed to parse JSON: {str(e)}")

class KeyValueDataGenerator(DataGenerator):
    """
    Data generator for key-value pair formatted LLM output.
    """

    def generate(self, llm_output: str) -> Dict[str, Any]:
        try:
            data = {}
            for line in llm_output.split('\n'):
                key, value = line.split(':', 1)
                data[key.strip()] = value.strip()
            
            if not self.validate_schema(data):
                raise DataGenerationError("Generated data does not match the schema")
            return data
        except ValueError as e:
            raise DataGenerationError(f"Failed to parse key-value pairs: {str(e)}")

class RegexDataGenerator(DataGenerator):
    """
    Data generator using regex patterns to extract structured data from LLM output.
    """

    def generate(self, llm_output: str) -> Dict[str, Any]:
        try:
            data = {}
            for key, pattern in self.config.get('patterns', {}).items():
                match = re.search(pattern, llm_output)
                if match:
                    data[key] = match.group(1)
            
            if not self.validate_schema(data):
                raise DataGenerationError("Generated data does not match the schema")
            return data
        except re.error as e:
            raise DataGenerationError(f"Failed to apply regex pattern: {str(e)}")

class DataGeneratorFactory:
    @staticmethod
    def create_data_generator(generator_type: str, config: Dict[str, Any]) -> DataGenerator:
        """
        Create a DataGenerator instance based on the provided type and configuration.

        Args:
            generator_type (str): Type of data generator to create.
            config (Dict[str, Any]): Configuration for the data generator.

        Returns:
            DataGenerator: An instance of a DataGenerator subclass.

        Raises:
            ValueError: If an unsupported generator type is specified.
        """
        if generator_type == 'json':
            return JSONDataGenerator(config)
        elif generator_type == 'key_value':
            return KeyValueDataGenerator(config)
        elif generator_type == 'regex':
            return RegexDataGenerator(config)
        else:
            raise ValueError(f"Unsupported data generator type: {generator_type}")