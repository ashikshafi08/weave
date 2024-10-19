# weave/core/data_processor.py

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Callable
from .exceptions import DataProcessingError
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import asyncio

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

class DataProcessor(ABC):
    """
    Abstract base class for data processors.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data processor.

        Args:
            config (Dict[str, Any]): Configuration for the data processor.
        """
        self.config = config

    @abstractmethod
    async def process(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process the input data.

        Args:
            data (List[Dict[str, Any]]): Input data to process.

        Returns:
            List[Dict[str, Any]]: Processed data.

        Raises:
            DataProcessingError: If data processing fails.
        """
        pass

    async def batch_process(self, data: List[Dict[str, Any]], batch_size: int = 100) -> List[Dict[str, Any]]:
        """
        Process data in batches.

        Args:
            data (List[Dict[str, Any]]): Input data to process.
            batch_size (int): Size of each batch.

        Returns:
            List[Dict[str, Any]]: Processed data.

        Raises:
            DataProcessingError: If batch processing fails.
        """
        try:
            processed_data = []
            for i in range(0, len(data), batch_size):
                batch = data[i:i+batch_size]
                processed_batch = await self.process(batch)
                processed_data.extend(processed_batch)
            return processed_data
        except Exception as e:
            raise DataProcessingError(f"Batch processing failed: {str(e)}")

class TextProcessor(DataProcessor):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    async def process(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process text data by cleaning, tokenizing, removing stop words, and lemmatizing.

        Args:
            data (List[Dict[str, Any]]): Input data to process.

        Returns:
            List[Dict[str, Any]]: Processed data with cleaned and normalized text.

        Raises:
            DataProcessingError: If text processing fails.
        """
        try:
            tasks = [self._process_item(item) for item in data]
            return await asyncio.gather(*tasks)
        except Exception as e:
            raise DataProcessingError(f"Text processing failed: {str(e)}")

    async def _process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        if 'text' in item:
            cleaned_text = self._clean_text(item['text'])
            tokens = self._tokenize(cleaned_text)
            tokens = self._remove_stop_words(tokens)
            lemmatized_tokens = self._lemmatize(tokens)
            item['processed_text'] = ' '.join(lemmatized_tokens)
        return item

    def _clean_text(self, text: str) -> str:
        """
        Clean the input text by removing special characters and extra whitespace.

        Args:
            text (str): Input text to clean.

        Returns:
            str: Cleaned text.
        """
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return ' '.join(text.split()).lower()

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize the input text.

        Args:
            text (str): Input text to tokenize.

        Returns:
            List[str]: List of tokens.
        """
        return word_tokenize(text)

    def _remove_stop_words(self, tokens: List[str]) -> List[str]:
        """
        Remove stop words from the list of tokens.

        Args:
            tokens (List[str]): List of tokens.

        Returns:
            List[str]: List of tokens with stop words removed.
        """
        return [token for token in tokens if token not in self.stop_words]

    def _lemmatize(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize the list of tokens.

        Args:
            tokens (List[str]): List of tokens to lemmatize.

        Returns:
            List[str]: List of lemmatized tokens.
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]

class NumericProcessor(DataProcessor):
    async def process(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process numeric data by scaling and normalizing.

        Args:
            data (List[Dict[str, Any]]): Input data to process.

        Returns:
            List[Dict[str, Any]]: Processed data with scaled and normalized numeric values.

        Raises:
            DataProcessingError: If numeric processing fails.
        """
        try:
            numeric_fields = self.config.get('numeric_fields', [])
            tasks = [self._process_item(item, numeric_fields) for item in data]
            return await asyncio.gather(*tasks)
        except Exception as e:
            raise DataProcessingError(f"Numeric processing failed: {str(e)}")

    async def _process_item(self, item: Dict[str, Any], numeric_fields: List[str]) -> Dict[str, Any]:
        for field in numeric_fields:
            if field in item and isinstance(item[field], (int, float)):
                scaled_value = self._scale_value(item[field])
                normalized_value = self._normalize_value(scaled_value)
                item[f'processed_{field}'] = normalized_value
        return item

    def _scale_value(self, value: float) -> float:
        """
        Scale the input value based on the configuration.

        Args:
            value (float): Input value to scale.

        Returns:
            float: Scaled value.
        """
        scale_factor = self.config.get('scale_factor', 1.0)
        return value * scale_factor

    def _normalize_value(self, value: float) -> float:
        """
        Normalize the input value to a range between 0 and 1.

        Args:
            value (float): Input value to normalize.

        Returns:
            float: Normalized value.
        """
        min_value = self.config.get('min_value', 0.0)
        max_value = self.config.get('max_value', 1.0)
        return (value - min_value) / (max_value - min_value)

class CustomProcessor(DataProcessor):
    def __init__(self, config: Dict[str, Any], custom_functions: List[Callable]):
        super().__init__(config)
        self.custom_functions = custom_functions

    async def process(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply custom processing functions to the data.

        Args:
            data (List[Dict[str, Any]]): Input data to process.

        Returns:
            List[Dict[str, Any]]: Processed data.

        Raises:
            DataProcessingError: If custom processing fails.
        """
        try:
            for func in self.custom_functions:
                if asyncio.iscoroutinefunction(func):
                    data = await func(data)
                else:
                    data = func(data)
            return data
        except Exception as e:
            raise DataProcessingError(f"Custom processing failed: {str(e)}")

class ProcessingPipeline:
    def __init__(self, processors: List[DataProcessor]):
        self.processors = processors

    async def process(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process data through the pipeline of processors.

        Args:
            data (List[Dict[str, Any]]): Input data to process.

        Returns:
            List[Dict[str, Any]]: Processed data.

        Raises:
            DataProcessingError: If pipeline processing fails.
        """
        try:
            for processor in self.processors:
                data = await processor.process(data)
            return data
        except Exception as e:
            raise DataProcessingError(f"Pipeline processing failed: {str(e)}")

class DataProcessorFactory:
    @staticmethod
    def create_processor(processor_type: str, config: Dict[str, Any]) -> DataProcessor:
        """
        Create a DataProcessor instance based on the provided type and configuration.

        Args:
            processor_type (str): Type of processor to create.
            config (Dict[str, Any]): Configuration for the processor.

        Returns:
            DataProcessor: An instance of a DataProcessor subclass.

        Raises:
            ValueError: If an unsupported processor type is specified.
        """
        if processor_type == 'text':
            return TextProcessor(config)
        elif processor_type == 'numeric':
            return NumericProcessor(config)
        elif processor_type == 'custom':
            custom_functions = config.get('custom_functions', [])
            return CustomProcessor(config, custom_functions)
        else:
            raise ValueError(f"Unsupported processor type: {processor_type}")