# llm_interface.py

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import asyncio
import time
from .exceptions import LLMError, RateLimitError
from .prompt_manager import PromptManager
from .cache import Cache

class LLMInterface(ABC):
    """
    Abstract base class for LLM interfaces.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.prompt_manager = PromptManager(config.get('prompt_manager', {}))
        self.cache = Cache()
        self.rate_limit = config.get('rate_limit', 60)  # requests per minute
        self.last_request_time = 0
        self.request_count = 0

    async def _rate_limit_check(self):
        current_time = time.time()
        if current_time - self.last_request_time >= 60:
            self.last_request_time = current_time
            self.request_count = 0
        self.request_count += 1
        if self.request_count > self.rate_limit:
            raise RateLimitError("Rate limit exceeded")

    @abstractmethod
    async def _api_call(self, prompt: str) -> str:
        """
        Make an API call to the LLM provider.

        Args:
            prompt (str): Input prompt for the LLM.

        Returns:
            str: Generated text.

        Raises:
            LLMError: If the API call fails.
        """
        pass

    async def generate(self, prompt: str) -> str:
        """
        Generate text using the LLM.

        Args:
            prompt (str): Input prompt for the LLM.

        Returns:
            str: Generated text.

        Raises:
            LLMError: If text generation fails.
            RateLimitError: If the rate limit is exceeded.
        """
        try:
            await self._rate_limit_check()
            cached_response = self.cache.get(prompt)
            if cached_response:
                return cached_response

            response = await self._api_call(prompt)
            self.cache.set(prompt, response)
            return response
        except RateLimitError as e:
            raise e
        except Exception as e:
            raise LLMError(f"Text generation failed: {str(e)}")

    async def validate(self, data: Dict[str, Any]) -> bool:
        """
        Validate generated data using the LLM.

        Args:
            data (Dict[str, Any]): Data to validate.

        Returns:
            bool: True if valid, False otherwise.

        Raises:
            LLMError: If validation fails.
            RateLimitError: If the rate limit is exceeded.
        """
        try:
            await self._rate_limit_check()
            validation_prompt = self.prompt_manager.get_prompt("validation", data=data)
            response = await self._api_call(validation_prompt)
            return "valid" in response.lower()
        except RateLimitError as e:
            raise e
        except Exception as e:
            raise LLMError(f"Data validation failed: {str(e)}")

    async def batch_generate(self, prompts: List[str], batch_size: int = 5) -> List[str]:
        """
        Generate text for multiple prompts in batches.

        Args:
            prompts (List[str]): List of input prompts.
            batch_size (int): Number of prompts to process in each batch.

        Returns:
            List[str]: List of generated texts.

        Raises:
            LLMError: If batch text generation fails.
        """
        try:
            results = []
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i:i+batch_size]
                batch_results = await asyncio.gather(*[self.generate(prompt) for prompt in batch])
                results.extend(batch_results)
            return results
        except Exception as e:
            raise LLMError(f"Batch text generation failed: {str(e)}")

class OpenAILLM(LLMInterface):
    async def _api_call(self, prompt: str) -> str:
        # Implement OpenAI API call here
        pass

class HuggingFaceLLM(LLMInterface):
    async def _api_call(self, prompt: str) -> str:
        # Implement Hugging Face API call here
        pass

class LLMFactory:
    @staticmethod
    def create_llm(llm_type: str, config: Dict[str, Any]) -> LLMInterface:
        """
        Create an LLMInterface instance based on the provided type and configuration.

        Args:
            llm_type (str): Type of LLM to create.
            config (Dict[str, Any]): Configuration for the LLM.

        Returns:
            LLMInterface: An instance of an LLMInterface subclass.

        Raises:
            ValueError: If an unsupported LLM type is specified.
        """
        if llm_type == 'openai':
            return OpenAILLM(config)
        elif llm_type == 'huggingface':
            return HuggingFaceLLM(config)
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")