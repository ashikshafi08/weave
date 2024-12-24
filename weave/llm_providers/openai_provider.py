from typing import Dict, Any, List, Optional
import asyncio
import time
import openai
from openai import AsyncOpenAI
from ..core.llm_interface import LLMInterface
from ..core.exceptions import LLMError, RateLimitError
from ..core.cache import Cache

class OpenAILLM(LLMInterface):
    """
    OpenAI LLM provider implementation.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the OpenAI LLM provider.

        Args:
            config (Dict[str, Any]): Configuration for the LLM provider.
                Required fields:
                - api_key: OpenAI API key
                - model: Model to use (e.g., "gpt-4", "gpt-3.5-turbo")
                Optional fields:
                - temperature: Sampling temperature (default: 0.7)
                - max_tokens: Maximum tokens to generate (default: 150)
                - rate_limit: Maximum requests per minute (default: 60)
        """
        super().__init__(config)
        self.client = AsyncOpenAI(api_key=config.get('api_key'))
        self.model = config.get('model', 'gpt-3.5-turbo')
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens = config.get('max_tokens', 150)
        self.cache = Cache(max_size=1000)  # Cache for responses
        self._last_request_time = 0
        self._request_count = 0
        self._lock = asyncio.Lock()

    async def _api_call(self, prompt: str) -> str:
        """
        Make an API call to OpenAI.

        Args:
            prompt (str): Input prompt for the LLM.

        Returns:
            str: Generated text.

        Raises:
            LLMError: If the API call fails.
            RateLimitError: If rate limit is exceeded.
        """
        try:
            # Check cache first
            cached_response = self.cache.get(prompt)
            if cached_response:
                return cached_response

            async with self._lock:
                # Rate limiting
                current_time = time.time()
                if current_time - self._last_request_time >= 60:
                    self._last_request_time = current_time
                    self._request_count = 0
                self._request_count += 1
                if self._request_count > self.rate_limit:
                    raise RateLimitError("OpenAI rate limit exceeded")

                # Make API call
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )

                # Extract and cache response
                generated_text = response.choices[0].message.content
                self.cache.set(prompt, generated_text)
                return generated_text

        except openai.RateLimitError:
            raise RateLimitError("OpenAI API rate limit exceeded")
        except openai.APIError as e:
            raise LLMError(f"OpenAI API error: {str(e)}")
        except Exception as e:
            raise LLMError(f"Unexpected error in OpenAI API call: {str(e)}")

    async def generate_batch(self, prompts: List[str], batch_size: int = 5) -> List[str]:
        """
        Generate responses for multiple prompts in batches.

        Args:
            prompts (List[str]): List of prompts to process.
            batch_size (int): Number of prompts to process in parallel.

        Returns:
            List[str]: List of generated responses.

        Raises:
            LLMError: If batch generation fails.
        """
        try:
            results = []
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i:i + batch_size]
                batch_results = await asyncio.gather(
                    *[self._api_call(prompt) for prompt in batch],
                    return_exceptions=True
                )
                
                # Handle any exceptions in the batch
                for result in batch_results:
                    if isinstance(result, Exception):
                        raise result
                    results.append(result)
                
                # Add a small delay between batches to help with rate limiting
                if i + batch_size < len(prompts):
                    await asyncio.sleep(1)
            
            return results
        except Exception as e:
            raise LLMError(f"Batch generation failed: {str(e)}")

    async def validate_response(self, prompt: str, response: str) -> bool:
        """
        Validate the quality of an LLM response.

        Args:
            prompt (str): Original prompt.
            response (str): Generated response.

        Returns:
            bool: True if response is valid, False otherwise.
        """
        validation_prompt = f"""
        Please validate the following response to the given prompt:
        
        Prompt: {prompt}
        Response: {response}
        
        Is this response:
        1. Relevant to the prompt?
        2. Well-formed and coherent?
        3. Free of harmful or inappropriate content?
        
        Answer with 'valid' or 'invalid' followed by a brief explanation.
        """
        
        try:
            validation_result = await self._api_call(validation_prompt)
            return validation_result.lower().startswith('valid')
        except Exception as e:
            raise LLMError(f"Response validation failed: {str(e)}")

    def get_token_count(self, text: str) -> int:
        """
        Estimate the number of tokens in the text.

        Args:
            text (str): Input text.

        Returns:
            int: Estimated token count.
        """
        # Simple estimation: ~4 characters per token
        return len(text) // 4 