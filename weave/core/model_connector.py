from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import os
import json
import logging
import asyncio
from enum import Enum
from datetime import datetime, timedelta
import aiohttp
from .exceptions import ModelError, ModelConnectionError, ModelAPIError, ModelTokenLimitError

class ModelType(Enum):
    """Supported model types/providers."""
    OPENAI = "openai"
    AZURE = "azure"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"

class RateLimiter:
    """Rate limiter for API calls."""
    
    def __init__(self, calls_per_minute: int = 60):
        self.calls_per_minute = calls_per_minute
        self.calls = []
        self._lock = asyncio.Lock()
        
    async def acquire(self):
        """Acquire permission to make an API call."""
        async with self._lock:
            now = datetime.now()
            # Remove calls older than 1 minute
            self.calls = [t for t in self.calls if now - t < timedelta(minutes=1)]
            
            if len(self.calls) >= self.calls_per_minute:
                # Wait until oldest call is more than 1 minute old
                wait_time = 60 - (now - self.calls[0]).total_seconds()
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                self.calls = self.calls[1:]
            
            self.calls.append(now)

class ModelConnector:
    """Handles interactions with various LLM APIs and model endpoints.
    
    This class provides a unified interface for making requests to different
    LLM providers (OpenAI, Azure, Hugging Face, etc.) and handles common
    functionality like API key management, rate limiting, and error handling.
    """
    
    def __init__(
        self,
        model_type: Union[ModelType, str],
        model_name: str,
        api_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the model connector.
        
        Args:
            model_type: Type/provider of the model (e.g., "openai", "azure").
            model_name: Name of the specific model to use.
            api_key: Optional API key. If not provided, will look for environment variable.
            config: Additional configuration options.
        """
        self.model_type = ModelType(model_type) if isinstance(model_type, str) else model_type
        self.model_name = model_name
        self.config = config or {}
        
        # Set up API key
        self._api_key = api_key or self._get_api_key_from_env()
        
        # Initialize provider-specific client
        self._client = self._initialize_client()
        
        # Set up rate limiter
        self._rate_limiter = RateLimiter(
            calls_per_minute=self.config.get("calls_per_minute", 60)
        )
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Set up session for API calls
        self._session = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        self._session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._session:
            await self._session.close()
            
    def _get_api_key_from_env(self) -> Optional[str]:
        """Get API key from environment variables based on model type."""
        env_var_map = {
            ModelType.OPENAI: "OPENAI_API_KEY",
            ModelType.AZURE: "AZURE_OPENAI_API_KEY",
            ModelType.HUGGINGFACE: "HUGGINGFACE_API_KEY",
            ModelType.LOCAL: None
        }
        
        env_var = env_var_map.get(self.model_type)
        if not env_var:
            return None
            
        api_key = os.getenv(env_var)
        if not api_key:
            self.logger.warning(f"No API key found in environment variable: {env_var}")
        return api_key
        
    def _initialize_client(self) -> Any:
        """Initialize the appropriate client based on model type."""
        try:
            if self.model_type == ModelType.OPENAI:
                import openai
                openai.api_key = self._api_key
                return openai
                
            elif self.model_type == ModelType.AZURE:
                import openai
                openai.api_type = "azure"
                openai.api_key = self._api_key
                openai.api_base = self.config.get("api_base", "https://your-resource.openai.azure.com/")
                openai.api_version = self.config.get("api_version", "2023-05-15")
                return openai
                
            elif self.model_type == ModelType.HUGGINGFACE:
                from transformers import AutoTokenizer, AutoModelForCausalLM
                return {
                    "tokenizer": AutoTokenizer.from_pretrained(self.model_name),
                    "model": AutoModelForCausalLM.from_pretrained(self.model_name)
                }
                
            elif self.model_type == ModelType.LOCAL:
                # Implement connection to local models (e.g., via FastAPI endpoint)
                return None
                
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
                
        except Exception as e:
            raise ModelConnectionError(f"Failed to initialize {self.model_type} client: {str(e)}")
            
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate text from the model.
        
        Args:
            prompt: Input prompt/context.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative).
            **kwargs: Additional model-specific parameters.
            
        Returns:
            Generated text response.
            
        Raises:
            ModelError: If generation fails.
            ModelTokenLimitError: If prompt exceeds token limit.
            ModelAPIError: If API call fails.
        """
        try:
            # Check token limit
            token_count = self.get_token_count(prompt)
            model_token_limit = self.config.get("max_tokens", 4096)  # Default to 4096
            if token_count + max_tokens > model_token_limit:
                raise ModelTokenLimitError(
                    f"Total tokens ({token_count + max_tokens}) would exceed "
                    f"model limit ({model_token_limit})"
                )
                
            # Apply rate limiting
            await self._rate_limiter.acquire()
            
            if self.model_type == ModelType.OPENAI:
                response = await self._client.ChatCompletion.acreate(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs
                )
                return response.choices[0].message.content
                
            elif self.model_type == ModelType.AZURE:
                response = await self._client.ChatCompletion.acreate(
                    deployment_id=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs
                )
                return response.choices[0].message.content
                
            elif self.model_type == ModelType.HUGGINGFACE:
                inputs = self._client["tokenizer"](prompt, return_tensors="pt")
                outputs = self._client["model"].generate(
                    inputs["input_ids"],
                    max_length=max_tokens,
                    temperature=temperature,
                    **kwargs
                )
                return self._client["tokenizer"].decode(outputs[0], skip_special_tokens=True)
                
            elif self.model_type == ModelType.LOCAL:
                raise NotImplementedError("Local model inference not yet implemented")
                
        except ModelError:
            raise
        except Exception as e:
            self.logger.error(f"Error generating text: {str(e)}")
            raise ModelAPIError(f"Generation failed: {str(e)}")
            
    async def batch_generate(
        self,
        prompts: List[str],
        max_tokens: int = 100,
        temperature: float = 0.7,
        batch_size: int = 10,
        **kwargs
    ) -> List[str]:
        """Generate text for multiple prompts in parallel.
        
        Args:
            prompts: List of input prompts/contexts.
            max_tokens: Maximum number of tokens to generate per prompt.
            temperature: Sampling temperature.
            batch_size: Number of prompts to process in parallel.
            **kwargs: Additional model-specific parameters.
            
        Returns:
            List of generated text responses.
        """
        results = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            tasks = [
                self.generate(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs
                )
                for prompt in batch
            ]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions
            for result in batch_results:
                if isinstance(result, Exception):
                    self.logger.error(f"Batch generation error: {str(result)}")
                    results.append(None)
                else:
                    results.append(result)
                    
        return results
        
    def get_token_count(self, text: str) -> int:
        """Estimate the number of tokens in the text.
        
        Args:
            text: Input text to count tokens for.
            
        Returns:
            Estimated number of tokens.
        """
        try:
            if self.model_type in [ModelType.OPENAI, ModelType.AZURE]:
                import tiktoken
                encoding = tiktoken.encoding_for_model(self.model_name)
                return len(encoding.encode(text))
                
            elif self.model_type == ModelType.HUGGINGFACE:
                return len(self._client["tokenizer"].encode(text))
                
            else:
                # Rough estimate: words / 0.75 (typical tokens per word ratio)
                return int(len(text.split()) / 0.75)
                
        except Exception as e:
            self.logger.warning(f"Error counting tokens: {str(e)}")
            # Fallback to rough estimate
            return int(len(text.split()) / 0.75)
            
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration.
        
        Returns:
            Dictionary containing model information.
        """
        return {
            "model_type": self.model_type.value,
            "model_name": self.model_name,
            "config": self.config,
            "rate_limit": self._rate_limiter.calls_per_minute
        }
        
    def __repr__(self) -> str:
        """Return string representation of the model connector."""
        return f"ModelConnector(type={self.model_type.value}, model={self.model_name})"