"""OpenAI LLM provider for the Weave framework."""

import os
from typing import Any, Dict, List, Optional, Union
import json
import aiohttp
import asyncio
from ..core import ModelConnector, ModelType, ModelError

class OpenAILLM(ModelConnector):
    """OpenAI LLM provider using their API.
    
    This provider supports:
    - GPT-4 models
    - GPT-3.5 models
    - Text embeddings
    - Function calling
    """
    
    # Available models and their properties
    MODELS = {
        "gpt-4": {
            "max_tokens": 8192,
            "supports_functions": True,
            "cost_per_1k_tokens": 0.03,  # Input tokens
            "output_cost_per_1k_tokens": 0.06  # Output tokens
        },
        "gpt-4-32k": {
            "max_tokens": 32768,
            "supports_functions": True,
            "cost_per_1k_tokens": 0.06,
            "output_cost_per_1k_tokens": 0.12
        },
        "gpt-3.5-turbo": {
            "max_tokens": 4096,
            "supports_functions": True,
            "cost_per_1k_tokens": 0.0015,
            "output_cost_per_1k_tokens": 0.002
        },
        "text-embedding-ada-002": {
            "max_tokens": 8191,
            "supports_functions": False,
            "cost_per_1k_tokens": 0.0001,
            "output_cost_per_1k_tokens": 0.0
        }
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        organization: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the OpenAI LLM provider.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var).
            model: Model to use for generation.
            organization: Optional organization ID.
            base_url: Base URL for API requests.
            timeout: Timeout for API requests in seconds.
            max_retries: Maximum number of retries for failed requests.
            retry_delay: Delay between retries in seconds.
            config: Additional configuration options.
        """
        super().__init__(config)
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
            
        if model not in self.MODELS:
            raise ValueError(f"Unsupported model: {model}")
        self.model = model
        
        self.organization = organization
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Set up aiohttp session
        self.session = None
        self._setup_session()
        
        # Track usage for cost estimation
        self._token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        
    def _setup_session(self) -> None:
        """Set up the aiohttp session with headers."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
            
        self.session = aiohttp.ClientSession(
            base_url=self.base_url,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        
    async def _make_request(
        self,
        endpoint: str,
        payload: Dict[str, Any],
        stream: bool = False
    ) -> Dict[str, Any]:
        """Make an API request with retries.
        
        Args:
            endpoint: API endpoint to call.
            payload: Request payload.
            stream: Whether to stream the response.
            
        Returns:
            API response data.
            
        Raises:
            ModelError: If the request fails after retries.
        """
        if not self.session:
            self._setup_session()
            
        for attempt in range(self.max_retries):
            try:
                async with self.session.post(endpoint, json=payload) as response:
                    if stream:
                        return response  # Return response object for streaming
                        
                    if response.status == 200:
                        data = await response.json()
                        # Update token usage
                        if "usage" in data:
                            usage = data["usage"]
                            self._token_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                            self._token_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                            self._token_usage["total_tokens"] += usage.get("total_tokens", 0)
                        return data
                        
                    error_data = await response.json()
                    raise ModelError(
                        f"OpenAI API error: {error_data.get('error', {}).get('message', 'Unknown error')}"
                    )
                    
            except aiohttp.ClientError as e:
                if attempt == self.max_retries - 1:
                    raise ModelError(f"OpenAI API request failed: {str(e)}")
                    
                await asyncio.sleep(self.retry_delay * (2 ** attempt))
                
    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[Union[str, List[str]]] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Optional[Union[str, Dict[str, str]]] = None,
        stream: bool = False
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Generate text using the OpenAI API.
        
        Args:
            prompt: Input text to generate from.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0.0 to 2.0).
            top_p: Nucleus sampling parameter.
            frequency_penalty: Frequency penalty (-2.0 to 2.0).
            presence_penalty: Presence penalty (-2.0 to 2.0).
            stop: Stop sequences to end generation.
            functions: Optional function definitions for function calling.
            function_call: Optional function to call.
            stream: Whether to stream the response.
            
        Returns:
            Generated text or async generator for streaming.
            
        Raises:
            ModelError: If generation fails.
        """
        if not max_tokens:
            max_tokens = self.MODELS[self.model]["max_tokens"] - 100  # Leave room for prompt
            
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stream": stream
        }
        
        if stop:
            payload["stop"] = stop
            
        if functions and self.MODELS[self.model]["supports_functions"]:
            payload["functions"] = functions
            if function_call:
                payload["function_call"] = function_call
                
        if stream:
            response = await self._make_request("/chat/completions", payload, stream=True)
            return self._stream_response(response)
        else:
            response = await self._make_request("/chat/completions", payload)
            return response["choices"][0]["message"]["content"]
            
    async def _stream_response(self, response: aiohttp.ClientResponse) -> AsyncGenerator[str, None]:
        """Stream response chunks from the API.
        
        Args:
            response: aiohttp response object.
            
        Yields:
            Text chunks as they arrive.
        """
        async for line in response.content:
            if line:
                chunk = json.loads(line.decode("utf-8").strip("data: "))
                if chunk and "choices" in chunk and chunk["choices"]:
                    delta = chunk["choices"][0].get("delta", {})
                    if "content" in delta:
                        yield delta["content"]
                        
    async def get_embeddings(
        self,
        texts: Union[str, List[str]],
        model: str = "text-embedding-ada-002"
    ) -> Union[List[float], List[List[float]]]:
        """Get embeddings for text(s).
        
        Args:
            texts: Single text or list of texts.
            model: Embedding model to use.
            
        Returns:
            Single embedding vector or list of vectors.
            
        Raises:
            ModelError: If embedding generation fails.
        """
        if model not in self.MODELS or not model.startswith("text-embedding"):
            raise ValueError(f"Invalid embedding model: {model}")
            
        if isinstance(texts, str):
            texts = [texts]
            
        payload = {
            "model": model,
            "input": texts
        }
        
        response = await self._make_request("/embeddings", payload)
        embeddings = [data["embedding"] for data in response["data"]]
        
        return embeddings[0] if len(embeddings) == 1 else embeddings
        
    def get_token_usage(self) -> Dict[str, int]:
        """Get token usage statistics.
        
        Returns:
            Dictionary with prompt, completion and total token counts.
        """
        return self._token_usage.copy()
        
    def estimate_cost(self) -> float:
        """Estimate cost based on token usage.
        
        Returns:
            Estimated cost in USD.
        """
        model_info = self.MODELS[self.model]
        prompt_cost = (
            self._token_usage["prompt_tokens"] *
            model_info["cost_per_1k_tokens"] / 1000
        )
        completion_cost = (
            self._token_usage["completion_tokens"] *
            model_info["output_cost_per_1k_tokens"] / 1000
        )
        return prompt_cost + completion_cost
        
    async def close(self) -> None:
        """Close the aiohttp session."""
        if self.session:
            await self.session.close()
            self.session = None
            
    def __repr__(self) -> str:
        """Return string representation of the provider."""
        return (
            f"OpenAILLM(model={self.model}, "
            f"usage={self._token_usage}, "
            f"cost=${self.estimate_cost():.4f})"
        ) 