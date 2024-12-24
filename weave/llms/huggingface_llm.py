"""Hugging Face LLM provider for the Weave framework."""

import os
from typing import Any, Dict, List, Optional, Union
import json
import aiohttp
import asyncio
from ..core import ModelConnector, ModelType, ModelError

class HuggingFaceLLM(ModelConnector):
    """Hugging Face LLM provider using their Inference API.
    
    This provider supports:
    - Text generation models
    - Text-to-text models
    - Embeddings models
    - Custom deployed models
    """
    
    # Default API endpoints
    INFERENCE_ENDPOINT = "https://api-inference.huggingface.co/models"
    
    def __init__(
        self,
        model_id: str,
        api_key: Optional[str] = None,
        task: str = "text-generation",
        base_url: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the Hugging Face LLM provider.
        
        Args:
            model_id: Hugging Face model ID (e.g., "gpt2", "t5-base").
            api_key: Hugging Face API token (defaults to HF_API_TOKEN env var).
            task: Model task (text-generation, text2text-generation, etc.).
            base_url: Optional custom API endpoint.
            timeout: Timeout for API requests in seconds.
            max_retries: Maximum number of retries for failed requests.
            retry_delay: Delay between retries in seconds.
            config: Additional configuration options.
        """
        super().__init__(config)
        
        self.api_key = api_key or os.getenv("HF_API_TOKEN")
        if not self.api_key:
            raise ValueError("Hugging Face API token is required")
            
        self.model_id = model_id
        self.task = task
        self.base_url = base_url or self.INFERENCE_ENDPOINT
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Set up aiohttp session
        self.session = None
        self._setup_session()
        
        # Track request usage
        self._request_count = 0
        self._total_generated_tokens = 0
        
    def _setup_session(self) -> None:
        """Set up the aiohttp session with headers."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        self.session = aiohttp.ClientSession(
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        
    async def _make_request(
        self,
        payload: Dict[str, Any],
        stream: bool = False
    ) -> Dict[str, Any]:
        """Make an API request with retries.
        
        Args:
            payload: Request payload.
            stream: Whether to stream the response.
            
        Returns:
            API response data.
            
        Raises:
            ModelError: If the request fails after retries.
        """
        if not self.session:
            self._setup_session()
            
        url = f"{self.base_url}/{self.model_id}"
        
        for attempt in range(self.max_retries):
            try:
                async with self.session.post(url, json=payload) as response:
                    if stream:
                        return response  # Return response object for streaming
                        
                    if response.status == 200:
                        data = await response.json()
                        self._request_count += 1
                        # Estimate token count from response length
                        if isinstance(data, list) and data:
                            text = data[0].get("generated_text", "")
                            self._total_generated_tokens += len(text.split())
                        return data
                        
                    error_data = await response.json()
                    raise ModelError(
                        f"Hugging Face API error: {error_data.get('error', 'Unknown error')}"
                    )
                    
            except aiohttp.ClientError as e:
                if attempt == self.max_retries - 1:
                    raise ModelError(f"Hugging Face API request failed: {str(e)}")
                    
                await asyncio.sleep(self.retry_delay * (2 ** attempt))
                
    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 50,
        num_return_sequences: int = 1,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Generate text using the Hugging Face model.
        
        Args:
            prompt: Input text to generate from.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0.0 to 2.0).
            top_p: Nucleus sampling parameter.
            top_k: Top-k sampling parameter.
            num_return_sequences: Number of sequences to return.
            stop: Stop sequences to end generation.
            stream: Whether to stream the response.
            
        Returns:
            Generated text or async generator for streaming.
            
        Raises:
            ModelError: If generation fails.
        """
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "num_return_sequences": num_return_sequences,
                "return_full_text": False
            }
        }
        
        if max_tokens:
            payload["parameters"]["max_new_tokens"] = max_tokens
            
        if stop:
            payload["parameters"]["stop_sequence"] = stop if isinstance(stop, str) else stop[0]
            
        if stream:
            response = await self._make_request(payload, stream=True)
            return self._stream_response(response)
        else:
            response = await self._make_request(payload)
            if isinstance(response, list) and response:
                return response[0]["generated_text"]
            return ""
            
    async def _stream_response(self, response: aiohttp.ClientResponse) -> AsyncGenerator[str, None]:
        """Stream response chunks from the API.
        
        Args:
            response: aiohttp response object.
            
        Yields:
            Text chunks as they arrive.
        """
        async for line in response.content:
            if line:
                try:
                    chunk = json.loads(line)
                    if isinstance(chunk, list) and chunk:
                        text = chunk[0].get("token", {}).get("text", "")
                        if text:
                            yield text
                except json.JSONDecodeError:
                    continue
                    
    async def get_embeddings(
        self,
        texts: Union[str, List[str]]
    ) -> Union[List[float], List[List[float]]]:
        """Get embeddings for text(s).
        
        Args:
            texts: Single text or list of texts.
            
        Returns:
            Single embedding vector or list of vectors.
            
        Raises:
            ModelError: If embedding generation fails.
        """
        if isinstance(texts, str):
            texts = [texts]
            
        payload = {
            "inputs": texts,
            "options": {
                "wait_for_model": True
            }
        }
        
        response = await self._make_request(payload)
        
        if isinstance(response, list):
            return response[0] if len(texts) == 1 else response
        return []
        
    def get_usage_stats(self) -> Dict[str, int]:
        """Get usage statistics.
        
        Returns:
            Dictionary with request count and estimated token count.
        """
        return {
            "request_count": self._request_count,
            "estimated_tokens": self._total_generated_tokens
        }
        
    async def close(self) -> None:
        """Close the aiohttp session."""
        if self.session:
            await self.session.close()
            self.session = None
            
    def __repr__(self) -> str:
        """Return string representation of the provider."""
        return (
            f"HuggingFaceLLM(model={self.model_id}, "
            f"task={self.task}, "
            f"requests={self._request_count})"
        ) 