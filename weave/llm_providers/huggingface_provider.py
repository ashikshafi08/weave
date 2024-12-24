from typing import Dict, Any, List, Optional
import asyncio
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from ..core.llm_interface import LLMInterface
from ..core.exceptions import LLMError, RateLimitError
from ..core.cache import Cache

class HuggingFaceLLM(LLMInterface):
    """
    HuggingFace LLM provider implementation.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the HuggingFace LLM provider.

        Args:
            config (Dict[str, Any]): Configuration for the LLM provider.
                Required fields:
                - model_name: Name of the model to use (e.g., "google/flan-t5-base")
                Optional fields:
                - device: Device to use for inference ("cpu" or "cuda", default: "cuda" if available)
                - max_length: Maximum length of generated text (default: 128)
                - temperature: Sampling temperature (default: 0.7)
                - use_auth_token: HuggingFace auth token for private models
        """
        super().__init__(config)
        self.model_name = config['model_name']
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = config.get('max_length', 128)
        self.temperature = config.get('temperature', 0.7)
        self.use_auth_token = config.get('use_auth_token')
        self.cache = Cache(max_size=1000)
        self._initialize_model()

    def _initialize_model(self):
        """
        Initialize the model and tokenizer.

        Raises:
            LLMError: If model initialization fails.
        """
        try:
            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_auth_token=self.use_auth_token
            )

            # Initialize model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                use_auth_token=self.use_auth_token,
                device_map=self.device
            )

            # Create generation pipeline
            self.generator = pipeline(
                'text-generation',
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == 'cuda' else -1
            )
        except Exception as e:
            raise LLMError(f"Failed to initialize HuggingFace model: {str(e)}")

    async def _api_call(self, prompt: str) -> str:
        """
        Generate text using the HuggingFace model.

        Args:
            prompt (str): Input prompt for the LLM.

        Returns:
            str: Generated text.

        Raises:
            LLMError: If text generation fails.
        """
        try:
            # Check cache first
            cached_response = self.cache.get(prompt)
            if cached_response:
                return cached_response

            # Run generation in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.generator(
                    prompt,
                    max_length=self.max_length,
                    temperature=self.temperature,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            )

            generated_text = response[0]['generated_text']
            # Remove the prompt from the generated text
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()

            # Cache the response
            self.cache.set(prompt, generated_text)
            return generated_text

        except Exception as e:
            raise LLMError(f"HuggingFace text generation failed: {str(e)}")

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
            
            return results
        except Exception as e:
            raise LLMError(f"Batch generation failed: {str(e)}")

    def get_token_count(self, text: str) -> int:
        """
        Get the exact token count using the model's tokenizer.

        Args:
            text (str): Input text.

        Returns:
            int: Token count.
        """
        return len(self.tokenizer.encode(text))

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
        Validate the following response:
        Prompt: {prompt}
        Response: {response}
        Is this response valid? Answer with 'valid' or 'invalid'.
        """
        
        try:
            validation_result = await self._api_call(validation_prompt)
            return 'valid' in validation_result.lower()
        except Exception as e:
            raise LLMError(f"Response validation failed: {str(e)}")

    def cleanup(self):
        """
        Clean up resources used by the model.
        """
        try:
            del self.model
            del self.generator
            torch.cuda.empty_cache()
        except Exception as e:
            raise LLMError(f"Failed to clean up resources: {str(e)}")

class HuggingFaceInferenceAPI(LLMInterface):
    """
    Implementation using HuggingFace's Inference API for hosted models.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the HuggingFace Inference API client.

        Args:
            config (Dict[str, Any]): Configuration for the API.
                Required fields:
                - api_token: HuggingFace API token
                - model_name: Name of the hosted model
        """
        super().__init__(config)
        self.api_token = config['api_token']
        self.model_name = config['model_name']
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        self.cache = Cache(max_size=1000)

    async def _api_call(self, prompt: str) -> str:
        """
        Make an API call to HuggingFace's Inference API.

        Args:
            prompt (str): Input prompt for the LLM.

        Returns:
            str: Generated text.

        Raises:
            LLMError: If the API call fails.
        """
        try:
            # Check cache first
            cached_response = self.cache.get(prompt)
            if cached_response:
                return cached_response

            headers = {"Authorization": f"Bearer {self.api_token}"}
            payload = {"inputs": prompt}

            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, json=payload, headers=headers) as response:
                    response.raise_for_status()
                    result = await response.json()
                    
                    if isinstance(result, list) and len(result) > 0:
                        generated_text = result[0].get('generated_text', '')
                        self.cache.set(prompt, generated_text)
                        return generated_text
                    else:
                        raise LLMError("Invalid response format from Inference API")

        except aiohttp.ClientError as e:
            raise LLMError(f"HuggingFace Inference API request failed: {str(e)}")
        except Exception as e:
            raise LLMError(f"Unexpected error in HuggingFace Inference API call: {str(e)}") 