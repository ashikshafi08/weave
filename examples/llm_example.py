import asyncio
import os
from dotenv import load_dotenv
from weave.llm_providers.factory import LLMProviderFactory
from weave.core.exceptions import LLMError
import json

# Load environment variables from .env file
load_dotenv()

# Example configurations
OPENAI_CONFIG = {
    'type': 'openai',
    'api_key': os.getenv('OPENAI_API_KEY'),
    'model': 'gpt-4-1106-preview',  # Latest GPT-4 Turbo
    'temperature': 0.7,
    'max_tokens': 4096,
    'rate_limit': 60,
    'cache': {
        'enabled': True,
        'max_size': 1000,
        'ttl': 3600
    },
    'advanced': {
        'response_format': {"type": "json"},
        'seed': 42,
        'frequency_penalty': 0.0,
        'presence_penalty': 0.0
    }
}

HUGGINGFACE_CONFIG = {
    'type': 'huggingface',
    'model_name': 'mistralai/Mixtral-8x7B-Instruct-v0.1',  # Latest Mixtral
    'device': 'cuda',
    'max_length': 4096,
    'temperature': 0.7,
    'cache': {
        'enabled': True,
        'max_size': 1000,
        'ttl': 3600
    },
    'advanced': {
        'quantization': 4,
        'trust_remote_code': True,
        'torch_dtype': 'bfloat16',
        'load_in_4bit': True,
        'use_flash_attention_2': True
    }
}

LLAMA2_CONFIG = {
    'type': 'huggingface_api',
    'api_token': os.getenv('HUGGINGFACE_API_TOKEN'),
    'model_name': 'meta-llama/Llama-2-70b-chat-hf',
    'cache': {
        'enabled': True,
        'max_size': 1000,
        'ttl': 3600
    },
    'advanced': {
        'wait_for_model': True,
        'use_cache': True,
        'max_retries': 3
    }
}

async def generate_structured_data(provider, prompt: str) -> dict:
    """
    Generate structured data using the specified provider.

    Args:
        provider: LLM provider instance
        prompt (str): Input prompt

    Returns:
        dict: Generated structured data
    """
    try:
        response = await provider._api_call(prompt)
        return json.loads(response) if isinstance(response, str) else response
    except json.JSONDecodeError:
        print("Warning: Response is not valid JSON")
        return {"error": "Invalid JSON response", "raw_response": response}
    except LLMError as e:
        print(f"Error generating text: {e}")
        return {"error": str(e)}

async def generate_with_streaming(provider, prompt: str):
    """
    Generate text with streaming response.

    Args:
        provider: LLM provider instance
        prompt (str): Input prompt
    """
    try:
        async for chunk in provider.stream(prompt):
            print(chunk, end='', flush=True)
        print()
    except LLMError as e:
        print(f"Error in streaming generation: {e}")

async def main():
    # Test OpenAI GPT-4 Turbo
    try:
        openai_provider = LLMProviderFactory.create_provider(OPENAI_CONFIG)
        print("\nTesting GPT-4 Turbo:")
        
        # Generate structured data
        prompt = """Generate a JSON object describing a machine learning model with the following fields:
        - name
        - type
        - parameters
        - performance_metrics
        """
        response = await generate_structured_data(openai_provider, prompt)
        print(f"\nStructured Response:\n{json.dumps(response, indent=2)}")

        # Batch generation with specific topics
        prompts = [
            "Explain the concept of transformers in deep learning",
            "What are attention mechanisms?",
            "Describe the role of self-attention in language models"
        ]
        print("\nBatch generation on advanced ML topics:")
        responses = await provider.generate_batch(prompts)
        for prompt, response in zip(prompts, responses):
            print(f"\nQ: {prompt}")
            print(f"A: {response}")

    except Exception as e:
        print(f"Error with GPT-4 Turbo: {e}")

    # Test Mixtral-8x7B
    try:
        mixtral_provider = LLMProviderFactory.create_provider(HUGGINGFACE_CONFIG)
        print("\nTesting Mixtral-8x7B:")
        
        # Complex reasoning task
        prompt = """
        Solve this step by step:
        A company has three products: A, B, and C.
        Product A costs $100 and has a profit margin of 20%.
        Product B costs $150 and has a profit margin of 30%.
        Product C costs $200 and has a profit margin of 25%.
        If they sell 100 units of each product, what is the total profit?
        """
        print("\nComplex reasoning task:")
        response = await generate_text(mixtral_provider, prompt)
        print(f"Response: {response}")

        # Clean up resources
        mixtral_provider.cleanup()

    except Exception as e:
        print(f"Error with Mixtral: {e}")

    # Test Llama 2 70B
    try:
        llama_provider = LLMProviderFactory.create_provider(LLAMA2_CONFIG)
        print("\nTesting Llama 2 70B:")
        
        # Creative writing task
        prompt = """
        Write a short story about artificial intelligence becoming self-aware,
        but with an unexpected twist. The story should be both thought-provoking
        and entertaining.
        """
        print("\nCreative writing task:")
        await generate_with_streaming(llama_provider, prompt)

    except Exception as e:
        print(f"Error with Llama 2: {e}")

if __name__ == "__main__":
    # Run the example
    asyncio.run(main()) 