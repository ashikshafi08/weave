import pytest
import os
from typing import Dict, Any
import asyncio
from weave.llm_providers import OpenAILLM, HuggingFaceLLM, HuggingFaceInferenceAPI
from weave.llm_providers.factory import LLMProviderFactory
from weave.core.exceptions import ConfigurationError, LLMError, RateLimitError

# Test configurations
OPENAI_CONFIG = {
    'type': 'openai',
    'api_key': 'sk-' + 'a' * 48,
    'model': 'gpt-3.5-turbo',
    'temperature': 0.7,
    'max_tokens': 150,
    'rate_limit': 60
}

HUGGINGFACE_CONFIG = {
    'type': 'huggingface',
    'model_name': 'google/flan-t5-base',
    'device': 'cpu',
    'max_length': 128,
    'temperature': 0.7
}

HUGGINGFACE_API_CONFIG = {
    'type': 'huggingface_api',
    'api_token': 'hf_' + 'a' * 20,
    'model_name': 'gpt2'
}

@pytest.fixture
def openai_provider():
    return OpenAILLM(OPENAI_CONFIG)

@pytest.fixture
def huggingface_provider():
    return HuggingFaceLLM(HUGGINGFACE_CONFIG)

@pytest.fixture
def huggingface_api_provider():
    return HuggingFaceInferenceAPI(HUGGINGFACE_API_CONFIG)

class TestLLMProviderFactory:
    def test_register_provider(self):
        class CustomProvider(OpenAILLM):
            pass

        LLMProviderFactory.register_provider('custom', CustomProvider)
        assert 'custom' in LLMProviderFactory.list_providers()
        assert LLMProviderFactory.get_provider_class('custom') == CustomProvider

    def test_register_duplicate_provider(self):
        with pytest.raises(ValueError):
            LLMProviderFactory.register_provider('openai', OpenAILLM)

    def test_create_provider(self):
        provider = LLMProviderFactory.create_provider(OPENAI_CONFIG)
        assert isinstance(provider, OpenAILLM)

    def test_create_provider_invalid_type(self):
        config = OPENAI_CONFIG.copy()
        config['type'] = 'invalid'
        with pytest.raises(ConfigurationError):
            LLMProviderFactory.create_provider(config)

class TestOpenAIProvider:
    @pytest.mark.asyncio
    async def test_api_call(self, openai_provider):
        prompt = "Test prompt"
        try:
            response = await openai_provider._api_call(prompt)
            assert isinstance(response, str)
            assert len(response) > 0
        except LLMError as e:
            if "OpenAI API key" in str(e):
                pytest.skip("OpenAI API key not configured")

    @pytest.mark.asyncio
    async def test_generate_batch(self, openai_provider):
        prompts = ["Test prompt 1", "Test prompt 2"]
        try:
            responses = await openai_provider.generate_batch(prompts)
            assert len(responses) == len(prompts)
            assert all(isinstance(r, str) for r in responses)
        except LLMError as e:
            if "OpenAI API key" in str(e):
                pytest.skip("OpenAI API key not configured")

    def test_get_token_count(self, openai_provider):
        text = "This is a test text"
        count = openai_provider.get_token_count(text)
        assert isinstance(count, int)
        assert count > 0

class TestHuggingFaceProvider:
    @pytest.mark.asyncio
    async def test_api_call(self, huggingface_provider):
        prompt = "Test prompt"
        response = await huggingface_provider._api_call(prompt)
        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.asyncio
    async def test_generate_batch(self, huggingface_provider):
        prompts = ["Test prompt 1", "Test prompt 2"]
        responses = await huggingface_provider.generate_batch(prompts)
        assert len(responses) == len(prompts)
        assert all(isinstance(r, str) for r in responses)

    def test_get_token_count(self, huggingface_provider):
        text = "This is a test text"
        count = huggingface_provider.get_token_count(text)
        assert isinstance(count, int)
        assert count > 0

    def test_cleanup(self, huggingface_provider):
        huggingface_provider.cleanup()
        # Verify that resources are cleaned up (no error should be raised)

class TestHuggingFaceAPIProvider:
    @pytest.mark.asyncio
    async def test_api_call(self, huggingface_api_provider):
        prompt = "Test prompt"
        try:
            response = await huggingface_api_provider._api_call(prompt)
            assert isinstance(response, str)
            assert len(response) > 0
        except LLMError as e:
            if "HuggingFace API token" in str(e):
                pytest.skip("HuggingFace API token not configured")

    @pytest.mark.asyncio
    async def test_rate_limit(self, huggingface_api_provider):
        prompt = "Test prompt"
        huggingface_api_provider.rate_limit = 1
        
        # First call should succeed
        await huggingface_api_provider._api_call(prompt)
        
        # Second call should raise RateLimitError
        with pytest.raises(RateLimitError):
            await huggingface_api_provider._api_call(prompt)

class TestConfigValidation:
    def test_openai_config_validation(self):
        # Test valid config
        LLMProviderFactory.create_provider(OPENAI_CONFIG)

        # Test missing required field
        invalid_config = OPENAI_CONFIG.copy()
        del invalid_config['api_key']
        with pytest.raises(ConfigurationError):
            LLMProviderFactory.create_provider(invalid_config)

        # Test invalid temperature
        invalid_config = OPENAI_CONFIG.copy()
        invalid_config['temperature'] = 2.0
        with pytest.raises(ConfigurationError):
            LLMProviderFactory.create_provider(invalid_config)

    def test_huggingface_config_validation(self):
        # Test valid config
        LLMProviderFactory.create_provider(HUGGINGFACE_CONFIG)

        # Test missing required field
        invalid_config = HUGGINGFACE_CONFIG.copy()
        del invalid_config['model_name']
        with pytest.raises(ConfigurationError):
            LLMProviderFactory.create_provider(invalid_config)

        # Test invalid device
        invalid_config = HUGGINGFACE_CONFIG.copy()
        invalid_config['device'] = 'invalid'
        with pytest.raises(ConfigurationError):
            LLMProviderFactory.create_provider(invalid_config)

class TestIntegration:
    @pytest.mark.asyncio
    async def test_provider_switching(self):
        providers = [
            LLMProviderFactory.create_provider(OPENAI_CONFIG),
            LLMProviderFactory.create_provider(HUGGINGFACE_CONFIG)
        ]
        
        prompt = "Test prompt"
        for provider in providers:
            try:
                response = await provider._api_call(prompt)
                assert isinstance(response, str)
                assert len(response) > 0
            except LLMError as e:
                if "API key" in str(e):
                    continue  # Skip if API key not configured

    @pytest.mark.asyncio
    async def test_caching(self):
        provider = LLMProviderFactory.create_provider(OPENAI_CONFIG)
        prompt = "Test prompt"
        
        # First call should hit the API
        try:
            response1 = await provider._api_call(prompt)
            # Second call should hit the cache
            response2 = await provider._api_call(prompt)
            assert response1 == response2
        except LLMError as e:
            if "OpenAI API key" in str(e):
                pytest.skip("OpenAI API key not configured") 