from typing import Dict, Any, Optional
from ..exceptions import ConfigurationError
import re

class LLMConfigValidator:
    """
    Validator for LLM provider configurations.
    """

    @staticmethod
    def validate_openai_config(config: Dict[str, Any]) -> None:
        """
        Validate OpenAI configuration.

        Args:
            config (Dict[str, Any]): OpenAI configuration dictionary.

        Raises:
            ConfigurationError: If configuration is invalid.
        """
        required_fields = ['api_key', 'model']
        for field in required_fields:
            if field not in config:
                raise ConfigurationError(f"Missing required field '{field}' in OpenAI configuration")

        # Validate API key format
        if not re.match(r'^sk-[A-Za-z0-9]{48}$', config['api_key']):
            raise ConfigurationError("Invalid OpenAI API key format")

        # Validate model name
        valid_models = [
            # GPT-4 Models
            'gpt-4-1106-preview',  # Latest GPT-4 Turbo
            'gpt-4-vision-preview',  # GPT-4 with vision capabilities
            'gpt-4',  # Base GPT-4
            'gpt-4-32k',  # GPT-4 with 32k context
            # GPT-3.5 Models
            'gpt-3.5-turbo-1106',  # Latest GPT-3.5 Turbo
            'gpt-3.5-turbo',  # GPT-3.5 Turbo
            'gpt-3.5-turbo-16k',  # GPT-3.5 with 16k context
            # Fine-tuning models
            'gpt-3.5-turbo-0613',  # For fine-tuning
            'gpt-3.5-turbo-1106',  # Latest for fine-tuning
            'babbage-002',  # Updated Babbage
            'davinci-002',  # Updated Davinci
            # Embedding models
            'text-embedding-3-small',  # Latest small embedding model
            'text-embedding-3-large',  # Latest large embedding model
            'text-embedding-ada-002'  # Previous embedding model
        ]
        if config['model'] not in valid_models:
            raise ConfigurationError(f"Invalid model name. Must be one of: {', '.join(valid_models)}")

        # Validate optional parameters
        if 'temperature' in config:
            if not isinstance(config['temperature'], (int, float)) or not 0 <= config['temperature'] <= 1:
                raise ConfigurationError("Temperature must be a float between 0 and 1")

        if 'max_tokens' in config:
            if not isinstance(config['max_tokens'], int) or config['max_tokens'] <= 0:
                raise ConfigurationError("max_tokens must be a positive integer")

        if 'rate_limit' in config:
            if not isinstance(config['rate_limit'], int) or config['rate_limit'] <= 0:
                raise ConfigurationError("rate_limit must be a positive integer")

    @staticmethod
    def validate_huggingface_config(config: Dict[str, Any]) -> None:
        """
        Validate HuggingFace configuration.

        Args:
            config (Dict[str, Any]): HuggingFace configuration dictionary.

        Raises:
            ConfigurationError: If configuration is invalid.
        """
        required_fields = ['model_name']
        for field in required_fields:
            if field not in config:
                raise ConfigurationError(f"Missing required field '{field}' in HuggingFace configuration")

        # Recommended models list (these are some of the latest and most popular models)
        recommended_models = [
            'meta-llama/Llama-2-70b-chat-hf',  # Llama 2 70B
            'meta-llama/Llama-2-13b-chat-hf',  # Llama 2 13B
            'meta-llama/Llama-2-7b-chat-hf',   # Llama 2 7B
            'mistralai/Mistral-7B-Instruct-v0.2',  # Latest Mistral
            'mistralai/Mixtral-8x7B-Instruct-v0.1',  # Mixtral (MoE model)
            'HuggingFaceH4/zephyr-7b-beta',    # Zephyr
            'tiiuae/falcon-180B-chat',         # Falcon
            'google/gemma-7b',                 # Gemma
            'google/gemma-2b',                 # Gemma 2B
            'microsoft/phi-2',                  # Phi-2
            'google/flan-t5-xxl',              # Flan-T5
            'google/flan-ul2',                 # Flan-UL2
            'stabilityai/stable-beluga-7b',    # Stable Beluga
            'THUDM/chatglm3-6b',              # ChatGLM3
            'openchat/openchat-3.5',           # OpenChat
            'Intel/neural-chat-7b-v3-1',       # Neural Chat
            'anthropic/claude-3-opus',         # Claude 3 Opus
            'anthropic/claude-3-sonnet',       # Claude 3 Sonnet
            'anthropic/claude-3-haiku'         # Claude 3 Haiku
        ]

        # Validate model name format but don't restrict to recommended models
        if not re.match(r'^[\w\-/]+$', config['model_name']):
            raise ConfigurationError("Invalid model name format")

        # Add a warning if not using a recommended model
        if config['model_name'] not in recommended_models:
            print(f"Warning: Using non-recommended model '{config['model_name']}'. Consider using one of the recommended models for better performance.")

        # Validate device
        if 'device' in config:
            if config['device'] not in ['cpu', 'cuda']:
                raise ConfigurationError("device must be either 'cpu' or 'cuda'")

        # Validate optional parameters
        if 'max_length' in config:
            if not isinstance(config['max_length'], int) or config['max_length'] <= 0:
                raise ConfigurationError("max_length must be a positive integer")

        if 'temperature' in config:
            if not isinstance(config['temperature'], (int, float)) or not 0 <= config['temperature'] <= 1:
                raise ConfigurationError("temperature must be a float between 0 and 1")

        if 'use_auth_token' in config:
            if not isinstance(config['use_auth_token'], str):
                raise ConfigurationError("use_auth_token must be a string")

    @staticmethod
    def validate_huggingface_api_config(config: Dict[str, Any]) -> None:
        """
        Validate HuggingFace Inference API configuration.

        Args:
            config (Dict[str, Any]): HuggingFace API configuration dictionary.

        Raises:
            ConfigurationError: If configuration is invalid.
        """
        required_fields = ['api_token', 'model_name']
        for field in required_fields:
            if field not in config:
                raise ConfigurationError(f"Missing required field '{field}' in HuggingFace API configuration")

        # Validate API token
        if not isinstance(config['api_token'], str) or len(config['api_token']) < 8:
            raise ConfigurationError("Invalid API token format")

        # Validate model name format
        if not re.match(r'^[\w\-/]+$', config['model_name']):
            raise ConfigurationError("Invalid model name format")

    @staticmethod
    def validate_cache_config(config: Dict[str, Any]) -> None:
        """
        Validate cache configuration.

        Args:
            config (Dict[str, Any]): Cache configuration dictionary.

        Raises:
            ConfigurationError: If configuration is invalid.
        """
        if 'cache' in config:
            cache_config = config['cache']
            
            if 'enabled' in cache_config and not isinstance(cache_config['enabled'], bool):
                raise ConfigurationError("cache.enabled must be a boolean")

            if 'max_size' in cache_config:
                if not isinstance(cache_config['max_size'], int) or cache_config['max_size'] <= 0:
                    raise ConfigurationError("cache.max_size must be a positive integer")

            if 'ttl' in cache_config:
                if not isinstance(cache_config['ttl'], int) or cache_config['ttl'] <= 0:
                    raise ConfigurationError("cache.ttl must be a positive integer")

    @staticmethod
    def validate_advanced_config(config: Dict[str, Any]) -> None:
        """
        Validate advanced configuration options.

        Args:
            config (Dict[str, Any]): Advanced configuration dictionary.

        Raises:
            ConfigurationError: If configuration is invalid.
        """
        if 'advanced' in config:
            advanced = config['advanced']

            # Validate retry configuration
            if 'retry' in advanced:
                retry = advanced['retry']
                if 'max_attempts' in retry:
                    if not isinstance(retry['max_attempts'], int) or retry['max_attempts'] <= 0:
                        raise ConfigurationError("retry.max_attempts must be a positive integer")

                if 'initial_delay' in retry:
                    if not isinstance(retry['initial_delay'], (int, float)) or retry['initial_delay'] <= 0:
                        raise ConfigurationError("retry.initial_delay must be a positive number")

            # Validate timeout configuration
            if 'timeout' in advanced:
                timeout = advanced['timeout']
                for key in ['connect', 'read']:
                    if key in timeout:
                        if not isinstance(timeout[key], (int, float)) or timeout[key] <= 0:
                            raise ConfigurationError(f"timeout.{key} must be a positive number")

            # Validate logging configuration
            if 'logging' in advanced:
                logging = advanced['logging']
                if 'level' in logging:
                    valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
                    if logging['level'] not in valid_levels:
                        raise ConfigurationError(f"logging.level must be one of: {', '.join(valid_levels)}")

    @classmethod
    def validate(cls, config: Dict[str, Any]) -> None:
        """
        Validate the complete LLM configuration.

        Args:
            config (Dict[str, Any]): Complete configuration dictionary.

        Raises:
            ConfigurationError: If configuration is invalid.
        """
        if 'type' not in config:
            raise ConfigurationError("Missing 'type' field in configuration")

        config_type = config['type']
        if config_type == 'openai':
            cls.validate_openai_config(config)
        elif config_type == 'huggingface':
            cls.validate_huggingface_config(config)
        elif config_type == 'huggingface_api':
            cls.validate_huggingface_api_config(config)
        else:
            raise ConfigurationError(f"Unknown configuration type: {config_type}")

        # Validate common configurations
        cls.validate_cache_config(config)
        cls.validate_advanced_config(config) 