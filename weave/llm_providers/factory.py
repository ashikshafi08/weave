from typing import Dict, Any, Type
from ..core.llm_interface import LLMInterface
from ..core.exceptions import ConfigurationError
from .openai_provider import OpenAILLM
from .huggingface_provider import HuggingFaceLLM, HuggingFaceInferenceAPI
from ..core.config_validators.llm_config_validator import LLMConfigValidator

class LLMProviderFactory:
    """
    Factory class for creating LLM provider instances.
    """

    _providers: Dict[str, Type[LLMInterface]] = {
        'openai': OpenAILLM,
        'huggingface': HuggingFaceLLM,
        'huggingface_api': HuggingFaceInferenceAPI
    }

    @classmethod
    def register_provider(cls, name: str, provider_class: Type[LLMInterface]) -> None:
        """
        Register a new LLM provider.

        Args:
            name (str): Name of the provider.
            provider_class (Type[LLMInterface]): Provider class.

        Raises:
            ValueError: If provider name is already registered.
        """
        if name in cls._providers:
            raise ValueError(f"Provider '{name}' is already registered")
        cls._providers[name] = provider_class

    @classmethod
    def create_provider(cls, config: Dict[str, Any]) -> LLMInterface:
        """
        Create an LLM provider instance based on configuration.

        Args:
            config (Dict[str, Any]): Provider configuration.

        Returns:
            LLMInterface: LLM provider instance.

        Raises:
            ConfigurationError: If configuration is invalid or provider type is unknown.
        """
        # Validate configuration
        LLMConfigValidator.validate(config)

        provider_type = config['type']
        if provider_type not in cls._providers:
            raise ConfigurationError(f"Unknown provider type: {provider_type}")

        # Create provider instance
        provider_class = cls._providers[provider_type]
        return provider_class(config)

    @classmethod
    def list_providers(cls) -> Dict[str, Type[LLMInterface]]:
        """
        Get a dictionary of registered providers.

        Returns:
            Dict[str, Type[LLMInterface]]: Dictionary of provider names and classes.
        """
        return cls._providers.copy()

    @classmethod
    def get_provider_class(cls, name: str) -> Type[LLMInterface]:
        """
        Get a provider class by name.

        Args:
            name (str): Name of the provider.

        Returns:
            Type[LLMInterface]: Provider class.

        Raises:
            ValueError: If provider is not found.
        """
        if name not in cls._providers:
            raise ValueError(f"Provider '{name}' not found")
        return cls._providers[name] 