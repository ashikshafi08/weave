# weave/core/config_validator.py

from typing import Dict, Any
from .exceptions import ConfigurationError

class ConfigurationValidator:
    """
    Validates framework and component configurations.
    """

    @staticmethod
    def validate_config(config: Dict[str, Any], schema: Dict[str, Any]) -> None:
        """
        Validate configuration against a schema.

        Args:
            config (Dict[str, Any]): Configuration to validate
            schema (Dict[str, Any]): Validation schema

        Raises:
            ConfigurationError: If configuration is invalid
        """
        try:
            for key, requirements in schema.items():
                if requirements.get('required', False) and key not in config:
                    raise ConfigurationError(f"Missing required configuration key: {key}")
                
                if key in config:
                    value = config[key]
                    if not isinstance(value, requirements.get('type', object)):
                        raise ConfigurationError(
                            f"Invalid type for {key}. Expected {requirements['type']}, got {type(value)}"
                        )
        except Exception as e:
            raise ConfigurationError(f"Configuration validation failed: {str(e)}")