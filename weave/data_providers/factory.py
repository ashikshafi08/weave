from typing import Dict, Any
from .registry import data_provider_registry

def create_data_provider(config: Dict[str, Any]):
    provider_name = config["type"]
    provider_class = data_provider_registry.get(provider_name)
    return provider_class.from_config(config)