from .registry import data_provider_registry

def register_data_provider(name: str):
    def decorator(cls):
        data_provider_registry.register(name, cls)
        return cls
    return decorator