# weave/weave/core/plugin.py
import importlib
import pkgutil
from typing import Dict, Type, TypeVar

T = TypeVar('T')

class PluginRegistry:
    def __init__(self):
        self.plugins: Dict[str, Type[T]] = {}

    def register(self, name: str, plugin: Type[T]):
        self.plugins[name] = plugin

    def get(self, name: str) -> Type[T]:
        return self.plugins[name]

    def list(self):
        return list(self.plugins.keys())

def load_plugins(package_name: str, base_class: Type[T]):
    package = importlib.import_module(package_name)
    registry = PluginRegistry()

    for _, name, _ in pkgutil.iter_modules(package.__path__):
        module = importlib.import_module(f'{package_name}.{name}')
        for item_name in dir(module):
            item = getattr(module, item_name)
            if isinstance(item, type) and issubclass(item, base_class) and item != base_class:
                registry.register(name, item)

    return registry