# weave/core/plugin_manager.py

from typing import Dict, Any, Type, Optional
from .exceptions import PluginError
import importlib
import pkg_resources

class PluginManager:
    """
    Manager for plugins in the Weave framework.
    """

    def __init__(self):
        self._plugins: Dict[str, Dict[str, Any]] = {}
        self._loaded_plugins: Dict[str, Any] = {}
        self.component_types: Dict[str, str] = {
            'pipeline': 'Pipeline',
            'data_source': 'DataSource',
            'data_processor': 'DataProcessor',
            'data_generator': 'DataGenerator',
            'data_validator': 'DataValidator',
            'llm_interface': 'LLMInterface',
            'prompt_manager': 'PromptManager'
        }

    def register_plugin(self, name: str, module_path: str, version: str = "latest") -> None:
        """
        Register a new plugin.

        Args:
            name (str): Name of the plugin.
            module_path (str): Import path for the plugin module.
            version (str): Version of the plugin. Defaults to "latest".

        Raises:
            PluginError: If the plugin is already registered.
        """
        if name in self._plugins and version in self._plugins[name]:
            raise PluginError(f"Plugin {name} version {version} is already registered")
        
        if name not in self._plugins:
            self._plugins[name] = {}
        
        self._plugins[name][version] = {
            "module_path": module_path,
            "loaded": False,
            "instance": None
        }

    def _load_plugin(self, name: str, version: str = "latest") -> Any:
        """
        Lazy load a plugin.

        Args:
            name (str): Name of the plugin.
            version (str): Version of the plugin. Defaults to "latest".

        Returns:
            Any: Loaded plugin instance.

        Raises:
            PluginError: If the plugin is not found or fails to load.
        """
        if name not in self._plugins or version not in self._plugins[name]:
            raise PluginError(f"Plugin {name} version {version} not found")

        plugin_info = self._plugins[name][version]
        if not plugin_info["loaded"]:
            try:
                module = importlib.import_module(plugin_info["module_path"])
                plugin_class = getattr(module, name)
                plugin_instance = plugin_class()
                if not self._validate_plugin(plugin_instance):
                    raise PluginError(f"Invalid plugin: {name}")
                plugin_info["instance"] = plugin_instance
                plugin_info["loaded"] = True
            except Exception as e:
                raise PluginError(f"Failed to load plugin {name}: {str(e)}")

        return plugin_info["instance"]

    def _validate_plugin(self, plugin: Any) -> bool:
        """
        Validate if the plugin adheres to the required interface.

        Args:
            plugin (Any): Plugin instance to validate.

        Returns:
            bool: True if valid, False otherwise.
        """
        for component_type, base_class_name in self.component_types.items():
            if component_type in plugin.__class__.__name__.lower():
                base_class = globals().get(base_class_name)
                if base_class and isinstance(plugin, base_class):
                    return True
        
        required_methods = ['initialize', 'execute']
        return all(hasattr(plugin, method) for method in required_methods)

    def get_component(self, name: str, version: str = "latest") -> Type:
        """
        Get a component by name and version.

        Args:
            name (str): Name of the component.
            version (str): Version of the component. Defaults to "latest".

        Returns:
            Type: Component class.

        Raises:
            PluginError: If the component is not found.
        """
        component_type = name.split('_')[-1]
        if component_type not in self.component_types:
            raise PluginError(f"Invalid component type: {component_type}")

        return self._load_plugin(name, version)

    def get_pipeline(self, name: str, version: str = "latest") -> Type:
        """
        Get a pipeline by name and version.

        Args:
            name (str): Name of the pipeline.
            version (str): Version of the pipeline. Defaults to "latest".

        Returns:
            Type: Pipeline class.
        """
        return self.get_component(f"{name}_pipeline", version)

    def get_data_source(self, name: str, version: str = "latest") -> Type:
        """
        Get a data source by name and version.

        Args:
            name (str): Name of the data source.
            version (str): Version of the data source. Defaults to "latest".

        Returns:
            Type: DataSource class.
        """
        return self.get_component(f"{name}_data_source", version)

    def get_data_processor(self, name: str, version: str = "latest") -> Type:
        """
        Get a data processor by name and version.

        Args:
            name (str): Name of the data processor.
            version (str): Version of the data processor. Defaults to "latest".

        Returns:
            Type: DataProcessor class.
        """
        return self.get_component(f"{name}_data_processor", version)

    def get_data_generator(self, name: str, version: str = "latest") -> Type:
        """
        Get a data generator by name and version.

        Args:
            name (str): Name of the data generator.
            version (str): Version of the data generator. Defaults to "latest".

        Returns:
            Type: DataGenerator class.
        """
        return self.get_component(f"{name}_data_generator", version)

    def get_validator(self, name: str, version: str = "latest") -> Type:
        """
        Get a validator by name and version.

        Args:
            name (str): Name of the validator.
            version (str): Version of the validator. Defaults to "latest".

        Returns:
            Type: DataValidator class.
        """
        return self.get_component(f"{name}_data_validator", version)

    def get_llm_interface(self, name: str, version: str = "latest") -> Type:
        """
        Get an LLM interface by name and version.

        Args:
            name (str): Name of the LLM interface.
            version (str): Version of the LLM interface. Defaults to "latest".

        Returns:
            Type: LLMInterface class.
        """
        return self.get_component(f"{name}_llm_interface", version)

    def get_prompt_manager(self, name: str, version: str = "latest") -> Type:
        """
        Get a prompt manager by name and version.

        Args:
            name (str): Name of the prompt manager.
            version (str): Version of the prompt manager. Defaults to "latest".

        Returns:
            Type: PromptManager class.
        """
        return self.get_component(f"{name}_prompt_manager", version)

    def list_plugins(self) -> Dict[str, Dict[str, Any]]:
        """
        List all registered plugins.

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of registered plugins.
        """
        return {name: {version: info["loaded"] for version, info in versions.items()}
                for name, versions in self._plugins.items()}

    def remove_plugin(self, name: str, version: Optional[str] = None) -> None:
        """
        Remove a plugin by name and optionally version.

        Args:
            name (str): Name of the plugin to remove.
            version (Optional[str]): Version of the plugin to remove. If None, removes all versions.

        Raises:
            PluginError: If the plugin is not found.
        """
        if name not in self._plugins:
            raise PluginError(f"Plugin {name} not found")
        
        if version:
            if version not in self._plugins[name]:
                raise PluginError(f"Plugin {name} version {version} not found")
            del self._plugins[name][version]
            if not self._plugins[name]:
                del self._plugins[name]
        else:
            del self._plugins[name]

    def get_component_types(self) -> Dict[str, str]:
        """
        Get all registered component types.

        Returns:
            Dict[str, str]: Dictionary of component types.
        """
        return self.component_types.copy()

    def register_component_type(self, name: str, class_name: str) -> None:
        """
        Register a new component type.

        Args:
            name (str): Name of the component type.
            class_name (str): Name of the class for this component type.

        Raises:
            PluginError: If the component type is already registered.
        """
        if name in self.component_types:
            raise PluginError(f"Component type {name} is already registered")
        self.component_types[name] = class_name

    def resolve_dependencies(self, plugin_name: str, version: str = "latest") -> None:
        """
        Resolve and load dependencies for a given plugin.

        Args:
            plugin_name (str): Name of the plugin.
            version (str): Version of the plugin. Defaults to "latest".

        Raises:
            PluginError: If dependencies cannot be resolved.
        """
        try:
            plugin = self._load_plugin(plugin_name, version)
            if hasattr(plugin, 'dependencies'):
                for dep in plugin.dependencies:
                    pkg_resources.require(dep)
        except pkg_resources.DistributionNotFound as e:
            raise PluginError(f"Dependency not found for plugin {plugin_name}: {str(e)}")
        except pkg_resources.VersionConflict as e:
            raise PluginError(f"Dependency version conflict for plugin {plugin_name}: {str(e)}")
        except Exception as e:
            raise PluginError(f"Failed to resolve dependencies for plugin {plugin_name}: {str(e)}")