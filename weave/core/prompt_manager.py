from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from jinja2 import Template
import json
import os

class PromptManager:
    """
    Manages and generates prompts for LLM interactions.
    """

    def __init__(self, config: Dict[str, Any]):
        self.templates: Dict[str, Dict[str, str]] = {}
        self.config = config
        self.default_language = config.get('default_language', 'en')
        self.load_templates()

    def load_templates(self):
        """
        Load templates from a JSON file or directory.
        """
        template_source = self.config.get('template_source', 'templates.json')
        if os.path.isdir(template_source):
            for filename in os.listdir(template_source):
                if filename.endswith('.json'):
                    with open(os.path.join(template_source, filename), 'r') as f:
                        self.templates.update(json.load(f))
        elif os.path.isfile(template_source):
            with open(template_source, 'r') as f:
                self.templates = json.load(f)
        else:
            raise ValueError(f"Invalid template source: {template_source}")

    def get_prompt(self, prompt_type: str, language: Optional[str] = None, **kwargs) -> str:
        """
        Get a prompt based on the prompt type and parameters.

        Args:
            prompt_type (str): Type of prompt to retrieve.
            language (Optional[str]): Language of the prompt. Defaults to the default language.
            **kwargs: Additional parameters for prompt generation.

        Returns:
            str: Generated prompt.

        Raises:
            ValueError: If the prompt type or language is not found.
        """
        language = language or self.default_language
        if prompt_type not in self.templates:
            raise ValueError(f"Prompt type '{prompt_type}' not found")
        if language not in self.templates[prompt_type]:
            raise ValueError(f"Language '{language}' not found for prompt type '{prompt_type}'")

        template = self.templates[prompt_type][language]
        return Template(template).render(**kwargs)

    def add_prompt_template(self, prompt_type: str, template: str, language: Optional[str] = None):
        """
        Add a new prompt template.

        Args:
            prompt_type (str): Type of prompt.
            template (str): Prompt template string.
            language (Optional[str]): Language of the template. Defaults to the default language.
        """
        language = language or self.default_language
        if prompt_type not in self.templates:
            self.templates[prompt_type] = {}
        self.templates[prompt_type][language] = template

    def chain_prompts(self, prompt_chain: list, **kwargs) -> str:
        """
        Generate a chained prompt from multiple prompt types.

        Args:
            prompt_chain (list): List of prompt types to chain.
            **kwargs: Additional parameters for prompt generation.

        Returns:
            str: Generated chained prompt.
        """
        return " ".join([self.get_prompt(prompt_type, **kwargs) for prompt_type in prompt_chain])

    def save_templates(self):
        """
        Save the current templates to the template source.
        """
        template_source = self.config.get('template_source', 'templates.json')
        with open(template_source, 'w') as f:
            json.dump(self.templates, f, indent=2)

class PromptManagerFactory:
    @staticmethod
    def create_prompt_manager(config: Dict[str, Any]) -> PromptManager:
        """
        Create a PromptManager instance based on the provided configuration.

        Args:
            config (Dict[str, Any]): Configuration for the PromptManager.

        Returns:
            PromptManager: An instance of PromptManager.
        """
        return PromptManager(config)