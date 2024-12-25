"""Library of reusable prompt templates."""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import json
from .prompt_template import PromptTemplate


class PromptLibrary:
    """Manager for a collection of prompt templates."""
    
    def __init__(self):
        """Initialize the prompt library."""
        self.templates: Dict[str, PromptTemplate] = {}
        self.categories: Dict[str, List[str]] = {}
        
        # Load default templates
        self._load_default_templates()
        
    def _load_default_templates(self) -> None:
        """Load built-in default templates."""
        self.add_template(
            "classification",
            PromptTemplate(
                "Classify the following text into $categories:\n\n$text",
                {"task": "classification", "version": "1.0"}
            )
        )
        
        self.add_template(
            "qa",
            PromptTemplate(
                "Answer the following question based on the context:\n\n"
                "Context: $context\n\nQuestion: $question",
                {"task": "question_answering", "version": "1.0"}
            )
        )
        
        self.add_template(
            "summarization",
            PromptTemplate(
                "Summarize the following text in $style style with "
                "$max_words words:\n\n$text",
                {"task": "summarization", "version": "1.0"}
            )
        )
        
        # Add categories
        self.categories = {
            "general": ["classification", "qa", "summarization"],
            "specialized": []
        }
        
    def add_template(self,
                    name: str,
                    template: PromptTemplate,
                    category: str = "general") -> None:
        """Add a template to the library.
        
        Args:
            name: Template name
            template: PromptTemplate instance
            category: Category to add template to
        """
        self.templates[name] = template
        
        if category not in self.categories:
            self.categories[category] = []
        if name not in self.categories[category]:
            self.categories[category].append(name)
            
    def get_template(self, name: str) -> PromptTemplate:
        """Get a template by name.
        
        Args:
            name: Template name
            
        Returns:
            PromptTemplate instance
        """
        if name not in self.templates:
            raise KeyError(f"Template not found: {name}")
        return self.templates[name]
        
    def remove_template(self, name: str) -> None:
        """Remove a template from the library.
        
        Args:
            name: Template name
        """
        if name in self.templates:
            del self.templates[name]
            for category in self.categories.values():
                if name in category:
                    category.remove(name)
                    
    def get_categories(self) -> List[str]:
        """Get list of available categories.
        
        Returns:
            List of category names
        """
        return list(self.categories.keys())
        
    def get_templates_in_category(self, category: str) -> List[str]:
        """Get templates in a category.
        
        Args:
            category: Category name
            
        Returns:
            List of template names
        """
        if category not in self.categories:
            raise KeyError(f"Category not found: {category}")
        return self.categories[category].copy()
        
    def save_to_directory(self, directory: Union[str, Path]) -> None:
        """Save all templates to a directory.
        
        Args:
            directory: Directory to save templates
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        # Save templates
        for name, template in self.templates.items():
            template.to_file(directory / f"{name}.json")
            
        # Save category information
        with (directory / "categories.json").open("w") as f:
            json.dump(self.categories, f, indent=2)
            
    @classmethod
    def from_directory(cls, directory: Union[str, Path]) -> "PromptLibrary":
        """Load templates from a directory.
        
        Args:
            directory: Directory containing templates
            
        Returns:
            PromptLibrary instance
        """
        library = cls()
        directory = Path(directory)
        
        # Load templates
        for template_file in directory.glob("*.json"):
            if template_file.name != "categories.json":
                name = template_file.stem
                template = PromptTemplate.from_file(template_file)
                library.add_template(name, template)
                
        # Load categories
        categories_file = directory / "categories.json"
        if categories_file.exists():
            with categories_file.open() as f:
                library.categories = json.load(f)
                
        return library
