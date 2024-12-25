"""Prompt template management for dynamic prompt generation."""

from typing import Any, Dict, List, Optional, Union
from string import Template
import json
from pathlib import Path


class PromptTemplate:
    """Class for managing and rendering prompt templates."""
    
    def __init__(self, template: str, metadata: Optional[Dict[str, Any]] = None):
        """Initialize the prompt template.
        
        Args:
            template: Template string with $variable placeholders
            metadata: Optional metadata about the template
        """
        self.template = Template(template)
        self.metadata = metadata or {}
        self.required_variables = self._extract_variables()
        
    def render(self, variables: Dict[str, Any]) -> str:
        """Render the template with provided variables.
        
        Args:
            variables: Dictionary of variable values
            
        Returns:
            Rendered prompt string
        """
        missing = self.required_variables - set(variables.keys())
        if missing:
            raise ValueError(f"Missing required variables: {missing}")
            
        return self.template.safe_substitute(variables)
        
    def _extract_variables(self) -> set:
        """Extract required variables from template.
        
        Returns:
            Set of variable names
        """
        return {
            name for _, name, _, _ in Template.pattern.findall(self.template.template)
        }
        
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "PromptTemplate":
        """Load template from a file.
        
        Args:
            path: Path to template file
            
        Returns:
            PromptTemplate instance
        """
        path = Path(path)
        with path.open() as f:
            if path.suffix == ".json":
                data = json.load(f)
                return cls(data["template"], data.get("metadata"))
            else:
                return cls(f.read().strip())
                
    def to_file(self, path: Union[str, Path]) -> None:
        """Save template to a file.
        
        Args:
            path: Path to save template
        """
        path = Path(path)
        with path.open("w") as f:
            if path.suffix == ".json":
                json.dump({
                    "template": self.template.template,
                    "metadata": self.metadata
                }, f, indent=2)
            else:
                f.write(self.template.template)
                
    def validate_variables(self, variables: Dict[str, Any]) -> bool:
        """Validate that all required variables are provided.
        
        Args:
            variables: Dictionary of variable values
            
        Returns:
            True if all required variables are present
        """
        return self.required_variables.issubset(variables.keys())
        
    def get_metadata(self) -> Dict[str, Any]:
        """Get template metadata.
        
        Returns:
            Dictionary of metadata
        """
        return self.metadata.copy()
        
    def update_metadata(self, metadata: Dict[str, Any]) -> None:
        """Update template metadata.
        
        Args:
            metadata: New metadata to merge
        """
        self.metadata.update(metadata)
