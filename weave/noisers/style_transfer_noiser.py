"""Style transfer noiser for text transformation."""

from typing import Any, Dict, List, Optional, Union
from ..core.base_noiser import BaseNoiser
from ..core.model_connector import ModelConnector


class StyleTransferNoiser(BaseNoiser):
    """Noiser for applying style transfer to text.
    
    This noiser can transform text to mimic different writing styles,
    such as famous authors, technical writing, or domain-specific text.
    """
    
    def __init__(self,
                 model_connector: ModelConnector,
                 style_config: Optional[Dict[str, Any]] = None):
        """Initialize the style transfer noiser.
        
        Args:
            model_connector: LLM connector for style transfer
            style_config: Configuration for the target style
        """
        super().__init__()
        self.model = model_connector
        self.style_config = style_config or {}
        
        # Load style templates
        self.style_templates = {
            "technical": "Transform this into technical writing style with precise terminology",
            "casual": "Rewrite this in a casual, conversational tone",
            "academic": "Convert this to academic writing style with formal language",
            "creative": "Transform this using creative and descriptive language",
            "business": "Rewrite this in professional business communication style"
        }
        
        # Add custom styles from config
        if "custom_styles" in self.style_config:
            self.style_templates.update(self.style_config["custom_styles"])
            
    def augment(self, query: str) -> str:
        """Apply style transfer to a single query.
        
        Args:
            query: Original text to transform
            
        Returns:
            Style-transferred text
        """
        style = self.style_config.get("style", "technical")
        if style not in self.style_templates:
            raise ValueError(f"Unknown style: {style}")
            
        prompt = f"{self.style_templates[style]}:\n\n{query}"
        
        response = self.model.generate(
            prompt=prompt,
            max_tokens=self.style_config.get("max_tokens", 150),
            temperature=self.style_config.get("temperature", 0.7)
        )
        
        return response.strip()
        
    def batch_augment(self, queries: List[str]) -> List[str]:
        """Apply style transfer to multiple queries.
        
        Args:
            queries: List of original texts
            
        Returns:
            List of style-transferred texts
        """
        return [self.augment(q) for q in queries]
        
    def add_custom_style(self, name: str, template: str) -> None:
        """Add a custom style template.
        
        Args:
            name: Name of the style
            template: Template for the style transformation
        """
        self.style_templates[name] = template
        
    def get_available_styles(self) -> List[str]:
        """Get list of available style templates.
        
        Returns:
            List of style names
        """
        return list(self.style_templates.keys())
        
    def get_augmentation_metadata(self) -> Dict[str, Any]:
        """Get metadata about the style transfer.
        
        Returns:
            Dictionary containing style transfer configuration
        """
        return {
            "style": self.style_config.get("style", "technical"),
            "temperature": self.style_config.get("temperature", 0.7),
            "available_styles": self.get_available_styles()
        }
