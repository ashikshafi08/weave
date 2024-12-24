"""Persona-based text noising for the Weave framework."""

from typing import Any, Dict, List, Optional, Union
from ..core import BaseNoiser, ModelError
from ..llms import OpenAILLM, HuggingFaceLLM

class PersonaNoiser(BaseNoiser):
    """A noiser that transforms text to match a specific persona or style.
    
    This noiser uses LLMs to rewrite text in the style of a given persona,
    which can be customized through templates and examples.
    """
    
    DEFAULT_TEMPLATE = """
    Transform the following text to match the style and characteristics of the specified persona.
    
    Persona: ${persona_name}
    Description: ${persona_description}
    Key traits: ${persona_traits}
    Examples of their style:
    ${persona_examples}
    
    Style instructions: ${style_instructions}
    Format instructions: ${format_instructions}
    
    Original text:
    ${original_text}
    
    Transformed text:
    """
    
    DEFAULT_FORMAT_INSTRUCTIONS = """
    - Maintain the original meaning and key information
    - Only change the style and tone to match the persona
    - Keep any technical terms, numbers, and specific references intact
    - Preserve any code blocks or special formatting
    """
    
    def __init__(
        self,
        model_connector: Optional[Union[OpenAILLM, HuggingFaceLLM]] = None,
        persona_name: str = "Technical Expert",
        persona_description: str = "A knowledgeable and precise technical professional",
        persona_traits: List[str] = None,
        persona_examples: List[str] = None,
        style_instructions: str = "Write in a clear, professional, and technically accurate manner",
        prompt_template: Optional[str] = None,
        format_instructions: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the PersonaNoiser.
        
        Args:
            model_connector: LLM provider to use for text transformation.
            persona_name: Name or title of the persona.
            persona_description: Brief description of the persona.
            persona_traits: List of key personality traits.
            persona_examples: List of example texts in the persona's style.
            style_instructions: Specific instructions for the style transformation.
            prompt_template: Custom template for the transformation prompt.
            format_instructions: Custom instructions for output formatting.
            config: Additional configuration options.
        """
        super().__init__(config)
        
        # Set up model connector
        self.model = model_connector or OpenAILLM()
        
        # Persona configuration
        self.persona_name = persona_name
        self.persona_description = persona_description
        self.persona_traits = persona_traits or [
            "precise", "knowledgeable", "professional",
            "analytical", "detail-oriented"
        ]
        self.persona_examples = persona_examples or [
            "The implementation leverages advanced algorithms to optimize performance.",
            "Based on the empirical analysis, we can conclude that..."
        ]
        
        # Template configuration
        self.prompt_template = prompt_template or self.DEFAULT_TEMPLATE
        self.style_instructions = style_instructions
        self.format_instructions = format_instructions or self.DEFAULT_FORMAT_INSTRUCTIONS
        
        # Validate template
        self._validate_template()
        
    def _validate_template(self) -> None:
        """Validate that the prompt template contains all required variables."""
        required_vars = {
            "${persona_name}", "${persona_description}", "${persona_traits}",
            "${persona_examples}", "${style_instructions}", "${format_instructions}",
            "${original_text}"
        }
        
        missing_vars = required_vars - set(
            var for var in required_vars if var in self.prompt_template
        )
        
        if missing_vars:
            raise ValueError(
                f"Prompt template is missing required variables: {missing_vars}"
            )
            
    def _build_prompt(self, text: str) -> str:
        """Build the complete prompt for text transformation.
        
        Args:
            text: Original text to transform.
            
        Returns:
            Complete prompt with all variables substituted.
        """
        return self.prompt_template.replace(
            "${persona_name}", self.persona_name
        ).replace(
            "${persona_description}", self.persona_description
        ).replace(
            "${persona_traits}", ", ".join(self.persona_traits)
        ).replace(
            "${persona_examples}", "\n".join(f"- {ex}" for ex in self.persona_examples)
        ).replace(
            "${style_instructions}", self.style_instructions
        ).replace(
            "${format_instructions}", self.format_instructions
        ).replace(
            "${original_text}", text
        )
        
    async def augment(self, text: str) -> str:
        """Transform text to match the specified persona.
        
        Args:
            text: Original text to transform.
            
        Returns:
            Transformed text in the persona's style.
            
        Raises:
            ModelError: If the transformation fails.
        """
        prompt = self._build_prompt(text)
        
        try:
            transformed_text = await self.model.generate(
                prompt=prompt,
                temperature=0.7,  # Allow some creativity while maintaining coherence
                max_tokens=len(text.split()) * 2  # Reasonable limit for transformation
            )
            return transformed_text.strip()
            
        except Exception as e:
            raise ModelError(f"Failed to transform text: {str(e)}")
            
    async def batch_augment(self, texts: List[str]) -> List[str]:
        """Transform multiple texts to match the specified persona.
        
        Args:
            texts: List of original texts to transform.
            
        Returns:
            List of transformed texts in the persona's style.
            
        Raises:
            ModelError: If the transformation fails.
        """
        results = []
        for text in texts:
            transformed = await self.augment(text)
            results.append(transformed)
        return results
        
    def update_template(
        self,
        template: str,
        validate: bool = True
    ) -> None:
        """Update the prompt template.
        
        Args:
            template: New template string.
            validate: Whether to validate the template variables.
            
        Raises:
            ValueError: If the template is invalid.
        """
        self.prompt_template = template
        if validate:
            self._validate_template()
            
    def get_template_variables(self) -> Dict[str, Any]:
        """Get the current values of all template variables.
        
        Returns:
            Dictionary of template variable names and their values.
        """
        return {
            "persona_name": self.persona_name,
            "persona_description": self.persona_description,
            "persona_traits": self.persona_traits,
            "persona_examples": self.persona_examples,
            "style_instructions": self.style_instructions,
            "format_instructions": self.format_instructions
        }
        
    def __repr__(self) -> str:
        """Return string representation of the noiser."""
        return (
            f"PersonaNoiser(persona='{self.persona_name}', "
            f"model={self.model.__class__.__name__})"
        )
