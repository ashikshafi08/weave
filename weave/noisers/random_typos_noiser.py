"""Random typo generation for text noising in the Weave framework."""

import random
from typing import Any, Dict, List, Optional, Union
from ..core import BaseNoiser, ModelError
from ..llms import OpenAILLM, HuggingFaceLLM

class RandomTyposNoiser(BaseNoiser):
    """A noiser that introduces realistic typos and errors into text.
    
    This noiser uses LLMs to introduce natural, human-like typos and errors
    while maintaining readability and semantic meaning.
    """
    
    DEFAULT_TEMPLATE = """
    Introduce realistic, human-like typos and errors into the text below.
    
    Error types to consider:
    ${error_types}
    
    Error frequency: ${error_frequency}
    Error severity: ${error_severity}
    
    Additional instructions:
    ${instructions}
    
    Original text:
    ${original_text}
    
    Text with typos:
    """
    
    DEFAULT_ERROR_TYPES = [
        "Misspellings (e.g., 'teh' instead of 'the')",
        "Missing letters (e.g., 'th' instead of 'the')",
        "Transposed letters (e.g., 'thsi' instead of 'this')",
        "Wrong case (e.g., 'Javascript' instead of 'JavaScript')",
        "Common homophone errors (e.g., 'their' vs 'there')",
        "Double letters (e.g., 'commmon' instead of 'common')",
        "Missing spaces or extra spaces",
        "Missing or wrong punctuation"
    ]
    
    DEFAULT_INSTRUCTIONS = """
    - Keep the text readable and understandable
    - Maintain code blocks and technical terms mostly intact
    - Make errors look natural and human-like
    - Preserve the overall structure and meaning
    - Don't change numbers or key technical information
    """
    
    def __init__(
        self,
        model_connector: Optional[Union[OpenAILLM, HuggingFaceLLM]] = None,
        error_types: Optional[List[str]] = None,
        error_frequency: str = "moderate",
        error_severity: str = "mild",
        instructions: Optional[str] = None,
        prompt_template: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the RandomTyposNoiser.
        
        Args:
            model_connector: LLM provider to use for text transformation.
            error_types: List of error types to introduce.
            error_frequency: How often to introduce errors ("low", "moderate", "high").
            error_severity: How severe the errors should be ("mild", "moderate", "severe").
            instructions: Custom instructions for error introduction.
            prompt_template: Custom template for the transformation prompt.
            config: Additional configuration options.
        """
        super().__init__(config)
        
        # Set up model connector
        self.model = model_connector or OpenAILLM()
        
        # Error configuration
        self.error_types = error_types or self.DEFAULT_ERROR_TYPES
        self.error_frequency = error_frequency
        self.error_severity = error_severity
        
        # Template configuration
        self.prompt_template = prompt_template or self.DEFAULT_TEMPLATE
        self.instructions = instructions or self.DEFAULT_INSTRUCTIONS
        
        # Validate configuration
        self._validate_config()
        
    def _validate_config(self) -> None:
        """Validate the noiser configuration."""
        valid_frequencies = {"low", "moderate", "high"}
        valid_severities = {"mild", "moderate", "severe"}
        
        if self.error_frequency not in valid_frequencies:
            raise ValueError(
                f"Invalid error frequency: {self.error_frequency}. "
                f"Must be one of: {valid_frequencies}"
            )
            
        if self.error_severity not in valid_severities:
            raise ValueError(
                f"Invalid error severity: {self.error_severity}. "
                f"Must be one of: {valid_severities}"
            )
            
        # Validate template variables
        required_vars = {
            "${error_types}", "${error_frequency}", "${error_severity}",
            "${instructions}", "${original_text}"
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
            "${error_types}", "\n".join(f"- {et}" for et in self.error_types)
        ).replace(
            "${error_frequency}", self.error_frequency
        ).replace(
            "${error_severity}", self.error_severity
        ).replace(
            "${instructions}", self.instructions
        ).replace(
            "${original_text}", text
        )
        
    async def augment(self, text: str) -> str:
        """Introduce typos and errors into the text.
        
        Args:
            text: Original text to transform.
            
        Returns:
            Text with introduced typos and errors.
            
        Raises:
            ModelError: If the transformation fails.
        """
        prompt = self._build_prompt(text)
        
        try:
            transformed_text = await self.model.generate(
                prompt=prompt,
                temperature=0.8,  # Higher temperature for more random errors
                max_tokens=len(text.split()) * 2  # Reasonable limit for transformation
            )
            return transformed_text.strip()
            
        except Exception as e:
            raise ModelError(f"Failed to introduce typos: {str(e)}")
            
    async def batch_augment(self, texts: List[str]) -> List[str]:
        """Introduce typos and errors into multiple texts.
        
        Args:
            texts: List of original texts to transform.
            
        Returns:
            List of texts with introduced typos and errors.
            
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
            self._validate_config()
            
    def get_config(self) -> Dict[str, Any]:
        """Get the current noiser configuration.
        
        Returns:
            Dictionary of configuration values.
        """
        return {
            "error_types": self.error_types,
            "error_frequency": self.error_frequency,
            "error_severity": self.error_severity,
            "instructions": self.instructions
        }
        
    def __repr__(self) -> str:
        """Return string representation of the noiser."""
        return (
            f"RandomTyposNoiser(frequency='{self.error_frequency}', "
            f"severity='{self.error_severity}', "
            f"model={self.model.__class__.__name__})"
        )
