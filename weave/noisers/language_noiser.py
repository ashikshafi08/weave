"""Language-specific noiser for multilingual text transformation."""

from typing import Any, Dict, List, Optional, Union
from ..core.base_noiser import BaseNoiser
from ..core.model_connector import ModelConnector
import json
from pathlib import Path


class LanguageNoiser(BaseNoiser):
    """Noiser for language-specific transformations and errors.
    
    This noiser can:
    1. Introduce language-specific errors
    2. Handle multilingual text
    3. Apply language-specific transformations
    """
    
    def __init__(self,
                 model_connector: ModelConnector,
                 language_config: Optional[Dict[str, Any]] = None):
        """Initialize the language noiser.
        
        Args:
            model_connector: LLM connector for transformations
            language_config: Configuration for language handling
        """
        super().__init__()
        self.model = model_connector
        self.language_config = language_config or {}
        
        # Load language-specific error patterns
        self.error_patterns = {
            "en": {
                "grammar": ["subject-verb agreement", "article usage", "preposition"],
                "spelling": ["common misspellings", "phonetic errors"],
                "punctuation": ["comma splices", "apostrophe misuse"]
            },
            "es": {
                "grammar": ["gender agreement", "ser/estar usage", "subjunctive"],
                "spelling": ["accent marks", "silent h"],
                "punctuation": ["inverted punctuation", "quotation marks"]
            },
            # Add more languages as needed
        }
        
        # Add custom error patterns from config
        if "custom_patterns" in self.language_config:
            for lang, patterns in self.language_config["custom_patterns"].items():
                if lang not in self.error_patterns:
                    self.error_patterns[lang] = {}
                self.error_patterns[lang].update(patterns)
                
    def augment(self, query: str) -> str:
        """Apply language-specific transformation to a single query.
        
        Args:
            query: Original text to transform
            
        Returns:
            Transformed text with language-specific modifications
        """
        language = self.language_config.get("language", "en")
        error_types = self.language_config.get("error_types", ["grammar"])
        error_rate = self.language_config.get("error_rate", 0.3)
        
        if language not in self.error_patterns:
            raise ValueError(f"Unsupported language: {language}")
            
        # Build prompt for language-specific transformation
        prompt = f"""Transform this text by introducing {language} language errors:
        Error types: {', '.join(error_types)}
        Error rate: {error_rate}
        
        Original text:
        {query}
        
        Instructions:
        1. Maintain the original meaning
        2. Introduce natural language errors
        3. Keep the error rate at approximately {error_rate * 100}%
        """
        
        response = self.model.generate(
            prompt=prompt,
            max_tokens=self.language_config.get("max_tokens", 150),
            temperature=self.language_config.get("temperature", 0.7)
        )
        
        return response.strip()
        
    def batch_augment(self, queries: List[str]) -> List[str]:
        """Apply language transformation to multiple queries.
        
        Args:
            queries: List of original texts
            
        Returns:
            List of transformed texts
        """
        return [self.augment(q) for q in queries]
        
    def add_language_patterns(self, language: str, patterns: Dict[str, List[str]]) -> None:
        """Add custom error patterns for a language.
        
        Args:
            language: Language code (e.g., 'en', 'es')
            patterns: Dictionary of error patterns
        """
        if language not in self.error_patterns:
            self.error_patterns[language] = {}
        self.error_patterns[language].update(patterns)
        
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages.
        
        Returns:
            List of language codes
        """
        return list(self.error_patterns.keys())
        
    def get_augmentation_metadata(self) -> Dict[str, Any]:
        """Get metadata about the language transformation.
        
        Returns:
            Dictionary containing language noiser configuration
        """
        return {
            "language": self.language_config.get("language", "en"),
            "error_types": self.language_config.get("error_types", ["grammar"]),
            "error_rate": self.language_config.get("error_rate", 0.3),
            "supported_languages": self.get_supported_languages()
        }
