"""Sentiment-based text transformation noiser."""

from typing import Any, Dict, List, Optional, Union
from ..core.base_noiser import BaseNoiser
from ..core.model_connector import ModelConnector


class SentimentNoiser(BaseNoiser):
    """Noiser for adjusting text sentiment and emotional tone."""
    
    def __init__(self,
                 model_connector: ModelConnector,
                 sentiment_config: Optional[Dict[str, Any]] = None):
        """Initialize the sentiment noiser.
        
        Args:
            model_connector: LLM connector for transformations
            sentiment_config: Configuration for sentiment adjustments
        """
        super().__init__()
        self.model = model_connector
        self.config = sentiment_config or {}
        
        self.sentiment_scales = {
            "polarity": [-1.0, 1.0],  # negative to positive
            "intensity": [0.0, 1.0],   # neutral to intense
            "formality": [0.0, 1.0],   # casual to formal
            "objectivity": [0.0, 1.0]  # subjective to objective
        }
        
        # Sentiment transformation templates
        self.templates = {
            "positive": "Transform this text to express a more positive sentiment while preserving the core meaning",
            "negative": "Transform this text to express a more negative sentiment while preserving the core meaning",
            "neutral": "Rewrite this text in a more neutral and objective tone",
            "formal": "Rewrite this text in a more formal and professional tone",
            "casual": "Rewrite this text in a more casual and conversational tone",
            "intense": "Amplify the emotional intensity of this text while maintaining its sentiment direction"
        }
        
    def augment(self, query: str) -> str:
        """Apply sentiment transformation to text.
        
        Args:
            query: Original text
            
        Returns:
            Transformed text with adjusted sentiment
        """
        target_sentiment = self.config.get("target_sentiment", "neutral")
        intensity = self.config.get("intensity", 0.5)
        preserve_keywords = self.config.get("preserve_keywords", [])
        
        if target_sentiment not in self.templates:
            raise ValueError(f"Unknown sentiment: {target_sentiment}")
            
        # Build prompt with sentiment instructions
        prompt = f"""{self.templates[target_sentiment]}
        
        Intensity: {intensity}
        Keywords to preserve: {', '.join(preserve_keywords)}
        
        Original text:
        {query}
        
        Instructions:
        1. Maintain the core meaning and key information
        2. Adjust sentiment towards {target_sentiment}
        3. Scale intensity to {intensity}
        4. Preserve these keywords: {preserve_keywords}
        """
        
        response = self.model.generate(
            prompt=prompt,
            max_tokens=self.config.get("max_tokens", 150),
            temperature=self.config.get("temperature", 0.7)
        )
        
        return response.strip()
        
    def batch_augment(self, queries: List[str]) -> List[str]:
        """Apply sentiment transformation to multiple texts.
        
        Args:
            queries: List of original texts
            
        Returns:
            List of transformed texts
        """
        return [self.augment(q) for q in queries]
        
    def adjust_sentiment(self,
                        text: str,
                        target_sentiment: Dict[str, float]) -> str:
        """Fine-grained sentiment adjustment using multiple scales.
        
        Args:
            text: Original text
            target_sentiment: Dictionary mapping sentiment scales to target values
            
        Returns:
            Transformed text with adjusted sentiment
        """
        # Validate sentiment values
        for scale, value in target_sentiment.items():
            if scale not in self.sentiment_scales:
                raise ValueError(f"Unknown sentiment scale: {scale}")
            min_val, max_val = self.sentiment_scales[scale]
            if not min_val <= value <= max_val:
                raise ValueError(
                    f"Value {value} for {scale} outside valid range "
                    f"[{min_val}, {max_val}]"
                )
                
        # Build prompt for fine-grained control
        prompt = f"""Transform this text according to these sentiment parameters:
        
        Text: {text}
        
        Target sentiment levels:
        {json.dumps(target_sentiment, indent=2)}
        
        Instructions:
        1. Maintain core meaning and structure
        2. Adjust each sentiment dimension:
           - Polarity (negative to positive)
           - Intensity (neutral to intense)
           - Formality (casual to formal)
           - Objectivity (subjective to objective)
        """
        
        response = self.model.generate(
            prompt=prompt,
            max_tokens=self.config.get("max_tokens", 150),
            temperature=self.config.get("temperature", 0.7)
        )
        
        return response.strip()
        
    def get_sentiment_scales(self) -> Dict[str, List[float]]:
        """Get available sentiment adjustment scales.
        
        Returns:
            Dictionary of sentiment scales and their ranges
        """
        return self.sentiment_scales.copy()
        
    def get_available_templates(self) -> List[str]:
        """Get list of available sentiment templates.
        
        Returns:
            List of template names
        """
        return list(self.templates.keys())
        
    def get_augmentation_metadata(self) -> Dict[str, Any]:
        """Get metadata about the sentiment transformation.
        
        Returns:
            Dictionary containing sentiment noiser configuration
        """
        return {
            "target_sentiment": self.config.get("target_sentiment", "neutral"),
            "intensity": self.config.get("intensity", 0.5),
            "available_templates": self.get_available_templates(),
            "sentiment_scales": self.get_sentiment_scales()
        }
