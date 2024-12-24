from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

class BaseNoiser(ABC):
    """Abstract base class for all data noisers/augmenters in the Weave framework.
    
    Noisers are responsible for applying transformations, augmentations, or
    "noise" to the generated data. This could include style changes, typos,
    persona modifications, etc.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the noiser with optional configuration.
        
        Args:
            config: Optional dictionary containing noiser-specific configuration.
        """
        self.config = config or {}
        
    @abstractmethod
    def augment(self, query: str) -> str:
        """Apply noise/augmentation to a single query.
        
        Args:
            query: The original query string to be transformed.
            
        Returns:
            The transformed/noised query string.
            
        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError
        
    @abstractmethod
    def batch_augment(self, queries: List[str]) -> List[str]:
        """Apply noise/augmentation to multiple queries.
        
        Args:
            queries: List of original query strings.
            
        Returns:
            List of transformed/noised query strings.
            
        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError
        
    def get_augmentation_metadata(self) -> Dict[str, Any]:
        """Get metadata about the applied augmentation.
        
        This can include information about what transformations were applied,
        their parameters, etc.
        
        Returns:
            Dictionary containing metadata about the augmentation.
        """
        return {
            "noiser_type": self.__class__.__name__,
            "config": self.config
        }
        
    def validate_config(self) -> bool:
        """Validate the noiser's configuration.
        
        Returns:
            bool: True if configuration is valid, False otherwise.
        """
        return True  # Default implementation assumes valid config
        
    def __repr__(self) -> str:
        """Return string representation of the noiser."""
        return f"{self.__class__.__name__}(config={self.config})" 