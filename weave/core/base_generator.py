from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

class BaseGenerator(ABC):
    """Abstract base class for all data generators in the Weave framework.
    
    This class defines the interface that all data generators must implement.
    Generators are responsible for producing synthetic data samples along with
    their reference answers.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the generator with optional configuration.
        
        Args:
            config: Optional dictionary containing generator-specific configuration.
        """
        self.config = config or {}
        
    @abstractmethod
    def generate(self) -> Tuple[str, Union[str, Dict[str, Any]]]:
        """Generate a single synthetic data sample.
        
        Returns:
            A tuple containing:
                - query (str): The generated query/prompt/question
                - reference (Union[str, Dict]): The reference answer. Can be a string
                  for simple tasks or a dictionary for complex outputs.
                  
        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError
        
    @abstractmethod
    def batch_generate(self, batch_size: int) -> list[Tuple[str, Union[str, Dict[str, Any]]]]:
        """Generate multiple synthetic data samples.
        
        Args:
            batch_size: Number of samples to generate.
            
        Returns:
            List of (query, reference) tuples.
            
        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError
        
    def validate_config(self) -> bool:
        """Validate the generator's configuration.
        
        Returns:
            bool: True if configuration is valid, False otherwise.
        """
        return True  # Default implementation assumes valid config
        
    def __repr__(self) -> str:
        """Return string representation of the generator."""
        return f"{self.__class__.__name__}(config={self.config})" 