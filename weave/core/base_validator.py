from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

class BaseValidator(ABC):
    """Abstract base class for all validators in the Weave framework.
    
    Validators are responsible for scoring the quality, correctness, or other
    metrics of generated answers. This could include rule-based checks,
    LLM-based validation, or composite scoring approaches.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the validator with optional configuration.
        
        Args:
            config: Optional dictionary containing validator-specific configuration.
        """
        self.config = config or {}
        
    @abstractmethod
    def score(
        self,
        query: str,
        reference: Union[str, Dict[str, Any]],
        generated: Union[str, Dict[str, Any]]
    ) -> float:
        """Score a single generated answer against its reference.
        
        Args:
            query: The original query/prompt.
            reference: The reference/ground truth answer.
            generated: The generated/predicted answer to validate.
            
        Returns:
            float: Score in [0.0, 1.0] indicating quality/correctness.
            
        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError
        
    @abstractmethod
    def batch_score(
        self,
        queries: List[str],
        references: List[Union[str, Dict[str, Any]]],
        generateds: List[Union[str, Dict[str, Any]]]
    ) -> List[float]:
        """Score multiple generated answers against their references.
        
        Args:
            queries: List of original queries/prompts.
            references: List of reference/ground truth answers.
            generateds: List of generated/predicted answers to validate.
            
        Returns:
            List[float]: Scores in [0.0, 1.0] for each sample.
            
        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError
        
    def get_validation_metadata(self) -> Dict[str, Any]:
        """Get metadata about the validation process.
        
        This can include information about what checks were performed,
        their weights if using composite validation, etc.
        
        Returns:
            Dictionary containing metadata about the validation.
        """
        return {
            "validator_type": self.__class__.__name__,
            "config": self.config
        }
        
    def validate_config(self) -> bool:
        """Validate the validator's configuration.
        
        Returns:
            bool: True if configuration is valid, False otherwise.
        """
        return True  # Default implementation assumes valid config
        
    def __repr__(self) -> str:
        """Return string representation of the validator."""
        return f"{self.__class__.__name__}(config={self.config})" 