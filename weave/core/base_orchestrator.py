from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .base_generator import BaseGenerator
from .base_noiser import BaseNoiser
from .base_validator import BaseValidator

class BaseOrchestrator(ABC):
    """Abstract base class for orchestrating the synthetic data generation pipeline.
    
    The Orchestrator coordinates the flow of data through the pipeline:
    Generator -> Noiser -> Miner -> Validator -> Storage/Logging.
    """
    
    def __init__(
        self,
        generator: BaseGenerator,
        validator: BaseValidator,
        noiser: Optional[BaseNoiser] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the orchestrator with components and configuration.
        
        Args:
            generator: Instance of BaseGenerator to produce synthetic data.
            validator: Instance of BaseValidator to score generated answers.
            noiser: Optional BaseNoiser instance for data augmentation.
            config: Optional orchestrator-specific configuration.
        """
        self.generator = generator
        self.validator = validator
        self.noiser = noiser
        self.config = config or {}
        
    @abstractmethod
    def run_single(
        self,
        miner_func: Callable[[str], Union[str, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Run the pipeline for a single sample.
        
        Args:
            miner_func: Callable that takes a query string and returns an answer.
                       This could be an LLM API call or any other prediction function.
                       
        Returns:
            Dictionary containing the sample results and metadata:
                - query: Original query
                - noised_query: Query after noise/augmentation (if noiser used)
                - reference: Reference answer
                - generated: Generated answer from miner
                - score: Validation score
                - metadata: Additional metadata from components
                
        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError
        
    @abstractmethod
    def run_batch(
        self,
        miner_func: Callable[[List[str]], List[Union[str, Dict[str, Any]]]],
        batch_size: int
    ) -> List[Dict[str, Any]]:
        """Run the pipeline for multiple samples in parallel.
        
        Args:
            miner_func: Callable that takes a list of queries and returns a list of answers.
            batch_size: Number of samples to process in parallel.
            
        Returns:
            List of dictionaries containing results and metadata for each sample.
            
        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError
        
    def validate_pipeline(self) -> bool:
        """Validate the entire pipeline configuration.
        
        Checks that all components are properly configured and compatible.
        
        Returns:
            bool: True if pipeline is valid, False otherwise.
        """
        return all([
            self.generator.validate_config(),
            self.validator.validate_config(),
            self.noiser.validate_config() if self.noiser else True,
            self.validate_config()
        ])
        
    def validate_config(self) -> bool:
        """Validate the orchestrator's configuration.
        
        Returns:
            bool: True if configuration is valid, False otherwise.
        """
        return True  # Default implementation assumes valid config
        
    def __repr__(self) -> str:
        """Return string representation of the orchestrator."""
        components = {
            "generator": self.generator.__class__.__name__,
            "validator": self.validator.__class__.__name__,
            "noiser": self.noiser.__class__.__name__ if self.noiser else None
        }
        return f"{self.__class__.__name__}(components={components}, config={self.config})" 