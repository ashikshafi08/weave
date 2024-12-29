"""Weave: A framework for AI-powered synthetic data generation and augmentation."""

from .core import (
    BaseGenerator,
    BaseNoiser,
    BaseValidator,
    BaseOrchestrator,
    ModelConnector,
    ModelType
)

from .datasets import (
    BaseDataset,
    DatasetLoader,
    DatasetMerger,
    HuggingFaceDataset,
    StreamingDataset
)

from .generators import (
    MathGenerator,
    CodeGenerator,
    NLUGenerator
)

from .llms import (
    OpenAILLM,
    HuggingFaceLLM
)

from .noisers import (
    ContextNoiser,
    # Add other noisers here
)

__version__ = "0.1.0"

__all__ = [
    # Core
    "BaseGenerator",
    "BaseNoiser", 
    "BaseValidator",
    "BaseOrchestrator",
    "ModelConnector",
    "ModelType",
    
    # Datasets
    "BaseDataset",
    "DatasetLoader",
    "DatasetMerger",
    "HuggingFaceDataset",
    "StreamingDataset",
    
    # Generators
    "MathGenerator",
    "CodeGenerator", 
    "NLUGenerator",
    
    # LLMs
    "OpenAILLM",
    "HuggingFaceLLM",
    
    # Noisers
    "ContextNoiser"
]
