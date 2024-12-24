"""Core module for the Weave framework.

This module provides the fundamental abstractions and utilities for synthetic
data generation, including base classes for generators, noisers, validators,
and orchestrators, as well as model connectivity and utility functions.
"""

# Base classes
from .base_generator import BaseGenerator
from .base_noiser import BaseNoiser
from .base_validator import BaseValidator
from .base_orchestrator import BaseOrchestrator

# Model connectivity
from .model_connector import ModelConnector, ModelType

# Exceptions
from .exceptions import (
    WeaveError,
    ConfigurationError,
    ValidationError,
    GenerationError,
    NoiserError,
    ModelError,
    ModelConnectionError,
    ModelAPIError,
    ModelTokenLimitError,
    PipelineError,
    DataError,
    StorageError,
    ResourceExhaustedError,
    InvalidArgumentError,
    NotImplementedInBaseClassError,
)

# Utilities
from .utils import (
    setup_logging,
    load_config,
    save_results,
    retry_with_exponential_backoff,
    validate_config_schema,
    merge_configs,
    get_timestamp,
    calculate_metrics,
)

__all__ = [
    # Base classes
    "BaseGenerator",
    "BaseNoiser",
    "BaseValidator",
    "BaseOrchestrator",
    
    # Model connectivity
    "ModelConnector",
    "ModelType",
    
    # Exceptions
    "WeaveError",
    "ConfigurationError",
    "ValidationError",
    "GenerationError",
    "NoiserError",
    "ModelError",
    "ModelConnectionError",
    "ModelAPIError",
    "ModelTokenLimitError",
    "PipelineError",
    "DataError",
    "StorageError",
    "ResourceExhaustedError",
    "InvalidArgumentError",
    "NotImplementedInBaseClassError",
    
    # Utilities
    "setup_logging",
    "load_config",
    "save_results",
    "retry_with_exponential_backoff",
    "validate_config_schema",
    "merge_configs",
    "get_timestamp",
    "calculate_metrics",
] 