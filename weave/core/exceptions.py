# weave/core/exceptions.py

class WeaveException(Exception):
    """Base exception class for all Weave-related exceptions."""
    pass

class ConfigurationError(WeaveException):
    """Exception raised for configuration-related errors."""
    pass

class PipelineConfigurationError(ConfigurationError):
    """Exception raised for pipeline configuration errors."""
    pass

class PipelineExecutionError(WeaveException):
    """Exception raised for errors during pipeline execution."""
    pass

class DataSourceError(WeaveException):
    """Exception raised for errors related to data sources."""
    pass

class DataProcessingError(WeaveException):
    """Exception raised for errors during data processing."""
    pass

class DataGenerationError(WeaveException):
    """Exception raised for errors during data generation."""
    pass

class ValidationError(WeaveException):
    """Exception raised for data validation errors."""
    pass

class LLMError(WeaveException):
    """Exception raised for errors related to LLM operations."""
    pass

class RateLimitError(LLMError):
    """Exception raised when LLM rate limit is exceeded."""
    pass

class PluginError(WeaveException):
    """Exception raised for plugin-related errors."""
    pass

class PromptError(WeaveException):
    """Exception raised for errors related to prompt management."""
    pass
