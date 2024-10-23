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


class DataQualityError(WeaveException):
    """Exception raised when generated data doesn't meet quality requirements."""
    pass

class PromptTemplateError(WeaveException):
    """Exception raised for errors in prompt template formatting or rendering."""
    pass

class LLMResponseError(LLMError):
    """Exception raised when LLM response is invalid or cannot be parsed."""
    pass

class DataSourceEmptyError(DataSourceError):
    """Exception raised when a data source is empty or exhausted."""
    pass

class InvalidPluginError(PluginError):
    """Exception raised when a plugin doesn't implement required interfaces."""
    pass

class PipelineStageError(PipelineExecutionError):
    """Exception raised when a specific pipeline stage fails."""
    def __init__(self, stage_name: str, message: str):
        self.stage_name = stage_name
        super().__init__(f"Stage '{stage_name}' failed: {message}")