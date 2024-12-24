class WeaveError(Exception):
    """Base exception class for all Weave-related errors."""
    pass

class ConfigurationError(WeaveError):
    """Raised when there's an error in component configuration."""
    pass

class ValidationError(WeaveError):
    """Raised when validation fails or produces invalid results."""
    pass

class GenerationError(WeaveError):
    """Raised when data generation fails."""
    pass

class NoiserError(WeaveError):
    """Raised when data augmentation/noising fails."""
    pass

class ModelError(WeaveError):
    """Raised when there's an error with model operations (API calls, etc.)."""
    pass

class ModelConnectionError(ModelError):
    """Raised when connecting to a model endpoint fails."""
    pass

class ModelAPIError(ModelError):
    """Raised when an API call to a model service fails."""
    pass

class ModelTokenLimitError(ModelError):
    """Raised when a prompt exceeds the model's token limit."""
    pass

class PipelineError(WeaveError):
    """Raised when there's an error in the orchestration pipeline."""
    pass

class DataError(WeaveError):
    """Raised when there's an issue with data format or content."""
    pass

class StorageError(WeaveError):
    """Raised when there's an error storing or retrieving data."""
    pass

class ResourceExhaustedError(WeaveError):
    """Raised when a resource limit is reached (API quota, memory, etc.)."""
    pass

class InvalidArgumentError(WeaveError):
    """Raised when an invalid argument is passed to a function."""
    def __init__(self, argument_name: str, message: str):
        self.argument_name = argument_name
        super().__init__(f"Invalid argument '{argument_name}': {message}")

class NotImplementedInBaseClassError(WeaveError):
    """Raised when an abstract method is not implemented by a subclass."""
    def __init__(self, class_name: str, method_name: str):
        super().__init__(
            f"Method '{method_name}' must be implemented by subclass of '{class_name}'"
        ) 