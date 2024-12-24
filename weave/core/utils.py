import os
import json
import yaml
import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import time
from functools import wraps

from .exceptions import ConfigurationError, StorageError

def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to log file.
        log_format: Optional custom log format string.
        
    Returns:
        Configured logger instance.
    """
    # Set default format if none provided
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
    # Create logger
    logger = logging.getLogger("weave")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if log_file specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Dictionary containing configuration.
        
    Raises:
        ConfigurationError: If file cannot be loaded or has invalid format.
    """
    try:
        config_path = Path(config_path)
        with open(config_path, "r") as f:
            if config_path.suffix in [".yaml", ".yml"]:
                return yaml.safe_load(f)
            elif config_path.suffix == ".json":
                return json.load(f)
            else:
                raise ConfigurationError(
                    f"Unsupported config file format: {config_path.suffix}"
                )
    except Exception as e:
        raise ConfigurationError(f"Error loading config file: {str(e)}")

def save_results(
    results: Union[Dict[str, Any], List[Dict[str, Any]]],
    output_path: Union[str, Path],
    format: str = "jsonl"
) -> None:
    """Save generation results to file.
    
    Args:
        results: Results dictionary or list of dictionaries to save.
        output_path: Path to save results to.
        format: Output format ("json", "jsonl", or "yaml").
        
    Raises:
        StorageError: If results cannot be saved.
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "jsonl":
            # Save as JSON Lines (one JSON object per line)
            with open(output_path, "w") as f:
                if isinstance(results, dict):
                    f.write(json.dumps(results) + "\n")
                else:
                    for result in results:
                        f.write(json.dumps(result) + "\n")
                        
        elif format == "json":
            # Save as regular JSON
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
                
        elif format == "yaml":
            # Save as YAML
            with open(output_path, "w") as f:
                yaml.dump(results, f)
                
        else:
            raise StorageError(f"Unsupported output format: {format}")
            
    except Exception as e:
        raise StorageError(f"Error saving results: {str(e)}")

def retry_with_exponential_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    errors: tuple = (Exception,)
):
    """Decorator for retrying functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts.
        initial_delay: Initial delay between retries in seconds.
        max_delay: Maximum delay between retries in seconds.
        exponential_base: Base for exponential backoff calculation.
        errors: Tuple of exceptions to catch and retry on.
        
    Returns:
        Decorated function that implements retry logic.
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except errors as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        raise last_exception
                        
                    # Calculate next delay with exponential backoff
                    delay = min(delay * exponential_base, max_delay)
                    
                    # Log retry attempt
                    logging.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}. "
                        f"Retrying in {delay:.1f} seconds..."
                    )
                    
                    # Wait before next attempt
                    time.sleep(delay)
                    
        return wrapper
    return decorator

def validate_config_schema(config: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """Validate configuration against a schema.
    
    Args:
        config: Configuration dictionary to validate.
        schema: Schema dictionary defining required fields and types.
        
    Returns:
        True if valid, raises ConfigurationError if invalid.
        
    Raises:
        ConfigurationError: If configuration is invalid.
    """
    try:
        import jsonschema
        jsonschema.validate(instance=config, schema=schema)
        return True
    except jsonschema.exceptions.ValidationError as e:
        raise ConfigurationError(f"Configuration validation failed: {str(e)}")

def merge_configs(
    base_config: Dict[str, Any],
    override_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Deep merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration dictionary.
        override_config: Configuration to override base with.
        
    Returns:
        Merged configuration dictionary.
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
            
    return merged

def get_timestamp() -> str:
    """Get current timestamp in ISO format.
    
    Returns:
        Timestamp string in ISO format.
    """
    from datetime import datetime
    return datetime.now().isoformat()

def calculate_metrics(scores: List[float]) -> Dict[str, float]:
    """Calculate basic statistics from a list of scores.
    
    Args:
        scores: List of validation scores.
        
    Returns:
        Dictionary containing calculated metrics (mean, std, min, max).
    """
    import numpy as np
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores))
    } 