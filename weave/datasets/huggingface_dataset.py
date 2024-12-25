"""Dataset loader for HuggingFace datasets."""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import pandas as pd
from .base_dataset import BaseDataset


class HuggingFaceDataset(BaseDataset):
    """Dataset loader for HuggingFace Hub datasets."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the HuggingFace dataset loader.
        
        Args:
            config: Configuration for dataset loading
        """
        super().__init__(config)
        self.dataset = None
        
    def load(self, source: str) -> "HuggingFaceDataset":
        """Load dataset from HuggingFace Hub.
        
        Args:
            source: Dataset name on HuggingFace Hub
            
        Returns:
            Self for method chaining
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "Please install datasets package: pip install datasets"
            )
            
        # Parse dataset path
        path_parts = source.split("/")
        if len(path_parts) < 2:
            raise ValueError(
                "Source should be in format: 'owner/dataset' or "
                "'owner/dataset/config'"
            )
            
        # Load dataset
        config = self.config.get("config")
        split = self.config.get("split", "train")
        
        self.dataset = load_dataset(
            path=source,
            name=config,
            split=split
        )
        
        # Convert to pandas DataFrame
        self.data = pd.DataFrame(self.dataset)
        
        return self
        
    def preprocess(self) -> "HuggingFaceDataset":
        """Apply preprocessing specific to HuggingFace datasets.
        
        Returns:
            Self for method chaining
        """
        if self.data is None:
            raise ValueError("Dataset must be loaded before preprocessing")
            
        # Handle common HuggingFace dataset preprocessing
        
        # 1. Convert special sequence types
        for col in self.data.columns:
            if self.data[col].dtype == 'object':
                # Convert sequence columns to lists
                if hasattr(self.data[col].iloc[0], '__iter__'):
                    self.data[col] = self.data[col].apply(list)
                    
        # 2. Apply configured transformations
        transforms = self.config.get("transforms", {})
        for col, transform in transforms.items():
            if col in self.data.columns:
                if transform == "join_tokens":
                    self.data[col] = self.data[col].apply(
                        lambda x: " ".join(x) if isinstance(x, list) else x
                    )
                elif transform == "flatten_list":
                    self.data[col] = self.data[col].apply(
                        lambda x: x[0] if isinstance(x, list) else x
                    )
                    
        return self
        
    def get_features(self) -> Dict[str, Any]:
        """Get dataset feature information.
        
        Returns:
            Dictionary of feature names and types
        """
        if self.dataset is None:
            raise ValueError("Dataset must be loaded first")
        return self.dataset.features
        
    def get_splits(self) -> List[str]:
        """Get available dataset splits.
        
        Returns:
            List of split names
        """
        if self.dataset is None:
            raise ValueError("Dataset must be loaded first")
        return self.dataset.split
        
    def to_huggingface(self) -> Any:
        """Convert back to HuggingFace Dataset format.
        
        Returns:
            HuggingFace Dataset object
        """
        if self.data is None:
            raise ValueError("Dataset must be loaded first")
            
        try:
            from datasets import Dataset
        except ImportError:
            raise ImportError(
                "Please install datasets package: pip install datasets"
            )
            
        return Dataset.from_pandas(self.data)
