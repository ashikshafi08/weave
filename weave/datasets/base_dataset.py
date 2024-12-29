"""Base dataset class for Weave framework."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import pandas as pd


class BaseDataset(ABC):
    """Abstract base class for all datasets in the Weave framework.
    
    This class provides the interface for dataset operations including loading,
    preprocessing, splitting, and merging with synthetic data.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the dataset with optional configuration.
        
        Args:
            config: Optional dictionary containing dataset-specific configuration.
        """
        self.config = config or {}
        self.data: Optional[pd.DataFrame] = None
        
    @abstractmethod
    def load(self, source: Union[str, Path]) -> "BaseDataset":
        """Load data from a source.
        
        Args:
            source: Path or URL to the data source.
            
        Returns:
            Self for method chaining.
        """
        raise NotImplementedError
        
    @abstractmethod
    def preprocess(self) -> "BaseDataset":
        """Apply preprocessing steps to the dataset.
        
        Returns:
            Self for method chaining.
        """
        raise NotImplementedError
        
    def split(self, 
             train_ratio: float = 0.8,
             val_ratio: float = 0.1,
             test_ratio: float = 0.1,
             shuffle: bool = True,
             seed: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """Split dataset into train, validation, and test sets.
        
        Args:
            train_ratio: Proportion of data for training.
            val_ratio: Proportion of data for validation.
            test_ratio: Proportion of data for testing.
            shuffle: Whether to shuffle data before splitting.
            seed: Random seed for reproducibility.
            
        Returns:
            Dictionary containing train, val, and test DataFrames.
        """
        assert self.data is not None, "Data must be loaded before splitting"
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, \
            "Split ratios must sum to 1"
            
        if shuffle and seed is not None:
            self.data = self.data.sample(frac=1, random_state=seed)
            
        n = len(self.data)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        return {
            "train": self.data[:train_end],
            "val": self.data[train_end:val_end],
            "test": self.data[val_end:]
        }
        
    def save(self, path: Union[str, Path], format: str = "csv") -> None:
        """Save the dataset to disk.
        
        Args:
            path: Path where to save the dataset.
            format: Format to save in (csv, json, parquet).
        """
        assert self.data is not None, "Data must be loaded before saving"
        path = Path(path)
        
        if format == "csv":
            self.data.to_csv(path, index=False)
        elif format == "json":
            self.data.to_json(path, orient="records")
        elif format == "parquet":
            self.data.to_parquet(path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
    def __len__(self) -> int:
        return 0 if self.data is None else len(self.data)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.data is None:
            raise ValueError("Data must be loaded first")
        return self.data.iloc[idx].to_dict()
