"""Dataset merger for combining synthetic and real data."""

from typing import Any, Dict, List, Optional, Union
import pandas as pd
from pathlib import Path
from .base_dataset import BaseDataset


class DatasetMerger:
    """Class for merging synthetic and real datasets."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the dataset merger.
        
        Args:
            config: Optional configuration for merging behavior.
        """
        self.config = config or {}
        
    def merge(self,
             real_data: Union[BaseDataset, pd.DataFrame],
             synthetic_data: Union[BaseDataset, pd.DataFrame],
             strategy: str = "append",
             ratio: float = 0.5) -> pd.DataFrame:
        """Merge real and synthetic datasets.
        
        Args:
            real_data: Real dataset
            synthetic_data: Synthetic dataset
            strategy: Merging strategy ("append", "replace", "mix")
            ratio: Ratio of synthetic to real data for "mix" strategy
            
        Returns:
            Merged DataFrame
        """
        # Convert inputs to DataFrames if needed
        real_df = real_data.data if isinstance(real_data, BaseDataset) else real_data
        synth_df = synthetic_data.data if isinstance(synthetic_data, BaseDataset) else synthetic_data
        
        if strategy == "append":
            return pd.concat([real_df, synth_df], ignore_index=True)
            
        elif strategy == "replace":
            return synth_df
            
        elif strategy == "mix":
            # Randomly sample from both datasets according to ratio
            real_sample = real_df.sample(
                n=int(len(real_df) * (1 - ratio)),
                random_state=self.config.get("seed")
            )
            synth_sample = synth_df.sample(
                n=int(len(real_df) * ratio),
                random_state=self.config.get("seed")
            )
            return pd.concat([real_sample, synth_sample], ignore_index=True)
            
        else:
            raise ValueError(f"Unsupported merge strategy: {strategy}")
            
    def validate_compatibility(self,
                             real_data: Union[BaseDataset, pd.DataFrame],
                             synthetic_data: Union[BaseDataset, pd.DataFrame]) -> bool:
        """Validate that datasets are compatible for merging.
        
        Args:
            real_data: Real dataset
            synthetic_data: Synthetic dataset
            
        Returns:
            True if datasets are compatible
        """
        real_df = real_data.data if isinstance(real_data, BaseDataset) else real_data
        synth_df = synthetic_data.data if isinstance(synthetic_data, BaseDataset) else synthetic_data
        
        # Check column compatibility
        real_cols = set(real_df.columns)
        synth_cols = set(synth_df.columns)
        
        if not real_cols == synth_cols:
            missing = real_cols - synth_cols
            extra = synth_cols - real_cols
            raise ValueError(
                f"Column mismatch. Missing: {missing}, Extra: {extra}"
            )
            
        # Check data types
        for col in real_cols:
            if real_df[col].dtype != synth_df[col].dtype:
                raise ValueError(
                    f"Type mismatch for column {col}: "
                    f"{real_df[col].dtype} vs {synth_df[col].dtype}"
                )
                
        return True
        
    def analyze_distribution(self,
                           real_data: Union[BaseDataset, pd.DataFrame],
                           synthetic_data: Union[BaseDataset, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze statistical properties of real vs synthetic data.
        
        Args:
            real_data: Real dataset
            synthetic_data: Synthetic dataset
            
        Returns:
            Dictionary containing distribution analysis
        """
        real_df = real_data.data if isinstance(real_data, BaseDataset) else real_data
        synth_df = synthetic_data.data if isinstance(synthetic_data, BaseDataset) else synthetic_data
        
        analysis = {}
        
        # Basic statistics
        analysis["real_stats"] = real_df.describe()
        analysis["synthetic_stats"] = synth_df.describe()
        
        # Column-wise correlation
        if self.config.get("compute_correlation", True):
            analysis["real_corr"] = real_df.corr()
            analysis["synthetic_corr"] = synth_df.corr()
            
        # Data type distribution
        analysis["real_dtypes"] = real_df.dtypes.value_counts().to_dict()
        analysis["synthetic_dtypes"] = synth_df.dtypes.value_counts().to_dict()
        
        return analysis
