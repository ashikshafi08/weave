"""Dataset loader with support for multiple data sources."""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import pandas as pd
import requests
from abc import ABC, abstractmethod
import json
import sqlite3
from .base_dataset import BaseDataset


class DatasetLoader(BaseDataset):
    """Dataset loader with support for multiple data formats and sources."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the dataset loader.
        
        Args:
            config: Optional configuration for data loading.
        """
        super().__init__(config)
        self.source_type: Optional[str] = None
        
    def load(self, source: Union[str, Path]) -> "DatasetLoader":
        """Load data from various sources.
        
        Supports:
        - Local files (CSV, JSON, Parquet)
        - URLs (CSV, JSON)
        - SQL databases
        - Kaggle datasets
        
        Args:
            source: Path, URL, or connection string to the data.
            
        Returns:
            Self for method chaining.
        """
        source = str(source)
        
        if source.startswith(("http://", "https://")):
            self._load_from_url(source)
        elif source.startswith("sqlite://"):
            self._load_from_sql(source)
        elif source.startswith("kaggle://"):
            self._load_from_kaggle(source)
        else:
            self._load_from_file(source)
            
        return self
        
    def _load_from_file(self, path: str) -> None:
        """Load data from a local file."""
        path = Path(path)
        suffix = path.suffix.lower()
        
        if suffix == ".csv":
            self.data = pd.read_csv(path)
        elif suffix == ".json":
            self.data = pd.read_json(path)
        elif suffix == ".parquet":
            self.data = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
            
    def _load_from_url(self, url: str) -> None:
        """Load data from a URL."""
        response = requests.get(url)
        response.raise_for_status()
        
        if url.endswith(".csv"):
            self.data = pd.read_csv(pd.io.common.StringIO(response.text))
        elif url.endswith(".json"):
            self.data = pd.DataFrame(response.json())
        else:
            raise ValueError("URL must end with .csv or .json")
            
    def _load_from_sql(self, connection_string: str) -> None:
        """Load data from a SQL database."""
        # Remove sqlite:// prefix
        db_path = connection_string[9:]
        query = self.config.get("query", "SELECT * FROM main")
        
        with sqlite3.connect(db_path) as conn:
            self.data = pd.read_sql_query(query, conn)
            
    def _load_from_kaggle(self, dataset_path: str) -> None:
        """Load data from Kaggle.
        
        Format: kaggle://username/dataset-name/file.csv
        """
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError:
            raise ImportError("Please install kaggle package: pip install kaggle")
            
        # Remove kaggle:// prefix and split path
        _, username, dataset_name, filename = dataset_path.split("/")
        
        api = KaggleApi()
        api.authenticate()
        
        # Download to temporary file
        temp_dir = Path.home() / ".weave" / "kaggle_cache"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        api.dataset_download_file(
            f"{username}/{dataset_name}",
            filename,
            path=str(temp_dir)
        )
        
        # Load the downloaded file
        self._load_from_file(temp_dir / filename)
        
    def preprocess(self) -> "DatasetLoader":
        """Apply basic preprocessing steps.
        
        Override this method for custom preprocessing.
        """
        if self.data is not None:
            # Handle missing values
            self.data = self.data.fillna(self.config.get("fill_value", 0))
            
            # Drop duplicates if configured
            if self.config.get("drop_duplicates", False):
                self.data = self.data.drop_duplicates()
                
            # Apply type conversions
            type_conversions = self.config.get("type_conversions", {})
            for column, dtype in type_conversions.items():
                if column in self.data.columns:
                    self.data[column] = self.data[column].astype(dtype)
                    
        return self
