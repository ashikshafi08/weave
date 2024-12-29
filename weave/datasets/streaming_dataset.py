"""Streaming dataset for handling large-scale data."""

from typing import Any, Dict, List, Optional, Union, Iterator
from pathlib import Path
import pandas as pd
from .base_dataset import BaseDataset
import json
import csv
from itertools import islice


class StreamingDataset(BaseDataset):
    """Dataset handler for streaming large datasets.
    
    Features:
    - Memory-efficient loading of large files
    - Streaming processing of data
    - Support for various file formats
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the streaming dataset.
        
        Args:
            config: Configuration for streaming behavior
        """
        super().__init__(config)
        self.chunk_size = config.get("chunk_size", 1000)
        self.stream = None
        self._file_handle = None
        
    def load(self, source: Union[str, Path]) -> "StreamingDataset":
        """Set up streaming from a data source.
        
        Args:
            source: Path to data file
            
        Returns:
            Self for method chaining
        """
        source = Path(source)
        
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source}")
            
        # Close any existing file handle
        if self._file_handle is not None:
            self._file_handle.close()
            
        self._file_handle = source.open("r")
        
        # Set up appropriate stream based on file type
        if source.suffix == ".csv":
            self.stream = self._stream_csv()
        elif source.suffix == ".json":
            self.stream = self._stream_json()
        elif source.suffix == ".jsonl":
            self.stream = self._stream_jsonl()
        else:
            raise ValueError(f"Unsupported file format: {source.suffix}")
            
        return self
        
    def _stream_csv(self) -> Iterator[Dict[str, Any]]:
        """Stream CSV file."""
        reader = csv.DictReader(self._file_handle)
        for row in reader:
            yield row
            
    def _stream_json(self) -> Iterator[Dict[str, Any]]:
        """Stream JSON file (assumes array of objects)."""
        data = json.load(self._file_handle)
        if not isinstance(data, list):
            raise ValueError("JSON file must contain an array of objects")
        for item in data:
            yield item
            
    def _stream_jsonl(self) -> Iterator[Dict[str, Any]]:
        """Stream JSONL file."""
        for line in self._file_handle:
            yield json.loads(line)
            
    def preprocess(self) -> "StreamingDataset":
        """Set up preprocessing for streamed data.
        
        Returns:
            Self for method chaining
        """
        # Configure preprocessing steps
        self.preprocessors = []
        
        # Add configured preprocessing steps
        if "fill_na" in self.config:
            self.preprocessors.append(
                lambda x: {k: v or self.config["fill_na"] for k, v in x.items()}
            )
            
        if "drop_columns" in self.config:
            cols_to_drop = set(self.config["drop_columns"])
            self.preprocessors.append(
                lambda x: {k: v for k, v in x.items() if k not in cols_to_drop}
            )
            
        return self
        
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over the dataset in chunks.
        
        Yields:
            Dictionary of record data
        """
        if self.stream is None:
            raise ValueError("Dataset must be loaded first")
            
        for record in self.stream:
            # Apply preprocessing
            for preprocessor in self.preprocessors:
                record = preprocessor(record)
            yield record
            
    def iter_chunks(self) -> Iterator[pd.DataFrame]:
        """Iterate over the dataset in DataFrame chunks.
        
        Yields:
            Pandas DataFrame containing chunk_size records
        """
        chunk = []
        for record in self:
            chunk.append(record)
            if len(chunk) >= self.chunk_size:
                yield pd.DataFrame(chunk)
                chunk = []
                
        # Yield remaining records
        if chunk:
            yield pd.DataFrame(chunk)
            
    def take(self, n: int) -> pd.DataFrame:
        """Take first n records from the stream.
        
        Args:
            n: Number of records to take
            
        Returns:
            DataFrame containing n records
        """
        records = list(islice(self, n))
        return pd.DataFrame(records)
        
    def to_pandas(self) -> pd.DataFrame:
        """Convert entire stream to DataFrame.
        
        Warning: This loads all data into memory.
        
        Returns:
            DataFrame containing all records
        """
        return pd.DataFrame(list(self))
        
    def __enter__(self) -> "StreamingDataset":
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None
            
    def get_augmentation_metadata(self) -> Dict[str, Any]:
        """Get metadata about the streaming configuration.
        
        Returns:
            Dictionary containing streaming settings
        """
        return {
            "chunk_size": self.chunk_size,
            "preprocessors": [p.__name__ for p in self.preprocessors],
            "streaming_active": self.stream is not None
        }
