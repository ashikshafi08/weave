# weave/core/data_source.py

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import aiofiles
import json
import csv
import aiosqlite
import random
import aiohttp
from .exceptions import DataSourceError

class DataSource(ABC):
    """
    Abstract base class for data sources.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data source.

        Args:
            config (Dict[str, Any]): Configuration for the data source.
        """
        self.config = config
        self.cache: Optional[List[Dict[str, Any]]] = None

    @abstractmethod
    async def fetch(self, num_samples: int) -> List[Dict[str, Any]]:
        """
        Fetch data from the source.

        Args:
            num_samples (int): Number of samples to fetch.

        Returns:
            List[Dict[str, Any]]: Fetched data.

        Raises:
            DataSourceError: If data fetching fails.
        """
        pass

    @abstractmethod
    async def load_data(self, source: str) -> None:
        """
        Load data from a specified source.

        Args:
            source (str): Source of the data (e.g., file path, database connection string, API endpoint).

        Raises:
            DataSourceError: If data loading fails.
        """
        pass

    async def _load_json(self, file_path: str) -> None:
        try:
            async with aiofiles.open(file_path, mode='r') as f:
                content = await f.read()
                self.cache = json.loads(content)
        except Exception as e:
            raise DataSourceError(f"Failed to load JSON from {file_path}: {str(e)}")

    async def _load_csv(self, file_path: str) -> None:
        try:
            self.cache = []
            async with aiofiles.open(file_path, mode='r') as f:
                reader = csv.DictReader(await f.read().splitlines())
                for row in reader:
                    self.cache.append(row)
        except Exception as e:
            raise DataSourceError(f"Failed to load CSV from {file_path}: {str(e)}")

    async def _load_sqlite(self, db_path: str, table_name: str) -> None:
        try:
            self.cache = []
            async with aiosqlite.connect(db_path) as db:
                async with db.execute(f"SELECT * FROM {table_name}") as cursor:
                    columns = [column[0] for column in cursor.description]
                    async for row in cursor:
                        self.cache.append(dict(zip(columns, row)))
        except Exception as e:
            raise DataSourceError(f"Failed to load data from SQLite database {db_path}: {str(e)}")

    async def _load_api(self, api_url: str) -> None:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(api_url) as response:
                    response.raise_for_status()
                    self.cache = await response.json()
        except Exception as e:
            raise DataSourceError(f"Failed to load data from API {api_url}: {str(e)}")

    def filter_data(self, filter_func: callable) -> None:
        """
        Filter the cached data based on a given function.

        Args:
            filter_func (callable): A function that takes a data item and returns a boolean.
        """
        if self.cache is not None:
            self.cache = list(filter(filter_func, self.cache))

    def sample_data(self, num_samples: int) -> List[Dict[str, Any]]:
        """
        Sample data from the cache.

        Args:
            num_samples (int): Number of samples to return.

        Returns:
            List[Dict[str, Any]]: Sampled data.
        """
        if self.cache is None:
            raise DataSourceError("Data not loaded. Call load_data() first.")
        
        if num_samples >= len(self.cache):
            return self.cache
        else:
            return random.sample(self.cache, num_samples)

class FileDataSource(DataSource):
    async def fetch(self, num_samples: int) -> List[Dict[str, Any]]:
        return self.sample_data(num_samples)

    async def load_data(self, source: str) -> None:
        file_extension = source.split('.')[-1].lower()
        
        if file_extension == 'json':
            await self._load_json(source)
        elif file_extension == 'csv':
            await self._load_csv(source)
        elif file_extension == 'db':
            table_name = self.config.get('table_name')
            if not table_name:
                raise DataSourceError("Table name must be specified in config for SQLite databases.")
            await self._load_sqlite(source, table_name)
        else:
            raise DataSourceError(f"Unsupported file type: {file_extension}")

class DatabaseDataSource(DataSource):
    async def fetch(self, num_samples: int) -> List[Dict[str, Any]]:
        return self.sample_data(num_samples)

    async def load_data(self, source: str) -> None:
        db_type = self.config.get('db_type', '').lower()
        table_name = self.config.get('table_name')
        
        if not table_name:
            raise DataSourceError("Table name must be specified in config for database sources.")
        
        if db_type == 'sqlite':
            await self._load_sqlite(source, table_name)
        else:
            raise DataSourceError(f"Unsupported database type: {db_type}")

class APIDataSource(DataSource):
    async def fetch(self, num_samples: int) -> List[Dict[str, Any]]:
        if self.cache is None:
            raise DataSourceError("Data not loaded. Call load_data() first.")
        
        if isinstance(self.cache, list):
            return self.sample_data(num_samples)
        elif isinstance(self.cache, dict):
            data_key = self.config.get('data_key', 'data')
            if data_key not in self.cache:
                raise DataSourceError(f"Data key '{data_key}' not found in API response")
            data = self.cache[data_key]
            return self.sample_data(num_samples) if isinstance(data, list) else data
        else:
            raise DataSourceError("Unexpected cache format")

    async def load_data(self, source: str) -> None:
        await self._load_api(source)

class DataSourceFactory:
    @staticmethod
    def create_data_source(config: Dict[str, Any]) -> DataSource:
        """
        Create a DataSource instance based on the provided configuration.

        Args:
            config (Dict[str, Any]): Configuration for the data source.

        Returns:
            DataSource: An instance of a DataSource subclass.

        Raises:
            DataSourceError: If an unsupported data source type is specified.
        """
        source_type = config.get('type', '').lower()
        
        if source_type == 'file':
            return FileDataSource(config)
        elif source_type == 'database':
            return DatabaseDataSource(config)
        elif source_type == 'api':
            return APIDataSource(config)
        else:
            raise DataSourceError(f"Unsupported data source type: {source_type}")
