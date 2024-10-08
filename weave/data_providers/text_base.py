from .base import DataProvider

class TextDataProvider(DataProvider):
    def get_data_type(self) -> str:
        return "text"

    @abstractmethod
    async def get_text_content(self, data_point: Dict[str, Any]) -> str:
        """Extract text content from a data point."""
        pass