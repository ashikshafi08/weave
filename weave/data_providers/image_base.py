from .base import DataProvider

class ImageDataProvider(DataProvider):
    def get_data_type(self) -> str:
        return "image"

    @abstractmethod
    async def get_image_path(self, data_point: Dict[str, Any]) -> str:
        """Get the path to the image file for a data point."""
        pass