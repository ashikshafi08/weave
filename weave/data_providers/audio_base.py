from .base import DataProvider

class AudioDataProvider(DataProvider):
    def get_data_type(self) -> str:
        return "audio"

    @abstractmethod
    async def get_audio_path(self, data_point: Dict[str, Any]) -> str:
        """Get the path to the audio file for a data point."""
        pass