import os
from .audio_base import AudioDataProvider
from . import register_data_provider

@register_data_provider("audio_file")
class AudioFileProvider(AudioDataProvider):
    def __init__(self, directory: str):
        self.directory = directory
        self.audio_files = [f for f in os.listdir(directory) if f.endswith('.wav')]
        self.current_index = 0

    async def fetch(self, **kwargs) -> Optional[Dict[str, Any]]:
        if self.current_index < len(self.audio_files):
            audio_file = self.audio_files[self.current_index]
            self.current_index += 1
            return {"file_name": audio_file, "path": os.path.join(self.directory, audio_file)}
        return None

    async def fetch_batch(self, batch_size: int, **kwargs) -> List[Dict[str, Any]]:
        batch = []
        for _ in range(batch_size):
            item = await self.fetch()
            if item:
                batch.append(item)
            else:
                break
        return batch

    async def get_audio_path(self, data_point: Dict[str, Any]) -> str:
        return data_point["path"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'AudioFileProvider':
        return cls(directory=config["directory"])