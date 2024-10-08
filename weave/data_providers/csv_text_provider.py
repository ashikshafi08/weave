import pandas as pd
from .text_base import TextDataProvider
from . import register_data_provider

@register_data_provider("csv_text")
class CSVTextProvider(TextDataProvider):
    def __init__(self, file_path: str, text_column: str):
        self.df = pd.read_csv(file_path)
        self.text_column = text_column
        self.current_index = 0

    async def fetch(self, **kwargs) -> Optional[Dict[str, Any]]:
        if self.current_index < len(self.df):
            data = self.df.iloc[self.current_index].to_dict()
            self.current_index += 1
            return data
        return None

    async def fetch_batch(self, batch_size: int, **kwargs) -> List[Dict[str, Any]]:
        batch = self.df.iloc[self.current_index:self.current_index + batch_size].to_dict('records')
        self.current_index += len(batch)
        return batch

    async def get_text_content(self, data_point: Dict[str, Any]) -> str:
        return data_point[self.text_column]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'CSVTextProvider':
        return cls(file_path=config["file_path"], text_column=config["text_column"])