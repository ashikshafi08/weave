import wikipedia
from .text_base import TextDataProvider
from . import register_data_provider

@register_data_provider("wikipedia")
class WikipediaProvider(TextDataProvider):
    def __init__(self, language: str = "en"):
        self.language = language
        wikipedia.set_lang(language)

    async def fetch(self, **kwargs) -> Optional[Dict[str, Any]]:
        title = kwargs.get("title")
        if not title:
            title = wikipedia.random(1)[0]
        try:
            page = wikipedia.page(title)
            return {
                "title": page.title,
                "content": page.content,
                "url": page.url,
            }
        except wikipedia.exceptions.DisambiguationError:
            return None

    async def fetch_batch(self, batch_size: int, **kwargs) -> List[Dict[str, Any]]:
        titles = wikipedia.random(pages=batch_size)
        return [await self.fetch(title=title) for title in titles]

    async def get_text_content(self, data_point: Dict[str, Any]) -> str:
        return data_point["content"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'WikipediaProvider':
        return cls(language=config.get("language", "en"))