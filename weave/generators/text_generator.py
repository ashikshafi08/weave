# weave/data_generators/text_generator.py
import random
from typing import Any, Dict, Tuple, List
from weave.core.base import DataGenerator
from weave.core.registry import data_generator_registry

class TextGenerator(DataGenerator):
    def __init__(self):
        self.texts = [
            "The quick brown fox jumps over the lazy dog.",
            "To be or not to be, that is the question.",
            "I think, therefore I am.",
            "Life is like a box of chocolates.",
            "May the Force be with you."
        ]

    async def generate(self, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        text = random.choice(self.texts)
        return text, {"source": "predefined_texts"}

    def get_supported_types(self) -> List[str]:
        return ["text"]

    async def load_dataset(self, dataset_path: str) -> None:
        with open(dataset_path, 'r') as f:
            self.texts = f.read().splitlines()

    async def sample(self, n: int) -> List[Tuple[Any, Dict[str, Any]]]:
        return [await self.generate() for _ in range(n)]

    async def augment(self, data: Any, context: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        # Simple augmentation: add random punctuation
        punctuation = "!?.,;"
        augmented_data = data + random.choice(punctuation)
        return augmented_data, context

    def initialize(self, config: Dict[str, Any]) -> None:
        if 'dataset_path' in config:
            asyncio.run(self.load_dataset(config['dataset_path']))

data_generator_registry.register("text", TextGenerator)