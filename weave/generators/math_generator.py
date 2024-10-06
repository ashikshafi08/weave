from weave.core.data_generator import DataGenerator
from typing import Any, Dict, Tuple, List
import random

class MathGenerator(DataGenerator):
    def generate(self, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        operation = kwargs.get('operation', 'addition')
        if operation == 'addition':
            a, b = random.randint(1, 100), random.randint(1, 100)
            answer = a + b
            context = {"operation": "addition", "operands": [a, b]}
        else:
            raise ValueError(f"Unsupported operation: {operation}")
        
        return answer, context

    def get_supported_types(self) -> List[str]:
        return ["addition"]