from typing import Any, Dict, Tuple, List
from weave.core.data_generator import DataGenerator
import random

class ProgrammingGenerator(DataGenerator):
    def __init__(self):
        self.languages = ["Python", "JavaScript", "Java", "C++"]
        self.topics = ["Arrays", "Strings", "Linked Lists", "Trees", "Sorting", "Searching"]
        self.difficulties = ["Easy", "Medium", "Hard"]

    def generate(self, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        language = kwargs.get('language', random.choice(self.languages))
        topic = kwargs.get('topic', random.choice(self.topics))
        difficulty = kwargs.get('difficulty', random.choice(self.difficulties))

        # This is a simplified example. In a real scenario, you'd have a database of programming questions
        # or a more sophisticated method of generating them.
        question = f"Write a {difficulty} {language} function to solve a problem related to {topic}."
        answer = "Sample solution code here"

        context = {
            "language": language,
            "topic": topic,
            "difficulty": difficulty
        }

        return answer, context

    def get_supported_types(self) -> List[str]:
        return ["programming_question"]