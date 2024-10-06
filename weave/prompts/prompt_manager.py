from typing import Dict
from jinja2 import Template

class PromptManager:
    def __init__(self):
        self.templates: Dict[str, Template] = {}

    def set_template(self, prompt_type: str, template_string: str):
        self.templates[prompt_type] = Template(template_string)

    def render(self, prompt_type: str, **kwargs) -> str:
        if prompt_type not in self.templates:
            raise ValueError(f"No template found for prompt type: {prompt_type}")
        return self.templates[prompt_type].render(**kwargs)