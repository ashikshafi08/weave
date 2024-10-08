from jinja2 import Template
from typing import Dict, Any

class PromptTemplate:
    def __init__(self, template_string: str):
        self.template = Template(template_string)

    def render(self, **kwargs: Any) -> str:
        return self.template.render(**kwargs)

class PromptTemplateManager:
    def __init__(self):
        self.templates: Dict[str, PromptTemplate] = {}

    def add_template(self, name: str, template_string: str):
        self.templates[name] = PromptTemplate(template_string)

    def get_template(self, name: str) -> PromptTemplate:
        if name not in self.templates:
            raise KeyError(f"No template found with name: {name}")
        return self.templates[name]

    def render_template(self, name: str, **kwargs: Any) -> str:
        template = self.get_template(name)
        return template.render(**kwargs)