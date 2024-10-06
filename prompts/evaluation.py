from jinja2 import Template

DEFAULT_EVALUATION_TEMPLATE = Template("""
Context: {{ context }}
Criteria: {{ criteria }}
Evaluate the given context based on the specified criteria:
""".strip())

class EvaluationPrompt:
    def __init__(self, template=DEFAULT_EVALUATION_TEMPLATE):
        self.template = template

    def render(self, context, criteria):
        return self.template.render(context=context, criteria=criteria)

    def set_template(self, template_string):
        self.template = Template(template_string)