from jinja2 import Template

DEFAULT_QUESTION_TEMPLATE = Template("""
Given the answer '{{ answer }}' and the context {{ context }}, create a challenging question that would lead to this answer.
""".strip())

class QuestionGenerationPrompt:
    def __init__(self, template=DEFAULT_QUESTION_TEMPLATE):
        self.template = template

    def render(self, answer, context):
        return self.template.render(answer=answer, context=context)

    def set_template(self, template_string):
        self.template = Template(template_string)