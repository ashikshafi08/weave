from jinja2 import Template

DEFAULT_VALIDATION_TEMPLATE = Template("""
Question: {{ question }}
Proposed Answer: {{ proposed_answer }}
Correct Answer: {{ correct_answer }}
Is the proposed answer correct? Answer with 'Yes' or 'No'.
""".strip())

class AnswerValidationPrompt:
    def __init__(self, template=DEFAULT_VALIDATION_TEMPLATE):
        self.template = template

    def render(self, question, proposed_answer, correct_answer):
        return self.template.render(question=question, proposed_answer=proposed_answer, correct_answer=correct_answer)

    def set_template(self, template_string):
        self.template = Template(template_string)