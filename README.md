"""
# weave

![Weave Logo](weave.png)

Weave is a flexible framework for generating high-quality synthetic data using Language Models (LLMs). It provides a modular and extensible architecture that allows users to easily create, customize, and validate synthetic datasets for various applications.

**Note: This project is in its very early stages and is being actively developed in public. Expect frequent changes and improvements.**

GitHub Repository: [https://github.com/ashikshafi08/weave.git](https://github.com/ashikshafi08/weave.git)

## Installation

You can install weave directly from GitHub using pip:
```bash
pip install git+https://github.com/ashikshafi08/weave.git
```

For development, you can clone the repository and install it in editable mode:

```bash
git clone https://github.com/ashikshafi08/weave.git
cd weave
pip install -e .
```

## 🌟 Key Features

- 🔌 **Modular Architecture**: Easily extend and customize components
- 🤖 **Multiple LLM Support**: OpenAI and Hugging Face integration with customizable providers
- 📝 **Dynamic Prompts**: Customizable templates with variable substitution
- ✨ **Data Augmentation**: Flexible noising and transformation pipeline
- 🔄 **Async-First**: Built for high-performance async operations
- 🔧 **Type Safety**: Full type hints and runtime validation

## 🏗️ Architecture

The Weave framework consists of several core components:

### Core Components

1. **LLM Providers** (`llms/`)
   - `OpenAILLM`: Full support for GPT-4, GPT-3.5, and embeddings
   - `HuggingFaceLLM`: Integration with Hugging Face's model hub
   - Customizable API settings and error handling
   - Token usage tracking and cost estimation

2. **Generators** (`generators/`)
   - `MathGenerator`: Mathematical problem generation
   - `CodeGenerator`: Code problem and solution generation
   - `NLUGenerator`: Natural language understanding tasks
   - Customizable templates and difficulty levels

3. **Noisers** (`noisers/`)
   - `PersonaNoiser`: Style and persona-based text transformation
   - `RandomTyposNoiser`: Realistic typo and error introduction
   - Template-based noise generation
   - Configurable noise levels and types

4. **Core Utilities** (`core/`)
   - Base classes and interfaces
   - Error handling and validation
   - Common utilities and helpers

## 🚀 Quick Start

### Using OpenAI Provider

```python
from weave.llms import OpenAILLM
from weave.generators import MathGenerator
from weave.noisers import PersonaNoiser, RandomTyposNoiser

# Initialize LLM provider
llm = OpenAILLM(
    model="gpt-3.5-turbo",
    api_key="your-api-key"  # or use OPENAI_API_KEY env var
)

# Create noisers
persona_noiser = PersonaNoiser(
    model_connector=llm,
    persona_name="High School Student",
    persona_traits=["casual", "informal", "uses-emojis"]
)

typo_noiser = RandomTyposNoiser(
    model_connector=llm,
    error_frequency="moderate",
    error_severity="mild"
)

# Initialize generator with noisers
math_gen = MathGenerator(
    problem_type="word",
    difficulty="medium",
    model_connector=llm,
    noisers=[persona_noiser, typo_noiser]
)

# Generate problems
async def generate_problems():
    # Single problem
    problem, solution = await math_gen.generate()
    
    # Batch of problems
    problems = await math_gen.batch_generate(batch_size=5)
    return problems
```

### Using Hugging Face Provider

```python
from weave.llms import HuggingFaceLLM
from weave.generators import CodeGenerator

# Initialize Hugging Face provider
llm = HuggingFaceLLM(
    model_id="bigcode/starcoder",
    api_key="your-api-key"  # or use HF_API_TOKEN env var
)

# Initialize code generator
code_gen = CodeGenerator(
    problem_type="algorithm",
    language="python",
    difficulty="medium",
    model_connector=llm
)

# Generate coding problems
async def generate_code_problems():
    problem, solution = await code_gen.generate()
    return problem, solution
```

### Using NLU Generator

```python
from weave.generators import NLUGenerator

# Initialize NLU generator
nlu_gen = NLUGenerator(
    task_type="classification",
    difficulty="medium",
    model_connector=llm
)

# Generate NLU tasks
async def generate_nlu_tasks():
    # Generate a sentiment analysis task
    task, solution = await nlu_gen.generate()
    return task, solution
```

## 📝 Customizing Templates

Each generator and noiser supports customizable templates:

```python
# Custom math problem template
math_gen.word_problem_template = """
Generate a ${difficulty} word problem about ${topic}.
The problem should involve ${operations} and satisfy these constraints:
${constraints}

${format_instructions}
"""

# Custom persona noising template
persona_noiser.prompt_template = """
Rewrite the following text in the style of ${persona_name}.
Key traits: ${persona_traits}
Style guide: ${style_instructions}

Original text:
${original_text}

Rewritten text:
"""
```

## 🔧 Configuration

Generators and noisers accept various configuration options:

```python
# Math generator config
math_gen = MathGenerator(
    problem_type="algebra",  # arithmetic, algebra, word, calculus
    difficulty="medium",     # easy, medium, hard
    model_connector=llm,
    noisers=[persona_noiser],
    config={
        "max_attempts": 3,
        "timeout": 30.0,
        "validate_solutions": True
    }
)

# Code generator config
code_gen = CodeGenerator(
    problem_type="algorithm",    # algorithm, data_structure, system_design, bug_fixing
    language="python",          # python, javascript, java, cpp, go, rust, typescript
    difficulty="medium",
    model_connector=llm,
    config={
        "include_tests": True,
        "add_comments": True,
        "style_guide": "pep8"
    }
)

# NLU generator config
nlu_gen = NLUGenerator(
    task_type="classification",  # classification, qa, summarization, ner, paraphrase
    difficulty="medium",
    model_connector=llm,
    config={
        "include_metadata": True,
        "return_confidence": True,
        "multiple_labels": False
    }
)
```

## 🔍 Error Handling

The framework provides comprehensive error handling:

```python
from weave.core import GenerationError, ModelError

try:
    problem, solution = await math_gen.generate()
except GenerationError as e:
    print(f"Generation failed: {e}")
except ModelError as e:
    print(f"Model error: {e}")
```

## 📊 Monitoring and Metrics

Track token usage and costs with OpenAI:

```python
# Get token usage statistics
usage = llm.get_token_usage()
print(f"Total tokens: {usage['total_tokens']}")
print(f"Prompt tokens: {usage['prompt_tokens']}")
print(f"Completion tokens: {usage['completion_tokens']}")

# Estimate costs
cost = llm.estimate_cost()
print(f"Estimated cost: ${cost:.4f}")
```

## Contributing

As this project is in its early stages, contributions, suggestions, and feedback are highly welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This project is under active development. APIs may change, and features may be added or removed. Use in production environments is not recommended at this stage.
"""

