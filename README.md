

# weave

![Weave Logo](weave_logo_1.webp)

Weave is a flexible framework for generating and validating synthetic data across various domains. The system leverages Language Models (LLMs) to create high-quality, domain-specific datasets that can be used for training AI models, testing, and research purposes.

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

## Key Features

- Modular architecture for easy extensibility
- Support for various data generators and LLM interfaces (OpenAI, Hugging Face, vLLM)
- Customizable prompt templates for different tasks
- Data validation and quality checking
- Asynchronous operations for improved performance
- Comprehensive logging for debugging and monitoring

## Usage

Here's a basic example of how to use weave:

```python
import asyncio
import logging
from weave import SyntheticDataFramework, ProgrammingGenerator, OpenAIProvider

async def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Initialize components
    data_generator = ProgrammingGenerator()
    llm_provider = OpenAIProvider(model="gpt-4o-mini", api_key="YOUR_API_KEY")
    
    # Create framework
    framework = SyntheticDataFramework(data_generator, llm_provider)
    
    # Set custom prompt templates
    framework.set_prompt_template("question_generation", "Generate a {difficulty} {language} programming question about {topic}. The answer should be: {answer}")
    framework.set_prompt_template("answer_validation", "For the {language} question: {question}\nIs this a valid answer: {proposed_answer}? Answer with Yes or No.")
    
    # Generate dataset
    dataset = await framework.generate_dataset(10)
    
    # Validate dataset
    validations = await framework.validate_dataset(dataset)
    
    # Evaluate dataset
    criteria = {"aspect": "code_quality", "scale": "1-10"}
    evaluations = await framework.evaluate_dataset(dataset, criteria)
    
    print(f"Generated {len(dataset)} samples")
    print(f"First sample: {dataset[0]}")
    print(f"First validation: {validations[0]}")
    print(f"First evaluation: {evaluations[0]}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Project Structure

- `weave/core/`: Contains the core framework classes
- `weave/generators/`: Data generator implementations
- `weave/llm_interfaces/`: LLM interface implementations (OpenAI, Hugging Face, vLLM)
- `weave/prompts/`: Prompt management and templates
- `weave/config/`: Configuration files
- `weave/examples/`: Usage examples

## Customization

### Creating a Custom Data Generator

1. Create a new file in the `weave/generators/` directory.
2. Implement a class that inherits from `DataGenerator`.
3. Override the `generate()` and `get_supported_types()` methods.

### Creating a Custom LLM Interface

1. Create a new file in the `weave/llm_interfaces/` directory.
2. Implement a class that inherits from `BaseLLMProvider`.
3. Override the required methods such as `generate_question()`, `validate_answer()`, `evaluate()`, etc.

### Customizing Prompts

Use the `set_prompt_template()` method of the `SyntheticDataFramework` or LLM provider to customize prompts for different tasks.

## Roadmap

To see the rough plans for future development and features, check out our [roadmap](roadmap.md). This is not set in stone and is subject to change as we receive feedback and decide what features are most important.

## Configuration

The `config/config.yaml` file allows you to set up your data generator, LLM provider, and other framework parameters. Here's an example:

```yaml
data_generator:
  type: "ProgrammingGenerator"
  params:
    languages: ["python", "javascript", "java"]
    difficulties: ["easy", "medium", "hard"]

llm_provider:
  type: "OpenAIProvider"
  params:
    model: "gpt-4o-mini"
    api_key: "YOUR_API_KEY"

framework:
  num_samples: 100
  logging_level: "INFO"
```

## Contributing

As this project is in its early stages, contributions, suggestions, and feedback are highly welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This project is under active development. APIs may change, and features may be added or removed. It's a learning project and is not intended for production use as of now.
```

