# weave

Weave is a flexible framework for generating and validating synthetic data across various domains. The system leverages Language Models (LLMs) to create high-quality, domain-specific datasets that can be used for training AI models, testing, and research purposes.

**Note: This project is in its very early stages and is being actively developed in public. Expect frequent changes and improvements.**

GitHub Repository: [https://github.com/ashikshafi08/weave.git](https://github.com/ashikshafi08/weave.git)

## Key Features

- Modular architecture for easy extensibility
- Support for various data generators and LLM interfaces
- Data validation and quality checking
- Configuration-based setup
- Comprehensive documentation and examples

## Getting Started

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ashikshafi08/weave.git
   cd weave
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Example

1. Make sure you're in the project root directory.

2. Run the basic usage example:
   ```bash
   python -m weave.examples.basic_usage
   ```

   This will generate a small dataset of math problems using the MathGenerator and DummyInterface.

### Project Structure

- `weave/core/`: Contains the core framework classes
- `weave/generators/`: Data generator implementations
- `weave/llm_interfaces/`: LLM interface implementations
- `weave/config/`: Configuration files
- `weave/examples/`: Usage examples

### Customization

To create your own data generator:

1. Create a new file in the `weave/generators/` directory.
2. Implement a class that inherits from `DataGenerator`.
3. Override the `generate()` and `get_supported_types()` methods.

To create your own LLM interface:

1. Create a new file in the `weave/llm_interfaces/` directory.
2. Implement a class that inherits from `LLMInterface`.
3. Override the `generate_question()`, `validate_answer()`, and `get_model_info()` methods.

Update the `config/config.yaml` file to use your custom components.

## Contributing

As this project is in its early stages, contributions, suggestions, and feedback are highly welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This project is under active development. APIs may change, and features may be added or removed. Its a learning project and is not intended for production use as of now.