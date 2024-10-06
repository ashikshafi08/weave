from weave.core.framework import SyntheticDataFramework
from weave.generators.math_generator import MathGenerator
from weave.llm_interfaces.dummy_interface import DummyInterface
import yaml

# Load configuration
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize components
data_generator = MathGenerator()
llm_interface = DummyInterface()

# Create framework
framework = SyntheticDataFramework(data_generator, llm_interface)

# Generate dataset
dataset = framework.generate_dataset(config['framework']['num_samples'])

# Validate dataset
validation_results = framework.validate_dataset(dataset)

print(f"Generated {len(dataset)} samples")
print(f"First sample: {dataset[0]}")
print(f"Validation results: {validation_results}")