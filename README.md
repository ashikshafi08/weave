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

## Core Architecture

### 1. Base Classes
- `BaseGenerator`: Foundation for all data generators
- `BaseNoiser`: Abstract class for data transformation
- `BaseOrchestrator`: Coordinates generation pipeline
- `BaseValidator`: Validates generated data

### 2. Dataset Management
The `datasets` module provides comprehensive data handling:

```python
from weave.datasets import DatasetLoader, DatasetMerger

# Load data from various sources
loader = DatasetLoader()
data = loader.load("kaggle://username/dataset/file.csv")

# Merge synthetic and real data
merger = DatasetMerger()
combined = merger.merge(real_data, synthetic_data, strategy="mix", ratio=0.3)
```

### 3. Advanced Noisers
Specialized noisers for various transformations:

```python
from weave.noisers import (
    StyleTransferNoiser,
    LanguageNoiser,
    DomainErrorNoiser,
    SentimentNoiser,
    ContextNoiser
)

# Initialize LLM
llm = OpenAILLM(
    model="gpt-4o-mini",  # High-performance model
    api_key="your-api-key"
)

# Style transfer
style_noiser = StyleTransferNoiser(
    model_connector=llm,
    style_config={"style": "technical"}
)

# Language-specific noise
lang_noiser = LanguageNoiser(
    model_connector=llm,
    language_config={
        "language": "en",
        "error_types": ["grammar", "spelling"]
    }
)

# Domain-specific errors
domain_noiser = DomainErrorNoiser(
    model_connector=llm,
    domain_config={
        "domain": "programming",
        "error_categories": ["syntax", "logic"]
    }
)

# Sentiment transformation
sentiment_noiser = SentimentNoiser(
    model_connector=llm,
    sentiment_config={
        "target_sentiment": "positive",
        "intensity": 0.8
    }
)

# Context-aware transformation
context_noiser = ContextNoiser(
    model_connector=llm,
    context_config={
        "context_type": "conversation",
        "window_sizes": {"conversation": 3}
    }
)
```

### 4. Prompt Engineering
The `prompts` module manages and optimizes prompts:

```python
from weave.prompts import PromptTemplate, PromptLibrary, PromptOptimizer

# Use template from library
library = PromptLibrary()
template = library.get_template("classification")

# Optimize prompts
optimizer = PromptOptimizer(
    model_connector=llm,
    optimization_config={
        "max_tokens": 150,
        "temperature": 0.7
    }
)

optimized = optimizer.optimize(
    template,
    test_cases=test_data,
    evaluation_fn=evaluate_classification
)
```

## Extensibility

### Creating Custom Components
Extend base classes to create custom components:

```python
from weave.core import BaseNoiser

class CustomNoiser(BaseNoiser):
    def augment(self, query: str) -> str:
        # Custom implementation
        pass

    def batch_augment(self, queries: List[str]) -> List[str]:
        # Custom implementation
        pass
```

### Plugin System
Register custom components:

```python
from weave.core import plugin_registry

@plugin_registry.register("custom_noiser")
class CustomNoiser(BaseNoiser):
    pass
```

## Type Safety
All components use Python type hints and support static type checking:

```python
from typing import List, Dict, Any

def process_data(data: List[Dict[str, Any]]) -> List[str]:
    pass
```

## Examples
Check out our example notebooks in the `examples/` directory:
- `advanced_noising.ipynb`: Demonstrates all noiser capabilities
- `dataset_management.ipynb`: Shows dataset handling features
- `prompt_engineering.ipynb`: Covers prompt management and optimization