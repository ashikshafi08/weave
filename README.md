# weave

![Weave Logo](weave.png)

> 🚀 Transform your data with AI-powered synthetic generation and augmentation

Weave is a powerful Python framework that helps you create high-quality synthetic datasets using state-of-the-art Language Models. Whether you're training ML models, testing applications, or augmenting existing datasets, Weave makes it easy to generate diverse, realistic data at scale.

## ✨ Why Weave?

- 🎯 **Production-Ready Data Generation**: Create synthetic datasets that mirror real-world complexity and edge cases
- 🔄 **Smart Data Augmentation**: Enhance your training data with intelligent noise and variations
- 🎨 **Style Transfer & Persona Simulation**: Generate content in different writing styles and personas
- 🌍 **Multi-Language Support**: Work with content across different languages and domains
- 🔍 **Context-Aware Transformations**: Maintain coherence and relevance in your synthetic data
- 📊 **Advanced Dataset Management**: Seamlessly merge and manage synthetic and real datasets

## 🚀 Quick Start

```bash
pip install git+https://github.com/ashikshafi08/weave.git
```

```python
from weave.noisers import StyleTransferNoiser
from weave.llms import OpenAILLM

# Initialize with your favorite LLM
llm = OpenAILLM(model="gpt-4o-mini")

# Create a technical writer persona
noiser = StyleTransferNoiser(
    model_connector=llm,
    style_config={"style": "technical_documentation"}
)

# Transform casual text into technical documentation
casual_text = "This code helps you make fake data that looks real"
technical_doc = noiser.augment(casual_text)
print(technical_doc)
# Output: "This framework facilitates the generation of synthetic data 
#          that accurately simulates real-world characteristics..."
```

## 🎯 Use Cases

### Data Augmentation
```python
from weave.datasets import DatasetLoader, DatasetMerger

# Load your existing dataset
loader = DatasetLoader()
real_data = loader.load("path/to/data.csv")

# Generate complementary synthetic data
synthetic_data = generate_synthetic_samples(real_data)

# Intelligently merge real and synthetic data
merger = DatasetMerger()
enhanced_dataset = merger.merge(
    real_data, 
    synthetic_data,
    strategy="mix",
    ratio=0.3  # 30% synthetic data
)
```

### Multi-Style Content Generation
```python
from weave.noisers import LanguageNoiser, SentimentNoiser

# Create content variations
lang_noiser = LanguageNoiser(
    model_connector=llm,
    language_config={
        "language": "en",
        "locale": "UK"
    }
)

sentiment_noiser = SentimentNoiser(
    model_connector=llm,
    sentiment_config={
        "target_sentiment": "positive",
        "intensity": 0.8
    }
)

# Transform content
uk_text = lang_noiser.augment("Color the background blue")
# Output: "Colour the background blue"

positive_review = sentiment_noiser.augment("The service was okay")
# Output: "The service exceeded my expectations!"
```

## 📚 Documentation

Check out our example notebooks to see Weave in action:
- [Advanced Noising Techniques](examples/advanced_noising.ipynb)
- [Dataset Management](examples/dataset_management.ipynb)
- [Prompt Engineering](examples/prompt_engineering.ipynb)

## 🛠️ Features

### Advanced Noisers
- **Style Transfer**: Transform content between different writing styles
- **Language Adaptation**: Handle language-specific nuances and variations
- **Domain-Specific Errors**: Simulate realistic mistakes and edge cases
- **Sentiment Transformation**: Adjust content tone and emotional impact
- **Context-Aware Noising**: Maintain coherence across transformations

### Dataset Tools
- **Smart Merging**: Intelligently combine synthetic and real data
- **Quality Validation**: Ensure synthetic data meets quality standards
- **Format Support**: Work with CSV, JSON, JSONL, and streaming data
- **HuggingFace Integration**: Direct access to public datasets

## 🤝 Contributing

We welcome contributions! Check out our [contribution guidelines](CONTRIBUTING.md) to get started.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
Built with ❤️ by the Weave team