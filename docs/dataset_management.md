# Dataset Management Guide

This guide demonstrates the dataset management capabilities of the Weave framework.

## Loading Datasets

```python
from weave.datasets import DatasetLoader

# Initialize loader
loader = DatasetLoader()

# Load from various sources
csv_data = loader.load("path/to/data.csv")
json_data = loader.load("path/to/data.json")
hf_data = loader.load("huggingface://organization/dataset")
kaggle_data = loader.load("kaggle://username/dataset")

# Load with preprocessing
processed_data = loader.load(
    "path/to/data.csv",
    preprocessing={
        "drop_duplicates": True,
        "fill_na": "mean",
        "normalize": ["numeric_columns"]
    }
)
```

## Streaming Large Datasets

```python
from weave.datasets import StreamingDataset

# Create streaming dataset
with StreamingDataset(chunk_size=1000) as stream:
    # Load large file
    stream.load("path/to/large_file.csv")
    
    # Process in chunks
    for chunk in stream.iter_chunks():
        processed_chunk = process_chunk(chunk)
        save_results(processed_chunk)
    
    # Get summary statistics
    stats = stream.get_stats()
    print(f"Total records: {stats['total_records']}")
```

## Merging Datasets

```python
from weave.datasets import DatasetMerger

# Initialize merger
merger = DatasetMerger()

# Simple append
combined = merger.merge(
    real_data,
    synthetic_data,
    strategy="append"
)

# Smart mixing with ratio
mixed = merger.merge(
    real_data,
    synthetic_data,
    strategy="mix",
    ratio=0.3  # 30% synthetic
)

# Advanced mixing with constraints
constrained = merger.merge(
    real_data,
    synthetic_data,
    strategy="smart_mix",
    constraints={
        "preserve_distribution": ["age", "income"],
        "balance_classes": "category",
        "max_synthetic_ratio": 0.4
    }
)
```

## Quality Validation

```python
from weave.datasets import DatasetValidator

# Create validator
validator = DatasetValidator()

# Add validation rules
validator.add_rules({
    "age": {
        "type": "numeric",
        "range": [0, 120],
        "required": True
    },
    "email": {
        "type": "string",
        "pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
        "required": True
    },
    "category": {
        "type": "categorical",
        "values": ["A", "B", "C"],
        "distribution": "original"
    }
})

# Validate dataset
results = validator.validate(synthetic_data, reference_data=real_data)
print(f"Validation score: {results['score']}")
print(f"Issues found: {results['issues']}")
```

## Format Conversion

```python
from weave.datasets import DatasetConverter

# Initialize converter
converter = DatasetConverter()

# Convert between formats
converter.to_csv(data, "output.csv")
converter.to_json(data, "output.json")
converter.to_parquet(data, "output.parquet")

# Convert to HuggingFace format
hf_dataset = converter.to_huggingface(
    data,
    config={
        "features": {
            "text": "string",
            "label": "classification"
        }
    }
)
```

## Dataset Analysis

```python
from weave.datasets import DatasetAnalyzer

# Initialize analyzer
analyzer = DatasetAnalyzer()

# Compare distributions
comparison = analyzer.compare_distributions(
    real_data,
    synthetic_data,
    metrics=["ks_test", "chi_square", "jensen_shannon"]
)

# Generate report
report = analyzer.generate_report(
    synthetic_data,
    reference_data=real_data,
    include=[
        "statistical_tests",
        "distribution_plots",
        "correlation_matrix",
        "privacy_metrics"
    ]
)

# Save report
report.save("analysis_report.html")
