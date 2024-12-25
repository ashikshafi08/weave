# Advanced Noising Techniques

This guide demonstrates the advanced noising capabilities of the Weave framework.

## Style Transfer Noiser

```python
from weave.noisers import StyleTransferNoiser
from weave.llms import OpenAILLM

# Initialize LLM
llm = OpenAILLM(model="gpt-4o-mini")

# Create style noiser
noiser = StyleTransferNoiser(
    model_connector=llm,
    style_config={
        "style": "technical",
        "formality": "high"
    }
)

# Transform text
casual_text = "Hey! This code is pretty cool, it makes fake data look super real!"
technical_text = noiser.augment(casual_text)
print(technical_text)
# Output: "The implementation demonstrates remarkable efficacy in generating 
#          synthetic data with high fidelity to real-world distributions."
```

## Language Noiser

```python
from weave.noisers import LanguageNoiser

# Create language noiser
noiser = LanguageNoiser(
    model_connector=llm,
    language_config={
        "source_language": "en",
        "target_language": "en-GB",
        "preserve_meaning": True
    }
)

# Transform text
us_text = "The color of the center dialog is gray"
uk_text = noiser.augment(us_text)
print(uk_text)
# Output: "The colour of the centre dialogue is grey"
```

## Domain Error Noiser

```python
from weave.noisers import DomainErrorNoiser

# Create domain error noiser
noiser = DomainErrorNoiser(
    model_connector=llm,
    domain_config={
        "domain": "programming",
        "error_types": ["syntax", "logic", "runtime"],
        "severity": "medium"
    }
)

# Add realistic programming errors
code = """
def calculate_average(numbers):
    return sum(numbers) / len(numbers)
"""

noisy_code = noiser.augment(code)
print(noisy_code)
# Output: """
# def calculate_average(numbers):
#     total = 0
#     for num in numbers:
#         total += num
#     return total / length  # Common error: using undefined 'length' instead of len(numbers)
# """
```

## Sentiment Noiser

```python
from weave.noisers import SentimentNoiser

# Create sentiment noiser
noiser = SentimentNoiser(
    model_connector=llm,
    sentiment_config={
        "target_sentiment": "positive",
        "intensity": 0.8,
        "preserve_key_facts": True
    }
)

# Transform sentiment
neutral_review = "The restaurant was okay. Food came on time."
positive_review = noiser.augment(neutral_review)
print(positive_review)
# Output: "The restaurant was fantastic! The service was impressively prompt, 
#          delivering our delicious food right on schedule."
```

## Context Noiser

```python
from weave.noisers import ContextNoiser

# Create context noiser
noiser = ContextNoiser(
    model_connector=llm,
    context_config={
        "context_type": "conversation",
        "window_size": 3,
        "coherence_weight": 0.8
    }
)

# Transform text while maintaining context
conversation = [
    "User: What's the weather like?",
    "Assistant: It's sunny and warm today.",
    "User: Should I go to the beach?"
]

response = noiser.augment_with_context(
    text="Yes, it's perfect beach weather!",
    context=conversation
)
print(response)
# Output: "Given the sunny and warm conditions I mentioned, 
#          it's definitely a perfect day for the beach!"
```

## Batch Processing

All noisers support batch processing for efficient transformation of multiple texts:

```python
texts = [
    "The product works well.",
    "Delivery was on time.",
    "Setup was easy."
]

# Batch transform
positive_reviews = sentiment_noiser.batch_augment(texts)
for review in positive_reviews:
    print(review)
# Output:
# "The product exceeds expectations with its outstanding performance!"
# "Impressed by the incredibly punctual delivery service!"
# "Setup was a breeze - incredibly user-friendly and intuitive!"
```
