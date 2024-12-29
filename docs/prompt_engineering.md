# Prompt Engineering Guide

This guide demonstrates the prompt engineering capabilities of the Weave framework.

## Working with Templates

```python
from weave.prompts import PromptTemplate, PromptLibrary
from weave.llms import OpenAILLM

# Initialize LLM
llm = OpenAILLM(model="gpt-4o-mini")

# Create template
template = PromptTemplate(
    """Classify the sentiment of this ${language} text:
    
Text: ${text}
Sentiment (positive/negative/neutral):""",
    metadata={
        "task": "sentiment_analysis",
        "version": "1.0"
    }
)

# Render template
prompt = template.render({
    "language": "English",
    "text": "This product is amazing!"
})

# Use with LLM
response = llm.generate(prompt)
print(response)  # Output: "positive"
```

## Using the Prompt Library

```python
# Initialize library
library = PromptLibrary()

# Add templates
library.add_template(
    "qa",
    PromptTemplate(
        """Answer the question based on the context:
        
Context: ${context}
Question: ${question}
Answer:""",
        metadata={"task": "question_answering"}
    )
)

# Use template
template = library.get_template("qa")
prompt = template.render({
    "context": "The cat sat on the mat.",
    "question": "Where did the cat sit?"
})

# List available templates
print("Available templates:")
for category in library.get_categories():
    templates = library.get_templates_in_category(category)
    print(f"{category}: {templates}")
```

## Prompt Optimization

```python
from weave.prompts import PromptOptimizer

# Create optimizer
optimizer = PromptOptimizer(
    model_connector=llm,
    optimization_config={
        "max_tokens": 150,
        "temperature": 0.7
    }
)

# Define test cases
test_cases = [
    {
        "input": {
            "text": "This movie was fantastic!",
            "categories": "positive, negative, neutral"
        },
        "expected": "positive"
    },
    {
        "input": {
            "text": "I didn't enjoy this book at all.",
            "categories": "positive, negative, neutral"
        },
        "expected": "negative"
    }
]

# Define evaluation function
def evaluate_classification(response: str, expected: str) -> float:
    return 1.0 if response.strip().lower() == expected.lower() else 0.0

# Optimize template
optimized = optimizer.optimize(
    template=template,
    test_cases=test_cases,
    evaluation_fn=evaluate_classification,
    num_iterations=3
)

print("Optimization history:")
for record in optimizer.get_optimization_history():
    print(f"Iteration {record['iteration']}: {record['score']}")
```

## Multi-Turn Prompts

```python
# Create conversation template
conversation_template = PromptTemplate(
    """Previous conversation:
${history}

User: ${user_input}
Assistant: Let me help you with that."""
)

# Build conversation history
history = [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi! How can I help you today?"},
    {"role": "user", "content": "I need help with my code."}
]

# Format history
formatted_history = "\n".join(
    f"{turn['role'].title()}: {turn['content']}"
    for turn in history
)

# Render template
prompt = conversation_template.render({
    "history": formatted_history,
    "user_input": "Can you explain how to use functions?"
})
```

## Template Management

```python
import tempfile
from pathlib import Path

# Save template to file
with tempfile.TemporaryDirectory() as temp_dir:
    template_path = Path(temp_dir) / "template.json"
    template.to_file(template_path)
    
    # Load template from file
    loaded_template = PromptTemplate.from_file(template_path)
    
    # Save entire library
    library_dir = Path(temp_dir) / "templates"
    library.save_to_directory(library_dir)
    
    # Load library from directory
    loaded_library = PromptLibrary.from_directory(library_dir)
```

## Advanced Features

### Chain of Thought Prompting
```python
cot_template = PromptTemplate(
    """Solve this math problem step by step:
Problem: ${problem}

Let's solve this step by step:
1) First, let's ${first_step}
2) Then, we can ${second_step}
3) Finally, we ${final_step}

Therefore, the answer is:""")

prompt = cot_template.render({
    "problem": "If a train travels 120 km in 2 hours, what is its speed?",
    "first_step": "identify the key information (distance = 120 km, time = 2 hours)",
    "second_step": "recall the formula for speed (speed = distance ÷ time)",
    "final_step": "plug in our values and calculate (120 ÷ 2)"
})
```

### Few-Shot Learning
```python
few_shot_template = PromptTemplate(
    """Classify these sentences as formal or informal.

Examples:
Input: "Hey, what's up?"
Classification: informal

Input: "Dear Sir/Madam,"
Classification: formal

Input: "Could you please assist me?"
Classification: formal

Now classify this:
Input: "${text}"
Classification:""")

prompt = few_shot_template.render({
    "text": "Yo, check this out!"
})
```
