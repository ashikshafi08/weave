import asyncio
import logging
from weave.core.framework import SyntheticDataFramework
from weave.core.config import Config
from weave.data_providers.wikipedia_provider import WikipediaProvider
from weave.task_creators.physics_qa_creator import PhysicsQATaskCreator
from weave.llm_interfaces.openai_provider import OpenAIProvider

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main():
    # Step 1: Load configuration
    config = Config('config/physics_qa_config.json')
    
    # Step 2: Create framework
    framework = SyntheticDataFramework(config)
    
    # Step 3: Customize the framework (optional)
    # Add a custom pipeline stage to preprocess the data
    framework.add_pipeline_stage(preprocess_physics_data)
    
    # Add custom prompt templates
    framework.add_prompt_template(
        "physics_question_generation",
        """
        Based on the following physics-related text, generate a graduate-level question:
        
        Text: {{data}}
        
        Context: {{context}}
        
        Generate a question that requires deep understanding and analysis of advanced physics concepts.
        The question should be suitable for graduate-level physics students.
        """
    )
    
    # Step 4: Generate dataset
    dataset = await framework.generate_dataset(config.get('num_samples'))
    
    # Step 5: Process and display results
    logger.info(f"Generated {len(dataset)} physics Q&A pairs")
    for i, sample in enumerate(dataset[:5], 1):  # Print first 5 samples
        logger.info(f"\nSample {i}:")
        logger.info(f"Question: {sample['question']}")
        logger.info(f"Answer: {sample['answer']}")
    
    # Step 6: Save the dataset
    save_dataset(dataset, 'physics_qa_dataset.json')

def preprocess_physics_data(data: str, context: dict) -> tuple:
    """Custom preprocessing function for physics data."""
    # Example: Remove any non-physics related content (simplified)
    physics_keywords = ['energy', 'force', 'mass', 'velocity', 'quantum', 'relativity']
    processed_data = ' '.join([sent for sent in data.split('.') if any(keyword in sent.lower() for keyword in physics_keywords)])
    context['preprocessed'] = True
    return processed_data, context

def save_dataset(dataset: list, filename: str):
    """Save the generated dataset to a JSON file."""
    import json
    with open(filename, 'w') as f:
        json.dump(dataset, f, indent=2)
    logger.info(f"Dataset saved to {filename}")

if __name__ == "__main__":
    asyncio.run(main())