# examples/generate_qa_dataset.py
import asyncio
from weave.core.framework import SyntheticDataFramework
from weave.core.config import Config


async def main():
    config = Config.from_cli()
    framework = SyntheticDataFramework(config)

    # Add custom stages to the pipeline
    framework.pipeline.add_stage(lambda data, context: (data.upper(), context))

    dataset = await framework.generate_dataset(num_samples=10)
    print(f"Generated {len(dataset)} samples:")
    for sample in dataset:
        print(sample)

    criteria = {"relevance": 0.8, "difficulty": "medium"}
    evaluations = await framework.evaluate_dataset(dataset, criteria)
    print("\nEvaluations:")
    for evaluation in evaluations:
        print(evaluation)


if __name__ == "__main__":
    asyncio.run(main())
