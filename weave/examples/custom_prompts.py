import asyncio
from weave.llm_interfaces.openai_provider import OpenAIProvider

async def main():
    # Initialize the provider
    provider = OpenAIProvider(model="gpt-3.5-turbo", api_key="YOUR_API_KEY")

    # Customize the prompt for generating questions
    provider.set_question_template(
        "Given the answer '{{ answer }}' and the context {{ context }}, create a challenging question that would lead to this answer. Make sure the question is suitable for a {{ context.difficulty }} level in {{ context.topic }}."
    )

    # Use the provider
    answer = 42
    context = {"topic": "mathematics", "difficulty": "medium"}
    question = await provider.generate_question(answer, context)
    print(f"Generated question: {question}")

    # Customize the validation prompt
    provider.set_validation_template(
        "Question: {{ question }}\nProposed Answer: {{ proposed_answer }}\nCorrect Answer: {{ correct_answer }}\nIs the proposed answer mathematically equivalent to the correct answer? Answer with 'Yes' or 'No'."
    )

    # Validate an answer
    is_correct = await provider.validate_answer(question, "6 * 7", 42)
    print(f"Is the answer correct? {'Yes' if is_correct else 'No'}")

if __name__ == "__main__":
    asyncio.run(main())