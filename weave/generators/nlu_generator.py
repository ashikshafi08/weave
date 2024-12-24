"""Natural Language Understanding task generator for the Weave framework."""

import random
from typing import Any, Dict, List, Optional, Tuple, Union
import json
from string import Template

from ..core import (
    BaseGenerator,
    BaseNoiser,
    ModelError,
    GenerationError
)
from ..llms import OpenAILLM, HuggingFaceLLM

class NLUGenerator(BaseGenerator):
    """Generator for natural language understanding tasks.
    
    This generator can create various types of NLU tasks:
    - Classification (sentiment, topic, intent)
    - Question answering
    - Text summarization
    - Entity recognition
    - Paraphrasing
    - Text generation
    - Machine translation
    
    It uses customizable prompt templates with variables:
    - ${difficulty}: Task difficulty level
    - ${task_type}: Type of NLU task
    - ${category}: Specific category or domain
    - ${constraints}: Task constraints and requirements
    - ${format_instructions}: Output format instructions
    """
    
    TASK_TYPES = [
        "classification",
        "question_answering",
        "summarization",
        "entity_recognition",
        "paraphrasing",
        "text_generation",
        "translation"
    ]
    
    # Default templates for each task type
    DEFAULT_TEMPLATES = {
        "classification": """Generate a ${difficulty} ${category} classification task.

Task requirements:
- Labels: ${constraints}
- Complexity: ${constraints}
- Domain: ${category}
- Edge cases: ${constraints}

${format_instructions}""",

        "question_answering": """Generate a ${difficulty} ${category} question answering task.

Task requirements:
- Question type: ${constraints}
- Answer scope: ${constraints}
- Context length: ${constraints}
- Reasoning depth: ${constraints}

${format_instructions}""",

        "summarization": """Generate a ${difficulty} ${category} summarization task.

Task requirements:
- Style: ${constraints}
- Length: ${constraints}
- Key points: ${constraints}
- Audience: ${constraints}

${format_instructions}""",

        "entity_recognition": """Generate a ${difficulty} named entity recognition task for ${category} entities.

Task requirements:
- Entity types: ${constraints}
- Context: ${constraints}
- Ambiguity: ${constraints}
- Domain specificity: ${constraints}

${format_instructions}""",

        "paraphrasing": """Generate a ${difficulty} ${category} paraphrasing task.

Task requirements:
- Style: ${constraints}
- Constraints: ${constraints}
- Semantic preservation: ${constraints}
- Creativity level: ${constraints}

${format_instructions}""",

        "text_generation": """Generate a ${difficulty} ${category} text generation task.

Task requirements:
- Style: ${constraints}
- Length: ${constraints}
- Creativity: ${constraints}
- Coherence: ${constraints}

${format_instructions}""",

        "translation": """Generate a ${difficulty} ${category} translation task.

Task requirements:
- Language pair: ${constraints}
- Domain: ${constraints}
- Cultural context: ${constraints}
- Idiomatic usage: ${constraints}

${format_instructions}"""
    }
    
    # Default format instructions for each task type
    DEFAULT_FORMAT_INSTRUCTIONS = {
        "classification": """Format the response as a JSON object with these fields:
- task: task description and input text
- label: correct classification label
- explanation: reasoning for the classification
- confidence: confidence score for the label
- alternatives: other possible labels with scores
- edge_cases: potential edge cases to consider""",

        "question_answering": """Format the response as a JSON object with these fields:
- context: background text
- question: specific question about the context
- answer: correct answer
- explanation: reasoning for the answer
- evidence: relevant parts of context
- alternative_answers: other valid interpretations
- follow_up: suggested follow-up questions""",

        "summarization": """Format the response as a JSON object with these fields:
- text: source text to summarize
- summary: reference summary
- key_points: main points covered
- metrics: evaluation metrics (ROUGE, etc.)
- alternative_summaries: different styles/lengths
- important_quotes: key quotes from text""",

        "entity_recognition": """Format the response as a JSON object with these fields:
- text: input text with entities
- entities: list of entities with positions
- labels: entity type labels
- context: relevant context for each entity
- relationships: entity relationships
- ambiguities: potential ambiguous cases""",

        "paraphrasing": """Format the response as a JSON object with these fields:
- source: original text
- paraphrase: reference paraphrase
- constraints: paraphrasing requirements
- similarity: semantic similarity score
- alternatives: alternative paraphrases
- style_variations: different style options""",

        "text_generation": """Format the response as a JSON object with these fields:
- prompt: generation prompt
- text: generated text
- constraints: generation constraints
- variations: alternative generations
- style_guide: style specifications
- evaluation: quality metrics""",

        "translation": """Format the response as a JSON object with these fields:
- source: original text
- translation: reference translation
- alternatives: alternative translations
- notes: translation notes/challenges
- cultural_context: relevant cultural notes
- idioms: idiomatic expressions used"""
    }
    
    def __init__(
        self,
        task_type: str = "classification",
        difficulty: str = "medium",
        model_connector: Optional[Union[OpenAILLM, HuggingFaceLLM]] = None,
        noisers: Optional[List[BaseNoiser]] = None,
        templates: Optional[Dict[str, str]] = None,
        format_instructions: Optional[Dict[str, str]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the NLU task generator.
        
        Args:
            task_type: Type of NLU task to generate.
            difficulty: Difficulty level ("easy", "medium", "hard").
            model_connector: LLM provider for task generation.
            noisers: Optional list of noisers for data augmentation.
            templates: Custom templates for each task type.
            format_instructions: Custom format instructions.
            config: Additional configuration options.
        """
        super().__init__(config)
        
        if task_type not in self.TASK_TYPES:
            raise ValueError(f"Invalid task type: {task_type}")
            
        self.task_type = task_type
        self.difficulty = difficulty
        self.model = model_connector or OpenAILLM()  # Default to OpenAI
        self.noisers = noisers or []
        
        # Set up templates and format instructions
        self.templates = {
            ttype: Template(template)
            for ttype, template in (templates or self.DEFAULT_TEMPLATES).items()
        }
        self.format_instructions = format_instructions or self.DEFAULT_FORMAT_INSTRUCTIONS
        
    async def generate(self) -> Tuple[str, Dict[str, Any]]:
        """Generate a single NLU task and its solution.
        
        Returns:
            Tuple containing:
                - Task description and input text (str)
                - Solution (dict with answer and metadata)
                
        Raises:
            GenerationError: If generation fails.
        """
        try:
            # Generate base task and solution
            if self.task_type == "classification":
                task, solution = await self._generate_classification()
            elif self.task_type == "question_answering":
                task, solution = await self._generate_question_answering()
            elif self.task_type == "summarization":
                task, solution = await self._generate_summarization()
            elif self.task_type == "entity_recognition":
                task, solution = await self._generate_entity_recognition()
            elif self.task_type == "paraphrasing":
                task, solution = await self._generate_paraphrasing()
            elif self.task_type == "text_generation":
                task, solution = await self._generate_text_generation()
            elif self.task_type == "translation":
                task, solution = await self._generate_translation()
            else:
                raise GenerationError(f"Unsupported task type: {self.task_type}")
                
            # Apply noisers to the task text if any are configured
            if self.noisers:
                # Keep track of noise transformations
                noise_metadata = []
                
                # Apply each noiser in sequence
                noised_task = task
                for noiser in self.noisers:
                    try:
                        noised_task = await noiser.augment(noised_task)
                        noise_metadata.append(noiser.get_augmentation_metadata())
                    except Exception as e:
                        self.logger.warning(f"Noiser {noiser} failed: {str(e)}")
                        
                # Add noise metadata to solution
                solution["noise_metadata"] = noise_metadata
                return noised_task, solution
                
            return task, solution
            
        except Exception as e:
            raise GenerationError(f"Failed to generate NLU task: {str(e)}")
            
    async def batch_generate(
        self,
        batch_size: int
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Generate multiple NLU tasks.
        
        Args:
            batch_size: Number of tasks to generate.
            
        Returns:
            List of (task, solution) tuples.
            
        Raises:
            GenerationError: If generation fails.
        """
        tasks = []
        for _ in range(batch_size):
            task = await self.generate()
            tasks.append(task)
        return tasks
        
    async def _generate_classification(self) -> Tuple[str, Dict[str, Any]]:
        """Generate classification tasks."""
        categories = {
            "easy": [
                "sentiment", "spam", "language",
                "toxicity", "news category", "product reviews"
            ],
            "medium": [
                "topic", "intent", "emotion",
                "stance", "fact-checking", "style"
            ],
            "hard": [
                "stance", "sarcasm", "bias",
                "hate speech", "misinformation", "subjectivity"
            ]
        }
        
        constraints = {
            "easy": "Binary or simple multi-class labels",
            "medium": "Multi-class with overlapping categories",
            "hard": "Multi-label with hierarchical structure"
        }
        
        template_vars = {
            "difficulty": self.difficulty,
            "category": random.choice(categories[self.difficulty]),
            "constraints": constraints[self.difficulty],
            "format_instructions": self.format_instructions[self.task_type]
        }
        
        prompt = self.templates[self.task_type].substitute(template_vars)
        
        try:
            response = await self.model.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=1000
            )
            
            result = json.loads(response)
            
            required_fields = {"task", "label", "explanation"}
            if not all(field in result for field in required_fields):
                raise GenerationError("Invalid response format from LLM")
                
            return result["task"], result
            
        except json.JSONDecodeError:
            raise GenerationError("Failed to parse LLM response as JSON")
        except Exception as e:
            raise GenerationError(f"Classification task generation failed: {str(e)}")
            
    async def _generate_question_answering(self) -> Tuple[str, Dict[str, Any]]:
        """Generate question answering tasks."""
        categories = {
            "easy": [
                "factual", "definition", "yes/no",
                "who/what/where", "simple reasoning"
            ],
            "medium": [
                "how/why", "comparison", "analysis",
                "multi-hop", "temporal reasoning"
            ],
            "hard": [
                "complex reasoning", "counterfactual",
                "causal", "ethical", "abstract"
            ]
        }
        
        constraints = {
            "easy": "Single-hop questions with direct answers",
            "medium": "Multi-hop questions requiring analysis",
            "hard": "Complex questions with multiple valid perspectives"
        }
        
        template_vars = {
            "difficulty": self.difficulty,
            "category": random.choice(categories[self.difficulty]),
            "constraints": constraints[self.difficulty],
            "format_instructions": self.format_instructions[self.task_type]
        }
        
        prompt = self.templates[self.task_type].substitute(template_vars)
        
        try:
            response = await self.model.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=1500
            )
            
            result = json.loads(response)
            
            required_fields = {"context", "question", "answer", "explanation"}
            if not all(field in result for field in required_fields):
                raise GenerationError("Invalid response format from LLM")
                
            return result["question"], result
            
        except json.JSONDecodeError:
            raise GenerationError("Failed to parse LLM response as JSON")
        except Exception as e:
            raise GenerationError(f"Question answering task generation failed: {str(e)}")
            
    async def _generate_summarization(self) -> Tuple[str, Dict[str, Any]]:
        """Generate summarization tasks."""
        categories = {
            "easy": [
                "news articles", "product descriptions",
                "short stories", "simple instructions"
            ],
            "medium": [
                "academic papers", "technical documents",
                "legal texts", "business reports"
            ],
            "hard": [
                "research papers", "medical literature",
                "philosophical texts", "multi-document"
            ]
        }
        
        constraints = {
            "easy": "Short texts with clear main points",
            "medium": "Longer texts with multiple themes",
            "hard": "Complex texts with technical content"
        }
        
        template_vars = {
            "difficulty": self.difficulty,
            "category": random.choice(categories[self.difficulty]),
            "constraints": constraints[self.difficulty],
            "format_instructions": self.format_instructions[self.task_type]
        }
        
        prompt = self.templates[self.task_type].substitute(template_vars)
        
        try:
            response = await self.model.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=2000
            )
            
            result = json.loads(response)
            
            required_fields = {"text", "summary", "key_points"}
            if not all(field in result for field in required_fields):
                raise GenerationError("Invalid response format from LLM")
                
            return result["text"], result
            
        except json.JSONDecodeError:
            raise GenerationError("Failed to parse LLM response as JSON")
        except Exception as e:
            raise GenerationError(f"Summarization task generation failed: {str(e)}")
            
    async def _generate_entity_recognition(self) -> Tuple[str, Dict[str, Any]]:
        """Generate entity recognition tasks."""
        categories = {
            "easy": [
                "person names", "locations", "organizations",
                "dates", "numbers", "basic entities"
            ],
            "medium": [
                "professional titles", "products", "events",
                "technical terms", "legal entities"
            ],
            "hard": [
                "domain-specific", "nested entities",
                "ambiguous entities", "coreference"
            ]
        }
        
        constraints = {
            "easy": "Clear entity boundaries and types",
            "medium": "Mixed entity types and contexts",
            "hard": "Complex nested and ambiguous entities"
        }
        
        template_vars = {
            "difficulty": self.difficulty,
            "category": random.choice(categories[self.difficulty]),
            "constraints": constraints[self.difficulty],
            "format_instructions": self.format_instructions[self.task_type]
        }
        
        prompt = self.templates[self.task_type].substitute(template_vars)
        
        try:
            response = await self.model.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=1500
            )
            
            result = json.loads(response)
            
            required_fields = {"text", "entities", "labels"}
            if not all(field in result for field in required_fields):
                raise GenerationError("Invalid response format from LLM")
                
            return result["text"], result
            
        except json.JSONDecodeError:
            raise GenerationError("Failed to parse LLM response as JSON")
        except Exception as e:
            raise GenerationError(f"Entity recognition task generation failed: {str(e)}")
            
    async def _generate_paraphrasing(self) -> Tuple[str, Dict[str, Any]]:
        """Generate paraphrasing tasks."""
        categories = {
            "easy": [
                "simple sentences", "common phrases",
                "basic instructions", "everyday language"
            ],
            "medium": [
                "complex sentences", "technical content",
                "idiomatic expressions", "formal language"
            ],
            "hard": [
                "academic writing", "legal text",
                "literary passages", "specialized jargon"
            ]
        }
        
        constraints = {
            "easy": "Keep same meaning with simple changes",
            "medium": "Preserve meaning with style changes",
            "hard": "Transform style while maintaining semantics"
        }
        
        template_vars = {
            "difficulty": self.difficulty,
            "category": random.choice(categories[self.difficulty]),
            "constraints": constraints[self.difficulty],
            "format_instructions": self.format_instructions[self.task_type]
        }
        
        prompt = self.templates[self.task_type].substitute(template_vars)
        
        try:
            response = await self.model.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=1000
            )
            
            result = json.loads(response)
            
            required_fields = {"source", "paraphrase", "similarity"}
            if not all(field in result for field in required_fields):
                raise GenerationError("Invalid response format from LLM")
                
            return result["source"], result
            
        except json.JSONDecodeError:
            raise GenerationError("Failed to parse LLM response as JSON")
        except Exception as e:
            raise GenerationError(f"Paraphrasing task generation failed: {str(e)}")
            
    async def _generate_text_generation(self) -> Tuple[str, Dict[str, Any]]:
        """Generate text generation tasks."""
        categories = {
            "easy": [
                "short stories", "product descriptions",
                "social media posts", "simple emails"
            ],
            "medium": [
                "blog posts", "news articles",
                "marketing copy", "technical guides"
            ],
            "hard": [
                "research papers", "creative writing",
                "technical documentation", "policy documents"
            ]
        }
        
        constraints = {
            "easy": "Basic structure and style requirements",
            "medium": "Specific tone and format guidelines",
            "hard": "Complex structure and style constraints"
        }
        
        template_vars = {
            "difficulty": self.difficulty,
            "category": random.choice(categories[self.difficulty]),
            "constraints": constraints[self.difficulty],
            "format_instructions": self.format_instructions[self.task_type]
        }
        
        prompt = self.templates[self.task_type].substitute(template_vars)
        
        try:
            response = await self.model.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=1500
            )
            
            result = json.loads(response)
            
            required_fields = {"prompt", "text", "constraints"}
            if not all(field in result for field in required_fields):
                raise GenerationError("Invalid response format from LLM")
                
            return result["prompt"], result
            
        except json.JSONDecodeError:
            raise GenerationError("Failed to parse LLM response as JSON")
        except Exception as e:
            raise GenerationError(f"Text generation task generation failed: {str(e)}")
            
    async def _generate_translation(self) -> Tuple[str, Dict[str, Any]]:
        """Generate translation tasks."""
        categories = {
            "easy": [
                "everyday conversation", "simple instructions",
                "basic correspondence", "common phrases"
            ],
            "medium": [
                "business documents", "technical content",
                "marketing materials", "news articles"
            ],
            "hard": [
                "literary works", "legal documents",
                "academic papers", "cultural content"
            ]
        }
        
        constraints = {
            "easy": "Direct translation between similar languages",
            "medium": "Idiomatic translation with cultural context",
            "hard": "Complex translation with cultural adaptation"
        }
        
        template_vars = {
            "difficulty": self.difficulty,
            "category": random.choice(categories[self.difficulty]),
            "constraints": constraints[self.difficulty],
            "format_instructions": self.format_instructions[self.task_type]
        }
        
        prompt = self.templates[self.task_type].substitute(template_vars)
        
        try:
            response = await self.model.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=1500
            )
            
            result = json.loads(response)
            
            required_fields = {"source", "translation", "notes"}
            if not all(field in result for field in required_fields):
                raise GenerationError("Invalid response format from LLM")
                
            return result["source"], result
            
        except json.JSONDecodeError:
            raise GenerationError("Failed to parse LLM response as JSON")
        except Exception as e:
            raise GenerationError(f"Translation task generation failed: {str(e)}")
            
    def __repr__(self) -> str:
        """Return string representation of the generator."""
        return (
            f"NLUGenerator(type='{self.task_type}', "
            f"difficulty='{self.difficulty}', "
            f"model={self.model.__class__.__name__})"
        ) 