"""Code problem generator for the Weave framework."""

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

class CodeGenerator(BaseGenerator):
    """Generator for coding problems and their solutions.
    
    This generator can create various types of coding problems:
    - Algorithms (sorting, searching, etc.)
    - Data structures (linked lists, trees, etc.)
    - System design (architecture, patterns)
    - Bug fixing (find and fix issues)
    
    It uses customizable prompt templates with variables:
    - ${difficulty}: Problem difficulty level
    - ${language}: Target programming language
    - ${problem_type}: Type of coding problem
    - ${topic}: Specific algorithm/structure/component
    - ${constraints}: Problem constraints and requirements
    - ${format_instructions}: Output format instructions
    """
    
    PROBLEM_TYPES = [
        "algorithm",
        "data_structure",
        "system_design",
        "bug_fixing"
    ]
    
    LANGUAGES = [
        "python",
        "javascript",
        "java",
        "cpp",
        "go",
        "rust",
        "typescript"
    ]
    
    # Default templates for each problem type
    DEFAULT_TEMPLATES = {
        "algorithm": """Generate a ${difficulty} algorithm problem about ${topic} in ${language}.

Required implementation:
- Time complexity: ${constraints}
- Space complexity: ${constraints}

${format_instructions}""",

        "data_structure": """Generate a ${difficulty} data structure problem about ${topic} in ${language}.

Implementation requirements:
- Operations: ${constraints}
- Efficiency: ${constraints}

${format_instructions}""",

        "system_design": """Generate a ${difficulty} system design problem about ${topic} in ${language}.

Design requirements:
- Components: ${constraints}
- Scalability: ${constraints}
- Performance: ${constraints}
- Maintainability: ${constraints}

${format_instructions}""",

        "bug_fixing": """Generate a ${difficulty} bug fixing problem with a ${topic} in ${language}.

Bug characteristics:
- Type: ${constraints}
- Complexity: ${constraints}
- Common pitfalls: ${constraints}

${format_instructions}"""
    }
    
    # Default format instructions for each problem type
    DEFAULT_FORMAT_INSTRUCTIONS = {
        "algorithm": """Format the response as a JSON object with these fields:
- problem: detailed problem description
- solution: complete code solution with comments
- explanation: step-by-step explanation of the algorithm
- complexity: time and space complexity analysis
- test_cases: list of test cases with inputs and expected outputs
- edge_cases: list of edge cases to consider""",

        "data_structure": """Format the response as a JSON object with these fields:
- problem: detailed problem description
- solution: complete implementation with comments
- explanation: implementation details and design choices
- complexity: time complexity for each operation
- test_cases: list of test cases with inputs and expected outputs
- usage_example: example usage of the data structure""",

        "system_design": """Format the response as a JSON object with these fields:
- problem: system requirements and constraints
- solution: high-level design and key components
- explanation: design decisions and tradeoffs
- implementation: sample code for core components
- scalability: scalability considerations
- testing: testing strategy and considerations""",

        "bug_fixing": """Format the response as a JSON object with these fields:
- problem: buggy code and issue description
- solution: fixed code with comments
- explanation: bug analysis and fix explanation
- test_cases: list of test cases that expose the bug
- prevention: how to prevent similar bugs
- debugging_steps: steps to identify the bug"""
    }
    
    def __init__(
        self,
        problem_type: str = "algorithm",
        language: str = "python",
        difficulty: str = "medium",
        model_connector: Optional[Union[OpenAILLM, HuggingFaceLLM]] = None,
        noisers: Optional[List[BaseNoiser]] = None,
        templates: Optional[Dict[str, str]] = None,
        format_instructions: Optional[Dict[str, str]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the code generator.
        
        Args:
            problem_type: Type of coding problems to generate.
            language: Target programming language.
            difficulty: Difficulty level ("easy", "medium", "hard").
            model_connector: LLM provider for code generation.
            noisers: Optional list of noisers for data augmentation.
            templates: Custom templates for each problem type.
            format_instructions: Custom format instructions.
            config: Additional configuration options.
        """
        super().__init__(config)
        
        if problem_type not in self.PROBLEM_TYPES:
            raise ValueError(f"Invalid problem type: {problem_type}")
            
        if language not in self.LANGUAGES:
            raise ValueError(f"Unsupported language: {language}")
            
        self.problem_type = problem_type
        self.language = language
        self.difficulty = difficulty
        self.model = model_connector or OpenAILLM()  # Default to OpenAI
        self.noisers = noisers or []
        
        # Set up templates and format instructions
        self.templates = {
            ptype: Template(template)
            for ptype, template in (templates or self.DEFAULT_TEMPLATES).items()
        }
        self.format_instructions = format_instructions or self.DEFAULT_FORMAT_INSTRUCTIONS
        
    async def generate(self) -> Tuple[str, Dict[str, Any]]:
        """Generate a single coding problem and its solution.
        
        Returns:
            Tuple containing:
                - Problem statement (str)
                - Solution (dict with code, explanation, and test cases)
                
        Raises:
            GenerationError: If generation fails.
        """
        try:
            # Generate base problem and solution
            if self.problem_type == "algorithm":
                problem, solution = await self._generate_algorithm()
            elif self.problem_type == "data_structure":
                problem, solution = await self._generate_data_structure()
            elif self.problem_type == "system_design":
                problem, solution = await self._generate_system_design()
            elif self.problem_type == "bug_fixing":
                problem, solution = await self._generate_bug_fixing()
            else:
                raise GenerationError(f"Unsupported problem type: {self.problem_type}")
                
            # Apply noisers to the problem text if any are configured
            if self.noisers:
                # Keep track of noise transformations
                noise_metadata = []
                
                # Apply each noiser in sequence
                noised_problem = problem
                for noiser in self.noisers:
                    try:
                        noised_problem = await noiser.augment(noised_problem)
                        noise_metadata.append(noiser.get_augmentation_metadata())
                    except Exception as e:
                        self.logger.warning(f"Noiser {noiser} failed: {str(e)}")
                        
                # Add noise metadata to solution
                solution["noise_metadata"] = noise_metadata
                return noised_problem, solution
                
            return problem, solution
            
        except Exception as e:
            raise GenerationError(f"Failed to generate coding problem: {str(e)}")
            
    async def batch_generate(
        self,
        batch_size: int
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Generate multiple coding problems.
        
        Args:
            batch_size: Number of problems to generate.
            
        Returns:
            List of (problem, solution) tuples.
            
        Raises:
            GenerationError: If generation fails.
        """
        problems = []
        for _ in range(batch_size):
            problem = await self.generate()
            problems.append(problem)
        return problems
        
    async def _generate_algorithm(self) -> Tuple[str, Dict[str, Any]]:
        """Generate algorithm problems."""
        algorithms = {
            "easy": [
                "binary search", "bubble sort", "fibonacci",
                "two sum", "palindrome check", "reverse string"
            ],
            "medium": [
                "quicksort", "merge sort", "binary tree traversal",
                "dynamic programming", "backtracking", "sliding window"
            ],
            "hard": [
                "graph algorithms", "advanced trees", "optimization",
                "string matching", "network flow", "computational geometry"
            ]
        }
        
        constraints = {
            "easy": "O(n) time, O(1) space preferred",
            "medium": "O(n log n) time, O(n) space acceptable",
            "hard": "Optimize for specific cases, consider tradeoffs"
        }
        
        # Build template variables
        template_vars = {
            "difficulty": self.difficulty,
            "language": self.language,
            "topic": random.choice(algorithms[self.difficulty]),
            "constraints": constraints[self.difficulty],
            "format_instructions": self.format_instructions[self.problem_type]
        }
        
        # Generate prompt from template
        prompt = self.templates[self.problem_type].substitute(template_vars)
        
        try:
            # Generate problem using LLM
            response = await self.model.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=1000
            )
            
            # Parse response as JSON
            result = json.loads(response)
            
            # Validate response format
            required_fields = {"problem", "solution", "explanation", "test_cases"}
            if not all(field in result for field in required_fields):
                raise GenerationError("Invalid response format from LLM")
                
            return result["problem"], result
            
        except json.JSONDecodeError:
            raise GenerationError("Failed to parse LLM response as JSON")
        except Exception as e:
            raise GenerationError(f"Algorithm generation failed: {str(e)}")
            
    async def _generate_data_structure(self) -> Tuple[str, Dict[str, Any]]:
        """Generate data structure problems."""
        structures = {
            "easy": [
                "stack", "queue", "linked list",
                "array list", "hash table", "binary search tree"
            ],
            "medium": [
                "heap", "trie", "graph",
                "AVL tree", "disjoint set", "LRU cache"
            ],
            "hard": [
                "red-black tree", "B-tree", "segment tree",
                "bloom filter", "skip list", "concurrent data structures"
            ]
        }
        
        constraints = {
            "easy": "Basic operations with good efficiency",
            "medium": "Advanced operations, handle edge cases",
            "hard": "Complex operations, thread safety, persistence"
        }
        
        template_vars = {
            "difficulty": self.difficulty,
            "language": self.language,
            "topic": random.choice(structures[self.difficulty]),
            "constraints": constraints[self.difficulty],
            "format_instructions": self.format_instructions[self.problem_type]
        }
        
        prompt = self.templates[self.problem_type].substitute(template_vars)
        
        try:
            response = await self.model.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=1000
            )
            
            result = json.loads(response)
            
            required_fields = {"problem", "solution", "explanation", "test_cases"}
            if not all(field in result for field in required_fields):
                raise GenerationError("Invalid response format from LLM")
                
            return result["problem"], result
            
        except json.JSONDecodeError:
            raise GenerationError("Failed to parse LLM response as JSON")
        except Exception as e:
            raise GenerationError(f"Data structure generation failed: {str(e)}")
            
    async def _generate_system_design(self) -> Tuple[str, Dict[str, Any]]:
        """Generate system design problems."""
        systems = {
            "easy": [
                "key-value store", "rate limiter", "url shortener",
                "logging system", "caching layer", "job queue"
            ],
            "medium": [
                "chat system", "notification service", "task scheduler",
                "file storage service", "search service", "recommendation system"
            ],
            "hard": [
                "distributed database", "social network", "video streaming",
                "payment system", "real-time analytics", "distributed cache"
            ]
        }
        
        constraints = {
            "easy": "Basic scalability and reliability",
            "medium": "High availability and performance",
            "hard": "Global scale, consistency, fault tolerance"
        }
        
        template_vars = {
            "difficulty": self.difficulty,
            "language": self.language,
            "topic": random.choice(systems[self.difficulty]),
            "constraints": constraints[self.difficulty],
            "format_instructions": self.format_instructions[self.problem_type]
        }
        
        prompt = self.templates[self.problem_type].substitute(template_vars)
        
        try:
            response = await self.model.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=1500
            )
            
            result = json.loads(response)
            
            required_fields = {"problem", "solution", "explanation", "implementation"}
            if not all(field in result for field in required_fields):
                raise GenerationError("Invalid response format from LLM")
                
            return result["problem"], result
            
        except json.JSONDecodeError:
            raise GenerationError("Failed to parse LLM response as JSON")
        except Exception as e:
            raise GenerationError(f"System design generation failed: {str(e)}")
            
    async def _generate_bug_fixing(self) -> Tuple[str, Dict[str, Any]]:
        """Generate bug fixing problems."""
        bug_types = {
            "easy": [
                "off-by-one errors", "null pointer exceptions", "type errors",
                "array bounds", "string manipulation", "basic logic errors"
            ],
            "medium": [
                "race conditions", "memory leaks", "performance issues",
                "API misuse", "error handling", "state management"
            ],
            "hard": [
                "deadlocks", "concurrency bugs", "security vulnerabilities",
                "distributed systems", "resource leaks", "complex edge cases"
            ]
        }
        
        constraints = {
            "easy": "Common programming mistakes",
            "medium": "Subtle bugs and edge cases",
            "hard": "Complex interactions and timing issues"
        }
        
        template_vars = {
            "difficulty": self.difficulty,
            "language": self.language,
            "topic": random.choice(bug_types[self.difficulty]),
            "constraints": constraints[self.difficulty],
            "format_instructions": self.format_instructions[self.problem_type]
        }
        
        prompt = self.templates[self.problem_type].substitute(template_vars)
        
        try:
            response = await self.model.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=1000
            )
            
            result = json.loads(response)
            
            required_fields = {"problem", "solution", "explanation", "test_cases"}
            if not all(field in result for field in required_fields):
                raise GenerationError("Invalid response format from LLM")
                
            return result["problem"], result
            
        except json.JSONDecodeError:
            raise GenerationError("Failed to parse LLM response as JSON")
        except Exception as e:
            raise GenerationError(f"Bug fixing generation failed: {str(e)}")
            
    def __repr__(self) -> str:
        """Return string representation of the generator."""
        return (
            f"CodeGenerator(type='{self.problem_type}', "
            f"language='{self.language}', "
            f"difficulty='{self.difficulty}', "
            f"model={self.model.__class__.__name__})"
        ) 