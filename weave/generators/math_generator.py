"""Math problem generator for the Weave framework."""

import random
from typing import Any, Dict, List, Optional, Tuple, Union
import sympy
from sympy.parsing.sympy_parser import parse_expr
import json
from string import Template

from ..core import (
    BaseGenerator,
    BaseNoiser,
    ModelError,
    GenerationError
)
from ..llms import OpenAILLM, HuggingFaceLLM

class MathGenerator(BaseGenerator):
    """Generator for mathematical problems and their solutions.
    
    This generator can create various types of math problems:
    - Arithmetic (basic operations)
    - Algebra (equations, expressions)
    - Word problems (using LLM for generation)
    - Calculus (derivatives, integrals)
    
    For word problems, it uses customizable prompt templates with variables:
    - ${difficulty}: Problem difficulty level
    - ${topic}: Math topic or scenario
    - ${operations}: Required mathematical operations
    - ${constraints}: Problem constraints
    - ${format_instructions}: Output format instructions
    """
    
    PROBLEM_TYPES = [
        "arithmetic",
        "algebra",
        "word",
        "calculus"
    ]
    
    # Default word problem template
    DEFAULT_WORD_PROBLEM_TEMPLATE = """Generate a ${difficulty} word problem about ${topic}.

Required operations: ${operations}
Constraints: ${constraints}

${format_instructions}"""

    # Default format instructions
    DEFAULT_FORMAT_INSTRUCTIONS = """Format the response as a JSON object with these fields:
- problem: the word problem text
- solution: the numerical answer
- explanation: step-by-step solution process"""
    
    def __init__(
        self,
        problem_type: str = "arithmetic",
        difficulty: str = "medium",
        model_connector: Optional[Union[OpenAILLM, HuggingFaceLLM]] = None,
        noisers: Optional[List[BaseNoiser]] = None,
        word_problem_template: Optional[str] = None,
        format_instructions: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the math generator.
        
        Args:
            problem_type: Type of math problems to generate.
            difficulty: Difficulty level ("easy", "medium", "hard").
            model_connector: Optional LLM provider for word problem generation.
            noisers: Optional list of noisers for data augmentation.
            word_problem_template: Custom template for word problems.
            format_instructions: Custom format instructions.
            config: Additional configuration options.
        """
        super().__init__(config)
        
        if problem_type not in self.PROBLEM_TYPES:
            raise ValueError(f"Invalid problem type: {problem_type}")
            
        self.problem_type = problem_type
        self.difficulty = difficulty
        self.model = model_connector or OpenAILLM()  # Default to OpenAI
        self.noisers = noisers or []
        
        # Set up templates for word problems
        self.word_problem_template = Template(word_problem_template or self.DEFAULT_WORD_PROBLEM_TEMPLATE)
        self.format_instructions = format_instructions or self.DEFAULT_FORMAT_INSTRUCTIONS
        
        # Set up SymPy for symbolic math
        self.x, self.y, self.z = sympy.symbols('x y z')
        
    async def generate(self) -> Tuple[str, Union[str, Dict[str, Any]]]:
        """Generate a single math problem and its solution.
        
        Returns:
            Tuple containing:
                - Problem statement (str)
                - Solution (str for simple answers, dict for detailed solutions)
                
        Raises:
            GenerationError: If generation fails.
        """
        try:
            # Generate base problem and solution
            if self.problem_type == "arithmetic":
                problem, solution = self._generate_arithmetic()
            elif self.problem_type == "algebra":
                problem, solution = self._generate_algebra()
            elif self.problem_type == "word":
                problem, solution = await self._generate_word_problem()
            elif self.problem_type == "calculus":
                problem, solution = self._generate_calculus()
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
                        
                # If solution is a dict, add noise metadata
                if isinstance(solution, dict):
                    solution["noise_metadata"] = noise_metadata
                    
                return noised_problem, solution
                
            return problem, solution
            
        except Exception as e:
            raise GenerationError(f"Failed to generate math problem: {str(e)}")
            
    async def batch_generate(
        self,
        batch_size: int
    ) -> List[Tuple[str, Union[str, Dict[str, Any]]]]:
        """Generate multiple math problems.
        
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
        
    def _generate_arithmetic(self) -> Tuple[str, str]:
        """Generate arithmetic problems (basic operations)."""
        if self.difficulty == "easy":
            # Single operation with small numbers
            a = random.randint(1, 20)
            b = random.randint(1, 20)
            op = random.choice(['+', '-', '*'])
            
            problem = f"{a} {op} {b}"
            solution = str(eval(problem))
            
        elif self.difficulty == "medium":
            # Two operations with medium numbers
            a = random.randint(10, 50)
            b = random.randint(10, 50)
            c = random.randint(10, 50)
            ops = random.sample(['+', '-', '*'], 2)
            
            problem = f"{a} {ops[0]} {b} {ops[1]} {c}"
            solution = str(eval(problem))
            
        else:  # hard
            # Multiple operations with larger numbers and division
            a = random.randint(50, 200)
            b = random.randint(50, 200)
            c = random.randint(50, 200)
            d = random.randint(1, 20)  # smaller number for division
            ops = random.sample(['+', '-', '*', '/'], 3)
            
            problem = f"({a} {ops[0]} {b}) {ops[1]} ({c} {ops[2]} {d})"
            solution = f"{eval(problem):.2f}"
            
        return f"Calculate: {problem}", solution
        
    def _generate_algebra(self) -> Tuple[str, Dict[str, Any]]:
        """Generate algebra problems (equations, expressions)."""
        if self.difficulty == "easy":
            # Linear equation with one variable
            a = random.randint(1, 10)
            b = random.randint(1, 20)
            
            equation = sympy.Eq(a * self.x + b, 0)
            solution = sympy.solve(equation, self.x)[0]
            
            return (
                f"Solve for x: {a}x + {b} = 0",
                {
                    "solution": str(solution),
                    "steps": [
                        f"Move {b} to right side: {a}x = -{b}",
                        f"Divide both sides by {a}: x = {solution}"
                    ]
                }
            )
            
        elif self.difficulty == "medium":
            # Quadratic equation
            a = random.randint(1, 5)
            b = random.randint(-10, 10)
            c = random.randint(-10, 10)
            
            equation = sympy.Eq(a * self.x**2 + b * self.x + c, 0)
            solutions = sympy.solve(equation, self.x)
            
            return (
                f"Solve for x: {a}x² + {b}x + {c} = 0",
                {
                    "solutions": [str(sol) for sol in solutions],
                    "steps": [
                        f"Using quadratic formula: x = (-{b} ± √({b}² - 4({a})({c}))) / (2({a}))",
                        f"x = {solutions[0]} or x = {solutions[1]}"
                    ]
                }
            )
            
        else:  # hard
            # System of equations
            a1 = random.randint(1, 5)
            b1 = random.randint(1, 5)
            c1 = random.randint(1, 20)
            a2 = random.randint(1, 5)
            b2 = random.randint(1, 5)
            c2 = random.randint(1, 20)
            
            eq1 = sympy.Eq(a1 * self.x + b1 * self.y, c1)
            eq2 = sympy.Eq(a2 * self.x + b2 * self.y, c2)
            solution = sympy.solve((eq1, eq2), (self.x, self.y))
            
            return (
                f"Solve the system of equations:\n"
                f"{a1}x + {b1}y = {c1}\n"
                f"{a2}x + {b2}y = {c2}",
                {
                    "solution": {
                        "x": str(solution[self.x]),
                        "y": str(solution[self.y])
                    },
                    "steps": [
                        "Using substitution method:",
                        f"From equation 1: y = ({c1} - {a1}x) / {b1}",
                        f"Substitute into equation 2: {a2}x + {b2}(({c1} - {a1}x) / {b1}) = {c2}",
                        f"Solve for x: x = {solution[self.x]}",
                        f"Substitute back: y = {solution[self.y]}"
                    ]
                }
            )
            
    async def _generate_word_problem(self) -> Tuple[str, Dict[str, Any]]:
        """Generate word problems using LLM."""
        # Select random topic and operations based on difficulty
        topics = {
            "easy": ["shopping", "distance", "time", "basic geometry"],
            "medium": ["percentages", "ratios", "compound interest", "probability"],
            "hard": ["optimization", "rate problems", "work problems", "advanced geometry"]
        }
        
        operations = {
            "easy": ["addition", "subtraction", "multiplication"],
            "medium": ["division", "percentages", "ratios", "proportions"],
            "hard": ["systems of equations", "quadratic equations", "calculus"]
        }
        
        topic = random.choice(topics[self.difficulty])
        required_ops = random.sample(operations[self.difficulty], 2)
        
        # Build prompt from template
        prompt = self.word_problem_template.substitute(
            difficulty=self.difficulty,
            topic=topic,
            operations=", ".join(required_ops),
            constraints="Numbers should be realistic and practical",
            format_instructions=self.format_instructions
        )
        
        try:
            # Generate problem using LLM
            response = await self.model.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=500
            )
            
            # Parse response as JSON
            result = json.loads(response)
            
            # Validate response format
            required_fields = {"problem", "solution", "explanation"}
            if not all(field in result for field in required_fields):
                raise GenerationError("Invalid response format from LLM")
                
            return result["problem"], {
                "solution": result["solution"],
                "explanation": result["explanation"]
            }
            
        except json.JSONDecodeError:
            raise GenerationError("Failed to parse LLM response as JSON")
        except Exception as e:
            raise GenerationError(f"Word problem generation failed: {str(e)}")
            
    def _generate_calculus(self) -> Tuple[str, Dict[str, Any]]:
        """Generate calculus problems (derivatives, integrals)."""
        if self.difficulty == "easy":
            # Simple derivatives
            a = random.randint(1, 5)
            n = random.randint(2, 4)
            
            expr = a * self.x**n
            derivative = sympy.diff(expr, self.x)
            
            return (
                f"Find the derivative of f(x) = {a}x^{n}",
                {
                    "solution": str(derivative),
                    "steps": [
                        f"Using power rule: d/dx(x^n) = n * x^(n-1)",
                        f"d/dx({a}x^{n}) = {a} * {n} * x^{n-1}",
                        f"f'(x) = {derivative}"
                    ]
                }
            )
            
        elif self.difficulty == "medium":
            # Product rule or chain rule
            a = random.randint(1, 5)
            b = random.randint(1, 5)
            
            expr = a * self.x**2 * sympy.sin(b * self.x)
            derivative = sympy.diff(expr, self.x)
            
            return (
                f"Find the derivative of f(x) = {a}x^2 * sin({b}x)",
                {
                    "solution": str(derivative),
                    "steps": [
                        "Using product rule: d/dx(u*v) = u'v + uv'",
                        f"u = {a}x^2, v = sin({b}x)",
                        f"u' = {2*a}x",
                        f"v' = {b}cos({b}x)",
                        f"f'(x) = {derivative}"
                    ]
                }
            )
            
        else:  # hard
            # Integration
            a = random.randint(1, 5)
            b = random.randint(1, 5)
            n = random.randint(2, 4)
            
            expr = a * self.x**n + b * sympy.sin(self.x)
            integral = sympy.integrate(expr, self.x)
            
            return (
                f"Find the indefinite integral of f(x) = {a}x^{n} + {b}sin(x)",
                {
                    "solution": str(integral) + " + C",
                    "steps": [
                        "Split into two integrals:",
                        f"∫({a}x^{n})dx + ∫({b}sin(x))dx",
                        f"For power term: ∫(x^n)dx = x^(n+1)/(n+1)",
                        f"For sin term: ∫sin(x)dx = -cos(x)",
                        f"Result: {integral} + C"
                    ]
                }
            )
            
    def __repr__(self) -> str:
        """Return string representation of the generator."""
        return (
            f"MathGenerator(type='{self.problem_type}', "
            f"difficulty='{self.difficulty}', "
            f"model={self.model.__class__.__name__})"
        ) 