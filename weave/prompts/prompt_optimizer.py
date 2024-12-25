"""Prompt optimization through feedback and auto-tuning."""

from typing import Any, Dict, List, Optional, Union, Callable
from .prompt_template import PromptTemplate
from ..core.model_connector import ModelConnector
import json
from pathlib import Path


class PromptOptimizer:
    """Class for optimizing prompts through feedback loops."""
    
    def __init__(self,
                 model_connector: ModelConnector,
                 optimization_config: Optional[Dict[str, Any]] = None):
        """Initialize the prompt optimizer.
        
        Args:
            model_connector: LLM connector for testing prompts
            optimization_config: Configuration for optimization
        """
        self.model = model_connector
        self.config = optimization_config or {}
        self.history: List[Dict[str, Any]] = []
        
    def optimize(self,
                template: PromptTemplate,
                test_cases: List[Dict[str, Any]],
                evaluation_fn: Callable[[str, Any], float],
                num_iterations: int = 5) -> PromptTemplate:
        """Optimize a prompt template using test cases.
        
        Args:
            template: Initial prompt template
            test_cases: List of test cases with inputs and expected outputs
            evaluation_fn: Function to evaluate prompt performance
            num_iterations: Number of optimization iterations
            
        Returns:
            Optimized PromptTemplate
        """
        best_template = template
        best_score = float("-inf")
        
        for i in range(num_iterations):
            # Generate variations
            variations = self._generate_variations(best_template)
            
            # Evaluate variations
            for variant in variations:
                score = self._evaluate_template(
                    variant,
                    test_cases,
                    evaluation_fn
                )
                
                if score > best_score:
                    best_template = variant
                    best_score = score
                    
                # Record history
                self.history.append({
                    "iteration": i,
                    "template": variant.template.template,
                    "score": score,
                    "metadata": variant.get_metadata()
                })
                
        return best_template
        
    def _generate_variations(self,
                           template: PromptTemplate,
                           num_variations: int = 3) -> List[PromptTemplate]:
        """Generate variations of a template.
        
        Args:
            template: Base template
            num_variations: Number of variations to generate
            
        Returns:
            List of template variations
        """
        variations = []
        base_prompt = f"""Generate {num_variations} variations of this prompt template
        while preserving its variables and core functionality:
        
        Original template:
        {template.template.template}
        
        Required variables: {list(template.required_variables)}
        """
        
        response = self.model.generate(
            prompt=base_prompt,
            max_tokens=self.config.get("max_tokens", 300),
            temperature=self.config.get("temperature", 0.7)
        )
        
        # Parse variations from response
        # This is a simplified version; in practice, you'd want more robust parsing
        variant_texts = response.strip().split("\n\n")
        for i, text in enumerate(variant_texts[:num_variations]):
            variations.append(PromptTemplate(
                text,
                {**template.get_metadata(), "variation": i}
            ))
            
        return variations
        
    def _evaluate_template(self,
                          template: PromptTemplate,
                          test_cases: List[Dict[str, Any]],
                          evaluation_fn: Callable[[str, Any], float]) -> float:
        """Evaluate a template using test cases.
        
        Args:
            template: Template to evaluate
            test_cases: List of test cases
            evaluation_fn: Function to compute score
            
        Returns:
            Average score across test cases
        """
        scores = []
        for test_case in test_cases:
            try:
                # Generate response using template
                prompt = template.render(test_case["input"])
                response = self.model.generate(
                    prompt=prompt,
                    max_tokens=self.config.get("max_tokens", 150)
                )
                
                # Evaluate response
                score = evaluation_fn(response, test_case["expected"])
                scores.append(score)
                
            except Exception as e:
                # Log error and continue
                print(f"Error evaluating template: {e}")
                scores.append(0.0)
                
        return sum(scores) / len(scores) if scores else 0.0
        
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get history of optimization attempts.
        
        Returns:
            List of optimization records
        """
        return self.history.copy()
        
    def save_history(self, path: Union[str, Path]) -> None:
        """Save optimization history to file.
        
        Args:
            path: Path to save history
        """
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
            
    def load_history(self, path: Union[str, Path]) -> None:
        """Load optimization history from file.
        
        Args:
            path: Path to history file
        """
        with open(path) as f:
            self.history = json.load(f)
            
    def get_best_template(self) -> Optional[PromptTemplate]:
        """Get the best performing template from history.
        
        Returns:
            Best PromptTemplate or None if no history
        """
        if not self.history:
            return None
            
        best_record = max(self.history, key=lambda x: x["score"])
        return PromptTemplate(
            best_record["template"],
            best_record["metadata"]
        )
