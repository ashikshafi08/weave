"""Domain-specific error noiser for specialized text."""

from typing import Any, Dict, List, Optional, Union
from ..core.base_noiser import BaseNoiser
from ..core.model_connector import ModelConnector


class DomainErrorNoiser(BaseNoiser):
    """Noiser for introducing domain-specific errors.
    
    Supports various domains like:
    - Programming/coding
    - Medical terminology
    - Legal text
    - Scientific writing
    """
    
    def __init__(self,
                 model_connector: ModelConnector,
                 domain_config: Optional[Dict[str, Any]] = None):
        """Initialize the domain error noiser.
        
        Args:
            model_connector: LLM connector for transformations
            domain_config: Configuration for domain-specific errors
        """
        super().__init__()
        self.model = model_connector
        self.domain_config = domain_config or {}
        
        # Load domain-specific error patterns
        self.domain_patterns = {
            "programming": {
                "syntax": ["missing semicolons", "bracket mismatches", "indentation"],
                "naming": ["camelCase/snake_case mixing", "reserved words"],
                "logic": ["off-by-one errors", "null pointer issues"]
            },
            "medical": {
                "terminology": ["similar drug names", "condition misspellings"],
                "abbreviations": ["ambiguous acronyms", "wrong expansions"],
                "dosage": ["unit confusion", "decimal placement"]
            },
            "legal": {
                "terminology": ["incorrect legal terms", "jurisdiction errors"],
                "formatting": ["citation errors", "clause numbering"],
                "references": ["wrong statute numbers", "case law errors"]
            },
            "scientific": {
                "notation": ["unit conversion errors", "significant figures"],
                "formatting": ["equation layout", "reference style"],
                "terminology": ["technical term confusion", "symbol misuse"]
            }
        }
        
        # Add custom domain patterns from config
        if "custom_domains" in self.domain_config:
            self.domain_patterns.update(self.domain_config["custom_domains"])
            
    def augment(self, query: str) -> str:
        """Apply domain-specific errors to a single query.
        
        Args:
            query: Original text to transform
            
        Returns:
            Text with domain-specific errors
        """
        domain = self.domain_config.get("domain", "programming")
        error_categories = self.domain_config.get("error_categories", ["syntax"])
        error_rate = self.domain_config.get("error_rate", 0.3)
        
        if domain not in self.domain_patterns:
            raise ValueError(f"Unsupported domain: {domain}")
            
        # Build prompt for domain-specific transformation
        prompt = f"""Transform this {domain} text by introducing domain-specific errors:
        Error categories: {', '.join(error_categories)}
        Error rate: {error_rate}
        
        Original text:
        {query}
        
        Instructions:
        1. Maintain overall structure and meaning
        2. Introduce realistic {domain}-specific errors
        3. Keep the error rate at approximately {error_rate * 100}%
        """
        
        response = self.model.generate(
            prompt=prompt,
            max_tokens=self.domain_config.get("max_tokens", 150),
            temperature=self.domain_config.get("temperature", 0.7)
        )
        
        return response.strip()
        
    def batch_augment(self, queries: List[str]) -> List[str]:
        """Apply domain-specific errors to multiple queries.
        
        Args:
            queries: List of original texts
            
        Returns:
            List of transformed texts
        """
        return [self.augment(q) for q in queries]
        
    def add_domain(self, domain: str, patterns: Dict[str, List[str]]) -> None:
        """Add a new domain with its error patterns.
        
        Args:
            domain: Name of the domain
            patterns: Dictionary of error patterns for the domain
        """
        self.domain_patterns[domain] = patterns
        
    def get_supported_domains(self) -> List[str]:
        """Get list of supported domains.
        
        Returns:
            List of domain names
        """
        return list(self.domain_patterns.keys())
        
    def get_error_categories(self, domain: str) -> List[str]:
        """Get available error categories for a domain.
        
        Args:
            domain: Name of the domain
            
        Returns:
            List of error category names
        """
        if domain not in self.domain_patterns:
            raise ValueError(f"Unsupported domain: {domain}")
        return list(self.domain_patterns[domain].keys())
        
    def get_augmentation_metadata(self) -> Dict[str, Any]:
        """Get metadata about the domain-specific transformation.
        
        Returns:
            Dictionary containing domain noiser configuration
        """
        domain = self.domain_config.get("domain", "programming")
        return {
            "domain": domain,
            "error_categories": self.domain_config.get("error_categories", ["syntax"]),
            "error_rate": self.domain_config.get("error_rate", 0.3),
            "available_categories": self.get_error_categories(domain),
            "supported_domains": self.get_supported_domains()
        }
