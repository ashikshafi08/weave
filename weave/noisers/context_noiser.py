"""Context-aware text transformation noiser."""

from typing import Any, Dict, List, Optional, Union
from ..core.base_noiser import BaseNoiser
from ..core.model_connector import ModelConnector


class ContextNoiser(BaseNoiser):
    """Noiser for context-aware text transformations.
    
    This noiser considers surrounding context when applying transformations,
    making it suitable for:
    - Document-level consistency
    - Conversation flow preservation
    - Topic-aware modifications
    """
    
    def __init__(self,
                 model_connector: ModelConnector,
                 context_config: Optional[Dict[str, Any]] = None):
        """Initialize the context noiser.
        
        Args:
            model_connector: LLM connector for transformations
            context_config: Configuration for context handling
        """
        super().__init__()
        self.model = model_connector
        self.config = context_config or {}
        
        # Default context window sizes
        self.window_sizes = {
            "document": 1000,    # characters for document context
            "conversation": 5,   # turns for conversation context
            "paragraph": 2       # paragraphs for local context
        }
        
        # Update with custom window sizes
        if "window_sizes" in self.config:
            self.window_sizes.update(self.config["window_sizes"])
            
    def augment(self, 
                query: str,
                context: Optional[Dict[str, Any]] = None) -> str:
        """Apply context-aware transformation.
        
        Args:
            query: Text to transform
            context: Additional context information
            
        Returns:
            Transformed text maintaining contextual consistency
        """
        context = context or {}
        context_type = self.config.get("context_type", "document")
        
        # Build context-aware prompt
        prompt = self._build_context_prompt(query, context, context_type)
        
        response = self.model.generate(
            prompt=prompt,
            max_tokens=self.config.get("max_tokens", 150),
            temperature=self.config.get("temperature", 0.7)
        )
        
        return response.strip()
        
    def batch_augment(self,
                     queries: List[str],
                     context: Optional[Dict[str, Any]] = None) -> List[str]:
        """Apply context-aware transformation to multiple texts.
        
        Args:
            queries: List of texts to transform
            context: Shared context information
            
        Returns:
            List of transformed texts
        """
        return [self.augment(q, context) for q in queries]
        
    def _build_context_prompt(self,
                            text: str,
                            context: Dict[str, Any],
                            context_type: str) -> str:
        """Build prompt incorporating context information.
        
        Args:
            text: Text to transform
            context: Context information
            context_type: Type of context to consider
            
        Returns:
            Context-aware prompt
        """
        if context_type == "document":
            return self._build_document_prompt(text, context)
        elif context_type == "conversation":
            return self._build_conversation_prompt(text, context)
        elif context_type == "paragraph":
            return self._build_paragraph_prompt(text, context)
        else:
            raise ValueError(f"Unknown context type: {context_type}")
            
    def _build_document_prompt(self,
                             text: str,
                             context: Dict[str, Any]) -> str:
        """Build prompt for document-level context."""
        document = context.get("document", "")
        topic = context.get("topic", "general")
        style = context.get("style", "neutral")
        
        return f"""Transform this text while maintaining consistency with the document context:
        
        Document excerpt:
        {document[:self.window_sizes["document"]]}
        
        Topic: {topic}
        Style: {style}
        
        Text to transform:
        {text}
        
        Instructions:
        1. Maintain consistency with document style and tone
        2. Preserve topic-specific terminology
        3. Ensure coherent flow with surrounding content
        """
        
    def _build_conversation_prompt(self,
                                 text: str,
                                 context: Dict[str, Any]) -> str:
        """Build prompt for conversation context."""
        history = context.get("conversation_history", [])
        speakers = context.get("speakers", {})
        
        # Take recent conversation turns
        recent_history = history[-self.window_sizes["conversation"]:]
        
        return f"""Transform this message while maintaining conversation flow:
        
        Recent conversation:
        {self._format_conversation(recent_history, speakers)}
        
        Message to transform:
        {text}
        
        Instructions:
        1. Maintain conversation coherence
        2. Preserve speaker characteristics
        3. Ensure natural dialogue flow
        """
        
    def _build_paragraph_prompt(self,
                              text: str,
                              context: Dict[str, Any]) -> str:
        """Build prompt for paragraph-level context."""
        before = context.get("previous_paragraphs", [])
        after = context.get("following_paragraphs", [])
        
        # Take adjacent paragraphs
        before = before[-self.window_sizes["paragraph"]:]
        after = after[:self.window_sizes["paragraph"]]
        
        return f"""Transform this paragraph while maintaining local context:
        
        Previous paragraphs:
        {' '.join(before)}
        
        Paragraph to transform:
        {text}
        
        Following paragraphs:
        {' '.join(after)}
        
        Instructions:
        1. Maintain logical flow between paragraphs
        2. Preserve transitions and connections
        3. Ensure consistent style and tone
        """
        
    def _format_conversation(self,
                           history: List[Dict[str, str]],
                           speakers: Dict[str, Dict[str, Any]]) -> str:
        """Format conversation history with speaker information."""
        formatted = []
        for turn in history:
            speaker_id = turn.get("speaker")
            speaker_info = speakers.get(speaker_id, {})
            speaker_name = speaker_info.get("name", speaker_id)
            formatted.append(f"{speaker_name}: {turn['text']}")
        return "\n".join(formatted)
        
    def get_augmentation_metadata(self) -> Dict[str, Any]:
        """Get metadata about the context-aware transformation.
        
        Returns:
            Dictionary containing context noiser configuration
        """
        return {
            "context_type": self.config.get("context_type", "document"),
            "window_sizes": self.window_sizes,
            "supported_contexts": ["document", "conversation", "paragraph"]
        }
