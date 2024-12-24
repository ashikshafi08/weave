"""LLM providers for the Weave framework.

This module provides concrete implementations of LLM providers that can be used
with the ModelConnector interface for text generation tasks.
"""

from .openai_llm import OpenAILLM
from .huggingface_llm import HuggingFaceLLM

__all__ = [
    "OpenAILLM",
    "HuggingFaceLLM",
] 