"""
LLM Adapters - Model-agnostic interface layer.

Provides a unified interface for multiple LLM providers:
- Ollama (local)
- Claude API (Anthropic)
- OpenAI API
- Groq
- Local GGUF models

Inspired by: aionui auto-detect and fallback patterns
"""

from adapters.base import LLMAdapter, LLMConfig, LLMProvider, LLMResponse
from adapters.router import DataClassification, LLMRouter, RoutingStrategy, get_router

__all__ = [
    "DataClassification",
    "LLMAdapter",
    "LLMConfig",
    "LLMProvider",
    "LLMResponse",
    "LLMRouter",
    "RoutingStrategy",
    "get_router",
]
