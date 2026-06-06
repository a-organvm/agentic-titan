"""
LLM Router - Intelligent provider selection and fallback.

Features:
- Auto-detection of available providers
- Fallback chains
- Cost-based routing
- Capability-based selection

Inspired by: aionui auto-detect and fallback patterns
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from adapters.base import (
    AnthropicAdapter,
    GroqAdapter,
    LLMAdapter,
    LLMConfig,
    LLMMessage,
    LLMProvider,
    LLMResponse,
    OllamaAdapter,
    OpenAIAdapter,
    Tool,
)
from agents.framework.errors import LLMAdapterError

logger = logging.getLogger("titan.adapters.router")


class RoutingStrategy(StrEnum):
    """Routing strategies for provider selection."""

    COST_OPTIMIZED = "cost_optimized"  # Prefer cheaper/local options
    QUALITY_FIRST = "quality_first"  # Prefer best quality
    SPEED_FIRST = "speed_first"  # Prefer fastest
    ROUND_ROBIN = "round_robin"  # Rotate between providers
    FALLBACK = "fallback"  # Use fallback chain
    SENSITIVITY = "sensitivity"  # Route based on data classification


class DataClassification(StrEnum):
    """Data sensitivity classification for routing decisions.

    Determines which providers are allowed for a given request:
    - PUBLIC: Any provider (cloud or local)
    - INTERNAL: Cloud or local (all providers)
    - CONFIDENTIAL: Local only (ollama, local GGUF)
    - REGULATED: Local only with audit logging
    """

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    REGULATED = "regulated"


# Which providers are allowed per classification
CLASSIFICATION_ALLOWED_PROVIDERS: dict[DataClassification, set[LLMProvider]] = {
    DataClassification.PUBLIC: {
        LLMProvider.OLLAMA,
        LLMProvider.ANTHROPIC,
        LLMProvider.OPENAI,
        LLMProvider.GROQ,
        LLMProvider.LOCAL,
    },
    DataClassification.INTERNAL: {
        LLMProvider.OLLAMA,
        LLMProvider.ANTHROPIC,
        LLMProvider.OPENAI,
        LLMProvider.GROQ,
        LLMProvider.LOCAL,
    },
    DataClassification.CONFIDENTIAL: {
        LLMProvider.OLLAMA,
        LLMProvider.LOCAL,
    },
    DataClassification.REGULATED: {
        LLMProvider.OLLAMA,
        LLMProvider.LOCAL,
    },
}


@dataclass
class ProviderInfo:
    """Information about a provider."""

    provider: LLMProvider
    available: bool
    models: list[str]
    supports_tools: bool
    supports_streaming: bool
    cost_tier: int  # 1=free, 2=cheap, 3=standard, 4=expensive
    quality_tier: int  # 1=basic, 2=good, 3=great, 4=best
    speed_tier: int  # 1=slow, 2=medium, 3=fast, 4=fastest


# Known provider characteristics
PROVIDER_INFO: dict[LLMProvider, dict[str, Any]] = {
    LLMProvider.OLLAMA: {
        "cost_tier": 1,
        "quality_tier": 2,
        "speed_tier": 2,
        "supports_tools": False,
    },
    LLMProvider.ANTHROPIC: {
        "cost_tier": 3,
        "quality_tier": 4,
        "speed_tier": 3,
        "supports_tools": True,
    },
    LLMProvider.OPENAI: {
        "cost_tier": 3,
        "quality_tier": 4,
        "speed_tier": 3,
        "supports_tools": True,
    },
    LLMProvider.GROQ: {
        "cost_tier": 2,
        "quality_tier": 3,
        "speed_tier": 4,
        "supports_tools": False,
    },
}


# Default models per provider
DEFAULT_MODELS: dict[LLMProvider, str] = {
    LLMProvider.OLLAMA: "llama3.2",
    LLMProvider.ANTHROPIC: "claude-sonnet-4-20250514",
    LLMProvider.OPENAI: "gpt-4o-mini",
    LLMProvider.GROQ: "llama-3.3-70b-versatile",
}


class LLMRouter:
    """
    Routes requests to appropriate LLM providers.

    Handles:
    - Provider auto-detection
    - Fallback chains
    - Cost/quality/speed optimization
    - Capability matching
    """

    def __init__(
        self,
        strategy: RoutingStrategy = RoutingStrategy.FALLBACK,
        preferred_providers: list[LLMProvider] | None = None,
    ) -> None:
        self.strategy = strategy
        self.preferred_providers = preferred_providers or []

        self._providers: dict[LLMProvider, ProviderInfo] = {}
        self._adapters: dict[LLMProvider, LLMAdapter] = {}
        self._fallback_chain: list[LLMProvider] = []
        self._initialized = False

    async def initialize(self) -> None:
        """Detect and initialize available providers."""
        if self._initialized:
            return

        # Check each provider
        for provider in LLMProvider:
            available, models = await self._check_provider(provider)
            info = PROVIDER_INFO.get(provider, {})

            self._providers[provider] = ProviderInfo(
                provider=provider,
                available=available,
                models=models,
                supports_tools=info.get("supports_tools", False),
                supports_streaming=True,
                cost_tier=info.get("cost_tier", 3),
                quality_tier=info.get("quality_tier", 2),
                speed_tier=info.get("speed_tier", 2),
            )

            if available:
                logger.info(f"Provider available: {provider.value} ({len(models)} models)")

        # Build fallback chain based on strategy
        self._build_fallback_chain()

        self._initialized = True
        logger.info(
            f"Router initialized with strategy={self.strategy.value}, "
            f"available={[p.value for p, i in self._providers.items() if i.available]}"
        )

    async def _check_provider(self, provider: LLMProvider) -> tuple[bool, list[str]]:
        """Check if a provider is available."""
        try:
            if provider == LLMProvider.OLLAMA:
                return await self._check_ollama()
            elif provider == LLMProvider.ANTHROPIC:
                return self._check_anthropic()
            elif provider == LLMProvider.OPENAI:
                return self._check_openai()
            elif provider == LLMProvider.GROQ:
                return self._check_groq()
        except Exception as e:
            logger.debug(f"Provider {provider.value} check failed: {e}")
        return False, []

    async def _check_ollama(self) -> tuple[bool, list[str]]:
        """Check Ollama availability."""
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "http://localhost:11434/api/tags",
                    timeout=2.0,
                )
                if response.status_code == 200:
                    data = response.json()
                    models = [m["name"] for m in data.get("models", [])]
                    return True, models
        except Exception:
            pass
        return False, []

    def _check_anthropic(self) -> tuple[bool, list[str]]:
        """Check Anthropic API availability."""
        if os.environ.get("ANTHROPIC_API_KEY"):
            return True, [
                "claude-opus-4-20250514",
                "claude-sonnet-4-20250514",
                "claude-3-5-haiku-20241022",
            ]
        return False, []

    def _check_openai(self) -> tuple[bool, list[str]]:
        """Check OpenAI API availability."""
        if os.environ.get("OPENAI_API_KEY"):
            return True, ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]
        return False, []

    def _check_groq(self) -> tuple[bool, list[str]]:
        """Check Groq API availability."""
        if os.environ.get("GROQ_API_KEY"):
            return True, [
                "llama-3.3-70b-versatile",
                "mixtral-8x7b-32768",
                "gemma2-9b-it",
            ]
        return False, []

    def _build_fallback_chain(self) -> None:
        """Build the fallback chain based on strategy."""
        available = [p for p, i in self._providers.items() if i.available]

        if self.preferred_providers:
            # Use preferred order, then add remaining
            chain = [p for p in self.preferred_providers if p in available]
            chain.extend([p for p in available if p not in chain])
        else:
            # Build based on strategy
            if self.strategy == RoutingStrategy.COST_OPTIMIZED:
                chain = sorted(
                    available,
                    key=lambda p: self._providers[p].cost_tier,
                )
            elif self.strategy == RoutingStrategy.QUALITY_FIRST:
                chain = sorted(
                    available,
                    key=lambda p: -self._providers[p].quality_tier,
                )
            elif self.strategy == RoutingStrategy.SPEED_FIRST:
                chain = sorted(
                    available,
                    key=lambda p: -self._providers[p].speed_tier,
                )
            else:
                # Default fallback order
                chain = available

        self._fallback_chain = chain
        logger.debug(f"Fallback chain: {[p.value for p in chain]}")

    def _get_adapter(self, provider: LLMProvider, model: str | None = None) -> LLMAdapter:
        """Get or create an adapter for a provider."""
        if provider in self._adapters:
            return self._adapters[provider]

        # Get default model or use first available from provider
        model = model or DEFAULT_MODELS.get(provider, "default")

        # For Ollama, use first available model if default isn't available
        if provider == LLMProvider.OLLAMA and provider in self._providers:
            available_models = self._providers[provider].models
            if available_models and model not in available_models:
                model = available_models[0]
                logger.info(f"Using available Ollama model: {model}")

        config = LLMConfig(
            provider=provider,
            model=model,
        )

        adapter: LLMAdapter
        if provider == LLMProvider.OLLAMA:
            adapter = OllamaAdapter(config)
        elif provider == LLMProvider.ANTHROPIC:
            adapter = AnthropicAdapter(config)
        elif provider == LLMProvider.OPENAI:
            adapter = OpenAIAdapter(config)
        elif provider == LLMProvider.GROQ:
            adapter = GroqAdapter(config)
        else:
            raise LLMAdapterError(f"Unknown provider: {provider}")

        self._adapters[provider] = adapter
        return adapter

    def _select_provider(
        self,
        requires_tools: bool = False,
        preferred_model: str | None = None,
        classification: DataClassification | None = None,
    ) -> tuple[LLMProvider, bool]:
        """
        Select the best provider for a request.

        Args:
            requires_tools: Whether the request needs tool calling support
            preferred_model: Specific model preference
            classification: Data sensitivity classification for routing

        Returns:
            Tuple of (provider, supports_native_tools)
            If tools are required but no provider supports them,
            returns a provider that can simulate tools via prompts.
        """
        # Filter providers by classification if specified
        allowed = None
        if classification:
            allowed = CLASSIFICATION_ALLOWED_PROVIDERS.get(classification)
            if allowed:
                logger.info(
                    "Classification=%s allows providers: %s",
                    classification.value,
                    [p.value for p in allowed],
                )

        chain = self._fallback_chain
        if allowed:
            chain = [p for p in chain if p in allowed]
            if not chain:
                classification_value = classification.value if classification else "unknown"
                raise LLMAdapterError(
                    f"No available provider for classification={classification_value}. "
                    f"Allowed: {[p.value for p in allowed]}. "
                    f"Available: {[p.value for p in self._fallback_chain]}"
                )

        # First, try to find a provider that supports tools natively
        if requires_tools:
            for provider in chain:
                info = self._providers[provider]
                if info.supports_tools:
                    return provider, True

        # Fall back to any available provider
        for provider in chain:
            return provider, self._providers[provider].supports_tools

        raise LLMAdapterError("No suitable provider available")

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        system: str | None = None,
        tools: list[Tool] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        provider: LLMProvider | None = None,
        model: str | None = None,
        classification: DataClassification | None = None,
    ) -> LLMResponse:
        """
        Route a completion request.

        Args:
            messages: Conversation messages
            system: System prompt
            tools: Available tools
            temperature: Sampling temperature
            max_tokens: Max output tokens
            provider: Force specific provider
            model: Force specific model
            classification: Data sensitivity classification for routing

        Returns:
            LLM response
        """
        await self._ensure_initialized()

        # Validate explicit provider against classification
        if provider and classification:
            allowed = CLASSIFICATION_ALLOWED_PROVIDERS.get(classification, set())
            if provider not in allowed:
                raise LLMAdapterError(
                    f"Provider {provider.value} not allowed for "
                    f"classification={classification.value}"
                )

        # Select provider
        supports_native_tools = True
        if provider:
            if not self._providers[provider].available:
                raise LLMAdapterError(f"Provider {provider.value} not available")
            supports_native_tools = self._providers[provider].supports_tools
        else:
            provider, supports_native_tools = self._select_provider(
                requires_tools=bool(tools),
                classification=classification,
            )

        # If tools are requested but provider doesn't support them natively,
        # simulate tools via prompt injection
        actual_tools = tools
        actual_system = system
        if tools and not supports_native_tools:
            logger.info(
                "Provider %s doesn't support tools natively, using prompt-based tools",
                provider.value,
            )
            actual_tools = None
            actual_system = self._build_tool_prompt(system, tools)

        adapter = self._get_adapter(provider, model)

        # Try with fallback
        last_error: Exception | None = None
        for fallback_provider in self._fallback_chain:
            if provider and fallback_provider != provider:
                continue

            try:
                adapter = self._get_adapter(fallback_provider, model)
                response = await adapter.complete(
                    messages,
                    system=actual_system,
                    tools=actual_tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                # If using simulated tools, parse tool calls from response
                if tools and not supports_native_tools:
                    response = self._parse_simulated_tools(response)

                # Audit log for regulated data
                if classification == DataClassification.REGULATED:
                    logger.info(
                        "AUDIT: regulated request completed via %s "
                        "(model=%s, tokens=%d)",
                        fallback_provider.value,
                        response.model,
                        response.total_tokens,
                    )

                return response
            except Exception as e:
                last_error = e
                logger.warning(f"Provider {fallback_provider.value} failed: {e}")
                if provider:
                    break  # Don't fallback if specific provider requested

        raise LLMAdapterError(
            f"All providers failed: {last_error}",
            provider=provider.value if provider else None,
        )

    def _build_tool_prompt(self, system: str | None, tools: list[Tool]) -> str:
        """Build a system prompt that includes tool definitions for non-native tool support."""
        tool_descriptions = []
        for i, tool in enumerate(tools, 1):
            params = tool.parameters
            if isinstance(params, dict) and "properties" in params:
                props = params["properties"]
                param_parts = [f"{name}: {info.get('type', 'any')}" for name, info in props.items()]
                param_str = ", ".join(param_parts) if param_parts else "no parameters"
            else:
                param_str = str(params)
            tool_descriptions.append(f"{i}. {tool.name}({param_str}): {tool.description}")

        tools_section = f"""
You have access to the following tools:

{chr(10).join(tool_descriptions)}

To use a tool, respond with:
<tool_call>
{{"name": "tool_name", "arguments": {{"arg1": "value1"}}}}
</tool_call>

After the tool result is provided, continue your response.
If you don't need to use a tool, just respond normally.
"""
        base_system = system or "You are a helpful assistant."
        return f"{base_system}\n\n{tools_section}"

    def _parse_simulated_tools(self, response: LLMResponse) -> LLMResponse:
        """Parse tool calls from text response for non-native tool support."""
        import json
        import re

        content = response.content
        tool_calls = []

        # Find all <tool_call>...</tool_call> blocks
        pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
        matches = re.findall(pattern, content, re.DOTALL)

        for i, match in enumerate(matches):
            try:
                call_data = json.loads(match)
                tool_calls.append(
                    {
                        "id": f"sim_call_{i}",
                        "name": call_data.get("name", "unknown"),
                        "arguments": call_data.get("arguments", {}),
                    }
                )
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse tool call: {match[:100]}")

        # Remove tool call blocks from content
        clean_content = re.sub(pattern, "", content, flags=re.DOTALL).strip()

        return LLMResponse(
            content=clean_content,
            model=response.model,
            provider=response.provider,
            finish_reason=response.finish_reason,
            usage=response.usage,
            tool_calls=tool_calls,
            raw_response=response.raw_response,
        )

    async def stream(
        self,
        messages: list[LLMMessage],
        *,
        system: str | None = None,
        tools: list[Tool] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        provider: LLMProvider | None = None,
        model: str | None = None,
        classification: DataClassification | None = None,
    ) -> AsyncIterator[str]:
        """
        Route a streaming request.

        Args:
            messages: Conversation messages
            system: System prompt
            tools: Available tools
            temperature: Sampling temperature
            max_tokens: Max output tokens
            provider: Force specific provider
            model: Force specific model
            classification: Data sensitivity classification for routing

        Yields:
            Response tokens
        """
        await self._ensure_initialized()

        # Validate explicit provider against classification
        if provider and classification:
            allowed = CLASSIFICATION_ALLOWED_PROVIDERS.get(classification, set())
            if provider not in allowed:
                raise LLMAdapterError(
                    f"Provider {provider.value} not allowed for "
                    f"classification={classification.value}"
                )

        supports_native_tools = True
        if provider:
            if not self._providers[provider].available:
                raise LLMAdapterError(f"Provider {provider.value} not available")
            supports_native_tools = self._providers[provider].supports_tools
        else:
            provider, supports_native_tools = self._select_provider(
                requires_tools=bool(tools),
                classification=classification,
            )

        # Audit log for regulated streaming
        if classification == DataClassification.REGULATED:
            logger.info(
                "AUDIT: regulated stream request via %s (model=%s)",
                provider.value,
                model or DEFAULT_MODELS.get(provider, "default"),
            )

        # If tools are requested but provider doesn't support them natively,
        # simulate tools via prompt injection
        actual_tools = tools
        actual_system = system
        if tools and not supports_native_tools:
            actual_tools = None
            actual_system = self._build_tool_prompt(system, tools)

        adapter = self._get_adapter(provider, model)

        async for token in adapter.stream(
            messages,
            system=actual_system,
            tools=actual_tools,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            yield token

    async def embed(
        self,
        text: str,
        provider: LLMProvider | None = None,
    ) -> list[float]:
        """
        Generate embeddings.

        Args:
            text: Text to embed
            provider: Force specific provider

        Returns:
            Embedding vector
        """
        await self._ensure_initialized()

        # Prefer OpenAI for embeddings if available
        if not provider:
            if self._providers.get(
                LLMProvider.OPENAI,
                ProviderInfo(
                    provider=LLMProvider.OPENAI,
                    available=False,
                    models=[],
                    supports_tools=False,
                    supports_streaming=False,
                    cost_tier=3,
                    quality_tier=3,
                    speed_tier=3,
                ),
            ).available:
                provider = LLMProvider.OPENAI
            else:
                provider = self._fallback_chain[0] if self._fallback_chain else LLMProvider.OLLAMA

        adapter = self._get_adapter(provider)
        return await adapter.embed(text)

    def list_providers(self) -> list[ProviderInfo]:
        """List all providers and their status."""
        return list(self._providers.values())

    def list_available_providers(self) -> list[LLMProvider]:
        """List available providers."""
        return [p for p, i in self._providers.items() if i.available]

    async def health_check(self) -> dict[str, bool]:
        """Check health of all providers."""
        await self._ensure_initialized()
        health = {}
        for provider, info in self._providers.items():
            if info.available:
                adapter = self._get_adapter(provider)
                health[provider.value] = await adapter.health_check()
            else:
                health[provider.value] = False
        return health

    async def _ensure_initialized(self) -> None:
        """Ensure router is initialized."""
        if not self._initialized:
            await self.initialize()

    def __repr__(self) -> str:
        available = [p.value for p in self.list_available_providers()]
        return f"<LLMRouter strategy={self.strategy.value} available={available}>"


# Singleton router
_default_router: LLMRouter | None = None


def get_router(
    strategy: RoutingStrategy = RoutingStrategy.FALLBACK,
) -> LLMRouter:
    """Get the default LLM router."""
    global _default_router
    if _default_router is None:
        _default_router = LLMRouter(strategy=strategy)
    return _default_router


async def reset_router() -> None:
    """Reset the default router."""
    global _default_router
    _default_router = None
