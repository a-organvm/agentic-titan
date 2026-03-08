"""Tests for LLM adapters — context window config (F-23) and sensitivity routing (F-24)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from adapters.base import LLMConfig, LLMMessage, LLMProvider, LLMResponse, OllamaAdapter
from adapters.router import (
    CLASSIFICATION_ALLOWED_PROVIDERS,
    DataClassification,
    LLMRouter,
    RoutingStrategy,
)
from agents.framework.errors import LLMAdapterError

# ── F-23: Context window configuration ──


class TestContextWindowConfig:
    """Verify num_ctx is passed to Ollama API calls."""

    def test_default_context_window(self):
        config = LLMConfig(provider=LLMProvider.OLLAMA, model="llama3.2:3b")
        assert config.context_window == 8192

    def test_custom_context_window(self):
        config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model="llama3.2:3b",
            context_window=32768,
        )
        assert config.context_window == 32768

    @pytest.mark.asyncio
    async def test_ollama_complete_passes_num_ctx(self):
        config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model="llama3.2:3b",
            context_window=16384,
        )
        adapter = OllamaAdapter(config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "test response"},
            "prompt_eval_count": 10,
            "eval_count": 5,
        }

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            messages = [LLMMessage(role="user", content="Hello")]
            await adapter.complete(messages)

            # Verify num_ctx was passed in options
            call_args = mock_client.post.call_args
            request_body = call_args.kwargs.get("json") or call_args[1].get("json")
            assert request_body["options"]["num_ctx"] == 16384

    @pytest.mark.asyncio
    async def test_ollama_stream_passes_num_ctx(self):
        config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model="llama3.2:3b",
            context_window=32768,
        )
        adapter = OllamaAdapter(config)

        mock_response = AsyncMock()
        mock_response.aiter_lines = AsyncMock(return_value=AsyncMock())

        async def mock_lines():
            yield '{"message": {"content": "hi"}, "done": false}'
            yield '{"message": {"content": ""}, "done": true}'

        mock_response.aiter_lines = mock_lines

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_stream_ctx = AsyncMock()
            mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
            mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_client.stream = MagicMock(return_value=mock_stream_ctx)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            messages = [LLMMessage(role="user", content="Hello")]
            tokens = []
            async for token in adapter.stream(messages):
                tokens.append(token)

            # Verify num_ctx was passed in options
            call_args = mock_client.stream.call_args
            request_body = call_args.kwargs.get("json") or call_args[1]["json"]
            assert request_body["options"]["num_ctx"] == 32768


# ── F-24: Sensitivity-based model routing ──


class TestDataClassification:
    """Verify data classification enum and allowed providers."""

    def test_classification_values(self):
        assert DataClassification.PUBLIC == "public"
        assert DataClassification.INTERNAL == "internal"
        assert DataClassification.CONFIDENTIAL == "confidential"
        assert DataClassification.REGULATED == "regulated"

    def test_public_allows_all_providers(self):
        allowed = CLASSIFICATION_ALLOWED_PROVIDERS[DataClassification.PUBLIC]
        assert LLMProvider.ANTHROPIC in allowed
        assert LLMProvider.OPENAI in allowed
        assert LLMProvider.OLLAMA in allowed
        assert LLMProvider.GROQ in allowed

    def test_confidential_allows_only_local(self):
        allowed = CLASSIFICATION_ALLOWED_PROVIDERS[DataClassification.CONFIDENTIAL]
        assert LLMProvider.OLLAMA in allowed
        assert LLMProvider.LOCAL in allowed
        assert LLMProvider.ANTHROPIC not in allowed
        assert LLMProvider.OPENAI not in allowed
        assert LLMProvider.GROQ not in allowed

    def test_regulated_allows_only_local(self):
        allowed = CLASSIFICATION_ALLOWED_PROVIDERS[DataClassification.REGULATED]
        assert LLMProvider.OLLAMA in allowed
        assert LLMProvider.LOCAL in allowed
        assert LLMProvider.ANTHROPIC not in allowed


class TestSensitivityRouting:
    """Verify router enforces classification constraints."""

    @pytest.mark.asyncio
    async def test_confidential_blocks_cloud_provider(self):
        router = LLMRouter(strategy=RoutingStrategy.FALLBACK)
        # Manually set up providers
        router._initialized = True
        router._providers = {
            LLMProvider.ANTHROPIC: MagicMock(available=True, supports_tools=True),
            LLMProvider.OPENAI: MagicMock(available=True, supports_tools=True),
        }
        router._fallback_chain = [LLMProvider.ANTHROPIC, LLMProvider.OPENAI]

        # Should raise because no local provider is available
        with pytest.raises(LLMAdapterError, match="No available provider"):
            router._select_provider(classification=DataClassification.CONFIDENTIAL)

    @pytest.mark.asyncio
    async def test_confidential_allows_ollama(self):
        router = LLMRouter(strategy=RoutingStrategy.FALLBACK)
        router._initialized = True
        router._providers = {
            LLMProvider.ANTHROPIC: MagicMock(
                available=True, supports_tools=True, cost_tier=3
            ),
            LLMProvider.OLLAMA: MagicMock(
                available=True, supports_tools=False, cost_tier=1
            ),
        }
        router._fallback_chain = [LLMProvider.ANTHROPIC, LLMProvider.OLLAMA]

        provider, _ = router._select_provider(
            classification=DataClassification.CONFIDENTIAL,
        )
        assert provider == LLMProvider.OLLAMA

    @pytest.mark.asyncio
    async def test_public_allows_any_provider(self):
        router = LLMRouter(strategy=RoutingStrategy.FALLBACK)
        router._initialized = True
        router._providers = {
            LLMProvider.ANTHROPIC: MagicMock(
                available=True, supports_tools=True, cost_tier=3
            ),
            LLMProvider.OLLAMA: MagicMock(
                available=True, supports_tools=False, cost_tier=1
            ),
        }
        router._fallback_chain = [LLMProvider.ANTHROPIC, LLMProvider.OLLAMA]

        provider, _ = router._select_provider(
            classification=DataClassification.PUBLIC,
        )
        # Should pick first in chain (Anthropic)
        assert provider == LLMProvider.ANTHROPIC

    @pytest.mark.asyncio
    async def test_explicit_provider_blocked_by_classification(self):
        router = LLMRouter(strategy=RoutingStrategy.FALLBACK)
        router._initialized = True
        router._providers = {
            LLMProvider.ANTHROPIC: MagicMock(available=True, supports_tools=True),
        }
        router._fallback_chain = [LLMProvider.ANTHROPIC]

        with pytest.raises(LLMAdapterError, match="not allowed"):
            await router.complete(
                [LLMMessage(role="user", content="secret data")],
                provider=LLMProvider.ANTHROPIC,
                classification=DataClassification.CONFIDENTIAL,
            )

    @pytest.mark.asyncio
    async def test_regulated_logs_audit(self, caplog):
        """Verify AUDIT log entry for regulated requests."""
        router = LLMRouter(strategy=RoutingStrategy.FALLBACK)
        router._initialized = True

        mock_adapter = AsyncMock()
        mock_adapter.complete = AsyncMock(
            return_value=LLMResponse(
                content="ok",
                model="llama3.2:3b",
                provider="ollama",
                usage={"total_tokens": 42},
            )
        )

        router._providers = {
            LLMProvider.OLLAMA: MagicMock(
                available=True, supports_tools=False, cost_tier=1
            ),
        }
        router._fallback_chain = [LLMProvider.OLLAMA]
        router._adapters = {LLMProvider.OLLAMA: mock_adapter}

        import logging

        with caplog.at_level(logging.INFO, logger="titan.adapters.router"):
            await router.complete(
                [LLMMessage(role="user", content="regulated data")],
                classification=DataClassification.REGULATED,
            )

        assert any("AUDIT" in record.message for record in caplog.records)
