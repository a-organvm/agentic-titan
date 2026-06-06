"""Tests for cross-model replay storage, dispatch, and diff."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from adapters.base import LLMMessage, LLMProvider, LLMResponse
from titan.replay import (
    ModelTarget,
    ReplayError,
    ReplayStore,
    build_diff,
    dispatch_replay,
    parse_model_targets,
    render_diff_markdown,
)


class FakeRouter:
    """Deterministic replay router for tests."""

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        provider: LLMProvider | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        del system, temperature, max_tokens
        assert provider is not None
        prompt = messages[-1].content
        return LLMResponse(
            content=f"{provider.value} recommends input validation for {prompt[:20]}",
            model=model or f"{provider.value}-default",
            provider=provider.value,
            usage={"total_tokens": 12},
        )


def test_parse_model_targets_supports_provider_and_model() -> None:
    targets = parse_model_targets("anthropic:claude, openai/gpt-4o, ollama")

    assert targets == [
        ModelTarget(LLMProvider.ANTHROPIC, "claude"),
        ModelTarget(LLMProvider.OPENAI, "gpt-4o"),
        ModelTarget(LLMProvider.OLLAMA, None),
    ]


def test_parse_model_targets_rejects_unknown_provider() -> None:
    with pytest.raises(ReplayError, match="Unknown provider"):
        parse_model_targets("not-a-provider")


@pytest.mark.asyncio
async def test_replay_store_dispatch_and_diff(tmp_path: Path) -> None:
    store = ReplayStore(tmp_path / "replays")
    record = store.capture(
        prompt="Review this function",
        system="Be concise",
        context="def add(a, b): return a + b",
    )

    run = await dispatch_replay(
        record,
        parse_model_targets("anthropic:claude-test,openai:gpt-test"),
        FakeRouter(),
        max_tokens=128,
    )
    store.save_run(run)

    loaded_record = store.load_record(record.id)
    loaded_run = store.load_run(record.id, run.id)
    latest_run = store.latest_run(record.id)
    diff = build_diff(loaded_run)
    markdown = render_diff_markdown(loaded_run, diff)

    assert loaded_record.context == "def add(a, b): return a + b"
    assert latest_run.id == run.id
    assert [output.target for output in loaded_run.outputs] == [
        "anthropic:claude-test",
        "openai:gpt-test",
    ]
    assert loaded_run.outputs[0].usage == {"total_tokens": 12}
    assert diff["comparisons"][0]["left"] == "anthropic:claude-test"
    assert diff["comparisons"][0]["right"] == "openai:gpt-test"
    assert "Pairwise Comparisons" in markdown


@pytest.mark.asyncio
async def test_dispatch_requires_two_targets(tmp_path: Path) -> None:
    store = ReplayStore(tmp_path / "replays")
    record = store.capture(prompt="Hello")

    with pytest.raises(ReplayError, match="at least two targets"):
        await dispatch_replay(
            record,
            [ModelTarget(LLMProvider.OPENAI)],
            FakeRouter(),
        )


class FailingRouter:
    """Router that fails one target and succeeds another."""

    async def complete(
        self,
        messages: list[LLMMessage],
        **kwargs: Any,
    ) -> LLMResponse:
        del messages
        provider = kwargs["provider"]
        assert isinstance(provider, LLMProvider)
        if provider == LLMProvider.OPENAI:
            raise RuntimeError("provider unavailable")
        return LLMResponse(content="local answer", model="llama", provider=provider.value)


@pytest.mark.asyncio
async def test_dispatch_stores_target_errors(tmp_path: Path) -> None:
    store = ReplayStore(tmp_path / "replays")
    record = store.capture(prompt="Hello")

    run = await dispatch_replay(
        record,
        parse_model_targets("openai,ollama"),
        FailingRouter(),
    )

    errors = [output for output in run.outputs if output.error]
    assert len(errors) == 1
    assert errors[0].target == "openai"
    assert build_diff(run)["errors"] == [
        {"target": "openai", "error": "provider unavailable"}
    ]
