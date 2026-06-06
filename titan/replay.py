"""Cross-model prompt replay and diff utilities."""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast
from uuid import uuid4

from adapters.base import LLMMessage, LLMProvider

DEFAULT_REPLAY_ROOT = Path(".titan") / "replays"
TOKEN_RE = re.compile(r"\b[a-zA-Z0-9_]{3,}\b")


class ReplayError(ValueError):
    """Raised for invalid replay storage or target requests."""


class CompletionRouter(Protocol):
    """Router surface used by replay dispatch."""

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        provider: LLMProvider | None = None,
        model: str | None = None,
    ) -> Any:
        """Complete a prompt for one provider/model target."""


@dataclass(frozen=True)
class ModelTarget:
    """A provider/model replay target."""

    provider: LLMProvider
    model: str | None = None

    @property
    def label(self) -> str:
        """Human-readable target label."""
        return f"{self.provider.value}:{self.model}" if self.model else self.provider.value


@dataclass(frozen=True)
class ReplayRecord:
    """Captured prompt and context for future replay."""

    id: str
    prompt: str
    system: str | None
    context: str | None
    created_at: str


@dataclass(frozen=True)
class ReplayOutput:
    """One model target output from a replay run."""

    target: str
    provider: str
    model: str | None
    content: str
    latency_ms: float
    usage: dict[str, int]
    error: str | None = None


@dataclass(frozen=True)
class ReplayRun:
    """Stored replay run results."""

    id: str
    record_id: str
    created_at: str
    outputs: list[ReplayOutput]


def utc_now() -> str:
    """Return a stable UTC timestamp for replay artifacts."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def make_id(prefix: str) -> str:
    """Create a compact replay artifact id."""
    stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    return f"{prefix}_{stamp}_{uuid4().hex[:8]}"


def parse_model_targets(raw_targets: str) -> list[ModelTarget]:
    """Parse comma-separated replay targets.

    Target forms:
    - `openai`
    - `openai:gpt-4o-mini`
    - `ollama/llama3.2`
    """
    targets: list[ModelTarget] = []
    for raw_target in raw_targets.split(","):
        target = raw_target.strip()
        if not target:
            continue
        separator = ":" if ":" in target else "/" if "/" in target else ""
        model: str | None
        if separator:
            provider_name, raw_model = target.split(separator, 1)
            model = raw_model.strip() or None
        else:
            provider_name, model = target, None
        try:
            provider = LLMProvider(provider_name.strip().lower())
        except ValueError as exc:
            valid = ", ".join(provider.value for provider in LLMProvider)
            message = f"Unknown provider '{provider_name}'. Valid providers: {valid}"
            raise ReplayError(message) from exc
        targets.append(ModelTarget(provider=provider, model=model))
    return targets


class ReplayStore:
    """JSON-backed replay record and result storage."""

    def __init__(self, root: Path = DEFAULT_REPLAY_ROOT) -> None:
        self.root = root
        self.records_dir = root / "records"
        self.runs_dir = root / "runs"

    def capture(
        self,
        *,
        prompt: str,
        system: str | None = None,
        context: str | None = None,
    ) -> ReplayRecord:
        """Capture and persist a replay record."""
        record = ReplayRecord(
            id=make_id("rec"),
            prompt=prompt,
            system=system,
            context=context,
            created_at=utc_now(),
        )
        self.records_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(self.record_path(record.id), asdict(record))
        return record

    def record_path(self, record_id: str) -> Path:
        """Return the path for a replay record."""
        return self.records_dir / f"{record_id}.json"

    def run_path(self, record_id: str, run_id: str) -> Path:
        """Return the path for a replay run."""
        return self.runs_dir / record_id / f"{run_id}.json"

    def load_record(self, record_id: str) -> ReplayRecord:
        """Load a captured replay record."""
        path = self.record_path(record_id)
        data = self._read_json(path)
        return ReplayRecord(**data)

    def save_run(self, run: ReplayRun) -> None:
        """Persist replay run results."""
        path = self.run_path(run.record_id, run.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._write_json(path, asdict(run))

    def load_run(self, record_id: str, run_id: str) -> ReplayRun:
        """Load a replay run."""
        data = self._read_json(self.run_path(record_id, run_id))
        outputs = [ReplayOutput(**output) for output in data["outputs"]]
        return ReplayRun(
            id=data["id"],
            record_id=data["record_id"],
            created_at=data["created_at"],
            outputs=outputs,
        )

    def latest_run(self, record_id: str) -> ReplayRun:
        """Load the most recent replay run for a record."""
        run_dir = self.runs_dir / record_id
        if not run_dir.is_dir():
            raise ReplayError(f"No runs found for replay record {record_id}")
        paths = sorted(run_dir.glob("run_*.json"), key=lambda path: path.stat().st_mtime)
        if not paths:
            raise ReplayError(f"No runs found for replay record {record_id}")
        return self.load_run(record_id, paths[-1].stem)

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise ReplayError(f"Replay artifact not found: {path}")
        return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def build_messages(record: ReplayRecord) -> list[LLMMessage]:
    """Build replay messages from a captured record."""
    content = record.prompt
    if record.context:
        content = f"{content}\n\nContext:\n{record.context}"
    return [LLMMessage(role="user", content=content)]


async def dispatch_replay(
    record: ReplayRecord,
    targets: list[ModelTarget],
    router: CompletionRouter,
    *,
    max_tokens: int = 1024,
    temperature: float | None = None,
) -> ReplayRun:
    """Replay a record across all requested targets."""
    if len(targets) < 2:
        raise ReplayError("Replay run requires at least two targets")

    async def complete_target(target: ModelTarget) -> ReplayOutput:
        start = time.perf_counter()
        try:
            response = await router.complete(
                build_messages(record),
                system=record.system,
                temperature=temperature,
                max_tokens=max_tokens,
                provider=target.provider,
                model=target.model,
            )
            latency_ms = (time.perf_counter() - start) * 1000
            content = str(getattr(response, "content", ""))
            model = getattr(response, "model", target.model)
            provider = str(getattr(response, "provider", target.provider.value))
            usage = getattr(response, "usage", {})
            return ReplayOutput(
                target=target.label,
                provider=provider,
                model=str(model) if model is not None else None,
                content=content,
                latency_ms=round(latency_ms, 3),
                usage=dict(usage) if isinstance(usage, dict) else {},
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            return ReplayOutput(
                target=target.label,
                provider=target.provider.value,
                model=target.model,
                content="",
                latency_ms=round(latency_ms, 3),
                usage={},
                error=str(exc),
            )

    outputs = await asyncio.gather(*(complete_target(target) for target in targets))
    return ReplayRun(
        id=make_id("run"),
        record_id=record.id,
        created_at=utc_now(),
        outputs=list(outputs),
    )


def tokenize(text: str) -> set[str]:
    """Tokenize response text for rough agreement scoring."""
    return {token.lower() for token in TOKEN_RE.findall(text)}


def comparison_category(agreement_rate: float) -> str:
    """Classify a pairwise comparison from token overlap."""
    if agreement_rate >= 0.75:
        return "agreement"
    if agreement_rate >= 0.45:
        return "phrasing_diff"
    if agreement_rate >= 0.2:
        return "semantic_diff"
    return "factual_diff"


def build_diff(run: ReplayRun) -> dict[str, Any]:
    """Build a structured pairwise diff for a replay run."""
    successful = [output for output in run.outputs if output.error is None]
    comparisons: list[dict[str, Any]] = []
    for left_index, left in enumerate(successful):
        for right in successful[left_index + 1 :]:
            left_tokens = tokenize(left.content)
            right_tokens = tokenize(right.content)
            union = left_tokens | right_tokens
            shared = left_tokens & right_tokens
            agreement_rate = len(shared) / len(union) if union else 1.0
            right_length = max(len(right.content), 1)
            comparisons.append(
                {
                    "left": left.target,
                    "right": right.target,
                    "agreement_rate": round(agreement_rate, 3),
                    "length_ratio": round(len(left.content) / right_length, 3),
                    "shared_terms": sorted(shared)[:25],
                    "category": comparison_category(agreement_rate),
                }
            )

    return {
        "run_id": run.id,
        "record_id": run.record_id,
        "created_at": run.created_at,
        "targets": [output.target for output in run.outputs],
        "errors": [
            {"target": output.target, "error": output.error}
            for output in run.outputs
            if output.error
        ],
        "comparisons": comparisons,
    }


def render_diff_markdown(run: ReplayRun, diff: dict[str, Any]) -> str:
    """Render replay results and diff as Markdown."""
    lines = [
        f"# Replay Diff: {run.record_id}",
        "",
        f"- Run: `{run.id}`",
        f"- Created: `{run.created_at}`",
        "",
        "## Outputs",
    ]
    for output in run.outputs:
        status = "error" if output.error else "ok"
        lines.extend(
            [
                "",
                f"### {output.target}",
                "",
                f"- Status: `{status}`",
                f"- Model: `{output.model or 'unknown'}`",
                f"- Latency: `{output.latency_ms:.3f} ms`",
            ]
        )
        if output.error:
            lines.append(f"- Error: `{output.error}`")
        else:
            preview = output.content.strip() or "(empty response)"
            lines.extend(["", "```text", preview, "```"])

    lines.extend(["", "## Pairwise Comparisons", ""])
    comparisons = diff["comparisons"]
    if not comparisons:
        lines.append("No successful output pairs were available for comparison.")
    else:
        lines.append("| Left | Right | Category | Agreement | Length Ratio |")
        lines.append("| --- | --- | --- | ---: | ---: |")
        for comparison in comparisons:
            lines.append(
                "| {left} | {right} | {category} | {agreement_rate:.3f} | "
                "{length_ratio:.3f} |".format(**comparison)
            )

    if diff["errors"]:
        lines.extend(["", "## Errors"])
        for error in diff["errors"]:
            lines.append(f"- `{error['target']}`: {error['error']}")

    return "\n".join(lines) + "\n"
