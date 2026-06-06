# F-67: Cross-Model Interaction Replay/Diff

> Replay the same prompt across multiple models and diff outputs for consistency, bias detection, and quality comparison.

## Implementation Status

Implemented CLI surface:

```bash
titan replay capture --prompt "Review this code for security issues"
titan replay run --id rec_... --targets anthropic:claude-sonnet-4-20250514,openai:gpt-4o
titan replay diff --id rec_... --format markdown
```

Replay records and run results are stored as JSON under `.titan/replays/` by
default. The diff command renders pairwise agreement, length ratio, shared
terms, target errors, and side-by-side output previews.

## Use Cases

| Use Case | Description |
|----------|-------------|
| Consistency verification | Ensure critical prompts produce similar outputs across models |
| Bias detection | Identify systematic differences in reasoning or recommendations |
| Quality comparison | Compare local (Ollama) vs cloud (Anthropic/OpenAI) output quality |
| Regression testing | Detect quality changes after model updates |
| Cost optimization | Verify cheaper models produce acceptable output for specific tasks |

## Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────┐
│  Replay      │────▶│  Multi-Target    │────▶│  Diff        │
│  Engine      │     │  Dispatcher      │     │  Generator   │
│              │     │                  │     │              │
│  - Prompt    │     │  - Model A       │     │  - Side-by-  │
│    storage   │     │  - Model B       │     │    side view │
│  - Replay    │     │  - Model C       │     │  - Metrics   │
│    schedule  │     │  - Model N       │     │  - Report    │
└──────────────┘     └──────────────────┘     └──────────────┘
```

### Replay Engine

Stores prompts with full context (system prompt, conversation history, tool definitions) and replays them on demand or on schedule.

```python
class ReplayEngine:
    def capture(self, prompt: Prompt, context: Context) -> ReplayRecord:
        """Store a prompt for future replay."""
        ...

    def replay(self, record_id: str, targets: list[ModelTarget]) -> list[ModelOutput]:
        """Send the same prompt to multiple models."""
        ...

    def schedule(self, record_id: str, targets: list[ModelTarget], cron: str) -> None:
        """Schedule periodic replay for regression detection."""
        ...
```

### Multi-Target Dispatcher

Uses titan's existing adapter layer to dispatch the same prompt to multiple models concurrently.

```python
# Leverages adapters/router.py
targets = [
    ModelTarget(provider="anthropic", model="claude-sonnet-4-20250514"),
    ModelTarget(provider="openai", model="gpt-4o"),
    ModelTarget(provider="ollama", model="llama3.1:70b"),
    ModelTarget(provider="groq", model="llama-3.1-70b-versatile"),
]
```

### Diff Generator

Produces structured comparison of outputs.

## Diff Format

### Side-by-Side Comparison

```
┌─ claude-sonnet-4-20250514 ──────────┬─ gpt-4o ────────────────────────┐
│ The function should validate input │ Input validation is required    │
│ before processing. Use a guard     │ at the entry point. A type      │
│ clause pattern:                    │ check should be performed:      │
│                                    │                                 │
│ ```python                          │ ```python                       │
│ if not isinstance(x, int):         │ if type(x) != int:              │
│     raise TypeError(...)           │     raise ValueError(...)       │
│ ```                                │ ```                             │
├─ DIFF ─────────────────────────────┴─────────────────────────────────┤
│ - Both recommend input validation (agreement)                        │
│ - isinstance vs type(): claude uses idiomatic Python (quality diff)  │
│ - TypeError vs ValueError: different exception choice (semantic diff)│
└──────────────────────────────────────────────────────────────────────┘
```

### Highlighted Differences

Differences are categorized:

| Category | Description | Severity |
|----------|-------------|----------|
| Agreement | Same recommendation, similar phrasing | None |
| Phrasing diff | Same meaning, different words | Low |
| Quality diff | One answer is more idiomatic/correct | Medium |
| Semantic diff | Different recommendations or conclusions | High |
| Factual diff | Contradictory factual claims | Critical |

## Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| Agreement rate | agreed_points / total_points | How often models agree |
| Response length ratio | len(output_a) / len(output_b) | Verbosity comparison |
| Factual consistency score | consistent_facts / total_facts | Cross-model fact agreement |
| Code equivalence | test_pass_rate across implementations | Functional equivalence of generated code |
| Latency ratio | time_a / time_b | Speed comparison |

## Integration

### Adapter Layer

The replay system uses titan's existing adapter layer for multi-target dispatch:

```python
# adapters/replay_adapter.py
from adapters.base import BaseAdapter

class ReplayAdapter:
    """Wraps multiple adapters for concurrent dispatch."""

    def __init__(self, adapters: dict[str, BaseAdapter]):
        self.adapters = adapters

    async def dispatch(self, prompt: str, targets: list[str]) -> dict[str, str]:
        results = await asyncio.gather(*[
            self.adapters[t].generate(prompt) for t in targets
        ])
        return dict(zip(targets, results))
```

### CLI Interface

```bash
# Capture a prompt for replay
titan replay capture --prompt "Review this code for security issues" --context ./context.json

# Replay across models
titan replay run --id rec_001 --targets anthropic,openai,ollama

# View diff
titan replay diff --id rec_001 --format side-by-side

# Schedule regression replay
titan replay schedule --id rec_001 --targets anthropic,openai --cron "0 0 * * MON"
```

## Reference

- `adapters/router.py` — Model routing logic
- `adapters/base.py` — Base adapter interface
- F-34 (Multi-model review) — Related multi-model quality assurance pattern
