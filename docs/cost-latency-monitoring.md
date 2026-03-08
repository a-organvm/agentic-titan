# F-31: Cost and Latency Monitoring

> Track per-interaction cost, latency, and output quality for all LLM API usage.

## Overview

Every LLM interaction in agentic-titan should be instrumented with three measurements: what it cost, how long it took, and whether the output was useful. This data enables budget management, provider comparison, and quality optimization.

## Per-Interaction Metrics

### Cost Tracking

Each LLM call records:

| Field | Type | Source |
|-------|------|--------|
| `provider` | string | Adapter config |
| `model` | string | Request params |
| `input_tokens` | int | API response usage |
| `output_tokens` | int | API response usage |
| `cache_read_tokens` | int | API response (Anthropic) |
| `cache_write_tokens` | int | API response (Anthropic) |
| `cost_usd` | float | Computed from token counts × price table |

### Price Table

```python
# adapters/pricing.py
PRICING = {
    "claude-sonnet-4-20250514": {
        "input_per_1k": 0.003,
        "output_per_1k": 0.015,
        "cache_read_per_1k": 0.0003,
        "cache_write_per_1k": 0.00375,
    },
    "claude-opus-4-20250514": {
        "input_per_1k": 0.015,
        "output_per_1k": 0.075,
        "cache_read_per_1k": 0.0015,
        "cache_write_per_1k": 0.01875,
    },
    "gpt-4o": {
        "input_per_1k": 0.0025,
        "output_per_1k": 0.01,
    },
    "ollama/*": {
        "input_per_1k": 0.0,
        "output_per_1k": 0.0,  # Local inference — electricity cost only
    },
}

def compute_cost(model: str, usage: TokenUsage) -> float:
    prices = PRICING.get(model, PRICING.get("default"))
    return (
        usage.input_tokens * prices["input_per_1k"] / 1000
        + usage.output_tokens * prices["output_per_1k"] / 1000
        + usage.cache_read_tokens * prices.get("cache_read_per_1k", 0) / 1000
        + usage.cache_write_tokens * prices.get("cache_write_per_1k", 0) / 1000
    )
```

### Latency Tracking

Each LLM call records:

| Field | Type | Description |
|-------|------|-------------|
| `time_to_first_token_ms` | int | Time from request sent to first streaming chunk |
| `total_response_time_ms` | int | Time from request sent to final chunk |
| `tokens_per_second` | float | `output_tokens / (total_response_time_s)` |
| `queue_time_ms` | int | Time spent waiting for rate limiter (if applicable) |

### Output Quality Proxy

Direct quality measurement is hard. Proxy metrics:

| Proxy | Measurement | Source |
|-------|-------------|--------|
| Test pass rate | % of generated code that passes tests on first try | CI results |
| Human acceptance rate | % of suggestions accepted without modification | Session logs |
| Retry rate | % of calls that required retry/regeneration | Adapter logs |
| Self-correction count | Number of correction loops per task | Orchestrator logs |

## Instrumentation

### Adapter Layer Integration

The base adapter logs cost and latency for every call:

```python
# adapters/base.py
class InstrumentedAdapter(BaseAdapter):
    async def generate(self, request: LLMRequest) -> LLMResponse:
        start = time.monotonic()
        first_token_time = None

        async for chunk in self._stream(request):
            if first_token_time is None:
                first_token_time = time.monotonic()
            yield chunk

        end = time.monotonic()

        metrics = InteractionMetrics(
            provider=self.provider,
            model=request.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cost_usd=compute_cost(request.model, response.usage),
            time_to_first_token_ms=int((first_token_time - start) * 1000),
            total_response_time_ms=int((end - start) * 1000),
            timestamp=datetime.utcnow(),
            session_id=request.session_id,
            feature_branch=request.metadata.get("branch"),
        )
        await self.metrics_store.record(metrics)
```

### Metrics Storage

```python
# runtime/metrics_store.py
class MetricsStore:
    """Append-only metrics log with aggregation queries."""

    def __init__(self, path: Path):
        self.path = path  # JSON Lines file

    async def record(self, metrics: InteractionMetrics):
        async with aiofiles.open(self.path, "a") as f:
            await f.write(metrics.model_dump_json() + "\n")

    async def aggregate(
        self,
        period: str = "daily",
        group_by: str = "provider",
    ) -> dict:
        """Aggregate metrics by period and grouping."""
        ...
```

Storage format: JSON Lines (one JSON object per line), rotated daily. Lightweight, greppable, no database dependency.

## Budget Tracking

### Budget Configuration

```yaml
# titan-config.yaml
budget:
  daily_limit_usd: 10.00
  weekly_limit_usd: 50.00
  monthly_limit_usd: 150.00
  alerts:
    - threshold: 0.80
      action: "warn"
      channel: "stderr"
    - threshold: 0.90
      action: "warn"
      channel: "notification"
    - threshold: 1.00
      action: "block"
      channel: "notification"
```

### Budget Enforcement

```python
class BudgetGuard:
    """Checks spend against budget before allowing LLM calls."""

    async def check(self, estimated_cost: float) -> BudgetDecision:
        current_spend = await self.metrics_store.period_spend("daily")
        projected = current_spend + estimated_cost

        if projected > self.config.daily_limit:
            return BudgetDecision.BLOCKED
        elif projected > self.config.daily_limit * 0.9:
            await self.notify("90% of daily budget consumed")
            return BudgetDecision.WARN
        elif projected > self.config.daily_limit * 0.8:
            await self.notify("80% of daily budget consumed")
            return BudgetDecision.WARN
        return BudgetDecision.OK
```

### Cost Estimation

Before sending a request, estimate cost from input token count:

```python
def estimate_cost(model: str, input_tokens: int, expected_output_ratio: float = 0.3) -> float:
    """Estimate cost before sending. Output ratio is empirically tuned per task type."""
    estimated_output = int(input_tokens * expected_output_ratio)
    return compute_cost(model, TokenUsage(input_tokens, estimated_output, 0, 0))
```

## Cost Per Feature

Aggregate API spend for all sessions associated with a feature branch:

```python
async def cost_per_feature(branch: str) -> FeatureCost:
    metrics = await metrics_store.query(feature_branch=branch)
    return FeatureCost(
        branch=branch,
        total_cost_usd=sum(m.cost_usd for m in metrics),
        total_tokens=sum(m.input_tokens + m.output_tokens for m in metrics),
        call_count=len(metrics),
        providers=Counter(m.provider for m in metrics),
        date_range=(min(m.timestamp for m in metrics), max(m.timestamp for m in metrics)),
    )
```

This enables questions like: "How much did F-18 cost to implement?" and informs future estimation.

## Provider Comparison

Weekly report comparing providers on cost-efficiency:

```
Provider Comparison — 2026-W10
──────────────────────────────────────────────────────────────
Provider       Calls   Avg Latency   Avg Cost   Quality Proxy
──────────────────────────────────────────────────────────────
Anthropic        142      1.2s        $0.045     92% accept
OpenAI            38      0.9s        $0.032     88% accept
Ollama (local)    67      3.1s        $0.000     76% accept
──────────────────────────────────────────────────────────────
```

## Alerts

| Alert | Trigger | Action |
|-------|---------|--------|
| Budget warning | 80% of daily/weekly/monthly limit | Log + stderr |
| Budget critical | 90% of limit | System notification |
| Budget exceeded | 100% of limit | Block further calls |
| Latency spike | p95 > 2x baseline | Log warning |
| Cost anomaly | Single call > $1.00 | Log warning |
| Quality drop | Accept rate < 70% for 10+ calls | Log warning |

## Dashboard Integration

Cost and latency metrics feed into the Reliability panel of the KPI dashboard (F-30):

```json
{
  "cost": {
    "daily_spend": 4.23,
    "daily_budget": 10.00,
    "weekly_spend": 28.50,
    "weekly_budget": 50.00,
    "top_model": "claude-sonnet-4-20250514",
    "top_model_spend": 3.80
  },
  "latency": {
    "p50_ttft_ms": 340,
    "p95_ttft_ms": 890,
    "p50_total_ms": 1200,
    "p95_total_ms": 4500
  }
}
```

## References

- `adapters/base.py` — `LLMConfig` and base adapter interface
- `adapters/router.py` — routing layer where instrumentation hooks in
- `conductors-scorecard.md` (F-29) — scorecard consumes cost metrics
- `kpi-dashboard-panels.md` (F-30) — dashboard displays cost/latency panels
- `removable-orchestration-layers.md` (F-18) — monitoring is an orchestration layer
