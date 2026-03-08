# F-18: Removable Orchestration Layers

> Design doc for making Taxis routing layers configurable and removable.

## Motivation

As base models improve — growing more capable of tool use, multi-step reasoning, and self-correction — the complex multi-agent routing that justified itself in 2024-25 becomes overhead rather than value. Agentic-titan must allow orchestration layers to collapse gracefully without architectural rework.

The principle: **every layer earns its keep or gets out of the way.**

## Layer Inventory

| Layer | Module | Purpose | Bypass Cost |
|-------|--------|---------|-------------|
| **Routing** | `adapters/router.py` | Select model/provider per task | Low — default to primary model |
| **Retry/Fallback** | `runtime/retry.py` | Retry failed LLM calls, fall back to alternate providers | Medium — single provider must be reliable |
| **Load Balancing** | `adapters/router.py` | Distribute across providers/instances | Low — single instance sufficient for solo operator |
| **Caching** | `runtime/cache.py` | Cache identical prompts/tool outputs | Low — increases latency and cost slightly |
| **Audit Logging** | `runtime/audit.py` | Log all LLM interactions for governance | None — always recommended, but can be made async/buffered |
| **Safety Gates** | `runtime/safety.py` | HITL gates, budget caps, tool allowlists | High — removing safety requires explicit opt-in |

## Configuration Model

### YAML Configuration

```yaml
# titan-config.yaml or within seed.yaml agent_config section
orchestration:
  layers:
    routing:
      enabled: true
      fallback: "use_default_model"
      bypass_when:
        - "single_provider_configured"
        - "model_explicitly_specified"

    retry_fallback:
      enabled: true
      fallback: "fail_fast"
      max_retries: 3
      bypass_when:
        - "provider_sla_above_99"

    load_balancing:
      enabled: false
      fallback: "round_robin"
      bypass_when:
        - "single_instance"

    caching:
      enabled: true
      fallback: "passthrough"
      ttl_seconds: 3600
      bypass_when:
        - "cache_backend_unavailable"

    audit_logging:
      enabled: true
      fallback: "stderr_fallback"
      bypass_when: []  # never auto-bypass

    safety_gates:
      enabled: true
      fallback: "block"
      bypass_when: []  # never auto-bypass
```

### Environment Variable Overrides

```bash
TITAN_LAYER_ROUTING_ENABLED=false
TITAN_LAYER_CACHING_ENABLED=false
TITAN_LAYER_LOAD_BALANCING_ENABLED=true
```

Environment variables override YAML. This enables per-session or per-environment tuning without config file changes.

## Layer Interface

Every orchestration layer implements a common interface:

```python
class OrchestrationLayer(Protocol):
    """Base protocol for all orchestration layers."""

    name: str
    enabled: bool

    async def process(self, request: LLMRequest, next_layer: Callable) -> LLMResponse:
        """Process a request, optionally delegating to the next layer."""
        ...

    async def passthrough(self, request: LLMRequest, next_layer: Callable) -> LLMResponse:
        """Bypass this layer entirely — delegate to next without modification."""
        return await next_layer(request)

    def should_bypass(self, request: LLMRequest, context: RuntimeContext) -> bool:
        """Evaluate bypass conditions against current context."""
        ...
```

### Pipeline Assembly

```python
def build_pipeline(config: OrchestrationConfig) -> Pipeline:
    """Assemble the layer pipeline from config, skipping disabled layers."""
    layers = []
    for layer_config in config.layers:
        layer = LAYER_REGISTRY[layer_config.name]
        if layer_config.enabled:
            layers.append(layer)
        # Disabled layers are simply not added — zero overhead
    return Pipeline(layers)
```

When a layer is disabled, it is not instantiated. No wrapper, no passthrough call, no overhead. The pipeline is assembled at startup from the active layer set.

## Fallback Behaviors

Each layer defines what happens when it is disabled or bypassed:

| Fallback Strategy | Behavior |
|-------------------|----------|
| `passthrough` | Request passes to next layer unchanged |
| `use_default_model` | Skip routing logic, use configured default |
| `fail_fast` | No retries — first failure is final |
| `block` | Reject the request (safety-critical layers) |
| `stderr_fallback` | Log to stderr instead of structured audit log |

## Bypass Conditions

Bypass conditions are evaluated at runtime. A layer can be enabled in config but bypassed dynamically when conditions are met:

```yaml
bypass_when:
  - "single_provider_configured"   # Only one provider → routing is pointless
  - "model_explicitly_specified"    # Caller chose the model → don't override
  - "context_window_under_4k"      # Small request → skip chunking/caching
  - "provider_sla_above_99"        # Provider reliable → skip retry logic
```

Conditions are simple string keys mapped to evaluator functions. Custom conditions can be registered.

## Future: Model Capability Detection

As models gain native tool use, self-correction, and structured output:

1. **Capability probing**: at startup, test the configured model's capabilities (tool calling, JSON mode, etc.)
2. **Auto-disable**: if a model handles retries internally, disable the retry layer; if it has native routing (e.g., multi-model orchestration), disable the routing layer
3. **Capability manifest**: models or providers publish a capability manifest that Titan reads to auto-configure layers

```yaml
# Future: auto-detected capability map
model_capabilities:
  claude-4:
    native_retry: true        # → disable retry_fallback layer
    native_tool_use: true     # → simplify routing
    structured_output: true   # → skip output parsing layer
    context_window: 200000    # → adjust chunking thresholds
```

## Migration Path

1. **Phase 1**: Add `enabled` toggle to all existing layers (backward compatible — all default to `true`)
2. **Phase 2**: Implement `bypass_when` conditions for routing and retry layers
3. **Phase 3**: Add capability detection for locally hosted models (Ollama)
4. **Phase 4**: Auto-configuration based on model capability manifests

## References

- `adapters/router.py` — current routing implementation
- `titan/` — topology engine
- `runtime/` — execution runtime
- seed.yaml `agent_config` section — per-repo configuration
