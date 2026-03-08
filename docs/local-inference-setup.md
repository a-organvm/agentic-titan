# Local Inference Setup — Ollama + agentic-titan

> **Features**: F-22, F-23, F-24
> **Scope**: Local model inference for data sovereignty and cost optimization
> **Version**: 1.0

---

## Prerequisites

- macOS (Apple Silicon recommended) or Linux
- Homebrew (macOS) or curl (Linux)
- 8GB+ RAM (16GB recommended for 7B models)

## Installation

### macOS (Homebrew)

```bash
brew install ollama
```

### Linux

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Verify

```bash
ollama --version
ollama serve &   # Start the server (runs on localhost:11434)
```

## Recommended Models

Models are selected for 16GB RAM constraint (Apple Silicon M3):

| Model | Size | Context | Use Case |
|-------|------|---------|----------|
| `llama3.2:3b` | 2.0 GB | 128K | General purpose, fast |
| `llama3.2:1b` | 1.3 GB | 128K | Lightweight tasks, embedding prep |
| `qwen2.5:3b` | 1.9 GB | 128K | Code-aware, multilingual |
| `gemma3:4b` | 3.3 GB | 128K | Balanced quality/speed |
| `nomic-embed-text` | 274 MB | 8K | Embeddings only |

### Pull Models

```bash
ollama pull llama3.2:3b          # Primary general model
ollama pull nomic-embed-text     # Embeddings
ollama pull qwen2.5:3b           # Code tasks (optional)
```

### Memory Budget

With 16GB system RAM, budget ~6-8GB for models. Rules of thumb:
- 1B model ≈ 1.3 GB VRAM
- 3B model ≈ 2.0 GB VRAM
- 7B model ≈ 4.5 GB VRAM (tight on 16GB with other apps)
- Only one model loaded at a time by default

## Context Window Configuration (F-23)

Ollama defaults to 2K-4K context, but agents need larger windows.
The `LLMConfig.context_window` parameter is passed as `num_ctx` to Ollama.

```python
from adapters.base import LLMConfig, LLMProvider, OllamaAdapter

config = LLMConfig(
    provider=LLMProvider.OLLAMA,
    model="llama3.2:3b",
    context_window=16384,  # 16K context — passed as num_ctx to Ollama
)
adapter = OllamaAdapter(config)
```

### Context Window Guidelines

| Task Type | Recommended `context_window` |
|-----------|------------------------------|
| Simple Q&A | 4096 |
| Code review | 16384 |
| Document analysis | 32768 |
| Multi-file agent | 65536+ |

Higher context windows use more memory. On 16GB RAM with a 3B model:
- 16K context: ~3 GB total
- 32K context: ~4 GB total
- 64K context: ~6 GB total

## Sensitivity-Based Routing (F-24)

The router enforces data classification constraints:

```python
from adapters.router import DataClassification, LLMRouter, RoutingStrategy

router = LLMRouter(strategy=RoutingStrategy.SENSITIVITY)
await router.initialize()

# Public data — any provider
response = await router.complete(
    messages,
    classification=DataClassification.PUBLIC,
)

# Confidential data — local only (Ollama)
response = await router.complete(
    messages,
    classification=DataClassification.CONFIDENTIAL,
)

# Regulated data — local only + audit logging
response = await router.complete(
    messages,
    classification=DataClassification.REGULATED,
)
```

### Classification Tiers

| Tier | Allowed Providers | Audit | Examples |
|------|-------------------|-------|----------|
| **PUBLIC** | All (cloud + local) | No | Public docs, open source code |
| **INTERNAL** | All (cloud + local) | No | Internal notes, drafts |
| **CONFIDENTIAL** | Local only (Ollama) | No | Personal data, credentials, PII |
| **REGULATED** | Local only (Ollama) | Yes | Financial data, health data, legal |

### What Happens When No Local Provider Is Available

If a request is classified as CONFIDENTIAL or REGULATED but no local
provider (Ollama) is running, the router raises `LLMAdapterError` rather
than silently falling back to a cloud provider. This is a safety
guarantee — confidential data never leaves the local machine.

## Integration with agentic-titan

### Via Router (recommended)

```python
from adapters.router import DataClassification, get_router

router = get_router()
await router.initialize()

# The router auto-detects Ollama and adds it to the fallback chain
response = await router.complete(
    messages,
    classification=DataClassification.CONFIDENTIAL,
)
```

### Via Direct Adapter

```python
from adapters.base import LLMConfig, LLMProvider, OllamaAdapter, LLMMessage

config = LLMConfig(
    provider=LLMProvider.OLLAMA,
    model="llama3.2:3b",
    context_window=16384,
)
adapter = OllamaAdapter(config)

response = await adapter.complete(
    [LLMMessage(role="user", content="Analyze this code...")],
    system="You are a code reviewer.",
)
```

## Troubleshooting

### Ollama not responding

```bash
# Check if server is running
curl http://localhost:11434/api/tags

# Restart
killall ollama
ollama serve &
```

### Out of memory

Reduce context window or use a smaller model:

```python
config = LLMConfig(
    provider=LLMProvider.OLLAMA,
    model="llama3.2:1b",       # Smaller model
    context_window=8192,        # Smaller context
)
```

### Slow inference

Apple Silicon uses GPU acceleration by default. If inference is slow:
- Close memory-heavy apps (browsers, Docker)
- Use a smaller model (1B vs 3B)
- Reduce `context_window`
