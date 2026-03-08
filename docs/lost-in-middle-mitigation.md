# F-68: "Lost in the Middle" Mitigation

> Preprocessing strategies for long-context LLM interactions.

## Problem

Research shows that LLMs perform worse on information placed in the middle of long contexts compared to information at the beginning or end ("Lost in the Middle" — Liu et al., 2023). As agentic-titan regularly constructs long prompts from multiple sources (repo context, tool outputs, conversation history, instructions), this effect degrades output quality.

## Strategies

### 1. Prompt Sandwiching

Place critical instructions at both the beginning and end of the prompt:

```python
def sandwich_prompt(system: str, context: str, instructions: str) -> str:
    """Place critical instructions at start AND end to combat position bias."""
    return f"""{instructions}

--- CONTEXT START ---
{context}
--- CONTEXT END ---

REMINDER: {instructions}"""
```

**When to use**: Always, for any prompt with context longer than 4,000 tokens.

**Implementation**: Preprocessing step in the adapter layer that detects long contexts and automatically duplicates the instruction block.

### 2. Chunking with Overlap

Split large inputs into overlapping windows to ensure no information falls entirely in a low-attention zone:

```python
def chunk_with_overlap(
    text: str,
    chunk_size: int = 4000,
    overlap: int = 500,
) -> list[str]:
    """Split text into overlapping chunks."""
    tokens = tokenize(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(detokenize(tokens[start:end]))
        start += chunk_size - overlap
    return chunks
```

**Overlap rationale**: 500-token overlap ensures that any passage split across chunk boundaries appears in full in at least one chunk.

**Processing pattern**: For each chunk, extract relevant information independently, then merge results:

```python
async def process_long_context(text: str, query: str, adapter: BaseAdapter) -> str:
    chunks = chunk_with_overlap(text, chunk_size=4000, overlap=500)

    # Phase 1: Extract from each chunk
    extractions = []
    for chunk in chunks:
        result = await adapter.generate(
            f"Extract information relevant to: {query}\n\nContext:\n{chunk}"
        )
        extractions.append(result)

    # Phase 2: Synthesize extractions
    combined = "\n---\n".join(extractions)
    return await adapter.generate(
        f"Synthesize these extractions into a coherent answer for: {query}\n\n{combined}"
    )
```

### 3. KV Cache Optimization (Local Models)

For locally hosted models via Ollama, KV cache management directly affects context quality:

```yaml
# ollama model configuration
parameters:
  num_ctx: 32768        # Context window size
  num_batch: 512        # Batch size for prompt processing
  num_keep: 256         # Tokens to always keep in KV cache (system prompt)
```

**Strategies**:
- **`num_keep`**: Pin system prompt and critical instructions in cache — they never get evicted
- **Quantized KV cache**: Use `q8_0` or `q4_0` KV quantization to fit more context in memory
- **Cache warming**: Pre-process common context (repo structure, governance rules) so it's cached across sessions

```python
async def warm_kv_cache(adapter: OllamaAdapter, common_context: str):
    """Pre-load common context into the model's KV cache."""
    await adapter.generate(
        model=adapter.model,
        prompt=f"Acknowledge this context:\n{common_context}",
        options={"num_predict": 1},  # Minimal output, just cache the input
    )
```

### 4. Retrieval-Based Context Injection

Instead of stuffing the full context into the prompt, use embeddings to select only the most relevant chunks:

```python
from chromadb import Client

class ContextRetriever:
    """Retrieve most relevant context chunks using embeddings."""

    def __init__(self, collection_name: str = "titan_context"):
        self.client = Client()
        self.collection = self.client.get_or_create_collection(collection_name)

    def index(self, documents: list[str], metadata: list[dict]):
        """Index documents for retrieval."""
        self.collection.add(
            documents=documents,
            metadatas=metadata,
            ids=[f"doc_{i}" for i in range(len(documents))],
        )

    def retrieve(self, query: str, n_results: int = 5) -> list[str]:
        """Retrieve the most relevant chunks for a query."""
        results = self.collection.query(query_texts=[query], n_results=n_results)
        return results["documents"][0]
```

**Integration with adapter layer**:

```python
async def generate_with_retrieval(
    adapter: BaseAdapter,
    query: str,
    retriever: ContextRetriever,
    max_context_tokens: int = 8000,
) -> str:
    # Retrieve only relevant context
    relevant_chunks = retriever.retrieve(query, n_results=5)
    context = "\n---\n".join(relevant_chunks)

    # Sandwich the prompt
    return await adapter.generate(
        sandwich_prompt(
            system="You are a helpful assistant.",
            context=context,
            instructions=query,
        )
    )
```

### 5. Context Window Budgeting

Explicitly allocate the context window across competing demands:

```python
@dataclass
class ContextBudget:
    """Allocate context window tokens across categories."""

    total_window: int           # Model's context window (e.g., 200000)
    reserved_output: int        # Tokens reserved for generation (e.g., 4096)
    system_prompt: int          # Fixed system prompt allocation
    instructions: int           # Task instructions (sandwiched)
    conversation_history: int   # Recent conversation turns
    retrieved_context: int      # RAG results
    repo_context: int           # Repo map, file contents

    @classmethod
    def from_window(cls, window: int, output_reserve: int = 4096) -> "ContextBudget":
        available = window - output_reserve
        return cls(
            total_window=window,
            reserved_output=output_reserve,
            system_prompt=int(available * 0.05),       # 5%
            instructions=int(available * 0.10),         # 10% (appears twice with sandwich)
            conversation_history=int(available * 0.20), # 20%
            retrieved_context=int(available * 0.35),    # 35%
            repo_context=int(available * 0.30),         # 30%
        )
```

**Budget enforcement**:

```python
def enforce_budget(content: dict[str, str], budget: ContextBudget) -> dict[str, str]:
    """Truncate each content category to its budget allocation."""
    result = {}
    for category, text in content.items():
        limit = getattr(budget, category)
        tokens = tokenize(text)
        if len(tokens) > limit:
            result[category] = detokenize(tokens[:limit])
            logger.warning(f"Truncated {category}: {len(tokens)} → {limit} tokens")
        else:
            result[category] = text
    return result
```

## Priority Order

When context must be cut, priority order (highest first):

1. **Instructions** — always kept in full (sandwiched)
2. **System prompt** — always kept
3. **Retrieved context** — most relevant chunks
4. **Conversation history** — recent turns preferred, older turns summarized
5. **Repo context** — repo map first, file contents by relevance

## Implementation in Adapter Layer

The mitigation strategies are implemented as a preprocessing step in `adapters/base.py`:

```python
class ContextPreprocessor:
    """Preprocess context before sending to LLM."""

    def __init__(self, config: LLMConfig):
        self.budget = ContextBudget.from_window(config.context_window)
        self.retriever = ContextRetriever() if config.use_retrieval else None

    async def preprocess(self, request: LLMRequest) -> LLMRequest:
        # Step 1: Budget allocation
        content = enforce_budget(request.content_parts, self.budget)

        # Step 2: Retrieval (if enabled)
        if self.retriever and request.query:
            content["retrieved_context"] = "\n---\n".join(
                self.retriever.retrieve(request.query)
            )

        # Step 3: Sandwich critical instructions
        request.prompt = sandwich_prompt(
            system=content["system_prompt"],
            context=assemble_context(content),
            instructions=content["instructions"],
        )

        return request
```

## Measurement

Track whether mitigation is working:

| Metric | Baseline (no mitigation) | Target |
|--------|--------------------------|--------|
| Instruction compliance | ~70% for long contexts | >= 90% |
| Relevant detail recall | ~60% for mid-context info | >= 80% |
| Output quality proxy | Varies | Measurably improved |

Test with synthetic benchmarks: place target information at beginning, middle, and end of context; measure retrieval accuracy with and without mitigation.

## References

- Liu et al. (2023) — "Lost in the Middle: How Language Models Use Long Contexts"
- `adapters/base.py` — `LLMConfig` with `context_window` parameter
- `cost-latency-monitoring.md` (F-31) — token budget affects cost
- `removable-orchestration-layers.md` (F-18) — context preprocessing is a removable layer
