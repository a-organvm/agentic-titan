"""
RAG Tool - Retrieval Augmented Generation.

Implements RAG patterns from anthropic-cookbook:
- Vector-based semantic search
- Chunk management with metadata
- Multi-level retrieval (basic, summary-indexed, reranking)
- Source citation support

Reference: vendor/cookbooks/anthropic-cookbook/skills/retrieval_augmented_generation/
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from tools.base import Tool, ToolParameter, ToolResult, register_tool

logger = logging.getLogger("titan.tools.rag")


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class Chunk:
    """A document chunk for RAG."""

    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    summary: str | None = None

    # Source tracking
    source: str = ""
    source_type: str = "text"  # text, pdf, url, code
    page_number: int | None = None
    line_start: int | None = None
    line_end: int | None = None

    # Timestamp
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "summary": self.summary,
            "source": self.source,
            "source_type": self.source_type,
            "page_number": self.page_number,
            "line_start": self.line_start,
            "line_end": self.line_end,
        }


@dataclass
class SearchResult:
    """A search result with relevance score."""

    chunk: Chunk
    score: float
    relevance_explanation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = self.chunk.to_dict()
        result["score"] = self.score
        if self.relevance_explanation:
            result["relevance_explanation"] = self.relevance_explanation
        return result


# ============================================================================
# Embedding Functions
# ============================================================================


def hash_embedding(text: str, dim: int = 384) -> list[float]:
    """
    Fast hash-based embedding for text.

    Uses character n-grams hashed to a fixed dimension.
    Suitable for development; use sentence-transformers for production.
    """
    ngrams: list[str] = []
    text_lower = text.lower()
    for n in [2, 3, 4]:
        for i in range(len(text_lower) - n + 1):
            ngrams.append(text_lower[i : i + n])

    embedding = [0.0] * dim
    for ngram in ngrams:
        h = int(hashlib.md5(ngram.encode()).hexdigest(), 16)
        idx = h % dim
        sign = 1 if (h // dim) % 2 == 0 else -1
        embedding[idx] += sign * 1.0

    # Normalize
    magnitude = sum(x * x for x in embedding) ** 0.5
    if magnitude > 0:
        embedding = [x / magnitude for x in embedding]

    return embedding


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(x * x for x in b) ** 0.5

    if mag_a == 0 or mag_b == 0:
        return 0.0

    return float(dot / (mag_a * mag_b))


# ============================================================================
# RAG Store
# ============================================================================


class RAGStore:
    """
    In-memory RAG store with vector search.

    For production, integrate with ChromaDB or similar.
    """

    def __init__(
        self,
        name: str = "default",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self.name = name
        self.embedding_fn = embedding_fn or hash_embedding
        self._chunks: dict[str, Chunk] = {}
        self._embeddings: dict[str, list[float]] = {}

    def add_chunk(self, chunk: Chunk) -> str:
        """Add a chunk to the store."""
        # Generate embedding if not present
        if chunk.embedding is None:
            chunk.embedding = self.embedding_fn(chunk.content)

        self._chunks[chunk.id] = chunk
        self._embeddings[chunk.id] = chunk.embedding

        logger.debug(f"Added chunk {chunk.id} to RAG store {self.name}")
        return chunk.id

    def add_text(
        self,
        text: str,
        source: str = "",
        metadata: dict[str, Any] | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> list[str]:
        """
        Add text to the store, splitting into chunks.

        Args:
            text: Text to add
            source: Source identifier
            metadata: Additional metadata
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks

        Returns:
            List of chunk IDs
        """
        chunks = self._split_text(text, chunk_size, chunk_overlap)
        chunk_ids = []

        for i, content in enumerate(chunks):
            chunk_id = f"{source or 'text'}-{hashlib.md5(content.encode()).hexdigest()[:8]}-{i}"
            chunk = Chunk(
                id=chunk_id,
                content=content,
                metadata=metadata or {},
                source=source,
                source_type="text",
            )
            self.add_chunk(chunk)
            chunk_ids.append(chunk_id)

        logger.info(f"Added {len(chunk_ids)} chunks from source '{source}'")
        return chunk_ids

    def _split_text(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[str]:
        """Split text into overlapping chunks."""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                # Look for period, newline, etc.
                for sep in [". ", ".\n", "\n\n", "\n", " "]:
                    sep_pos = text.rfind(sep, start + chunk_size // 2, end)
                    if sep_pos != -1:
                        end = sep_pos + len(sep)
                        break

            chunks.append(text[start:end].strip())
            start = end - chunk_overlap

        return chunks

    def search(
        self,
        query: str,
        k: int = 5,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """
        Search for relevant chunks.

        Args:
            query: Search query
            k: Number of results
            min_score: Minimum similarity score

        Returns:
            List of search results
        """
        if not self._chunks:
            return []

        query_embedding = self.embedding_fn(query)

        # Compute similarities
        scores: list[tuple[str, float]] = []
        for chunk_id, embedding in self._embeddings.items():
            score = cosine_similarity(query_embedding, embedding)
            if score >= min_score:
                scores.append((chunk_id, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        # Return top k results
        results = []
        for chunk_id, score in scores[:k]:
            chunk = self._chunks[chunk_id]
            results.append(SearchResult(chunk=chunk, score=score))

        return results

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """Get a chunk by ID."""
        return self._chunks.get(chunk_id)

    def delete_chunk(self, chunk_id: str) -> bool:
        """Delete a chunk."""
        if chunk_id in self._chunks:
            del self._chunks[chunk_id]
            del self._embeddings[chunk_id]
            return True
        return False

    def clear(self) -> None:
        """Clear all chunks."""
        self._chunks.clear()
        self._embeddings.clear()

    def count(self) -> int:
        """Get number of chunks."""
        return len(self._chunks)

    def save(self, path: str | Path) -> None:
        """Save store to disk."""
        path = Path(path)
        data = {
            "name": self.name,
            "chunks": [c.to_dict() for c in self._chunks.values()],
            "embeddings": self._embeddings,
        }
        path.write_text(json.dumps(data))
        logger.info(f"Saved RAG store to {path}")

    def load(self, path: str | Path) -> None:
        """Load store from disk."""
        path = Path(path)
        data = json.loads(path.read_text())

        self.name = data.get("name", "default")
        self._chunks.clear()
        self._embeddings.clear()

        for chunk_data in data.get("chunks", []):
            chunk = Chunk(
                id=chunk_data["id"],
                content=chunk_data["content"],
                metadata=chunk_data.get("metadata", {}),
                summary=chunk_data.get("summary"),
                source=chunk_data.get("source", ""),
                source_type=chunk_data.get("source_type", "text"),
                page_number=chunk_data.get("page_number"),
                line_start=chunk_data.get("line_start"),
                line_end=chunk_data.get("line_end"),
            )
            self._chunks[chunk.id] = chunk

        self._embeddings = data.get("embeddings", {})
        logger.info(f"Loaded RAG store from {path}")


# ============================================================================
# Global Store Registry
# ============================================================================

_stores: dict[str, RAGStore] = {}


def get_store(name: str = "default") -> RAGStore:
    """Get or create a RAG store."""
    if name not in _stores:
        _stores[name] = RAGStore(name)
    return _stores[name]


def list_stores() -> list[str]:
    """List all store names."""
    return list(_stores.keys())


# ============================================================================
# RAG Tool Implementation
# ============================================================================


class RAGTool(Tool):
    """
    Retrieval Augmented Generation tool.

    Allows agents to:
    - Index documents into a vector store
    - Search for relevant context
    - Retrieve specific chunks
    """

    @property
    def name(self) -> str:
        return "rag"

    @property
    def description(self) -> str:
        return (
            "Retrieval Augmented Generation tool for indexing and searching documents. "
            "Use 'index' action to add documents, 'search' to find relevant content, "
            "'get' to retrieve specific chunks, 'list' to see indexed sources."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="action",
                type="string",
                description="Action to perform: 'index', 'search', 'get', 'list', 'clear'",
                required=True,
                enum=["index", "search", "get", "list", "clear"],
            ),
            ToolParameter(
                name="text",
                type="string",
                description="Text to index (for 'index' action)",
                required=False,
            ),
            ToolParameter(
                name="query",
                type="string",
                description="Search query (for 'search' action)",
                required=False,
            ),
            ToolParameter(
                name="chunk_id",
                type="string",
                description="Chunk ID to retrieve (for 'get' action)",
                required=False,
            ),
            ToolParameter(
                name="source",
                type="string",
                description="Source identifier for indexed content",
                required=False,
            ),
            ToolParameter(
                name="store_name",
                type="string",
                description="RAG store name (default: 'default')",
                required=False,
            ),
            ToolParameter(
                name="k",
                type="integer",
                description="Number of search results (default: 5)",
                required=False,
            ),
        ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        action = kwargs.get("action", "")
        store_name = kwargs.get("store_name", "default")
        store = get_store(store_name)

        try:
            if action == "index":
                text = kwargs.get("text", "")
                if not text:
                    return ToolResult(
                        success=False,
                        output=None,
                        error="'text' parameter required for index action",
                    )

                source = kwargs.get("source", "")
                chunk_ids = store.add_text(text, source=source)

                return ToolResult(
                    success=True,
                    output={
                        "indexed": True,
                        "chunk_count": len(chunk_ids),
                        "chunk_ids": chunk_ids,
                        "store": store_name,
                    },
                )

            elif action == "search":
                query = kwargs.get("query", "")
                if not query:
                    return ToolResult(
                        success=False,
                        output=None,
                        error="'query' parameter required for search action",
                    )

                k = kwargs.get("k", 5)
                results = store.search(query, k=k)

                return ToolResult(
                    success=True,
                    output={
                        "results": [r.to_dict() for r in results],
                        "count": len(results),
                        "query": query,
                    },
                )

            elif action == "get":
                chunk_id = kwargs.get("chunk_id", "")
                if not chunk_id:
                    return ToolResult(
                        success=False,
                        output=None,
                        error="'chunk_id' parameter required for get action",
                    )

                chunk = store.get_chunk(chunk_id)
                if chunk:
                    return ToolResult(
                        success=True,
                        output=chunk.to_dict(),
                    )
                else:
                    return ToolResult(
                        success=False,
                        output=None,
                        error=f"Chunk '{chunk_id}' not found",
                    )

            elif action == "list":
                sources = set()
                for chunk in store._chunks.values():
                    if chunk.source:
                        sources.add(chunk.source)

                return ToolResult(
                    success=True,
                    output={
                        "store": store_name,
                        "chunk_count": store.count(),
                        "sources": list(sources),
                    },
                )

            elif action == "clear":
                count = store.count()
                store.clear()
                return ToolResult(
                    success=True,
                    output={
                        "cleared": True,
                        "chunks_removed": count,
                    },
                )

            else:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Unknown action: {action}",
                )

        except Exception as e:
            logger.exception(f"RAG tool error: {e}")
            return ToolResult(
                success=False,
                output=None,
                error=str(e),
            )


# Register the tool
rag_tool = RAGTool()
register_tool(rag_tool)
