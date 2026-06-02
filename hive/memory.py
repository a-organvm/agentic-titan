"""
Hive Mind - Memory Layer

Provides shared memory infrastructure for the agent swarm:
- Long-term memory: ChromaDB vector store for semantic search
- Working memory: Redis for fast key-value access
- Event bus: Redis Pub/Sub or NATS for agent communication
- State management: Distributed state with Redis

This is the "collective consciousness" that enables agents to share
knowledge and coordinate in real-time.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

logger = logging.getLogger("titan.hive.memory")


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class MemoryConfig:
    """Configuration for Hive Mind memory."""

    # Redis configuration
    redis_url: str = "redis://localhost:6379"
    redis_db: int = 0
    redis_prefix: str = "titan:"

    # ChromaDB configuration
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_collection: str = "titan_memories"

    # NATS configuration (optional, falls back to Redis Pub/Sub)
    nats_url: str | None = None

    # Memory settings
    max_short_term_items: int = 100
    memory_ttl_seconds: int = 3600
    embedding_model: str = "hash"  # "hash" or "e5-small" or custom

    # Performance
    connection_pool_size: int = 10
    connection_timeout: float = 5.0


@dataclass
class Memory:
    """A single memory entry."""

    id: str
    agent_id: str
    content: str
    importance: float
    tags: list[str]
    metadata: dict[str, Any]
    timestamp: datetime
    embedding: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "content": self.content,
            "importance": self.importance,
            "tags": self.tags,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class Message:
    """A message between agents."""

    id: str
    source_agent_id: str
    target_agent_id: str | None  # None for broadcast
    topic: str
    content: dict[str, Any]
    timestamp: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_agent_id": self.source_agent_id,
            "target_agent_id": self.target_agent_id,
            "topic": self.topic,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================================
# Embedding Functions
# ============================================================================


def hash_embedding(text: str, dim: int = 128) -> list[float]:
    """
    Fast hash-based embedding for text.

    Uses character n-grams hashed to a fixed dimension.
    Much faster than ML-based embeddings, suitable for development.
    """
    # Create n-grams
    ngrams: list[str] = []
    text_lower = text.lower()
    for n in [2, 3, 4]:
        for i in range(len(text_lower) - n + 1):
            ngrams.append(text_lower[i : i + n])

    # Hash n-grams to embedding dimensions
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


# ============================================================================
# Hive Mind Implementation
# ============================================================================


class HiveMind:
    """
    Shared intelligence layer for the agent swarm.

    Provides:
    - remember() / recall() - Long-term semantic memory
    - set() / get() - Fast working memory
    - broadcast() / send() / subscribe() - Agent communication
    - get_topology() / set_topology() - Swarm state
    """

    def __init__(self, config: MemoryConfig | None = None) -> None:
        self.config = config or MemoryConfig()

        # Connection state
        self._redis: Any = None  # redis.asyncio.Redis
        self._chroma_client: Any = None  # chromadb.HttpClient
        self._chroma_collection: Any = None
        self._nats: Any = None  # nats.NATS
        self._initialized = False

        # In-memory fallbacks for development
        self._memory_store: dict[str, Memory] = {}
        self._working_memory: dict[str, Any] = {}
        self._subscriptions: dict[str, list[Callable[[dict[str, Any]], Any]]] = {}

        # State
        self._topology_state: dict[str, Any] = {}
        self._agent_registry: dict[str, dict[str, Any]] = {}

        logger.info("Hive Mind created")

    async def initialize(self) -> None:
        """Initialize connections to backing stores."""
        if self._initialized:
            return

        # Try to connect to Redis
        try:
            import redis.asyncio as redis

            from_url = cast(Any, redis.from_url)
            self._redis = from_url(
                self.config.redis_url,
                db=self.config.redis_db,
                socket_connect_timeout=self.config.connection_timeout,
            )
            await self._redis.ping()
            logger.info(f"Connected to Redis at {self.config.redis_url}")
        except Exception as e:
            logger.warning(f"Redis unavailable, using in-memory fallback: {e}")
            self._redis = None

        # Try to connect to ChromaDB
        try:
            import chromadb

            self._chroma_client = chromadb.HttpClient(
                host=self.config.chroma_host,
                port=self.config.chroma_port,
            )
            self._chroma_collection = self._chroma_client.get_or_create_collection(
                name=self.config.chroma_collection,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                f"Connected to ChromaDB at {self.config.chroma_host}:{self.config.chroma_port}"
            )
        except Exception as e:
            logger.warning(f"ChromaDB unavailable, using in-memory fallback: {e}")
            self._chroma_client = None

        # Try to connect to NATS (optional)
        if self.config.nats_url:
            try:
                import nats

                self._nats = await nats.connect(self.config.nats_url)
                logger.info(f"Connected to NATS at {self.config.nats_url}")
            except Exception as e:
                logger.warning(f"NATS unavailable, using Redis Pub/Sub: {e}")
                self._nats = None

        self._initialized = True
        logger.info("Hive Mind initialized")

    async def shutdown(self) -> None:
        """Close all connections."""
        if self._redis:
            await self._redis.close()
        if self._nats:
            await self._nats.close()
        self._initialized = False
        logger.info("Hive Mind shutdown")

    # =========================================================================
    # Long-Term Memory (Vector Store)
    # =========================================================================

    async def remember(
        self,
        agent_id: str,
        content: str,
        importance: float = 0.5,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Store content in long-term memory.

        Args:
            agent_id: ID of agent storing the memory
            content: Content to remember
            importance: Importance score (0.0 to 1.0)
            tags: Optional tags for categorization
            metadata: Optional metadata

        Returns:
            Memory ID
        """
        await self._ensure_initialized()

        memory_id = f"mem-{uuid.uuid4().hex[:12]}"
        loop = asyncio.get_running_loop()

        # Track executor wait time
        start_wait = time.time()
        embedding = await loop.run_in_executor(None, hash_embedding, content)
        wait_duration = time.time() - start_wait

        if self._initialized:
            from titan.metrics import get_metrics

            get_metrics().embedding_wait(wait_duration)

        memory = Memory(
            id=memory_id,
            agent_id=agent_id,
            content=content,
            importance=importance,
            tags=tags or [],
            metadata=metadata or {},
            timestamp=datetime.now(),
            embedding=embedding,
        )

        # Store in ChromaDB if available
        if self._chroma_collection is not None:
            self._chroma_collection.add(
                ids=[memory_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[
                    {
                        "agent_id": agent_id,
                        "importance": importance,
                        "tags": json.dumps(tags or []),
                        "metadata": json.dumps(metadata or {}),
                        "timestamp": memory.timestamp.isoformat(),
                    }
                ],
            )
        else:
            # In-memory fallback
            self._memory_store[memory_id] = memory

        logger.debug(f"Remembered: {memory_id}")
        return memory_id

    async def recall(
        self,
        query: str,
        k: int = 10,
        tags: list[str] | None = None,
        agent_id: str | None = None,
        min_importance: float = 0.0,
    ) -> list[dict[str, Any]]:
        """
        Recall memories using semantic search.

        Args:
            query: Search query
            k: Number of results
            tags: Filter by tags
            agent_id: Filter by agent
            min_importance: Minimum importance threshold

        Returns:
            List of matching memories
        """
        await self._ensure_initialized()

        loop = asyncio.get_running_loop()
        query_embedding = await loop.run_in_executor(None, hash_embedding, query)

        # Query ChromaDB if available
        if self._chroma_collection is not None:
            where_filter: dict[str, Any] = {}
            if agent_id:
                where_filter["agent_id"] = agent_id
            if min_importance > 0:
                where_filter["importance"] = {"$gte": min_importance}

            results = self._chroma_collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where_filter if where_filter else None,
            )

            memories = []
            for i, doc in enumerate(results["documents"][0] if results["documents"] else []):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                memories.append(
                    {
                        "id": results["ids"][0][i] if results["ids"] else f"mem-{i}",
                        "content": doc,
                        "distance": results["distances"][0][i] if results.get("distances") else 0,
                        "agent_id": meta.get("agent_id"),
                        "importance": meta.get("importance", 0.5),
                        "tags": json.loads(meta.get("tags", "[]")),
                        "timestamp": meta.get("timestamp"),
                    }
                )
            return memories

        # In-memory fallback with simple similarity
        results = []
        for mem in self._memory_store.values():
            if agent_id and mem.agent_id != agent_id:
                continue
            if mem.importance < min_importance:
                continue
            if tags and not any(t in mem.tags for t in tags):
                continue

            # Compute similarity
            if mem.embedding:
                similarity = sum(a * b for a, b in zip(query_embedding, mem.embedding))
            else:
                similarity = 0

            results.append((similarity, mem))

        # Sort by similarity
        results.sort(key=lambda x: x[0], reverse=True)
        return [mem.to_dict() for _, mem in results[:k]]

    async def forget(self, memory_id: str) -> bool:
        """
        Remove a memory.

        Args:
            memory_id: Memory ID to remove

        Returns:
            True if removed
        """
        await self._ensure_initialized()

        if self._chroma_collection is not None:
            try:
                self._chroma_collection.delete(ids=[memory_id])
                return True
            except Exception:
                return False

        # In-memory fallback
        if memory_id in self._memory_store:
            del self._memory_store[memory_id]
            return True
        return False

    # =========================================================================
    # Working Memory (Redis KV)
    # =========================================================================

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
    ) -> None:
        """
        Set a value in working memory.

        Args:
            key: Key
            value: Value (will be JSON serialized)
            ttl: Time-to-live in seconds
        """
        await self._ensure_initialized()

        full_key = f"{self.config.redis_prefix}{key}"

        if self._redis:
            serialized = json.dumps(value)
            if ttl:
                await self._redis.setex(full_key, ttl, serialized)
            else:
                await self._redis.set(full_key, serialized)
        else:
            # In-memory fallback
            self._working_memory[full_key] = {
                "value": value,
                "expires": time.time() + ttl if ttl else None,
            }

    async def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from working memory.

        Args:
            key: Key
            default: Default value if not found

        Returns:
            Value or default
        """
        await self._ensure_initialized()

        full_key = f"{self.config.redis_prefix}{key}"

        if self._redis:
            value = await self._redis.get(full_key)
            if value:
                return json.loads(value)
            return default

        # In-memory fallback
        entry = self._working_memory.get(full_key)
        if entry:
            if entry["expires"] and time.time() > entry["expires"]:
                del self._working_memory[full_key]
                return default
            return entry["value"]
        return default

    async def delete(self, key: str) -> bool:
        """Delete a key from working memory."""
        await self._ensure_initialized()

        full_key = f"{self.config.redis_prefix}{key}"

        if self._redis:
            result = await self._redis.delete(full_key)
            return isinstance(result, int) and result > 0

        # In-memory fallback
        if full_key in self._working_memory:
            del self._working_memory[full_key]
            return True
        return False

    # =========================================================================
    # Agent Communication
    # =========================================================================

    async def broadcast(
        self,
        source_agent_id: str,
        message: dict[str, Any],
        topic: str = "general",
    ) -> None:
        """
        Broadcast a message to all agents.

        Args:
            source_agent_id: Sender agent ID
            message: Message content
            topic: Message topic
        """
        await self._ensure_initialized()

        msg = Message(
            id=f"msg-{uuid.uuid4().hex[:12]}",
            source_agent_id=source_agent_id,
            target_agent_id=None,
            topic=topic,
            content=message,
            timestamp=datetime.now(),
        )

        channel = f"{self.config.redis_prefix}channel:{topic}"

        if self._nats:
            await self._nats.publish(channel, json.dumps(msg.to_dict()).encode())
        elif self._redis:
            await self._redis.publish(channel, json.dumps(msg.to_dict()))
        else:
            # In-memory fallback
            handlers = self._subscriptions.get(topic, [])
            for handler in handlers:
                try:
                    await self._invoke_handler(handler, msg.to_dict())
                except Exception as e:
                    logger.error(f"Handler error: {e}")

        logger.debug(f"Broadcast to {topic}: {msg.id}")

    async def send(
        self,
        source_agent_id: str,
        target_agent_id: str,
        message: dict[str, Any],
    ) -> None:
        """
        Send a direct message to another agent.

        Args:
            source_agent_id: Sender agent ID
            target_agent_id: Recipient agent ID
            message: Message content
        """
        await self._ensure_initialized()

        msg = Message(
            id=f"msg-{uuid.uuid4().hex[:12]}",
            source_agent_id=source_agent_id,
            target_agent_id=target_agent_id,
            topic="direct",
            content=message,
            timestamp=datetime.now(),
        )

        channel = f"{self.config.redis_prefix}agent:{target_agent_id}"

        if self._nats:
            await self._nats.publish(channel, json.dumps(msg.to_dict()).encode())
        elif self._redis:
            await self._redis.publish(channel, json.dumps(msg.to_dict()))
        else:
            # In-memory fallback
            handlers = self._subscriptions.get(f"agent:{target_agent_id}", [])
            for handler in handlers:
                try:
                    await self._invoke_handler(handler, msg.to_dict())
                except Exception as e:
                    logger.error(f"Handler error: {e}")

        logger.debug(f"Sent to {target_agent_id}: {msg.id}")

    async def subscribe(
        self,
        agent_id: str,
        topic: str,
        handler: Callable[[dict[str, Any]], Any],
    ) -> None:
        """
        Subscribe to messages on a topic.

        Args:
            agent_id: Subscribing agent ID
            topic: Topic to subscribe to
            handler: Callback for messages
        """
        await self._ensure_initialized()

        channel = f"{self.config.redis_prefix}channel:{topic}"

        if self._nats:

            async def nats_handler(msg: Any) -> None:
                data = json.loads(msg.data.decode())
                await self._invoke_handler(handler, data)

            await self._nats.subscribe(channel, cb=nats_handler)
        elif self._redis:
            pubsub = self._redis.pubsub()
            await pubsub.subscribe(channel)

            async def redis_reader() -> None:
                async for message in pubsub.listen():
                    if message["type"] == "message":
                        data = json.loads(message["data"])
                        await self._invoke_handler(handler, data)

            asyncio.create_task(redis_reader())
        else:
            # In-memory fallback
            if topic not in self._subscriptions:
                self._subscriptions[topic] = []
            self._subscriptions[topic].append(handler)

        logger.debug(f"Agent {agent_id} subscribed to {topic}")

    # =========================================================================
    # Topology State
    # =========================================================================

    async def get_topology(self) -> dict[str, Any]:
        """Get current topology state."""
        default_topology: dict[str, Any] = {"type": "swarm", "agents": []}
        value = await self.get("topology", default_topology)
        if isinstance(value, dict):
            return value
        return default_topology

    async def set_topology(self, topology: dict[str, Any]) -> None:
        """Set topology state."""
        await self.set("topology", topology)

    async def register_agent(
        self,
        agent_id: str,
        name: str,
        capabilities: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register an agent with the Hive Mind."""
        agent_info = {
            "id": agent_id,
            "name": name,
            "capabilities": capabilities,
            "metadata": metadata or {},
            "registered_at": datetime.now().isoformat(),
            "last_seen": datetime.now().isoformat(),
        }

        await self.set(f"agent:{agent_id}", agent_info, ttl=self.config.memory_ttl_seconds)
        self._agent_registry[agent_id] = agent_info
        logger.info(f"Agent registered: {agent_id}")

    async def get_agent_status(self, agent_id: str) -> dict[str, Any] | None:
        """Get status of a registered agent."""
        value = await self.get(f"agent:{agent_id}")
        if isinstance(value, dict):
            return value
        return None

    async def list_agents(self) -> list[dict[str, Any]]:
        """List all registered agents."""
        agents = []

        if self._redis:
            pattern = f"{self.config.redis_prefix}agent:*"
            keys = await self._redis.keys(pattern)
            if keys:
                from titan.metrics import get_metrics

                get_metrics().memory_mget()
                values = await self._redis.mget(keys)
                for data in values:
                    if data:
                        agents.append(json.loads(data))
        else:
            agents = list(self._agent_registry.values())

        return agents

    # =========================================================================
    # Utilities
    # =========================================================================

    async def _ensure_initialized(self) -> None:
        """Ensure Hive Mind is initialized."""
        if not self._initialized:
            await self.initialize()

    async def _invoke_handler(
        self,
        handler: Callable[[dict[str, Any]], Any],
        payload: dict[str, Any],
    ) -> None:
        """Invoke message handler, supporting sync and async callbacks."""
        result = handler(payload)
        if inspect.isawaitable(result):
            await result

    async def health_check(self) -> dict[str, Any]:
        """Check health of all components."""
        health = {
            "redis": False,
            "chromadb": False,
            "nats": False,
            "initialized": self._initialized,
        }

        if self._redis:
            try:
                await self._redis.ping()
                health["redis"] = True
            except Exception:
                pass

        if self._chroma_client:
            try:
                self._chroma_client.heartbeat()
                health["chromadb"] = True
            except Exception:
                pass

        if self._nats:
            health["nats"] = self._nats.is_connected

        return health

    def __repr__(self) -> str:
        return (
            f"<HiveMind initialized={self._initialized} "
            f"redis={self._redis is not None} "
            f"chroma={self._chroma_client is not None}>"
        )
