"""Topological Neighborhood for Agent Interactions.

Implements scale-free neighbor-based interactions inspired by
starling murmuration research. Each agent interacts with N=7
topological neighbors (based on relationship strength, not distance).

Key insight from murmuration research:
- Each bird interacts with ~6-7 nearest neighbors regardless of density
- This creates scale-free correlations across the flock
- Information can propagate through the entire group
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

logger = logging.getLogger("titan.hive.neighborhood")


# Default number of topological neighbors (from murmuration research)
DEFAULT_NEIGHBOR_COUNT = 7


class NeighborLayer(StrEnum):
    """Layers of topological coupling based on murmuration research.

    Multi-scale coupling allows systems to have:
    - Strong local interactions (primary)
    - Weaker medium-range correlations (secondary)
    - Rare global events (tertiary)
    """

    PRIMARY = "primary"  # 6-7 neighbors, strong coupling (main interactions)
    SECONDARY = "secondary"  # 20-30 neighbors, weak coupling (information spread)
    TERTIARY = "tertiary"  # Global, rare events only (emergency broadcast)


@dataclass
class LayeredNeighborConfig:
    """Configuration for multi-scale neighbor layers."""

    primary_count: int = 7  # N=7 from murmuration research
    secondary_count: int = 25  # Weaker, broader connections
    secondary_weight: float = 0.3  # Interaction strength for secondary
    tertiary_probability: float = 0.05  # Probability of tertiary connection


class InteractionType(StrEnum):
    """Types of agent-to-agent interactions."""

    COLLABORATION = "collaboration"  # Worked together on task
    COMMUNICATION = "communication"  # Exchanged messages
    DELEGATION = "delegation"  # Delegated task
    REVIEW = "review"  # Reviewed work
    ASSISTANCE = "assistance"  # Provided help
    CONFLICT = "conflict"  # Had disagreement


@dataclass
class InteractionRecord:
    """Record of an interaction between two agents."""

    agent_a: str
    agent_b: str
    interaction_type: InteractionType
    success: bool
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    weight: float = 1.0  # Interaction strength
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_a": self.agent_a,
            "agent_b": self.agent_b,
            "interaction_type": self.interaction_type.value,
            "success": self.success,
            "timestamp": self.timestamp.isoformat(),
            "weight": self.weight,
            "metadata": self.metadata,
        }


@dataclass
class AgentProfile:
    """Profile of an agent for neighbor calculation."""

    agent_id: str
    capabilities: list[str] = field(default_factory=list)
    current_task: str | None = None
    task_tags: list[str] = field(default_factory=list)
    performance_score: float = 1.0
    active: bool = True
    joined_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_seen: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    def capability_overlap(self, other: AgentProfile) -> float:
        """Calculate capability overlap with another agent (0-1)."""
        if not self.capabilities or not other.capabilities:
            return 0.0

        set_a = set(self.capabilities)
        set_b = set(other.capabilities)
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)

        return intersection / union if union > 0 else 0.0

    def task_similarity(self, other: AgentProfile) -> float:
        """Calculate task similarity with another agent (0-1)."""
        if not self.task_tags or not other.task_tags:
            return 0.0

        set_a = set(self.task_tags)
        set_b = set(other.task_tags)
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)

        return intersection / union if union > 0 else 0.0


@dataclass
class NeighborScore:
    """Score components for a potential neighbor."""

    agent_id: str
    total_score: float
    collaboration_score: float = 0.0
    capability_score: float = 0.0
    task_similarity_score: float = 0.0
    recency_score: float = 0.0
    success_rate: float = 0.0


class TopologicalNeighborhood:
    """Manages topological neighbor relationships for agents.

    Key features:
    - Each agent has N topological neighbors (default 7)
    - Neighbors are determined by relationship strength, not distance
    - Relationships are based on collaboration history, capability overlap,
      and task similarity
    - Information propagates through neighbor networks
    """

    def __init__(
        self,
        neighbor_count: int = DEFAULT_NEIGHBOR_COUNT,
        history_weight: float = 0.4,
        capability_weight: float = 0.3,
        task_weight: float = 0.3,
        recency_decay: float = 0.1,
        max_history: int = 1000,
        layer_config: LayeredNeighborConfig | None = None,
    ) -> None:
        """Initialize the neighborhood manager.

        Args:
            neighbor_count: Number of topological neighbors per agent.
            history_weight: Weight for collaboration history in scoring.
            capability_weight: Weight for capability overlap.
            task_weight: Weight for task similarity.
            recency_decay: Decay factor for old interactions.
            max_history: Maximum interaction records to keep.
            layer_config: Configuration for multi-scale neighbor layers.
        """
        self._neighbor_count = neighbor_count
        self._history_weight = history_weight
        self._capability_weight = capability_weight
        self._task_weight = task_weight
        self._recency_decay = recency_decay
        self._max_history = max_history
        self._layer_config = layer_config or LayeredNeighborConfig()

        # Agent profiles
        self._profiles: dict[str, AgentProfile] = {}

        # Interaction history
        self._interactions: list[InteractionRecord] = []

        # Cached neighbor lists (agent_id -> list of neighbor ids)
        self._neighbor_cache: dict[str, list[str]] = {}
        # Layered neighbor caches
        self._secondary_cache: dict[str, list[str]] = {}
        self._cache_valid = False

    def _invalidate_cache(self) -> None:
        self._neighbor_cache.clear()
        self._secondary_cache.clear()
        self._cache_valid = False

    @property
    def neighbor_count(self) -> int:
        """Number of neighbors per agent."""
        return self._neighbor_count

    def register_agent(
        self,
        agent_id: str,
        capabilities: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentProfile:
        """Register an agent in the neighborhood.

        Args:
            agent_id: Unique agent identifier.
            capabilities: List of agent capabilities.
            metadata: Optional metadata.

        Returns:
            The created AgentProfile.
        """
        profile = AgentProfile(
            agent_id=agent_id,
            capabilities=capabilities or [],
            metadata=metadata or {},
        )
        self._profiles[agent_id] = profile
        self._invalidate_cache()

        logger.debug(f"Registered agent in neighborhood: {agent_id}")
        return profile

    def unregister_agent(self, agent_id: str) -> bool:
        """Remove an agent from the neighborhood.

        Args:
            agent_id: Agent to remove.

        Returns:
            True if removed, False if not found.
        """
        if agent_id in self._profiles:
            del self._profiles[agent_id]
            self._invalidate_cache()
            return True
        return False

    def update_agent(
        self,
        agent_id: str,
        current_task: str | None = None,
        task_tags: list[str] | None = None,
        capabilities: list[str] | None = None,
    ) -> AgentProfile | None:
        """Update an agent's profile.

        Args:
            agent_id: Agent to update.
            current_task: Current task being worked on.
            task_tags: Tags describing current task.
            capabilities: Updated capabilities.

        Returns:
            Updated profile, or None if not found.
        """
        profile = self._profiles.get(agent_id)
        if not profile:
            return None

        if current_task is not None:
            profile.current_task = current_task
        if task_tags is not None:
            profile.task_tags = task_tags
        if capabilities is not None:
            profile.capabilities = capabilities

        profile.last_seen = datetime.now(UTC)
        self._invalidate_cache()

        return profile

    def record_interaction(
        self,
        agent_a: str,
        agent_b: str,
        interaction_type: InteractionType,
        success: bool,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> InteractionRecord:
        """Record an interaction between two agents.

        Args:
            agent_a: First agent.
            agent_b: Second agent.
            interaction_type: Type of interaction.
            success: Whether the interaction was successful.
            weight: Interaction strength.
            metadata: Additional metadata.

        Returns:
            The created InteractionRecord.
        """
        record = InteractionRecord(
            agent_a=agent_a,
            agent_b=agent_b,
            interaction_type=interaction_type,
            success=success,
            weight=weight,
            metadata=metadata or {},
        )

        self._interactions.append(record)

        # Trim history if needed
        if len(self._interactions) > self._max_history:
            self._interactions = self._interactions[-self._max_history :]

        self._invalidate_cache()

        logger.debug(
            f"Recorded interaction: {agent_a} <-> {agent_b} "
            f"({interaction_type.value}, success={success})"
        )

        return record

    def get_neighbors(
        self,
        agent_id: str,
        layer: NeighborLayer = NeighborLayer.PRIMARY,
        force_recalculate: bool = False,
    ) -> list[str]:
        """Get the topological neighbors for an agent.

        Multi-scale coupling based on murmuration research:
        - PRIMARY: 6-7 neighbors, strong coupling
        - SECONDARY: 20-30 neighbors, weak coupling
        - TERTIARY: Global, rare events only

        Args:
            agent_id: Agent to get neighbors for.
            layer: Which neighbor layer to retrieve.
            force_recalculate: Whether to force recalculation.

        Returns:
            List of neighbor agent IDs.
        """
        if layer == NeighborLayer.PRIMARY:
            if not force_recalculate and self._cache_valid and agent_id in self._neighbor_cache:
                return self._neighbor_cache[agent_id]

            neighbors = self.calculate_neighbors(agent_id)
            self._neighbor_cache[agent_id] = neighbors
            self._cache_valid = True
            return neighbors

        elif layer == NeighborLayer.SECONDARY:
            if not force_recalculate and self._cache_valid and agent_id in self._secondary_cache:
                return self._secondary_cache[agent_id]

            # Secondary layer: more neighbors with weaker connections
            neighbors = self._calculate_secondary_neighbors(agent_id)
            self._secondary_cache[agent_id] = neighbors
            self._cache_valid = True
            return neighbors

        elif layer == NeighborLayer.TERTIARY:
            # Tertiary layer: all agents with probability filter
            return self._get_tertiary_neighbors(agent_id)

        return []

    def _calculate_secondary_neighbors(self, agent_id: str) -> list[str]:
        """Calculate secondary (weak) neighbors for an agent.

        Secondary neighbors are more numerous but with weaker coupling.
        """
        profile = self._profiles.get(agent_id)
        if not profile:
            return []

        scores: list[NeighborScore] = []

        for other_id, other_profile in self._profiles.items():
            if other_id == agent_id or not other_profile.active:
                continue

            # Calculate scores (same as primary but take more)
            collab_score = self._calculate_collaboration_score(agent_id, other_id)
            cap_score = profile.capability_overlap(other_profile)
            task_score = profile.task_similarity(other_profile)

            total = (
                self._history_weight * collab_score
                + self._capability_weight * cap_score
                + self._task_weight * task_score
            )

            scores.append(
                NeighborScore(
                    agent_id=other_id,
                    total_score=total,
                    collaboration_score=collab_score,
                    capability_score=cap_score,
                    task_similarity_score=task_score,
                )
            )

        # Sort by score and take top N (secondary count)
        scores.sort(key=lambda x: x.total_score, reverse=True)
        neighbors = [s.agent_id for s in scores[: self._layer_config.secondary_count]]

        return neighbors

    def _get_tertiary_neighbors(self, agent_id: str) -> list[str]:
        """Get tertiary (global, rare) neighbors.

        Tertiary connections are used for emergency broadcasts or
        rare global events. Selection is probabilistic.
        """
        all_agents = [aid for aid, p in self._profiles.items() if aid != agent_id and p.active]

        # Probabilistic selection
        selected = [
            aid for aid in all_agents if random.random() < self._layer_config.tertiary_probability
        ]

        return selected

    def calculate_neighbors(self, agent_id: str) -> list[str]:
        """Calculate topological neighbors for an agent.

        Neighbors are determined by:
        1. Collaboration history (weighted by success and recency)
        2. Capability overlap
        3. Task similarity

        Args:
            agent_id: Agent to calculate neighbors for.

        Returns:
            List of neighbor agent IDs, ordered by score.
        """
        profile = self._profiles.get(agent_id)
        if not profile:
            return []

        scores: list[NeighborScore] = []

        for other_id, other_profile in self._profiles.items():
            if other_id == agent_id or not other_profile.active:
                continue

            # Calculate collaboration score from history
            collab_score = self._calculate_collaboration_score(agent_id, other_id)

            # Calculate capability overlap
            cap_score = profile.capability_overlap(other_profile)

            # Calculate task similarity
            task_score = profile.task_similarity(other_profile)

            # Combined score
            total = (
                self._history_weight * collab_score
                + self._capability_weight * cap_score
                + self._task_weight * task_score
            )

            scores.append(
                NeighborScore(
                    agent_id=other_id,
                    total_score=total,
                    collaboration_score=collab_score,
                    capability_score=cap_score,
                    task_similarity_score=task_score,
                )
            )

        # Sort by total score and take top N
        scores.sort(key=lambda x: x.total_score, reverse=True)
        neighbors = [s.agent_id for s in scores[: self._neighbor_count]]

        return neighbors

    def _calculate_collaboration_score(
        self,
        agent_a: str,
        agent_b: str,
    ) -> float:
        """Calculate collaboration score from interaction history."""
        now = datetime.now(UTC)
        total_score = 0.0
        interaction_count = 0

        for record in self._interactions:
            if not (
                (record.agent_a == agent_a and record.agent_b == agent_b)
                or (record.agent_a == agent_b and record.agent_b == agent_a)
            ):
                continue

            # Apply recency decay
            age_days = (now - record.timestamp).total_seconds() / 86400
            recency_factor = max(0.1, 1.0 - (self._recency_decay * age_days))

            # Success bonus
            success_factor = 1.5 if record.success else 0.5

            # Weight by interaction type
            type_weight = {
                InteractionType.COLLABORATION: 1.0,
                InteractionType.COMMUNICATION: 0.5,
                InteractionType.DELEGATION: 0.8,
                InteractionType.REVIEW: 0.7,
                InteractionType.ASSISTANCE: 0.9,
                InteractionType.CONFLICT: -0.5,
            }.get(record.interaction_type, 0.5)

            total_score += recency_factor * success_factor * type_weight * record.weight
            interaction_count += 1

        # Normalize to 0-1 range
        if interaction_count == 0:
            return 0.0

        return min(1.0, total_score / (interaction_count * 2))

    async def propagate_information(
        self,
        source_agent: str,
        information: dict[str, Any],
        max_hops: int = 3,
        decay_factor: float = 0.7,
    ) -> dict[str, float]:
        """Propagate information through the neighbor network.

        Information spreads from source through neighbors, decaying
        at each hop. This models how information flows through
        scale-free networks like murmurations.

        Args:
            source_agent: Agent originating the information.
            information: Data to propagate.
            max_hops: Maximum number of hops.
            decay_factor: Information strength decay per hop.

        Returns:
            Dict mapping agent_id -> information strength received.
        """
        reached: dict[str, float] = {source_agent: 1.0}
        frontier = [(source_agent, 1.0, 0)]  # (agent, strength, hop)

        while frontier:
            current, strength, hop = frontier.pop(0)

            if hop >= max_hops:
                continue

            next_strength = strength * decay_factor
            if next_strength < 0.01:
                continue

            neighbors = self.get_neighbors(current)
            for neighbor in neighbors:
                if neighbor not in reached or reached[neighbor] < next_strength:
                    reached[neighbor] = next_strength
                    frontier.append((neighbor, next_strength, hop + 1))

        logger.debug(
            f"Propagated information from {source_agent}: "
            f"reached {len(reached)} agents in {max_hops} hops"
        )

        return reached

    async def propagate_information_layered(
        self,
        source_agent: str,
        information: dict[str, Any],
        use_layers: list[NeighborLayer] | None = None,
        perturbation_magnitude: float = 1.0,
        max_hops: int = 3,
        decay_factor: float = 0.7,
    ) -> dict[str, float]:
        """Propagate information through multi-scale neighbor network.

        Uses layered neighbor structure for multi-scale information spread:
        - Primary layer: Strong, local propagation
        - Secondary layer: Weaker, broader propagation
        - Tertiary layer: Rare global events

        Based on murmuration research showing scale-free correlations
        emerge from topological (not metric) coupling.

        Args:
            source_agent: Agent originating the information.
            information: Data to propagate.
            use_layers: Which layers to use (default: all).
            perturbation_magnitude: Strength of the perturbation (for tracking).
            max_hops: Maximum hops for primary layer.
            decay_factor: Decay per hop for primary layer.

        Returns:
            Dict mapping agent_id -> information strength received.
        """
        if use_layers is None:
            use_layers = [NeighborLayer.PRIMARY, NeighborLayer.SECONDARY]

        reached: dict[str, float] = {source_agent: 1.0}

        # Primary layer propagation (standard BFS with decay)
        if NeighborLayer.PRIMARY in use_layers:
            frontier = [(source_agent, 1.0, 0)]  # (agent, strength, hop)

            while frontier:
                current, strength, hop = frontier.pop(0)

                if hop >= max_hops:
                    continue

                next_strength = strength * decay_factor
                if next_strength < 0.01:
                    continue

                neighbors = self.get_neighbors(current, NeighborLayer.PRIMARY)
                for neighbor in neighbors:
                    if neighbor not in reached or reached[neighbor] < next_strength:
                        reached[neighbor] = next_strength
                        frontier.append((neighbor, next_strength, hop + 1))

        # Secondary layer propagation (direct, weaker)
        if NeighborLayer.SECONDARY in use_layers:
            secondary_strength = self._layer_config.secondary_weight
            secondary_neighbors = self.get_neighbors(source_agent, NeighborLayer.SECONDARY)

            for neighbor in secondary_neighbors:
                if neighbor not in reached:
                    reached[neighbor] = secondary_strength
                else:
                    # Combine strengths (don't exceed 1.0)
                    reached[neighbor] = min(1.0, reached[neighbor] + secondary_strength * 0.5)

        # Tertiary layer propagation (rare, global)
        if NeighborLayer.TERTIARY in use_layers:
            tertiary_neighbors = self.get_neighbors(source_agent, NeighborLayer.TERTIARY)
            tertiary_strength = self._layer_config.tertiary_probability

            for neighbor in tertiary_neighbors:
                if neighbor not in reached:
                    reached[neighbor] = tertiary_strength
                else:
                    reached[neighbor] = min(1.0, reached[neighbor] + tertiary_strength * 0.3)

        logger.debug(
            f"Layered propagation from {source_agent}: "
            f"reached {len(reached)} agents using layers "
            f"{[layer.value for layer in use_layers]}"
        )

        return reached

    def get_network_stats(self) -> dict[str, Any]:
        """Get statistics about the neighbor network.

        Returns:
            Dictionary with network statistics.
        """
        active_agents = [a for a in self._profiles.values() if a.active]
        total_agents = len(active_agents)

        if total_agents == 0:
            return {
                "total_agents": 0,
                "neighbor_count": self._neighbor_count,
            }

        # Calculate network density
        total_edges = 0
        for agent in active_agents:
            neighbors = self.get_neighbors(agent.agent_id)
            total_edges += len(neighbors)

        max_edges = total_agents * (total_agents - 1)
        density = total_edges / max_edges if max_edges > 0 else 0.0

        # Calculate average clustering coefficient
        clustering_sum = 0.0
        for agent in active_agents:
            neighbor_set = set(self.get_neighbors(agent.agent_id))
            if len(neighbor_set) < 2:
                continue

            # Count connections between neighbors
            neighbor_connections = 0
            for n1 in neighbor_set:
                n1_neighbors = set(self.get_neighbors(n1))
                neighbor_connections += len(neighbor_set & n1_neighbors)

            possible = len(neighbor_set) * (len(neighbor_set) - 1)
            if possible > 0:
                clustering_sum += neighbor_connections / possible

        avg_clustering = clustering_sum / total_agents if total_agents > 0 else 0.0

        return {
            "total_agents": total_agents,
            "neighbor_count": self._neighbor_count,
            "total_edges": total_edges,
            "density": density,
            "average_clustering": avg_clustering,
            "total_interactions": len(self._interactions),
        }

    def get_most_connected_agents(self, limit: int = 10) -> list[tuple[str, int]]:
        """Get agents with the most neighbor connections.

        Args:
            limit: Maximum number of agents to return.

        Returns:
            List of (agent_id, connection_count) tuples.
        """
        connection_counts: dict[str, int] = defaultdict(int)

        for agent_id in self._profiles:
            neighbors = self.get_neighbors(agent_id)
            connection_counts[agent_id] = len(neighbors)

            # Also count reverse connections
            for neighbor in neighbors:
                if agent_id in self.get_neighbors(neighbor):
                    connection_counts[agent_id] += 1

        sorted_agents = sorted(
            connection_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return sorted_agents[:limit]

    def find_bridges(self) -> list[str]:
        """Find agents that act as bridges between clusters.

        Bridge agents connect otherwise disconnected groups.

        Returns:
            List of bridge agent IDs.
        """
        bridges: list[str] = []

        for agent_id, profile in self._profiles.items():
            if not profile.active:
                continue

            neighbors = set(self.get_neighbors(agent_id))
            if len(neighbors) < 2:
                continue

            # Check if removing this agent would disconnect neighbors
            connections_between_neighbors = 0
            for n1 in neighbors:
                n1_neighbors = set(self.get_neighbors(n1))
                connections_between_neighbors += len(neighbors & n1_neighbors)

            # Low connectivity between neighbors indicates bridge
            max_connections = len(neighbors) * (len(neighbors) - 1)
            if max_connections > 0:
                connectivity = connections_between_neighbors / max_connections
                if connectivity < 0.3:  # Threshold for bridge detection
                    bridges.append(agent_id)

        return bridges

    def invalidate_cache(self) -> None:
        """Invalidate the neighbor cache."""
        self._cache_valid = False
        self._neighbor_cache.clear()

    def to_dict(self) -> dict[str, Any]:
        """Serialize state to dictionary."""
        return {
            "neighbor_count": self._neighbor_count,
            "history_weight": self._history_weight,
            "capability_weight": self._capability_weight,
            "task_weight": self._task_weight,
            "recency_decay": self._recency_decay,
            "profiles": {
                agent_id: {
                    "agent_id": p.agent_id,
                    "capabilities": p.capabilities,
                    "current_task": p.current_task,
                    "task_tags": p.task_tags,
                    "performance_score": p.performance_score,
                    "active": p.active,
                }
                for agent_id, p in self._profiles.items()
            },
            "interactions": [r.to_dict() for r in self._interactions[-100:]],
        }
