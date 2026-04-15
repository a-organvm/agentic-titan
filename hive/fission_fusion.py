"""Fission-Fusion Dynamics for Agent Coordination.

Implements crow roost-inspired dynamics where the swarm switches between
sparse (fission) and dense (fusion) topologies based on task correlation
and coordination needs.

Key concepts:
- Fission: Sparse, independent clusters for parallel exploration
- Fusion: Dense, information-sharing mode for complex coordination
- Information centers: Hub nodes that aggregate and broadcast patterns

Based on research on:
- Crow roost dynamics and information centers
- Fission-fusion social systems (dolphins, elephants, primates)
- Collective decision-making in animal groups
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, cast

from titan.metrics import get_metrics

if TYPE_CHECKING:
    from hive.criticality import CriticalityMonitor
    from hive.events import EventBus
    from hive.neighborhood import TopologicalNeighborhood
    from hive.stigmergy import PheromoneField

logger = logging.getLogger("titan.hive.fission_fusion")


class MetricsSink(Protocol):
    """Typed metrics surface used by fission-fusion manager."""

    def fission_event(self) -> None: ...

    def fusion_event(self) -> None: ...

    def set_cluster_count(self, count: int) -> None: ...

    def set_fission_fusion_state(self, state: str) -> None: ...


class FissionFusionState(StrEnum):
    """State of fission-fusion dynamics."""

    FISSION = "fission"  # Sparse, independent clusters
    FUSION = "fusion"  # Dense, information-sharing mode
    TRANSITIONING = "transitioning"  # Moving between states


@dataclass
class Cluster:
    """A cluster of agents during fission state."""

    cluster_id: str
    agent_ids: list[str] = field(default_factory=list)
    task_focus: str | None = None
    centroid_agent: str | None = None  # Most central agent
    formation_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_agent(self, agent_id: str) -> None:
        """Add an agent to the cluster."""
        if agent_id not in self.agent_ids:
            self.agent_ids.append(agent_id)

    def remove_agent(self, agent_id: str) -> None:
        """Remove an agent from the cluster."""
        if agent_id in self.agent_ids:
            self.agent_ids.remove(agent_id)

    @property
    def size(self) -> int:
        """Number of agents in cluster."""
        return len(self.agent_ids)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cluster_id": self.cluster_id,
            "agent_ids": self.agent_ids,
            "task_focus": self.task_focus,
            "centroid_agent": self.centroid_agent,
            "formation_time": self.formation_time.isoformat(),
            "size": self.size,
            "metadata": self.metadata,
        }


@dataclass
class FissionFusionMetrics:
    """Metrics for evaluating fission-fusion state."""

    task_correlation: float = 0.5  # 0-1, how correlated current tasks are
    information_spread: float = 0.5  # 0-1, how well information is propagating
    cluster_count: int = 1  # Number of clusters
    avg_cluster_size: float = 0.0  # Average cluster size
    cohesion: float = 0.5  # 0-1, overall group cohesion
    crisis_level: float = 0.0  # 0-1, urgency requiring fusion
    exploration_need: float = 0.0  # 0-1, need for parallel exploration
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def suggest_state(self) -> FissionFusionState:
        """Suggest optimal state based on metrics."""
        # Fusion triggers:
        # - High task correlation (need coordination)
        # - Crisis (need collective response)
        # - Low information spread (need better communication)
        if self.task_correlation > 0.7 or self.crisis_level > 0.6:
            return FissionFusionState.FUSION

        # Fission triggers:
        # - Low task correlation (independent work)
        # - High exploration need
        # - Already good information spread
        if self.task_correlation < 0.3 or self.exploration_need > 0.6:
            return FissionFusionState.FISSION

        # Transitioning zone
        return FissionFusionState.TRANSITIONING


@dataclass
class FissionFusionEvent:
    """Record of a fission-fusion state change."""

    event_id: str
    event_type: str  # "fission" or "fusion"
    previous_state: FissionFusionState
    new_state: FissionFusionState
    metrics: FissionFusionMetrics
    clusters_formed: int = 0
    info_center_id: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "previous_state": self.previous_state.value,
            "new_state": self.new_state.value,
            "clusters_formed": self.clusters_formed,
            "info_center_id": self.info_center_id,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class FissionFusionManager:
    """Manages fission-fusion dynamics for the agent swarm.

    Implements crow roost-inspired dynamics:
    - During FISSION: Agents split into independent clusters for exploration
    - During FUSION: Agents gather at information centers for coordination

    The manager monitors task correlation and other metrics to determine
    when to switch between states.
    """

    # Thresholds for state transitions
    FISSION_THRESHOLD = 0.3  # Low task correlation -> fission
    FUSION_THRESHOLD = 0.7  # High task correlation -> fusion
    TRANSITION_HYSTERESIS = 0.1  # Prevent rapid oscillation

    # Cluster parameters
    MIN_CLUSTER_SIZE = 2
    MAX_CLUSTER_SIZE = 20
    DEFAULT_CLUSTER_COUNT = 5

    def __init__(
        self,
        neighborhood: TopologicalNeighborhood | None = None,
        pheromone_field: PheromoneField | None = None,
        event_bus: EventBus | None = None,
        evaluation_interval: float = 30.0,
        criticality_monitor: CriticalityMonitor | None = None,
    ) -> None:
        """Initialize the fission-fusion manager.

        Args:
            neighborhood: Topological neighborhood for clustering.
            pheromone_field: Pheromone field for stigmergic coordination.
            event_bus: Event bus for publishing events.
            evaluation_interval: Seconds between state evaluations.
            criticality_monitor: Optional monitor for deriving crisis_level from
                criticality state (supercritical → high crisis).
        """
        self._neighborhood = neighborhood
        self._pheromone_field = pheromone_field
        self._event_bus = event_bus
        self._evaluation_interval = evaluation_interval
        self._criticality_monitor = criticality_monitor

        self._state = FissionFusionState.FISSION  # Start distributed
        self._metrics = FissionFusionMetrics()

        # Clusters (active during fission)
        self._clusters: dict[str, Cluster] = {}
        self._agent_cluster: dict[str, str] = {}  # agent_id -> cluster_id

        # Information center (active during fusion)
        self._info_center_id: str | None = None

        # History
        self._events: list[FissionFusionEvent] = []
        self._event_counter = 0

        # Background task
        self._eval_task: asyncio.Task[None] | None = None
        self._running = False

        # Post-transition cooldown (refractory period)
        self._last_transition_time: datetime | None = None
        self._cooldown_seconds = evaluation_interval * 2  # Default: 2× eval interval

        # Metrics history for windowed evaluation (5-cycle window)
        self._metrics_history: list[FissionFusionMetrics] = []
        self._window_size = 5
        self._majority_threshold = 3  # 3-of-5 must agree for transition

        # Manual crisis override (set via set_crisis_level)
        self._manual_crisis_level: float | None = None

        # Callbacks
        self._on_state_change: list[Callable[[FissionFusionState], None]] = []

    @property
    def state(self) -> FissionFusionState:
        """Get current fission-fusion state."""
        return self._state

    @property
    def metrics(self) -> FissionFusionMetrics:
        """Get current metrics."""
        return self._metrics

    @property
    def clusters(self) -> list[Cluster]:
        """Get current clusters."""
        return list(self._clusters.values())

    async def start(self) -> None:
        """Start the fission-fusion manager."""
        if self._running:
            return

        self._running = True
        self._eval_task = asyncio.create_task(self._evaluation_loop())
        logger.info("Started fission-fusion manager")

    async def stop(self) -> None:
        """Stop the fission-fusion manager."""
        self._running = False
        if self._eval_task:
            self._eval_task.cancel()
            try:
                await self._eval_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped fission-fusion manager")

    async def _evaluation_loop(self) -> None:
        """Background loop for state evaluation."""
        while self._running:
            try:
                await asyncio.sleep(self._evaluation_interval)
                await self._evaluate_and_respond()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in fission-fusion evaluation: {e}")

    async def _evaluate_and_respond(self) -> None:
        """Evaluate state and respond if needed."""
        self._metrics = await self.evaluate_state()

        # Record metrics for windowed evaluation
        self._metrics_history.append(self._metrics)
        if len(self._metrics_history) > self._window_size:
            self._metrics_history = self._metrics_history[-self._window_size :]

        suggested = await self.should_transition()
        if suggested and suggested != self._state:
            if suggested == FissionFusionState.FISSION:
                await self.perform_fission()
            elif suggested == FissionFusionState.FUSION:
                await self.perform_fusion()

    async def evaluate_state(self) -> FissionFusionMetrics:
        """Evaluate current fission-fusion metrics.

        Returns:
            Updated FissionFusionMetrics.
        """
        # Calculate task correlation from neighborhood
        task_correlation = await self._calculate_task_correlation()

        # Calculate information spread
        information_spread = await self._calculate_information_spread()

        # Calculate cluster metrics
        cluster_count = len(self._clusters)
        avg_cluster_size = 0.0
        if cluster_count > 0:
            sizes = [c.size for c in self._clusters.values()]
            avg_cluster_size = statistics.mean(sizes)

        # Calculate cohesion
        cohesion = await self._calculate_cohesion()

        # Crisis level: manual override > criticality-derived > preserved from last cycle
        crisis_level = self._resolve_crisis_level()
        exploration_need = 1.0 - task_correlation  # Inverse of correlation

        return FissionFusionMetrics(
            task_correlation=task_correlation,
            information_spread=information_spread,
            cluster_count=cluster_count,
            avg_cluster_size=avg_cluster_size,
            cohesion=cohesion,
            crisis_level=crisis_level,
            exploration_need=exploration_need,
        )

    async def _calculate_task_correlation(self) -> float:
        """Calculate how correlated current tasks are across agents."""
        if not self._neighborhood:
            return 0.5

        profiles = list(self._neighborhood._profiles.values())
        if len(profiles) < 2:
            return 0.5

        # Calculate pairwise task similarity
        similarities: list[float] = []
        for i, p1 in enumerate(profiles):
            for p2 in profiles[i + 1 :]:
                sim = p1.task_similarity(p2)
                similarities.append(sim)

        if not similarities:
            return 0.5

        return statistics.mean(similarities)

    async def _calculate_information_spread(self) -> float:
        """Calculate how well information is spreading."""
        if not self._neighborhood:
            return 0.5

        stats = self._neighborhood.get_network_stats()
        density = float(stats.get("density", 0.5))
        avg_clustering = float(stats.get("average_clustering", 0.5))

        # Good information spread = high density + high clustering
        return (density + avg_clustering) / 2

    async def _calculate_cohesion(self) -> float:
        """Calculate overall group cohesion."""
        if not self._neighborhood:
            return 0.5

        stats = self._neighborhood.get_network_stats()
        return float(stats.get("average_clustering", 0.5))

    def _resolve_crisis_level(self) -> float:
        """Resolve crisis_level from available signal sources.

        Priority: manual override > criticality-derived > preserved value.
        """
        # 1. Manual override (set via set_crisis_level)
        if self._manual_crisis_level is not None:
            return self._manual_crisis_level

        # 2. Derive from criticality monitor if available
        if self._criticality_monitor is not None:
            from hive.criticality import CriticalityState

            state = self._criticality_monitor.current_state
            metrics = self._criticality_monitor.current_metrics

            if state == CriticalityState.SUPERCRITICAL:
                # Chaotic system → high crisis → push toward FUSION
                return min(1.0, 0.5 + metrics.criticality_score * 0.5)
            elif state == CriticalityState.SUBCRITICAL:
                # Rigid system → low crisis
                return max(0.0, 0.2 * metrics.criticality_score)
            else:
                # Critical (optimal) → moderate baseline
                return 0.1

        # 3. Preserve current value from last cycle
        return self._metrics.crisis_level

    async def should_transition(self) -> FissionFusionState | None:
        """Determine if state transition should occur.

        Uses a windowed majority gate: the suggested state must be
        consistent across at least 3 of the last 5 evaluation cycles.
        This prevents transient metric spikes from triggering premature
        transitions.

        Returns:
            Target state, or None if no transition needed.
        """
        # Cooldown gate: suppress transitions during refractory period
        if self._last_transition_time is not None:
            elapsed = (datetime.now(UTC) - self._last_transition_time).total_seconds()
            if elapsed < self._cooldown_seconds:
                return None

        if self._state == FissionFusionState.TRANSITIONING:
            # Complete current transition
            return self._metrics.suggest_state()

        suggested = self._metrics.suggest_state()

        # Apply hysteresis
        if suggested == self._state:
            return None

        # Windowed majority gate: need majority of recent evaluations
        # to agree on the transition target (3-of-5 default)
        if len(self._metrics_history) >= self._majority_threshold:
            window = self._metrics_history[-self._window_size :]
            votes = [m.suggest_state() for m in window]
            agree_count = sum(1 for v in votes if v == suggested)
            if agree_count < self._majority_threshold:
                return None

        # Check thresholds with hysteresis
        if self._state == FissionFusionState.FISSION:
            if self._metrics.task_correlation > self.FUSION_THRESHOLD:
                return FissionFusionState.FUSION
        elif self._state == FissionFusionState.FUSION:
            if self._metrics.task_correlation < self.FISSION_THRESHOLD:
                return FissionFusionState.FISSION

        return None

    async def perform_fission(self) -> int:
        """Perform fission: split into independent clusters.

        Returns:
            Number of clusters formed.
        """
        old_state = self._state
        self._state = FissionFusionState.TRANSITIONING

        # Clear existing clusters
        self._clusters.clear()
        self._agent_cluster.clear()

        # Get all agents
        if not self._neighborhood:
            self._state = FissionFusionState.FISSION
            self._last_transition_time = datetime.now(UTC)
            return 0

        agents = list(self._neighborhood._profiles.keys())
        if not agents:
            self._state = FissionFusionState.FISSION
            self._last_transition_time = datetime.now(UTC)
            return 0

        # Cluster based on task similarity
        clusters_formed = await self._form_clusters(agents)

        # Clear info center
        self._info_center_id = None

        self._state = FissionFusionState.FISSION

        # Record event
        self._event_counter += 1
        event = FissionFusionEvent(
            event_id=f"ff_event_{self._event_counter}",
            event_type="fission",
            previous_state=old_state,
            new_state=FissionFusionState.FISSION,
            metrics=self._metrics,
            clusters_formed=clusters_formed,
        )
        self._events.append(event)

        # Emit event
        if self._event_bus:
            from hive.events import EventType

            await self._event_bus.emit(
                EventType.FISSION_COMPLETED,
                {
                    "clusters_formed": clusters_formed,
                    "cluster_ids": list(self._clusters.keys()),
                },
                source_id="fission_fusion_manager",
            )

        # Record metrics
        typed_get_metrics = cast(Callable[[], MetricsSink], get_metrics)
        metrics = typed_get_metrics()
        metrics.fission_event()
        metrics.set_cluster_count(clusters_formed)
        metrics.set_fission_fusion_state(FissionFusionState.FISSION.value)

        self._last_transition_time = datetime.now(UTC)
        logger.info(f"Fission completed: {clusters_formed} clusters formed")

        # Notify callbacks
        for callback in self._on_state_change:
            try:
                callback(FissionFusionState.FISSION)
            except Exception as e:
                logger.error(f"State change callback error: {e}")

        return clusters_formed

    async def _form_clusters(self, agents: list[str]) -> int:
        """Form clusters from agents based on task similarity.

        Uses a simple greedy clustering algorithm.
        """
        neighborhood = self._neighborhood
        if neighborhood is None:
            return 0

        if not agents:
            return 0

        unassigned = set(agents)
        cluster_count = min(
            self.DEFAULT_CLUSTER_COUNT,
            len(agents) // self.MIN_CLUSTER_SIZE,
        )

        if cluster_count < 1:
            cluster_count = 1

        # Initialize clusters with seed agents
        import random

        seeds = random.sample(list(unassigned), min(cluster_count, len(unassigned)))

        for seed in seeds:
            cluster_id = f"cluster_{uuid.uuid4().hex[:8]}"
            cluster = Cluster(
                cluster_id=cluster_id,
                agent_ids=[seed],
                centroid_agent=seed,
            )
            self._clusters[cluster_id] = cluster
            self._agent_cluster[seed] = cluster_id
            unassigned.remove(seed)

        # Assign remaining agents to nearest cluster
        for agent_id in list(unassigned):
            best_cluster = None
            best_score = -1.0

            profile = neighborhood._profiles.get(agent_id)
            if not profile:
                continue

            for cluster in self._clusters.values():
                if cluster.size >= self.MAX_CLUSTER_SIZE:
                    continue

                # Calculate similarity to cluster centroid
                if cluster.centroid_agent is None:
                    continue
                centroid_profile = neighborhood._profiles.get(cluster.centroid_agent)
                if centroid_profile:
                    score = profile.task_similarity(centroid_profile)
                    if score > best_score:
                        best_score = score
                        best_cluster = cluster

            if best_cluster:
                best_cluster.add_agent(agent_id)
                self._agent_cluster[agent_id] = best_cluster.cluster_id
                unassigned.remove(agent_id)

        # Handle any remaining unassigned (put in smallest cluster)
        for agent_id in unassigned:
            smallest = min(self._clusters.values(), key=lambda c: c.size)
            smallest.add_agent(agent_id)
            self._agent_cluster[agent_id] = smallest.cluster_id

        return len(self._clusters)

    async def perform_fusion(self) -> str:
        """Perform fusion: gather at information center.

        Returns:
            Information center ID.
        """
        old_state = self._state
        self._state = FissionFusionState.TRANSITIONING

        # Create information center
        self._info_center_id = f"info_center_{uuid.uuid4().hex[:8]}"

        # Clear clusters
        self._clusters.clear()
        self._agent_cluster.clear()

        self._state = FissionFusionState.FUSION

        # Record event
        self._event_counter += 1
        event = FissionFusionEvent(
            event_id=f"ff_event_{self._event_counter}",
            event_type="fusion",
            previous_state=old_state,
            new_state=FissionFusionState.FUSION,
            metrics=self._metrics,
            info_center_id=self._info_center_id,
        )
        self._events.append(event)

        # Emit event
        if self._event_bus:
            from hive.events import EventType

            await self._event_bus.emit(
                EventType.FUSION_COMPLETED,
                {
                    "info_center_id": self._info_center_id,
                },
                source_id="fission_fusion_manager",
            )

        # Record metrics
        typed_get_metrics = cast(Callable[[], MetricsSink], get_metrics)
        metrics = typed_get_metrics()
        metrics.fusion_event()
        metrics.set_cluster_count(0)
        metrics.set_fission_fusion_state(FissionFusionState.FUSION.value)

        self._last_transition_time = datetime.now(UTC)
        logger.info(f"Fusion completed: info center {self._info_center_id}")

        # Notify callbacks
        for callback in self._on_state_change:
            try:
                callback(FissionFusionState.FUSION)
            except Exception as e:
                logger.error(f"State change callback error: {e}")

        assert self._info_center_id is not None
        return self._info_center_id

    def get_agent_cluster(self, agent_id: str) -> Cluster | None:
        """Get the cluster an agent belongs to."""
        cluster_id = self._agent_cluster.get(agent_id)
        if cluster_id:
            return self._clusters.get(cluster_id)
        return None

    def set_crisis_level(self, level: float) -> None:
        """Set the crisis level (triggers fusion if high).

        This value persists across evaluation cycles until cleared
        with clear_crisis_override().
        """
        clamped = max(0.0, min(1.0, level))
        self._manual_crisis_level = clamped
        self._metrics.crisis_level = clamped

    def clear_crisis_override(self) -> None:
        """Clear manual crisis override, returning to signal-derived values."""
        self._manual_crisis_level = None

    def set_exploration_need(self, level: float) -> None:
        """Set the exploration need (triggers fission if high)."""
        self._metrics.exploration_need = max(0.0, min(1.0, level))

    def on_state_change(self, callback: Callable[[FissionFusionState], None]) -> None:
        """Register a callback for state changes."""
        self._on_state_change.append(callback)

    def get_events(self, limit: int = 20) -> list[FissionFusionEvent]:
        """Get recent fission-fusion events."""
        return self._events[-limit:]

    def to_dict(self) -> dict[str, Any]:
        """Serialize state to dictionary."""
        return {
            "state": self._state.value,
            "metrics": {
                "task_correlation": self._metrics.task_correlation,
                "information_spread": self._metrics.information_spread,
                "cluster_count": self._metrics.cluster_count,
                "avg_cluster_size": self._metrics.avg_cluster_size,
                "cohesion": self._metrics.cohesion,
                "crisis_level": self._metrics.crisis_level,
                "exploration_need": self._metrics.exploration_need,
            },
            "clusters": [c.to_dict() for c in self._clusters.values()],
            "info_center_id": self._info_center_id,
            "events_count": len(self._events),
        }
