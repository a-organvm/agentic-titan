"""Criticality Detection and Phase Transition Monitoring.

Implements monitoring for criticality in agent systems based on
statistical physics concepts. Systems operating at the "edge of chaos"
are maximally sensitive while maintaining coherence.

Key concepts:
- Criticality: State where system exhibits power-law correlations
- Phase transitions: Abrupt changes in system organization
- Correlation length: How far influence propagates
- Susceptibility: Response to perturbations

Based on research in:
- Statistical mechanics of phase transitions
- Self-organized criticality
- Starling murmuration physics
"""

from __future__ import annotations

import asyncio
import logging
import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from titan.metrics import get_metrics

if TYPE_CHECKING:
    from hive.events import EventBus
    from hive.neighborhood import TopologicalNeighborhood

logger = logging.getLogger("titan.hive.criticality")


class CriticalityState(StrEnum):
    """States of system criticality."""

    SUBCRITICAL = "subcritical"  # Too rigid/bureaucratic, low responsiveness
    CRITICAL = "critical"  # Optimal - edge of chaos, maximally adaptive
    SUPERCRITICAL = "supercritical"  # Too chaotic, unstable dynamics


@dataclass
class CriticalityMetrics:
    """Metrics for evaluating system criticality.

    Based on statistical physics indicators of critical behavior:
    - correlation_length: How far perturbations propagate
    - susceptibility: Response magnitude to small perturbations
    - relaxation_time: Time to return to equilibrium
    - fluctuation_size: Variance in system state
    """

    correlation_length: float = 0.0  # Range 0-1, normalized by system size
    susceptibility: float = 0.0  # Range 0-inf, typically 0-10
    relaxation_time: float = 0.0  # Seconds to equilibrium
    fluctuation_size: float = 0.0  # Variance in order parameter
    order_parameter: float = 0.5  # Measure of system organization (0-1)
    functional_connectivity: float = 0.5  # Ratio of meaningful to total interactions (0-1)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def criticality_score(self) -> float:
        """Calculate overall criticality score.

        Higher scores indicate closer to critical point.
        At criticality:
        - correlation_length is high
        - susceptibility is high
        - relaxation_time is high (critical slowing down)
        - fluctuation_size is high

        Returns:
            Score from 0 (far from critical) to 1 (at critical point)
        """
        # Normalize components
        corr_norm = min(1.0, self.correlation_length)
        susc_norm = min(1.0, self.susceptibility / 5.0)  # Normalize to typical max
        relax_norm = min(1.0, self.relaxation_time / 60.0)  # Normalize to 60s max
        fluct_norm = min(1.0, self.fluctuation_size * 4)  # Amplify small fluctuations

        # Weighted combination - all indicators should be high at criticality
        return 0.3 * corr_norm + 0.3 * susc_norm + 0.2 * relax_norm + 0.2 * fluct_norm

    def infer_state(self) -> CriticalityState:
        """Infer criticality state from metrics."""
        score = self.criticality_score

        # Also consider order parameter
        if self.order_parameter > 0.8:
            # High order = too structured
            return CriticalityState.SUBCRITICAL
        elif self.order_parameter < 0.2:
            # Low order = too chaotic
            return CriticalityState.SUPERCRITICAL

        # Use criticality score
        if score > 0.6:
            return CriticalityState.CRITICAL
        elif self.susceptibility < 0.5:
            return CriticalityState.SUBCRITICAL
        else:
            return CriticalityState.SUPERCRITICAL


@dataclass
class PhaseTransition:
    """Record of a detected phase transition."""

    transition_id: str
    from_state: CriticalityState
    to_state: CriticalityState
    trigger: str
    metrics_before: CriticalityMetrics
    metrics_after: CriticalityMetrics
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "transition_id": self.transition_id,
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "trigger": self.trigger,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class CriticalityMonitor:
    """Monitors system criticality and detects phase transitions.

    Periodically samples system state to calculate criticality metrics.
    When phase transitions are detected, can recommend topology changes
    to maintain optimal criticality.

    Subcritical systems need deterritorialization (more fluidity).
    Supercritical systems need territorialization (more structure).
    Critical systems are optimal and should be maintained.
    """

    # Thresholds for state detection
    SUBCRITICAL_THRESHOLD = 0.3
    SUPERCRITICAL_THRESHOLD = 0.7
    TRANSITION_HYSTERESIS = 0.1  # Prevent rapid oscillation

    # Sampling configuration
    DEFAULT_SAMPLE_INTERVAL = 30.0  # Seconds between samples
    HISTORY_SIZE = 100  # Number of samples to keep

    def __init__(
        self,
        neighborhood: TopologicalNeighborhood | None = None,
        event_bus: EventBus | None = None,
        sample_interval: float = DEFAULT_SAMPLE_INTERVAL,
    ) -> None:
        """Initialize the criticality monitor.

        Args:
            neighborhood: Topological neighborhood for measuring correlations.
            event_bus: Event bus for publishing criticality events.
            sample_interval: Seconds between state samples.
        """
        self._neighborhood = neighborhood
        self._event_bus = event_bus
        self._sample_interval = sample_interval

        self._current_state = CriticalityState.CRITICAL  # Assume starting balanced
        self._current_metrics = CriticalityMetrics()
        self._metrics_history: list[CriticalityMetrics] = []
        self._transitions: list[PhaseTransition] = []

        # State tracking for perturbation response
        self._perturbations: list[tuple[datetime, float, float]] = []  # (time, magnitude, response)
        self._state_samples: list[float] = []  # Order parameter samples

        # Background task
        self._monitor_task: asyncio.Task[None] | None = None
        self._running = False
        self._transition_counter = 0

        # Callbacks for external response
        self._on_transition_callbacks: list[Callable[[PhaseTransition], None]] = []

    @property
    def current_state(self) -> CriticalityState:
        """Get current criticality state."""
        return self._current_state

    @property
    def current_metrics(self) -> CriticalityMetrics:
        """Get current criticality metrics."""
        return self._current_metrics

    async def start(self) -> None:
        """Start the criticality monitoring loop."""
        if self._running:
            return

        self._running = True
        self._monitor_task = asyncio.create_task(self._run_monitoring_loop())
        logger.info("Started criticality monitor")

    async def stop(self) -> None:
        """Stop the criticality monitoring loop."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped criticality monitor")

    async def _run_monitoring_loop(self) -> None:
        """Background loop for periodic criticality sampling."""
        while self._running:
            try:
                await asyncio.sleep(self._sample_interval)
                await self._sample_and_evaluate()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in criticality monitoring: {e}")

    async def _sample_and_evaluate(self) -> None:
        """Sample system state and evaluate criticality."""
        metrics = await self.sample_state()
        self._metrics_history.append(metrics)

        # Trim history
        if len(self._metrics_history) > self.HISTORY_SIZE:
            self._metrics_history = self._metrics_history[-self.HISTORY_SIZE :]

        # Check for phase transition
        new_state = metrics.infer_state()
        if await self.detect_phase_transition():
            old_state = self._current_state
            self._current_state = new_state

            # Create transition record
            self._transition_counter += 1
            transition = PhaseTransition(
                transition_id=f"transition_{self._transition_counter}",
                from_state=old_state,
                to_state=new_state,
                trigger="criticality_evaluation",
                metrics_before=(
                    self._metrics_history[-2] if len(self._metrics_history) >= 2 else metrics
                ),
                metrics_after=metrics,
            )
            self._transitions.append(transition)

            # Notify callbacks
            for callback in self._on_transition_callbacks:
                try:
                    callback(transition)
                except Exception as e:
                    logger.error(f"Transition callback error: {e}")

            # Emit event
            if self._event_bus:
                from hive.events import EventType

                await self._event_bus.emit(
                    EventType.PHASE_TRANSITION_DETECTED,
                    {
                        "from_state": old_state.value,
                        "to_state": new_state.value,
                        "metrics": {
                            "correlation_length": metrics.correlation_length,
                            "susceptibility": metrics.susceptibility,
                            "criticality_score": metrics.criticality_score,
                        },
                    },
                    source_id="criticality_monitor",
                )

            # Record metric
            get_metrics().phase_transition(old_state.value, new_state.value)

            logger.info(
                f"Phase transition detected: {old_state.value} -> {new_state.value} "
                f"(score={metrics.criticality_score:.2f})"
            )

        # Update metrics
        self._current_metrics = metrics
        get_metrics().set_criticality_state(self._current_state.value)
        get_metrics().set_criticality_metrics(
            metrics.correlation_length,
            metrics.susceptibility,
            metrics.relaxation_time,
            metrics.fluctuation_size,
        )

    async def sample_state(self) -> CriticalityMetrics:
        """Sample the current system state and calculate metrics.

        Returns:
            CriticalityMetrics for current system state.
        """
        correlation_length = await self._measure_correlation_length()
        susceptibility = await self._measure_susceptibility()
        relaxation_time = await self._measure_relaxation_time()
        fluctuation_size = await self._measure_fluctuation_size()
        functional_connectivity = await self._measure_functional_connectivity()
        order_parameter = await self._measure_order_parameter(functional_connectivity)

        return CriticalityMetrics(
            correlation_length=correlation_length,
            susceptibility=susceptibility,
            relaxation_time=relaxation_time,
            fluctuation_size=fluctuation_size,
            order_parameter=order_parameter,
            functional_connectivity=functional_connectivity,
        )

    async def _measure_correlation_length(self) -> float:
        """Measure how far influence propagates through the system.

        Uses neighbor network to estimate correlation decay.
        """
        if not self._neighborhood:
            return 0.5  # Default moderate correlation

        # Get network statistics
        stats = self._neighborhood.get_network_stats()
        total_agents = int(stats.get("total_agents", 0))
        avg_clustering = float(stats.get("average_clustering", 0.0))

        if total_agents < 2:
            return 0.0

        # Correlation length relates to clustering and network size
        # High clustering = correlations propagate well
        base_length = avg_clustering

        # Scale by network size (larger networks have longer potential correlation)
        size_factor = min(1.0, math.log(total_agents + 1) / 5)

        return min(1.0, base_length * (1 + size_factor))

    async def _measure_susceptibility(self) -> float:
        """Measure system response to perturbations.

        Uses recorded perturbation responses.
        """
        if not self._perturbations:
            return 1.0  # Default moderate susceptibility

        # Calculate average response/magnitude ratio
        recent = self._perturbations[-20:]
        if not recent:
            return 1.0

        ratios = [r / m if m > 0 else 0 for _, m, r in recent]
        return statistics.mean(ratios) if ratios else 1.0

    async def _measure_relaxation_time(self) -> float:
        """Measure time to return to equilibrium after perturbation."""
        # Use variance decay rate from state samples
        if len(self._state_samples) < 10:
            return 10.0  # Default moderate relaxation

        # Calculate autocorrelation at different lags
        samples = self._state_samples[-50:]
        variance = statistics.variance(samples) if len(samples) > 1 else 1.0

        if variance < 0.001:
            return 5.0  # Very stable, fast relaxation

        # Estimate relaxation from variance decay
        # Higher variance = longer relaxation
        return min(60.0, 5.0 + variance * 50)

    async def _measure_fluctuation_size(self) -> float:
        """Measure variance in system state."""
        if len(self._state_samples) < 5:
            return 0.1  # Default low fluctuation

        samples = self._state_samples[-30:]
        if len(samples) < 2:
            return 0.1

        return min(1.0, statistics.stdev(samples))

    async def _measure_functional_connectivity(self) -> float:
        """Measure ratio of meaningful interactions to total interactions.

        Functional connectivity separates semantic throughput from structural
        connectivity. A fully-connected graph (SWARM) can have low functional
        connectivity if agents cannot comprehend each other's outputs.

        Uses interaction count vs. edge count from the neighborhood network.
        """
        if not self._neighborhood:
            return 0.5  # Default moderate

        stats = self._neighborhood.get_network_stats()
        total_agents = int(stats.get("total_agents", 0))
        total_interactions = int(stats.get("total_interactions", 0))
        total_edges = int(stats.get("total_edges", 0))

        if total_edges == 0 or total_agents < 2:
            return 0.5

        # Ratio of actual interactions to possible edges
        # High ratio = agents are meaningfully using their connections
        # Low ratio = structural connections exist but aren't producing dialogue
        interaction_ratio = min(1.0, total_interactions / max(1, total_edges))

        return interaction_ratio

    async def _measure_order_parameter(
        self, functional_connectivity: float | None = None,
    ) -> float:
        """Measure degree of system organization.

        Order parameter ~ 1: highly organized
        Order parameter ~ 0: disordered

        When structural density is saturated (≈1.0, e.g. SWARM topology),
        uses topological neighbor ratio (k/N) as effective density instead
        of raw structural density. This follows Cavagna et al. 2010:
        functional interactions are topological, not metric.
        """
        if not self._neighborhood:
            return 0.5  # Default balanced

        stats = self._neighborhood.get_network_stats()
        total_agents = int(stats.get("total_agents", 0))
        density = float(stats.get("density", 0.0))
        avg_clustering = float(stats.get("average_clustering", 0.0))

        if total_agents < 2:
            return 0.5

        # Structural order: density + clustering (who CAN talk)
        # When density is saturated (all-to-all, e.g. SWARM), use topological
        # neighbor ratio as effective density (Cavagna et al. 2010).
        if density > 0.95 and total_agents > 2:
            neighbor_count = min(
                int(stats.get("neighbor_count", total_agents - 1)),
                total_agents - 1,  # Clamp k to feasible neighbors in small swarms
            )
            effective_density = neighbor_count / (total_agents - 1)
            structural_order = effective_density * 0.5 + avg_clustering * 0.5
        else:
            structural_order = density * 0.5 + avg_clustering * 0.5

        # Blend structural and functional connectivity
        # Functional connectivity captures who IS meaningfully communicating
        func_conn = functional_connectivity if functional_connectivity is not None else 0.5
        order = structural_order * 0.6 + func_conn * 0.4

        # Record for tracking
        self._state_samples.append(order)
        if len(self._state_samples) > 200:
            self._state_samples = self._state_samples[-200:]

        return order

    async def detect_phase_transition(self) -> bool:
        """Detect if a phase transition is occurring.

        Returns:
            True if transition detected, False otherwise.
        """
        if len(self._metrics_history) < 3:
            return False

        current = self._metrics_history[-1]
        previous = self._metrics_history[-2]

        # Detect state change with hysteresis
        current_state = current.infer_state()
        if current_state == self._current_state:
            return False

        # Check for significant change in criticality score
        score_change = abs(current.criticality_score - previous.criticality_score)
        if score_change < self.TRANSITION_HYSTERESIS:
            return False

        # Check for sustained change (not just noise)
        if len(self._metrics_history) >= 5:
            recent_states = [m.infer_state() for m in self._metrics_history[-5:]]
            state_counts: dict[CriticalityState, int] = {}
            for s in recent_states:
                state_counts[s] = state_counts.get(s, 0) + 1

            # Need majority in new state to confirm transition
            if state_counts.get(current_state, 0) < 3:
                return False

        return True

    async def recommend_intervention(self) -> str | None:
        """Recommend topology intervention based on criticality state.

        Returns:
            Recommended TopologyType name, or None if no intervention needed.
        """
        if self._current_state == CriticalityState.CRITICAL:
            return None  # At optimal point, no intervention needed

        if self._current_state == CriticalityState.SUBCRITICAL:
            # Too rigid - need deterritorialization / more fluidity
            # Recommend rhizomatic or deterritorialized topology
            logger.info("Recommending deterritorialization (subcritical system)")
            return "deterritorialized"

        if self._current_state == CriticalityState.SUPERCRITICAL:
            # Too chaotic - need territorialization / more structure
            # Recommend hierarchy or territorialized topology
            logger.info("Recommending territorialization (supercritical system)")
            return "hierarchy"

        return None

    def record_perturbation(
        self,
        magnitude: float,
        response: float,
    ) -> None:
        """Record a perturbation and its response for susceptibility calculation.

        Args:
            magnitude: Size of the perturbation applied.
            response: Size of the system's response.
        """
        self._perturbations.append((datetime.now(UTC), magnitude, response))

        # Trim history
        if len(self._perturbations) > 100:
            self._perturbations = self._perturbations[-100:]

    def on_transition(self, callback: Callable[[PhaseTransition], None]) -> None:
        """Register a callback for phase transitions.

        Args:
            callback: Function to call when transition occurs.
        """
        self._on_transition_callbacks.append(callback)

    def get_transitions(self, limit: int = 20) -> list[PhaseTransition]:
        """Get recent phase transitions.

        Args:
            limit: Maximum number of transitions to return.

        Returns:
            List of recent PhaseTransition objects.
        """
        return self._transitions[-limit:]

    def get_metrics_history(self, limit: int = 50) -> list[CriticalityMetrics]:
        """Get recent metrics history.

        Args:
            limit: Maximum number of samples to return.

        Returns:
            List of recent CriticalityMetrics.
        """
        return self._metrics_history[-limit:]

    def to_dict(self) -> dict[str, Any]:
        """Serialize state to dictionary."""
        return {
            "current_state": self._current_state.value,
            "current_metrics": {
                "correlation_length": self._current_metrics.correlation_length,
                "susceptibility": self._current_metrics.susceptibility,
                "relaxation_time": self._current_metrics.relaxation_time,
                "fluctuation_size": self._current_metrics.fluctuation_size,
                "order_parameter": self._current_metrics.order_parameter,
                "functional_connectivity": self._current_metrics.functional_connectivity,
                "criticality_score": self._current_metrics.criticality_score,
            },
            "transitions_count": len(self._transitions),
            "samples_count": len(self._metrics_history),
        }
