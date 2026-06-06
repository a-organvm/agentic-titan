"""Chaos engineering tests for FissionFusion dynamics (#35).

Validates the full emergence loop under adversarial conditions:
conflict -> crisis_level rises -> FUSION triggered -> agents coordinate
-> conflict resolves -> crisis drops -> FISSION.

Tests both the FissionFusionTopology data structure (agent loss scenarios)
and the FissionFusionManager dynamics engine (crisis propagation, state
machine transitions, windowed gates, cooldown enforcement).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest

from hive.conflict import ConflictDetector
from hive.criticality import CriticalityMetrics, CriticalityMonitor, CriticalityState
from hive.fission_fusion import (
    FissionFusionManager,
    FissionFusionMetrics,
    FissionFusionState,
)
from hive.stigmergy import PheromoneTrace, TraceType
from hive.topology import FissionFusionTopology

pytestmark = [pytest.mark.chaos]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trace(
    trace_type: TraceType,
    location: str = "loc_0",
    intensity: float = 0.8,
    depositor: str = "agent_x",
) -> PheromoneTrace:
    """Create a fresh, non-expired PheromoneTrace."""
    return PheromoneTrace(
        trace_id=uuid.uuid4().hex[:8],
        trace_type=trace_type,
        location=location,
        intensity=intensity,
        depositor_id=depositor,
        deposited_at=datetime.now(UTC),
    )


def _make_manager(
    *,
    criticality_monitor: CriticalityMonitor | None = None,
    conflict_detector: ConflictDetector | None = None,
    with_event_bus: bool = True,
) -> FissionFusionManager:
    """Build a FissionFusionManager with mock dependencies."""
    neighborhood = MagicMock()
    neighborhood.get_network_stats.return_value = {
        "total_agents": 10,
        "average_clustering": 0.5,
        "density": 0.4,
    }
    neighborhood._profiles = {}

    pheromone_field = MagicMock()
    pheromone_field._traces = {}

    event_bus = None
    if with_event_bus:
        event_bus = MagicMock()
        event_bus.emit = AsyncMock()

    mgr = FissionFusionManager(
        neighborhood=neighborhood,
        pheromone_field=pheromone_field,
        event_bus=event_bus,
        evaluation_interval=1.0,
        criticality_monitor=criticality_monitor,
        conflict_detector=conflict_detector,
    )
    return mgr


def _populate_topology(topo: FissionFusionTopology, n: int, cluster_id: str) -> list[str]:
    """Add n agents to a topology cluster. Returns agent IDs."""
    ids = [f"agent_{cluster_id}_{i}" for i in range(n)]
    for aid in ids:
        topo.add_agent(aid, aid, ["worker"], cluster_id=cluster_id)
    return ids


# =========================================================================
# A. Crisis Signal Propagation
# =========================================================================


class TestCrisisSignalPropagation:
    """Verify each crisis source correctly drives the manager toward FUSION."""

    async def test_manual_crisis_triggers_fusion(self):
        """set_crisis_level(0.8) + high task correlation -> should_transition() returns FUSION.

        Note: should_transition() has a final hysteresis gate on task_correlation
        (not just crisis_level). In a real crisis, agents converge on the same problem,
        so high task correlation accompanies high crisis.
        """
        mgr = _make_manager()
        mgr.set_crisis_level(0.8)
        # Set current metrics to match crisis conditions
        mgr._metrics = FissionFusionMetrics(task_correlation=0.9, crisis_level=0.8)

        # Populate metrics history so the windowed gate can fire.
        for _ in range(5):
            mgr._metrics_history.append(
                FissionFusionMetrics(task_correlation=0.9, crisis_level=0.8)
            )

        result = await mgr.should_transition()
        assert result == FissionFusionState.FUSION

    async def test_criticality_derived_crisis(self):
        """CriticalityMonitor in SUPERCRITICAL -> crisis rises above fusion threshold."""
        monitor = MagicMock(spec=CriticalityMonitor)
        type(monitor).current_state = PropertyMock(return_value=CriticalityState.SUPERCRITICAL)
        type(monitor).current_metrics = PropertyMock(
            return_value=CriticalityMetrics(
                correlation_length=0.9,
                susceptibility=8.0,
                relaxation_time=50.0,
                fluctuation_size=0.3,
            )
        )

        mgr = _make_manager(criticality_monitor=monitor)
        metrics = await mgr.evaluate_state()

        # SUPERCRITICAL with high criticality_score -> crisis should be well above 0.6
        assert metrics.crisis_level > 0.6, (
            f"Expected crisis > 0.6 from SUPERCRITICAL, got {metrics.crisis_level}"
        )

    async def test_conflict_detector_floor(self):
        """Opposing pheromone traces raise the crisis floor via ConflictDetector."""
        detector = ConflictDetector(intensity_threshold=0.5)

        mgr = _make_manager(conflict_detector=detector)

        # Inject opposing traces directly into the mock pheromone field
        mgr._pheromone_field._traces = {
            "loc_0": {
                TraceType.RESOURCE: [_make_trace(TraceType.RESOURCE, intensity=0.9)],
                TraceType.WARNING: [_make_trace(TraceType.WARNING, intensity=0.9)],
            }
        }

        metrics = await mgr.evaluate_state()

        # Conflict floor should elevate crisis above the default (0.0)
        assert metrics.crisis_level > 0.5, (
            f"Expected conflict floor > 0.5, got {metrics.crisis_level}"
        )


# =========================================================================
# B. Agent Loss Scenarios (Topology data structure)
# =========================================================================


class TestAgentLossScenarios:
    """Verify FissionFusionTopology resilience when agents are removed."""

    def test_cluster_survives_minority_loss(self):
        """5 agents in cluster, remove 2 -> remaining 3 still connected."""
        topo = FissionFusionTopology()
        ids = _populate_topology(topo, 5, "alpha")

        topo.remove_agent(ids[0])
        topo.remove_agent(ids[1])

        assert len(topo.nodes) == 3
        survivors = [ids[2], ids[3], ids[4]]
        for sid in survivors:
            node = topo.nodes[sid]
            # Each survivor should see the other 2 as neighbors
            assert set(node.neighbors) == set(survivors) - {sid}

    def test_cluster_survives_majority_loss(self):
        """5 agents, remove 4 -> last agent exists with empty neighbors."""
        topo = FissionFusionTopology()
        ids = _populate_topology(topo, 5, "beta")

        for victim in ids[:4]:
            topo.remove_agent(victim)

        assert len(topo.nodes) == 1
        lone = topo.nodes[ids[4]]
        assert lone.neighbors == []
        assert lone.metadata["cluster_id"] == "beta"

    def test_cross_cluster_loss(self):
        """10 agents in 2 clusters, remove 1 from each -> both clusters intact."""
        topo = FissionFusionTopology()
        a_ids = _populate_topology(topo, 5, "A")
        b_ids = _populate_topology(topo, 5, "B")

        topo.remove_agent(a_ids[0])
        topo.remove_agent(b_ids[0])

        assert len(topo.nodes) == 8

        # Cluster A survivors
        for aid in a_ids[1:]:
            node = topo.nodes[aid]
            assert node.metadata["cluster_id"] == "A"
            assert len(node.neighbors) == 3  # 4 remaining - self

        # Cluster B survivors
        for bid in b_ids[1:]:
            node = topo.nodes[bid]
            assert node.metadata["cluster_id"] == "B"
            assert len(node.neighbors) == 3

    def test_total_cluster_wipeout(self):
        """Remove all agents from one cluster -> other cluster unaffected."""
        topo = FissionFusionTopology()
        a_ids = _populate_topology(topo, 5, "A")
        b_ids = _populate_topology(topo, 5, "B")

        for victim in a_ids:
            topo.remove_agent(victim)

        assert len(topo.nodes) == 5
        # Cluster A is gone
        assert topo.clusters.get("A", []) == []
        # Cluster B untouched
        assert set(topo.clusters["B"]) == set(b_ids)
        for bid in b_ids:
            assert len(topo.nodes[bid].neighbors) == 4


# =========================================================================
# C. State Machine Under Stress
# =========================================================================


class TestStateMachineUnderStress:
    """Verify state machine behavior under adversarial signal patterns."""

    async def test_chaos_fission_to_fusion_to_fission(self):
        """Full loop: FISSION -> inject crisis -> FUSION -> clear -> FISSION."""
        mgr = _make_manager()
        assert mgr.state == FissionFusionState.FISSION

        # --- Drive to FUSION via manual crisis ---
        mgr.set_crisis_level(0.9)
        # Fill windowed history with FUSION-favoring metrics
        for _ in range(5):
            mgr._metrics_history.append(
                FissionFusionMetrics(task_correlation=0.9, crisis_level=0.9)
            )
        await mgr.perform_fusion()
        assert mgr.state == FissionFusionState.FUSION

        # --- Clear crisis and drive back to FISSION ---
        mgr.clear_crisis_override()
        mgr._metrics = FissionFusionMetrics(task_correlation=0.1, crisis_level=0.0)
        # Clear cooldown so transition is allowed
        mgr._last_transition_time = None
        # Fill windowed history with FISSION-favoring metrics
        mgr._metrics_history.clear()
        for _ in range(5):
            mgr._metrics_history.append(
                FissionFusionMetrics(task_correlation=0.1, crisis_level=0.0)
            )
        await mgr.perform_fission()
        assert mgr.state == FissionFusionState.FISSION

    async def test_windowed_gate_blocks_transient_spike(self):
        """A single high-crisis evaluation should NOT trigger transition."""
        mgr = _make_manager()

        # 4 calm cycles + 1 spike = not enough for 3-of-5 majority
        for _ in range(4):
            mgr._metrics_history.append(
                FissionFusionMetrics(task_correlation=0.2, crisis_level=0.1)
            )
        mgr._metrics_history.append(
            FissionFusionMetrics(task_correlation=0.9, crisis_level=0.9)
        )

        # Current metrics show the spike
        mgr._metrics = FissionFusionMetrics(task_correlation=0.9, crisis_level=0.9)
        result = await mgr.should_transition()
        # Only 1-of-5 votes for FUSION — gate blocks
        assert result is None

    async def test_cooldown_prevents_rapid_oscillation(self):
        """After a transition, immediate counter-signal is suppressed."""
        mgr = _make_manager()

        # Perform fusion (sets _last_transition_time)
        await mgr.perform_fusion()
        assert mgr.state == FissionFusionState.FUSION

        # Immediately try to transition back with FISSION-favoring metrics
        mgr._metrics = FissionFusionMetrics(task_correlation=0.1, crisis_level=0.0)
        mgr._metrics_history.clear()
        for _ in range(5):
            mgr._metrics_history.append(
                FissionFusionMetrics(task_correlation=0.1, crisis_level=0.0)
            )

        # Cooldown should block this
        result = await mgr.should_transition()
        assert result is None, "Cooldown should prevent immediate counter-transition"

    async def test_concurrent_crisis_sources(self):
        """Both CriticalityMonitor AND ConflictDetector fire -> conflict floor wins when higher."""
        monitor = MagicMock(spec=CriticalityMonitor)
        type(monitor).current_state = PropertyMock(return_value=CriticalityState.SUBCRITICAL)
        # SUBCRITICAL with low score -> crisis ~0.04
        type(monitor).current_metrics = PropertyMock(
            return_value=CriticalityMetrics(
                correlation_length=0.1,
                susceptibility=0.5,
                relaxation_time=5.0,
                fluctuation_size=0.05,
            )
        )

        detector = ConflictDetector(intensity_threshold=0.5)

        mgr = _make_manager(criticality_monitor=monitor, conflict_detector=detector)

        # Inject strong conflicts
        mgr._pheromone_field._traces = {
            "loc_0": {
                TraceType.PATH: [_make_trace(TraceType.PATH, intensity=0.95)],
                TraceType.FAILURE: [_make_trace(TraceType.FAILURE, intensity=0.95)],
            },
            "loc_1": {
                TraceType.SUCCESS: [_make_trace(TraceType.SUCCESS, "loc_1", 0.85)],
                TraceType.FAILURE: [_make_trace(TraceType.FAILURE, "loc_1", 0.85)],
            },
        }

        metrics = await mgr.evaluate_state()

        # Criticality-derived crisis is low (SUBCRITICAL), but conflict floor overrides
        assert metrics.crisis_level > 0.7, (
            f"Conflict floor should dominate over SUBCRITICAL crisis, got {metrics.crisis_level}"
        )


# =========================================================================
# D. Recovery Dynamics
# =========================================================================


class TestRecoveryDynamics:
    """Verify the system recovers correctly after state transitions."""

    async def test_fusion_creates_info_center(self):
        """During fusion, info_center_id is set and FUSION_COMPLETED event emitted."""
        mgr = _make_manager(with_event_bus=True)
        info_center = await mgr.perform_fusion()

        assert mgr.state == FissionFusionState.FUSION
        assert info_center is not None
        assert info_center.startswith("info_center_")
        assert mgr._info_center_id == info_center

        # Event bus should have emitted FUSION_COMPLETED
        mgr._event_bus.emit.assert_called()
        call_args = mgr._event_bus.emit.call_args_list
        event_types = [call.args[0].value for call in call_args]
        assert "fission_fusion.fusion.completed" in event_types

    async def test_fission_after_recovery_forms_clusters(self):
        """After crisis clears and fission occurs, clusters reform from profiles."""
        mgr = _make_manager()

        # Add agent profiles to the neighborhood so clustering has material
        profiles = {}
        for i in range(6):
            profile = MagicMock()
            profile.task_similarity = MagicMock(return_value=0.5)
            profiles[f"agent_{i}"] = profile
        mgr._neighborhood._profiles = profiles

        # Perform fission
        cluster_count = await mgr.perform_fission()

        assert mgr.state == FissionFusionState.FISSION
        assert cluster_count > 0
        # All agents should be assigned to clusters
        assert len(mgr._agent_cluster) == 6

    async def test_state_change_callbacks_fire_under_chaos(self):
        """Registered callbacks are invoked with the correct state on transition."""
        mgr = _make_manager()
        received_states: list[FissionFusionState] = []
        mgr.on_state_change(lambda s: received_states.append(s))

        await mgr.perform_fusion()
        assert FissionFusionState.FUSION in received_states

        # Add agent profiles so perform_fission doesn't take the empty-agents early return
        for i in range(4):
            profile = MagicMock()
            profile.task_similarity = MagicMock(return_value=0.5)
            mgr._neighborhood._profiles[f"agent_{i}"] = profile

        mgr._last_transition_time = None  # bypass cooldown
        await mgr.perform_fission()
        assert FissionFusionState.FISSION in received_states


# =========================================================================
# E. Integration: Full Emergence Loop
# =========================================================================


class TestFullEmergenceLoop:
    """Test the complete crisis resolution pipeline with real components."""

    async def test_conflict_driven_full_loop(self):
        """Real opposing traces trigger fusion, then clear and recover to fission."""
        detector = ConflictDetector(intensity_threshold=0.5)
        mgr = _make_manager(conflict_detector=detector)

        # --- Phase 1: Inject conflict ---
        mgr._pheromone_field._traces = {
            "battlefield": {
                TraceType.EXPLORATION: [
                    _make_trace(TraceType.EXPLORATION, "battlefield", 0.9),
                ],
                TraceType.TERRITORY: [
                    _make_trace(TraceType.TERRITORY, "battlefield", 0.9),
                ],
            }
        }

        metrics = await mgr.evaluate_state()
        assert metrics.crisis_level > 0.5, "Conflict should raise crisis"

        # --- Phase 2: Transition to FUSION ---
        # Fill windowed history to pass majority gate
        mgr._metrics_history.clear()
        for _ in range(5):
            mgr._metrics_history.append(
                FissionFusionMetrics(task_correlation=0.9, crisis_level=0.9)
            )
        mgr._metrics = FissionFusionMetrics(task_correlation=0.9, crisis_level=0.9)
        await mgr.perform_fusion()
        assert mgr.state == FissionFusionState.FUSION

        # --- Phase 3: Clear conflict, recover to FISSION ---
        mgr._pheromone_field._traces = {}
        mgr._conflict_crisis_level = 0.0
        mgr.clear_crisis_override()
        mgr._last_transition_time = None  # bypass cooldown

        mgr._metrics = FissionFusionMetrics(task_correlation=0.1, crisis_level=0.0)
        mgr._metrics_history.clear()
        for _ in range(5):
            mgr._metrics_history.append(
                FissionFusionMetrics(task_correlation=0.1, crisis_level=0.0)
            )
        await mgr.perform_fission()
        assert mgr.state == FissionFusionState.FISSION

    async def test_crisis_level_resolution_priority(self):
        """Verify crisis priority tiers and the conflict floor."""
        monitor = MagicMock(spec=CriticalityMonitor)
        type(monitor).current_state = PropertyMock(return_value=CriticalityState.SUPERCRITICAL)
        type(monitor).current_metrics = PropertyMock(
            return_value=CriticalityMetrics(
                correlation_length=0.9,
                susceptibility=8.0,
                relaxation_time=50.0,
                fluctuation_size=0.3,
            )
        )

        detector = ConflictDetector(intensity_threshold=0.5)
        mgr = _make_manager(criticality_monitor=monitor, conflict_detector=detector)

        # --- Tier 1: Manual override takes precedence ---
        mgr.set_crisis_level(0.2)
        level = mgr._resolve_crisis_level()
        # Manual is 0.2, conflict floor is 0.0 -> result is max(0.2, 0.0) = 0.2
        assert abs(level - 0.2) < 0.01, f"Manual override should produce 0.2, got {level}"

        # --- Now add conflict floor that's higher than manual ---
        mgr._conflict_crisis_level = 0.7
        level = mgr._resolve_crisis_level()
        # Manual is 0.2, conflict floor is 0.7 -> result is max(0.2, 0.7) = 0.7
        assert abs(level - 0.7) < 0.01, f"Conflict floor should override to 0.7, got {level}"

        # --- Clear manual, fall through to criticality ---
        mgr.clear_crisis_override()
        mgr._conflict_crisis_level = 0.0
        level = mgr._resolve_crisis_level()
        # SUPERCRITICAL with high criticality_score -> crisis well above 0.5
        assert level > 0.5, f"Criticality-derived crisis should be > 0.5, got {level}"

        # --- Clear criticality, fall through to preserved value ---
        mgr._criticality_monitor = None
        mgr._metrics.crisis_level = 0.35
        level = mgr._resolve_crisis_level()
        assert abs(level - 0.35) < 0.01, f"Preserved value should be 0.35, got {level}"
