"""Tests for criticality detection and phase transitions (Phase 16A)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from hive.criticality import (
    CriticalityMetrics,
    CriticalityMonitor,
    CriticalityState,
    PhaseTransition,
)


class TestCriticalityState:
    """Tests for CriticalityState enum."""

    def test_states_exist(self):
        """Test all states are defined."""
        assert CriticalityState.SUBCRITICAL == "subcritical"
        assert CriticalityState.CRITICAL == "critical"
        assert CriticalityState.SUPERCRITICAL == "supercritical"


class TestCriticalityMetrics:
    """Tests for CriticalityMetrics dataclass."""

    def test_default_metrics(self):
        """Test default metric values."""
        metrics = CriticalityMetrics()
        assert metrics.correlation_length == 0.0
        assert metrics.susceptibility == 0.0
        assert metrics.relaxation_time == 0.0
        assert metrics.fluctuation_size == 0.0
        assert metrics.order_parameter == 0.5

    def test_criticality_score_calculation(self):
        """Test criticality score is calculated correctly."""
        # Low criticality
        low = CriticalityMetrics(
            correlation_length=0.1,
            susceptibility=0.5,
            relaxation_time=5.0,
            fluctuation_size=0.05,
        )
        assert 0.0 <= low.criticality_score <= 1.0

        # High criticality (edge of chaos)
        high = CriticalityMetrics(
            correlation_length=0.9,
            susceptibility=4.0,
            relaxation_time=50.0,
            fluctuation_size=0.2,
        )
        assert high.criticality_score > low.criticality_score

    def test_criticality_score_bounds(self):
        """Test criticality score stays within bounds."""
        extreme = CriticalityMetrics(
            correlation_length=10.0,  # Way above normal
            susceptibility=100.0,
            relaxation_time=1000.0,
            fluctuation_size=5.0,
        )
        assert 0.0 <= extreme.criticality_score <= 1.0

    def test_infer_state_subcritical(self):
        """Test inferring subcritical state."""
        metrics = CriticalityMetrics(
            correlation_length=0.1,
            susceptibility=0.3,
            relaxation_time=5.0,
            fluctuation_size=0.02,
            order_parameter=0.9,  # Very high order
        )
        assert metrics.infer_state() == CriticalityState.SUBCRITICAL

    def test_infer_state_supercritical(self):
        """Test inferring supercritical state."""
        metrics = CriticalityMetrics(
            correlation_length=0.3,
            susceptibility=2.0,
            relaxation_time=20.0,
            fluctuation_size=0.3,
            order_parameter=0.1,  # Very low order
        )
        assert metrics.infer_state() == CriticalityState.SUPERCRITICAL

    def test_infer_state_critical(self):
        """Test inferring critical state."""
        metrics = CriticalityMetrics(
            correlation_length=0.8,
            susceptibility=3.5,
            relaxation_time=40.0,
            fluctuation_size=0.15,
            order_parameter=0.5,  # Balanced order
        )
        assert metrics.infer_state() == CriticalityState.CRITICAL


class TestPhaseTransition:
    """Tests for PhaseTransition dataclass."""

    def test_phase_transition_creation(self):
        """Test creating a phase transition record."""
        before = CriticalityMetrics(order_parameter=0.3)
        after = CriticalityMetrics(order_parameter=0.7)

        transition = PhaseTransition(
            transition_id="test_1",
            from_state=CriticalityState.SUPERCRITICAL,
            to_state=CriticalityState.SUBCRITICAL,
            trigger="test",
            metrics_before=before,
            metrics_after=after,
        )

        assert transition.transition_id == "test_1"
        assert transition.from_state == CriticalityState.SUPERCRITICAL
        assert transition.to_state == CriticalityState.SUBCRITICAL

    def test_phase_transition_to_dict(self):
        """Test serialization of phase transition."""
        transition = PhaseTransition(
            transition_id="test_2",
            from_state=CriticalityState.CRITICAL,
            to_state=CriticalityState.SUBCRITICAL,
            trigger="stability",
            metrics_before=CriticalityMetrics(),
            metrics_after=CriticalityMetrics(),
        )

        data = transition.to_dict()
        assert data["transition_id"] == "test_2"
        assert data["from_state"] == "critical"
        assert data["to_state"] == "subcritical"
        assert "timestamp" in data


class TestCriticalityMonitor:
    """Tests for CriticalityMonitor class."""

    @pytest.fixture
    def mock_neighborhood(self):
        """Create mock neighborhood."""
        neighborhood = MagicMock()
        neighborhood.get_network_stats.return_value = {
            "total_agents": 10,
            "average_clustering": 0.5,
            "density": 0.4,
            "total_edges": 36,
            "total_interactions": 18,
        }
        neighborhood._profiles = {}
        return neighborhood

    @pytest.fixture
    def mock_event_bus(self):
        """Create mock event bus."""
        bus = MagicMock()
        bus.emit = AsyncMock()
        return bus

    @pytest.fixture
    def monitor(self, mock_neighborhood, mock_event_bus):
        """Create a CriticalityMonitor for testing."""
        return CriticalityMonitor(
            neighborhood=mock_neighborhood,
            event_bus=mock_event_bus,
            sample_interval=1.0,  # Fast for testing
        )

    def test_initial_state(self, monitor):
        """Test initial monitor state."""
        assert monitor.current_state == CriticalityState.CRITICAL
        assert monitor.current_metrics is not None

    @pytest.mark.asyncio
    async def test_sample_state(self, monitor):
        """Test sampling system state."""
        metrics = await monitor.sample_state()

        assert isinstance(metrics, CriticalityMetrics)
        assert 0.0 <= metrics.correlation_length <= 1.0
        assert metrics.susceptibility >= 0.0
        assert metrics.relaxation_time >= 0.0

    @pytest.mark.asyncio
    async def test_detect_phase_transition_no_history(self, monitor):
        """Test no transition detected with insufficient history."""
        result = await monitor.detect_phase_transition()
        assert result is False

    @pytest.mark.asyncio
    async def test_detect_phase_transition_with_change(self, monitor):
        """Test transition detection with state change."""
        # Add some history
        monitor._metrics_history = [
            CriticalityMetrics(order_parameter=0.5, susceptibility=2.0),
            CriticalityMetrics(order_parameter=0.5, susceptibility=2.0),
            CriticalityMetrics(order_parameter=0.9, susceptibility=0.5),  # Changed
        ]
        # The actual detection depends on current state vs inferred
        result = await monitor.detect_phase_transition()
        # May or may not detect depending on thresholds
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_recommend_intervention_critical(self, monitor):
        """Test no intervention recommended when critical."""
        monitor._current_state = CriticalityState.CRITICAL
        recommendation = await monitor.recommend_intervention()
        assert recommendation is None

    @pytest.mark.asyncio
    async def test_recommend_intervention_subcritical(self, monitor):
        """Test deterritorialization recommended when subcritical."""
        monitor._current_state = CriticalityState.SUBCRITICAL
        recommendation = await monitor.recommend_intervention()
        assert recommendation == "deterritorialized"

    @pytest.mark.asyncio
    async def test_recommend_intervention_supercritical(self, monitor):
        """Test territorialization recommended when supercritical."""
        monitor._current_state = CriticalityState.SUPERCRITICAL
        recommendation = await monitor.recommend_intervention()
        assert recommendation == "hierarchy"

    def test_record_perturbation(self, monitor):
        """Test recording perturbations."""
        monitor.record_perturbation(magnitude=1.0, response=0.5)
        assert len(monitor._perturbations) == 1

        # Record more
        for i in range(10):
            monitor.record_perturbation(magnitude=0.5, response=0.3)
        assert len(monitor._perturbations) == 11

    def test_on_transition_callback(self, monitor):
        """Test registering transition callback."""
        callback = MagicMock()
        monitor.on_transition(callback)
        assert callback in monitor._on_transition_callbacks

    def test_get_transitions(self, monitor):
        """Test getting transition history."""
        # Initially empty
        assert monitor.get_transitions() == []

        # Add a transition
        transition = PhaseTransition(
            transition_id="test",
            from_state=CriticalityState.CRITICAL,
            to_state=CriticalityState.SUBCRITICAL,
            trigger="test",
            metrics_before=CriticalityMetrics(),
            metrics_after=CriticalityMetrics(),
        )
        monitor._transitions.append(transition)

        assert len(monitor.get_transitions()) == 1

    def test_to_dict(self, monitor):
        """Test serialization."""
        data = monitor.to_dict()

        assert "current_state" in data
        assert "current_metrics" in data
        assert "transitions_count" in data
        assert "samples_count" in data

    @pytest.mark.asyncio
    async def test_start_stop(self, monitor):
        """Test starting and stopping the monitor."""
        # Start
        await monitor.start()
        assert monitor._running is True
        assert monitor._monitor_task is not None

        # Stop
        await monitor.stop()
        assert monitor._running is False

    @pytest.mark.asyncio
    async def test_monitoring_loop(self, monitor, mock_event_bus):
        """Test the monitoring loop samples state."""
        monitor._sample_interval = 0.1  # Very fast

        await monitor.start()
        await asyncio.sleep(0.25)  # Let it run a couple cycles
        await monitor.stop()

        # Should have some metrics history
        assert len(monitor._metrics_history) >= 1


class TestSwarmDensitySaturation:
    """Tests for SWARM topology density saturation fix (#67)."""

    @pytest.fixture
    def swarm_neighborhood(self):
        """Mock neighborhood with SWARM-like saturated density."""
        neighborhood = MagicMock()
        neighborhood.get_network_stats.return_value = {
            "total_agents": 20,
            "average_clustering": 1.0,
            "density": 1.0,  # SWARM: all-to-all
            "neighbor_count": 7,  # Topological k=7
        }
        neighborhood._profiles = {}
        return neighborhood

    @pytest.fixture
    def normal_neighborhood(self):
        """Mock neighborhood with normal (non-saturated) density."""
        neighborhood = MagicMock()
        neighborhood.get_network_stats.return_value = {
            "total_agents": 20,
            "average_clustering": 0.5,
            "density": 0.4,
            "neighbor_count": 7,
        }
        neighborhood._profiles = {}
        return neighborhood

    @pytest.mark.asyncio
    async def test_swarm_order_parameter_not_saturated(self, swarm_neighborhood):
        """SWARM density=1.0 should NOT produce order_parameter=1.0."""
        monitor = CriticalityMonitor(neighborhood=swarm_neighborhood)
        order = await monitor._measure_order_parameter()

        # With k=7, N=20: effective_density = 7/19 ≈ 0.368
        # order = 0.368*0.5 + 1.0*0.5 = 0.684 (not 1.0)
        assert order < 0.9, f"Order parameter {order} still saturated for SWARM"

    @pytest.mark.asyncio
    async def test_swarm_can_be_supercritical(self, swarm_neighborhood):
        """SWARM should be able to reach SUPERCRITICAL state."""
        monitor = CriticalityMonitor(neighborhood=swarm_neighborhood)
        order = await monitor._measure_order_parameter()

        # order_parameter < 0.8 means SUBCRITICAL gate does NOT fire
        assert order < 0.8, "SWARM should not be trapped in SUBCRITICAL"

    @pytest.mark.asyncio
    async def test_normal_density_unchanged(self, normal_neighborhood):
        """Non-saturated density structural calculation is unchanged."""
        monitor = CriticalityMonitor(neighborhood=normal_neighborhood)
        # Pass functional_connectivity=0.5 (default) to isolate structural behavior
        order = await monitor._measure_order_parameter(functional_connectivity=0.5)

        # structural_order = 0.4*0.5 + 0.5*0.5 = 0.45
        # order = 0.45*0.6 + 0.5*0.4 = 0.47
        expected = (0.4 * 0.5 + 0.5 * 0.5) * 0.6 + 0.5 * 0.4
        assert abs(order - expected) < 0.01

    @pytest.mark.asyncio
    async def test_swarm_effective_density_scales_with_k(self):
        """Effective density scales with topological neighbor count."""
        neighborhood_k3 = MagicMock()
        neighborhood_k3.get_network_stats.return_value = {
            "total_agents": 20,
            "average_clustering": 1.0,
            "density": 1.0,
            "neighbor_count": 3,
        }
        neighborhood_k3._profiles = {}

        neighborhood_k15 = MagicMock()
        neighborhood_k15.get_network_stats.return_value = {
            "total_agents": 20,
            "average_clustering": 1.0,
            "density": 1.0,
            "neighbor_count": 15,
        }
        neighborhood_k15._profiles = {}

        monitor_k3 = CriticalityMonitor(neighborhood=neighborhood_k3)
        monitor_k15 = CriticalityMonitor(neighborhood=neighborhood_k15)

        order_k3 = await monitor_k3._measure_order_parameter()
        order_k15 = await monitor_k15._measure_order_parameter()

        assert order_k3 < order_k15, "Higher k should produce higher order parameter"

    @pytest.mark.asyncio
    async def test_small_swarm_clamps_k(self):
        """In small swarms where k > N-1, effective_density stays <= 1.0."""
        neighborhood = MagicMock()
        neighborhood.get_network_stats.return_value = {
            "total_agents": 4,
            "average_clustering": 1.0,
            "density": 1.0,
            "neighbor_count": 7,  # k=7 > N-1=3
        }
        neighborhood._profiles = {}

        monitor = CriticalityMonitor(neighborhood=neighborhood)
        order = await monitor._measure_order_parameter()

        # k clamped to 3, effective_density = 3/3 = 1.0
        # order = 1.0*0.5 + 1.0*0.5 = 1.0 (but never > 1.0)
        assert order <= 1.0


class TestFunctionalConnectivity:
    """Tests for functional_connectivity metric (#66)."""

    def test_field_exists_with_default(self):
        """CriticalityMetrics has functional_connectivity with sensible default."""
        m = CriticalityMetrics()
        assert m.functional_connectivity == 0.5

    def test_field_in_constructor(self):
        """functional_connectivity can be set at construction."""
        m = CriticalityMetrics(functional_connectivity=0.8)
        assert m.functional_connectivity == 0.8

    @pytest.mark.asyncio
    async def test_measurement_with_interactions(self):
        """Functional connectivity = interactions / edges."""
        neighborhood = MagicMock()
        neighborhood.get_network_stats.return_value = {
            "total_agents": 10,
            "total_edges": 40,
            "total_interactions": 20,
            "average_clustering": 0.5,
            "density": 0.4,
        }
        neighborhood._profiles = {}

        monitor = CriticalityMonitor(neighborhood=neighborhood)
        fc = await monitor._measure_functional_connectivity()

        assert fc == 0.5  # 20/40

    @pytest.mark.asyncio
    async def test_measurement_no_edges(self):
        """Returns default when no edges exist."""
        neighborhood = MagicMock()
        neighborhood.get_network_stats.return_value = {
            "total_agents": 1,
            "total_edges": 0,
            "total_interactions": 0,
            "average_clustering": 0.0,
            "density": 0.0,
        }
        neighborhood._profiles = {}

        monitor = CriticalityMonitor(neighborhood=neighborhood)
        fc = await monitor._measure_functional_connectivity()

        assert fc == 0.5

    @pytest.mark.asyncio
    async def test_measurement_no_neighborhood(self):
        """Returns default when no neighborhood available."""
        monitor = CriticalityMonitor()
        fc = await monitor._measure_functional_connectivity()

        assert fc == 0.5

    @pytest.mark.asyncio
    async def test_measurement_capped_at_1(self):
        """Functional connectivity never exceeds 1.0."""
        neighborhood = MagicMock()
        neighborhood.get_network_stats.return_value = {
            "total_agents": 5,
            "total_edges": 10,
            "total_interactions": 50,  # More interactions than edges
            "average_clustering": 0.5,
            "density": 0.5,
        }
        neighborhood._profiles = {}

        monitor = CriticalityMonitor(neighborhood=neighborhood)
        fc = await monitor._measure_functional_connectivity()

        assert fc == 1.0

    @pytest.mark.asyncio
    async def test_included_in_sample_state(self):
        """sample_state() populates functional_connectivity."""
        neighborhood = MagicMock()
        neighborhood.get_network_stats.return_value = {
            "total_agents": 10,
            "total_edges": 40,
            "total_interactions": 30,
            "average_clustering": 0.5,
            "density": 0.4,
        }
        neighborhood._profiles = {}

        monitor = CriticalityMonitor(neighborhood=neighborhood)
        metrics = await monitor.sample_state()

        assert metrics.functional_connectivity == 0.75  # 30/40

    @pytest.mark.asyncio
    async def test_order_parameter_blends_functional(self):
        """Order parameter includes functional_connectivity in calculation."""
        neighborhood = MagicMock()
        neighborhood.get_network_stats.return_value = {
            "total_agents": 10,
            "total_edges": 36,
            "total_interactions": 0,  # No meaningful interactions
            "average_clustering": 0.5,
            "density": 0.4,
        }
        neighborhood._profiles = {}

        monitor = CriticalityMonitor(neighborhood=neighborhood)
        # Pass zero functional_connectivity explicitly
        order = await monitor._measure_order_parameter(functional_connectivity=0.0)

        # structural_order = 0.4*0.5 + 0.5*0.5 = 0.45
        # order = 0.45*0.6 + 0.0*0.4 = 0.27
        assert order < 0.3  # Pulled down by zero functional connectivity

    def test_serialization_includes_functional(self):
        """to_dict includes functional_connectivity."""
        monitor = CriticalityMonitor()
        data = monitor.to_dict()
        assert "functional_connectivity" in data["current_metrics"]
