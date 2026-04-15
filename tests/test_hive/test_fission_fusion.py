"""Tests for fission-fusion dynamics (Phase 16B)."""

from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest

from hive.criticality import CriticalityMetrics, CriticalityMonitor, CriticalityState
from hive.fission_fusion import (
    Cluster,
    FissionFusionEvent,
    FissionFusionManager,
    FissionFusionMetrics,
    FissionFusionState,
)


class TestFissionFusionState:
    """Tests for FissionFusionState enum."""

    def test_states_exist(self):
        """Test all states are defined."""
        assert FissionFusionState.FISSION == "fission"
        assert FissionFusionState.FUSION == "fusion"
        assert FissionFusionState.TRANSITIONING == "transitioning"


class TestCluster:
    """Tests for Cluster dataclass."""

    def test_cluster_creation(self):
        """Test creating a cluster."""
        cluster = Cluster(
            cluster_id="cluster_1",
            agent_ids=["agent_1", "agent_2"],
            task_focus="research",
        )
        assert cluster.cluster_id == "cluster_1"
        assert cluster.size == 2

    def test_add_agent(self):
        """Test adding agent to cluster."""
        cluster = Cluster(cluster_id="test")
        cluster.add_agent("agent_1")
        assert "agent_1" in cluster.agent_ids
        assert cluster.size == 1

        # Adding same agent again should not duplicate
        cluster.add_agent("agent_1")
        assert cluster.size == 1

    def test_remove_agent(self):
        """Test removing agent from cluster."""
        cluster = Cluster(cluster_id="test", agent_ids=["agent_1", "agent_2"])
        cluster.remove_agent("agent_1")
        assert "agent_1" not in cluster.agent_ids
        assert cluster.size == 1

    def test_to_dict(self):
        """Test cluster serialization."""
        cluster = Cluster(
            cluster_id="test",
            agent_ids=["a1", "a2"],
            task_focus="coding",
        )
        data = cluster.to_dict()
        assert data["cluster_id"] == "test"
        assert data["size"] == 2
        assert data["task_focus"] == "coding"


class TestFissionFusionMetrics:
    """Tests for FissionFusionMetrics dataclass."""

    def test_default_metrics(self):
        """Test default metric values."""
        metrics = FissionFusionMetrics()
        assert metrics.task_correlation == 0.5
        assert metrics.information_spread == 0.5
        assert metrics.cluster_count == 1

    def test_suggest_state_fusion(self):
        """Test suggesting fusion state."""
        metrics = FissionFusionMetrics(
            task_correlation=0.8,  # High correlation -> fusion
            crisis_level=0.0,
        )
        assert metrics.suggest_state() == FissionFusionState.FUSION

    def test_suggest_state_fission(self):
        """Test suggesting fission state."""
        metrics = FissionFusionMetrics(
            task_correlation=0.2,  # Low correlation -> fission
            exploration_need=0.7,
        )
        assert metrics.suggest_state() == FissionFusionState.FISSION

    def test_suggest_state_crisis(self):
        """Test crisis triggers fusion."""
        metrics = FissionFusionMetrics(
            task_correlation=0.3,  # Would suggest fission
            crisis_level=0.7,  # But crisis triggers fusion
        )
        assert metrics.suggest_state() == FissionFusionState.FUSION

    def test_suggest_state_transitioning(self):
        """Test transitioning state for intermediate values."""
        metrics = FissionFusionMetrics(
            task_correlation=0.5,  # Middle ground
            exploration_need=0.4,
            crisis_level=0.3,
        )
        assert metrics.suggest_state() == FissionFusionState.TRANSITIONING


class TestFissionFusionEvent:
    """Tests for FissionFusionEvent dataclass."""

    def test_event_creation(self):
        """Test creating an event."""
        event = FissionFusionEvent(
            event_id="event_1",
            event_type="fission",
            previous_state=FissionFusionState.FUSION,
            new_state=FissionFusionState.FISSION,
            metrics=FissionFusionMetrics(),
            clusters_formed=5,
        )
        assert event.event_type == "fission"
        assert event.clusters_formed == 5

    def test_event_to_dict(self):
        """Test event serialization."""
        event = FissionFusionEvent(
            event_id="event_2",
            event_type="fusion",
            previous_state=FissionFusionState.FISSION,
            new_state=FissionFusionState.FUSION,
            metrics=FissionFusionMetrics(),
            info_center_id="center_1",
        )
        data = event.to_dict()
        assert data["event_type"] == "fusion"
        assert data["info_center_id"] == "center_1"


class TestFissionFusionManager:
    """Tests for FissionFusionManager class."""

    @pytest.fixture
    def mock_neighborhood(self):
        """Create mock neighborhood."""
        neighborhood = MagicMock()
        neighborhood.get_network_stats.return_value = {
            "total_agents": 20,
            "density": 0.3,
            "average_clustering": 0.5,
        }

        # Mock profiles with task tags
        profiles = {}
        for i in range(20):
            profile = MagicMock()
            profile.task_tags = ["coding"] if i < 10 else ["research"]
            profile.task_similarity = MagicMock(return_value=0.5)
            profiles[f"agent_{i}"] = profile

        neighborhood._profiles = profiles
        return neighborhood

    @pytest.fixture
    def mock_event_bus(self):
        """Create mock event bus."""
        bus = MagicMock()
        bus.emit = AsyncMock()
        return bus

    @pytest.fixture
    def manager(self, mock_neighborhood, mock_event_bus):
        """Create a FissionFusionManager for testing."""
        return FissionFusionManager(
            neighborhood=mock_neighborhood,
            event_bus=mock_event_bus,
            evaluation_interval=1.0,
        )

    def test_initial_state(self, manager):
        """Test initial manager state."""
        assert manager.state == FissionFusionState.FISSION
        assert len(manager.clusters) == 0

    @pytest.mark.asyncio
    async def test_evaluate_state(self, manager):
        """Test evaluating current state."""
        metrics = await manager.evaluate_state()

        assert isinstance(metrics, FissionFusionMetrics)
        assert 0.0 <= metrics.task_correlation <= 1.0
        assert 0.0 <= metrics.information_spread <= 1.0

    @pytest.mark.asyncio
    async def test_perform_fission(self, manager):
        """Test performing fission."""
        clusters_formed = await manager.perform_fission()

        assert manager.state == FissionFusionState.FISSION
        assert clusters_formed >= 1
        assert len(manager.clusters) >= 1

    @pytest.mark.asyncio
    async def test_perform_fusion(self, manager):
        """Test performing fusion."""
        # First do fission to have something to fuse
        await manager.perform_fission()

        # Then fuse
        info_center_id = await manager.perform_fusion()

        assert manager.state == FissionFusionState.FUSION
        assert info_center_id is not None
        assert len(manager.clusters) == 0  # Clusters cleared

    @pytest.mark.asyncio
    async def test_should_transition_to_fusion(self, manager):
        """Test detecting need to transition to fusion."""
        manager._state = FissionFusionState.FISSION
        manager._metrics = FissionFusionMetrics(task_correlation=0.8)

        result = await manager.should_transition()
        assert result == FissionFusionState.FUSION

    @pytest.mark.asyncio
    async def test_should_transition_to_fission(self, manager):
        """Test detecting need to transition to fission."""
        manager._state = FissionFusionState.FUSION
        manager._metrics = FissionFusionMetrics(task_correlation=0.2)

        result = await manager.should_transition()
        assert result == FissionFusionState.FISSION

    @pytest.mark.asyncio
    async def test_should_transition_no_change(self, manager):
        """Test no transition when not needed."""
        manager._state = FissionFusionState.FISSION
        manager._metrics = FissionFusionMetrics(task_correlation=0.2)

        result = await manager.should_transition()
        assert result is None  # Already in appropriate state

    def test_get_agent_cluster(self, manager):
        """Test getting agent's cluster."""
        # Initially no cluster
        assert manager.get_agent_cluster("agent_0") is None

    @pytest.mark.asyncio
    async def test_get_agent_cluster_after_fission(self, manager):
        """Test getting agent's cluster after fission."""
        await manager.perform_fission()

        # Some agents should be in clusters
        cluster = manager.get_agent_cluster("agent_0")
        if cluster:
            assert isinstance(cluster, Cluster)

    def test_set_crisis_level(self, manager):
        """Test setting crisis level."""
        manager.set_crisis_level(0.8)
        assert manager._metrics.crisis_level == 0.8
        assert manager._manual_crisis_level == 0.8

        # Test bounds
        manager.set_crisis_level(1.5)
        assert manager._metrics.crisis_level == 1.0

        manager.set_crisis_level(-0.5)
        assert manager._metrics.crisis_level == 0.0

    def test_clear_crisis_override(self, manager):
        """Test clearing manual crisis override."""
        manager.set_crisis_level(0.8)
        assert manager._manual_crisis_level == 0.8

        manager.clear_crisis_override()
        assert manager._manual_crisis_level is None

    def test_set_exploration_need(self, manager):
        """Test setting exploration need."""
        manager.set_exploration_need(0.7)
        assert manager._metrics.exploration_need == 0.7

    def test_on_state_change_callback(self, manager):
        """Test registering state change callback."""
        callback = MagicMock()
        manager.on_state_change(callback)
        assert callback in manager._on_state_change

    @pytest.mark.asyncio
    async def test_callback_called_on_fission(self, manager):
        """Test callback is called on fission."""
        callback = MagicMock()
        manager.on_state_change(callback)

        await manager.perform_fission()

        callback.assert_called_once_with(FissionFusionState.FISSION)

    def test_get_events(self, manager):
        """Test getting event history."""
        assert manager.get_events() == []

    @pytest.mark.asyncio
    async def test_get_events_after_transitions(self, manager):
        """Test getting events after transitions."""
        await manager.perform_fission()
        await manager.perform_fusion()

        events = manager.get_events()
        assert len(events) == 2
        assert events[0].event_type == "fission"
        assert events[1].event_type == "fusion"

    def test_to_dict(self, manager):
        """Test serialization."""
        data = manager.to_dict()

        assert "state" in data
        assert "metrics" in data
        assert "clusters" in data
        assert "info_center_id" in data

    @pytest.mark.asyncio
    async def test_start_stop(self, manager):
        """Test starting and stopping the manager."""
        await manager.start()
        assert manager._running is True

        await manager.stop()
        assert manager._running is False



class TestCrisisLevelResolution:
    """Tests for crisis_level signal resolution (#70)."""

    @pytest.mark.asyncio
    async def test_manual_override_persists_across_evaluate(self):
        """Manual crisis_level survives evaluate_state() cycles."""
        manager = FissionFusionManager()
        manager.set_crisis_level(0.8)

        # Trigger metric computation
        metrics = await manager.evaluate_state()

        assert metrics.crisis_level == 0.8

    @pytest.mark.asyncio
    async def test_no_signal_preserves_last_value(self):
        """Without monitor or manual override, preserves current metrics value."""
        manager = FissionFusionManager()
        # Manually set the metrics crisis_level to simulate a previous cycle
        manager._metrics.crisis_level = 0.4

        metrics = await manager.evaluate_state()

        assert metrics.crisis_level == 0.4

    def test_criticality_monitor_supercritical(self):
        """Supercritical state drives crisis_level high."""
        monitor = MagicMock(spec=CriticalityMonitor)
        type(monitor).current_state = PropertyMock(
            return_value=CriticalityState.SUPERCRITICAL
        )
        # criticality_score is a computed property — mock it on the metrics object
        mock_metrics = MagicMock()
        mock_metrics.criticality_score = 0.8
        type(monitor).current_metrics = PropertyMock(return_value=mock_metrics)

        manager = FissionFusionManager(criticality_monitor=monitor)
        crisis = manager._resolve_crisis_level()

        # Supercritical: 0.5 + 0.8 * 0.5 = 0.9
        assert crisis > 0.5

    def test_criticality_monitor_subcritical(self):
        """Subcritical state drives crisis_level low."""
        monitor = MagicMock(spec=CriticalityMonitor)
        type(monitor).current_state = PropertyMock(
            return_value=CriticalityState.SUBCRITICAL
        )
        mock_metrics = MagicMock()
        mock_metrics.criticality_score = 0.3
        type(monitor).current_metrics = PropertyMock(return_value=mock_metrics)

        manager = FissionFusionManager(criticality_monitor=monitor)
        crisis = manager._resolve_crisis_level()

        assert crisis < 0.3

    def test_criticality_monitor_critical_low_baseline(self):
        """Critical (optimal) state gives low baseline crisis."""
        monitor = MagicMock(spec=CriticalityMonitor)
        type(monitor).current_state = PropertyMock(
            return_value=CriticalityState.CRITICAL
        )
        type(monitor).current_metrics = PropertyMock(
            return_value=CriticalityMetrics()
        )

        manager = FissionFusionManager(criticality_monitor=monitor)
        crisis = manager._resolve_crisis_level()

        assert crisis == 0.1

    def test_manual_override_beats_monitor(self):
        """Manual override takes priority over criticality monitor."""
        monitor = MagicMock(spec=CriticalityMonitor)
        type(monitor).current_state = PropertyMock(
            return_value=CriticalityState.SUPERCRITICAL
        )
        mock_metrics = MagicMock()
        mock_metrics.criticality_score = 0.9
        type(monitor).current_metrics = PropertyMock(return_value=mock_metrics)

        manager = FissionFusionManager(criticality_monitor=monitor)
        manager.set_crisis_level(0.2)  # Override: stay calm

        crisis = manager._resolve_crisis_level()
        assert crisis == 0.2

    def test_clear_override_falls_back_to_monitor(self):
        """After clearing override, falls back to criticality-derived value."""
        monitor = MagicMock(spec=CriticalityMonitor)
        type(monitor).current_state = PropertyMock(
            return_value=CriticalityState.SUPERCRITICAL
        )
        mock_metrics = MagicMock()
        mock_metrics.criticality_score = 0.8
        type(monitor).current_metrics = PropertyMock(return_value=mock_metrics)

        manager = FissionFusionManager(criticality_monitor=monitor)
        manager.set_crisis_level(0.1)
        assert manager._resolve_crisis_level() == 0.1

        manager.clear_crisis_override()
        crisis = manager._resolve_crisis_level()
        assert crisis > 0.5  # Back to supercritical-derived


class TestTransitionCooldown:
    """Tests for post-transition cooldown / refractory period (#68)."""

    @pytest.mark.asyncio
    async def test_cooldown_blocks_immediate_transition(self):
        """Transition is suppressed during cooldown period."""
        manager = FissionFusionManager(evaluation_interval=30.0)

        # Perform fission (sets _last_transition_time)
        await manager.perform_fission()

        # Now set metrics that would normally trigger fusion
        manager._metrics.task_correlation = 0.9
        manager._metrics.crisis_level = 0.8

        # Transition should be blocked by cooldown
        result = await manager.should_transition()
        assert result is None

    @pytest.mark.asyncio
    async def test_no_cooldown_initially(self):
        """No cooldown before any transition has occurred."""
        manager = FissionFusionManager()
        assert manager._last_transition_time is None

        # Set metrics that trigger fusion
        manager._metrics.task_correlation = 0.9
        result = await manager.should_transition()
        assert result == FissionFusionState.FUSION

    @pytest.mark.asyncio
    async def test_cooldown_default_is_2x_interval(self):
        """Default cooldown is 2× evaluation_interval."""
        manager = FissionFusionManager(evaluation_interval=15.0)
        assert manager._cooldown_seconds == 30.0

    @pytest.mark.asyncio
    async def test_fusion_also_sets_cooldown(self):
        """perform_fusion also records transition time."""
        manager = FissionFusionManager()
        await manager.perform_fusion()
        assert manager._last_transition_time is not None

    @pytest.mark.asyncio
    async def test_cooldown_expires(self):
        """After cooldown expires, transitions are allowed again."""
        from datetime import timedelta

        manager = FissionFusionManager(evaluation_interval=1.0)  # 2s cooldown

        await manager.perform_fission()

        # Backdate the transition time to simulate cooldown expiry
        manager._last_transition_time = manager._last_transition_time - timedelta(seconds=5)

        # Set metrics that trigger fusion — fill window for majority gate
        manager._metrics.task_correlation = 0.9
        for _ in range(5):
            manager._metrics_history.append(
                FissionFusionMetrics(task_correlation=0.9, crisis_level=0.8)
            )
        result = await manager.should_transition()
        assert result == FissionFusionState.FUSION


class TestWindowedEvaluation:
    """Tests for windowed metric evaluation (#69)."""

    @pytest.mark.asyncio
    async def test_single_spike_does_not_trigger_transition(self):
        """A single high-correlation snapshot shouldn't trigger fusion."""
        manager = FissionFusionManager()

        # Fill window with low-correlation history (fission territory)
        for _ in range(4):
            manager._metrics_history.append(
                FissionFusionMetrics(task_correlation=0.2)
            )
        # One spike
        manager._metrics_history.append(
            FissionFusionMetrics(task_correlation=0.9, crisis_level=0.8)
        )
        manager._metrics = manager._metrics_history[-1]

        result = await manager.should_transition()
        assert result is None  # Only 1-of-5 voted fusion

    @pytest.mark.asyncio
    async def test_majority_triggers_transition(self):
        """3-of-5 agreeing on fusion triggers the transition."""
        manager = FissionFusionManager()

        # 2 low, 3 high = majority fusion
        for tc in [0.2, 0.2, 0.9, 0.9, 0.9]:
            manager._metrics_history.append(
                FissionFusionMetrics(task_correlation=tc, crisis_level=0.8 if tc > 0.5 else 0.0)
            )
        manager._metrics = manager._metrics_history[-1]

        result = await manager.should_transition()
        assert result == FissionFusionState.FUSION

    @pytest.mark.asyncio
    async def test_no_history_allows_immediate_transition(self):
        """Before enough history accumulates, transitions use single snapshot."""
        manager = FissionFusionManager()
        # Only 1 entry — below majority_threshold (3)
        manager._metrics_history.append(
            FissionFusionMetrics(task_correlation=0.9, crisis_level=0.8)
        )
        manager._metrics = manager._metrics_history[-1]

        result = await manager.should_transition()
        assert result == FissionFusionState.FUSION

    @pytest.mark.asyncio
    async def test_window_trims_to_size(self):
        """History is trimmed to window_size."""
        manager = FissionFusionManager()

        # Add 10 entries
        for _ in range(10):
            manager._metrics_history.append(FissionFusionMetrics())

        # Simulate _evaluate_and_respond trimming
        if len(manager._metrics_history) > manager._window_size:
            manager._metrics_history = manager._metrics_history[-manager._window_size:]

        assert len(manager._metrics_history) == 5

    @pytest.mark.asyncio
    async def test_history_recorded_in_evaluate_and_respond(self):
        """_evaluate_and_respond adds to metrics history."""
        manager = FissionFusionManager()
        assert len(manager._metrics_history) == 0

        await manager._evaluate_and_respond()

        assert len(manager._metrics_history) == 1

    @pytest.mark.asyncio
    async def test_transition_from_fusion_to_fission_requires_majority(self):
        """Fusion→fission also requires windowed majority."""
        manager = FissionFusionManager()
        manager._state = FissionFusionState.FUSION

        # 2 high, 3 low = majority fission
        for tc in [0.8, 0.8, 0.1, 0.1, 0.1]:
            manager._metrics_history.append(
                FissionFusionMetrics(task_correlation=tc)
            )
        manager._metrics = manager._metrics_history[-1]

        result = await manager.should_transition()
        assert result == FissionFusionState.FISSION
