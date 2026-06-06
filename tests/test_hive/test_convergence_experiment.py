"""Tests for the convergence experiment — emergence threshold sweep (#72)."""

from __future__ import annotations

import pytest

from hive.experiments.convergence import (
    ConvergenceExperiment,
    ConvergenceResult,
    SweepPoint,
)
from hive.topology import TopologyType

# ---------------------------------------------------------------------------
# TestSweepPoint
# ---------------------------------------------------------------------------


class TestSweepPoint:
    def test_to_dict(self):
        point = SweepPoint(
            agent_count=16,
            topology_type="swarm",
            emergence_detected=True,
            novelty_ratio=0.25,
            evidence_count=3,
            above_threshold=True,
            criticality_state="critical",
            vocab_coverage=0.6,
        )
        data = point.to_dict()
        assert data["agent_count"] == 16
        assert data["emergence_detected"] is True
        assert data["novelty_ratio"] == 0.25

    def test_frozen(self):
        point = SweepPoint(
            agent_count=8,
            topology_type="swarm",
            emergence_detected=False,
            novelty_ratio=0.0,
            evidence_count=0,
            above_threshold=True,
            criticality_state="subcritical",
            vocab_coverage=0.3,
        )
        with pytest.raises(AttributeError):
            point.agent_count = 16  # type: ignore[misc]

    def test_new_fields_have_defaults(self):
        """New criticality/network fields default to sensible values."""
        point = SweepPoint(
            agent_count=16,
            topology_type="swarm",
            emergence_detected=False,
            novelty_ratio=0.0,
            evidence_count=0,
            above_threshold=True,
            criticality_state="critical",
            vocab_coverage=0.5,
        )
        assert point.correlation_length == 0.0
        assert point.functional_connectivity == 0.0
        assert point.order_parameter == 0.5
        assert point.criticality_score == 0.0
        assert point.total_interactions == 0
        assert point.network_density == 0.0

    def test_all_14_fields_in_to_dict(self):
        """All 14 fields appear in serialized output."""
        point = SweepPoint(
            agent_count=32,
            topology_type="swarm",
            emergence_detected=True,
            novelty_ratio=0.15,
            evidence_count=3,
            above_threshold=True,
            criticality_state="critical",
            vocab_coverage=0.8,
            correlation_length=0.45,
            functional_connectivity=0.72,
            order_parameter=0.55,
            criticality_score=0.61,
            total_interactions=160,
            network_density=0.226,
        )
        d = point.to_dict()
        assert len(d) == 14
        assert d["correlation_length"] == 0.45
        assert d["functional_connectivity"] == 0.72
        assert d["total_interactions"] == 160


# ---------------------------------------------------------------------------
# TestConvergenceResult
# ---------------------------------------------------------------------------


class TestConvergenceResult:
    def _make_point(self, n: int, emerged: bool) -> SweepPoint:
        return SweepPoint(
            agent_count=n,
            topology_type="swarm",
            emergence_detected=emerged,
            novelty_ratio=0.15 if emerged else 0.0,
            evidence_count=2 if emerged else 0,
            above_threshold=True,
            criticality_state="critical",
            vocab_coverage=0.5,
        )

    def test_empty_result(self):
        result = ConvergenceResult()
        assert result.emergence_onset is None
        assert result.consistent_emergence is None

    def test_emergence_onset(self):
        result = ConvergenceResult(sweep_points=[
            self._make_point(8, False),
            self._make_point(16, False),
            self._make_point(24, True),
            self._make_point(32, True),
        ])
        assert result.emergence_onset == 24

    def test_consistent_emergence(self):
        result = ConvergenceResult(sweep_points=[
            self._make_point(8, False),
            self._make_point(16, True),
            self._make_point(24, False),  # blip
            self._make_point(32, True),
            self._make_point(40, True),
        ])
        assert result.emergence_onset == 16
        assert result.consistent_emergence == 32

    def test_no_consistent_emergence(self):
        result = ConvergenceResult(sweep_points=[
            self._make_point(8, True),
            self._make_point(16, False),
        ])
        assert result.emergence_onset == 8
        assert result.consistent_emergence is None

    def test_to_dict(self):
        result = ConvergenceResult(
            sweep_points=[self._make_point(8, False)],
            estimated_threshold=24,
            seed=42,
        )
        data = result.to_dict()
        assert data["estimated_threshold"] == 24
        assert data["seed"] == 42
        assert len(data["sweep_points"]) == 1
        # Verify new fields appear in nested SweepPoint dicts
        assert "correlation_length" in data["sweep_points"][0]
        assert "functional_connectivity" in data["sweep_points"][0]
        assert len(data["sweep_points"][0]) == 14


# ---------------------------------------------------------------------------
# TestConvergenceExperiment — pipeline integration
# ---------------------------------------------------------------------------


class TestConvergenceExperiment:
    _cached_result: ConvergenceResult | None = None

    @pytest.fixture
    async def default_result(self) -> ConvergenceResult:
        """Run default experiment once, cached across tests in this class."""
        if TestConvergenceExperiment._cached_result is None:
            TestConvergenceExperiment._cached_result = await ConvergenceExperiment(
                seed=42,
            ).run()
        return TestConvergenceExperiment._cached_result

    # --- Basic sweep behavior ---

    async def test_deterministic(self):
        """Same seed produces identical deterministic fields."""
        r1 = await ConvergenceExperiment(n_range=[8, 16, 24], seed=42).run()
        r2 = await ConvergenceExperiment(n_range=[8, 16, 24], seed=42).run()
        for p1, p2 in zip(r1.sweep_points, r2.sweep_points):
            assert p1.agent_count == p2.agent_count
            assert p1.emergence_detected == p2.emergence_detected
            assert p1.novelty_ratio == p2.novelty_ratio
            assert p1.evidence_count == p2.evidence_count
            assert p1.vocab_coverage == p2.vocab_coverage
            assert p1.above_threshold == p2.above_threshold
            assert p1.total_interactions == p2.total_interactions
            assert p1.functional_connectivity == p2.functional_connectivity
            assert p1.correlation_length == p2.correlation_length
            assert p1.order_parameter == p2.order_parameter

    async def test_different_seeds_differ(self):
        """Different seeds produce different vocabulary coverage."""
        r1 = await ConvergenceExperiment(n_range=[8], seed=1).run()
        r2 = await ConvergenceExperiment(n_range=[8], seed=999).run()
        assert r1.sweep_points[0].vocab_coverage != r2.sweep_points[0].vocab_coverage

    async def test_sweep_covers_all_n(self):
        """Sweep produces one point per N value."""
        n_range = [8, 16, 24, 32]
        result = await ConvergenceExperiment(n_range=n_range).run()
        assert len(result.sweep_points) == len(n_range)
        for i, n in enumerate(n_range):
            assert result.sweep_points[i].agent_count == n

    async def test_default_range(self, default_result: ConvergenceResult):
        """Default range is 8..128 in steps of 8."""
        assert len(default_result.sweep_points) == 16
        assert default_result.sweep_points[0].agent_count == 8
        assert default_result.sweep_points[-1].agent_count == 128

    async def test_custom_n_range(self):
        """Custom n_range produces correct number of points."""
        result = await ConvergenceExperiment(n_range=[10, 20, 50], seed=42).run()
        assert len(result.sweep_points) == 3
        assert [p.agent_count for p in result.sweep_points] == [10, 20, 50]

    # --- Emergence detection ---

    async def test_small_n_no_emergence(self):
        """Small N with low coverage should not produce emergence."""
        result = await ConvergenceExperiment(n_range=[8], seed=42).run()
        assert result.sweep_points[0].emergence_detected is False
        assert result.sweep_points[0].novelty_ratio == 0.0

    async def test_large_n_emergence(self):
        """At large N, emergence should be detected."""
        result = await ConvergenceExperiment(n_range=[128], seed=42).run()
        point = result.sweep_points[0]
        assert point.emergence_detected is True
        assert point.novelty_ratio > 0.0
        assert point.evidence_count > 0

    async def test_estimated_threshold_set(self, default_result: ConvergenceResult):
        """Estimated threshold equals consistent_emergence."""
        assert default_result.estimated_threshold == default_result.consistent_emergence

    async def test_novelty_ratio_bounded(self, default_result: ConvergenceResult):
        """Novelty ratio is always in [0, 1]."""
        for point in default_result.sweep_points:
            assert 0.0 <= point.novelty_ratio <= 1.0

    # --- Criticality metrics (pipeline integration) ---

    async def test_correlation_length_in_valid_range(self, default_result: ConvergenceResult):
        """All correlation_length values are in [0, 1]."""
        for point in default_result.sweep_points:
            assert 0.0 <= point.correlation_length <= 1.0

    async def test_criticality_metrics_populated(self, default_result: ConvergenceResult):
        """Every point has real criticality metrics from CriticalityMonitor."""
        for point in default_result.sweep_points:
            assert (
                point.correlation_length > 0
                or point.functional_connectivity > 0
                or point.criticality_score > 0
            )

    async def test_functional_connectivity_increases_with_n(
        self, default_result: ConvergenceResult,
    ):
        """Functional connectivity increases from low N to high N."""
        points = default_result.sweep_points
        assert points[-1].functional_connectivity > points[0].functional_connectivity

    async def test_order_parameter_in_valid_range(self, default_result: ConvergenceResult):
        """All order_parameter values are in [0, 1]."""
        for point in default_result.sweep_points:
            assert 0.0 <= point.order_parameter <= 1.0

    async def test_criticality_score_in_valid_range(self, default_result: ConvergenceResult):
        """All criticality_score values are in [0, 1]."""
        for point in default_result.sweep_points:
            assert 0.0 <= point.criticality_score <= 1.0

    async def test_functional_connectivity_in_valid_range(
        self, default_result: ConvergenceResult,
    ):
        """All functional_connectivity values are in [0, 1]."""
        for point in default_result.sweep_points:
            assert 0.0 <= point.functional_connectivity <= 1.0

    # --- Network metrics ---

    async def test_total_interactions_scales_with_n(self, default_result: ConvergenceResult):
        """More agents produce more total interactions."""
        points = default_result.sweep_points
        assert points[-1].total_interactions > points[0].total_interactions

    async def test_network_density_populated(self, default_result: ConvergenceResult):
        """Network density is positive for all points."""
        for point in default_result.sweep_points:
            assert point.network_density > 0

    async def test_vocab_coverage_increases_overall(self, default_result: ConvergenceResult):
        """Coverage at high N exceeds coverage at low N.

        Not strictly monotonic per step (random sampling can regress
        slightly between adjacent N values), but the overall trend
        from N=8 to N=128 is always increasing.
        """
        points = default_result.sweep_points
        assert points[-1].vocab_coverage > points[0].vocab_coverage
        # First quarter vs last quarter
        q1_avg = sum(p.vocab_coverage for p in points[:4]) / 4
        q4_avg = sum(p.vocab_coverage for p in points[-4:]) / 4
        assert q4_avg > q1_avg

    # --- Emergence threshold integration ---

    async def test_above_threshold_always_true(self, default_result: ConvergenceResult):
        """All points in default range (N>=8) are above SWARM threshold (8)."""
        for point in default_result.sweep_points:
            assert point.above_threshold is True

    async def test_small_n_below_threshold_forces_subcritical(self):
        """N < SWARM threshold (8) forces SUBCRITICAL via CriticalityMonitor gate."""
        exp = ConvergenceExperiment(n_range=[4, 6], seed=42)
        result = await exp.run()
        for point in result.sweep_points:
            assert point.above_threshold is False
            assert point.criticality_state == "subcritical"

    # --- Topology variants ---

    async def test_topology_type_recorded(self):
        """Each point records the topology type."""
        result = await ConvergenceExperiment(
            topology_type=TopologyType.HIERARCHY, n_range=[8],
        ).run()
        assert result.sweep_points[0].topology_type == "hierarchy"

    async def test_pipeline_topology(self):
        """Experiment works with non-SWARM topologies."""
        result = await ConvergenceExperiment(
            topology_type=TopologyType.PIPELINE, n_range=[8, 16], seed=42,
        ).run()
        assert len(result.sweep_points) == 2
        assert result.sweep_points[0].topology_type == "pipeline"

    # --- Serialization ---

    async def test_result_serialization(self, default_result: ConvergenceResult):
        """Full result serializes with all keys including new fields."""
        data = default_result.to_dict()
        assert "sweep_points" in data
        assert "estimated_threshold" in data
        assert "emergence_onset" in data
        assert "consistent_emergence" in data
        assert "seed" in data
        # Verify nested SweepPoint has all 14 fields
        assert len(data["sweep_points"][0]) == 14

    async def test_all_fields_populated(self):
        """Every SweepPoint has all fields with correct types."""
        result = await ConvergenceExperiment(n_range=[16]).run()
        point = result.sweep_points[0]
        assert isinstance(point.agent_count, int)
        assert isinstance(point.topology_type, str)
        assert isinstance(point.emergence_detected, bool)
        assert isinstance(point.novelty_ratio, float)
        assert isinstance(point.evidence_count, int)
        assert isinstance(point.above_threshold, bool)
        assert isinstance(point.criticality_state, str)
        assert isinstance(point.vocab_coverage, float)
        assert isinstance(point.correlation_length, float)
        assert isinstance(point.functional_connectivity, float)
        assert isinstance(point.order_parameter, float)
        assert isinstance(point.criticality_score, float)
        assert isinstance(point.total_interactions, int)
        assert isinstance(point.network_density, float)

    async def test_custom_detector_threshold(self):
        """Custom detector threshold changes emergence sensitivity."""
        low = await ConvergenceExperiment(
            n_range=[16], detector_threshold=0.01, seed=42,
        ).run()
        high = await ConvergenceExperiment(
            n_range=[16], detector_threshold=0.9, seed=42,
        ).run()
        # Same data, different threshold — ratio stays the same
        assert low.sweep_points[0].novelty_ratio == high.sweep_points[0].novelty_ratio
        # Low threshold is more sensitive
        if high.sweep_points[0].emergence_detected:
            assert low.sweep_points[0].emergence_detected
