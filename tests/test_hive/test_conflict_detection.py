"""Tests for pheromone field conflict detection (#64).

Covers:
- ConflictDetector.detect(): opposing trace pair logic
- ConflictDetector.compute_crisis_signal(): aggregation + cap
- Integration with FissionFusionManager: evaluate_state() wires the detector
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from hive.conflict import ConflictDetector, ConflictPair
from hive.fission_fusion import FissionFusionManager, FissionFusionState
from hive.stigmergy import PheromoneField, PheromoneTrace, TraceType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_trace(
    trace_type: TraceType,
    intensity: float,
    location: str = "loc_a",
    age_seconds: float = 10.0,
    ttl_seconds: float = 3600.0,
) -> PheromoneTrace:
    """Build a PheromoneTrace with a controlled age (for recency tests)."""
    trace = PheromoneTrace(
        trace_id=f"t_{trace_type}_{location}_{intensity}",
        trace_type=trace_type,
        location=location,
        intensity=intensity,
        ttl_seconds=ttl_seconds,
    )
    trace.deposited_at = datetime.now(UTC) - timedelta(seconds=age_seconds)
    return trace


# ---------------------------------------------------------------------------
# Core detection tests
# ---------------------------------------------------------------------------


class TestConflictDetection:
    """ConflictDetector.detect() — opposing-pair logic."""

    def test_opposing_traces_both_above_threshold(self):
        """Both traces above threshold on same location → one conflict detected."""
        detector = ConflictDetector(intensity_threshold=0.6)
        traces = {
            "zone_a": {
                TraceType.RESOURCE: [make_trace(TraceType.RESOURCE, 0.9, "zone_a")],
                TraceType.WARNING: [make_trace(TraceType.WARNING, 0.8, "zone_a")],
            }
        }
        conflicts = detector.detect(traces)
        assert len(conflicts) == 1
        assert conflicts[0].location == "zone_a"
        assert conflicts[0].trace_a.trace_type == TraceType.RESOURCE
        assert conflicts[0].trace_b.trace_type == TraceType.WARNING

    def test_same_location_non_opposing_types(self):
        """Non-opposing types at the same location → no conflict."""
        detector = ConflictDetector(intensity_threshold=0.6)
        # RESOURCE and SUCCESS are not in SEMANTIC_OPPOSITES against each other
        traces = {
            "zone_a": {
                TraceType.RESOURCE: [make_trace(TraceType.RESOURCE, 0.9, "zone_a")],
                TraceType.SUCCESS: [make_trace(TraceType.SUCCESS, 0.9, "zone_a")],
            }
        }
        conflicts = detector.detect(traces)
        assert len(conflicts) == 0

    def test_one_trace_below_threshold(self):
        """One trace above, opposing trace below threshold → no conflict."""
        detector = ConflictDetector(intensity_threshold=0.6)
        traces = {
            "zone_a": {
                TraceType.RESOURCE: [make_trace(TraceType.RESOURCE, 0.9, "zone_a")],
                TraceType.WARNING: [make_trace(TraceType.WARNING, 0.3, "zone_a")],
            }
        }
        conflicts = detector.detect(traces)
        assert len(conflicts) == 0

    def test_both_below_threshold(self):
        """Both traces below threshold → no conflict."""
        detector = ConflictDetector(intensity_threshold=0.6)
        traces = {
            "zone_a": {
                TraceType.RESOURCE: [make_trace(TraceType.RESOURCE, 0.4, "zone_a")],
                TraceType.WARNING: [make_trace(TraceType.WARNING, 0.5, "zone_a")],
            }
        }
        conflicts = detector.detect(traces)
        assert len(conflicts) == 0

    def test_conflict_intensity_is_geometric_mean(self):
        """conflict_intensity == sqrt(intensity_a * intensity_b)."""
        detector = ConflictDetector(intensity_threshold=0.6)
        ia, ib = 0.81, 0.64
        traces = {
            "zone_a": {
                TraceType.RESOURCE: [make_trace(TraceType.RESOURCE, ia, "zone_a")],
                TraceType.WARNING: [make_trace(TraceType.WARNING, ib, "zone_a")],
            }
        }
        conflicts = detector.detect(traces)
        assert len(conflicts) == 1
        assert abs(conflicts[0].conflict_intensity - math.sqrt(ia * ib)) < 1e-9

    def test_expired_traces_not_counted(self):
        """Traces with intensity=0 (expired) are ignored."""
        detector = ConflictDetector(intensity_threshold=0.6)
        expired = make_trace(TraceType.RESOURCE, 0.9, "zone_a")
        expired.intensity = 0.0  # forces is_expired == True
        traces = {
            "zone_a": {
                TraceType.RESOURCE: [expired],
                TraceType.WARNING: [make_trace(TraceType.WARNING, 0.9, "zone_a")],
            }
        }
        conflicts = detector.detect(traces)
        assert len(conflicts) == 0

    def test_stale_traces_beyond_recency_window(self):
        """Traces older than recency_window_seconds are excluded."""
        detector = ConflictDetector(intensity_threshold=0.6, recency_window_seconds=60.0)
        old = make_trace(TraceType.RESOURCE, 0.9, "zone_a", age_seconds=120.0)
        fresh = make_trace(TraceType.WARNING, 0.9, "zone_a", age_seconds=5.0)
        traces = {
            "zone_a": {
                TraceType.RESOURCE: [old],
                TraceType.WARNING: [fresh],
            }
        }
        conflicts = detector.detect(traces)
        assert len(conflicts) == 0

    def test_detect_multiple_locations(self):
        """Conflicts at distinct locations are all returned."""
        detector = ConflictDetector(intensity_threshold=0.6)
        traces = {
            "zone_a": {
                TraceType.RESOURCE: [make_trace(TraceType.RESOURCE, 0.9, "zone_a")],
                TraceType.WARNING: [make_trace(TraceType.WARNING, 0.8, "zone_a")],
            },
            "zone_b": {
                TraceType.SUCCESS: [make_trace(TraceType.SUCCESS, 0.85, "zone_b")],
                TraceType.FAILURE: [make_trace(TraceType.FAILURE, 0.75, "zone_b")],
            },
        }
        conflicts = detector.detect(traces)
        assert len(conflicts) == 2
        assert {c.location for c in conflicts} == {"zone_a", "zone_b"}

    def test_location_filter_restricts_scan(self):
        """Passing locations= restricts scan; other locations are skipped."""
        detector = ConflictDetector(intensity_threshold=0.6)
        traces = {
            "zone_a": {
                TraceType.RESOURCE: [make_trace(TraceType.RESOURCE, 0.9, "zone_a")],
                TraceType.WARNING: [make_trace(TraceType.WARNING, 0.8, "zone_a")],
            },
            "zone_b": {
                TraceType.SUCCESS: [make_trace(TraceType.SUCCESS, 0.9, "zone_b")],
                TraceType.FAILURE: [make_trace(TraceType.FAILURE, 0.9, "zone_b")],
            },
        }
        conflicts = detector.detect(traces, locations=["zone_b"])
        assert len(conflicts) == 1
        assert conflicts[0].location == "zone_b"


# ---------------------------------------------------------------------------
# Crisis signal tests
# ---------------------------------------------------------------------------


class TestCrisisSignal:
    """ConflictDetector.compute_crisis_signal() — aggregation logic."""

    def test_compute_crisis_empty(self):
        """No conflicts → 0.0."""
        detector = ConflictDetector()
        assert detector.compute_crisis_signal([]) == 0.0

    def test_compute_crisis_single_conflict(self):
        """Single conflict contributes its intensity directly."""
        detector = ConflictDetector()
        pair = ConflictPair(
            location="zone_a",
            trace_a=make_trace(TraceType.RESOURCE, 0.9),
            trace_b=make_trace(TraceType.WARNING, 0.71),
            conflict_intensity=0.8,
        )
        assert detector.compute_crisis_signal([pair]) == pytest.approx(0.8)

    def test_compute_crisis_multiple_capped_at_one(self):
        """Sum > 1.0 is capped at 1.0."""
        detector = ConflictDetector()
        pairs = [
            ConflictPair(
                location=f"zone_{i}",
                trace_a=make_trace(TraceType.RESOURCE, 0.9),
                trace_b=make_trace(TraceType.WARNING, 0.9),
                conflict_intensity=0.9,
            )
            for i in range(3)
        ]
        assert detector.compute_crisis_signal(pairs) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Integration: ConflictDetector wired into FissionFusionManager
# ---------------------------------------------------------------------------


class TestIntegration:
    """FissionFusionManager with conflict_detector parameter."""

    def test_no_conflict_detector_no_behavior_change(self):
        """Without a detector, _conflict_crisis_level stays 0.0."""
        manager = FissionFusionManager()
        assert manager._conflict_crisis_level == 0.0
        # _resolve_crisis_level with no monitor, no override → 0.0
        assert manager._resolve_crisis_level() == 0.0

    async def test_field_conflict_raises_crisis_level(self):
        """After evaluate_state(), conflicting field traces raise _conflict_crisis_level."""
        field = PheromoneField()
        field._traces["zone_a"] = {
            TraceType.RESOURCE: [make_trace(TraceType.RESOURCE, 0.9, "zone_a")],
            TraceType.WARNING: [make_trace(TraceType.WARNING, 0.85, "zone_a")],
        }
        detector = ConflictDetector(intensity_threshold=0.6)
        manager = FissionFusionManager(
            pheromone_field=field,
            conflict_detector=detector,
        )
        await manager.evaluate_state()
        assert manager._conflict_crisis_level > 0.6

    async def test_field_conflict_can_trigger_fusion(self):
        """Saturated conflict → crisis_level ≥ 0.6 → suggest_state() returns FUSION."""
        field = PheromoneField()
        field._traces["zone_a"] = {
            TraceType.RESOURCE: [make_trace(TraceType.RESOURCE, 1.0, "zone_a")],
            TraceType.WARNING: [make_trace(TraceType.WARNING, 1.0, "zone_a")],
        }
        detector = ConflictDetector(intensity_threshold=0.6)
        manager = FissionFusionManager(
            pheromone_field=field,
            conflict_detector=detector,
        )
        metrics = await manager.evaluate_state()
        assert metrics.suggest_state() == FissionFusionState.FUSION

    async def test_conflict_event_emitted_on_detection(self):
        """When conflicts are found, CONFLICT_DETECTED is emitted on the event bus."""
        from hive.events import EventBus, EventType

        bus = EventBus()
        received: list = []

        async def handler(event):  # type: ignore[misc]
            received.append(event)

        bus.subscribe(EventType.CONFLICT_DETECTED, handler)

        field = PheromoneField()
        field._traces["zone_a"] = {
            TraceType.RESOURCE: [make_trace(TraceType.RESOURCE, 0.9, "zone_a")],
            TraceType.WARNING: [make_trace(TraceType.WARNING, 0.85, "zone_a")],
        }
        detector = ConflictDetector(intensity_threshold=0.6)
        manager = FissionFusionManager(
            pheromone_field=field,
            event_bus=bus,
            conflict_detector=detector,
        )
        await manager.evaluate_state()

        assert len(received) == 1
        assert received[0].payload["conflict_count"] == 1

    async def test_conflict_floor_raises_above_manual_override(self):
        """Conflict floor raises crisis even when manual override is lower."""
        field = PheromoneField()
        field._traces["zone_a"] = {
            TraceType.RESOURCE: [make_trace(TraceType.RESOURCE, 1.0, "zone_a")],
            TraceType.WARNING: [make_trace(TraceType.WARNING, 1.0, "zone_a")],
        }
        detector = ConflictDetector(intensity_threshold=0.6)
        manager = FissionFusionManager(
            pheromone_field=field,
            conflict_detector=detector,
        )
        manager.set_crisis_level(0.1)  # manually set low
        await manager.evaluate_state()
        # conflict_crisis_level should be 1.0 (saturated), floor raises result
        resolved = manager._resolve_crisis_level()
        assert resolved >= manager._conflict_crisis_level
