"""Pheromone Field Conflict Detection.

Detects when opposing trace types persistently reinforce the same location,
creating signal conflicts that indicate agents are pursuing contradictory
goals in the same region. Conflict intensity feeds into fission-fusion
crisis_level, driving the swarm toward FUSION for direct coordination.

Semantic opposites (pairs that conflict when co-located):
    RESOURCE  ↔ WARNING      (value vs danger)
    PATH      ↔ FAILURE      (route vs dead end)
    SUCCESS   ↔ FAILURE      (completion vs dead end)
    EXPLORATION ↔ TERRITORY  (open vs claimed)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from hive.stigmergy import PheromoneTrace, TraceType

# Pairs of TraceTypes that semantically oppose each other.
# Key deposits claim value/progress; value deposits claim hazard/blockage.
SEMANTIC_OPPOSITES: dict[TraceType, TraceType] = {
    TraceType.RESOURCE: TraceType.WARNING,
    TraceType.PATH: TraceType.FAILURE,
    TraceType.SUCCESS: TraceType.FAILURE,
    TraceType.EXPLORATION: TraceType.TERRITORY,
}


@dataclass
class ConflictPair:
    """A detected conflict between two opposing trace types at one location."""

    location: str
    trace_a: PheromoneTrace  # The stronger trace of type_a (key in SEMANTIC_OPPOSITES)
    trace_b: PheromoneTrace  # The stronger trace of type_b (value in SEMANTIC_OPPOSITES)
    conflict_intensity: float  # Geometric mean of both intensities; range [0, 1]


class ConflictDetector:
    """Detects conflicting pheromone traces at shared locations.

    A conflict exists when two semantically opposing trace types are both
    active, non-expired, recently reinforced, and above the intensity
    threshold at the same location.
    """

    def __init__(
        self,
        intensity_threshold: float = 0.6,
        recency_window_seconds: float = 300.0,
    ) -> None:
        """
        Args:
            intensity_threshold: Both traces must exceed this value (0–1).
            recency_window_seconds: Ignore traces older than this window.
        """
        self._intensity_threshold = intensity_threshold
        self._recency_window_seconds = recency_window_seconds

    def _strongest_recent(
        self,
        traces: list[PheromoneTrace],
    ) -> PheromoneTrace | None:
        """Return the strongest non-expired, recently-reinforced trace.

        Args:
            traces: Candidate traces to evaluate.

        Returns:
            Best trace, or None if none qualify.
        """
        candidates = [
            t
            for t in traces
            if not t.is_expired and t.age_seconds < self._recency_window_seconds
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda t: t.intensity)

    def detect(
        self,
        traces: dict[str, dict[TraceType, list[PheromoneTrace]]],
        locations: list[str] | None = None,
    ) -> list[ConflictPair]:
        """Scan for conflict pairs in the pheromone field.

        Args:
            traces: The field's internal ``_traces`` dict
                    (location → trace_type → list of traces).
            locations: Optional subset of locations to scan.
                       Defaults to all locations in *traces*.

        Returns:
            List of ConflictPair objects, one per conflicting (location, pair).
        """
        conflicts: list[ConflictPair] = []
        scan_locations = locations if locations is not None else list(traces.keys())

        for location in scan_locations:
            location_traces = traces.get(location, {})

            for type_a, type_b in SEMANTIC_OPPOSITES.items():
                best_a = self._strongest_recent(location_traces.get(type_a, []))
                best_b = self._strongest_recent(location_traces.get(type_b, []))

                if best_a is None or best_b is None:
                    continue
                if best_a.intensity < self._intensity_threshold:
                    continue
                if best_b.intensity < self._intensity_threshold:
                    continue

                conflict_intensity = math.sqrt(best_a.intensity * best_b.intensity)
                conflicts.append(
                    ConflictPair(
                        location=location,
                        trace_a=best_a,
                        trace_b=best_b,
                        conflict_intensity=conflict_intensity,
                    )
                )

        return conflicts

    def compute_crisis_signal(self, conflicts: list[ConflictPair]) -> float:
        """Aggregate conflict pairs into a single crisis contribution.

        The total is the sum of all conflict intensities, capped at 1.0.
        Multiple simultaneous conflicts accumulate proportionally.

        Args:
            conflicts: Output of :meth:`detect`.

        Returns:
            Crisis signal in [0.0, 1.0]. Returns 0.0 for an empty list.
        """
        if not conflicts:
            return 0.0
        return min(1.0, sum(c.conflict_intensity for c in conflicts))
