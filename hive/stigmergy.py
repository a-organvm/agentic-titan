"""Stigmergic Communication Layer.

Implements environment-mediated indirect communication inspired by
ant pheromone systems. Agents communicate by depositing traces in
a shared environment, and other agents respond to these traces.

Key concepts:
- PheromoneTrace: A deposited signal in the environment
- PheromoneField: The shared environment containing traces
- Decay: Traces evaporate over time
- Diffusion: Traces spread to nearby locations
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from titan.metrics import get_metrics

logger = logging.getLogger("titan.hive.stigmergy")


class TraceType(StrEnum):
    """Types of pheromone traces agents can deposit."""

    PATH = "path"  # Successful solution routes
    RESOURCE = "resource"  # Valuable information found
    WARNING = "warning"  # Danger/failure indicators
    SUCCESS = "success"  # Task completion markers
    FAILURE = "failure"  # Dead ends to avoid
    COLLABORATION = "collaboration"  # Help requests
    EXPLORATION = "exploration"  # Areas being explored
    TERRITORY = "territory"  # Claimed regions


@dataclass
class PheromoneTrace:
    """A single pheromone trace deposited in the environment.

    Traces are deposited by agents and can be sensed by other agents.
    They decay over time and can diffuse to nearby locations.
    """

    trace_id: str
    trace_type: TraceType
    location: str  # Logical location key
    intensity: float  # 0.0 to 1.0
    payload: dict[str, Any] = field(default_factory=dict)
    depositor_id: str = ""
    deposited_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    decay_rate: float = 0.1  # Intensity loss per decay cycle
    ttl_seconds: float = 3600.0  # Time to live

    @property
    def age_seconds(self) -> float:
        """Age of the trace in seconds."""
        now = datetime.now(UTC)
        return (now - self.deposited_at).total_seconds()

    @property
    def is_expired(self) -> bool:
        """Whether the trace has expired."""
        return self.age_seconds > self.ttl_seconds or self.intensity <= 0.0

    def decay(self, cycles: int = 1) -> float:
        """Apply decay to the trace.

        Args:
            cycles: Number of decay cycles to apply.

        Returns:
            New intensity after decay.
        """
        for _ in range(cycles):
            self.intensity *= 1.0 - self.decay_rate
        self.intensity = max(0.0, self.intensity)
        return self.intensity

    def reinforce(self, amount: float) -> float:
        """Reinforce the trace intensity.

        Args:
            amount: Amount to add (will be bounded to 1.0).

        Returns:
            New intensity after reinforcement.
        """
        self.intensity = min(1.0, self.intensity + amount)
        return self.intensity

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trace_id": self.trace_id,
            "trace_type": self.trace_type.value,
            "location": self.location,
            "intensity": self.intensity,
            "payload": self.payload,
            "depositor_id": self.depositor_id,
            "deposited_at": self.deposited_at.isoformat(),
            "decay_rate": self.decay_rate,
            "ttl_seconds": self.ttl_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PheromoneTrace:
        """Create from dictionary."""
        deposited_at = data.get("deposited_at")
        if isinstance(deposited_at, str):
            deposited_at = datetime.fromisoformat(deposited_at)
        elif deposited_at is None:
            deposited_at = datetime.now(UTC)

        return cls(
            trace_id=data["trace_id"],
            trace_type=TraceType(data["trace_type"]),
            location=data["location"],
            intensity=data.get("intensity", 1.0),
            payload=data.get("payload", {}),
            depositor_id=data.get("depositor_id", ""),
            deposited_at=deposited_at,
            decay_rate=data.get("decay_rate", 0.1),
            ttl_seconds=data.get("ttl_seconds", 3600.0),
        )


@dataclass
class GradientInfo:
    """Information about a pheromone gradient at a location."""

    trace_type: TraceType
    location: str
    local_intensity: float
    strongest_neighbor: str | None
    strongest_neighbor_intensity: float
    gradient_direction: str | None  # Direction of increasing intensity


class PheromoneField:
    """Shared environment for stigmergic communication.

    Manages pheromone traces across a logical space, handling:
    - Trace deposition and sensing
    - Gradient calculation
    - Automatic decay
    - Diffusion to neighbors
    """

    DEFAULT_DECAY_INTERVAL = 60.0  # Seconds between decay cycles
    DEFAULT_DIFFUSION_RATE = 0.05  # Fraction of intensity that diffuses

    def __init__(
        self,
        field_id: str = "default",
        decay_interval: float = DEFAULT_DECAY_INTERVAL,
        diffusion_rate: float = DEFAULT_DIFFUSION_RATE,
        neighbor_map: dict[str, list[str]] | None = None,
        redis_client: Any = None,
    ) -> None:
        """Initialize the pheromone field.

        Args:
            field_id: Identifier for this field.
            decay_interval: Seconds between automatic decay cycles.
            diffusion_rate: Fraction of intensity that spreads to neighbors.
            neighbor_map: Dict mapping location -> list of neighbor locations.
            redis_client: Optional Redis client for distributed storage.
        """
        self._field_id = field_id
        self._decay_interval = decay_interval
        self._diffusion_rate = diffusion_rate
        self._neighbor_map = neighbor_map or {}
        self._redis = redis_client

        # In-memory storage: location -> trace_type -> list[PheromoneTrace]
        self._traces: dict[str, dict[TraceType, list[PheromoneTrace]]] = {}

        # Background task handles
        self._decay_task: asyncio.Task[None] | None = None
        self._running = False
        self._trace_counter = 0

    @property
    def field_id(self) -> str:
        """Get the field ID."""
        return self._field_id

    async def start(self) -> None:
        """Start background decay cycle."""
        if self._running:
            return

        self._running = True
        self._decay_task = asyncio.create_task(self._decay_loop())
        logger.info(f"Started pheromone field: {self._field_id}")

    async def stop(self) -> None:
        """Stop background processes."""
        self._running = False
        if self._decay_task:
            self._decay_task.cancel()
            try:
                await self._decay_task
            except asyncio.CancelledError:
                pass
        logger.info(f"Stopped pheromone field: {self._field_id}")

    async def _decay_loop(self) -> None:
        """Background loop for trace decay."""
        while self._running:
            try:
                await asyncio.sleep(self._decay_interval)
                await self.decay_cycle()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in decay loop: {e}")

    def set_neighbors(self, location: str, neighbors: list[str]) -> None:
        """Set the neighbors for a location.

        Args:
            location: The location.
            neighbors: List of neighboring location keys.
        """
        self._neighbor_map[location] = neighbors

    def get_neighbors(self, location: str) -> list[str]:
        """Get the neighbors of a location.

        Args:
            location: The location.

        Returns:
            List of neighbor location keys.
        """
        return self._neighbor_map.get(location, [])

    async def deposit(
        self,
        agent_id: str,
        trace_type: TraceType,
        location: str,
        intensity: float = 1.0,
        payload: dict[str, Any] | None = None,
        decay_rate: float = 0.1,
        ttl_seconds: float = 3600.0,
    ) -> PheromoneTrace:
        """Deposit a pheromone trace at a location.

        Args:
            agent_id: ID of the depositing agent.
            trace_type: Type of trace to deposit.
            location: Location key where trace is deposited.
            intensity: Initial intensity (0-1).
            payload: Optional data payload.
            decay_rate: Rate of decay per cycle.
            ttl_seconds: Time to live in seconds.

        Returns:
            The created PheromoneTrace.
        """
        self._trace_counter += 1
        trace_id = f"trace_{self._field_id}_{self._trace_counter}"

        trace = PheromoneTrace(
            trace_id=trace_id,
            trace_type=trace_type,
            location=location,
            intensity=min(1.0, max(0.0, intensity)),
            payload=payload or {},
            depositor_id=agent_id,
            decay_rate=decay_rate,
            ttl_seconds=ttl_seconds,
        )

        # Store trace
        if location not in self._traces:
            self._traces[location] = {}
        if trace_type not in self._traces[location]:
            self._traces[location][trace_type] = []

        # Check for existing trace from same depositor at same location
        existing = self._find_existing_trace(location, trace_type, agent_id)
        if existing:
            # Reinforce existing trace instead of creating new
            existing.reinforce(intensity * 0.5)
            logger.debug(
                f"Reinforced trace at {location}: type={trace_type.value}, "
                f"intensity={existing.intensity:.2f}"
            )
            return existing

        self._traces[location][trace_type].append(trace)

        # Store in Redis if available
        if self._redis:
            await self._store_trace_redis(trace)

        # Record metrics
        metrics = get_metrics()
        metrics.pheromone_deposited(trace_type.value, location, trace.intensity)

        logger.debug(
            f"Deposited trace at {location}: type={trace_type.value}, "
            f"intensity={intensity:.2f}, depositor={agent_id}"
        )

        return trace

    def _find_existing_trace(
        self,
        location: str,
        trace_type: TraceType,
        depositor_id: str,
    ) -> PheromoneTrace | None:
        """Find an existing trace from the same depositor."""
        traces = self._traces.get(location, {}).get(trace_type, [])
        for trace in traces:
            if trace.depositor_id == depositor_id and not trace.is_expired:
                return trace
        return None

    async def _store_trace_redis(self, trace: PheromoneTrace) -> None:
        """Store trace in Redis."""
        if not self._redis:
            return

        try:
            key = f"titan:pheromone:{self._field_id}:{trace.location}:{trace.trace_type.value}"
            await self._redis.hset(key, trace.trace_id, trace.to_dict())
            await self._redis.expire(key, int(trace.ttl_seconds))
        except Exception as e:
            logger.warning(f"Failed to store trace in Redis: {e}")

    async def sense(
        self,
        location: str,
        trace_types: list[TraceType] | None = None,
        min_intensity: float = 0.0,
    ) -> list[PheromoneTrace]:
        """Sense all traces at a location.

        Args:
            location: Location to sense.
            trace_types: Optional filter for specific trace types.
            min_intensity: Minimum intensity to include.

        Returns:
            List of PheromoneTrace objects at the location.
        """
        result: list[PheromoneTrace] = []

        location_traces = self._traces.get(location, {})

        for ttype, traces in location_traces.items():
            if trace_types and ttype not in trace_types:
                continue

            for trace in traces:
                if not trace.is_expired and trace.intensity >= min_intensity:
                    result.append(trace)

        return result

    async def sense_filtered(
        self,
        region: Any,
    ) -> list[PheromoneTrace]:
        """Sense traces within a perceptual region.

        Accepts a SensingRegion (from hive.topology) and returns all
        traces that pass its location, type, and intensity filters.
        This is the perceptual gating interface: topology constrains
        what an agent can perceive, not what gets deposited.

        Args:
            region: A SensingRegion defining the perceptual filter.
                Uses Any type to avoid circular import with topology.

        Returns:
            List of traces visible within the sensing region.
        """
        result: list[PheromoneTrace] = []

        region_locations = getattr(region, "locations", None)
        transition_buffer = getattr(region, "transition_buffer", None)
        invariant_types = getattr(region, "invariant_types", None)

        if region_locations is None or transition_buffer is not None or invariant_types:
            locations_to_scan = list(self._traces.keys())
        else:
            locations_to_scan = [
                loc for loc in region_locations
                if loc in self._traces
            ]

        min_intensity = getattr(region, "min_intensity", 0.0)

        for location in locations_to_scan:
            location_traces = self._traces[location]
            for ttype, traces in location_traces.items():
                for trace in traces:
                    if trace.is_expired:
                        continue

                    extra_type = trace.payload.get("atom_type")
                    is_invariant = region.is_invariant_type(ttype, extra_type)
                    if is_invariant:
                        result.append(trace)
                        continue

                    if not region.allows_trace_type(ttype, extra_type):
                        continue

                    is_local = region_locations is None or location in region_locations
                    visible_intensity = trace.intensity
                    if not is_local:
                        if transition_buffer is None or transition_buffer <= 0.0:
                            continue
                        visible_intensity *= transition_buffer

                    if visible_intensity < min_intensity:
                        continue

                    if visible_intensity == trace.intensity:
                        result.append(trace)
                    else:
                        result.append(replace(trace, intensity=visible_intensity))

        return result

    async def sense_radius(
        self,
        location: str,
        trace_types: list[TraceType] | None = None,
        radius: int = 1,
    ) -> dict[str, list[PheromoneTrace]]:
        """Sense traces at a location and its neighbors.

        Args:
            location: Center location.
            trace_types: Optional filter for specific trace types.
            radius: Number of neighbor hops to include.

        Returns:
            Dict mapping location -> list of traces.
        """
        result: dict[str, list[PheromoneTrace]] = {}
        visited: set[str] = set()
        to_visit = [(location, 0)]

        while to_visit:
            loc, depth = to_visit.pop(0)
            if loc in visited:
                continue
            visited.add(loc)

            traces = await self.sense(loc, trace_types)
            if traces:
                result[loc] = traces

            if depth < radius:
                for neighbor in self.get_neighbors(loc):
                    if neighbor not in visited:
                        to_visit.append((neighbor, depth + 1))

        return result

    async def sense_gradient(
        self,
        location: str,
        trace_type: TraceType,
    ) -> GradientInfo:
        """Sense the gradient of a trace type at a location.

        Returns information about the direction of increasing intensity.

        Args:
            location: Location to check.
            trace_type: Type of trace to measure gradient for.

        Returns:
            GradientInfo with local intensity and gradient direction.
        """
        # Get local intensity
        local_traces = await self.sense(location, [trace_type])
        local_intensity = sum(t.intensity for t in local_traces)

        # Find strongest neighbor
        strongest_neighbor = None
        strongest_intensity = 0.0

        for neighbor in self.get_neighbors(location):
            neighbor_traces = await self.sense(neighbor, [trace_type])
            neighbor_intensity = sum(t.intensity for t in neighbor_traces)

            if neighbor_intensity > strongest_intensity:
                strongest_intensity = neighbor_intensity
                strongest_neighbor = neighbor

        gradient_direction = None
        if strongest_intensity > local_intensity:
            gradient_direction = strongest_neighbor

        return GradientInfo(
            trace_type=trace_type,
            location=location,
            local_intensity=local_intensity,
            strongest_neighbor=strongest_neighbor,
            strongest_neighbor_intensity=strongest_intensity,
            gradient_direction=gradient_direction,
        )

    async def follow_strongest(
        self,
        location: str,
        trace_type: TraceType,
        possible_next: list[str] | None = None,
    ) -> str | None:
        """Determine the best next location by following strongest trace.

        Uses probabilistic selection weighted by intensity (like ACO).

        Args:
            location: Current location.
            trace_type: Type of trace to follow.
            possible_next: Allowed next locations (defaults to neighbors).

        Returns:
            Best next location, or None if no traces found.
        """
        import random

        candidates = possible_next or self.get_neighbors(location)
        if not candidates:
            return None

        # Calculate intensities for each candidate
        intensities: list[tuple[str, float]] = []
        total = 0.0

        for candidate in candidates:
            traces = await self.sense(candidate, [trace_type])
            intensity = sum(t.intensity for t in traces)
            if intensity > 0:
                intensities.append((candidate, intensity))
                total += intensity

        if not intensities:
            # No traces found, random choice
            return random.choice(candidates)

        # Probabilistic selection weighted by intensity
        r = random.random() * total
        cumulative = 0.0
        for candidate, intensity in intensities:
            cumulative += intensity
            if r <= cumulative:
                # Record trail follow metrics
                metrics = get_metrics()
                metrics.trail_followed(trace_type.value)
                return candidate

        # Record trail follow for last candidate
        metrics = get_metrics()
        metrics.trail_followed(trace_type.value)
        return intensities[-1][0]

    async def decay_cycle(self) -> int:
        """Apply decay to all traces and remove expired ones.

        Returns:
            Number of traces removed.
        """
        removed = 0

        for location in list(self._traces.keys()):
            for trace_type in list(self._traces[location].keys()):
                traces = self._traces[location][trace_type]

                # Apply decay and filter expired
                remaining = []
                for trace in traces:
                    trace.decay()
                    if not trace.is_expired:
                        remaining.append(trace)
                    else:
                        removed += 1

                if remaining:
                    self._traces[location][trace_type] = remaining
                else:
                    del self._traces[location][trace_type]

            # Clean up empty locations
            if not self._traces[location]:
                del self._traces[location]

        if removed > 0:
            logger.debug(f"Decay cycle removed {removed} expired traces")

        return removed

    async def diffuse(self) -> int:
        """Diffuse traces to neighboring locations.

        A fraction of each trace's intensity spreads to neighbors.

        Returns:
            Number of diffusion operations performed.
        """
        diffusions = 0

        for location, type_traces in list(self._traces.items()):
            neighbors = self.get_neighbors(location)
            if not neighbors:
                continue

            for trace_type, traces in type_traces.items():
                for trace in traces:
                    if trace.intensity < 0.1:
                        continue

                    diffuse_amount = trace.intensity * self._diffusion_rate / len(neighbors)

                    for neighbor in neighbors:
                        if neighbor not in self._traces:
                            self._traces[neighbor] = {}
                        if trace_type not in self._traces[neighbor]:
                            self._traces[neighbor][trace_type] = []

                        # Find or create diffused trace at neighbor
                        existing = None
                        for nt in self._traces[neighbor][trace_type]:
                            if nt.depositor_id == f"diffused_from_{location}":
                                existing = nt
                                break

                        if existing:
                            existing.reinforce(diffuse_amount)
                        else:
                            self._trace_counter += 1
                            diffused = PheromoneTrace(
                                trace_id=f"diffused_{self._trace_counter}",
                                trace_type=trace_type,
                                location=neighbor,
                                intensity=diffuse_amount,
                                payload={"diffused_from": location},
                                depositor_id=f"diffused_from_{location}",
                                decay_rate=trace.decay_rate * 1.5,  # Faster decay
                            )
                            self._traces[neighbor][trace_type].append(diffused)

                        diffusions += 1

        if diffusions > 0:
            logger.debug(f"Diffused to {diffusions} neighbor locations")

        return diffusions

    def get_total_intensity(
        self,
        trace_type: TraceType | None = None,
    ) -> float:
        """Get total intensity across all traces.

        Args:
            trace_type: Optional filter for specific trace type.

        Returns:
            Sum of all trace intensities.
        """
        total = 0.0

        for location_traces in self._traces.values():
            for ttype, traces in location_traces.items():
                if trace_type and ttype != trace_type:
                    continue
                total += sum(t.intensity for t in traces if not t.is_expired)

        return total

    def get_trace_count(
        self,
        trace_type: TraceType | None = None,
    ) -> int:
        """Get count of active traces.

        Args:
            trace_type: Optional filter for specific trace type.

        Returns:
            Number of non-expired traces.
        """
        count = 0

        for location_traces in self._traces.values():
            for ttype, traces in location_traces.items():
                if trace_type and ttype != trace_type:
                    continue
                count += sum(1 for t in traces if not t.is_expired)

        return count

    def get_locations_with_traces(
        self,
        trace_type: TraceType | None = None,
    ) -> list[str]:
        """Get all locations that have active traces.

        Args:
            trace_type: Optional filter for specific trace type.

        Returns:
            List of location keys.
        """
        locations = []

        for location, location_traces in self._traces.items():
            has_traces = False
            for ttype, traces in location_traces.items():
                if trace_type and ttype != trace_type:
                    continue
                if any(not t.is_expired for t in traces):
                    has_traces = True
                    break

            if has_traces:
                locations.append(location)

        return locations

    def clear(self) -> None:
        """Clear all traces from the field."""
        self._traces.clear()
        logger.info(f"Cleared pheromone field: {self._field_id}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize field state to dictionary."""
        traces_dict: dict[str, dict[str, list[dict[str, Any]]]] = {}

        for location, type_traces in self._traces.items():
            traces_dict[location] = {}
            for trace_type, traces in type_traces.items():
                traces_dict[location][trace_type.value] = [
                    t.to_dict() for t in traces if not t.is_expired
                ]

        return {
            "field_id": self._field_id,
            "decay_interval": self._decay_interval,
            "diffusion_rate": self._diffusion_rate,
            "neighbor_map": self._neighbor_map,
            "traces": traces_dict,
            "trace_counter": self._trace_counter,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PheromoneField:
        """Create field from dictionary."""
        field = cls(
            field_id=data.get("field_id", "default"),
            decay_interval=data.get("decay_interval", cls.DEFAULT_DECAY_INTERVAL),
            diffusion_rate=data.get("diffusion_rate", cls.DEFAULT_DIFFUSION_RATE),
            neighbor_map=data.get("neighbor_map", {}),
        )

        field._trace_counter = data.get("trace_counter", 0)

        for location, type_traces in data.get("traces", {}).items():
            field._traces[location] = {}
            for trace_type_str, traces in type_traces.items():
                trace_type = TraceType(trace_type_str)
                field._traces[location][trace_type] = [PheromoneTrace.from_dict(t) for t in traces]

        return field
