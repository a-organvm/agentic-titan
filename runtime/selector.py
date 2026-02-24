"""
Runtime Selector - Intelligent runtime selection.

Automatically chooses the best runtime based on:
- Task requirements (GPU, memory, CPU)
- Scale requirements (instances, auto-scaling)
- Cost optimization preferences
- Fault tolerance needs
- System load awareness (CPU, memory utilization)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from runtime.base import (
    AgentProcess,
    Runtime,
    RuntimeConstraints,
    RuntimeType,
)
from runtime.docker import DockerRuntime
from runtime.local import LocalRuntime

logger = logging.getLogger("titan.runtime.selector")


class SelectionStrategy(StrEnum):
    """Strategy for runtime selection."""

    AUTO = "auto"  # Intelligent selection based on constraints
    PREFER_LOCAL = "local"  # Prefer local, fallback to others
    PREFER_DOCKER = "docker"  # Prefer Docker containers
    COST_OPTIMIZED = "cost"  # Minimize resource usage
    PERFORMANCE = "perf"  # Maximum performance
    LOAD_AWARE = "load_aware"  # Select based on system load


class LoadLevel(StrEnum):
    """System load level classification."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SystemLoad:
    """Current system resource utilization."""

    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_percent: float = 0.0
    load_average_1m: float = 0.0
    load_average_5m: float = 0.0
    load_average_15m: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def level(self) -> LoadLevel:
        """Classify current load level."""
        if self.cpu_percent > 90 or self.memory_percent > 90:
            return LoadLevel.CRITICAL
        if self.cpu_percent > 80 or self.memory_percent > 85:
            return LoadLevel.HIGH
        if self.cpu_percent > 50 or self.memory_percent > 60:
            return LoadLevel.MODERATE
        return LoadLevel.LOW

    @property
    def should_offload(self) -> bool:
        """Whether tasks should be offloaded to remote workers."""
        return self.level in (LoadLevel.HIGH, LoadLevel.CRITICAL)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cpu_percent": round(self.cpu_percent, 1),
            "memory_percent": round(self.memory_percent, 1),
            "disk_percent": round(self.disk_percent, 1),
            "load_average_1m": round(self.load_average_1m, 2),
            "load_average_5m": round(self.load_average_5m, 2),
            "load_average_15m": round(self.load_average_15m, 2),
            "level": self.level.value,
            "should_offload": self.should_offload,
            "timestamp": self.timestamp.isoformat(),
        }


def get_system_load() -> SystemLoad:
    """
    Get current system resource utilization.

    Uses psutil if available, falls back to os.getloadavg().
    """
    try:
        import psutil

        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        load_avg = os.getloadavg()

        return SystemLoad(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_percent=disk.percent,
            load_average_1m=load_avg[0],
            load_average_5m=load_avg[1],
            load_average_15m=load_avg[2],
        )

    except ImportError:
        # psutil not installed
        load_avg = os.getloadavg()
        # Estimate CPU from load average (rough approximation)
        cpu_count = os.cpu_count() or 1
        cpu_estimate = min(100, (load_avg[0] / cpu_count) * 100)

        return SystemLoad(
            cpu_percent=cpu_estimate,
            memory_percent=50.0,  # Unknown
            disk_percent=50.0,  # Unknown
            load_average_1m=load_avg[0],
            load_average_5m=load_avg[1],
            load_average_15m=load_avg[2],
        )


@dataclass
class RuntimeScore:
    """Score for a runtime option."""

    runtime_type: RuntimeType
    score: float  # 0-100
    reasons: list[str]
    available: bool = True


class RuntimeSelector:
    """
    Intelligent runtime selector.

    Analyzes constraints and task requirements to select
    the optimal runtime for agent execution.

    Features:
    - Load-aware selection based on CPU/memory utilization
    - Automatic offloading to Docker/K3s under high load
    - Strategy-based selection (auto, prefer_local, load_aware, etc.)
    """

    def __init__(
        self,
        strategy: SelectionStrategy = SelectionStrategy.AUTO,
        runtimes: dict[RuntimeType, Runtime] | None = None,
        load_threshold_cpu: float = 80.0,
        load_threshold_memory: float = 85.0,
    ) -> None:
        self.strategy = strategy
        self._runtimes: dict[RuntimeType, Runtime] = runtimes or {}
        self._initialized = False

        # Load thresholds for offloading
        self._load_threshold_cpu = load_threshold_cpu
        self._load_threshold_memory = load_threshold_memory

        # Cache system load (refresh every 5 seconds)
        self._cached_load: SystemLoad | None = None
        self._load_cache_time: datetime | None = None

    async def initialize(self) -> None:
        """Initialize available runtimes."""
        logger.info("Initializing runtime selector...")

        # Create default runtimes if not provided
        if not self._runtimes:
            self._runtimes = {
                RuntimeType.LOCAL: LocalRuntime(),
                RuntimeType.DOCKER: DockerRuntime(),
            }

        # Initialize each runtime
        for runtime_type, runtime in self._runtimes.items():
            try:
                await runtime.initialize()
                logger.info(f"Runtime {runtime_type.value} initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize {runtime_type.value}: {e}")

        self._initialized = True
        logger.info("Runtime selector initialized")

    async def shutdown(self) -> None:
        """Shutdown all runtimes."""
        for runtime in self._runtimes.values():
            try:
                await runtime.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down runtime: {e}")

        self._initialized = False

    def get_system_load(self, refresh: bool = False) -> SystemLoad:
        """
        Get current system load with caching.

        Args:
            refresh: Force refresh of cached value

        Returns:
            Current SystemLoad
        """
        now = datetime.now()

        # Check cache (5 second TTL)
        if (
            not refresh
            and self._cached_load
            and self._load_cache_time
            and (now - self._load_cache_time).total_seconds() < 5.0
        ):
            return self._cached_load

        # Refresh cache
        self._cached_load = get_system_load()
        self._load_cache_time = now
        return self._cached_load

    def select(
        self,
        constraints: RuntimeConstraints | None = None,
        agent_spec: dict[str, Any] | None = None,
    ) -> RuntimeType:
        """
        Select the best runtime for given constraints.

        Args:
            constraints: Runtime constraints
            agent_spec: Agent specification (optional)

        Returns:
            Selected runtime type
        """
        constraints = constraints or RuntimeConstraints()

        # Score each runtime
        scores = self._score_runtimes(constraints, agent_spec)

        # Sort by score (descending)
        scores.sort(key=lambda s: s.score, reverse=True)

        # Log selection rationale
        logger.info("Runtime selection scores:")
        for score in scores:
            status = "✓" if score.available else "✗"
            logger.info(
                f"  {status} {score.runtime_type.value}: {score.score:.1f} - "
                f"{', '.join(score.reasons)}"
            )

        # Return highest scoring available runtime
        for score in scores:
            if score.available:
                return score.runtime_type

        # Fallback to local
        logger.warning("No suitable runtime found, falling back to local")
        return RuntimeType.LOCAL

    def select_with_load_awareness(
        self,
        constraints: RuntimeConstraints | None = None,
        agent_spec: dict[str, Any] | None = None,
    ) -> tuple[RuntimeType, SystemLoad]:
        """
        Select runtime with load awareness.

        Adjusts constraints based on current system load to
        automatically offload to Docker/K3s when local is under pressure.

        Args:
            constraints: Runtime constraints
            agent_spec: Agent specification

        Returns:
            Tuple of (selected runtime, current system load)
        """
        constraints = constraints or RuntimeConstraints()
        system_load = self.get_system_load()

        # Adjust constraints based on load
        if system_load.cpu_percent > self._load_threshold_cpu:
            logger.info(
                f"High CPU load ({system_load.cpu_percent:.1f}%), preferring non-local runtime"
            )
            constraints.prefer_local = False

        if system_load.memory_percent > self._load_threshold_memory:
            logger.info(
                f"High memory pressure ({system_load.memory_percent:.1f}%), requiring isolation"
            )
            constraints.needs_isolation = True

        # Use load-aware strategy
        original_strategy = self.strategy
        if system_load.should_offload:
            self.strategy = SelectionStrategy.PREFER_DOCKER

        try:
            runtime_type = self.select(constraints, agent_spec)
        finally:
            self.strategy = original_strategy

        return runtime_type, system_load

    def _score_runtimes(
        self,
        constraints: RuntimeConstraints,
        agent_spec: dict[str, Any] | None,
    ) -> list[RuntimeScore]:
        """Score all available runtimes."""
        scores = []

        runtime_types = [RuntimeType.LOCAL, RuntimeType.DOCKER]

        # Include Firecracker if available
        if RuntimeType.FIRECRACKER in self._runtimes:
            runtime_types.append(RuntimeType.FIRECRACKER)

        for runtime_type in runtime_types:
            score = self._score_runtime(runtime_type, constraints, agent_spec)
            scores.append(score)

        return scores

    def _score_runtime(
        self,
        runtime_type: RuntimeType,
        constraints: RuntimeConstraints,
        agent_spec: dict[str, Any] | None,
    ) -> RuntimeScore:
        """Score a specific runtime."""
        score = 50.0  # Base score
        reasons: list[str] = []

        # Check availability
        runtime = self._runtimes.get(runtime_type)
        available = runtime is not None and runtime._initialized

        if runtime_type == RuntimeType.LOCAL:
            score, reasons = self._score_local(constraints, agent_spec)
        elif runtime_type == RuntimeType.DOCKER:
            score, reasons = self._score_docker(constraints, agent_spec)
        elif runtime_type == RuntimeType.FIRECRACKER:
            score, reasons = self._score_firecracker(constraints, agent_spec)

        # Apply strategy modifiers
        score = self._apply_strategy(runtime_type, score, constraints)

        return RuntimeScore(
            runtime_type=runtime_type,
            score=score,
            reasons=reasons,
            available=available,
        )

    def _score_local(
        self,
        constraints: RuntimeConstraints,
        agent_spec: dict[str, Any] | None,
    ) -> tuple[float, list[str]]:
        """Score local runtime."""
        score = 60.0  # Good baseline
        reasons = []

        # Advantages of local
        if constraints.prefer_local:
            score += 20
            reasons.append("Local preferred")

        if not constraints.needs_isolation:
            score += 10
            reasons.append("No isolation needed")

        # Local is fastest for single instances
        if constraints.expected_instances <= 2:
            score += 15
            reasons.append("Low instance count")

        # GPU access (local has direct GPU)
        if constraints.requires_gpu:
            score += 10
            reasons.append("Direct GPU access")

        # Disadvantages
        if constraints.expected_instances > 5:
            score -= 20
            reasons.append("High scale reduces score")

        if constraints.needs_isolation:
            score -= 15
            reasons.append("Needs isolation")

        if constraints.auto_scale:
            score -= 10
            reasons.append("No auto-scaling")

        return score, reasons

    def _score_docker(
        self,
        constraints: RuntimeConstraints,
        agent_spec: dict[str, Any] | None,
    ) -> tuple[float, list[str]]:
        """Score Docker runtime."""
        score = 50.0  # Good baseline
        reasons = []

        # Advantages of Docker
        if constraints.needs_isolation:
            score += 25
            reasons.append("Isolation provided")

        if constraints.expected_instances > 3:
            score += 20
            reasons.append("Good for scaling")

        if constraints.needs_persistence:
            score += 10
            reasons.append("Volume support")

        # Resource limits enforceable
        if constraints.min_memory_mb > 512:
            score += 5
            reasons.append("Resource limits enforced")

        # Disadvantages
        if constraints.prefer_local:
            score -= 15
            reasons.append("Local preferred")

        if constraints.requires_gpu:
            # GPU in Docker is more complex
            score -= 5
            reasons.append("GPU overhead")

        # Check if agent spec has container config
        if agent_spec:
            runtimes = agent_spec.get("spec", {}).get("runtimes", {})
            if "container" in runtimes:
                score += 15
                reasons.append("Container config present")

        return score, reasons

    def _score_firecracker(
        self,
        constraints: RuntimeConstraints,
        agent_spec: dict[str, Any] | None,
    ) -> tuple[float, list[str]]:
        """Score Firecracker runtime."""
        score = 45.0  # Slightly lower baseline (requires more setup)
        reasons = []

        # Advantages of Firecracker
        if constraints.needs_isolation:
            score += 30
            reasons.append("Strong HW isolation")

        # Firecracker has very fast boot times
        if constraints.max_execution_time < 60:
            score += 15
            reasons.append("Fast boot for short tasks")

        # Good for security-sensitive workloads
        if agent_spec:
            task_type = agent_spec.get("spec", {}).get("task_type", "")
            if "security" in task_type.lower() or "sandbox" in task_type.lower():
                score += 20
                reasons.append("Security-sensitive task")

        # Resource efficiency
        if constraints.min_memory_mb <= 256:
            score += 10
            reasons.append("Low memory fits microVM")

        # Disadvantages
        if constraints.requires_gpu:
            score -= 40
            reasons.append("No GPU support")

        if constraints.needs_persistence:
            score -= 15
            reasons.append("Limited persistence")

        if constraints.auto_scale:
            score -= 10
            reasons.append("Manual scaling only")

        # Platform check (Linux only)
        import platform

        if platform.system() != "Linux":
            score = 0
            reasons = ["Linux only"]

        return score, reasons

    def _apply_strategy(
        self,
        runtime_type: RuntimeType,
        score: float,
        constraints: RuntimeConstraints,
    ) -> float:
        """Apply strategy modifiers to score."""
        if self.strategy == SelectionStrategy.PREFER_LOCAL:
            if runtime_type == RuntimeType.LOCAL:
                score += 30
        elif self.strategy == SelectionStrategy.PREFER_DOCKER:
            if runtime_type == RuntimeType.DOCKER:
                score += 30
        elif self.strategy == SelectionStrategy.COST_OPTIMIZED:
            # Local is cheapest
            if runtime_type == RuntimeType.LOCAL:
                score += 20
        elif self.strategy == SelectionStrategy.PERFORMANCE:
            # Local is fastest for simple tasks
            if runtime_type == RuntimeType.LOCAL and not constraints.needs_isolation:
                score += 15
        elif self.strategy == SelectionStrategy.LOAD_AWARE:
            # Apply load-based modifiers
            system_load = self.get_system_load()
            if system_load.should_offload:
                # Prefer Docker under high load
                if runtime_type == RuntimeType.DOCKER:
                    score += 25
                elif runtime_type == RuntimeType.LOCAL:
                    score -= 20
            else:
                # Prefer local under low load
                if runtime_type == RuntimeType.LOCAL:
                    score += 10

        return score

    def get_runtime(self, runtime_type: RuntimeType) -> Runtime | None:
        """Get a specific runtime instance."""
        return self._runtimes.get(runtime_type)

    async def spawn(
        self,
        agent_id: str,
        agent_spec: dict[str, Any],
        prompt: str | None = None,
        constraints: RuntimeConstraints | None = None,
        runtime_type: RuntimeType | None = None,
    ) -> AgentProcess:
        """
        Spawn an agent with automatic runtime selection.

        Args:
            agent_id: Unique agent identifier
            agent_spec: Agent specification
            prompt: Optional initial prompt
            constraints: Runtime constraints (for selection)
            runtime_type: Override automatic selection

        Returns:
            AgentProcess from selected runtime
        """
        # Select runtime
        if runtime_type is None:
            runtime_type = self.select(constraints, agent_spec)

        runtime = self._runtimes.get(runtime_type)
        if not runtime:
            raise RuntimeError(f"Runtime {runtime_type.value} not available")

        logger.info(f"Spawning agent {agent_id} on {runtime_type.value}")
        return await runtime.spawn(agent_id, agent_spec, prompt)

    async def health_check(self) -> dict[str, Any]:
        """Check health of all runtimes."""
        runtimes_health: dict[str, Any] = {}
        results: dict[str, Any] = {
            "strategy": self.strategy.value,
            "initialized": self._initialized,
            "runtimes": runtimes_health,
        }

        for runtime_type, runtime in self._runtimes.items():
            try:
                health = await runtime.health_check()
                runtimes_health[runtime_type.value] = health
            except Exception as e:
                runtimes_health[runtime_type.value] = {
                    "error": str(e),
                    "available": False,
                }

        return results

    def suggest(
        self,
        constraints: RuntimeConstraints | None = None,
        agent_spec: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Get a runtime suggestion with explanation.

        Args:
            constraints: Runtime constraints
            agent_spec: Agent specification

        Returns:
            Dict with recommendation and reasoning
        """
        constraints = constraints or RuntimeConstraints()
        scores = self._score_runtimes(constraints, agent_spec)
        scores.sort(key=lambda s: s.score, reverse=True)

        best = scores[0]

        return {
            "recommended": best.runtime_type.value,
            "score": best.score,
            "reasons": best.reasons,
            "available": best.available,
            "alternatives": [
                {
                    "type": s.runtime_type.value,
                    "score": s.score,
                    "reasons": s.reasons,
                    "available": s.available,
                }
                for s in scores[1:]
            ],
            "constraints_summary": {
                "requires_gpu": constraints.requires_gpu,
                "needs_isolation": constraints.needs_isolation,
                "expected_instances": constraints.expected_instances,
                "auto_scale": constraints.auto_scale,
                "cost_sensitive": constraints.cost_sensitive,
            },
            "system_load": self.get_system_load().to_dict(),
        }


# =============================================================================
# Factory Functions
# =============================================================================

_default_selector: RuntimeSelector | None = None


def get_runtime_selector() -> RuntimeSelector:
    """Get the default runtime selector instance."""
    global _default_selector
    if _default_selector is None:
        _default_selector = RuntimeSelector(
            strategy=SelectionStrategy.LOAD_AWARE,
        )
    return _default_selector
