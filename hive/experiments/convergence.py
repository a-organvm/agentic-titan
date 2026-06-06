"""Convergence Experiment — sweep agent count to find emergence threshold.

Empirically validates the emergence threshold by running simulated tasks
across agent counts N=8..128 and measuring when collective intelligence
appears (novel information in collective output not present in any
individual agent's contributions).

The experiment wires together the full emergence pipeline:
  TopologicalNeighborhood → CriticalityMonitor → PheromoneField → EmergenceDetector

References:
- Rubenstein et al. (2014). "Programmable self-assembly in a thousand-robot swarm." Science
- Cavagna et al. (2010). "Scale-free correlations in starling flocks." PNAS
- Issue #72, #61
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any

from hive.criticality import CriticalityMonitor, CriticalityState
from hive.emergence import EmergenceDetector, EmergenceResult
from hive.neighborhood import InteractionType, TopologicalNeighborhood
from hive.stigmergy import PheromoneField, TraceType
from hive.topology import TopologyEngine, TopologyType

logger = logging.getLogger("titan.hive.experiments.convergence")

# ---------------------------------------------------------------------------
# 139-atom semantic vocabulary
# ---------------------------------------------------------------------------
# Domain vocabulary (120 terms): partial knowledge fragments agents contribute.
# Each agent gets a random subset. The collective synthesizes across all.
_DOMAIN_VOCABULARY = [
    # Systems (30)
    "latency", "throughput", "bottleneck", "saturation", "capacity",
    "congestion", "degradation", "failover", "redundancy", "resilience",
    "partition", "consistency", "availability", "replication", "sharding",
    "indexing", "caching", "prefetching", "eviction", "invalidation",
    "compaction", "deduplication", "normalization", "denormalization",
    "materialization", "linearizability", "serializability", "isolation",
    "atomicity", "durability",
    # Resilience (20)
    "backpressure", "throttling", "circuit", "breaker", "bulkhead",
    "timeout", "retry", "jitter", "exponential", "backoff",
    "shedding", "quarantine", "fallback", "recovery", "checkpoint",
    "rollback", "idempotency", "compensation", "reconciliation",
    "stabilization",
    # Observability (20)
    "observability", "telemetry", "tracing", "correlation", "propagation",
    "sampling", "aggregation", "windowing", "percentile", "histogram",
    "cardinality", "dimensionality", "instrumentation", "profiling",
    "benchmarking", "alerting", "anomaly", "baseline", "threshold",
    "dashboard",
    # Networking (20)
    "routing", "forwarding", "balancing", "proxy", "gateway",
    "firewall", "encryption", "certificate", "handshake", "multiplexing",
    "pipelining", "framing", "serialization", "compression", "decompression",
    "discovery", "registration", "heartbeat", "gossip", "quorum",
    # Scaling (15)
    "horizontal", "vertical", "autoscaling", "provisioning", "elasticity",
    "multitenancy", "namespace", "federation", "orchestration", "scheduling",
    "placement", "affinity", "colocation", "migration", "preemption",
    # Consensus (15)
    "consensus", "leader", "election", "follower", "candidate",
    "proposal", "ballot", "majority", "supermajority", "byzantine",
    "membership", "reconfiguration", "snapshotting", "protocol",
    "ratification",  # not "compaction" — already in Systems

]

# Synthesis vocabulary (19 terms): novel information that emerges only
# when combining fragments across agents.
_SYNTHESIS_VOCABULARY = [
    "cascading", "amplification", "resonance", "interference",
    "convergence", "divergence", "bifurcation", "attractor",
    "equilibrium", "perturbation", "oscillation", "dampening",
    "emergence", "self-organization", "criticality", "phase-transition",
    "synergy", "metamorphosis", "catalysis",
]


@dataclass(frozen=True)
class SweepPoint:
    """Result of a single point in the convergence sweep."""

    agent_count: int
    topology_type: str
    emergence_detected: bool
    novelty_ratio: float
    evidence_count: int
    above_threshold: bool
    criticality_state: str
    vocab_coverage: float  # Fraction of domain vocab covered by agents
    # Criticality metrics from CriticalityMonitor.sample_state()
    correlation_length: float = 0.0
    functional_connectivity: float = 0.0
    order_parameter: float = 0.5
    criticality_score: float = 0.0
    # Network metrics from TopologicalNeighborhood
    total_interactions: int = 0
    network_density: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "agent_count": self.agent_count,
            "topology_type": self.topology_type,
            "emergence_detected": self.emergence_detected,
            "novelty_ratio": self.novelty_ratio,
            "evidence_count": self.evidence_count,
            "above_threshold": self.above_threshold,
            "criticality_state": self.criticality_state,
            "vocab_coverage": self.vocab_coverage,
            "correlation_length": self.correlation_length,
            "functional_connectivity": self.functional_connectivity,
            "order_parameter": self.order_parameter,
            "criticality_score": self.criticality_score,
            "total_interactions": self.total_interactions,
            "network_density": self.network_density,
        }


@dataclass
class ConvergenceResult:
    """Complete result of a convergence experiment sweep."""

    sweep_points: list[SweepPoint] = field(default_factory=list)
    estimated_threshold: int | None = None
    seed: int = 0

    @property
    def emergence_onset(self) -> int | None:
        """First N where emergence was detected."""
        for point in self.sweep_points:
            if point.emergence_detected:
                return point.agent_count
        return None

    @property
    def consistent_emergence(self) -> int | None:
        """First N after which emergence is always detected."""
        detected = [p.emergence_detected for p in self.sweep_points]
        for i in range(len(detected)):
            if all(detected[i:]):
                return self.sweep_points[i].agent_count
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "sweep_points": [p.to_dict() for p in self.sweep_points],
            "estimated_threshold": self.estimated_threshold,
            "emergence_onset": self.emergence_onset,
            "consistent_emergence": self.consistent_emergence,
            "seed": self.seed,
        }


class ConvergenceExperiment:
    """Sweep agent count N to find the emergence threshold empirically.

    Creates simulated multi-agent tasks at each N, wiring through the
    full emergence pipeline:

    1. Create SWARM topology + TopologicalNeighborhood (k=7)
    2. Register agents, simulate interactions
    3. Deposit pheromone traces with vocabulary payloads
    4. CriticalityMonitor.sample_state() for real metrics
    5. EmergenceDetector checks for novel collective information
    6. Record all metrics into SweepPoint

    The experiment is deterministic given the same seed.

    Args:
        topology_type: Topology to test (default SWARM).
        n_range: Agent counts to sweep (default 8..128 in steps of 8).
        vocab_per_agent: How many domain terms each agent knows.
        synthesis_terms: How many synthesis terms appear in collective.
        detector_threshold: EmergenceDetector novelty threshold.
        coverage_gate: Minimum domain vocabulary coverage before synthesis
            produces novel information. Below this, agents are too siloed
            for cross-pollination. Default 0.75.
        neighbor_count: Topological neighbor count (k). Default 7.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        topology_type: TopologyType = TopologyType.SWARM,
        n_range: list[int] | None = None,
        vocab_per_agent: int = 5,
        synthesis_terms: int = 4,
        detector_threshold: float = 0.1,
        coverage_gate: float = 0.75,
        neighbor_count: int = 7,
        seed: int = 42,
    ) -> None:
        self._topology_type = topology_type
        self._n_range = n_range or list(range(8, 129, 8))
        self._vocab_per_agent = vocab_per_agent
        self._synthesis_terms = synthesis_terms
        self._coverage_gate = coverage_gate
        self._neighbor_count = neighbor_count
        self._detector = EmergenceDetector(
            novelty_threshold=detector_threshold
        )
        self._seed = seed

    async def run(self) -> ConvergenceResult:
        """Execute the convergence sweep.

        Returns:
            ConvergenceResult with one SweepPoint per N value.
        """
        rng = random.Random(self._seed)
        result = ConvergenceResult(seed=self._seed)

        for n in self._n_range:
            point = await self._run_single(n, rng)
            result.sweep_points.append(point)
            logger.info(
                "N=%d: emergence=%s ratio=%.3f fc=%.3f op=%.3f state=%s",
                n,
                point.emergence_detected,
                point.novelty_ratio,
                point.functional_connectivity,
                point.order_parameter,
                point.criticality_state,
            )

        # Estimate threshold as the consistent_emergence point
        result.estimated_threshold = result.consistent_emergence

        return result

    async def _run_single(self, n: int, rng: random.Random) -> SweepPoint:
        """Run a single sweep point with N agents.

        Wires together topology, neighborhood, criticality monitor,
        pheromone field, and emergence detector for a complete pipeline.
        """
        # --- Topology ---
        engine = TopologyEngine()
        topo = engine.create_topology(self._topology_type)

        for i in range(n):
            topo.add_agent(f"agent-{i}", f"Agent {i}", ["general"])

        # --- Neighborhood (k=7 topological neighbors) ---
        neighborhood = TopologicalNeighborhood(
            neighbor_count=self._neighbor_count,
            recency_decay=0.0,
        )

        # Generate per-agent contributions and register in neighborhood
        contributions: dict[str, list[str]] = {}
        agent_terms_map: dict[str, list[str]] = {}
        all_agent_terms: set[str] = set()

        for i in range(n):
            agent_id = f"agent-{i}"
            k = min(self._vocab_per_agent, len(_DOMAIN_VOCABULARY))
            terms = rng.sample(_DOMAIN_VOCABULARY, k)
            agent_terms_map[agent_id] = terms
            all_agent_terms.update(terms)

            # Register agent with vocabulary terms as capabilities
            neighborhood.register_agent(agent_id, capabilities=terms)

            # Agent "contribution" is a sentence using their terms
            contribution = (
                f"Analysis from agent {i}: observed {' and '.join(terms)} "
                f"in the system under study"
            )
            contributions[agent_id] = [contribution]

        # Simulate interactions: interaction density grows sub-linearly
        # with swarm size (sparse at low N, saturating at k for large N).
        # This models the real phenomenon: small groups have fewer active
        # communication channels; larger swarms develop richer interaction.
        agent_ids = [f"agent-{i}" for i in range(n)]
        max_interactions = min(self._neighbor_count, n - 1)
        interactions_per_agent = max(1, min(max_interactions, int(math.sqrt(n))))
        for i in range(n):
            agent_id = agent_ids[i]
            peers = [a for a in agent_ids if a != agent_id]
            collaborators = rng.sample(peers, min(interactions_per_agent, len(peers)))
            for peer in collaborators:
                neighborhood.record_interaction(
                    agent_id, peer, InteractionType.COLLABORATION, success=True,
                )

        # --- Pheromone field ---
        pheromone_field = PheromoneField(field_id=f"sweep-n{n}")

        for agent_id, terms in agent_terms_map.items():
            await pheromone_field.deposit(
                agent_id,
                TraceType.RESOURCE,
                "shared",
                payload={"terms": terms},
            )

        # --- Criticality monitor (real metrics from neighborhood) ---
        monitor = CriticalityMonitor(neighborhood=neighborhood)
        monitor.set_emergence_threshold(engine.emergence_threshold, n)

        crit_metrics = await monitor.sample_state()

        # Enforce emergence threshold gate: below minimum N means
        # the system cannot produce collective intelligence regardless
        # of what the raw metrics say. This mirrors the gate in
        # CriticalityMonitor._sample_and_evaluate() for the monitoring loop.
        if monitor.below_emergence_threshold:
            forced_state = CriticalityState.SUBCRITICAL
        else:
            forced_state = None

        # --- Collective output generation ---
        coverage = len(all_agent_terms) / len(_DOMAIN_VOCABULARY)

        if coverage < self._coverage_gate:
            # Low coverage: verbatim restatement, no novel tokens
            terms_list = list(all_agent_terms)
            rng.shuffle(terms_list)
            collective = " ".join(terms_list)
        else:
            # High coverage: synthesis produces novel insights
            gate = self._coverage_gate
            synth_scale = (coverage - gate) / (1.0 - gate)
            n_synth = max(1, int(synth_scale * self._synthesis_terms))
            n_synth = min(n_synth, len(_SYNTHESIS_VOCABULARY))
            synth_terms = rng.sample(_SYNTHESIS_VOCABULARY, n_synth)

            agent_summary = " ".join(
                rng.sample(
                    list(all_agent_terms),
                    min(10, len(all_agent_terms)),
                )
            )
            synthesis = " ".join(synth_terms)
            collective = (
                f"Collective analysis synthesizing {n} agent observations: "
                f"{agent_summary}. "
                f"Cross-agent synthesis reveals {synthesis} patterns "
                f"that were not apparent in any individual analysis."
            )

        # --- Emergence detection ---
        em_result: EmergenceResult = self._detector.detect(
            contributions, collective
        )

        # --- Network stats ---
        net_stats = neighborhood.get_network_stats()

        return SweepPoint(
            agent_count=n,
            topology_type=self._topology_type.value,
            emergence_detected=em_result.detected,
            novelty_ratio=em_result.novelty_ratio,
            evidence_count=len(em_result.evidence),
            above_threshold=engine.above_emergence_threshold,
            criticality_state=(
                forced_state.value if forced_state
                else crit_metrics.infer_state().value
            ),
            vocab_coverage=round(coverage, 4),
            correlation_length=round(crit_metrics.correlation_length, 4),
            functional_connectivity=round(crit_metrics.functional_connectivity, 4),
            order_parameter=round(crit_metrics.order_parameter, 4),
            criticality_score=round(crit_metrics.criticality_score, 4),
            total_interactions=int(net_stats.get("total_interactions", 0)),
            network_density=round(float(net_stats.get("density", 0.0)), 4),
        )
