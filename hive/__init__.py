"""
Hive Mind - Shared Intelligence Layer

Provides collective memory and real-time coordination for the agent swarm:
- Long-term memory (ChromaDB/Vector store)
- Working memory (Redis)
- Event bus (NATS/Redis Streams)
- Distributed state
- Event-driven topology transitions
- LLM-powered task analysis
- Episodic learning from outcomes
"""

from hive.analyzer import (
    AnalysisResult,
    TaskAnalyzer,
    analyze_task,
)
from hive.assembly import (
    AssemblyEvent,
    AssemblyManager,
    AssemblyState,
    DeterritorializationType,
    StabilityMetrics,
    TerritorizationType,
)
from hive.conflict import (
    SEMANTIC_OPPOSITES,
    ConflictDetector,
    ConflictPair,
)
from hive.criticality import (
    CriticalityMetrics,
    CriticalityMonitor,
    CriticalityState,
    PhaseTransition,
)
from hive.emergence import (
    EmergenceDetector,
    EmergenceResult,
)
from hive.events import (
    Event,
    EventBus,
    EventType,
    get_event_bus,
)
from hive.fission_fusion import (
    Cluster,
    FissionFusionManager,
    FissionFusionMetrics,
    FissionFusionState,
)
from hive.information_center import (
    InformationCenter,
    InformationCenterManager,
    InformationCenterRole,
    LearnedPattern,
)
from hive.learning import (
    Episode,
    EpisodeOutcome,
    EpisodicLearner,
    TopologyPreference,
    get_episodic_learner,
)
from hive.machines import (
    MachineDynamics,
    MachineOperation,
    MachineState,
    MachineType,
    OperationType,
)
from hive.memory import HiveMind, MemoryConfig
from hive.neighborhood import (
    AgentProfile,
    InteractionRecord,
    InteractionType,
    LayeredNeighborConfig,
    NeighborLayer,
    NeighborScore,
    TopologicalNeighborhood,
)
from hive.stigmergy import (
    GradientInfo,
    PheromoneField,
    PheromoneTrace,
    TraceType,
)
from hive.topology import (
    AgentNode,
    BaseTopology,
    HierarchyTopology,
    MeshTopology,
    PipelineTopology,
    RingTopology,
    SensingRegion,
    StarTopology,
    SwarmTopology,
    TaskProfile,
    TopologyEngine,
    TopologyType,
)
from hive.topology_extended import (
    ArborealTopology,
    Connection,
    ConnectionType,
    DeterritorializedTopology,
    ExtendedTopologyType,
    RhizomaticTopology,
    TerritorializedTopology,
    Territory,
)

__all__ = [
    # Memory
    "HiveMind",
    "MemoryConfig",
    # Topology
    "TopologyEngine",
    "TopologyType",
    "TaskProfile",
    "SensingRegion",
    "AgentNode",
    "BaseTopology",
    "SwarmTopology",
    "HierarchyTopology",
    "PipelineTopology",
    "MeshTopology",
    "RingTopology",
    "StarTopology",
    # Events
    "EventBus",
    "Event",
    "EventType",
    "get_event_bus",
    # Analyzer
    "TaskAnalyzer",
    "AnalysisResult",
    "analyze_task",
    # Learning
    "EpisodicLearner",
    "Episode",
    "EpisodeOutcome",
    "TopologyPreference",
    "get_episodic_learner",
    # Stigmergy
    "PheromoneField",
    "PheromoneTrace",
    "TraceType",
    "GradientInfo",
    # Neighborhood
    "TopologicalNeighborhood",
    "InteractionRecord",
    "InteractionType",
    "AgentProfile",
    "NeighborScore",
    "NeighborLayer",
    "LayeredNeighborConfig",
    # Extended Topologies
    "ExtendedTopologyType",
    "RhizomaticTopology",
    "ArborealTopology",
    "TerritorializedTopology",
    "DeterritorializedTopology",
    "Territory",
    "Connection",
    "ConnectionType",
    # Assembly
    "AssemblyManager",
    "AssemblyState",
    "AssemblyEvent",
    "StabilityMetrics",
    "TerritorizationType",
    "DeterritorializationType",
    # Machines
    "MachineDynamics",
    "MachineType",
    "MachineState",
    "MachineOperation",
    "OperationType",
    # Emergence Detection
    "EmergenceDetector",
    "EmergenceResult",
    # Criticality (Phase 16)
    "CriticalityMonitor",
    "CriticalityState",
    "CriticalityMetrics",
    "PhaseTransition",
    # Conflict Detection (#64)
    "ConflictDetector",
    "ConflictPair",
    "SEMANTIC_OPPOSITES",
    # Fission-Fusion (Phase 16)
    "FissionFusionManager",
    "FissionFusionState",
    "FissionFusionMetrics",
    "Cluster",
    # Information Centers (Phase 16)
    "InformationCenterManager",
    "InformationCenter",
    "InformationCenterRole",
    "LearnedPattern",
]
