"""
Stress Test Scenarios

Pre-built scenarios for stress testing different aspects of the system:
- SwarmBrainstorm: All agents communicate freely
- PipelineWorkflow: Sequential processing stages
- HierarchyDelegation: Tree-based task delegation
- Chaos: Random failures and topology switches
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from agents.framework.base_agent import BaseAgent


class ScenarioPhase(Enum):
    """Phases of a stress test scenario."""

    SETUP = "setup"
    WARMUP = "warmup"
    STRESS = "stress"
    COOLDOWN = "cooldown"
    TEARDOWN = "teardown"


@dataclass
class Scenario(ABC):
    """
    Base class for stress test scenarios.

    Scenarios define:
    - How agents are spawned
    - What tasks they perform
    - How they interact
    - What failures to inject
    """

    name: str
    description: str
    target_agents: int = 50
    duration_seconds: int = 60
    ramp_up_seconds: int = 10
    warmup_seconds: int = 5

    # Failure injection
    failure_rate: float = 0.0  # 0-1, probability of injected failure
    timeout_rate: float = 0.0  # 0-1, probability of timeout injection
    topology_switch_interval: int = 0  # 0 = no switching, >0 = switch every N seconds

    # Communication patterns
    broadcast_probability: float = 0.1  # Probability an agent broadcasts
    direct_message_probability: float = 0.2  # Probability of direct messaging

    # Current state
    phase: ScenarioPhase = field(default=ScenarioPhase.SETUP)
    active_agents: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @abstractmethod
    async def spawn_agent(self, index: int) -> BaseAgent:
        """
        Create an agent for this scenario.

        Args:
            index: Agent index (0 to target_agents-1)

        Returns:
            Configured agent instance
        """
        pass

    @abstractmethod
    async def get_task(self, agent: BaseAgent) -> str:
        """
        Get the task/prompt for an agent.

        Args:
            agent: The agent to get task for

        Returns:
            Task description/prompt
        """
        pass

    def should_inject_failure(self) -> bool:
        """Check if we should inject a failure."""
        return random.random() < self.failure_rate

    def should_inject_timeout(self) -> bool:
        """Check if we should inject a timeout."""
        return random.random() < self.timeout_rate

    def should_broadcast(self) -> bool:
        """Check if agent should broadcast."""
        return random.random() < self.broadcast_probability

    def should_direct_message(self) -> bool:
        """Check if agent should send direct message."""
        return random.random() < self.direct_message_probability

    def get_random_target_agent(self, exclude: str | None = None) -> str | None:
        """Get a random active agent ID for messaging."""
        candidates = [a for a in self.active_agents if a != exclude]
        return random.choice(candidates) if candidates else None


@dataclass
class SwarmBrainstormScenario(Scenario):
    """
    Swarm topology stress test.

    All agents operate in a flat structure, broadcasting ideas
    and building on each other's work. Tests:
    - High message throughput
    - Broadcast efficiency
    - Memory pressure from shared state
    """

    name: str = "swarm_brainstorm"
    description: str = "All agents brainstorm and communicate freely"
    broadcast_probability: float = 0.3
    direct_message_probability: float = 0.4

    topics: list[str] = field(
        default_factory=lambda: [
            "AI safety best practices",
            "Sustainable energy solutions",
            "Space exploration technologies",
            "Healthcare automation",
            "Education innovation",
        ]
    )

    async def spawn_agent(self, index: int) -> BaseAgent:
        """Spawn a researcher-type agent for brainstorming."""
        from agents.archetypes.researcher import ResearcherAgent

        agent = ResearcherAgent(
            name=f"brainstormer-{index}",
            capabilities=["brainstorm", "analyze", "synthesize"],
            max_turns=5,
            timeout_ms=30_000,
        )
        return agent

    async def get_task(self, agent: BaseAgent) -> str:
        """Get a brainstorming task."""
        topic = random.choice(self.topics)
        return f"Brainstorm innovative ideas for: {topic}. Build on ideas from other agents."


@dataclass
class PipelineWorkflowScenario(Scenario):
    """
    Pipeline topology stress test.

    Agents work in sequential stages: research -> code -> review.
    Tests:
    - Stage transitions
    - Handoff latency
    - Stage bottlenecks
    """

    name: str = "pipeline_workflow"
    description: str = "Sequential stages: research -> code -> review"
    broadcast_probability: float = 0.0
    direct_message_probability: float = 0.8  # High for handoffs

    stages: list[str] = field(default_factory=lambda: ["research", "code", "review"])

    async def spawn_agent(self, index: int) -> BaseAgent:
        """Spawn agent for a specific pipeline stage."""
        stage_index = index % len(self.stages)
        stage = self.stages[stage_index]

        if stage == "research":
            from agents.archetypes.researcher import ResearcherAgent

            return ResearcherAgent(
                name=f"researcher-{index // len(self.stages)}",
                max_turns=5,
                timeout_ms=30_000,
            )
        elif stage == "code":
            from agents.archetypes.coder import CoderAgent

            return CoderAgent(
                name=f"coder-{index // len(self.stages)}",
                max_turns=10,
                timeout_ms=60_000,
            )
        else:  # review
            from agents.archetypes.reviewer import ReviewerAgent

            return ReviewerAgent(
                name=f"reviewer-{index // len(self.stages)}",
                max_turns=5,
                timeout_ms=30_000,
            )

    async def get_task(self, agent: BaseAgent) -> str:
        """Get task based on agent's stage."""
        if "researcher" in agent.name:
            return "Research best practices for implementing a REST API"
        elif "coder" in agent.name:
            return "Implement the API based on research findings"
        else:
            return "Review the implementation for bugs and improvements"


@dataclass
class HierarchyDelegationScenario(Scenario):
    """
    Hierarchy topology stress test.

    One orchestrator delegates to multiple teams.
    Tests:
    - Delegation efficiency
    - Multi-level communication
    - Coordinator bottlenecks
    """

    name: str = "hierarchy_delegation"
    description: str = "Orchestrator delegates to worker teams"
    broadcast_probability: float = 0.1
    direct_message_probability: float = 0.5
    team_size: int = 5
    num_teams: int = 4

    async def spawn_agent(self, index: int) -> BaseAgent:
        """Spawn orchestrator or worker agents."""
        if index == 0:
            # Orchestrator
            from agents.archetypes.orchestrator import OrchestratorAgent

            return OrchestratorAgent(
                name="orchestrator",
                max_turns=20,
                timeout_ms=120_000,
            )
        else:
            # Workers assigned to teams
            team_id = (index - 1) // self.team_size
            worker_id = (index - 1) % self.team_size

            from agents.archetypes.coder import CoderAgent

            return CoderAgent(
                name=f"team{team_id}-worker{worker_id}",
                max_turns=10,
                timeout_ms=60_000,
            )

    async def get_task(self, agent: BaseAgent) -> str:
        """Get task based on role."""
        if "orchestrator" in agent.name:
            return "Coordinate the implementation of a microservices architecture across all teams"
        else:
            team_tasks = [
                "Implement the user authentication service",
                "Build the order processing service",
                "Create the notification service",
                "Develop the analytics dashboard",
            ]
            team_id = int(agent.name.split("-")[0].replace("team", ""))
            return team_tasks[team_id % len(team_tasks)]


@dataclass
class ChaosScenario(Scenario):
    """
    Chaos engineering stress test.

    Injects failures, timeouts, and topology switches.
    Tests:
    - Fault tolerance
    - Recovery mechanisms
    - Circuit breakers
    """

    name: str = "chaos"
    description: str = "Random failures and topology switches"
    failure_rate: float = 0.1
    timeout_rate: float = 0.05
    topology_switch_interval: int = 15  # Switch every 15 seconds
    broadcast_probability: float = 0.2
    direct_message_probability: float = 0.3

    topologies: list[str] = field(
        default_factory=lambda: ["swarm", "hierarchy", "pipeline", "mesh", "ring", "star"]
    )

    async def spawn_agent(self, index: int) -> BaseAgent:
        """Spawn random agent types."""
        agent_types = [
            ("researcher", "agents.archetypes.researcher", "ResearcherAgent"),
            ("coder", "agents.archetypes.coder", "CoderAgent"),
            ("reviewer", "agents.archetypes.reviewer", "ReviewerAgent"),
        ]

        agent_type, module_path, class_name = random.choice(agent_types)
        module = __import__(module_path, fromlist=[class_name])
        agent_class = getattr(module, class_name)

        return cast(
            "BaseAgent",
            agent_class(
                name=f"{agent_type}-{index}",
                max_turns=random.randint(3, 10),
                timeout_ms=random.randint(15_000, 45_000),
            ),
        )

    async def get_task(self, agent: BaseAgent) -> str:
        """Get random task with potential chaos."""
        tasks = [
            "Analyze system architecture",
            "Implement a new feature",
            "Review recent changes",
            "Debug production issue",
            "Write documentation",
        ]

        # Inject failures via task
        if self.should_inject_failure():
            return "CHAOS_INJECT_FAILURE: This task should fail"
        if self.should_inject_timeout():
            return "CHAOS_INJECT_TIMEOUT: This task should timeout"

        return random.choice(tasks)

    def get_next_topology(self, current: str) -> str:
        """Get next topology for switching."""
        candidates = [t for t in self.topologies if t != current]
        return random.choice(candidates)


@dataclass
class ScaleTestScenario(Scenario):
    """
    Pure scale stress test.

    Spawns maximum agents with minimal work to find scale limits.
    """

    name: str = "scale_test"
    description: str = "Maximum agents, minimal work"
    target_agents: int = 100
    broadcast_probability: float = 0.0
    direct_message_probability: float = 0.0

    async def spawn_agent(self, index: int) -> BaseAgent:
        """Spawn lightweight agents."""
        from agents.archetypes.researcher import ResearcherAgent

        return ResearcherAgent(
            name=f"agent-{index}",
            max_turns=1,
            timeout_ms=10_000,
        )

    async def get_task(self, agent: BaseAgent) -> str:
        """Minimal task."""
        return "Return 'OK'"


# Scenario registry
SCENARIOS: dict[str, type[Scenario]] = {
    "swarm": SwarmBrainstormScenario,
    "pipeline": PipelineWorkflowScenario,
    "hierarchy": HierarchyDelegationScenario,
    "chaos": ChaosScenario,
    "scale": ScaleTestScenario,
}


def get_scenario(name: str, **kwargs: Any) -> Scenario:
    """
    Get a scenario by name.

    Args:
        name: Scenario name
        **kwargs: Override scenario parameters

    Returns:
        Configured scenario instance
    """
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {name}. Available: {list(SCENARIOS.keys())}")

    return SCENARIOS[name](**kwargs)


def list_scenarios() -> list[dict[str, str]]:
    """List available scenarios."""
    return [
        {"name": name, "description": cast(Any, cls).description}
        for name, cls in SCENARIOS.items()
    ]
