"""
Hive Mind - Episodic Learning System

Learns from topology decisions and task outcomes to improve future selections.
Implements a feedback loop that strengthens good decisions over time.

Inspired by: iGOR's episodic learning patterns
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from hive.topology import TaskProfile, TopologyType

if TYPE_CHECKING:
    from hive.memory import HiveMind

logger = logging.getLogger("titan.hive.learning")


@dataclass
class Episode:
    """
    Represents a single learning episode.

    An episode captures a topology decision and its outcome.
    """

    episode_id: str
    task_description: str
    task_embedding: list[float] | None
    selected_topology: TopologyType
    task_profile: TaskProfile
    agent_count: int
    start_time: datetime
    end_time: datetime | None = None
    outcome: EpisodeOutcome | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "episode_id": self.episode_id,
            "task_description": self.task_description,
            "task_embedding": self.task_embedding,
            "selected_topology": self.selected_topology.value,
            "task_profile": {
                "requires_consensus": self.task_profile.requires_consensus,
                "has_sequential_stages": self.task_profile.has_sequential_stages,
                "needs_fault_tolerance": self.task_profile.needs_fault_tolerance,
                "has_clear_leader": self.task_profile.has_clear_leader,
                "is_voting_based": self.task_profile.is_voting_based,
                "parallel_subtasks": self.task_profile.parallel_subtasks,
                "complexity": self.task_profile.complexity,
                "estimated_agents": self.task_profile.estimated_agents,
            },
            "agent_count": self.agent_count,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "outcome": self.outcome.to_dict() if self.outcome else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Episode:
        """Deserialize from dictionary."""
        profile_data = data.get("task_profile", {})
        profile = TaskProfile(
            requires_consensus=profile_data.get("requires_consensus", False),
            has_sequential_stages=profile_data.get("has_sequential_stages", False),
            needs_fault_tolerance=profile_data.get("needs_fault_tolerance", False),
            has_clear_leader=profile_data.get("has_clear_leader", False),
            is_voting_based=profile_data.get("is_voting_based", False),
            parallel_subtasks=profile_data.get("parallel_subtasks", 0),
            complexity=profile_data.get("complexity", "medium"),
            estimated_agents=profile_data.get("estimated_agents", 2),
        )

        outcome_data = data.get("outcome")
        outcome = EpisodeOutcome.from_dict(outcome_data) if outcome_data else None

        return cls(
            episode_id=data["episode_id"],
            task_description=data["task_description"],
            task_embedding=data.get("task_embedding"),
            selected_topology=TopologyType(data["selected_topology"]),
            task_profile=profile,
            agent_count=data["agent_count"],
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            outcome=outcome,
            metadata=data.get("metadata", {}),
        )


@dataclass
class EpisodeOutcome:
    """Outcome of an episode."""

    success: bool
    completion_time_ms: float
    agent_utilization: float  # 0-1, how busy were agents
    communication_overhead: float  # Number of messages per agent
    topology_switches: int  # How many times topology changed
    error_rate: float  # 0-1
    user_feedback: float | None = None  # -1 to 1 if provided
    notes: str | None = None
    emergence_detected: bool = False  # Collective output contained novel information
    emergence_evidence: list[str] = field(default_factory=list)  # Descriptions of emergent content

    @property
    def score(self) -> float:
        """
        Calculate overall outcome score (0-1).

        Higher is better.
        """
        base_score = 0.5 if self.success else 0.0

        # Penalize long completion times (normalize to 0-0.2)
        time_penalty = min(0.2, self.completion_time_ms / 600000)  # 10 min max
        time_score = 0.2 - time_penalty

        # Reward good utilization (0-0.2)
        util_score = self.agent_utilization * 0.2

        # Penalize high error rate (0-0.1)
        error_score = (1 - self.error_rate) * 0.1

        # User feedback override (if provided)
        if self.user_feedback is not None:
            feedback_score = (self.user_feedback + 1) / 2  # Normalize -1..1 to 0..1
            return feedback_score * 0.5 + (base_score + time_score + util_score + error_score) * 0.5

        return base_score + time_score + util_score + error_score

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "success": self.success,
            "completion_time_ms": self.completion_time_ms,
            "agent_utilization": self.agent_utilization,
            "communication_overhead": self.communication_overhead,
            "topology_switches": self.topology_switches,
            "error_rate": self.error_rate,
            "user_feedback": self.user_feedback,
            "notes": self.notes,
            "score": self.score,
            "emergence_detected": self.emergence_detected,
            "emergence_evidence": self.emergence_evidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EpisodeOutcome:
        """Deserialize from dictionary."""
        return cls(
            success=data["success"],
            completion_time_ms=data["completion_time_ms"],
            agent_utilization=data["agent_utilization"],
            communication_overhead=data["communication_overhead"],
            topology_switches=data["topology_switches"],
            error_rate=data["error_rate"],
            user_feedback=data.get("user_feedback"),
            notes=data.get("notes"),
            emergence_detected=data.get("emergence_detected", False),
            emergence_evidence=data.get("emergence_evidence", []),
        )


@dataclass
class TopologyPreference:
    """Learned preference for a topology given task characteristics."""

    topology: TopologyType
    weight: float = 1.0  # Higher = more preferred
    sample_count: int = 0
    avg_score: float = 0.5
    last_updated: datetime = field(default_factory=datetime.now)

    def update(self, score: float, learning_rate: float = 0.1) -> None:
        """Update preference based on new outcome."""
        self.sample_count += 1
        self.avg_score = self.avg_score * (1 - learning_rate) + score * learning_rate
        self.weight = self._calculate_weight()
        self.last_updated = datetime.now()

    def _calculate_weight(self) -> float:
        """Calculate weight from average score and confidence."""
        # Confidence increases with sample count (sigmoid)
        confidence = 1 / (1 + math.exp(-0.5 * (self.sample_count - 5)))
        # Weight combines score and confidence
        return self.avg_score * confidence

    def to_dict(self) -> dict[str, Any]:
        return {
            "topology": self.topology.value,
            "weight": self.weight,
            "sample_count": self.sample_count,
            "avg_score": self.avg_score,
            "last_updated": self.last_updated.isoformat(),
        }


class EpisodicLearner:
    """
    Learns from topology decisions and outcomes.

    Maintains:
    - Episode history for similar task lookup
    - Topology preferences per task profile
    - Adaptive learning rate based on convergence
    """

    def __init__(
        self,
        hive_mind: HiveMind | None = None,
        persistence_path: Path | str | None = None,
        learning_rate: float = 0.1,
        max_episodes: int = 10000,
    ) -> None:
        self._hive_mind = hive_mind
        self._persistence_path = Path(persistence_path) if persistence_path else None
        self._learning_rate = learning_rate
        self._max_episodes = max_episodes

        self._episodes: list[Episode] = []
        self._preferences: dict[str, dict[TopologyType, TopologyPreference]] = {}
        self._current_episode: Episode | None = None

        # Load persisted data
        if self._persistence_path and self._persistence_path.exists():
            self._load()

    def start_episode(
        self,
        task_description: str,
        selected_topology: TopologyType,
        task_profile: TaskProfile,
        agent_count: int,
        task_embedding: list[float] | None = None,
    ) -> Episode:
        """
        Start recording a new episode.

        Args:
            task_description: Description of the task
            selected_topology: Chosen topology
            task_profile: Analyzed task profile
            agent_count: Number of agents involved
            task_embedding: Optional embedding for similarity lookup

        Returns:
            New episode instance
        """
        episode = Episode(
            episode_id=str(uuid4()),
            task_description=task_description,
            task_embedding=task_embedding,
            selected_topology=selected_topology,
            task_profile=task_profile,
            agent_count=agent_count,
            start_time=datetime.now(),
        )

        self._current_episode = episode
        logger.info(f"Started episode {episode.episode_id} with {selected_topology.value}")
        return episode

    def end_episode(
        self,
        outcome: EpisodeOutcome,
        episode: Episode | None = None,
    ) -> None:
        """
        End an episode with its outcome.

        Args:
            outcome: Episode outcome
            episode: Episode to end (uses current if not provided)
        """
        episode = episode or self._current_episode
        if not episode:
            logger.warning("No episode to end")
            return

        episode.end_time = datetime.now()
        episode.outcome = outcome

        # Store episode
        self._episodes.append(episode)
        if len(self._episodes) > self._max_episodes:
            self._episodes.pop(0)

        # Update preferences
        self._update_preferences(episode)

        # Persist if configured
        if self._persistence_path:
            self._save()

        # Store in Hive Mind if available
        if self._hive_mind:
            asyncio.create_task(self._store_in_hive(episode))

        logger.info(f"Ended episode {episode.episode_id} with score {outcome.score:.2f}")

        if self._current_episode == episode:
            self._current_episode = None

    def _update_preferences(self, episode: Episode) -> None:
        """Update preferences based on episode outcome."""
        if not episode.outcome:
            return

        # Create profile key
        profile_key = self._profile_key(episode.task_profile)

        if profile_key not in self._preferences:
            self._preferences[profile_key] = {}

        preferences = self._preferences[profile_key]

        # Initialize preference for this topology if needed
        if episode.selected_topology not in preferences:
            preferences[episode.selected_topology] = TopologyPreference(
                topology=episode.selected_topology
            )

        # Update with outcome score
        preferences[episode.selected_topology].update(
            episode.outcome.score,
            self._learning_rate,
        )

    def _profile_key(self, profile: TaskProfile) -> str:
        """Create a hashable key from task profile."""
        return (
            f"{profile.requires_consensus}:"
            f"{profile.has_sequential_stages}:"
            f"{profile.needs_fault_tolerance}:"
            f"{profile.has_clear_leader}:"
            f"{profile.is_voting_based}:"
            f"{profile.complexity}"
        )

    def get_recommendation(
        self,
        task_profile: TaskProfile,
        available_topologies: list[TopologyType] | None = None,
    ) -> tuple[TopologyType, float]:
        """
        Get topology recommendation based on learned preferences.

        Args:
            task_profile: Task profile to match
            available_topologies: Restrict to these topologies

        Returns:
            Tuple of (recommended topology, confidence)
        """
        profile_key = self._profile_key(task_profile)
        preferences = self._preferences.get(profile_key, {})

        if not preferences:
            # No learned data, return default based on profile
            return self._default_recommendation(task_profile), 0.3

        available = available_topologies or list(TopologyType)

        # Find best weighted topology
        best_topology = None
        best_weight = -1.0
        total_samples = sum(p.sample_count for p in preferences.values())

        for topology in available:
            if topology in preferences:
                pref = preferences[topology]
                if pref.weight > best_weight:
                    best_weight = pref.weight
                    best_topology = topology

        if best_topology:
            confidence = min(1.0, total_samples / 20)  # Caps at 20 samples
            return best_topology, confidence

        return self._default_recommendation(task_profile), 0.3

    def _default_recommendation(self, profile: TaskProfile) -> TopologyType:
        """Default recommendation when no learning data exists."""
        if profile.is_voting_based:
            return TopologyType.RING
        if profile.requires_consensus:
            return TopologyType.SWARM
        if profile.has_sequential_stages:
            return TopologyType.PIPELINE
        if profile.needs_fault_tolerance:
            return TopologyType.MESH
        if profile.has_clear_leader:
            return TopologyType.STAR
        return TopologyType.SWARM

    def find_similar_episodes(
        self,
        task_description: str,
        task_embedding: list[float] | None = None,
        limit: int = 5,
    ) -> list[Episode]:
        """
        Find similar past episodes.

        Args:
            task_description: Current task description
            task_embedding: Task embedding for similarity
            limit: Maximum episodes to return

        Returns:
            List of similar episodes
        """
        if task_embedding and self._hive_mind:
            # Use vector similarity
            # This would query ChromaDB in a real implementation
            pass

        # Fallback to keyword matching
        keywords = set(task_description.lower().split())
        scored = []

        for episode in self._episodes:
            ep_keywords = set(episode.task_description.lower().split())
            overlap = len(keywords & ep_keywords)
            if overlap > 0:
                scored.append((overlap, episode))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:limit]]

    def get_statistics(self) -> dict[str, Any]:
        """Get learning statistics."""
        total_episodes = len(self._episodes)
        completed = [e for e in self._episodes if e.outcome is not None]

        topology_stats = {}
        for topology in TopologyType:
            episodes = [e for e in completed if e.selected_topology == topology]
            if episodes:
                outcomes = [e.outcome for e in episodes if e.outcome is not None]
                avg_score = sum(outcome.score for outcome in outcomes) / len(outcomes)
                success_rate = sum(1 for outcome in outcomes if outcome.success) / len(outcomes)
                topology_stats[topology.value] = {
                    "count": len(episodes),
                    "avg_score": avg_score,
                    "success_rate": success_rate,
                }

        return {
            "total_episodes": total_episodes,
            "completed_episodes": len(completed),
            "topology_stats": topology_stats,
            "unique_profiles": len(self._preferences),
            "learning_rate": self._learning_rate,
        }

    def _save(self) -> None:
        """Save state to persistence path."""
        if not self._persistence_path:
            return

        self._persistence_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "episodes": [e.to_dict() for e in self._episodes[-1000:]],  # Keep last 1000
            "preferences": {
                k: {t.value: p.to_dict() for t, p in v.items()}
                for k, v in self._preferences.items()
            },
        }

        with open(self._persistence_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.debug(f"Saved {len(self._episodes)} episodes to {self._persistence_path}")

    def _load(self) -> None:
        """Load state from persistence path."""
        if not self._persistence_path or not self._persistence_path.exists():
            return

        try:
            with open(self._persistence_path) as f:
                data = json.load(f)

            self._episodes = [Episode.from_dict(e) for e in data.get("episodes", [])]

            for profile_key, prefs in data.get("preferences", {}).items():
                self._preferences[profile_key] = {}
                for topology_str, pref_data in prefs.items():
                    topology = TopologyType(topology_str)
                    self._preferences[profile_key][topology] = TopologyPreference(
                        topology=topology,
                        weight=pref_data["weight"],
                        sample_count=pref_data["sample_count"],
                        avg_score=pref_data["avg_score"],
                        last_updated=datetime.fromisoformat(pref_data["last_updated"]),
                    )

            logger.info(f"Loaded {len(self._episodes)} episodes from {self._persistence_path}")

        except Exception as e:
            logger.error(f"Failed to load episodes: {e}")

    async def _store_in_hive(self, episode: Episode) -> None:
        """Store episode in Hive Mind for vector similarity."""
        if not self._hive_mind:
            return

        try:
            await self._hive_mind.remember(
                agent_id="episodic_learner",
                content=episode.task_description,
                importance=episode.outcome.score if episode.outcome else 0.5,
                metadata={
                    "episode_id": episode.episode_id,
                    "topology": episode.selected_topology.value,
                    "score": episode.outcome.score if episode.outcome else None,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to store episode in Hive Mind: {e}")


# Singleton learner
_default_learner: EpisodicLearner | None = None


def get_episodic_learner(
    hive_mind: HiveMind | None = None,
    persistence_path: Path | str | None = None,
) -> EpisodicLearner:
    """Get the default episodic learner."""
    global _default_learner
    if _default_learner is None:
        _default_learner = EpisodicLearner(
            hive_mind=hive_mind,
            persistence_path=persistence_path or Path(".titan/learning.json"),
        )
    return _default_learner
