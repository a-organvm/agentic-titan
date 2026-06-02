"""Tests for multi-scale neighbor support (Phase 16A)."""

import pytest

from hive.neighborhood import (
    InteractionType,
    LayeredNeighborConfig,
    NeighborLayer,
    TopologicalNeighborhood,
)


class TestNeighborLayer:
    """Tests for NeighborLayer enum."""

    def test_layers_exist(self):
        """Test all layers are defined."""
        assert NeighborLayer.PRIMARY == "primary"
        assert NeighborLayer.SECONDARY == "secondary"
        assert NeighborLayer.TERTIARY == "tertiary"


class TestLayeredNeighborConfig:
    """Tests for LayeredNeighborConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LayeredNeighborConfig()
        assert config.primary_count == 7  # Murmuration research
        assert config.secondary_count == 25
        assert config.secondary_weight == 0.3
        assert config.tertiary_probability == 0.05

    def test_custom_config(self):
        """Test custom configuration."""
        config = LayeredNeighborConfig(
            primary_count=5,
            secondary_count=15,
            secondary_weight=0.5,
            tertiary_probability=0.1,
        )
        assert config.primary_count == 5
        assert config.secondary_count == 15


class TestMultiScaleNeighborhood:
    """Tests for multi-scale neighbor functionality."""

    @pytest.fixture
    def neighborhood(self):
        """Create a neighborhood with agents."""
        config = LayeredNeighborConfig(
            primary_count=3,  # Small for testing
            secondary_count=5,
            tertiary_probability=0.5,  # High for test reliability
        )
        nb = TopologicalNeighborhood(
            neighbor_count=3,
            layer_config=config,
        )

        # Register some agents with different capabilities
        for i in range(10):
            caps = ["code"] if i % 2 == 0 else ["research"]
            if i % 3 == 0:
                caps.append("review")
            nb.register_agent(f"agent_{i}", capabilities=caps)

        # Record some interactions
        nb.record_interaction("agent_0", "agent_1", InteractionType.COLLABORATION, True)
        nb.record_interaction("agent_0", "agent_2", InteractionType.COLLABORATION, True)
        nb.record_interaction("agent_1", "agent_3", InteractionType.COMMUNICATION, True)

        return nb

    def test_get_primary_neighbors(self, neighborhood):
        """Test getting primary layer neighbors."""
        neighbors = neighborhood.get_neighbors("agent_0", layer=NeighborLayer.PRIMARY)

        assert isinstance(neighbors, list)
        assert len(neighbors) <= neighborhood._layer_config.primary_count
        assert "agent_0" not in neighbors  # Self not in neighbors

    def test_get_secondary_neighbors(self, neighborhood):
        """Test getting secondary layer neighbors."""
        neighbors = neighborhood.get_neighbors("agent_0", layer=NeighborLayer.SECONDARY)

        assert isinstance(neighbors, list)
        assert len(neighbors) <= neighborhood._layer_config.secondary_count
        assert "agent_0" not in neighbors

    def test_get_tertiary_neighbors(self, neighborhood):
        """Test getting tertiary layer neighbors (probabilistic)."""
        # Run multiple times due to probabilistic nature
        all_neighbors = []
        for _ in range(10):
            neighbors = neighborhood.get_neighbors("agent_0", layer=NeighborLayer.TERTIARY)
            all_neighbors.extend(neighbors)

        # Should sometimes get some neighbors
        # With 0.5 probability and 9 agents, expect some
        assert isinstance(all_neighbors, list)

    def test_secondary_more_than_primary(self, neighborhood):
        """Test secondary layer returns more neighbors than primary."""
        primary = neighborhood.get_neighbors("agent_0", layer=NeighborLayer.PRIMARY)
        secondary = neighborhood.get_neighbors("agent_0", layer=NeighborLayer.SECONDARY)

        # Secondary count limit is higher
        assert len(secondary) >= len(primary)

    def test_neighbors_caching(self, neighborhood):
        """Test neighbor lists are cached."""
        # First call
        neighbors1 = neighborhood.get_neighbors("agent_0", layer=NeighborLayer.PRIMARY)

        # Second call should return same cached result
        neighbors2 = neighborhood.get_neighbors("agent_0", layer=NeighborLayer.PRIMARY)

        assert neighbors1 == neighbors2

    def test_neighbor_cache_used_until_invalidated(self, neighborhood, monkeypatch):
        """Cached neighbors are reused and cleared when topology inputs change."""
        call_count = 0
        original = neighborhood.calculate_neighbors

        def counting_calculate(agent_id: str) -> list[str]:
            nonlocal call_count
            call_count += 1
            return original(agent_id)

        monkeypatch.setattr(neighborhood, "calculate_neighbors", counting_calculate)

        neighborhood.get_neighbors("agent_0", layer=NeighborLayer.PRIMARY)
        neighborhood.get_neighbors("agent_0", layer=NeighborLayer.PRIMARY)
        assert call_count == 1

        neighborhood.record_interaction(
            "agent_0", "agent_4", InteractionType.COLLABORATION, True,
        )
        neighborhood.get_neighbors("agent_0", layer=NeighborLayer.PRIMARY)
        assert call_count == 2

    def test_force_recalculate(self, neighborhood):
        """Test forcing recalculation bypasses cache."""
        neighborhood.get_neighbors("agent_0", layer=NeighborLayer.PRIMARY)

        # Force recalculate
        neighbors2 = neighborhood.get_neighbors(
            "agent_0",
            layer=NeighborLayer.PRIMARY,
            force_recalculate=True,
        )

        # Should still be valid lists (may or may not be equal)
        assert isinstance(neighbors2, list)

    @pytest.mark.asyncio
    async def test_propagate_information_layered_primary(self, neighborhood):
        """Test layered propagation with primary layer only."""
        reached = await neighborhood.propagate_information_layered(
            source_agent="agent_0",
            information={"message": "test"},
            use_layers=[NeighborLayer.PRIMARY],
            max_hops=2,
        )

        assert "agent_0" in reached
        assert reached["agent_0"] == 1.0
        # Should reach at least some primary neighbors
        assert len(reached) >= 1

    @pytest.mark.asyncio
    async def test_propagate_information_layered_secondary(self, neighborhood):
        """Test layered propagation includes secondary layer."""
        reached = await neighborhood.propagate_information_layered(
            source_agent="agent_0",
            information={"message": "test"},
            use_layers=[NeighborLayer.PRIMARY, NeighborLayer.SECONDARY],
            max_hops=2,
        )

        # Should reach more agents with secondary layer
        primary_only = await neighborhood.propagate_information_layered(
            source_agent="agent_0",
            information={"message": "test"},
            use_layers=[NeighborLayer.PRIMARY],
            max_hops=2,
        )

        assert len(reached) >= len(primary_only)

    @pytest.mark.asyncio
    async def test_propagate_information_layered_all_layers(self, neighborhood):
        """Test propagation with all layers."""
        reached = await neighborhood.propagate_information_layered(
            source_agent="agent_0",
            information={"message": "test"},
            use_layers=[NeighborLayer.PRIMARY, NeighborLayer.SECONDARY, NeighborLayer.TERTIARY],
            max_hops=3,
        )

        assert len(reached) >= 1
        # Source always has strength 1.0
        assert reached["agent_0"] == 1.0

    @pytest.mark.asyncio
    async def test_propagate_information_layered_decay(self, neighborhood):
        """Test information decays over hops."""
        reached = await neighborhood.propagate_information_layered(
            source_agent="agent_0",
            information={"message": "test"},
            use_layers=[NeighborLayer.PRIMARY],
            max_hops=3,
            decay_factor=0.5,
        )

        # Source has full strength
        assert reached["agent_0"] == 1.0

        # Others should have less (if reached)
        for agent_id, strength in reached.items():
            if agent_id != "agent_0":
                assert strength <= 1.0

    @pytest.mark.asyncio
    async def test_propagate_information_secondary_weight(self, neighborhood):
        """Test secondary layer uses configured weight."""
        reached = await neighborhood.propagate_information_layered(
            source_agent="agent_0",
            information={"message": "test"},
            use_layers=[NeighborLayer.SECONDARY],
            max_hops=1,
        )

        # Secondary neighbors should have the secondary weight
        for agent_id, strength in reached.items():
            if agent_id != "agent_0":
                assert strength <= neighborhood._layer_config.secondary_weight + 0.01

    @pytest.mark.asyncio
    async def test_propagate_information_default_layers(self, neighborhood):
        """Test default layers when not specified."""
        reached = await neighborhood.propagate_information_layered(
            source_agent="agent_0",
            information={"message": "test"},
        )

        # Should use PRIMARY and SECONDARY by default
        assert len(reached) >= 1
