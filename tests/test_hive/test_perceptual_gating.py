"""Tests for perceptual gating — topology as pheromone field filter."""

import pytest

from hive.stigmergy import PheromoneField, TraceType
from hive.topology import (
    FissionFusionTopology,
    HierarchyTopology,
    MeshTopology,
    PipelineTopology,
    RingTopology,
    SensingRegion,
    StarTopology,
    StigmergicTopology,
    SwarmTopology,
)


class TestSensingRegion:
    def test_full_field(self):
        region = SensingRegion.full_field()
        assert region.locations is None
        assert region.trace_types is None
        assert region.contains_location("anywhere")

    def test_restricted(self):
        region = SensingRegion.restricted({"loc-a", "loc-b"})
        assert region.contains_location("loc-a")
        assert region.contains_location("loc-b")
        assert not region.contains_location("loc-c")

    def test_min_intensity(self):
        region = SensingRegion.full_field(min_intensity=0.5)
        assert region.min_intensity == 0.5

    def test_restricted_with_invariant_types(self):
        region = SensingRegion.restricted(
            {"loc-a"},
            trace_types={"path"},
            invariant_types={"c:sync"},
            transition_buffer=0.25,
        )
        assert region.allows_trace_type("path")
        assert not region.allows_trace_type("resource")
        assert region.is_invariant_type("collaboration", "c:sync")

    def test_transition_buffer_advances(self):
        region = SensingRegion.restricted({"loc-a"}, transition_buffer=0.5)
        advanced = region.next_transition_buffer(step=0.3)
        assert advanced.transition_buffer == pytest.approx(0.8)

    def test_frozen(self):
        region = SensingRegion.full_field()
        with pytest.raises(AttributeError):
            region.min_intensity = 0.9  # type: ignore[misc]


class TestSwarmSensing:
    def test_full_field(self):
        """SWARM agents sense the entire field."""
        topo = SwarmTopology()
        topo.add_agent("a1", "Agent 1", [])
        topo.add_agent("a2", "Agent 2", [])
        region = topo.get_sensing_region("a1")
        assert region.locations is None  # Full field

    def test_all_agents_full_field(self):
        """Every SWARM agent gets full-field sensing."""
        topo = SwarmTopology()
        for i in range(5):
            topo.add_agent(f"a{i}", f"Agent {i}", [])
        for i in range(5):
            assert topo.get_sensing_region(f"a{i}").locations is None


class TestHierarchySensing:
    def test_root_sees_children(self):
        """Root agent sees itself + children."""
        topo = HierarchyTopology()
        topo.add_agent("root", "Root", [], role="leader")
        topo.add_agent("c1", "Child 1", [], parent_id="root")
        topo.add_agent("c2", "Child 2", [], parent_id="root")
        region = topo.get_sensing_region("root")
        assert "root" in region.locations
        assert "c1" in region.locations
        assert "c2" in region.locations

    def test_child_sees_parent(self):
        """Child agent sees itself + parent."""
        topo = HierarchyTopology()
        topo.add_agent("root", "Root", [], role="leader")
        topo.add_agent("child", "Child", [], parent_id="root")
        region = topo.get_sensing_region("child")
        assert "child" in region.locations
        assert "root" in region.locations

    def test_child_does_not_see_sibling(self):
        """Child agent does not see sibling directly."""
        topo = HierarchyTopology()
        topo.add_agent("root", "Root", [], role="leader")
        topo.add_agent("c1", "Child 1", [], parent_id="root")
        topo.add_agent("c2", "Child 2", [], parent_id="root")
        region = topo.get_sensing_region("c1")
        assert "c2" not in region.locations


class TestPipelineSensing:
    def test_middle_stage_sees_adjacent(self):
        """Middle pipeline stage sees upstream + downstream."""
        topo = PipelineTopology()
        topo.add_agent("s0", "Stage 0", [], stage=0)
        topo.add_agent("s1", "Stage 1", [], stage=1)
        topo.add_agent("s2", "Stage 2", [], stage=2)
        region = topo.get_sensing_region("s1")
        assert "s0" in region.locations  # upstream
        assert "s1" in region.locations  # self
        assert "s2" in region.locations  # downstream

    def test_first_stage_no_upstream(self):
        """First stage has no upstream to sense."""
        topo = PipelineTopology()
        topo.add_agent("s0", "Stage 0", [], stage=0)
        topo.add_agent("s1", "Stage 1", [], stage=1)
        region = topo.get_sensing_region("s0")
        assert "s0" in region.locations
        assert "s1" in region.locations
        assert len(region.locations) == 2


class TestRingSensing:
    def test_sees_neighbors(self):
        """Ring agent sees self + adjacent ring positions."""
        topo = RingTopology()
        topo.add_agent("a", "Agent A", [])
        topo.add_agent("b", "Agent B", [])
        topo.add_agent("c", "Agent C", [])
        region = topo.get_sensing_region("b")
        assert "b" in region.locations
        # Ring neighbors (at least one adjacent)
        assert len(region.locations) >= 2


class TestStarSensing:
    def test_hub_sees_all(self):
        """Star hub has full-field sensing."""
        topo = StarTopology()
        topo.add_agent("hub", "Hub", [], role="hub")
        topo.add_agent("s1", "Spoke 1", [])
        topo.add_agent("s2", "Spoke 2", [])
        region = topo.get_sensing_region("hub")
        assert region.locations is None  # Full field

    def test_spoke_sees_hub_only(self):
        """Star spoke senses only hub + self."""
        topo = StarTopology()
        topo.add_agent("hub", "Hub", [], role="hub")
        topo.add_agent("s1", "Spoke 1", [])
        topo.add_agent("s2", "Spoke 2", [])
        region = topo.get_sensing_region("s1")
        assert "s1" in region.locations
        assert "hub" in region.locations
        assert "s2" not in region.locations


class TestFissionFusionSensing:
    def test_cluster_local(self):
        """Agent senses only its own cluster."""
        topo = FissionFusionTopology()
        topo.add_agent("a1", "Agent 1", [], cluster_id="alpha")
        topo.add_agent("a2", "Agent 2", [], cluster_id="alpha")
        topo.add_agent("b1", "Agent B1", [], cluster_id="beta")
        region = topo.get_sensing_region("a1")
        assert "a1" in region.locations
        assert "a2" in region.locations
        assert "b1" not in region.locations

    def test_after_fission(self):
        """After fission, sensing region changes to new cluster."""
        topo = FissionFusionTopology()
        topo.add_agent("a1", "Agent 1", [], cluster_id="main")
        topo.add_agent("a2", "Agent 2", [], cluster_id="main")
        topo.add_agent("a3", "Agent 3", [], cluster_id="main")
        topo.fission("main", "split", ["a3"])
        region = topo.get_sensing_region("a3")
        assert "a3" in region.locations
        assert "a1" not in region.locations

    def test_transition_buffer_preserves_pre_fusion_boundary(self):
        topo = FissionFusionTopology()
        topo.add_agent("a1", "Agent 1", [], cluster_id="alpha")
        topo.add_agent("a2", "Agent 2", [], cluster_id="alpha")
        topo.add_agent("b1", "Agent B1", [], cluster_id="beta")

        topo.fusion("beta", "alpha")

        region = topo.get_sensing_region("a1")
        assert region.locations == frozenset({"a1", "a2"})
        assert region.transition_buffer == 0.0
        assert "c:sync" in region.invariant_types

    def test_transition_buffer_clears_after_full_pass(self):
        topo = FissionFusionTopology(transition_buffer_step=0.6)
        topo.add_agent("a1", "Agent 1", [], cluster_id="alpha")
        topo.add_agent("b1", "Agent B1", [], cluster_id="beta")
        topo.fusion("beta", "alpha")

        assert topo.advance_transition_buffer() == pytest.approx(0.6)
        assert topo.advance_transition_buffer() is None
        region = topo.get_sensing_region("a1")
        assert region.transition_buffer is None
        assert region.locations == frozenset({"a1", "b1"})


class TestStigmergicSensing:
    def test_full_field(self):
        """Stigmergic agents have full-field sensing (native mode)."""
        topo = StigmergicTopology()
        topo.add_agent("a1", "Agent 1", [])
        region = topo.get_sensing_region("a1")
        assert region.locations is None


class TestMeshSensing:
    def test_sees_neighbors(self):
        """Mesh agent sees connected neighbors."""
        topo = MeshTopology()
        topo.add_agent("a1", "Agent 1", [])
        topo.add_agent("a2", "Agent 2", [])
        topo.add_agent("a3", "Agent 3", [])
        region = topo.get_sensing_region("a1")
        assert "a1" in region.locations
        # Mesh connects agents to neighbors
        assert len(region.locations) >= 1


class TestPheromoneFieldFiltered:
    @pytest.mark.asyncio
    async def test_full_field_sensing(self):
        """Full-field region returns all traces."""
        field = PheromoneField()
        await field.deposit("a1", TraceType.PATH, "loc-a", 0.8)
        await field.deposit("a2", TraceType.RESOURCE, "loc-b", 0.6)

        region = SensingRegion.full_field()
        traces = await field.sense_filtered(region)
        assert len(traces) == 2

    @pytest.mark.asyncio
    async def test_restricted_region(self):
        """Restricted region only returns traces at specified locations."""
        field = PheromoneField()
        await field.deposit("a1", TraceType.PATH, "loc-a", 0.8)
        await field.deposit("a2", TraceType.RESOURCE, "loc-b", 0.6)
        await field.deposit("a3", TraceType.WARNING, "loc-c", 0.9)

        region = SensingRegion.restricted({"loc-a", "loc-c"})
        traces = await field.sense_filtered(region)
        assert len(traces) == 2
        locations = {t.location for t in traces}
        assert locations == {"loc-a", "loc-c"}

    @pytest.mark.asyncio
    async def test_min_intensity_filter(self):
        """Traces below min_intensity are filtered."""
        field = PheromoneField()
        await field.deposit("a1", TraceType.PATH, "loc-a", 0.3)
        await field.deposit("a2", TraceType.PATH, "loc-a", 0.8)

        region = SensingRegion.full_field(min_intensity=0.5)
        traces = await field.sense_filtered(region)
        assert len(traces) == 1
        assert traces[0].intensity == 0.8

    @pytest.mark.asyncio
    async def test_empty_field(self):
        """Empty field returns no traces."""
        field = PheromoneField()
        region = SensingRegion.full_field()
        traces = await field.sense_filtered(region)
        assert traces == []

    @pytest.mark.asyncio
    async def test_no_matching_locations(self):
        """Region with no matching locations returns empty."""
        field = PheromoneField()
        await field.deposit("a1", TraceType.PATH, "loc-a", 0.8)

        region = SensingRegion.restricted({"loc-z"})
        traces = await field.sense_filtered(region)
        assert traces == []

    @pytest.mark.asyncio
    async def test_invariant_type_bypasses_location_type_and_intensity(self):
        field = PheromoneField()
        await field.deposit("a1", TraceType.PATH, "loc-a", 0.9)
        invariant = await field.deposit(
            "b1",
            TraceType.COLLABORATION,
            "loc-b",
            0.1,
            payload={"atom_type": "c:sync"},
        )

        region = SensingRegion.restricted(
            {"loc-a"},
            min_intensity=0.5,
            trace_types={"path"},
            invariant_types={"c:sync"},
        )
        traces = await field.sense_filtered(region)

        assert {trace.trace_id for trace in traces} == {"trace_default_1", invariant.trace_id}

    @pytest.mark.asyncio
    async def test_transition_buffer_attenuates_foreign_non_invariant_traces(self):
        field = PheromoneField()
        local = await field.deposit("a1", TraceType.PATH, "loc-a", 0.8)
        foreign = await field.deposit(
            "b1",
            TraceType.RESOURCE,
            "loc-b",
            0.8,
            payload={"atom_type": "o:agree"},
        )
        invariant = await field.deposit(
            "b2",
            TraceType.COLLABORATION,
            "loc-b",
            0.3,
            payload={"atom_type": "c:sync"},
        )

        region = SensingRegion.restricted(
            {"loc-a"},
            invariant_types={"c:sync"},
            transition_buffer=0.25,
        )
        traces = await field.sense_filtered(region)
        by_id = {trace.trace_id: trace for trace in traces}

        assert by_id[local.trace_id].intensity == pytest.approx(0.8)
        assert by_id[foreign.trace_id].intensity == pytest.approx(0.2)
        assert by_id[invariant.trace_id].intensity == pytest.approx(0.3)
        assert foreign.intensity == pytest.approx(0.8)

    @pytest.mark.asyncio
    async def test_zero_transition_buffer_blocks_foreign_non_invariant_traces(self):
        field = PheromoneField()
        local = await field.deposit("a1", TraceType.PATH, "loc-a", 0.8)
        foreign = await field.deposit("b1", TraceType.RESOURCE, "loc-b", 0.8)

        region = SensingRegion.restricted({"loc-a"}, transition_buffer=0.0)
        traces = await field.sense_filtered(region)

        assert {trace.trace_id for trace in traces} == {local.trace_id}
        assert foreign.trace_id not in {trace.trace_id for trace in traces}
