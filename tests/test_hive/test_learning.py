"""Tests for hive episodic learning system."""


from hive.learning import EpisodeOutcome


class TestEpisodeOutcome:
    """Tests for EpisodeOutcome dataclass."""

    def _make_outcome(self, **overrides) -> EpisodeOutcome:
        """Create an EpisodeOutcome with sensible defaults."""
        defaults = {
            "success": True,
            "completion_time_ms": 5000.0,
            "agent_utilization": 0.8,
            "communication_overhead": 3.0,
            "topology_switches": 2,
            "error_rate": 0.05,
        }
        defaults.update(overrides)
        return EpisodeOutcome(**defaults)

    def test_default_emergence_fields(self):
        """New fields default to no-emergence state."""
        outcome = self._make_outcome()
        assert outcome.emergence_detected is False
        assert outcome.emergence_evidence == []

    def test_emergence_detected_flag(self):
        """emergence_detected can be set to True."""
        outcome = self._make_outcome(emergence_detected=True)
        assert outcome.emergence_detected is True

    def test_emergence_evidence_list(self):
        """emergence_evidence stores descriptions of novel information."""
        evidence = [
            (
                "Agents A+B deposited partial route data; collective output "
                "contained optimal route neither had"
            ),
            "Synthesis produced a classification schema not present in any individual trace",
        ]
        outcome = self._make_outcome(
            emergence_detected=True,
            emergence_evidence=evidence,
        )
        assert len(outcome.emergence_evidence) == 2
        assert "optimal route" in outcome.emergence_evidence[0]

    def test_score_unaffected_by_emergence(self):
        """Score calculation is unchanged — emergence is metadata, not a score modifier."""
        base = self._make_outcome()
        with_emergence = self._make_outcome(
            emergence_detected=True,
            emergence_evidence=["novel synthesis found"],
        )
        assert base.score == with_emergence.score

    def test_to_dict_includes_emergence(self):
        """Serialization includes emergence fields."""
        outcome = self._make_outcome(
            emergence_detected=True,
            emergence_evidence=["novel pattern"],
        )
        data = outcome.to_dict()
        assert data["emergence_detected"] is True
        assert data["emergence_evidence"] == ["novel pattern"]

    def test_to_dict_default_emergence(self):
        """Serialization includes default emergence values."""
        data = self._make_outcome().to_dict()
        assert data["emergence_detected"] is False
        assert data["emergence_evidence"] == []

    def test_from_dict_with_emergence(self):
        """Deserialization restores emergence fields."""
        data = self._make_outcome(
            emergence_detected=True,
            emergence_evidence=["emergent behavior X"],
        ).to_dict()
        restored = EpisodeOutcome.from_dict(data)
        assert restored.emergence_detected is True
        assert restored.emergence_evidence == ["emergent behavior X"]

    def test_from_dict_without_emergence_fields(self):
        """Deserialization of legacy data (no emergence fields) defaults safely."""
        legacy_data = {
            "success": True,
            "completion_time_ms": 5000.0,
            "agent_utilization": 0.8,
            "communication_overhead": 3.0,
            "topology_switches": 2,
            "error_rate": 0.05,
        }
        outcome = EpisodeOutcome.from_dict(legacy_data)
        assert outcome.emergence_detected is False
        assert outcome.emergence_evidence == []

    def test_roundtrip_serialization(self):
        """Full roundtrip: create -> to_dict -> from_dict preserves all fields."""
        original = self._make_outcome(
            success=False,
            completion_time_ms=12000.0,
            agent_utilization=0.6,
            communication_overhead=8.0,
            topology_switches=5,
            error_rate=0.15,
            user_feedback=0.3,
            notes="partial success with emergence",
            emergence_detected=True,
            emergence_evidence=["novel inference A", "novel inference B"],
        )
        restored = EpisodeOutcome.from_dict(original.to_dict())
        assert restored.success == original.success
        assert restored.completion_time_ms == original.completion_time_ms
        assert restored.emergence_detected == original.emergence_detected
        assert restored.emergence_evidence == original.emergence_evidence
        assert restored.user_feedback == original.user_feedback
        assert restored.notes == original.notes
