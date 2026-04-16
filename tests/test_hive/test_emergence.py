"""Tests for emergence detection in agent swarms."""

import pytest

from hive.emergence import EmergenceDetector, EmergenceResult


class TestEmergenceResult:
    def test_no_emergence_defaults(self):
        result = EmergenceResult(detected=False)
        assert result.detected is False
        assert result.evidence == []
        assert result.novelty_ratio == 0.0

    def test_to_dict(self):
        result = EmergenceResult(
            detected=True,
            evidence=["novel synthesis found"],
            novelty_ratio=0.15,
        )
        data = result.to_dict()
        assert data["detected"] is True
        assert data["evidence"] == ["novel synthesis found"]
        assert data["novelty_ratio"] == 0.15

    def test_frozen(self):
        result = EmergenceResult(detected=False)
        with pytest.raises(AttributeError):
            result.detected = True  # type: ignore[misc]


class TestEmergenceDetector:
    def setup_method(self):
        self.detector = EmergenceDetector(novelty_threshold=0.1)

    def test_no_emergence_subset_output(self):
        """Collective output that is a strict subset of individual contributions."""
        contributions = {
            "agent-1": ["The weather is warm today"],
            "agent-2": ["Traffic congestion reported downtown"],
        }
        # Collective just restates what agents said
        collective = "The weather is warm and traffic congestion downtown"
        result = self.detector.detect(contributions, collective)
        assert result.detected is False

    def test_clear_emergence(self):
        """Collective output contains novel synthesis not in any individual."""
        contributions = {
            "agent-1": ["Server alpha has high memory usage"],
            "agent-2": ["Database queries are running slowly"],
        }
        # Collective synthesizes a novel conclusion
        collective = (
            "Server alpha has high memory usage and database queries are slow. "
            "Root cause analysis indicates a connection pool exhaustion event "
            "triggering cascading failures across the infrastructure."
        )
        result = self.detector.detect(contributions, collective)
        assert result.detected is True
        assert result.novelty_ratio > 0.0
        assert len(result.evidence) > 0

    def test_empty_collective_output(self):
        """Empty collective output produces no emergence."""
        contributions = {"agent-1": ["data"], "agent-2": ["more data"]}
        result = self.detector.detect(contributions, "")
        assert result.detected is False

    def test_whitespace_collective_output(self):
        """Whitespace-only collective output produces no emergence."""
        contributions = {"agent-1": ["data"], "agent-2": ["more data"]}
        result = self.detector.detect(contributions, "   \n\t  ")
        assert result.detected is False

    def test_single_agent_no_emergence(self):
        """A single agent cannot produce emergence — requires collective."""
        contributions = {
            "agent-1": ["partial data about systems"],
        }
        collective = "Novel conclusion about cascading infrastructure failures"
        result = self.detector.detect(contributions, collective)
        assert result.detected is False

    def test_no_agents_no_emergence(self):
        """No agent contributions produces no emergence."""
        result = self.detector.detect({}, "Some collective output")
        assert result.detected is False

    def test_empty_contributions_no_emergence(self):
        """Agents with empty contribution lists don't count."""
        contributions = {
            "agent-1": [],
            "agent-2": [],
        }
        result = self.detector.detect(contributions, "Some output")
        assert result.detected is False

    def test_overlapping_contributions_with_emergence(self):
        """Agents share some knowledge but collective still produces novel synthesis."""
        contributions = {
            "agent-1": ["System performance metrics show degradation patterns"],
            "agent-2": ["Performance metrics indicate resource contention"],
            "agent-3": ["Metrics collection pipeline is operational"],
        }
        collective = (
            "Performance metrics show degradation patterns consistent with "
            "resource contention. Probabilistic analysis suggests a resonance "
            "phenomenon between garbage collection cycles and checkpoint intervals."
        )
        result = self.detector.detect(contributions, collective)
        assert result.detected is True
        # Novel tokens: probabilistic, resonance, garbage, checkpoint, etc.
        novel_words = {"resonance", "probabilistic"}
        assert any(w in e.lower() for e in result.evidence for w in novel_words)

    def test_custom_threshold_high(self):
        """High threshold requires more novelty to trigger."""
        detector = EmergenceDetector(novelty_threshold=0.9)
        contributions = {
            "agent-1": ["System alpha is operational"],
            "agent-2": ["System beta is operational"],
        }
        # Some novel content but not 90%
        collective = "Systems alpha and beta are operational with minor latency anomalies"
        result = detector.detect(contributions, collective)
        assert result.detected is False

    def test_custom_threshold_zero(self):
        """Zero threshold detects any novelty at all."""
        detector = EmergenceDetector(novelty_threshold=0.0)
        contributions = {
            "agent-1": ["word alpha"],
            "agent-2": ["word beta"],
        }
        # Even a single novel token triggers
        collective = "alpha beta gamma"
        result = detector.detect(contributions, collective)
        assert result.detected is True
        assert result.novelty_ratio > 0.0

    def test_invalid_threshold(self):
        """Threshold must be 0.0-1.0."""
        with pytest.raises(ValueError, match="novelty_threshold"):
            EmergenceDetector(novelty_threshold=1.5)
        with pytest.raises(ValueError, match="novelty_threshold"):
            EmergenceDetector(novelty_threshold=-0.1)

    def test_evidence_limited_to_five(self):
        """Evidence is capped at 5 items."""
        contributions = {
            "agent-1": ["alpha"],
            "agent-2": ["beta"],
        }
        # Many sentences with novel content
        sentences = [f"Novel concept {chr(65 + i)} discovered unexpectedly" for i in range(10)]
        collective = ". ".join(sentences)
        result = EmergenceDetector(novelty_threshold=0.0).detect(contributions, collective)
        assert len(result.evidence) <= 5

    def test_novelty_ratio_precision(self):
        """Novelty ratio is rounded to 4 decimal places."""
        contributions = {
            "agent-1": ["word"],
            "agent-2": ["another word"],
        }
        collective = "word another novel"
        result = EmergenceDetector(novelty_threshold=0.0).detect(contributions, collective)
        # novel_ratio should have at most 4 decimal places
        ratio_str = str(result.novelty_ratio)
        if "." in ratio_str:
            assert len(ratio_str.split(".")[1]) <= 4

    def test_short_tokens_filtered(self):
        """Tokens shorter than min_token_length are filtered as noise."""
        contributions = {
            "agent-1": ["the cat is on the mat"],
            "agent-2": ["a dog is by the door"],
        }
        # "xyz" is only 3 chars — below default min_token_length of 4
        collective = "the cat and dog xyz"
        result = EmergenceDetector(novelty_threshold=0.0).detect(contributions, collective)
        # "xyz" should be filtered (3 chars < 4), no novel tokens remain
        assert result.detected is False

    def test_custom_min_token_length(self):
        """Custom min_token_length changes filtering."""
        detector = EmergenceDetector(novelty_threshold=0.0, min_token_length=2)
        contributions = {
            "agent-1": ["the cat"],
            "agent-2": ["the dog"],
        }
        collective = "cat dog ox"
        result = detector.detect(contributions, collective)
        # "ox" is 2 chars, now passes the filter
        assert result.detected is True

    def test_to_dict_roundtrip_data(self):
        """EmergenceResult.to_dict produces usable data."""
        contributions = {
            "agent-1": ["Server alpha experiencing latency spikes"],
            "agent-2": ["Database connections reaching pool limits"],
        }
        collective = (
            "Server alpha latency correlates with database connection exhaustion. "
            "Implementing circuit breaker pattern recommended."
        )
        result = self.detector.detect(contributions, collective)
        data = result.to_dict()
        assert isinstance(data["detected"], bool)
        assert isinstance(data["evidence"], list)
        assert isinstance(data["novelty_ratio"], float)
