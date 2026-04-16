"""Emergence Detection for Agent Swarms.

Compares individual agent contributions against collective output to
detect novel information — content that appeared in the collective
synthesis but was not present in any individual agent's trace.

V1 uses token set-difference as a heuristic. The architecture supports
swapping in embedding-based detection (via HiveMind's ChromaDB) later.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("titan.hive.emergence")

# Tokens shorter than this are noise (articles, prepositions)
_MIN_TOKEN_LENGTH = 4

# Default: 10% novel content in collective triggers emergence
_DEFAULT_NOVELTY_THRESHOLD = 0.1


@dataclass(frozen=True)
class EmergenceResult:
    """Result of emergence detection analysis."""

    detected: bool
    evidence: list[str] = field(default_factory=list)
    novelty_ratio: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "detected": self.detected,
            "evidence": list(self.evidence),
            "novelty_ratio": self.novelty_ratio,
        }


_NO_EMERGENCE = EmergenceResult(detected=False)


class EmergenceDetector:
    """Detects emergent information in collective agent output.

    Emergence = information in the collective output that is not
    traceable to any individual agent's contributions.

    Args:
        novelty_threshold: Fraction of novel tokens required to
            trigger emergence detection (0.0 to 1.0). Default 0.1.
        min_token_length: Minimum token length to consider. Shorter
            tokens (articles, prepositions) are filtered as noise.
    """

    def __init__(
        self,
        novelty_threshold: float = _DEFAULT_NOVELTY_THRESHOLD,
        min_token_length: int = _MIN_TOKEN_LENGTH,
    ) -> None:
        if not 0.0 <= novelty_threshold <= 1.0:
            raise ValueError(f"novelty_threshold must be 0.0-1.0, got {novelty_threshold}")
        self._threshold = novelty_threshold
        self._min_token_length = min_token_length

    def detect(
        self,
        agent_contributions: dict[str, list[str]],
        collective_output: str,
    ) -> EmergenceResult:
        """Detect emergence by comparing individual vs collective content.

        Args:
            agent_contributions: Mapping of agent_id to list of text
                contributions (memories, messages, outputs) from that agent.
            collective_output: The synthesized collective output to check
                for novel information.

        Returns:
            EmergenceResult with detection flag, evidence, and novelty ratio.
        """
        if not collective_output or not collective_output.strip():
            return _NO_EMERGENCE

        # Emergence requires multiple agents — one agent alone cannot produce it
        contributing_agents = {
            aid: texts for aid, texts in agent_contributions.items() if texts
        }
        if len(contributing_agents) < 2:
            return _NO_EMERGENCE

        # Tokenize individual contributions into per-agent sets, then union
        individual_union: set[str] = set()
        for texts in contributing_agents.values():
            for text in texts:
                individual_union |= self._tokenize(text)

        collective_tokens = self._tokenize(collective_output)

        if not collective_tokens:
            return _NO_EMERGENCE

        # Novel tokens = in collective but not in any individual
        novel_tokens = collective_tokens - individual_union

        novelty_ratio = len(novel_tokens) / len(collective_tokens) if collective_tokens else 0.0
        detected = novelty_ratio >= self._threshold

        evidence: list[str] = []
        if detected:
            evidence = self._extract_evidence(novel_tokens, collective_output)

        if detected:
            logger.info(
                "Emergence detected: %.1f%% novel content (%d/%d tokens), %d evidence items",
                novelty_ratio * 100,
                len(novel_tokens),
                len(collective_tokens),
                len(evidence),
            )

        return EmergenceResult(
            detected=detected,
            evidence=evidence,
            novelty_ratio=round(novelty_ratio, 4),
        )

    def _tokenize(self, text: str) -> set[str]:
        """Normalize and split text into a token set.

        Lowercases, strips punctuation, filters short tokens.
        """
        # Split on whitespace and punctuation boundaries
        words = re.findall(r"[a-zA-Z0-9]+", text.lower())
        return {w for w in words if len(w) >= self._min_token_length}

    def _extract_evidence(self, novel_tokens: set[str], collective_output: str) -> list[str]:
        """Extract sentences from collective output that contain novel tokens.

        Returns up to 5 evidence sentences, prioritized by novel token density.
        """
        # Split into sentences (period, exclamation, question mark, or newline)
        sentences = re.split(r"[.!?\n]+", collective_output)
        sentences = [s.strip() for s in sentences if s.strip()]

        scored: list[tuple[int, str]] = []
        for sentence in sentences:
            sentence_tokens = self._tokenize(sentence)
            novel_count = len(sentence_tokens & novel_tokens)
            if novel_count > 0:
                scored.append((novel_count, sentence))

        # Sort by novel token density (most novel first), take top 5
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:5]]
