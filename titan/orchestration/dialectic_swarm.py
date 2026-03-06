"""
Titan Orchestration - Dialectic Swarm

Orchestrates a triadic agent swarm (Thesis, Antithesis, Synthesis) 
to resolve complex system problems with high-fidelity reasoning.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

from titan.analysis.dialectic import DialecticSynthesizer, SynthesisResult, SynthesisStrategy
from titan.analysis.contradictions import Contradiction, ContradictionType
from titan.learning.local_trainer import get_trainer

logger = logging.getLogger("titan.orchestration.dialectic_swarm")

@dataclass
class SwarmTurn:
    """A single turn in the dialectic dialogue."""
    role: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

class DialecticSwarm:
    """
    Triadic Swarm Orchestrator.
    
    Roles:
    - Thesis: Proposes a solution
    - Antithesis: Identifies flaws and counter-perspectives
    - Synthesis: Resolves the conflict into a verified plan
    """
    
    def __init__(self, llm_caller: Any):
        self.llm_caller = llm_caller
        self.synthesizer = DialecticSynthesizer(llm_caller=llm_caller)
        self.trainer = get_trainer()

    async def resolve(self, problem_description: str) -> SynthesisResult:
        """Execute the triadic swarm loop to resolve a problem."""
        logger.info(f"Swarm activated for problem: {problem_description[:50]}...")
        
        # 1. THESIS: Generate proposal
        thesis_content = await self._spawn_thesis(problem_description)
        
        # 2. ANTITHESIS: Challenge the proposal
        antithesis_content = await self._spawn_antithesis(problem_description, thesis_content)
        
        # 3. CONTRADICTION EXTRACTION
        contradiction = Contradiction(
            source_a="ThesisAgent",
            content_a=thesis_content,
            source_b="AntithesisAgent",
            content_b=antithesis_content,
            explanation=f"Conflict regarding: {problem_description}",
            contradiction_type=ContradictionType.LOGICAL
        )
        
        # 4. SYNTHESIS: Generate the final plan
        result = await self.synthesizer.synthesize_contradiction(contradiction)
        
        # 5. STYLE ADAPTATION
        style_context = self.trainer.get_adapter().get_prompt_context()
        final_prompt = f"Given this synthesized solution:\n{result.synthesis}\n\nAdapt it to this style guide:\n{style_context}\n\nProduce the FINAL EXECUTABLE PLAN."
        
        final_plan = await self.llm_caller(final_prompt, "titan-synthesis")
        result.synthesis = final_plan if isinstance(final_plan, str) else str(final_plan)
        
        return result

    async def _spawn_thesis(self, problem: str) -> str:
        prompt = f"Role: ARCHITECT. Task: Propose a comprehensive solution for: {problem}"
        resp = await self.llm_caller(prompt, "titan-thesis")
        return resp if isinstance(resp, str) else str(resp)

    async def _spawn_antithesis(self, problem: str, thesis: str) -> str:
        prompt = f"Role: ADVERSARY. Task: Find critical flaws, security risks, and edge cases in this proposal:\n{thesis}\n\nContext: {problem}"
        resp = await self.llm_caller(prompt, "titan-antithesis")
        return resp if isinstance(resp, str) else str(resp)
