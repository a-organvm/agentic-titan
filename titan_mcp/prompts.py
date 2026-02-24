"""
MCP Prompts API - Predefined prompts for inquiry workflows.

Provides prompt templates that MCP clients can use to start
common inquiry types with optimal configurations.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger("titan.mcp.prompts")


class PromptCategory(StrEnum):
    """Categories of MCP prompts."""

    INQUIRY = "inquiry"
    MODEL = "model"
    ANALYSIS = "analysis"
    WORKFLOW = "workflow"


@dataclass
class MCPPromptArgument:
    """
    An argument for an MCP prompt.

    Follows MCP prompt argument schema.
    """

    name: str
    description: str
    required: bool = True
    default: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to MCP format."""
        result: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "required": self.required,
        }
        if self.default is not None:
            result["default"] = self.default
        return result


@dataclass
class MCPPrompt:
    """
    An MCP prompt template.

    Prompts are pre-defined conversation starters that help users
    invoke common operations with the right parameters.
    """

    name: str
    description: str
    category: PromptCategory
    arguments: list[MCPPromptArgument] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to MCP format."""
        return {
            "name": self.name,
            "description": self.description,
            "arguments": [a.to_dict() for a in self.arguments],
        }

    def get_messages(self, arguments: dict[str, str]) -> list[dict[str, Any]]:
        """
        Generate prompt messages from arguments.

        Args:
            arguments: User-provided argument values

        Returns:
            List of message dicts suitable for LLM
        """
        # Validate required arguments
        for arg in self.arguments:
            if arg.required and arg.name not in arguments:
                if arg.default is not None:
                    arguments[arg.name] = arg.default
                else:
                    raise ValueError(f"Missing required argument: {arg.name}")

        # Get generator for this prompt
        generator = PROMPT_GENERATORS.get(self.name)
        if generator:
            return generator(arguments)

        # Default simple prompt
        return [
            {
                "role": "user",
                "content": f"Execute {self.name} with: {arguments}",
            }
        ]


# =============================================================================
# Prompt Definitions
# =============================================================================

EXPANSIVE_INQUIRY_PROMPT = MCPPrompt(
    name="expansive-inquiry",
    description=(
        "Start a comprehensive 6-stage inquiry exploring a topic from multiple "
        "cognitive perspectives: scope clarification, logical analysis, intuitive "
        "exploration, lateral thinking, recursive design, and pattern recognition."
    ),
    category=PromptCategory.INQUIRY,
    arguments=[
        MCPPromptArgument(
            name="topic",
            description="The topic or question to explore in depth",
            required=True,
        ),
        MCPPromptArgument(
            name="context",
            description="Optional background context or constraints",
            required=False,
            default="",
        ),
        MCPPromptArgument(
            name="focus_areas",
            description="Specific areas to emphasize (comma-separated)",
            required=False,
            default="",
        ),
    ],
    metadata={
        "workflow": "expansive",
        "stages": 6,
        "estimated_duration": "5-10 minutes",
    },
)

QUICK_INQUIRY_PROMPT = MCPPrompt(
    name="quick-inquiry",
    description=(
        "Start a streamlined 3-stage inquiry for faster exploration. "
        "Covers scope clarification, logical analysis, and pattern synthesis."
    ),
    category=PromptCategory.INQUIRY,
    arguments=[
        MCPPromptArgument(
            name="topic",
            description="The topic or question to explore",
            required=True,
        ),
        MCPPromptArgument(
            name="depth",
            description="Depth level: shallow, moderate, or deep",
            required=False,
            default="moderate",
        ),
    ],
    metadata={
        "workflow": "quick",
        "stages": 3,
        "estimated_duration": "2-3 minutes",
    },
)

CREATIVE_INQUIRY_PROMPT = MCPPrompt(
    name="creative-inquiry",
    description=(
        "Start a 4-stage creative inquiry emphasizing intuitive and lateral thinking. "
        "Best for artistic, philosophical, or open-ended topics."
    ),
    category=PromptCategory.INQUIRY,
    arguments=[
        MCPPromptArgument(
            name="topic",
            description="The creative topic or question to explore",
            required=True,
        ),
        MCPPromptArgument(
            name="style",
            description="Exploration style: metaphorical, narrative, experimental",
            required=False,
            default="metaphorical",
        ),
    ],
    metadata={
        "workflow": "creative",
        "stages": 4,
        "estimated_duration": "3-5 minutes",
    },
)

MODEL_COMPARISON_PROMPT = MCPPrompt(
    name="model-comparison",
    description=(
        "Compare two AI models across cognitive dimensions. "
        "Shows radar chart of capabilities and recommendations for task types."
    ),
    category=PromptCategory.MODEL,
    arguments=[
        MCPPromptArgument(
            name="model_a",
            description="First model ID to compare",
            required=True,
        ),
        MCPPromptArgument(
            name="model_b",
            description="Second model ID to compare",
            required=True,
        ),
        MCPPromptArgument(
            name="task_type",
            description="Optional task type for context-specific comparison",
            required=False,
            default="",
        ),
    ],
    metadata={
        "visualization": "radar",
        "output_format": "structured",
    },
)

ROUTE_TASK_PROMPT = MCPPrompt(
    name="route-task",
    description=(
        "Get a model routing recommendation for a specific task. "
        "Analyzes task requirements and suggests the optimal model."
    ),
    category=PromptCategory.MODEL,
    arguments=[
        MCPPromptArgument(
            name="task_description",
            description="Description of the task to route",
            required=True,
        ),
        MCPPromptArgument(
            name="requirements",
            description="Specific requirements (speed, accuracy, creativity, etc.)",
            required=False,
            default="",
        ),
        MCPPromptArgument(
            name="constraints",
            description="Constraints (budget, latency, model preferences)",
            required=False,
            default="",
        ),
    ],
    metadata={
        "output_format": "recommendation",
    },
)

CONTRADICTION_ANALYSIS_PROMPT = MCPPrompt(
    name="contradiction-analysis",
    description=(
        "Analyze inquiry results for contradictions and synthesize dialectically. "
        "Identifies logical, semantic, and value contradictions."
    ),
    category=PromptCategory.ANALYSIS,
    arguments=[
        MCPPromptArgument(
            name="session_id",
            description="Inquiry session ID to analyze",
            required=True,
        ),
        MCPPromptArgument(
            name="sensitivity",
            description="Detection sensitivity: low, medium, high",
            required=False,
            default="medium",
        ),
    ],
    metadata={
        "output_format": "report",
    },
)

WORKFLOW_EXECUTE_PROMPT = MCPPrompt(
    name="workflow-execute",
    description=(
        "Execute a named workflow with custom configuration. "
        "Supports sequential, parallel, and staged execution modes."
    ),
    category=PromptCategory.WORKFLOW,
    arguments=[
        MCPPromptArgument(
            name="workflow_name",
            description="Name of workflow to execute (expansive, quick, creative)",
            required=True,
        ),
        MCPPromptArgument(
            name="topic",
            description="Topic for the workflow",
            required=True,
        ),
        MCPPromptArgument(
            name="execution_mode",
            description="Execution mode: sequential, parallel, staged",
            required=False,
            default="staged",
        ),
    ],
    metadata={
        "output_format": "session",
    },
)


# =============================================================================
# Prompt Message Generators
# =============================================================================


def _generate_expansive_inquiry_messages(args: dict[str, str]) -> list[dict[str, Any]]:
    """Generate messages for expansive inquiry prompt."""
    topic = args["topic"]
    context = args.get("context", "")
    focus_areas = args.get("focus_areas", "")

    content_parts = [
        f"Please conduct a comprehensive multi-perspective inquiry on: {topic}",
        "",
        "This inquiry should proceed through 6 cognitive stages:",
        "1. Scope Clarification - Refine and focus the question",
        "2. Logical Analysis - Systematic rational exploration",
        "3. Intuitive Exploration - Metaphorical and narrative thinking",
        "4. Lateral Thinking - Cross-domain connections",
        "5. Recursive Design - Meta-analysis and self-improvement",
        "6. Pattern Recognition - Emergent patterns across all stages",
    ]

    if context:
        content_parts.extend(["", f"Context: {context}"])

    if focus_areas:
        content_parts.extend(["", f"Focus areas: {focus_areas}"])

    return [{"role": "user", "content": "\n".join(content_parts)}]


def _generate_quick_inquiry_messages(args: dict[str, str]) -> list[dict[str, Any]]:
    """Generate messages for quick inquiry prompt."""
    topic = args["topic"]
    depth = args.get("depth", "moderate")

    return [
        {
            "role": "user",
            "content": (
                f"Conduct a streamlined inquiry on: {topic}\n\n"
                f"Depth level: {depth}\n\n"
                "Focus on:\n"
                "1. Clarifying the scope and key questions\n"
                "2. Logical analysis of core aspects\n"
                "3. Pattern synthesis and key insights"
            ),
        }
    ]


def _generate_creative_inquiry_messages(args: dict[str, str]) -> list[dict[str, Any]]:
    """Generate messages for creative inquiry prompt."""
    topic = args["topic"]
    style = args.get("style", "metaphorical")

    return [
        {
            "role": "user",
            "content": (
                f"Conduct a creative exploration of: {topic}\n\n"
                f"Exploration style: {style}\n\n"
                "Approach this through:\n"
                "1. Scope clarification with creative framing\n"
                "2. Intuitive/metaphorical exploration\n"
                "3. Lateral/cross-domain connections\n"
                "4. Pattern synthesis and emergent insights"
            ),
        }
    ]


def _generate_model_comparison_messages(args: dict[str, str]) -> list[dict[str, Any]]:
    """Generate messages for model comparison prompt."""
    model_a = args["model_a"]
    model_b = args["model_b"]
    task_type = args.get("task_type", "")

    content = (
        f"Compare the cognitive capabilities of {model_a} and {model_b}.\n\n"
        "Analyze across dimensions:\n"
        "- Structured reasoning\n"
        "- Creative synthesis\n"
        "- Mathematical analysis\n"
        "- Cross-domain thinking\n"
        "- Meta-analysis\n"
        "- Pattern recognition"
    )

    if task_type:
        content += f"\n\nFor context: This comparison is for {task_type} tasks."

    return [{"role": "user", "content": content}]


def _generate_route_task_messages(args: dict[str, str]) -> list[dict[str, Any]]:
    """Generate messages for task routing prompt."""
    task = args["task_description"]
    requirements = args.get("requirements", "")
    constraints = args.get("constraints", "")

    content_parts = [
        f"Recommend the optimal AI model for this task: {task}",
        "",
        "Consider:",
        "- Task cognitive requirements",
        "- Model strengths and specializations",
        "- Efficiency and cost trade-offs",
    ]

    if requirements:
        content_parts.extend(["", f"Requirements: {requirements}"])

    if constraints:
        content_parts.extend(["", f"Constraints: {constraints}"])

    return [{"role": "user", "content": "\n".join(content_parts)}]


def _generate_contradiction_analysis_messages(
    args: dict[str, str],
) -> list[dict[str, Any]]:
    """Generate messages for contradiction analysis prompt."""
    session_id = args["session_id"]
    sensitivity = args.get("sensitivity", "medium")

    return [
        {
            "role": "user",
            "content": (
                f"Analyze inquiry session {session_id} for contradictions.\n\n"
                f"Detection sensitivity: {sensitivity}\n\n"
                "Identify:\n"
                "- Logical contradictions\n"
                "- Semantic conflicts\n"
                "- Value tensions\n"
                "- Temporal inconsistencies\n\n"
                "For each contradiction, suggest dialectic synthesis approaches."
            ),
        }
    ]


def _generate_workflow_execute_messages(args: dict[str, str]) -> list[dict[str, Any]]:
    """Generate messages for workflow execution prompt."""
    workflow = args["workflow_name"]
    topic = args["topic"]
    mode = args.get("execution_mode", "staged")

    return [
        {
            "role": "user",
            "content": (
                f"Execute the '{workflow}' workflow on topic: {topic}\n\n"
                f"Execution mode: {mode}\n\n"
                "Process through all stages, accumulating context and "
                "building toward comprehensive insights."
            ),
        }
    ]


# Mapping of prompt names to message generators
_PromptGenerator = Callable[[dict[str, str]], list[dict[str, Any]]]

PROMPT_GENERATORS: dict[str, _PromptGenerator] = {
    "expansive-inquiry": _generate_expansive_inquiry_messages,
    "quick-inquiry": _generate_quick_inquiry_messages,
    "creative-inquiry": _generate_creative_inquiry_messages,
    "model-comparison": _generate_model_comparison_messages,
    "route-task": _generate_route_task_messages,
    "contradiction-analysis": _generate_contradiction_analysis_messages,
    "workflow-execute": _generate_workflow_execute_messages,
}


# =============================================================================
# Registry and API Functions
# =============================================================================

# All available prompts
_PROMPTS: dict[str, MCPPrompt] = {
    "expansive-inquiry": EXPANSIVE_INQUIRY_PROMPT,
    "quick-inquiry": QUICK_INQUIRY_PROMPT,
    "creative-inquiry": CREATIVE_INQUIRY_PROMPT,
    "model-comparison": MODEL_COMPARISON_PROMPT,
    "route-task": ROUTE_TASK_PROMPT,
    "contradiction-analysis": CONTRADICTION_ANALYSIS_PROMPT,
    "workflow-execute": WORKFLOW_EXECUTE_PROMPT,
}


def get_inquiry_prompts() -> list[MCPPrompt]:
    """Get all inquiry-related prompts."""
    return [p for p in _PROMPTS.values() if p.category == PromptCategory.INQUIRY]


def get_model_prompts() -> list[MCPPrompt]:
    """Get all model-related prompts."""
    return [p for p in _PROMPTS.values() if p.category == PromptCategory.MODEL]


def get_analysis_prompts() -> list[MCPPrompt]:
    """Get all analysis-related prompts."""
    return [p for p in _PROMPTS.values() if p.category == PromptCategory.ANALYSIS]


def get_workflow_prompts() -> list[MCPPrompt]:
    """Get all workflow-related prompts."""
    return [p for p in _PROMPTS.values() if p.category == PromptCategory.WORKFLOW]


def get_all_prompts() -> list[MCPPrompt]:
    """Get all available prompts."""
    return list(_PROMPTS.values())


def get_prompt(name: str) -> MCPPrompt | None:
    """Get a prompt by name."""
    return _PROMPTS.get(name)


def get_prompt_messages(
    name: str,
    arguments: dict[str, str],
) -> list[dict[str, Any]]:
    """
    Get generated messages for a prompt.

    Args:
        name: Prompt name
        arguments: User-provided arguments

    Returns:
        List of message dicts

    Raises:
        ValueError: If prompt not found or required args missing
    """
    prompt = _PROMPTS.get(name)
    if not prompt:
        raise ValueError(f"Unknown prompt: {name}")

    return prompt.get_messages(arguments)


def list_prompts_by_category() -> dict[str, list[str]]:
    """Get prompt names organized by category."""
    result: dict[str, list[str]] = {}
    for prompt in _PROMPTS.values():
        category = prompt.category.value
        if category not in result:
            result[category] = []
        result[category].append(prompt.name)
    return result
