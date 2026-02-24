"""Tests for MCP prompts module."""

from __future__ import annotations

import pytest

from titan_mcp.prompts import (
    CREATIVE_INQUIRY_PROMPT,
    EXPANSIVE_INQUIRY_PROMPT,
    MODEL_COMPARISON_PROMPT,
    QUICK_INQUIRY_PROMPT,
    ROUTE_TASK_PROMPT,
    MCPPrompt,
    MCPPromptArgument,
    PromptCategory,
    get_all_prompts,
    get_inquiry_prompts,
    get_model_prompts,
    get_prompt,
    get_prompt_messages,
    list_prompts_by_category,
)


class TestMCPPromptArgument:
    """Tests for MCPPromptArgument dataclass."""

    def test_create_required_argument(self) -> None:
        """Test creating a required argument."""
        arg = MCPPromptArgument(
            name="topic",
            description="The topic to explore",
            required=True,
        )

        assert arg.name == "topic"
        assert arg.description == "The topic to explore"
        assert arg.required is True
        assert arg.default is None

    def test_create_optional_argument_with_default(self) -> None:
        """Test creating an optional argument with default."""
        arg = MCPPromptArgument(
            name="depth",
            description="Exploration depth",
            required=False,
            default="medium",
        )

        assert arg.required is False
        assert arg.default == "medium"

    def test_to_dict(self) -> None:
        """Test converting argument to dict."""
        arg = MCPPromptArgument(
            name="topic",
            description="Topic desc",
            required=True,
        )

        d = arg.to_dict()

        assert d["name"] == "topic"
        assert d["description"] == "Topic desc"
        assert d["required"] is True
        assert "default" not in d

    def test_to_dict_with_default(self) -> None:
        """Test dict conversion includes default when set."""
        arg = MCPPromptArgument(
            name="style",
            description="Style",
            required=False,
            default="casual",
        )

        d = arg.to_dict()

        assert d["default"] == "casual"


class TestMCPPrompt:
    """Tests for MCPPrompt dataclass."""

    def test_create_prompt(self) -> None:
        """Test creating a prompt."""
        prompt = MCPPrompt(
            name="test-prompt",
            description="A test prompt",
            category=PromptCategory.INQUIRY,
            arguments=[MCPPromptArgument(name="topic", description="Topic", required=True)],
        )

        assert prompt.name == "test-prompt"
        assert prompt.description == "A test prompt"
        assert prompt.category == PromptCategory.INQUIRY
        assert len(prompt.arguments) == 1

    def test_to_dict(self) -> None:
        """Test converting prompt to dict."""
        prompt = MCPPrompt(
            name="test",
            description="Test",
            category=PromptCategory.MODEL,
            arguments=[],
        )

        d = prompt.to_dict()

        assert d["name"] == "test"
        assert d["description"] == "Test"
        assert d["arguments"] == []

    def test_get_messages_with_generator(self) -> None:
        """Test getting messages calls generator."""
        messages = EXPANSIVE_INQUIRY_PROMPT.get_messages({"topic": "quantum computing"})

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "quantum computing" in messages[0]["content"]

    def test_get_messages_validates_required_args(self) -> None:
        """Test that missing required args raises error."""
        prompt = MCPPrompt(
            name="test",
            description="Test",
            category=PromptCategory.INQUIRY,
            arguments=[
                MCPPromptArgument(name="required_arg", description="Required", required=True)
            ],
        )

        with pytest.raises(ValueError, match="Missing required argument"):
            prompt.get_messages({})

    def test_get_messages_uses_default(self) -> None:
        """Test that defaults are applied for missing optional args."""
        messages = QUICK_INQUIRY_PROMPT.get_messages({"topic": "AI safety"})

        assert len(messages) == 1
        # Default depth should be applied
        assert "moderate" in messages[0]["content"]


class TestPromptRegistry:
    """Tests for prompt registry functions."""

    def test_get_all_prompts(self) -> None:
        """Test getting all prompts."""
        prompts = get_all_prompts()

        assert len(prompts) >= 7
        names = [p.name for p in prompts]
        assert "expansive-inquiry" in names
        assert "quick-inquiry" in names
        assert "creative-inquiry" in names

    def test_get_prompt_by_name(self) -> None:
        """Test getting prompt by name."""
        prompt = get_prompt("expansive-inquiry")

        assert prompt is not None
        assert prompt.name == "expansive-inquiry"
        assert prompt.category == PromptCategory.INQUIRY

    def test_get_prompt_unknown_returns_none(self) -> None:
        """Test getting unknown prompt returns None."""
        prompt = get_prompt("nonexistent-prompt")

        assert prompt is None

    def test_get_inquiry_prompts(self) -> None:
        """Test filtering inquiry prompts."""
        prompts = get_inquiry_prompts()

        assert len(prompts) >= 3
        for p in prompts:
            assert p.category == PromptCategory.INQUIRY

    def test_get_model_prompts(self) -> None:
        """Test filtering model prompts."""
        prompts = get_model_prompts()

        assert len(prompts) >= 2
        for p in prompts:
            assert p.category == PromptCategory.MODEL

    def test_list_prompts_by_category(self) -> None:
        """Test listing prompts organized by category."""
        by_category = list_prompts_by_category()

        assert "inquiry" in by_category
        assert "model" in by_category
        assert "expansive-inquiry" in by_category["inquiry"]


class TestPromptMessages:
    """Tests for get_prompt_messages function."""

    def test_get_prompt_messages_success(self) -> None:
        """Test getting prompt messages."""
        messages = get_prompt_messages(
            "expansive-inquiry",
            {"topic": "machine learning"},
        )

        assert len(messages) == 1
        assert "machine learning" in messages[0]["content"]

    def test_get_prompt_messages_unknown_prompt(self) -> None:
        """Test error for unknown prompt."""
        with pytest.raises(ValueError, match="Unknown prompt"):
            get_prompt_messages("nonexistent", {})

    def test_expansive_inquiry_messages(self) -> None:
        """Test expansive inquiry message generation."""
        messages = get_prompt_messages(
            "expansive-inquiry",
            {
                "topic": "consciousness",
                "context": "philosophical perspective",
                "focus_areas": "neuroscience, philosophy",
            },
        )

        content = messages[0]["content"]
        assert "consciousness" in content
        assert "philosophical perspective" in content
        assert "neuroscience, philosophy" in content

    def test_model_comparison_messages(self) -> None:
        """Test model comparison message generation."""
        messages = get_prompt_messages(
            "model-comparison",
            {"model_a": "gpt-4", "model_b": "claude-3"},
        )

        content = messages[0]["content"]
        assert "gpt-4" in content
        assert "claude-3" in content
        assert "cognitive" in content.lower()

    def test_route_task_messages(self) -> None:
        """Test route task message generation."""
        messages = get_prompt_messages(
            "route-task",
            {
                "task_description": "Write a Python parser",
                "requirements": "fast, accurate",
            },
        )

        content = messages[0]["content"]
        assert "Python parser" in content
        assert "fast, accurate" in content


class TestPromptDefinitions:
    """Tests for predefined prompts."""

    def test_expansive_inquiry_prompt(self) -> None:
        """Test expansive inquiry prompt definition."""
        assert EXPANSIVE_INQUIRY_PROMPT.name == "expansive-inquiry"
        assert EXPANSIVE_INQUIRY_PROMPT.metadata.get("stages") == 6
        assert len(EXPANSIVE_INQUIRY_PROMPT.arguments) >= 1

        # Check required argument
        topic_arg = next(a for a in EXPANSIVE_INQUIRY_PROMPT.arguments if a.name == "topic")
        assert topic_arg.required is True

    def test_quick_inquiry_prompt(self) -> None:
        """Test quick inquiry prompt definition."""
        assert QUICK_INQUIRY_PROMPT.name == "quick-inquiry"
        assert QUICK_INQUIRY_PROMPT.metadata.get("stages") == 3

    def test_creative_inquiry_prompt(self) -> None:
        """Test creative inquiry prompt definition."""
        assert CREATIVE_INQUIRY_PROMPT.name == "creative-inquiry"
        assert CREATIVE_INQUIRY_PROMPT.metadata.get("stages") == 4

    def test_model_comparison_prompt(self) -> None:
        """Test model comparison prompt definition."""
        assert MODEL_COMPARISON_PROMPT.category == PromptCategory.MODEL

        # Both model args should be required
        for arg in MODEL_COMPARISON_PROMPT.arguments:
            if arg.name in ["model_a", "model_b"]:
                assert arg.required is True

    def test_route_task_prompt(self) -> None:
        """Test route task prompt definition."""
        assert ROUTE_TASK_PROMPT.name == "route-task"
        assert ROUTE_TASK_PROMPT.category == PromptCategory.MODEL
