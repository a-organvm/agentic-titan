"""
Titan Core - Configuration System

Centralized Pydantic configuration for the Agentic Titan ecosystem.
Supports environment variable overrides via .env.
"""

# mypy: disable-error-code="misc,untyped-decorator"

from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseSettings):
    """LLM Provider settings."""

    model_config = SettingsConfigDict(env_prefix="TITAN_LLM_")

    default_model: str = "claude-3-5-sonnet-20241022"
    fast_model: str = "claude-3-5-haiku-20241022"
    creative_model: str = "gpt-4o"
    max_tokens: int = 4000
    temperature: float = 0.7


class RedisConfig(BaseSettings):
    """Redis connection settings."""

    model_config = SettingsConfigDict(env_prefix="TITAN_REDIS_")

    url: str = "redis://localhost:6379"
    db: int = 0
    timeout: int = 5


class MCPServerConfig(BaseSettings):
    """Configuration for an upstream MCP server."""
    name: str
    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)


class TitanConfig(BaseSettings):
    """Global configuration for Agentic Titan."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="TITAN_",
        extra="ignore",
    )

    # General
    version: str = "0.1.0"
    environment: str = "development"
    debug: bool = False

    # LLM
    llm: LLMConfig = Field(default_factory=LLMConfig)

    # Infrastructure
    redis: RedisConfig = Field(default_factory=RedisConfig)
    chroma_url: str = "http://localhost:8000"
    
    # MCP Integration
    mcp_servers: list[MCPServerConfig] = Field(default_factory=list)

    # Swarm/Topology
    default_topology: str = "swarm"
    agent_timeout_ms: int = 300000
    max_agent_turns: int = 20

    # Inquiry Engine
    max_context_tokens: int = 4000
    dialectic_friction_threshold: float = 0.7

    @field_validator("environment")
    @classmethod
    def validate_env(cls, v: str) -> str:
        if v not in ["development", "staging", "production"]:
            raise ValueError("Environment must be development, staging, or production")
        return v


# Singleton instance
_config: TitanConfig | None = None


def get_config() -> TitanConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = TitanConfig()
    return _config
