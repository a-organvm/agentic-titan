"""
Titan API - Admin Routes

REST API endpoints for system administration.

Endpoints:
    GET  /api/admin/health/detailed    - Detailed system health
    GET  /api/admin/metrics/summary    - Metrics summary
    GET  /api/admin/users              - List users
    POST /api/admin/users              - Create user
    PUT  /api/admin/users/{user_id}    - Update user
    DELETE /api/admin/users/{user_id}  - Delete user
    GET  /api/admin/config             - Get configuration
    PUT  /api/admin/config/{key}       - Update configuration
    GET  /api/admin/batches/stalled    - Get stalled batches
    POST /api/admin/batches/{batch_id}/recover - Recover stalled batch
    DELETE /api/admin/batches/cleanup  - Clean up old batches
    POST /api/admin/cache/flush        - Flush cache
    GET  /api/admin/audit/events       - Get audit events
"""

from __future__ import annotations

import logging
import os
import time
from datetime import UTC, datetime
from typing import Any, cast
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, status

from titan.api.typing_helpers import (
    BaseModel,
    Field,
    typed_delete,
    typed_get,
    typed_post,
    typed_put,
)
from titan.auth.middleware import require_admin
from titan.auth.models import UserRole

logger = logging.getLogger("titan.api.admin")
_HEALTH_START_TIME = time.time()

admin_router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(require_admin)],  # All routes require admin
)


# =============================================================================
# Request/Response Models
# =============================================================================


class DetailedHealthResponse(BaseModel):
    """Detailed system health response."""

    status: str
    version: str
    uptime_seconds: float
    components: dict[str, dict[str, Any]]


class MetricsSummaryResponse(BaseModel):
    """Summary of system metrics."""

    total_users: int
    active_users: int
    total_api_keys: int
    total_batches: int
    active_batches: int
    total_inquiries: int
    redis_connected: bool
    postgres_connected: bool


class UserCreateRequest(BaseModel):
    """Request to create a new user."""

    username: str = Field(..., min_length=3, max_length=100)
    email: str | None = Field(default=None)
    password: str = Field(..., min_length=8)  # allow-secret
    role: UserRole = Field(default=UserRole.USER)
    metadata: dict[str, Any] = Field(default_factory=dict)


class UserUpdateRequest(BaseModel):
    """Request to update a user."""

    email: str | None = None
    password: str | None = None  # allow-secret
    role: UserRole | None = None
    is_active: bool | None = None
    metadata: dict[str, Any] | None = None


class UserResponse(BaseModel):
    """User response for admin views."""

    id: str
    username: str
    email: str | None
    role: str
    is_active: bool
    created_at: str
    last_login: str | None


class ConfigResponse(BaseModel):
    """Configuration response."""

    key: str
    value: Any
    description: str | None
    editable: bool


class ConfigUpdateRequest(BaseModel):
    """Request to update configuration."""

    value: Any


class StalledBatchResponse(BaseModel):
    """Stalled batch information."""

    batch_id: str
    status: str
    topics: list[str]
    stalled_since: str
    recommended_action: str


class RecoveryRequest(BaseModel):
    """Batch recovery request."""

    strategy: str = Field(
        default="retry",
        description="Recovery strategy: retry, skip, or fail",
    )


class AuditEventResponse(BaseModel):
    """Audit event response."""

    id: str
    timestamp: str
    event_type: str
    action: str
    agent_id: str | None
    session_id: str | None
    user_id: str | None


# =============================================================================
# Health & Metrics
# =============================================================================


@typed_get(admin_router, "/health/detailed", response_model=DetailedHealthResponse)
async def detailed_health() -> DetailedHealthResponse:
    """
    Get detailed system health including all components.

    Checks Redis, PostgreSQL, ChromaDB, and other services.
    """
    components: dict[str, dict[str, Any]] = {}

    # Check Redis
    try:
        import redis

        from_url = cast(Any, redis.from_url)
        r = from_url(os.getenv("TITAN_REDIS_URL", "redis://localhost:6379"))
        r.ping()
        components["redis"] = {"status": "healthy", "connected": True}
    except Exception as e:
        components["redis"] = {"status": "unhealthy", "error": str(e)}

    # Check PostgreSQL
    try:
        from titan.persistence.postgres import get_postgres_client

        client = get_postgres_client()
        health = await client.health_check()
        components["postgres"] = health
    except Exception as e:
        components["postgres"] = {"status": "unhealthy", "error": str(e)}

    # Determine overall status
    all_healthy = all(
        c.get("status") == "healthy" or c.get("healthy", False) for c in components.values()
    )

    return DetailedHealthResponse(
        status="healthy" if all_healthy else "degraded",
        version="0.1.0",
        uptime_seconds=time.time() - _HEALTH_START_TIME,
        components=components,
    )


@typed_get(admin_router, "/metrics/summary", response_model=MetricsSummaryResponse)
async def metrics_summary() -> MetricsSummaryResponse:
    """
    Get summary of system metrics.
    """
    try:
        from titan.auth.storage import get_auth_storage

        storage = await get_auth_storage()
        total_users = await storage.count_users()
        # Note: These would need proper implementations
        active_users = await storage.count_users()  # Simplified
    except Exception:
        total_users = 0
        active_users = 0

    try:
        from titan.batch.orchestrator import get_batch_orchestrator

        orchestrator = get_batch_orchestrator()
        all_batches = orchestrator.list_batches()
        total_batches = len(all_batches)
        active_batches = len(
            [b for b in all_batches if b.status.value in ("processing", "pending")]
        )
    except Exception:
        total_batches = 0
        active_batches = 0

    try:
        from titan.workflows.inquiry_engine import get_inquiry_engine

        engine = get_inquiry_engine()
        total_inquiries = len(engine.list_sessions())
    except Exception:
        total_inquiries = 0

    # Check connections
    redis_connected = False
    try:
        import redis

        from_url = cast(Any, redis.from_url)
        r = from_url(os.getenv("TITAN_REDIS_URL", "redis://localhost:6379"))
        redis_connected = bool(r.ping())
    except Exception:
        pass

    postgres_connected = False
    try:
        from titan.persistence.postgres import get_postgres_client

        client = get_postgres_client()
        postgres_connected = client.is_connected
    except Exception:
        pass

    return MetricsSummaryResponse(
        total_users=total_users,
        active_users=active_users,
        total_api_keys=0,  # Would need implementation
        total_batches=total_batches,
        active_batches=active_batches,
        total_inquiries=total_inquiries,
        redis_connected=redis_connected,
        postgres_connected=postgres_connected,
    )


# =============================================================================
# User Management
# =============================================================================


@typed_get(admin_router, "/users", response_model=list[UserResponse])
async def list_users(
    role: str | None = Query(None, description="Filter by role"),
    is_active: bool | None = Query(None, description="Filter by active status"),
    limit: int = Query(100, le=1000),
    offset: int = Query(0),
) -> list[UserResponse]:
    """List all users with optional filtering."""
    from titan.auth.storage import get_auth_storage

    storage = await get_auth_storage()
    users = await storage.list_users(
        role=role,
        is_active=is_active,
        limit=limit,
        offset=offset,
    )

    return [
        UserResponse(
            id=str(u["id"]),
            username=u["username"],
            email=u.get("email"),
            role=u["role"],
            is_active=u.get("is_active", True),
            created_at=(
                u["created_at"].isoformat()
                if isinstance(u["created_at"], datetime)
                else u["created_at"]
            ),
            last_login=(
                u["last_login"].isoformat()
                if u.get("last_login") and isinstance(u["last_login"], datetime)
                else u.get("last_login")
            ),
        )
        for u in users
    ]


@typed_post(
    admin_router, "/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED
)
async def create_user(request: UserCreateRequest) -> UserResponse:
    """Create a new user."""
    from titan.auth.storage import get_auth_storage

    storage = await get_auth_storage()

    # Check if username already exists
    existing = await storage.get_user_by_username(request.username)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Username '{request.username}' already exists",
        )

    # Check if email already exists
    if request.email:
        existing = await storage.get_user_by_email(request.email)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Email '{request.email}' already exists",
            )

    # Create user
    user_id = uuid4()
    hashed_password = storage.hash_password(request.password)  # allow-secret
    now = datetime.now(UTC)

    success = await storage.create_user(
        user_id=user_id,
        username=request.username,
        hashed_password=hashed_password,  # allow-secret
        email=request.email,
        role=request.role.value,
        metadata=request.metadata,
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user",
        )

    logger.info(f"Admin created user: {request.username}")

    return UserResponse(
        id=str(user_id),
        username=request.username,
        email=request.email,
        role=request.role.value,
        is_active=True,
        created_at=now.isoformat(),
        last_login=None,
    )


@typed_put(admin_router, "/users/{user_id}", response_model=UserResponse)
async def update_user(user_id: str, request: UserUpdateRequest) -> UserResponse:
    """Update a user."""
    from titan.auth.storage import get_auth_storage

    storage = await get_auth_storage()

    # Check user exists
    user_data = await storage.get_user(user_id)
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    # Build updates
    updates: dict[str, Any] = {}
    if request.email is not None:
        updates["email"] = request.email
    if request.password is not None:  # allow-secret
        updates["hashed_password"] = storage.hash_password(request.password)  # allow-secret
    if request.role is not None:
        updates["role"] = request.role.value
    if request.is_active is not None:
        updates["is_active"] = request.is_active
    if request.metadata is not None:
        updates["metadata"] = request.metadata

    if updates:
        success = await storage.update_user(user_id, updates)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update user",
            )

    # Fetch updated user
    user_data = await storage.get_user(user_id)
    if user_data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found after update",
        )

    logger.info(f"Admin updated user: {user_id}")

    return UserResponse(
        id=str(user_data["id"]),
        username=user_data["username"],
        email=user_data.get("email"),
        role=user_data["role"],
        is_active=user_data.get("is_active", True),
        created_at=(
            user_data["created_at"].isoformat()
            if isinstance(user_data["created_at"], datetime)
            else user_data["created_at"]
        ),
        last_login=(
            user_data["last_login"].isoformat()
            if user_data.get("last_login") and isinstance(user_data["last_login"], datetime)
            else user_data.get("last_login")
        ),
    )


@typed_delete(admin_router, "/users/{user_id}")
async def delete_user(user_id: str) -> dict[str, str]:
    """Delete a user."""
    from titan.auth.storage import get_auth_storage

    storage = await get_auth_storage()

    # Check user exists
    user_data = await storage.get_user(user_id)
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    success = await storage.delete_user(user_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete user",
        )

    logger.info(f"Admin deleted user: {user_id}")

    return {"message": "User deleted successfully"}


# =============================================================================
# Configuration
# =============================================================================


# In-memory config store (would be PostgreSQL in production)
_config_store: dict[str, dict[str, Any]] = {
    "rate_limit_default": {
        "value": "100/minute",
        "description": "Default rate limit for API endpoints",
        "editable": True,
    },
    "batch_max_concurrent": {
        "value": 10,
        "description": "Maximum concurrent batch sessions",
        "editable": True,
    },
    "inquiry_timeout_seconds": {
        "value": 300,
        "description": "Timeout for inquiry stages in seconds",
        "editable": True,
    },
    "cleanup_retention_days": {
        "value": 30,
        "description": "Days to retain completed batches",
        "editable": True,
    },
}


@typed_get(admin_router, "/config", response_model=list[ConfigResponse])
async def get_config() -> list[ConfigResponse]:
    """Get all configuration values."""
    return [
        ConfigResponse(
            key=key,
            value=data["value"],
            description=data.get("description"),
            editable=data.get("editable", False),
        )
        for key, data in _config_store.items()
    ]


@typed_put(admin_router, "/config/{key}", response_model=ConfigResponse)
async def update_config(key: str, request: ConfigUpdateRequest) -> ConfigResponse:
    """Update a configuration value."""
    if key not in _config_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Configuration key not found: {key}",
        )

    if not _config_store[key].get("editable", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Configuration key is not editable: {key}",
        )

    _config_store[key]["value"] = request.value

    logger.info(f"Admin updated config: {key} = {request.value}")

    return ConfigResponse(
        key=key,
        value=_config_store[key]["value"],
        description=_config_store[key].get("description"),
        editable=_config_store[key].get("editable", False),
    )


# =============================================================================
# Batch Management
# =============================================================================


@typed_get(admin_router, "/batches/stalled", response_model=list[StalledBatchResponse])
async def get_stalled_batches(
    threshold_minutes: int = Query(30, description="Minutes without progress"),
) -> list[StalledBatchResponse]:
    """Get batches that appear to be stalled."""
    try:
        from titan.batch.orchestrator import get_batch_orchestrator

        orchestrator = get_batch_orchestrator()
        stalled_ids = await orchestrator.get_stalled_batches(threshold_minutes=threshold_minutes)

        response: list[StalledBatchResponse] = []
        for batch_id in stalled_ids:
            batch = orchestrator.get_batch(batch_id)
            if batch is None:
                continue
            response.append(
                StalledBatchResponse(
                    batch_id=str(batch.id),
                    status=batch.status.value,
                    topics=batch.topics[:5],  # Limit for response size
                    stalled_since=(
                        batch.started_at.isoformat()
                        if batch.started_at
                        else batch.created_at.isoformat()
                    ),
                    recommended_action="retry",
                )
            )
        return response
    except Exception as e:
        logger.error(f"Failed to get stalled batches: {e}")
        return []


@typed_post(admin_router, "/batches/{batch_id}/recover")
async def recover_batch(batch_id: str, request: RecoveryRequest) -> dict[str, Any]:
    """Recover a stalled batch."""
    try:
        from titan.batch.orchestrator import get_batch_orchestrator

        orchestrator = get_batch_orchestrator()

        # Validate strategy
        if request.strategy not in ("retry", "skip", "fail"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid strategy: {request.strategy}",
            )

        recovery_result = await orchestrator.recover_stalled_batch(
            batch_id=batch_id,
            strategy=request.strategy,
        )

        logger.info(f"Admin recovered batch {batch_id} with strategy {request.strategy}")

        recovered_sessions = 0
        failed_sessions = 0
        recovered = False

        if isinstance(recovery_result, dict):
            recovered_sessions = int(recovery_result.get("recovered", 0))
            failed_sessions = int(recovery_result.get("failed", 0))
            recovered = bool(recovery_result.get("success", recovered_sessions > 0))
        else:
            recovered = bool(recovery_result)
            recovered_sessions = int(recovered)

        return {
            "batch_id": batch_id,
            "strategy": request.strategy,
            "recovered": recovered,
            "recovered_sessions": recovered_sessions,
            "failed_sessions": failed_sessions,
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@typed_delete(admin_router, "/batches/cleanup")
async def cleanup_batches(
    retention_days: int = Query(30, description="Keep batches newer than this"),
) -> dict[str, Any]:
    """Clean up old completed batches."""
    try:
        from titan.batch.cleanup import full_cleanup

        result = await full_cleanup(retention_days=retention_days)

        logger.info(f"Admin triggered batch cleanup: {result}")

        if not isinstance(result, dict):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Cleanup returned non-dict result",
            )
        return {str(key): value for key, value in result.items()}
    except Exception as e:
        logger.error(f"Batch cleanup failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cleanup failed: {e}",
        )


# =============================================================================
# System Operations
# =============================================================================


@typed_post(admin_router, "/cache/flush")
async def flush_cache(
    pattern: str = Query("*", description="Key pattern to flush"),
) -> dict[str, Any]:
    """Flush Redis cache."""
    try:
        import redis

        from_url = cast(Any, redis.from_url)
        r = from_url(os.getenv("TITAN_REDIS_URL", "redis://localhost:6379"))

        keys_deleted: str | int
        if pattern == "*":
            # Full flush - dangerous!
            r.flushdb()
            keys_deleted = "all"
        else:
            # Pattern-based deletion
            matched_keys = list(r.keys(pattern))
            if matched_keys:
                r.delete(*matched_keys)
            keys_deleted = len(matched_keys)

        logger.info(f"Admin flushed cache: pattern={pattern}, deleted={keys_deleted}")

        return {
            "pattern": pattern,
            "keys_deleted": keys_deleted,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cache flush failed: {e}",
        )


@typed_get(admin_router, "/audit/events", response_model=list[AuditEventResponse])
async def get_audit_events(
    event_type: str | None = Query(None, description="Filter by event type"),
    agent_id: str | None = Query(None, description="Filter by agent ID"),
    session_id: str | None = Query(None, description="Filter by session ID"),
    limit: int = Query(100, le=1000),
    offset: int = Query(0),
) -> list[AuditEventResponse]:
    """Get audit events with optional filtering."""
    try:
        from titan.persistence.postgres import get_postgres_client

        client = get_postgres_client()
        events = await client.get_audit_events(
            event_type=event_type,
            agent_id=agent_id,
            session_id=session_id,
            limit=limit,
            offset=offset,
        )

        return [
            AuditEventResponse(
                id=str(e["id"]),
                timestamp=(
                    e["timestamp"].isoformat()
                    if isinstance(e["timestamp"], datetime)
                    else e["timestamp"]
                ),
                event_type=e["event_type"],
                action=e["action"],
                agent_id=e.get("agent_id"),
                session_id=e.get("session_id"),
                user_id=e.get("user_id"),
            )
            for e in events
        ]
    except Exception as e:
        logger.error(f"Failed to get audit events: {e}")
        return []
